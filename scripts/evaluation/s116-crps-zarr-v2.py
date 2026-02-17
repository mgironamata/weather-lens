import xarray as xr
import numpy as np
import time
import dask
import argparse
from pathlib import Path

# --- PERFORMANCE TUNING ---
# Adjust as needed for landau; 8 threads is conservative.
dask.config.set(scheduler="threads", num_workers=8)

# ==========================================
# 1. ROBUST DATA LOADER
# ==========================================

def open_as_dataarray(
    zarr_path,
    variable=None,
    sel_dict=None,
    rename_coords=None,
    standardize_lon=False,
    to_valid_time=False,
    fixed_lead_hours=None,
):
    """
    Opens Zarr, extracts variable, handles slicing, renaming, lon-fix, and time-shift.
    INCLUDES FIX FOR DUPLICATE DATES.
    """
    # Try consolidated first, fall back to non-consolidated for Zarr v3
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except (KeyError, FileNotFoundError):
        ds = xr.open_zarr(zarr_path, consolidated=False)

    # 1. Variable Selection
    var_names = list(ds.data_vars)
    if variable and variable in var_names:
        var_name = variable
    elif len(var_names) == 1:
        var_name = var_names[0]
    else:
        raise ValueError(f"Cannot identify variable in {zarr_path}. Choices: {var_names}")

    da = ds[var_name]

    # 2. Drop duplicate times if present
    if "time" in da.dims and not da.indexes["time"].is_unique:
        print(
            f"   ⚠️  WARNING: Found duplicate times in {Path(zarr_path).name}. "
            "Dropping duplicates..."
        )
        da = da.drop_duplicates(dim="time")

    # 3. Dimensional Slicing (e.g. select lead time)
    if sel_dict:
        da = da.sel(sel_dict)

    # 4. Convert to Validity Time (Forecast only)
    if to_valid_time and "time" in da.coords:
        valid_time = None

        if "prediction_timedelta" in da.coords:
            # WN2-style: time (init) + prediction_timedelta (timedelta64)
            delta = da["prediction_timedelta"]
            valid_time = da["time"] + delta

        elif "step" in da.coords:
            # GenCast-style: time (init) + step (hours or timedelta)
            delta = da["step"]
            if np.issubdtype(delta.dtype, np.timedelta64):
                dt = delta
            else:
                dt = delta * np.timedelta64(1, "h")
            valid_time = da["time"] + dt

        elif fixed_lead_hours is not None:
            # Fixed-lead files (e.g., IFS f24/f72): shift init time by a constant
            dt = np.timedelta64(int(fixed_lead_hours), "h")
            valid_time = da["time"] + dt

        else:
            print(
                "   ⚠️  WARNING: Requested valid_time shift but "
                "no 'prediction_timedelta', 'step' or fixed_lead_hours found."
            )

        if valid_time is not None:
            da = da.assign_coords(valid_time=valid_time)
            da = da.swap_dims({"time": "valid_time"})
            da = da.drop_vars("time")
            da = da.rename({"valid_time": "time"})

            # Check for duplicates AGAIN after shifting time
            if not da.indexes["time"].is_unique:
                da = da.drop_duplicates(dim="time")

    # 5. Coordinate Renaming
    if rename_coords:
        clean_rename = {
            k: v for k, v in rename_coords.items()
            if k in da.dims or k in da.coords
        }
        da = da.rename(clean_rename)

    # 6. Longitude Standardization
    if standardize_lon:
        lon_name = "longitude"
        if lon_name in da.coords:
            new_lon = (da[lon_name] + 180) % 360 - 180
            da = da.assign_coords({lon_name: new_lon}).sortby(lon_name)

    da.name = "data"
    return da

# ==========================================
# 2. CRPS CALCULATION (chunk-wise, no vectorize)
# ==========================================

def crps_fair_fast(forecast, obs, member_dim="number"):
    """
    Fair CRPS for ensemble forecasts.

    Works chunk-wise over all non-member dims; only the last axis (member_dim)
    is treated as the core dimension and sorted.

    forecast: ensemble field with dimension `member_dim`
    obs: deterministic field (no member_dim); will be broadcast against members
    """

    # Force float32 to save RAM
    forecast = forecast.astype(np.float32)
    obs = obs.astype(np.float32)

    # Ensure member_dim is last for forecast only
    if member_dim not in forecast.dims:
        raise ValueError(
            f"Forecast is missing ensemble dimension {member_dim!r}. "
            f"Dims: {forecast.dims}"
        )

    if forecast.dims[-1] != member_dim:
        forecast = forecast.transpose(..., member_dim)

    # Term 1: mean absolute error over members
    term1 = np.abs(forecast - obs).mean(dim=member_dim)

    # Term 2: spread term, operating on whole chunks
    def _spread_block(a):
        """
        a: ndarray with shape (..., M) where last axis is member_dim.
        Returns spread with shape (...,).
        """
        a = np.asarray(a)
        M = a.shape[-1]
        if M < 2:
            return np.zeros(a.shape[:-1], dtype=a.dtype)

        # sort along member axis for entire block at once
        a_sorted = np.sort(a, axis=-1)

        # weights shape (M,)
        i = np.arange(M, dtype=a.dtype)
        weights = 2 * i - (M - 1)

        weighted = a_sorted * weights
        weighted_sum = weighted.sum(axis=-1)

        return weighted_sum / (M * (M - 1))

    term2 = xr.apply_ufunc(
        _spread_block,
        forecast,
        input_core_dims=[[member_dim]],   # last axis is member_dim
        output_core_dims=[[]],            # scalar per (time, lat, lon)
        dask="parallelized",
        output_dtypes=[np.float32],
    )

    return term1 - term2

# ==========================================
# 3. SINGLE-PASS PIPELINE (let dask handle chunking)
# ==========================================

def run_crps_pipeline(config):
    spec = MODEL_SPECS[config["model_name"]]

    print(f"🔹 Loading Forecast: {config['forecast_path']}")
    da_fcst = open_as_dataarray(config["forecast_path"], **config["forecast_kwargs"])

    print(f"🔹 Loading Obs: {config['obs_path']}")
    da_obs = open_as_dataarray(config["obs_path"], **config["obs_kwargs"])

    # Align (Inner Join)
    print("🔹 Aligning datasets...")
    da_fcst, da_obs = xr.align(da_fcst, da_obs, join="inner")

    total_times = da_fcst.sizes["time"]
    if total_times == 0:
        print("❌ ALIGNMENT FAILED: Intersection is empty.")
        return

    print(f"🚀 Processing {total_times} times in one dask graph (chunked)...")

    # Choose chunking for computation
    member_dim = spec["member_dim"]
    # Reasonable default: multi-day time chunks + spatial tiles, all members together
    fcst_chunks = {"time": 16, "latitude": 90, "longitude": 180, member_dim: -1}
    obs_chunks  = {"time": 16, "latitude": 90, "longitude": 180}

    da_fcst = da_fcst.chunk(fcst_chunks)
    da_obs  = da_obs.chunk(obs_chunks)

    print("   💾 Forecast chunks:", da_fcst.chunks)
    print("   💾 Obs chunks:",      da_obs.chunks)

    # Compute CRPS lazily (dask graph)
    da_crps = crps_fair_fast(da_fcst, da_obs, member_dim=member_dim)

    # Naming
    var_name = config["forecast_kwargs"].get("variable", "data")
    da_crps.name = f"{var_name}_crps"

    # For output, we can re-chunk the CRPS to something similar (no member_dim now)
    crps_chunks = {"time": 16, "latitude": 90, "longitude": 180}
    da_crps = da_crps.chunk(crps_chunks)

    print("   💾 CRPS chunks:", da_crps.chunks)

    # Write once to Zarr (dask will execute in chunks)
    ds_out = da_crps.to_dataset()
    enc_var = da_crps.name

    # Avoid old numcodecs/BytesBytesCodec issues: no compressor
    encoding = {enc_var: {"compressor": None}}

    print(f"💾 Writing CRPS to {config['output_path']} ...")
    ds_out.to_zarr(
        config["output_path"],
        mode="w",
        consolidated=True,
        encoding=encoding,
        # let xarray/zarr pick the format (likely v3)
    )
    print(f"🎉 DONE! Saved to {config['output_path']}")

# ==========================================
# 4. CONFIGURATION
# ==========================================

BASE_OUTPUT_DIR = Path("/scratch2/mg963/results")

VAR_MAP = {
    "t2m":   {"wn2": "2m_temperature", "ifs": "t2m"},
    "winds": {"wn2": "10m_wind_speed",  "ifs": "wspd10"},  # forecast-side variable names
    "tp":   {"wn2": "total_precipitation_12hr", "ifs": "tp_6h"}
}

# Path name mapping (for file names, which differ from variable names)
PATH_MAP = {
    "WN2": {
        "t2m": "t2m",
        "winds": "10m_wind_speed",  # uses wn2_2024_10m_wind_speed.zarr
    },
    "gencast": {
        "t2m": "t2m", 
        "winds": "10m_wind_speed",  # adjust if gencast uses different naming
    },
}

OBS_VAR_MAP = {
    "t2m":   "t2m",
    "winds": "ws10",   # analysis/control name
}

MODEL_SPECS = {
    "WN2": {
        # Per-variable Zarr - path uses PATH_MAP, variable uses VAR_MAP
        "path": "/scratch2/mg963/data/weathernext/wn2/wn2_2024_{var}.zarr",
        "output_subpath": "weathernext/WN2",
        "member_dim": "sample",
        "standardize_lon": True,
        "rename_coords": {"lat": "latitude", "lon": "longitude"},
        "needs_time_selection": True,
        "lead_coord": "prediction_timedelta",
        "to_valid_time": True,
    },
    "gencast": {
        # Per-variable Zarr - path uses PATH_MAP, variable uses VAR_MAP
        "path": "/scratch2/mg963/data/weathernext/gencast/gencast_2024_{var}.zarr",
        "output_subpath": "weathernext/gencast",
        "member_dim": "sample",
        "standardize_lon": True,
        "rename_coords": {"lat": "latitude", "lon": "longitude"},
        "needs_time_selection": True,    # select a single 'step'
        "lead_coord": "step",
        "to_valid_time": True,           # time + step*h
    },
    "IFS_ENS": {
        # Per-variable, per-lead Zarr layout
        "path_template": "/scratch2/mg963/data/ecmwf/ensembles/ifs/zarr_ens/{var}_f{lead}.zarr",
        "output_subpath": "ecmwf/ifs-ens",
        "member_dim": "number",
        "standardize_lon": False,
        "rename_coords": {},
        "needs_time_selection": False,   # each file already one lead
        "lead_coord": None,
        "to_valid_time": True,           # Shift init time to valid time
    },
}

OBS_PATH = "/scratch2/mg963/data/ecmwf/analysis/ifs_ens_control_06z.zarr"

# ==========================================
# 5. EXECUTION
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CRPS for weather forecasts")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["WN2", "gencast", "IFS_ENS"],
                        help="Model name: WN2, gencast, or IFS_ENS")
    parser.add_argument("--var", type=str, required=True,
                        choices=["t2m", "winds"],
                        help="Variable: t2m or winds")
    parser.add_argument("--lead", type=int, required=True,
                        help="Lead time in hours (e.g., 24, 72)")
    
    args = parser.parse_args()
    
    SELECTED_MODEL = args.model
    SELECTED_VAR   = args.var
    SELECTED_LEAD  = args.lead

    spec = MODEL_SPECS[SELECTED_MODEL]

    # --- Forecast path logic (single Zarr vs per-var-per-lead Zarr) ---

    if SELECTED_MODEL in ("WN2", "gencast"):
        # Single per-variable Zarr for all leads - use PATH_MAP for file name
        path_var = PATH_MAP[SELECTED_MODEL][SELECTED_VAR]
        forecast_path = spec["path"].format(var=path_var)
    elif SELECTED_MODEL == "IFS_ENS":
        # Per-variable, per-lead Zarr
        if "path_template" not in spec:
            raise ValueError("IFS_ENS spec is missing 'path_template'.")
        ifs_var_name = VAR_MAP[SELECTED_VAR]["ifs"]
        forecast_path = spec["path_template"].format(var=ifs_var_name, lead=SELECTED_LEAD)
    else:
        raise ValueError(f"Unknown model: {SELECTED_MODEL}")

    print(f"Using forecast path: {forecast_path}")

    # --- Forecast kwargs ---
    if SELECTED_MODEL in ("WN2", "gencast"):
        fcst_var_name = VAR_MAP[SELECTED_VAR]["wn2"]
    elif SELECTED_MODEL == "IFS_ENS":
        fcst_var_name = VAR_MAP[SELECTED_VAR]["ifs"]
    else:
        raise ValueError(f"Unknown model: {SELECTED_MODEL}")

    fcst_kwargs = {
        "variable": fcst_var_name,
        "standardize_lon": spec["standardize_lon"],
        "rename_coords": spec["rename_coords"],
        "to_valid_time": spec.get("to_valid_time", False),
        "fixed_lead_hours": SELECTED_LEAD if SELECTED_MODEL == "IFS_ENS" else None,
    }

    if spec.get("needs_time_selection", False):
        lead_coord = spec.get("lead_coord", "prediction_timedelta")
        if lead_coord == "prediction_timedelta":
            # WN2: prediction_timedelta is a timedelta64
            fcst_kwargs["sel_dict"] = {"prediction_timedelta": np.timedelta64(SELECTED_LEAD, "h")}
        elif lead_coord == "step":
            # GenCast: step is timedelta64 (e.g. 72h) after decode
            fcst_kwargs["sel_dict"] = {"step": np.timedelta64(SELECTED_LEAD, "h")}
        else:
            raise ValueError(f"Unknown lead_coord {lead_coord!r} for model {SELECTED_MODEL!r}")

    # --- Obs kwargs ---
    obs_var_name = OBS_VAR_MAP[SELECTED_VAR]
    obs_kwargs = {"variable": obs_var_name, "standardize_lon": False}

    # --- Output path ---
    out_dir = BASE_OUTPUT_DIR / spec["output_subpath"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"crps_{SELECTED_VAR}_{SELECTED_LEAD}h.zarr"

    config = {
        "model_name": SELECTED_MODEL,
        "forecast_path": str(forecast_path),
        "obs_path": OBS_PATH,
        "output_path": str(out_path),
        "forecast_kwargs": fcst_kwargs,
        "obs_kwargs": obs_kwargs,
    }

    # Run
    print(f"\n{'='*60}")
    print(f"Running CRPS: {SELECTED_MODEL} | {SELECTED_VAR} | {SELECTED_LEAD}h")
    print(f"{'='*60}\n")
    start = time.time()
    run_crps_pipeline(config)
    print(f"⏱️  Total Time: {(time.time() - start) / 60:.2f} min")
