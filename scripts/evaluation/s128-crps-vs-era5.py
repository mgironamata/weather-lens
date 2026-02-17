import xarray as xr
import numpy as np
import time
from dask import config as dask_config
import argparse
from pathlib import Path
import shutil

# --- PERFORMANCE TUNING ---
dask_config.set(scheduler="threads", num_workers=8)

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
    scale=1.0,
    sort_coords=None,
):
    """
    Opens Zarr/NetCDF, extracts variable, handles slicing, renaming, lon-fix, and time-shift.
    INCLUDES FIX FOR DUPLICATE DATES.
    
    Args:
        sort_coords: List of coordinate names to sort (e.g., ['latitude', 'longitude'])
    """
    # Handle both zarr and netcdf
    if zarr_path.endswith('.zarr'):
        # Try consolidated first, fall back to non-consolidated for Zarr v3
        try:
            ds = xr.open_zarr(zarr_path, consolidated=True)
        except (KeyError, FileNotFoundError):
            ds = xr.open_zarr(zarr_path, consolidated=False)
    else:
        ds = xr.open_dataset(zarr_path, chunks='auto')

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
    # Assumption: forecast datasets use init-time on 'time' coordinate.
    # We can shift to valid time using one of:
    # - prediction_timedelta (WN2)
    # - step (GenCast)
    # - lead_time (cumulative precip products)
    # - fixed_lead_hours (e.g., IFS files with one lead per file)
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

        elif "lead_time" in da.coords:
            # Cumulative-precip products: time (init) + lead_time (timedelta64 or hours)
            delta = da["lead_time"]
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
                "no 'prediction_timedelta' or 'step' coord found."
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

    # 7. Sort specified coordinates (e.g., ERA5 may have reversed lat/lon)
    if sort_coords:
        for coord in sort_coords:
            if coord in da.coords:
                da = da.sortby(coord)

    # 8. Adjust ERA5 time to match forecast valid time (add 6 hours for UTC+6)
    # ERA5 daily UTC+6 data has timestamps at 00:00, but represents 06:00-06:00 period
    # GenCast forecasts have valid times at 06:00 for daily accumulation
    if "time" in da.coords and zarr_path.endswith('.nc') and 'era5' in zarr_path:
        print("   ⚠️  Adjusting ERA5 time by +6 hours to match forecast valid time...")
        da = da.assign_coords(time=da.time + np.timedelta64(6, 'h'))

    # 9. Optional scaling (e.g., convert ERA5 mm -> m)
    if scale is not None and scale != 1.0:
        da = da * scale

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

    print(f"🔹 Loading ERA5 Reference: {config['obs_path']}")
    da_obs = open_as_dataarray(config["obs_path"], **config["obs_kwargs"])

    # For 72h accumulation, build ERA5 3-day accumulation ending at each valid time.
    # Assumes 24h ERA5 daily field is aligned to the end of the accumulation window.
    lead_hours = int(config.get("lead_hours", 24))
    if lead_hours == 72:
        da_obs = da_obs.rolling(time=3, min_periods=3).sum()

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
    print("   💾 ERA5 chunks:",      da_obs.chunks)

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

    out_path = Path(config["output_path"])
    if out_path.exists():
        # Avoid mixed-format partial stores on reruns
        shutil.rmtree(out_path)

    try:
        # Prefer Zarr v2 for maximum compatibility with older readers.
        ds_out.to_zarr(
            str(out_path),
            mode="w",
            consolidated=True,
            encoding=encoding,
            safe_chunks=False,
            zarr_format=2,
        )
    except ValueError as e:
        # Some zarr v3 stacks pass a `serializer` even when requesting v2,
        # which fails with: "Zarr format 2 arrays do not support `serializer`."
        msg = str(e)
        if "do not support `serializer`" not in msg and "do not support `serializer`" not in msg.replace("'", "`"):
            raise

        print("   ⚠️  Zarr v2 write failed due to serializer; retrying with Zarr v3 output...")
        if out_path.exists():
            shutil.rmtree(out_path)

        ds_out.to_zarr(
            str(out_path),
            mode="w",
            consolidated=False,
            encoding=encoding,
            safe_chunks=False,
            # Let xarray/zarr choose v3
        )
    print(f"🎉 DONE! Saved to {config['output_path']}")

# ==========================================
# 4. CONFIGURATION
# ==========================================

BASE_OUTPUT_DIR = Path("/scratch2/mg963/results")

VAR_MAP = {
    # Forecast variables are in meters; ERA5 is mm/day and will be scaled to meters.
    "tp": {"ifs": "tp", "gencast": "total_precipitation_cumulative", "wn2": "total_precipitation_cumulative", "era5": "tp_daily_utc6_mm"},
}

MODEL_SPECS = {
    "ifs": {
        "paths": {
            24: "/scratch2/mg963/data/ecmwf/ensembles/ifs/zarr_ens/tp_f24.zarr",
            72: "/scratch2/mg963/data/ecmwf/ensembles/ifs/zarr_ens/tp_f72.zarr",
        },
        "output_subpath": "ecmwf/ifs_vs_ERA5",
        "member_dim": "number",
        "standardize_lon": True,
        "rename_coords": {},
    },
    "gencast": {
        "path": "/scratch2/mg963/data/weathernext/gencast/gencast_2024_precip_cumulative_24_72.zarr",
        "output_subpath": "weathernext/gencast_vs_ERA5",
        "member_dim": "sample",
        "standardize_lon": True,
        "rename_coords": {"lat": "latitude", "lon": "longitude"},
        "to_valid_time": True,  # Convert init time to valid time via lead_time
    },
    "wn2": {
        "path": "/scratch2/mg963/data/weathernext/wn2/wn2_2024_precip_cumulative_24_72.zarr/",
        "output_subpath": "weathernext/wn2_vs_ERA5",
        "member_dim": "sample",
        "standardize_lon": True,
        "rename_coords": {"lat": "latitude", "lon": "longitude"},
        "to_valid_time": True,  # Convert init time to valid time via lead_time
    },
}

# ERA5 data paths
ERA5_PATHS = {
    "tp": "/scratch2/mg963/data/ecmwf/era5/tp/era5/daily_utc6_2024/era5_tp_daily_utc6_2024_processed.nc",
}

# ==========================================
# 5. EXECUTION
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CRPS for 24h/72h cumulative precipitation (IFS/GenCast/WN2) using ERA5 as reference")
    parser.add_argument(
        "--model",
        type=str,
        default="gencast",
        choices=["ifs", "gencast", "wn2"],
        help="Model name: ifs | gencast | wn2",
    )
    parser.add_argument(
        "--lead",
        type=int,
        default=24,
        choices=[24, 72],
        help="Lead time in hours for cumulative precipitation: 24 or 72",
    )
    parser.add_argument(
        "--var",
        type=str,
        default="tp",
        choices=["tp"],
        help="Variable: tp (total precipitation)",
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Extract arguments
    SELECTED_MODEL = args.model
    SELECTED_VAR = args.var
    LEAD_HOURS = int(args.lead)

    # what is this: it is 
    spec = MODEL_SPECS[SELECTED_MODEL]

    # --- Check if ERA5 data is available for this variable ---
    if SELECTED_VAR not in ERA5_PATHS:
        raise ValueError(
            f"ERA5 reference data not configured for variable '{SELECTED_VAR}'. "
            f"Available variables: {list(ERA5_PATHS.keys())}"
        )

    # --- Forecast path ---
    if SELECTED_MODEL == "ifs":
        forecast_path = spec["paths"][LEAD_HOURS]
    else:
        forecast_path = spec["path"]

    print(f"Using forecast path: {forecast_path}")

    # --- Forecast kwargs ---
    fcst_var_name = VAR_MAP[SELECTED_VAR][SELECTED_MODEL]

    # Forecast selection for lead time (GenCast/WN2 use lead_time coord; IFS uses fixed lead per file)
    sel_dict = None
    if SELECTED_MODEL in {"gencast", "wn2"}:
        sel_dict = {"lead_time": np.timedelta64(LEAD_HOURS, "h")}

    fcst_kwargs = {
        "variable": fcst_var_name,
        "sel_dict": sel_dict,
        "standardize_lon": True,
        "rename_coords": spec.get("rename_coords"),
        "to_valid_time": True,
        "fixed_lead_hours": LEAD_HOURS if SELECTED_MODEL == "ifs" else None,
        "sort_coords": ["latitude", "longitude"],
    }

    # --- ERA5 obs kwargs ---
    era5_path = ERA5_PATHS[SELECTED_VAR]
    era5_var_name = VAR_MAP[SELECTED_VAR]["era5"]
    
    obs_kwargs = {
        "variable": era5_var_name,
        "standardize_lon": True,
        "rename_coords": {"valid_time": "time", "latitude": "latitude", "longitude": "longitude"},
        "sort_coords": ["latitude", "longitude"],
        # ERA5 is mm/day; convert to meters
        "scale": 0.001,
    }

    # --- Output path ---
    out_dir = BASE_OUTPUT_DIR / spec["output_subpath"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"crps_{SELECTED_VAR}_{LEAD_HOURS}h.zarr"

    config = {
        "model_name": SELECTED_MODEL,
        "lead_hours": LEAD_HOURS,
        "forecast_path": str(forecast_path),
        "obs_path": era5_path,
        "output_path": str(out_path),
        "forecast_kwargs": fcst_kwargs,
        "obs_kwargs": obs_kwargs,
    }

    # Run
    print(f"\n{'='*60}")
    print(f"Running CRPS: {SELECTED_MODEL} | {SELECTED_VAR} | lead={LEAD_HOURS}h")
    print(f"Reference: ERA5 daily UTC+6")
    print(f"{'='*60}\n")
    start = time.time()
    run_crps_pipeline(config)
    print(f"⏱️  Total Time: {(time.time() - start) / 60:.2f} min")

# example usage:
# python s128-crps-vs-era5.py --model gencast --lead 24 --var tp