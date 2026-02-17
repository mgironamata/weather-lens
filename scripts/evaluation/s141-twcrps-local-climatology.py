import xarray as xr
import numpy as np
import argparse
import dask.array as da
from dask.base import compute
from dask import config as dask_config
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

# -----------------------------------------
# CONFIG
# -----------------------------------------

parser = argparse.ArgumentParser(description="Compute local threshold-weighted CRPS")
parser.add_argument("--var", type=str, default="t2m", choices=["t2m", "winds", "tp"], help="Variable name")
parser.add_argument("--lead", type=int, default=24, help="Lead time in hours")
parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
parser.add_argument("--models", type=str, nargs="+", default=["WN2", "GenCast"], 
                    choices=["WN2", "GenCast", "wn2", "gencast"], 
                    help="AI models to include (default: both)")
args = parser.parse_args()

# Set dask configuration for parallel processing
dask_config.set(scheduler="threads", num_workers=8)

variable = args.var
lead_hours = args.lead

# Normalize model names to match internal keys
selected_models = []
for m in args.models:
    ml = m.lower()
    if ml == "wn2":
        selected_models.append("WN2")
    elif ml == "gencast":
        selected_models.append("GenCast")
    else:
        selected_models.append(m)

# Always include IFS-ENS for baseline if available
models_to_include = selected_models + ["IFS-ENS"]

# Optional: Select a specific time period
start_date = args.start
if start_date is None:
    if lead_hours == 24:
        start_date = "2024-02-04"
    elif lead_hours == 72:
        start_date = "2024-02-06"

end_date = args.end

VAR_MAP = {
    "t2m": {"WN2": "2m_temperature_crps",
            "GenCast": "2m_temperature_crps",
            "IFS-ENS": "t2m_crps"},
    "winds": {"WN2": "10m_wind_speed_crps",
              "GenCast": "10m_wind_speed_crps",
              "IFS-ENS": "wspd10_crps"},
    "tp": {"WN2": "total_precipitation_cumulative_crps",
           "GenCast": "total_precipitation_cumulative_crps",
           "IFS-ENS": "tp_crps"}
}

OBS_VAR_MAP = {
    "t2m":   "t2m",
    "winds": "ws10",
    "tp":    "tp_daily_utc6_mm",
}

VAR_CONFIG = {
    "t2m": {
        "long_name": "2m Temperature",
        "obs_units": "K",
        "display_units": "°C",
        "convert_obs": lambda x: x,
        "convert_threshold_display": lambda x: x - 273.15,
    },
    "winds": {
        "long_name": "10m Wind Speed",
        "obs_units": "m/s",
        "display_units": "m/s",
        "convert_obs": lambda x: x,
        "convert_threshold_display": lambda x: x,
    },
    "tp": {
        "long_name": "Total Precipitation",
        "obs_units": "m",
        "display_units": "mm",
        "convert_obs": lambda x: x * 0.001,
        "convert_threshold_display": lambda x: x * 1000.0,
    },
}

var_cfg = VAR_CONFIG[variable]

OBS_PATHS = {
    "t2m": "/scratch2/mg963/data/ecmwf/analysis/ifs_ens_control_06z.zarr",
    "winds": "/scratch2/mg963/data/ecmwf/analysis/ifs_ens_control_06z.zarr",
    "tp": "/scratch2/mg963/data/ecmwf/era5/tp/era5/daily_utc6_2024/era5_tp_daily_utc6_2024_processed.nc",
}
OBS_PATH = OBS_PATHS[variable]
OBS_VAR  = OBS_VAR_MAP[variable]

if variable == "tp":
    CRPS_CONFIG = {
        "WN2": {
            "path": f"/scratch2/mg963/results/weathernext/wn2_vs_ERA5/crps_tp_{lead_hours}h.zarr",
            "var": VAR_MAP[variable]["WN2"],
        },
        "GenCast": {
            "path": f"/scratch2/mg963/results/weathernext/gencast_vs_ERA5/crps_tp_{lead_hours}h.zarr",
            "var": VAR_MAP[variable]["GenCast"],
        },
        "IFS-ENS": {
            "path": f"/scratch2/mg963/results/ecmwf/ifs_vs_ERA5/crps_tp_{lead_hours}h.zarr",
            "var": VAR_MAP[variable]["IFS-ENS"],
        },
    }
else:
    CRPS_CONFIG = {
        "WN2": {
            "path": f"/scratch2/mg963/results/weathernext/WN2/crps_{variable}_{lead_hours}h.zarr",
            "var": VAR_MAP[variable]["WN2"],
        },
        "GenCast": {
            "path": f"/scratch2/mg963/results/weathernext/gencast/crps_{variable}_{lead_hours}h.zarr",
            "var": VAR_MAP[variable]["GenCast"],
        },
        "IFS-ENS": {
            "path": f"/scratch2/mg963/results/ecmwf/ifs-ens/crps_{variable}_{lead_hours}h.zarr",
            "var": VAR_MAP[variable]["IFS-ENS"],
        },
    }

CRPS_CONFIG = {name: cfg for name, cfg in CRPS_CONFIG.items() if name in models_to_include}

Q_LIST = [0.5, 0.9, 0.95, 0.99, 0.999, 0.9999]

OUT_DIR = Path("/scratch2/mg963/results/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DATA = OUT_DIR / f"twcrps_local_{variable}_{lead_hours}h.nc"

def open_zarr_robust(path):
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:
        return xr.open_zarr(path, consolidated=False)

# -----------------------------------------
# 1. Load Obs & Setup Grid
# -----------------------------------------

print("🔹 Loading observations...")
if OBS_PATH.endswith(".zarr"):
    ds_obs = open_zarr_robust(OBS_PATH)
else:
    ds_obs = xr.open_dataset(OBS_PATH, chunks="auto")

if variable == "tp" and "valid_time" in ds_obs.coords and "time" not in ds_obs.coords:
    ds_obs = ds_obs.rename({"valid_time": "time"})

obs = ds_obs[OBS_VAR]

if variable == "tp" and OBS_PATH.endswith(".nc") and "era5" in OBS_PATH:
    print("   ⚠️  Adjusting ERA5 time by +6 hours...")
    obs = obs.assign_coords(time=obs.time + np.timedelta64(6, "h"))

obs = var_cfg["convert_obs"](obs)

if variable == "tp" and lead_hours == 72:
    print("   ⚠️  Computing 3-day rolling sum for ERA5...")
    obs = obs.rolling(time=3, min_periods=3).sum()

if "time" in obs.dims:
    if not obs.indexes["time"].is_unique:
        obs = obs.drop_duplicates(dim="time")
    obs = obs.sortby("time")

if start_date or end_date:
    obs = obs.sel(time=slice(start_date, end_date))

lat_dim = "latitude" if "latitude" in obs.dims else "lat"
lon_dim = "longitude" if "longitude" in obs.dims else "lon"

# -----------------------------------------
# 2. Compute Local Thresholds
# -----------------------------------------

def compute_local_thresholds(obs_da, q_list, window_size=25):
    print(f"🔹 Computing local thresholds with {window_size}x{window_size} window...")
    
    # Ensure contiguous in time and chunked spatially
    # We use smaller spatial chunks to keep memory usage per worker manageable
    obs_da = obs_da.chunk({"time": -1, lat_dim: 64, lon_dim: 64})
    obs_arr = obs_da.data
    
    overlap = window_size // 2
    
    # Prepare padded array with boundary handling
    # axis 0: time (no overlap)
    # axis 1: lat (nearest)
    # axis 2: lon (wrap)
    padded_arr = da.overlap.overlap(obs_arr, 
                                    depth={0: 0, 1: overlap, 2: overlap},
                                    boundary={0: 'none', 1: 'nearest', 2: 'wrap'})
    
    def block_func(block, q_list, window_size):
        # block shape: (T, H_p, W_p)
        T, H_p, W_p = block.shape
        ov = window_size // 2
        H_c = H_p - 2 * ov
        W_c = W_p - 2 * ov
        
        if H_c <= 0 or W_c <= 0:
            # Return an empty chunk with the correct number of dimensions and size
            return np.zeros((len(q_list), max(0, H_c), max(0, W_c)), dtype=block.dtype)

        # Use sliding_window_view for spatial pooling
        # result: (T, H_c, W_c, window_size, window_size)
        windows = sliding_window_view(block, (window_size, window_size), axis=(1, 2))
        
        # Reshape to (H_c, W_c, T * window_size * window_size)
        reshaped = windows.transpose(1, 2, 0, 3, 4).reshape(H_c, W_c, -1)
        
        # Compute quantiles
        return np.quantile(reshaped, q_list, axis=-1) # (len(q_list), H_c, W_c)

    # Output chunk definition
    lat_chunks = obs_arr.chunks[1]
    lon_chunks = obs_arr.chunks[2]
    out_chunks = ((len(q_list),), lat_chunks, lon_chunks)
    
    thresholds_arr = da.map_blocks(
        block_func,
        padded_arr,
        q_list=q_list,
        window_size=window_size,
        dtype=obs_arr.dtype,
        chunks=out_chunks,
        drop_axis=0,      # time
        new_axis=0        # quantile
    )
    
    thresholds = xr.DataArray(
        thresholds_arr,
        coords={
            "quantile": q_list,
            lat_dim: obs_da[lat_dim],
            lon_dim: obs_da[lon_dim]
        },
        dims=["quantile", lat_dim, lon_dim]
    ).chunk({"quantile": 1, lat_dim: 180, lon_dim: 180})
    
    return thresholds

local_thresholds = compute_local_thresholds(obs, Q_LIST)

# -----------------------------------------
# 3. Load CRPS & Align
# -----------------------------------------

print("\n🔹 Loading and aligning CRPS datasets...")
datasets_to_align = [obs]
names_order = []

for name, cfg in CRPS_CONFIG.items():
    ds = open_zarr_robust(cfg["path"])
    crps = ds[cfg["var"]]
    if "time" in crps.dims:
        if not crps.indexes["time"].is_unique:
            crps = crps.drop_duplicates(dim="time")
        crps = crps.sortby("time")
    datasets_to_align.append(crps)
    names_order.append(name)

aligned = xr.align(*datasets_to_align, join="inner")
obs_use = aligned[0]
crps_models = {name: aligned[i+1] for i, name in enumerate(names_order)}

# Final time selection
if start_date or end_date:
    obs_use = obs_use.sel(time=slice(start_date, end_date))
    crps_models = {name: crps.sel(time=slice(start_date, end_date)) for name, crps in crps_models.items()}

# Ensure consistent chunking
obs_use = obs_use.chunk({"time": 50, lat_dim: 180, lon_dim: 180})
crps_models = {name: crps.chunk({"time": 50, lat_dim: 180, lon_dim: 180}) for name, crps in crps_models.items()}
local_thresholds = local_thresholds.reindex_like(obs_use, method="nearest")

# -----------------------------------------
# 4. Compute Local TWCRPS
# -----------------------------------------

print("\n🔹 Computing local TWCRPS...")

def compute_twcrps_maps(crps_da, obs_da, thresholds_da):
    # Broadcast threshold to (quantile, time, lat, lon) via lazy mask
    mask = obs_da >= thresholds_da
    
    # Conditional mean: sum(CRPS * mask) / sum(mask) over time
    num = (crps_da * mask).sum(dim="time")
    den = mask.sum(dim="time")
    
    # Use where to avoid division by zero
    return num / den.where(den > 0)

twcrps_results = {}
for name, crps_da in crps_models.items():
    print(f"   Processing {name}...")
    twcrps_results[name] = compute_twcrps_maps(crps_da, obs_use, local_thresholds)

# -----------------------------------------
# 5. Save Results
# -----------------------------------------

print("\n🔹 Packaging and saving...")

ds_out = xr.Dataset(
    data_vars={
        f"twcrps_{name}": twcrps_results[name] for name in twcrps_results
    },
    coords=local_thresholds.coords
)
ds_out["thresholds_native"] = local_thresholds
ds_out["thresholds_display"] = var_cfg["convert_threshold_display"](local_thresholds)

# Add attributes
ds_out.attrs["variable"] = variable
ds_out.attrs["lead_hours"] = lead_hours
ds_out.attrs["window_size"] = "25x25"
ds_out.attrs["description"] = "Local threshold-weighted CRPS using spatial-window climatology"

if OUT_DATA.exists():
    OUT_DATA.unlink()

print(f"💾 Saving to {OUT_DATA}")
# Use compute() to trigger dask and then save
# Optimization: better to use to_netcdf directly on the delayed object
ds_out.to_netcdf(OUT_DATA)

print("✅ Done.")
