import xarray as xr
import numpy as np
import argparse
from dask.base import compute
from dask import config as dask_config

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None
from pathlib import Path

# -----------------------------------------
# CONFIG
# -----------------------------------------

parser = argparse.ArgumentParser(description="Compute threshold-weighted CRPS")
parser.add_argument("--var", type=str, default="t2m", choices=["t2m", "winds", "tp"], help="Variable name")
parser.add_argument("--lead", type=int, default=24, help="Lead time in hours")
parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
parser.add_argument("--mask", type=str, default="all", choices=["all", "land", "ocean"], help="Mask type")
parser.add_argument("--models", type=str, nargs="+", default=["WN2", "GenCast"], 
                    choices=["WN2", "GenCast", "wn2", "gencast"], 
                    help="AI models to include (default: both)")
args = parser.parse_args()

dask_config.set(scheduler="threads", num_workers=8)

variable = args.var
lead_hours = args.lead
mask_type = args.mask

# Normalize model names to match internal keys
selected_models = []
for m in args.models:
    ml = m.lower()
    if ml == "wn2":
        selected_models.append("WN2")
    elif ml == "gencast":
        selected_models.append("GenCast")
    else:
        selected_models.append(m) # fallback

# Always include IFS-ENS for baseline if available in the config
models_to_include = selected_models + ["IFS-ENS"]

area_weighted = True  # Use latitude-weighted (cosine) area weighting in CRPS computation

# Optional: Select a specific time period
start_date = args.start
if start_date is None:
    if lead_hours == 24:
        start_date = "2024-02-04"
    elif lead_hours == 72:
        start_date = "2024-02-06"

end_date = args.end

VAR_MAP = {"t2m": {"WN2": "2m_temperature_crps",
                   "GenCast": "2m_temperature_crps",
                   "IFS-ENS": "t2m_crps"},
            "winds": {"WN2": "10m_wind_speed_crps",
                      "GenCast": "10m_wind_speed_crps",
                      "IFS-ENS": "wspd10_crps"}
        ,
        "tp": {"WN2": "total_precipitation_cumulative_crps",
            "GenCast": "total_precipitation_cumulative_crps",
            "IFS-ENS": "tp_crps"}
}

OBS_VAR_MAP = {
    "t2m":   "t2m",
    "winds": "ws10",   # analysis/control name
    "tp":    "tp_daily_utc6_mm",  # ERA5 daily UTC+6 precipitation (mm/day)
}

# Variable-specific configuration for units and conversions
VAR_CONFIG = {
    "t2m": {
        "long_name": "2m Temperature",
        "short_name": "T2m",
        "obs_units": "K",           # units in observation file
        "display_units": "°C",      # units for display/plotting
        "crps_units": "K",          # units for CRPS
        "convert_obs": lambda x: x,           # Keep in Kelvin for native calculations
        "convert_threshold_display": lambda x: x - 273.15,  # K to °C for threshold labels
        "threshold_label": "Temperature",
        "crps_label": "CRPS (K)",
    },
    "winds": {
        "long_name": "10m Wind Speed",
        "short_name": "WS10",
        "obs_units": "m/s",
        "display_units": "m/s",
        "crps_units": "m/s",
        "convert_obs": lambda x: x,  # no conversion
        "convert_threshold_display": lambda x: x,  # no conversion
        "threshold_label": "Wind Speed",
        "crps_label": "CRPS (m/s)",
    },
    "tp": {
        "long_name": "Total Precipitation",
        "short_name": "TP",
        # We will convert ERA5 mm/day -> meters before computing thresholds.
        "obs_units": "m",
        "display_units": "mm",
        "crps_units": "m",
        "convert_obs": lambda x: x * 0.001,  # mm -> m (for the obs field)
        "convert_threshold_display": lambda x: x * 1000.0,  # m -> mm (for labels)
        "threshold_label": "Precipitation",
        "crps_label": "CRPS (m)",
    },
}

# Get configuration for current variable
var_cfg = VAR_CONFIG[variable]

OBS_PATHS = {
    "t2m": "/scratch2/mg963/data/ecmwf/analysis/ifs_ens_control_06z.zarr",
    "winds": "/scratch2/mg963/data/ecmwf/analysis/ifs_ens_control_06z.zarr",
    "tp": "/scratch2/mg963/data/ecmwf/era5/tp/era5/daily_utc6_2024/era5_tp_daily_utc6_2024_processed.nc",
}
OBS_PATH = OBS_PATHS[variable]
OBS_VAR  = OBS_VAR_MAP[variable]

if variable == "tp":
    # New precipitation CRPS outputs (computed vs ERA5, already on valid-time axis)
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
    # Legacy winds/t2m CRPS outputs
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

# Filter CRPS_CONFIG based on selected models
CRPS_CONFIG = {name: cfg for name, cfg in CRPS_CONFIG.items() if name in models_to_include}

# Quantiles to define thresholds (upper tail emphasis)
Q_LIST = [
          0.50, 0.90, 0.95, 0.99, 
          0.995, 0.999, 0.9995, 0.9999, 
          0.99995, 0.99999,
          0.999995, 
          0.999999
          ]

# Output
OUT_DIR = Path("/scratch2/mg963/results/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

mask_suffix = f"_{mask_type}" if mask_type != "all" else ""
# Add model suffix if not default (both)
if len(selected_models) == 1:
    models_suffix = f"_{selected_models[0].lower()}"
else:
    models_suffix = ""

OUT_FIG = OUT_DIR / f"twcrps_{variable}_{lead_hours}h_thresholds{mask_suffix}{models_suffix}.png"
OUT_FIG_SKILL = OUT_DIR / f"twcrps_skill_vs_ifs_{variable}_{lead_hours}h_thresholds{mask_suffix}{models_suffix}.png"
OUT_DATA = OUT_DIR / f"twcrps_{variable}_{lead_hours}h_thresholds{mask_suffix}{models_suffix}.nc"


def open_zarr_robust(path):
    """Opens Zarr with consolidated=True, falls back to False."""
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:
        return xr.open_zarr(path, consolidated=False)


# -----------------------------------------
# 1. Load obs & compute thresholds
# -----------------------------------------

print("🔹 Loading observations...")
if OBS_PATH.endswith(".zarr"):
    ds_obs = open_zarr_robust(OBS_PATH)
else:
    ds_obs = xr.open_dataset(OBS_PATH, chunks="auto")

# Normalize ERA5 naming if needed
if variable == "tp" and "valid_time" in ds_obs.coords and "time" not in ds_obs.coords:
    ds_obs = ds_obs.rename({"valid_time": "time"})

obs = ds_obs[OBS_VAR]

# ERA5 daily UTC+6 data: timestamps may be at 00:00 but represent 06:00-06:00 period
if variable == "tp" and OBS_PATH.endswith(".nc") and "era5" in OBS_PATH:
    print("   ⚠️  Adjusting ERA5 time by +6 hours to match forecast valid time...")
    obs = obs.assign_coords(time=obs.time + np.timedelta64(6, "h"))

# Unit conversion for obs field if configured (e.g., mm -> m)
obs = var_cfg["convert_obs"](obs)

# For 72h accumulation, build ERA5 3-day accumulation ending at each valid time.
if variable == "tp" and lead_hours == 72:
    print("   ⚠️  Computing 3-day rolling sum for ERA5 (72h lead)...")
    obs = obs.rolling(time=3, min_periods=3).sum()

# 🔧 NEW: drop duplicate times in obs (we know this dataset has them)
if "time" in obs.dims:
    if not obs.indexes["time"].is_unique:
        print("   ⚠️  WARNING: obs has duplicate times. Dropping duplicates along 'time'...")
        obs = obs.drop_duplicates(dim="time")
    # Always sort to ensure monotonic index for slicing and alignment
    obs = obs.sortby("time")

# --- Time Selection ---
if start_date or end_date:
    print(f"   Selecting time period for thresholds: {start_date or 'start'} to {end_date or 'end'}...")
    obs = obs.sel(time=slice(start_date, end_date))

# Chunk reasonably for quantiles
if "latitude" in obs.dims and "longitude" in obs.dims:
    obs = obs.chunk({"time": 16, "latitude": 90, "longitude": 180})
else:
    # Some legacy obs may use lat/lon
    obs = obs.chunk({"time": 16, "lat": 90, "lon": 180})

print("   Computing quantile-based thresholds...")
quantiles = obs.quantile(Q_LIST, dim=("time", "latitude", "longitude"))
# quantiles is DataArray with dim 'quantile'
thresholds_native = quantiles.values  # native units (K for t2m, m/s for winds)
thresholds_display = var_cfg["convert_threshold_display"](thresholds_native)

# Filter out duplicate thresholds (common for precipitation) while keeping Q_LIST aligned
unique_indices = np.unique(thresholds_native, return_index=True)[1]
unique_indices.sort()

thresholds_native = thresholds_native[unique_indices]
thresholds_display = thresholds_display[unique_indices]
q_list_use = np.array(Q_LIST)[unique_indices]

print(f"   Thresholds ({var_cfg['obs_units']}, {var_cfg['display_units']}):")
for q, native, display in zip(q_list_use, thresholds_native, thresholds_display):
    print(f"      q={q:.6f} -> {native:.6f} {var_cfg['obs_units']} ({display:.2f} {var_cfg['display_units']})")


# -----------------------------------------
# 2. Area weights (cos(lat)) - optional
# -----------------------------------------

lat_name = "latitude"
if lat_name not in obs.coords:
    lat_name = "lat"

if area_weighted:
    lat = obs[lat_name]
    weights_lat = np.cos(np.deg2rad(lat))
    print(f"   Using area-weighted (cos(lat)) CRPS computation")
else:
    # Uniform weights (no area weighting)
    weights_lat = xr.ones_like(obs[lat_name])
    print(f"   Using unweighted CRPS computation")


# -----------------------------------------
# 3. Load CRPS fields for each model and align with obs (OPTIMIZED: single alignment)
# -----------------------------------------

print("\n🔹 Loading all CRPS datasets...")
datasets_to_align = [obs]
names_order = []

for name, cfg in CRPS_CONFIG.items():
    ds = open_zarr_robust(cfg["path"])
    crps = ds[cfg["var"]]
    
    # Drop duplicate times and sort
    if "time" in crps.dims:
        if not crps.indexes["time"].is_unique:
            print(f"   ⚠️  Cleaning duplicates in {name}...")
            crps = crps.drop_duplicates(dim="time")
        crps = crps.sortby("time")
    
    datasets_to_align.append(crps)
    names_order.append(name)

print("   Aligning all datasets (inner join)...")
aligned = xr.align(*datasets_to_align, join="inner")
obs_use = aligned[0]
crps_models = {name: aligned[i+1] for i, name in enumerate(names_order)}

# --- Masking (NEW) ---
if mask_type != "all":
    LSM_PATH = "/scratch2/mg963/data/ecmwf/era5/constants/era5_lsm.nc"
    print(f"🎭 Applying {mask_type} mask from {LSM_PATH}...")
    try:
        ds_lsm = xr.open_dataset(LSM_PATH)
        lsm = ds_lsm["lsm"]
        
        # Handle time if present in LSM
        if "time" in lsm.dims:
            lsm = lsm.isel(time=0, drop=True)
        elif "valid_time" in lsm.dims:
            lsm = lsm.isel(valid_time=0, drop=True)

        # Standardize longitude to [-180, 180]
        if lsm.longitude.max() > 180:
            lsm = lsm.assign_coords(longitude=(((lsm.longitude + 180) % 360) - 180)).sortby("longitude")
        
        # Align LSM with data grid
        lsm_aligned = lsm.reindex_like(obs_use, method="nearest")
        
        if mask_type == "land":
            mask = lsm_aligned > 0.5
        else:  # ocean
            mask = lsm_aligned <= 0.5
            
        # Apply mask to obs and all CRPS models
        obs_use = obs_use.where(mask)
        crps_models = {name: crps.where(mask) for name, crps in crps_models.items()}
        print(f"   Mask applied. Grid points remaining: {int(mask.sum())}")
        
    except Exception as e:
        print(f"   ⚠️  Failed to apply mask: {e}. Proceeding without mask.")

# --- Time Selection (Final) ---
if start_date or end_date:
    print(f"   Applying final time selection: {start_date or 'start'} to {end_date or 'end'}...")
    # time is already sorted above
    obs_use = obs_use.sel(time=slice(start_date, end_date))
    crps_models = {name: crps.sel(time=slice(start_date, end_date)) for name, crps in crps_models.items()}

print(f"   Aligned shape: {obs_use.sizes}")

# Rechunk with smaller time chunks to avoid OOM
print("   Rechunking for memory-efficient computation...")
obs_use = obs_use.chunk({"time": 50, "latitude": 200, "longitude": 200})
crps_models = {
    name: crps.chunk({"time": 50, "latitude": 200, "longitude": 200})
    for name, crps in crps_models.items()
}

# Precompute weights (reused across all models)
print("   Preparing area weights...")
# We'll use weights_lat directly in the sum to save memory
# weights_lat has dimension 'latitude' (or 'lat')


# -----------------------------------------
# 4. Compute threshold-weighted CRPS for each model & threshold
# -----------------------------------------

def twcrps_for_model_efficient(crps_da, obs_da, thresholds_native, weights_lat, lat_name="latitude"):
    """
    Compute threshold-weighted CRPS one threshold at a time (memory-efficient).

    crps_da: CRPS DataArray (time, lat, lon)
    obs_da:  obs DataArray (time, lat, lon)
    thresholds_native: 1D array of thresholds in native units
    weights_lat: 1D DataArray of weights (lat)
    """
    if lat_name not in crps_da.coords:
        lat_name = "lat"

    twcrps_vals = []
    
    for i, thr in enumerate(thresholds_native):
        print(f"      Threshold {i+1}/{len(thresholds_native)}: {thr:.2f}", end="\r")
        
        # Create mask for this threshold (3D: time, lat, lon)
        mask = obs_da >= thr
        
        # Compute weighted sum: sum over (time, lon) first, then apply lat weights
        # This is more memory efficient than broadcasting weights to 3D
        
        # 1. Sum CRPS * mask over time and longitude
        # We use .where(mask, 0.0) to treat NaNs as 0 in the sum
        num_lat = (crps_da.where(mask, 0.0)).sum(dim=("time", "longitude"))
        # 2. Sum mask over time and longitude
        den_lat = (mask.astype(float)).sum(dim=("time", "longitude"))
        
        # 3. Apply latitude weights and sum
        num = (num_lat * weights_lat).sum(dim=lat_name)
        den = (den_lat * weights_lat).sum(dim=lat_name)
        
        # Compute both together
        num_val, den_val = compute(num, den)
        
        # Avoid divide-by-zero
        if den_val > 0:
            twcrps_vals.append(float(num_val / den_val))
        else:
            twcrps_vals.append(np.nan)
    
    print()  # newline after progress
    return np.array(twcrps_vals)

print("\n🔹 Computing threshold-weighted CRPS...")
twcrps_results = {}

for name, crps_da in crps_models.items():
    print(f"   {name}...")
    twcrps = twcrps_for_model_efficient(
        crps_da, obs_use, thresholds_native, weights_lat, lat_name=lat_name
    )
    twcrps_results[name] = twcrps


# -----------------------------------------
# 5. Package results as Dataset and save
# -----------------------------------------

print("\n🔹 Packaging results...")

# Sanitize units for netCDF coordinate names (replace / with _per_)
obs_units_clean = var_cfg['obs_units'].replace('/', '_per_')
display_units_clean = var_cfg['display_units'].replace('°', 'deg').replace('/', '_per_')

ds_tw = xr.Dataset(
    data_vars={
        f"twcrps_{name}": (["threshold"], vals)
        for name, vals in twcrps_results.items()
    },
    coords={
        f"threshold_{obs_units_clean}": (["threshold"], thresholds_native),
        f"threshold_{display_units_clean}": (["threshold"], thresholds_display),
        "quantile":    (["threshold"], q_list_use),
    },
)

# Skill scores relative to IFS-ENS (1 - model/ifs)
if "IFS-ENS" in twcrps_results:
    ifs_vals = twcrps_results["IFS-ENS"].astype(float)
    for name, vals in twcrps_results.items():
        if name == "IFS-ENS":
            continue
        vals = vals.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            skill = 1.0 - (vals / ifs_vals)
        ds_tw[f"skill_vs_ifs_{name}"] = (["threshold"], skill)

# Add attributes for clarity
ds_tw.attrs["variable"] = variable
ds_tw.attrs["variable_long_name"] = var_cfg["long_name"]
ds_tw.attrs["lead_hours"] = lead_hours
ds_tw.attrs["obs_units"] = var_cfg["obs_units"]
ds_tw.attrs["display_units"] = var_cfg["display_units"]
ds_tw.attrs["crps_units"] = var_cfg["crps_units"]

print(ds_tw)

if OUT_DATA.exists():
    print(f"   ⚠️  Removing existing file {OUT_DATA}")
    OUT_DATA.unlink()

print(f"💾 Saving results to {OUT_DATA}")
ds_tw.to_netcdf(OUT_DATA)


# -----------------------------------------
# 6. Plot: threshold (°C) vs twCRPS
# -----------------------------------------

print(f"📈 Plotting to {OUT_FIG} ...")

if plt is None:
    print("   ⚠️  matplotlib is not available; skipping plot output.")
    print("✅ Done.")
    raise SystemExit(0)

plt.figure(figsize=(7, 5))

mask_title = f" ({mask_type.capitalize()})" if mask_type != "all" else ""

for name, vals in twcrps_results.items():
    plt.plot(thresholds_display, vals, marker="o", label=name)

plt.xlabel(f"Threshold {var_cfg['threshold_label']} ({var_cfg['display_units']})")
plt.ylabel(f"Threshold-weighted {var_cfg['crps_label']}")
model_names_str = ", ".join(twcrps_results.keys())
plt.title(f"{lead_hours}h {var_cfg['long_name']} threshold-weighted CRPS{mask_title}\n({model_names_str})")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150)

print("✅ Done.")


# -----------------------------------------
# 7. Plot: threshold vs Skill (relative to IFS-ENS)
# -----------------------------------------

if "IFS-ENS" in twcrps_results:
    print(f"📈 Plotting skill vs IFS-ENS to {OUT_FIG_SKILL} ...")
    ifs_vals = twcrps_results["IFS-ENS"].astype(float)

    plt.figure(figsize=(7, 5))
    for name in ["GenCast", "WN2"]:
        if name not in twcrps_results:
            continue
        vals = twcrps_results[name].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            skill = 1.0 - (vals / ifs_vals)
        plt.plot(thresholds_display, skill, marker="o", label=name)

    plt.axhline(0.0, color="k", linewidth=1, alpha=0.4)
    plt.xlabel(f"Threshold {var_cfg['threshold_label']} ({var_cfg['display_units']})")
    plt.ylabel("Skill vs IFS-ENS (1 - twCRPS / twCRPS_IFS)")
    ai_models_plotted = [name for name in ["GenCast", "WN2"] if name in twcrps_results]
    skill_title_models = ", ".join(ai_models_plotted)
    plt.title(f"{lead_hours}h {var_cfg['long_name']} skill vs IFS-ENS{mask_title}\n({skill_title_models})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG_SKILL, dpi=150)

    print("✅ Skill plot saved.")