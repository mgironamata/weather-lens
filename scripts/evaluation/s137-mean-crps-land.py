import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------------
# CONFIG
# -----------------------------------------
VARIABLE = "t2m"
LEAD = 24
START_DATE = "2024-02-04"
END_DATE = "2024-12-31"

LSM_PATH = "/scratch2/mg963/data/ecmwf/era5/constants/era5_lsm.nc"

CRPS_PATHS = {
    "IFS": f"/scratch2/mg963/results/ecmwf/ifs-ens/crps_{VARIABLE}_{LEAD}h.zarr",
    "GenCast": f"/scratch2/mg963/results/weathernext/gencast/crps_{VARIABLE}_{LEAD}h.zarr",
    "WN2": f"/scratch2/mg963/results/weathernext/WN2/crps_{VARIABLE}_{LEAD}h.zarr"
}

def open_crps(path):
    """Robust loader for CRPS Zarr files."""
    try:
        ds = xr.open_zarr(path, consolidated=True)
    except Exception:
        ds = xr.open_zarr(path, consolidated=False)
    
    # Identify variable (handles different naming conventions)
    var_names = list(ds.data_vars)
    if f"{VARIABLE}_crps" in var_names:
        da = ds[f"{VARIABLE}_crps"]
    elif "2m_temperature_crps" in var_names:
        da = ds["2m_temperature_crps"]
    elif len(var_names) == 1:
        da = ds[var_names[0]]
    else:
        raise ValueError(f"Could not identify CRPS variable in {path}")
    
    return da.sortby("time")

def main():
    print(f"🌍 Computing Mean Land CRPS for {VARIABLE} @ {LEAD}h")
    print(f"📅 Period: {START_DATE} to {END_DATE}")

    # 1. Load Land-Sea Mask
    print(f"🎭 Loading LSM from {LSM_PATH}...")
    ds_lsm = xr.open_dataset(LSM_PATH)
    
    # Handle different time coordinate names in LSM
    time_name = "time" if "time" in ds_lsm.dims else ("valid_time" if "valid_time" in ds_lsm.dims else None)
    if time_name:
        lsm = ds_lsm["lsm"].isel({time_name: 0}, drop=True)
    else:
        lsm = ds_lsm["lsm"]
    
    # Standardize longitude to [-180, 180] if needed
    if lsm.longitude.max() > 180:
        lsm = lsm.assign_coords(longitude=(((lsm.longitude + 180) % 360) - 180)).sortby("longitude")

    # 2. Load and Align CRPS Datasets
    das = {}
    for name, path in CRPS_PATHS.items():
        print(f"📂 Loading {name}...")
        das[name] = open_crps(path).sel(time=slice(START_DATE, END_DATE))

    # Align all models and the mask
    print("🔗 Aligning datasets...")
    # We align the models first
    da_ifs, da_gencast, da_wn2 = xr.align(das["IFS"], das["GenCast"], das["WN2"], join="inner")
    
    # Reindex LSM to match the data grid
    lsm_aligned = lsm.reindex_like(da_ifs, method="nearest")
    land_mask = lsm_aligned > 0.5

    # 3. Compute Area Weights
    weights = np.cos(np.deg2rad(da_ifs.latitude))
    weights.name = "weights"

    # 4. Compute Mean CRPS over Land
    results = {}
    models = {"IFS": da_ifs, "GenCast": da_gencast, "WN2": da_wn2}
    
    print("\n📊 Results (Mean Land CRPS):")
    print("-" * 30)
    
    for name, da in models.items():
        # Mask for land only
        da_land = da.where(land_mask)
        
        # Weighted mean over space (lat, lon) and simple mean over time
        # First mean over space
        spatial_mean = da_land.weighted(weights).mean(dim=("latitude", "longitude"))
        # Then mean over time
        final_mean = spatial_mean.mean(dim="time").compute().item()
        
        results[name] = final_mean
        print(f"{name:<10}: {final_mean:.4f}")

    print("-" * 30)

if __name__ == "__main__":
    main()
