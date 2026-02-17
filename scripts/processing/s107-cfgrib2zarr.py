import xarray as xr
import pandas as pd
from pathlib import Path
import numpy as np
import cfgrib
import os

# --- CONFIGURATION ---
# Directory where your 'ifs_ens_cf_*.grib2' files are located
source_dir = Path("/scratch2/mg963/data/ecmwf/analysis/raw_06z")
zarr_path = "/scratch2/mg963/data/ecmwf/analysis/ifs_ens_control_06z.zarr"

# Filter for the Modern Era (Feb 1 2024 onwards) to avoid grid mismatches
# If you need Jan 2024, process it into a separate Zarr file (e.g. ifs_legacy.zarr)
start_date = "2024-02-03" 
end_date = "2024-12-31"

# Generate the list of dates to look for
dates = pd.date_range(start=start_date, end=end_date, freq="D")

print(f"🐢 Processing files from {source_dir} into {zarr_path}")

for date in dates:
    # Construct your filename (matching the naming convention you used to download)
    filename = f"ifs_ens_cf_{date.strftime('%Y%m%d')}_06z.grib2"
    file_path = source_dir / filename
    
    if not file_path.exists():
        # It's okay if some days are missing, just skip them
        print(f"⏭️  Skipping missing file: {filename}")
        continue
        
    try:
        # 1. Open the GRIB file
        # usage of open_datasets (plural) handles the mixed 2m/10m levels
        datasets = cfgrib.open_datasets(file_path)
        
        # 2. Merge 2m and 10m variables
        # compat='override' ignores the conflicting heightAboveGround coordinate
        ds = xr.merge(datasets, compat='override')
        
        # 3. Clean up Dimensions/Coords
        # Drop conflicting coordinates and expand time
        ds = ds.drop_vars(['heightAboveGround', 'step', 'valid_time', 'number'], errors='ignore')
        if 'time' not in ds.dims:
            ds = ds.expand_dims(dim='time')

        # 4. Standardize Variable Names
        rename_map = {'10u': 'u10', '10v': 'v10', '2t': 't2m'}
        # Only rename what actually exists in the file
        current_map = {k: v for k, v in rename_map.items() if k in ds}
        ds = ds.rename(current_map)

        # 5. 🌪️ Compute Wind Speed (Magnitude)
        # We do this NOW so it's saved permanently in the Zarr
        ds['ws10'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
        ds['ws10'].attrs = {'units': 'm s**-1', 'long_name': '10 metre wind speed'}

        # 6. Save to Zarr
        # Chunking: 1 time step, full spatial field (lat/lon = -1 means "all")
        # This is optimal for appending daily files.
        chunk_encoding = {
            'time': 1, 
            'latitude': -1, 
            'longitude': -1
        }
        
        # Apply chunking to all variables
        for var in ds.data_vars:
            ds[var] = ds[var].chunk(chunk_encoding)

        if not os.path.exists(zarr_path):
            # First write
            ds.to_zarr(zarr_path, mode='w', consolidated=True)
            print(f"🆕 Created Zarr: {date.strftime('%Y-%m-%d')}")
        else:
            # Append
            ds.to_zarr(zarr_path, append_dim='time', consolidated=True)
            print(f"➕ Appended:     {date.strftime('%Y-%m-%d')}")
            
    except Exception as e:
        print(f"❌ Error on {filename}: {e}")

print("✅ Processing Complete.")