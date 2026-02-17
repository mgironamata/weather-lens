import xarray as xr
import pandas as pd
from pathlib import Path
import cfgrib
import os

# --- CONFIGURATION ---
source_dir = Path("/scratch2/mg963/data/ecmwf/analysis/raw")
start_date = "2024-02-01"
end_date = "2024-02-03"

# The "Correct" Expected Dimensions (for flagging issues)
EXPECTED_LON = 1440
EXPECTED_LAT = 721  # 0.25 degree global (90 to -90)

dates = pd.date_range(start=start_date, end=end_date, freq="D")

print(f"🔍 Scanning files in {source_dir}...\n")
print(f"{'Date':<12} | {'Status':<8} | {'Lat':<5} | {'Lon':<5} | {'Filename'}")
print("-" * 60)

for date in dates:
    filename = f"ifs_ens_cf_{date.strftime('%Y%m%d')}.grib2"
    file_path = source_dir / filename
    
    if not file_path.exists():
        # print(f"{date.strftime('%Y-%m-%d'):<12} | MISSING  | {'-':<5} | {'-':<5} | {filename}")
        continue
        
    try:
        # Open strictly to check metadata (no data loading)
        # We use open_datasets to handle the 2m/10m split, but just check the first one
        # as lat/lon should be identical across variables in the same file.
        datasets = cfgrib.open_datasets(file_path, backend_kwargs={'indexpath': ''})
        
        if not datasets:
            print(f"{date.strftime('%Y-%m-%d'):<12} | ❌ EMPTY | {'-':<5} | {'-':<5} | {filename}")
            continue

        # Peek at the first dataset found
        ds = datasets[0]
        
        lat_len = ds.sizes.get('latitude', 0)
        lon_len = ds.sizes.get('longitude', 0)
        
        # Determine status flag
        if lat_len == EXPECTED_LAT and lon_len == EXPECTED_LON:
            status = "✅ OK"
        else:
            status = "⚠️ BAD"

        print(f"{date.strftime('%Y-%m-%d'):<12} | {status}   | {lat_len:<5} | {lon_len:<5} | {filename}")

    except Exception as e:
        print(f"{date.strftime('%Y-%m-%d'):<12} | ❌ ERROR | {'-':<5} | {'-':<5} | {e}")

print("\nScan complete.")