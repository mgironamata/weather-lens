#!/usr/bin/env python
"""
Download ERA5 daily total precipitation (UTC+6) for 2024.

Uses the derived-era5-single-levels-daily-statistics dataset,
which directly provides daily sums at a specified time zone.
"""

import os
from pathlib import Path
import cdsapi
import xarray as xr
import numpy as np

# ---- Configuration ----
out_dir = Path("/scratch2/mg963/data/ecmwf/era5/tp/era5/daily_utc6_2024")
out_dir.mkdir(parents=True, exist_ok=True)

year = "2024"
variable = "total_precipitation"
dataset = "derived-era5-single-levels-daily-statistics"

# All months and days for a full year
months = [f"{m:02d}" for m in range(1, 13)]
days = [f"{d:02d}" for d in range(1, 32)]

# ---- Step 1: Download daily sum data from CDS ----
target_file = out_dir / f"era5_tp_daily_utc6_{year}.nc"

if target_file.exists():
    print(f"[skip] {target_file} already exists")
else:
    print(f"[download] ERA5 daily total precipitation (UTC+6) for {year}")
    print(f"           Target: {target_file}")
    
    client = cdsapi.Client()
    
    request = {
        "product_type": "reanalysis",
        "variable": [variable],
        "year": year,
        "month": months,
        "day": days,
        "daily_statistic": "daily_sum",
        "time_zone": "utc+06:00",
        "frequency": "1_hourly"
    }
    
    client.retrieve(dataset, request, str(target_file))
    print(f"[done] Downloaded {target_file}")

# ---- Step 2: Open and inspect the data ----
print(f"\n[inspect] Opening {target_file}")
ds = xr.open_dataset(target_file)
print(ds)

# ---- Step 3: Convert units if needed and add metadata ----
# The CDS daily statistics dataset returns precipitation in meters
# Convert to mm for convenience
if "tp" in ds.data_vars:
    tp_var = "tp"
elif "total_precipitation" in ds.data_vars:
    tp_var = "total_precipitation"
else:
    print(f"Warning: Could not find precipitation variable. Available: {list(ds.data_vars)}")
    tp_var = list(ds.data_vars)[0]  # Use first variable as fallback

print(f"\n[process] Converting {tp_var} from meters to mm")
tp_daily = ds[tp_var]

# Check current units
current_units = tp_daily.attrs.get("units", "unknown")
print(f"Current units: {current_units}")

# Convert m to mm if needed
if current_units in ["m", "meters"]:
    tp_daily_mm = tp_daily * 1000.0
else:
    print(f"Warning: Unexpected units '{current_units}', assuming meters")
    tp_daily_mm = tp_daily * 1000.0

# Update metadata
tp_daily_mm = tp_daily_mm.rename("tp_daily_utc6_mm")
tp_daily_mm.attrs.update({
    "long_name": "Daily total precipitation (UTC+6 time zone)",
    "units": "mm/day",
    "source": "ERA5 derived daily statistics",
    "time_zone": "UTC+06:00",
    "daily_statistic": "sum",
    "description": "24-hour precipitation sum in UTC+6 time zone (06:00-06:00 local)"
})

# Create output dataset
out_ds = tp_daily_mm.to_dataset()

# ---- Step 4: Write processed NetCDF ----
out_file = out_dir / f"era5_tp_daily_utc6_{year}_processed.nc"
print(f"\n[write] Saving processed data to {out_file}")

encoding = {
    "tp_daily_utc6_mm": {
        "zlib": True,
        "complevel": 4,
        "_FillValue": np.nan,
        "dtype": "float32"
    }
}

out_ds.to_netcdf(out_file, encoding=encoding)
print(f"[done] Wrote {out_file}")

# ---- Step 5: Show summary statistics ----
print(f"\n[summary] Data summary:")
print(f"Time range: {tp_daily_mm.time.values[0]} to {tp_daily_mm.time.values[-1]}")
print(f"Number of time steps: {len(tp_daily_mm.time)}")
print(f"Spatial shape: {tp_daily_mm.shape}")
print(f"\nPrecipitation statistics (mm/day):")
print(f"  Mean:   {float(tp_daily_mm.mean()):.3f}")
print(f"  Median: {float(tp_daily_mm.median()):.3f}")
print(f"  Max:    {float(tp_daily_mm.max()):.3f}")
print(f"  Min:    {float(tp_daily_mm.min()):.3f}")

print("\n[complete] Script finished successfully")
