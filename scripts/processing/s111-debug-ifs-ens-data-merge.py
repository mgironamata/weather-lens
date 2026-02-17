import xarray as xr
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
zarr_dir = Path("/scratch2/mg963/data/ecmwf/ensembles/ifs/zarr_ens")
start_date = "2024-02-01"
end_date = "2024-12-31" # The full range you EXPECT to have
target_var = "t2m_f24.zarr" # We check one variable as a proxy for all

# Path to the specific store
store_path = zarr_dir / target_var

print(f"🕵️‍♀️ Auditing Zarr Store: {store_path}")

if not store_path.exists():
    print("❌ Zarr store not found!")
else:
    # Open the store
    ds = xr.open_zarr(store_path, consolidated=True)
    
    # 1. Check Dimensions
    actual_lon = ds.sizes['longitude']
    actual_lat = ds.sizes['latitude']
    
    print(f"   Shape: Lat={actual_lat}, Lon={actual_lon}")
    if actual_lon == 1440:
        print("   ✅ Grid dimensions are correct (Global 0.25°).")
    else:
        print(f"   ⚠️  WARNING: Grid dimensions look suspicious! ({actual_lon})")

    # 2. Check Dates
    stored_dates = pd.DatetimeIndex(ds.time.values).normalize()
    expected_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # Calculate Missing
    # We find dates in 'expected' that are NOT in 'stored'
    missing_dates = expected_dates.difference(stored_dates)
    
    total_expected = len(expected_dates)
    total_stored = len(stored_dates)
    
    print(f"\n📅 Timeline Status:")
    print(f"   Expected Days: {total_expected}")
    print(f"   Stored Days:   {total_stored}")
    print(f"   Missing Days:  {len(missing_dates)}")
    
    if len(missing_dates) > 0:
        print("\n❌ MISSING DATES (First 10):")
        for d in missing_dates[:10]:
            print(f"   - {d.strftime('%Y-%m-%d')}")
        
        if len(missing_dates) > 10:
            print(f"   ... and {len(missing_dates) - 10} more.")
            
        print("\n💡 TIP: These are the dates you need to re-download or re-process.")
    else:
        print("\n🎉 COMPLETE! You have a full timeline.")