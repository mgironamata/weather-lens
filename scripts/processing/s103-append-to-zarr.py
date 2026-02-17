import xarray as xr
import os
import glob

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Path where the temporary NetCDF files are located
NC_SOURCE_DIR = "/scratch2/mg963/data/weathernext/wn2/"

# Path to your EXISTING Zarr store
ZARR_STORE_PATH = "/scratch2/mg963/data/weathernext/wn2/wn2_2024_global.zarr"

# Optimization Strategy (Must match what you used for the rest of the data)
CHUNK_PLAN = {
    'time': 1, 
    'sample': -1, 
    'lat': 360, 
    'lon': 720
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    # 1. Find the specific missing files (Jan 01 to Jan 05)
    # We use a specific pattern to avoid grabbing files you don't want
    pattern = os.path.join(NC_SOURCE_DIR, "temp_download_2024010*_06.nc")
    
    # glob returns unsorted list, so we MUST sort it to append Jan 1 before Jan 2
    files_to_append = sorted(glob.glob(pattern))
    
    # Optional: Filter strictly for days 01-05 if you have other junk in that folder
    files_to_append = [f for f in files_to_append if any(x in f for x in ['20240101', '20240102', '20240103', '20240104', '20240105'])]

    print(f"Found {len(files_to_append)} files to append:")
    for f in files_to_append:
        print(f" - {os.path.basename(f)}")

    if not os.path.exists(ZARR_STORE_PATH):
        print(f"\nERROR: Zarr store not found at {ZARR_STORE_PATH}")
        print("This script is for APPENDING to an existing store only.")
        return

    # 2. Loop and Append
    print("\nStarting Append Process...")
    
    for i, nc_file in enumerate(files_to_append):
        print(f"[{i+1}/{len(files_to_append)}] Appending {os.path.basename(nc_file)}...", end=" ", flush=True)
        
        try:
            with xr.open_dataset(nc_file) as local_ds:
                # A. Apply Chunking (Critical for Zarr alignment)
                local_ds = local_ds.chunk(CHUNK_PLAN)
                
                # B. Clean Encoding (Remove conflicting chunk specs from NetCDF)
                for var in local_ds.data_vars:
                    if 'chunks' in local_ds[var].encoding:
                        del local_ds[var].encoding['chunks']

                # C. Append
                # We use 'a' (append) and explicitly say we are extending 'time'
                local_ds.to_zarr(
                    ZARR_STORE_PATH, 
                    mode='a', 
                    append_dim='time', 
                    consolidated=True
                )
                
            print("Done.")
            
            # Optional: Delete file after success
            # os.remove(nc_file) 

        except Exception as e:
            print(f"FAILED: {e}")

    print("\nJob Complete.")
    
    # 3. Verification
    print("Verifying final dataset...")
    ds = xr.open_zarr(ZARR_STORE_PATH, consolidated=True)
    print(f"Total Timesteps: {ds.sizes['time']}")
    print("NOTE: Remember to use .sortby('time') when reading this dataset later!")

if __name__ == "__main__":
    main()