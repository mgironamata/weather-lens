import zarr
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ZARR_STORE_PATH = "/scratch2/mg963/data/weathernext/wn2/wn2_2024_global.zarr"

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
def main():
    print(f"Opening Zarr store: {ZARR_STORE_PATH}")
    
    # 1. Open using Xarray first to read the time coordinates easily
    # We use consolidated=False to ensure we see the raw latest state
    try:
        ds = xr.open_zarr(ZARR_STORE_PATH, consolidated=False)
    except Exception:
        # Fallback if metadata is messy
        ds = xr.open_zarr(ZARR_STORE_PATH, consolidated=True)

    times = pd.to_datetime(ds.time.values)
    total_len = len(times)
    
    print(f"\nTotal Timesteps: {total_len}")
    
    # 2. Show the tail of the dataset (The "Crime Scene")
    # We look at the last 20 entries to find where the duplicates start
    lookback = 20
    print(f"\n--- Last {lookback} entries in the store ---")
    print(f"{'INDEX':<8} | {'DATE':<20}")
    print("-" * 30)
    
    for i in range(total_len - lookback, total_len):
        t_str = times[i].strftime('%Y-%m-%d %H:%M')
        print(f"{i:<8} | {t_str}")

    print("-" * 30)
    print("\nCheck the list above.")
    print("Identify the FIRST index you want to DELETE.")
    print("(Everything from that index onwards will be removed)")
    
    # 3. Ask for user input to prevent accidents
    cut_index_input = input("\nEnter the Index to CUT at (or press Enter to cancel): ")
    
    if not cut_index_input.strip().isdigit():
        print("Cancelled.")
        return

    cut_index = int(cut_index_input)
    
    if cut_index >= total_len:
        print("Error: Cut index is beyond the end of the array.")
        return
        
    print(f"\nWARNING: You are about to resize the store from {total_len} to {cut_index}.")
    print(f"This will DELETE {total_len - cut_index} timesteps permanently.")
    confirm = input("Type 'yes' to confirm: ")
    
    if confirm.lower() != 'yes':
        print("Cancelled.")
        return

    # 4. PERFORM THE SURGERY (Using Zarr directly)
    # We must resize every array that shares the 'time' dimension.
    
    # Open the store in Read/Write mode
    store = zarr.open_group(ZARR_STORE_PATH, mode='r+')
    
    print("\nResizing arrays...")
    resized_count = 0
    
    # Iterate through all arrays (variables + coords) in the store
    for name, arr in store.arrays():
        # Check if the array shape matches the total time length (dimension 0)
        # We assume 'time' is always the first dimension (index 0)
        if arr.shape[0] == total_len:
            # Construct the new shape as a single tuple
            new_shape = (cut_index,) + arr.shape[1:]
            
            print(f" - Resizing {name} from {arr.shape} -> {new_shape}")
            arr.resize(new_shape)
            resized_count += 1
            
    print(f"\nSuccessfully resized {resized_count} arrays.")
    
    # 5. CLEANUP: Consolidate Metadata
    # This updates the hidden .zmetadata file so Xarray doesn't get confused later
    print("Consolidating metadata...")
    zarr.consolidate_metadata(ZARR_STORE_PATH)
    
    print("Done. The duplicates are gone.")

if __name__ == "__main__":
    main()