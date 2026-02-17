import xarray as xr
import gcsfs
import pandas as pd
import os
import shutil
import numpy as np

# ==============================================================================
# 1. CONFIGURATION PARAMETERS
# ==============================================================================
CONFIG = {
    # --- SAFETY FLAG ---
    "PERFORM_DOWNLOAD": True, 

    # Google Cloud Project ID
    "project_id": "weathernextforecasts", 
    
    # Input Bucket Base Path
    "bucket_base": "gs://weathernext/126478713_1_0/zarr/126478713_2024_to_present",
    
    # Output Paths
    "zarr_store_path": "/scratch2/mg963/data/weathernext/gencast/gencast_2024_global.zarr",
    "temp_nc_prefix":  "/scratch2/mg963/data/weathernext/gencast/temp_gencast_", 

    # Filter Settings
    "start_date": "2024-02-02",
    "end_date":   "2024-06-30",
    "init_hour":  6,  # 06:00 UTC
    
    # Variables
    "variables": [
        'total_precipitation_12hr',
        '2m_temperature',
        '10m_u_component_of_wind', 
        '10m_v_component_of_wind'
    ],
    
    # Forecast Steps (INTEGERS)
    "steps": [24, 72],

    # Chunking Strategy
    "chunk_plan": {
        'time': 1, 
        'sample': -1,  # Keep all 64 samples together
        'lat': 360, 
        'lon': 720
    }
}

# ==============================================================================
# HELPER: Sanitize Attributes
# ==============================================================================
def clean_attributes(ds):
    """Converts boolean attributes to integers to prevent NetCDF errors."""
    for key, val in ds.attrs.items():
        if isinstance(val, (bool, np.bool_)):
            ds.attrs[key] = int(val)
    for var in ds.variables:
        for key, val in ds[var].attrs.items():
            if isinstance(val, (bool, np.bool_)):
                ds[var].attrs[key] = int(val)
    return ds

# ==============================================================================
# 2. MAIN SCRIPT
# ==============================================================================
def main():
    print(f"--- Starting GenCast Download Job ---")
    print(f"Mode: {'DOWNLOAD ENABLED' if CONFIG['PERFORM_DOWNLOAD'] else 'SAFE MODE'}")
    
    # --- FIX: Ensure Output Directories Exist ---
    # This fixes the [Errno 13] Permission denied if the folder is missing
    output_dir = os.path.dirname(CONFIG['temp_nc_prefix'])
    zarr_dir = os.path.dirname(CONFIG['zarr_store_path'])
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(zarr_dir, exist_ok=True)
        print(f"Output directory confirmed: {output_dir}")
    except Exception as e:
        print(f"\nCRITICAL ERROR: Could not create output directory {output_dir}")
        print(f"System reported: {e}")
        return

    # Auth check
    try:
        fs = gcsfs.GCSFileSystem(project=CONFIG['project_id'], token='google_default')
    except Exception as e:
        print(f"Auth Error: {e}")
        return

    # Generate target dates
    target_dates = pd.date_range(
        start=CONFIG['start_date'], 
        end=CONFIG['end_date'], 
        freq='D'
    )
    
    print(f"Targeting {len(target_dates)} daily forecasts at {CONFIG['init_hour']}Z.")

    # --- DOWNLOAD LOOP ---
    for i, date in enumerate(target_dates):
        date_str = date.strftime('%Y%m%d')
        folder_name = f"{date_str}_{CONFIG['init_hour']:02d}hr_01_preds"
        remote_zarr_path = f"{CONFIG['bucket_base']}/{folder_name}/predictions.zarr"
        
        temp_file = f"{CONFIG['temp_nc_prefix']}{date_str}.nc"
        
        print(f"[{i+1}/{len(target_dates)}] Checking {folder_name}...", end=" ", flush=True)

        if not fs.exists(remote_zarr_path):
            print("Not Found (Skipping).")
            continue

        if not CONFIG['PERFORM_DOWNLOAD']:
            print("Found (Safe Mode).")
            continue

        try:
            # 1. Open Daily Zarr
            ds = xr.open_dataset(
                remote_zarr_path, 
                engine='zarr', 
                chunks={}, 
                decode_times=False, 
                backend_kwargs={'consolidated': True}
            )

            # --- CRITICAL ADJUSTMENTS FOR GENCAST STRUCTURE ---
            
            # A. Rename 'time' -> 'step'
            if 'time' in ds.dims and 'step' not in ds.dims:
                ds = ds.rename({'time': 'step'})
            
            # B. Select Variables & Steps
            subset = ds[CONFIG['variables']]
            subset = subset.sel(step=CONFIG['steps']) 

            # C. Create the Initialization Time Dimension
            init_time_val = date + pd.Timedelta(hours=CONFIG['init_hour'])
            subset = subset.expand_dims(time=[init_time_val])

            # 4. Sanitize & Save NetCDF
            subset = clean_attributes(subset)
            comp = dict(zlib=True, complevel=4)
            encoding = {var: comp for var in subset.data_vars}
            subset.to_netcdf(temp_file, encoding=encoding)
            
            # 5. Append to Master Zarr
            with xr.open_dataset(temp_file) as local_ds:
                local_ds = local_ds.chunk(CONFIG['chunk_plan'])
                
                for var in local_ds.data_vars:
                    if 'chunks' in local_ds[var].encoding:
                        del local_ds[var].encoding['chunks']

                if not os.path.exists(CONFIG['zarr_store_path']):
                    local_ds.to_zarr(CONFIG['zarr_store_path'], mode='w', consolidated=True)
                else:
                    local_ds.to_zarr(CONFIG['zarr_store_path'], mode='a', append_dim='time', consolidated=True)

            # 6. Cleanup
            os.remove(temp_file)
            print("Done.")

        except KeyError as e:
            print(f"\nVariable Error: {e}. Check names!")
            break
        except Exception as e:
            print(f"\nFAILED on {date_str}: {e}")
            continue

    print("\nJob Complete.")
    
    if os.path.exists(CONFIG['zarr_store_path']):
        xr.open_zarr(CONFIG['zarr_store_path'], consolidated=True)
        print("Verification successful.")

if __name__ == "__main__":
    main()