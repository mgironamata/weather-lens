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
    
    # Input Bucket
    "bucket_path": "gs://weathernext/weathernext_2_0_0/zarr/2024_to_2025/predictions.zarr",
    
    # Output Paths
    "zarr_store_path": "/scratch2/mg963/data/weathernext/wn2/wn2_2024_global.zarr",
    "temp_nc_prefix":  "/scratch2/mg963/data/weathernext/wn2/temp_download_", 

    # Filter Settings
    "start_date": "2025-01-01",
    "end_date":   "2025-02-01",
    "init_hour":  6,
    
    # Variables
    "variables": [
        'total_precipitation_6hr',
        '2m_temperature',
        '10m_u_component_of_wind', 
        '10m_v_component_of_wind'
    ],
    
    # Steps (24h and 72h)
    "steps": [pd.Timedelta("24h"), pd.Timedelta("72h")],

    # Chunking Strategy
    "chunk_plan": {
        'time': 1, 
        'sample': -1, 
        'lat': 360, 
        'lon': 720
    }
}

# ==============================================================================
# HELPER: Sanitize Attributes
# ==============================================================================
def clean_attributes(ds):
    """
    Converts boolean attributes to integers to prevent NetCDF errors.
    NetCDF4 does not support boolean attributes (b1).
    """
    # Clean Global Attributes
    for key, val in ds.attrs.items():
        if isinstance(val, (bool, np.bool_)):
            ds.attrs[key] = int(val)
            
    # Clean Variable Attributes
    for var in ds.variables:
        for key, val in ds[var].attrs.items():
            if isinstance(val, (bool, np.bool_)):
                ds[var].attrs[key] = int(val)
    return ds

# ==============================================================================
# 2. MAIN SCRIPT
# ==============================================================================
def main():
    print(f"--- Starting WeatherNext 2 Job ---")
    print(f"Mode: {'DOWNLOAD ENABLED' if CONFIG['PERFORM_DOWNLOAD'] else 'SAFE MODE'}")
    
    # Auth check
    try:
        fs = gcsfs.GCSFileSystem(project=CONFIG['project_id'], token='google_default')
    except Exception as e:
        print(f"Auth Error: {e}")
        return

    print("Opening remote bucket...")
    ds = xr.open_dataset(
        CONFIG['bucket_path'], 
        engine='zarr', 
        chunks={}, 
        backend_kwargs={'consolidated': True}
    )

    # --- PRE-FILTERING ---
    subset = ds[CONFIG['variables']]
    step_dim = 'prediction_timedelta' if 'prediction_timedelta' in ds.dims else 'step'
    subset = subset.sel({step_dim: CONFIG['steps']})

    all_times = pd.to_datetime(ds.time.values)
    valid_times = all_times[
        (all_times >= pd.to_datetime(CONFIG['start_date'])) & 
        (all_times <= pd.to_datetime(CONFIG['end_date'])) & 
        (all_times.hour == CONFIG['init_hour'])
    ]
    
    print(f"Found {len(valid_times)} valid initialization times.")

    if not CONFIG['PERFORM_DOWNLOAD']:
        print("SAFE MODE: Stopping before download.")
        return

    # --- DOWNLOAD LOOP ---
    for i, init_time in enumerate(valid_times):
        date_str = init_time.strftime('%Y%m%d_%H')
        temp_file = f"{CONFIG['temp_nc_prefix']}{date_str}.nc"
        
        print(f"[{i+1}/{len(valid_times)}] Processing {date_str}...", end=" ", flush=True)

        try:
            # STEP A: Slice
            time_loc = list(ds.time.values).index(init_time)
            daily_slice = subset.isel(time=slice(time_loc, time_loc+1))
            
            # FIX: Sanitize attributes before saving
            daily_slice = clean_attributes(daily_slice)
            
            # Save NetCDF
            comp = dict(zlib=True, complevel=4)
            encoding = {var: comp for var in daily_slice.data_vars}
            daily_slice.to_netcdf(temp_file, encoding=encoding)
            
            # STEP B: Append to Zarr
            with xr.open_dataset(temp_file) as local_ds:
                local_ds = local_ds.chunk(CONFIG['chunk_plan'])
                
                # Clean encoding
                for var in local_ds.data_vars:
                    if 'chunks' in local_ds[var].encoding:
                        del local_ds[var].encoding['chunks']

                if not os.path.exists(CONFIG['zarr_store_path']):
                    local_ds.to_zarr(CONFIG['zarr_store_path'], mode='w', consolidated=True)
                else:
                    local_ds.to_zarr(CONFIG['zarr_store_path'], mode='a', append_dim='time', consolidated=True)

            # STEP C: Cleanup
            os.remove(temp_file)
            print("Done.")

        except Exception as e:
            print(f"\nFAILED on {date_str}: {e}")
            continue

    print("\nJob Complete.")
    
    # Only verify if something was actually created
    if os.path.exists(CONFIG['zarr_store_path']):
        xr.open_zarr(CONFIG['zarr_store_path'], consolidated=True)
        print("Verification successful.")
    else:
        print("Warning: No Zarr store was created (all downloads failed).")

if __name__ == "__main__":
    main()