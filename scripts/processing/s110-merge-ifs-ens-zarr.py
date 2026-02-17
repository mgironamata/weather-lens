import xarray as xr
import pandas as pd
from pathlib import Path
import cfgrib
import os

# --- CONFIGURATION ---
source_dir = Path("/scratch2/mg963/data/ecmwf/ensembles/ifs/raw")
zarr_output_dir = Path("/scratch2/mg963/data/ecmwf/ensembles/ifs/zarr_ens")
start_date = "2024-02-03" 
end_date = "2024-12-31"

steps = [18, 24, 66, 72]

# Ensure output exists
zarr_output_dir.mkdir(parents=True, exist_ok=True)

variable_map = {
    '2t': 't2m', 't2m': 't2m', 
    'tp': 'tp',
    '10u': 'u10', 'u10': 'u10', 'p165': 'u10',
    '10v': 'v10', 'v10': 'v10', 'p166': 'v10'
}

# 🛠️ FIX: Removed 'step' from chunks
chunks = {
    'time': 1, 
    'number': -1, 
    'latitude': -1, 
    'longitude': -1
}

dates = pd.date_range(start=start_date, end=end_date, freq="D")

print(f"🏭 Processing Ensemble (1-50) to Zarr...")

for date in dates:
    for step in steps:
        filename = f"ifs-ens_{date.strftime('%Y%m%d')}_06z_f{step:02d}.grib2"
        file_path = source_dir / filename

        if not file_path.exists():
            continue

        try:
            # 1. Open GRIBs
            ds_list = cfgrib.open_datasets(file_path, backend_kwargs={'indexpath': ''})
            
            if not ds_list:
                continue

            # 2. Filter: Keep ONLY Ensemble datasets (size > 1)
            ensemble_parts = []
            for ds in ds_list:
                if 'number' in ds.dims and ds.sizes['number'] > 1:
                    
                    # We drop 'step' here, which caused your previous error
                    ds = ds.drop_vars(['heightAboveGround', 'surface', 'valid_time', 'step'], errors='ignore')
                    
                    current_vars = list(ds.data_vars)
                    rename_dict = {k: variable_map[k] for k in current_vars if k in variable_map}
                    ds = ds.rename(rename_dict)
                    
                    ensemble_parts.append(ds)

            if not ensemble_parts:
                print(f"⚠️  No ensemble members found in {filename}")
                continue

            # 3. Merge Parts
            ds_ens = xr.merge(ensemble_parts, compat='override')

            # 4. Final Polish
            if 'time' not in ds_ens.dims:
                ds_ens = ds_ens.expand_dims(dim='time')

            # 5. Save Individual Variables
            target_vars = ['t2m', 'tp', 'u10', 'v10']
            
            for var in target_vars:
                if var in ds_ens:
                    ds_single = ds_ens[[var]]
                    
                    zarr_name = f"{var}_f{step:02d}.zarr"
                    zarr_path = zarr_output_dir / zarr_name

                    # Apply chunks (now safe because 'step' is removed)
                    ds_single = ds_single.chunk(chunks)

                    if not zarr_path.exists():
                        ds_single.to_zarr(zarr_path, mode='w', consolidated=True)
                        print(f"   🆕 Created {zarr_name} ({date.strftime('%Y-%m-%d')})")
                    else:
                        ds_single.to_zarr(zarr_path, append_dim='time', consolidated=True)
                        # print(f"   ➕ Appended to {zarr_name}")

        except Exception as e:
            print(f"❌ Error on {filename}: {e}")
            import traceback; traceback.print_exc()

print("✅ Conversion Complete.")