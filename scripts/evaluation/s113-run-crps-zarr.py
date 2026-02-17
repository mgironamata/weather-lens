import xarray as xr
import numpy as np
import time
import dask
from pathlib import Path

# --- PERFORMANCE TUNING ---
# Limit threads to prevent I/O thrashing on /scratch2
dask.config.set(scheduler='threads', num_workers=8)

# ==========================================
# 1. ROBUST DATA LOADER
# ==========================================

def open_as_dataarray(zarr_path, variable=None, sel_dict=None, rename_coords=None, standardize_lon=False, to_valid_time=False):
    """
    Opens Zarr, extracts variable, handles slicing, renaming, lon-fix, and time-shift.
    INCLUDES FIX FOR DUPLICATE DATES.
    """
    ds = xr.open_zarr(zarr_path, consolidated=True)
    
    # 1. Variable Selection
    var_names = list(ds.data_vars)
    if variable and variable in var_names:
        var_name = variable
    elif len(var_names) == 1:
        var_name = var_names[0]
    else:
        raise ValueError(f"Cannot identify variable in {zarr_path}. Choices: {var_names}")
    
    da = ds[var_name]

    # 2. 🧹 CLEANUP: Drop Duplicates
    # This fixes the ValueError you just saw. It keeps the first occurrence.
    if not da.indexes['time'].is_unique:
        print(f"   ⚠️  WARNING: Found duplicate times in {Path(zarr_path).name}. Dropping duplicates...")
        da = da.drop_duplicates(dim='time')

    # 3. Dimensional Slicing
    if sel_dict:
        da = da.sel(sel_dict)
    
    # 4. Convert to Validity Time (Forecast only)
    if to_valid_time:
        if 'prediction_timedelta' in da.coords:
            valid_time = da.time + da.prediction_timedelta
            da = da.assign_coords(valid_time=valid_time)
            da = da.swap_dims({'time': 'valid_time'})
            da = da.drop_vars('time') 
            da = da.rename({'valid_time': 'time'})
            
            # Check for duplicates AGAIN after shifting time
            if not da.indexes['time'].is_unique:
                 da = da.drop_duplicates(dim='time')
        else:
            print("   ⚠️  WARNING: Requested valid_time shift but 'prediction_timedelta' not found.")

    # 5. Coordinate Renaming
    if rename_coords:
        clean_rename = {k: v for k, v in rename_coords.items() if k in da.dims or k in da.coords}
        da = da.rename(clean_rename)

    # 6. Longitude Standardization
    if standardize_lon:
        lon_name = 'longitude'
        if lon_name in da.coords:
            new_lon = (da[lon_name] + 180) % 360 - 180
            da = da.assign_coords({lon_name: new_lon}).sortby(lon_name)
    
    da.name = "data"
    return da

# ==========================================
# 2. CRPS CALCULATION
# ==========================================

def crps_fair_fast(forecast, obs, member_dim="number"):
    # Force float32 to save RAM
    forecast = forecast.astype(np.float32)
    obs = obs.astype(np.float32)
    
    # Term 1: MAE
    term1 = np.abs(forecast - obs).mean(dim=member_dim)

    # Term 2: Spread (Sorted)
    def _compute_spread_sorted(a):
        a_sorted = np.sort(a, axis=-1)
        M = a.shape[-1]
        if M < 2: return np.zeros_like(a[..., 0])
        i = np.arange(M)
        weights = 2 * i - M + 1  
        weighted_sum = np.sum(a_sorted * weights, axis=-1)
        return weighted_sum / (M * (M - 1))
    
    term2 = xr.apply_ufunc(
        _compute_spread_sorted, 
        forecast, 
        input_core_dims=[[member_dim]], 
        output_core_dims=[[]], 
        vectorize=True, 
        dask="parallelized", 
        output_dtypes=[np.float32]
    )
    return term1 - term2

# ==========================================
# 3. BATCHED PIPELINE (Solves Memory Issues)
# ==========================================

def run_batched_pipeline(config, batch_size_days=30):
    
    spec = MODEL_SPECS[config['model_name']]
    print(f"🔹 Loading Forecast: {spec['path']}")
    da_fcst = open_as_dataarray(spec['path'], **config['forecast_kwargs'])
    
    print(f"🔹 Loading Obs: {config['obs_path']}")
    da_obs  = open_as_dataarray(config['obs_path'], **config['obs_kwargs'])
    
    # Align (Inner Join)
    print("🔹 Aligning datasets...")
    da_fcst, da_obs = xr.align(da_fcst, da_obs, join="inner")
    
    total_times = da_fcst.sizes['time']
    if total_times == 0:
        print("❌ ALIGNMENT FAILED: Intersection is empty.")
        return

    print(f"🚀 Processing {total_times} days in batches of {batch_size_days}...")

    # BATCH LOOP
    for i in range(0, total_times, batch_size_days):
        
        # Slice Batch
        sl = slice(i, i + batch_size_days)
        fcst_batch = da_fcst.isel(time=sl)
        obs_batch  = da_obs.isel(time=sl)
        
        current_dates = fcst_batch.time.values
        print(f"   ⚙️  Computing Batch {i//batch_size_days + 1}: {current_dates[0]} to {current_dates[-1]}")
        
        # Compute
        da_crps = crps_fair_fast(fcst_batch, obs_batch, member_dim=spec['member_dim'])
        
        # Naming
        var_name = config['forecast_kwargs'].get('variable', 'data')
        da_crps.name = f"{var_name}_crps"
        
        # Chunking
        chunks = {'time': 1, 'latitude': -1, 'longitude': -1}
        da_crps = da_crps.chunk(chunks)

        # Save / Append
        mode = 'w' if i == 0 else 'a'
        append_dim = None if i == 0 else 'time'
        
        da_crps.to_dataset().to_zarr(
            config['output_path'], 
            mode=mode, 
            append_dim=append_dim, 
            consolidated=True
        )
        print(f"      ✅ Saved batch.")

    print(f"🎉 DONE! Saved to {config['output_path']}")

# ==========================================
# 4. CONFIGURATION
# ==========================================

BASE_OUTPUT_DIR = Path("/scratch2/mg963/results")

VAR_MAP = {
    't2m': {'wn2': '2m_temperature',          'ifs': 't2m'},
    # Add other vars here...
}

MODEL_SPECS = {
    'WN2': {
        'path': "/scratch2/mg963/data/weathernext/wn2/wn2_2024_global.zarr",
        'output_subpath': "weathernext/WN2",
        'member_dim': 'sample',
        'standardize_lon': True,
        'rename_coords': {'lat': 'latitude', 'lon': 'longitude'},
        'needs_time_selection': True,
        'to_valid_time': True
    },
    'IFS_ENS': {
        'path': "/scratch2/mg963/data/ecmwf/ens/ifs_enqs.zarr",
        'output_subpath': "ecmwf/ifs-ens",
        'member_dim': 'number',
        'standardize_lon': False,
        'rename_coords': {},
        'needs_time_selection': False,
        'to_valid_time': False
    }
}

OBS_PATH = "/scratch2/mg963/data/ecmwf/analysis/ifs_ens_control_06z.zarr"

# ==========================================
# 5. EXECUTION
# ==========================================

SELECTED_MODEL = 'WN2'
SELECTED_VAR   = 't2m'
SELECTED_LEAD  = 72

# Build Config Dict
spec = MODEL_SPECS[SELECTED_MODEL]
fcst_kwargs = {
    'variable': VAR_MAP[SELECTED_VAR]['wn2'] if SELECTED_MODEL == 'WN2' else VAR_MAP[SELECTED_VAR]['ifs'],
    'standardize_lon': spec['standardize_lon'],
    'rename_coords': spec['rename_coords'],
    'to_valid_time': spec.get('to_valid_time', False)
}
if spec['needs_time_selection']:
    fcst_kwargs['sel_dict'] = {'prediction_timedelta': np.timedelta64(SELECTED_LEAD, 'h')}

obs_kwargs = {'variable': VAR_MAP[SELECTED_VAR]['ifs'], 'standardize_lon': False}

out_dir = BASE_OUTPUT_DIR / spec['output_subpath']
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"crps_{SELECTED_VAR}_{SELECTED_LEAD}h.zarr"

config = {
    'model_name': SELECTED_MODEL,
    'obs_path': OBS_PATH,
    'output_path': str(out_path),
    'forecast_kwargs': fcst_kwargs,
    'obs_kwargs': obs_kwargs
}

# Run
start = time.time()
run_batched_pipeline(config, batch_size_days=30)
print(f"⏱️  Total Time: {(time.time()-start)/60:.2f} min")