import xarray as xr
import numpy as np
import os
import shutil
from pathlib import Path
import zarr

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    # Input zarr stores (12hr incremental precipitation)
    "zarr_12_36_48_60": "/scratch2/mg963/data/weathernext/gencast/gencast_2024_precip_12_36_48_60.zarr",
    "zarr_24_72": "/scratch2/mg963/data/weathernext/gencast/gencast_2024_precip.zarr",
    
    # Output zarr store for cumulative precipitation
    "output_zarr": "/scratch2/mg963/data/weathernext/gencast/gencast_2024_precip_cumulative_24_72.zarr",
    
    # Overwrite existing output
    "OVERWRITE_EXISTING": True,
    
    # Chunking strategy
    "chunk_plan": {
        "time": 8,
        "sample": -1,
        "lat": 90,
        "lon": 180,
    },
    
    # Batch size for writing (number of time steps per batch)
    "write_batch_size": 16,
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def clean_attributes(ds: xr.Dataset) -> xr.Dataset:
    """Converts boolean attributes to integers to prevent NetCDF/Zarr errors."""
    for key, val in list(ds.attrs.items()):
        if isinstance(val, (bool, np.bool_)):
            ds.attrs[key] = int(val)

    for var in ds.variables:
        attrs = ds[var].attrs
        for key, val in list(attrs.items()):
            if isinstance(val, (bool, np.bool_)):
                attrs[key] = int(val)

    return ds


# ==============================================================================
# MAIN PROCESSING FUNCTION
# ==============================================================================
def main():
    print("=" * 80)
    print("CUMULATIVE PRECIPITATION COMPUTATION")
    print("Summing incremental 12hr precipitation to get 24hr and 72hr cumulative totals")
    print("=" * 80)
    
    # Validate input paths
    zarr_intermediate = CONFIG["zarr_12_36_48_60"]
    zarr_endpoints = CONFIG["zarr_24_72"]
    output_zarr = CONFIG["output_zarr"]
    
    print(f"\nInput zarr (12, 36, 48, 60 hr steps): {zarr_intermediate}")
    print(f"Input zarr (24, 72 hr steps):          {zarr_endpoints}")
    print(f"Output zarr (cumulative 24, 72 hr):    {output_zarr}")
    
    if not os.path.exists(zarr_intermediate):
        print(f"\nERROR: Intermediate zarr not found at {zarr_intermediate}")
        return
    
    if not os.path.exists(zarr_endpoints):
        print(f"\nERROR: Endpoints zarr not found at {zarr_endpoints}")
        return
    
    # Handle output zarr
    output_dir = os.path.dirname(output_zarr)
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output_zarr):
        if CONFIG["OVERWRITE_EXISTING"]:
            print(f"\nRemoving existing output zarr: {output_zarr}")
            shutil.rmtree(output_zarr)
        else:
            print(f"\nERROR: Output zarr already exists at {output_zarr}")
            print("Set OVERWRITE_EXISTING=True to overwrite.")
            return
    
    # Open input datasets (don't decode timedelta to keep as integers)
    print("\nOpening input datasets...")
    ds_intermediate = xr.open_zarr(zarr_intermediate, chunks="auto", decode_timedelta=False)
    ds_endpoints = xr.open_zarr(zarr_endpoints, chunks="auto", decode_timedelta=False)
    
    print(f"Intermediate dataset:\n{ds_intermediate}")
    print(f"\nEndpoints dataset:\n{ds_endpoints}")
    
    # Extract precipitation variable
    precip_var = "total_precipitation_12hr"
    if precip_var not in ds_intermediate:
        raise ValueError(f"Variable {precip_var} not found in intermediate dataset")
    if precip_var not in ds_endpoints:
        raise ValueError(f"Variable {precip_var} not found in endpoints dataset")
    
    precip_intermediate = ds_intermediate[precip_var]
    precip_endpoints = ds_endpoints[precip_var]
    
    print(f"\nIntermediate steps (raw): {precip_intermediate.step.values}")
    print(f"Endpoint steps (raw): {precip_endpoints.step.values}")
    
    # Convert step values from days to hours if needed
    def convert_steps_to_hours(da):
        step_values = da.step.values
        if np.max(step_values) <= 10:  # Likely in days
            step_hours = step_values * 24
            return da.assign_coords(step=step_hours)
        return da
    
    precip_intermediate = convert_steps_to_hours(precip_intermediate)
    precip_endpoints = convert_steps_to_hours(precip_endpoints)
    
    print(f"\nIntermediate steps (hours): {precip_intermediate.step.values}")
    print(f"Endpoint steps (hours): {precip_endpoints.step.values}")
    
    # ==========================================
    # COMPUTE 24HR CUMULATIVE PRECIPITATION
    # ==========================================
    print("\n" + "=" * 80)
    print("Computing 24hr cumulative precipitation...")
    print("=" * 80)
    
    # For 24hr: need increments at hours 0-12 and 12-24
    # Hour 0-12: step=12 from either dataset
    # Hour 12-24: need step=24, which should be in endpoints
    
    if 12 in precip_intermediate.step.values:
        incr_0_12 = precip_intermediate.sel(step=12)
        print("Using step=12 from intermediate dataset (hours 0-12)")
    elif 12 in precip_endpoints.step.values:
        incr_0_12 = precip_endpoints.sel(step=12)
        print("Using step=12 from endpoints dataset (hours 0-12)")
    else:
        raise ValueError("Step 12 not found in either dataset")
    
    if 24 in precip_endpoints.step.values:
        incr_12_24 = precip_endpoints.sel(step=24)
        print("Using step=24 from endpoints dataset (hours 12-24)")
    else:
        raise ValueError("Step 24 not found in endpoints dataset")
    
    # Align the two increments on time dimension
    incr_0_12, incr_12_24 = xr.align(incr_0_12, incr_12_24, join='inner')
    
    # Sum to get 24hr cumulative
    precip_24hr_cumulative = incr_0_12 + incr_12_24
    precip_24hr_cumulative = precip_24hr_cumulative.drop_vars('step', errors='ignore')
    
    print(f"24hr cumulative shape: {precip_24hr_cumulative.shape}")
    print(f"24hr time range: {precip_24hr_cumulative.time.values[0]} to {precip_24hr_cumulative.time.values[-1]}")
    
    # ==========================================
    # COMPUTE 72HR CUMULATIVE PRECIPITATION  
    # ==========================================
    print("\n" + "=" * 80)
    print("Computing 72hr cumulative precipitation...")
    print("=" * 80)
    
    # For 72hr: need to sum increments from hours 0-72
    # We need steps: 12, 24, 36, 48, 60, 72
    required_steps = [12, 24, 36, 48, 60, 72]
    
    # Collect all available increments
    increments_72hr = []
    
    for step in required_steps:
        if step in precip_intermediate.step.values:
            increments_72hr.append(precip_intermediate.sel(step=step))
            print(f"Found step={step}hr in intermediate dataset")
        elif step in precip_endpoints.step.values:
            increments_72hr.append(precip_endpoints.sel(step=step))
            print(f"Found step={step}hr in endpoints dataset")
        else:
            print(f"WARNING: Step {step}hr not found in either dataset - skipping")
    
    if len(increments_72hr) == 0:
        raise ValueError("No increments found for 72hr cumulative computation")
    
    # Align all increments on time dimension (inner join to handle missing forecasts)
    aligned_increments = xr.align(*increments_72hr, join='inner')
    
    # Sum all increments
    precip_72hr_cumulative = sum(aligned_increments)
    precip_72hr_cumulative = precip_72hr_cumulative.drop_vars('step', errors='ignore')
    
    print(f"72hr cumulative shape: {precip_72hr_cumulative.shape}")
    print(f"72hr time range: {precip_72hr_cumulative.time.values[0]} to {precip_72hr_cumulative.time.values[-1]}")
    print(f"Note: 72hr starts {len(increments_72hr)-1}*12 = {(len(increments_72hr)-1)*12}hr after 24hr due to forecast lead time")
    
    # ==========================================
    # CREATE OUTPUT DATASET
    # ==========================================
    print("\n" + "=" * 80)
    print("Creating output dataset...")
    print("=" * 80)
    
    # Since 24hr and 72hr have different time ranges, we'll use lead_time as a coordinate
    # This is cleaner than separate variables with different time lengths
    
    # Align both on their common time range
    precip_24hr_cumulative, precip_72hr_cumulative = xr.align(
        precip_24hr_cumulative, 
        precip_72hr_cumulative, 
        join='outer',  # Keep all times from both
        fill_value=np.nan  # Fill missing with NaN
    )
    
    # Stack into a single variable with lead_time dimension
    ds_cumulative = xr.Dataset({
        'total_precipitation_cumulative': xr.concat(
            [precip_24hr_cumulative, precip_72hr_cumulative],
            dim=xr.DataArray([24, 72], dims='lead_time', name='lead_time',
                           attrs={'units': 'hours', 'long_name': 'Forecast lead time'})
        )
    })
    
    # Update attributes
    ds_cumulative['total_precipitation_cumulative'].attrs.update({
        "long_name": "Cumulative precipitation",
        "description": "Total accumulated precipitation from forecast initialization to lead time",
        "units": precip_intermediate.attrs.get("units", "m"),
        "note": "Computed by summing 12-hourly incremental precipitation"
    })
    
    # Clean attributes
    ds_cumulative = clean_attributes(ds_cumulative)
    
    # Apply chunking
    chunk_dict = CONFIG["chunk_plan"].copy()
    chunk_dict['lead_time'] = -1  # Keep all lead times together
    ds_cumulative = ds_cumulative.chunk(chunk_dict)
    
    # ==========================================
    # WRITE TO ZARR IN BATCHES
    # ==========================================
    print(f"\nWriting cumulative precipitation to: {output_zarr}")
    print(f"Dataset:\n{ds_cumulative}")
    
    batch_size = CONFIG["write_batch_size"]
    n_times = len(ds_cumulative.time)
    n_batches = (n_times + batch_size - 1) // batch_size
    
    print(f"\nWriting in {n_batches} batches of {batch_size} time steps...")
    
    for i in range(0, n_times, batch_size):
        batch_num = i // batch_size + 1
        time_slice = slice(i, min(i + batch_size, n_times))
        subset = ds_cumulative.isel(time=time_slice)
        
        print(f"Batch {batch_num}/{n_batches}: writing times {i} to {min(i + batch_size, n_times)-1}...")
        
        # Compute this batch to avoid lazy loading issues
        subset = subset.compute()
        
        subset.to_zarr(
            output_zarr,
            mode="a" if i > 0 else "w",
            append_dim="time" if i > 0 else None,
            consolidated=False,
            safe_chunks=False,
            zarr_format=2,
        )
        
        print(f"  ✓ Batch {batch_num} written")
    
    # Consolidate metadata at the end
    print("\nConsolidating metadata...")
    zarr.consolidate_metadata(output_zarr)
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    
    # Verification (metadata only)
    print("\nVerifying output...")
    ds_verify = xr.open_zarr(output_zarr, consolidated=True, decode_timedelta=False)
    print(f"\nOutput dataset:\n{ds_verify}")
    print(f"\nDimensions: {dict(ds_verify.sizes)}")
    print(f"Lead times: {ds_verify.lead_time.values}")
    print(f"Time range: {ds_verify.time.values[0]} to {ds_verify.time.values[-1]}")
    print("\nVerification complete (metadata only)")


if __name__ == "__main__":
    main()
