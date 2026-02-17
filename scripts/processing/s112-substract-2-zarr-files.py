import xarray as xr
from pathlib import Path
import shutil

# --- CONFIGURATION ---
# Example: Calculate 6-hour precip (Step 24 - Step 18)
path_a = Path("/scratch2/mg963/data/ecmwf/ensembles/ifs/zarr_ens/tp_f72.zarr") # The larger accumulation
path_b = Path("/scratch2/mg963/data/ecmwf/ensembles/ifs/zarr_ens/tp_f66.zarr") # The smaller accumulation
output_path = Path("/scratch2/mg963/data/ecmwf/ensembles/ifs/zarr_ens/tp_f72_6h.zarr") # Where to save the result

# Variable name inside the Zarr (usually 'tp' or 't2m')
# If inputs have the same name, xarray handles this automatically.
# If you want to rename the RESULT, change this string.
result_var_name = 'tp_6h'

# Chunking strategy (Should match inputs for best performance)
chunks = {
    'time': 1, 
    'number': -1, 
    'latitude': -1, 
    'longitude': -1
}

# --- PROCESSING ---
print(f"➖ Subtracting:\n   A: {path_a}\n   B: {path_b}\n   =  {output_path}")

if not path_a.exists() or not path_b.exists():
    raise FileNotFoundError("One of the input Zarr stores is missing.")

# 1. Open Datasets (Lazy Loading)
# consolidated=True makes opening fast
ds_a = xr.open_zarr(path_a, consolidated=True)
ds_b = xr.open_zarr(path_b, consolidated=True)

# 2. Check Compatibility
# Xarray subtracts based on COORDINATE ALIGNMENT (Time, Lat, Lon, Number).
# If the coordinates don't match exactly, you will get NaNs or an empty result.
try:
    xr.align(ds_a, ds_b, join='exact')
    print("✅ Coordinates align perfectly.")
except ValueError:
    print("⚠️  WARNING: Coordinates do not align exactly (e.g., different dates).")
    print("   The operation will likely intersect the common dates only.")

# 3. Perform Subtraction
# This does not compute yet (it's a dask graph)
ds_diff = ds_a - ds_b

# 4. Cleanup Metadata
# The resulting variable will just be named 'tp' (inherited from ds_a).
# Let's rename it to something meaningful like 'tp_6h'.
existing_var = list(ds_diff.data_vars)[0] # Grab the variable name (e.g., 'tp')
ds_diff = ds_diff.rename({existing_var: result_var_name})

# Update attributes to reflect that this is a processed file
ds_diff[result_var_name].attrs = ds_a[existing_var].attrs
ds_diff[result_var_name].attrs['description'] = "Derived: Step 24h minus Step 18h"

# 5. Save to Zarr
# We re-chunk to ensure the output structure is optimal
ds_diff = ds_diff.chunk(chunks)

if output_path.exists():
    print(f"🗑️  Output path exists. Overwriting...")
    shutil.rmtree(output_path) # to_zarr(mode='w') works, but fresh delete is safer for structure changes

# Trigger the computation
print("💾 Computing and Saving... (This might take a moment)")
ds_diff.to_zarr(output_path, mode='w', consolidated=True)

print("🎉 Calculation Complete.")