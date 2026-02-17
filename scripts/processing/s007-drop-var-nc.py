import xarray as xr

def process_xr_datasets(paths, var_to_drop, output_paths):
    """
    Process multiple xarray datasets: open, drop variable, and save as netCDF
    
    Args:
        paths: list of input file paths
        var_to_drop: variable name to drop
        output_paths: list of output file paths
    """
    for input_path, output_path in zip(paths, output_paths):
        # Open dataset
        ds = xr.open_dataset(input_path)
        
        # Drop variable if it exists
        if var_to_drop in ds.variables:
            ds_drop = ds.drop_vars(var_to_drop)
        
        # Save as netCDF
        ds_drop.to_netcdf(output_path)

        # Close dataset to free memory
        ds.close()
        ds_drop.close()
        
        print(f"Processed: {input_path} -> {output_path}")

uber_var = "10uv"
var = "u10"  # 10-meter wind speed
var_to_drop = 'v10'
fxx = 72  # lead time
model = "ifs"

# Example usage:
paths = [f"/scratch2/mg963/data/ecmwf/ensembles/{uber_var}/{fxx}/{model}/{model.upper()}_{uber_var}_FXX{fxx}_NH.nc",
            f"/scratch2/mg963/data/ecmwf/ensembles/{uber_var}/{fxx}/{model}/{model.upper()}_{uber_var}_FXX{fxx}_Tropics.nc",
            f"/scratch2/mg963/data/ecmwf/ensembles/{uber_var}/{fxx}/{model}/{model.upper()}_{uber_var}_FXX{fxx}_SH.nc"]

output_paths = [f"/scratch2/mg963/data/ecmwf/ensembles/{uber_var}/{fxx}/{model}/{model.upper()}_{var}_FXX{fxx}_NH.nc",
                f"/scratch2/mg963/data/ecmwf/ensembles/{uber_var}/{fxx}/{model}/{model.upper()}_{var}_FXX{fxx}_Tropics.nc",
                f"/scratch2/mg963/data/ecmwf/ensembles/{uber_var}/{fxx}/{model}/{model.upper()}_{var}_FXX{fxx}_SH.nc"]

process_xr_datasets(paths, var_to_drop, output_paths)