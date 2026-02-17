
import time
import xarray as xr

def align_era_to_target_grid(target_dataset, era):
    # target grid (AIFS/IFS without the member dim)
    target = target_dataset.isel(number=0).drop_vars([v for v in target_dataset.coords if v not in ("time","latitude","longitude")], errors="ignore")

    # 0) Rename time coordinate if needed
    if 'valid_time' in era.coords:
        era = era.rename({'valid_time': 'time'})

    # 1) Convert ERA lon labels to [-180, 180) AND reorder data accordingly
    era = era.assign_coords(longitude=((era.longitude + 180) % 360) - 180).sortby("longitude")

    # 2) Make latitude orientation match the target
    era = era.sortby("latitude", ascending=bool(target.latitude[0] < target.latitude[-1]))

    # 3) Slice ERA to match target time range
    era = era.sel(time=slice(target.time.min(), target.time.max()))

    # 4) Put ERA *values* onto the exact target grid
    #    - If grids are identical: reindex (no interpolation)
    era_on_target = era.reindex_like(target, method=None)

    #   - If there are tiny differences in points/resolution: interpolate instead
    # era_on_target = era.interp_like(target)

    # Sanity checks
    # for d in ("time", "latitude", "longitude"):
    #     assert era_on_target[d].equals(target[d]), f"{d} still differs"
    
    return era_on_target    

if __name__ == "__main__":
    import xarray as xr

    uber_var = "10uv"
    var = "u10"  # 10-meter wind speed
    fxx = 72  # lead time
    model = "aifs"

    base_folder = "/scratch2/mg963/data/"
    era_files = [
        f"ecmwf/era5/{uber_var}/era5_daily_{var}_july2sept2025_NH.nc",
        f"ecmwf/era5/{uber_var}/era5_daily_{var}_july2sept2025_Tropics.nc",
        f"ecmwf/era5/{uber_var}/era5_daily_{var}_july2sept2025_SH.nc",
        # Add more ERA files here
    ]

    target_files = [
        f"ecmwf/ensembles/{uber_var}/{fxx}/{model}/{model.upper()}_{var}_FXX{fxx}_NH.nc",
        f"ecmwf/ensembles/{uber_var}/{fxx}/{model}/{model.upper()}_{var}_FXX{fxx}_Tropics.nc",
        f"ecmwf/ensembles/{uber_var}/{fxx}/{model}/{model.upper()}_{var}_FXX{fxx}_SH.nc",
        # Add corresponding target files here
    ]

    output_files = [
        f"ecmwf/era5/{uber_var}/era5_daily_{var}_july2sept2025_NH_aligned.nc",
        f"ecmwf/era5/{uber_var}/era5_daily_{var}_july2sept2025_Tropics_aligned.nc",
        f"ecmwf/era5/{uber_var}/era5_daily_{var}_july2sept2025_SH_aligned.nc",
        # Add corresponding output files here
    ]

    era_files = [base_folder + f for f in era_files]
    target_files = [base_folder + f for f in target_files]
    output_files = [base_folder + f for f in output_files]

    for era_file, target_file, output_file in zip(era_files, target_files, output_files):
        start = time.time()
        era = xr.open_dataarray(era_file)
        target = xr.open_dataset(target_file)

        # crop target to tropics if needed
        if 'Tropics' in target_file:
            target = target.sel(latitude=slice(23.4, -23.4))
        
        era_on_target = align_era_to_target_grid(target, era)
        
        era_on_target.to_netcdf(output_file)
        end = time.time()
        print(f"Time taken: {end - start:.2f} seconds")
        print(f"Processed: {era_file} -> {output_file}")