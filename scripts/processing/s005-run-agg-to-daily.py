import xarray as xr
import numpy as np

def aggregate_to_daily_custom_start(ds, var="tp", start_hour=6, units_out="mm/day"):
    """
    Aggregate hourly ERA5 variable to daily totals based on a custom 24-h window.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Input dataset containing an hourly variable (e.g. ERA5 'tp' in m/hour).
    var : str
        Name of variable in ds (ignored if ds is a DataArray).
    start_hour : int
        Start hour (UTC) defining the 24-h accumulation period.
        e.g. start_hour=6 -> 06:00→06:00 daily totals.
    units_out : str
        Desired output units (default 'mm/day').

    Returns
    -------
    xr.DataArray
        Daily totals with time coordinate marking the **end** of each 24-h period.
    """

    # Extract DataArray
    da = ds[var] if isinstance(ds, xr.Dataset) else ds

    # Ensure hourly regular spacing
    if not np.all(np.diff(da.valid_time.values).astype('timedelta64[h]') == np.timedelta64(1, 'h')):
        raise ValueError("Input time steps must be hourly and regular")

    # Shift time by -start_hour so resampling bins align with desired start time
    da_shifted = da.assign_coords(time=da.valid_time - np.timedelta64(start_hour, "h"))

    # Aggregate over 24 h
    da_daily_shifted = da_shifted.resample(time="1D").sum(keep_attrs=True)

    # Shift labels back +start_hour so each timestamp marks the *end* of the period
    da_daily = da_daily_shifted.assign_coords(
        time=da_daily_shifted.time + np.timedelta64(start_hour, "h")
    )

    # Convert from meters to mm
    if "mm" in units_out:
        da_daily = da_daily * 1000.0

    # Metadata
    da_daily.name = f"{da.name}_daily_{start_hour:02d}UTC"
    da_daily.attrs.update({
        "long_name": f"Daily total ({start_hour:02d}–{start_hour:02d} UTC)",
        "aggregation": f"sum of hourly {da.name} shifted by {start_hour}h",
        "units": units_out
    })

    return da_daily

def select_daily_timestamp(ds, var="tp", target_hour=6):
    """
    Select one timestamp per day from hourly data.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Input dataset containing hourly data.
    var : str
        Name of variable in ds (ignored if ds is a DataArray).
    target_hour : int
        Hour (UTC) to select for each day (0-23).

    Returns
    -------
    xr.DataArray
        Daily data with one value per day at the specified hour.
    """

    # Extract DataArray
    da = ds[var] if isinstance(ds, xr.Dataset) else ds

    # Ensure hourly regular spacing
    if not np.all(np.diff(da.valid_time.values).astype('timedelta64[h]') == np.timedelta64(1, 'h')):
        raise ValueError("Input time steps must be hourly and regular")

    # Select data at the target hour for each day
    da_daily = da.where(da.valid_time.dt.hour == target_hour, drop=True)

    # Metadata
    da_daily.name = f"{da.name}_daily_{target_hour:02d}UTC_selected"
    da_daily.attrs.update({
        "long_name": f"Daily values at {target_hour:02d} UTC",
        "selection": f"hourly {da.name} at {target_hour:02d} UTC each day"
    })

    return da_daily

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Aggregate hourly ERA5 variable to daily totals with custom start hour.")
    ap.add_argument("input_file", help="Input NetCDF file with hourly data")
    ap.add_argument("output_file", help="Output NetCDF file for daily aggregated data")
    ap.add_argument("--variable", "-v", default="tp", help="Variable name in the dataset (default: tp)")
    ap.add_argument("--start-hour", "-s", type=int, default=6, help="Start hour (UTC) for daily aggregation (default: 6)")
    ap.add_argument("--units-out", "-u", default="mm/day", help="Output units (default: mm/day)")
    args = ap.parse_args()
    # Example usage: python s005-run-agg-to-daily.py input.nc output.nc --variable tp --start-hour 6 --units-out mm/day

    # Load dataset
    ds_in = xr.open_dataset(args.input_file)
    print(f"Loaded input data with shape: {ds_in[args.variable].shape}")

    # Aggregate to daily
    if args.variable == "tp":
        da_daily = aggregate_to_daily_custom_start(
            ds_in,
            var=args.variable,
            start_hour=args.start_hour,
            units_out=args.units_out
        )
        print(f"Aggregated to daily data with shape: {da_daily.shape}")
    else:
        da_daily = select_daily_timestamp(
            ds_in,
            var=args.variable,
            target_hour=args.start_hour
        )
        print(f"Selected daily at {args.start_hour:02d} UTC - data with shape: {da_daily.shape}")
    

    # Northern Hemisphere, Tropics and Southern Hemisphere masks
    lat = da_daily['latitude']
    tropics_lat = 23.43688  # Approx. Tropic of Cancer/Capricorn
    nh_mask = lat > tropics_lat
    tropics_mask = (lat >= -tropics_lat) & (lat <= tropics_lat)
    sh_mask = lat < -tropics_lat

    da_daily_nh = da_daily.where(nh_mask, drop=True)
    da_daily_tropics = da_daily.where(tropics_mask, drop=True)
    da_daily_sh = da_daily.where(sh_mask, drop=True)

    # Save files to NetCDF
    da_daily_nh.to_netcdf(args.output_file.replace(".nc", "_NH.nc"))
    print(f"Saved daily aggregated data to {args.output_file.replace('.nc', '_NH.nc')}")
    da_daily_tropics.to_netcdf(args.output_file.replace(".nc", "_Tropics.nc"))
    print(f"Saved daily aggregated data to {args.output_file.replace('.nc', '_Tropics.nc')}")
    da_daily_sh.to_netcdf(args.output_file.replace(".nc", "_SH.nc"))
    print(f"Saved daily aggregated data to {args.output_file.replace('.nc', '_SH.nc')}")