import xarray as xr
from pathlib import Path


def compute_time_totals(
    input_path: str,
    output_path: str,
    n_days: int,
    time_dim: str = "time"
) -> None:
    """
    Load a DataArray from a file, compute totals across N-day periods, and save as netCDF.
    
    This function is designed for daily precipitation data and loads the DataArray directly
    without needing to specify the variable name.
    
    Parameters
    ----------
    input_path : str
        Path to the input netCDF/zarr file containing the DataArray
    output_path : str
        Path where the output netCDF file will be saved
    n_days : int
        Number of days to aggregate over (e.g., 7 for weekly totals)
    time_dim : str, optional
        Name of the time dimension, by default "time"
    
    Examples
    --------
    >>> # Compute 7-day precipitation totals
    >>> compute_time_totals("daily_precip.nc", "weekly_precip.nc", n_days=7)
    
    >>> # Compute monthly totals (30 days)
    >>> compute_time_totals("daily_precip.zarr", "monthly_precip.nc", n_days=30)
    """
    # Load the DataArray
    print(f"Loading data from {input_path}...")
    da = xr.open_dataarray(input_path, chunks="auto")
    
    # Verify time dimension exists
    if time_dim not in da.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in DataArray. "
                        f"Available dimensions: {list(da.dims)}")
    
    # Get the number of time steps
    n_times = len(da[time_dim])
    print(f"Original time steps: {n_times}")
    
    # Compute the number of complete periods
    n_periods = n_times // n_days
    n_complete = n_periods * n_days
    
    if n_complete < n_times:
        print(f"Warning: {n_times - n_complete} time steps at the end will be dropped "
              f"to form complete {n_days}-day periods")
    
    # Trim to complete periods
    da_trimmed = da.isel({time_dim: slice(0, n_complete)})
    
    # Reshape and sum across the period dimension
    # This groups consecutive n_days into periods and sums them
    print(f"Computing {n_days}-day totals...")
    
    # Stack the time dimension into (periods, days_in_period)
    new_shape = (n_periods, n_days) + da_trimmed.shape[1:]
    
    # Reshape the data
    reshaped = da_trimmed.values.reshape(new_shape)
    
    # Sum across the days_in_period axis (axis=1)
    totals = reshaped.sum(axis=1)
    
    # Create new coordinates for the aggregated time
    # Use the first time of each period as the coordinate
    time_coords = da_trimmed[time_dim].values[::n_days]
    
    # Create the output DataArray
    coords = {time_dim: time_coords}
    for dim in da.dims:
        if dim != time_dim:
            coords[dim] = da[dim].values
    
    da_totals = xr.DataArray(
        totals,
        dims=da.dims,
        coords=coords,
        attrs=da.attrs.copy()
    )
    
    # Update attributes to reflect the aggregation
    da_totals.attrs["aggregation"] = f"{n_days}-day totals"
    da_totals.attrs["original_time_steps"] = n_times
    da_totals.attrs["aggregated_time_steps"] = n_periods
    
    # Save to netCDF
    print(f"Saving to {output_path}...")
    da_totals.to_netcdf(output_path)
    print(f"Successfully saved {n_periods} {n_days}-day totals to {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python compute_precip_totals.py <input_path> <output_path> <n_days>")
        print("Example: python compute_precip_totals.py daily_precip.nc weekly_precip.nc 7")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    n_days = int(sys.argv[3])
    
    compute_time_totals(input_path, output_path, n_days)
