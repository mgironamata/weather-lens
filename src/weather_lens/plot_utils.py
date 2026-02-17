import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pdb

def plot_ensemble_maps(filepath, lon_slice=slice(-10, 30), lat_slice=slice(70, 35), time='2025-08-01', 
                       figsize=(12, 10), binary=False, era5_filepath=None, nrows=3, ncols=3):
    """
    Plot subplots showing ensemble members for a given variable.
    If ERA5 data is provided, plot ERA5 in first position with ensemble members filling remaining positions.
    Otherwise, plot ensemble members starting from first position.
    
    Parameters:
    filepath (str): Path to the ensemble netCDF file
    variable (str): Variable name to plot (default: 'tp')
    lon_slice (slice): Longitude slice for Europe (default: slice(-10, 30))
    lat_slice (slice): Latitude slice for Europe (default: slice(70, 35))
    time (str): Time for selection (default: '2025-08-01')
    figsize (tuple): Figure size for the plot
    binary (bool): If True, plot binary maps (zero vs non-zero) (default: False)
    era5_filepath (str): Path to ERA5 netCDF file (optional)
    nrows (int): Number of rows in subplot grid (default: 3)
    ncols (int): Number of columns in subplot grid (default: 3)
    """
    total_plots = nrows * ncols
    
    # Load ensemble data as DataArray
    ensemble_data = xr.open_dataarray(filepath)
    
    # Select and slice ensemble data
    if 'number' in ensemble_data.dims:
        num_ensemble_members = (total_plots - 1) if era5_filepath else total_plots
        ensemble_data = ensemble_data.isel(number=slice(0, num_ensemble_members))
    
    ensemble_data = ensemble_data.sel(
        longitude=lon_slice, 
        latitude=lat_slice).sel(time=time, method='nearest')
    
    # Load ERA5 data if provided
    era5_data = None
    if era5_filepath:
        era5_data = xr.open_dataarray(era5_filepath)
        era5_data = era5_data.sel(
            longitude=lon_slice,
            latitude=lat_slice).sel(time=time, method='nearest')
            
        if binary:
            era5_data = (era5_data > 0).astype(int)
    
    # Convert ensemble to binary if requested
    if binary:
        ensemble_data = (ensemble_data > 0).astype(int)
    
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if total_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot data
    ensemble_idx = 0
    for i in range(total_plots):        
        if era5_filepath and i == 0:  # First position for ERA5
            if binary:
                im = axes[i].contourf(era5_data.longitude, era5_data.latitude, era5_data, 
                                    levels=[0, 0.5, 1], colors=['white', 'blue'], extend='neither')
            else:
                im = axes[i].contourf(era5_data.longitude, era5_data.latitude, era5_data)
            axes[i].set_title('ERA5')
        else:  # Ensemble members
            if binary:
                im = axes[i].contourf(ensemble_data.longitude, ensemble_data.latitude, 
                                    ensemble_data.isel(number=ensemble_idx),
                                    levels=[0, 0.5, 1], colors=['white', 'blue'], extend='neither')
            else:
                im = axes[i].contourf(ensemble_data.longitude, ensemble_data.latitude, 
                                    ensemble_data.isel(number=ensemble_idx))
            axes[i].set_title(f'Ensemble {ensemble_idx+1}')
            ensemble_idx += 1
        
        # Remove tick labels and ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.show()

def plot_mixed_ensemble_maps(aifs_filepath, ifs_filepath, lon_slice=slice(-10, 30), lat_slice=slice(70, 35), 
                            time='2025-08-01', figsize=(12, 10), binary=False, era5_filepath=None, 
                            nrows=3, ncols=3):
    """
    Plot subplots showing both AIFS and IFS ensemble members for a given variable.
    If ERA5 data is provided, plot ERA5 at the beginning of AIFS rows and IFS rows.
    
    Parameters:
    aifs_filepath (str): Path to the AIFS ensemble netCDF file
    ifs_filepath (str): Path to the IFS ensemble netCDF file
    lon_slice (slice): Longitude slice for Europe (default: slice(-10, 30))
    lat_slice (slice): Latitude slice for Europe (default: slice(70, 35))
    time (str): Time for selection (default: '2025-08-01')
    figsize (tuple): Figure size for the plot
    binary (bool): If True, plot binary maps (zero vs non-zero) (default: False)
    era5_filepath (str): Path to ERA5 netCDF file (optional)
    nrows (int): Number of rows in subplot grid (default: 3)
    ncols (int): Number of columns in subplot grid (default: 3)
    """
    total_plots = nrows * ncols
    
    # Calculate rows for AIFS and IFS
    aifs_rows = nrows // 2
    ifs_rows = nrows - aifs_rows
    
    # Calculate members needed (subtract 2 for ERA5 if provided)
    era5_plots = 2 if era5_filepath else 0
    ensemble_plots = total_plots - era5_plots
    aifs_members = aifs_rows * ncols - (1 if era5_filepath else 0)
    ifs_members = ifs_rows * ncols - (1 if era5_filepath else 0)
    
    # Load AIFS ensemble data
    aifs_data = xr.open_dataarray(aifs_filepath)
    if 'number' in aifs_data.dims:
        aifs_data = aifs_data.isel(number=slice(0, aifs_members))
    aifs_data = aifs_data.sel(
        longitude=lon_slice, 
        latitude=lat_slice).sel(time=time, method='nearest')
    
    # Load IFS ensemble data
    ifs_data = xr.open_dataarray(ifs_filepath)
    if 'number' in ifs_data.dims:
        ifs_data = ifs_data.isel(number=slice(0, ifs_members))
    ifs_data = ifs_data.sel(
        longitude=lon_slice, 
        latitude=lat_slice).sel(time=time, method='nearest')
    
    # Load ERA5 data if provided
    era5_data = None
    if era5_filepath:
        era5_data = xr.open_dataarray(era5_filepath)
        era5_data = era5_data.sel(
            longitude=lon_slice,
            latitude=lat_slice).sel(time=time, method='nearest')
        if binary:
            era5_data = (era5_data > 0).astype(int)
    
    # Convert to binary if requested
    if binary:
        aifs_data = (aifs_data > 0).astype(int)
        ifs_data = (ifs_data > 0).astype(int)
    
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if total_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot data
    aifs_idx = 0
    ifs_idx = 0
    
    for i in range(total_plots):
        row = i // ncols
        col = i % ncols
        
        # First row (AIFS section)
        if row < aifs_rows:
            if era5_filepath and col == 0:  # ERA5 at beginning of AIFS section
                if binary:
                    im = axes[i].contourf(era5_data.longitude, era5_data.latitude, era5_data, 
                                        levels=[0, 0.5, 1], colors=['white', 'green'], extend='neither')
                else:
                    im = axes[i].contourf(era5_data.longitude, era5_data.latitude, era5_data)
                axes[i].set_title('ERA5')
            else:  # AIFS members
                if binary:
                    im = axes[i].contourf(aifs_data.longitude, aifs_data.latitude, 
                                        aifs_data.isel(number=aifs_idx),
                                        levels=[0, 0.5, 1], colors=['white', 'blue'], extend='neither')
                else:
                    im = axes[i].contourf(aifs_data.longitude, aifs_data.latitude, 
                                        aifs_data.isel(number=aifs_idx))
                axes[i].set_title(f'AIFS {aifs_idx+1}')
                aifs_idx += 1
        # Later rows (IFS section)
        else:
            if era5_filepath and col == 0:  # ERA5 at beginning of IFS section
                if binary:
                    im = axes[i].contourf(era5_data.longitude, era5_data.latitude, era5_data, 
                                        levels=[0, 0.5, 1], colors=['white', 'green'], extend='neither')
                else:
                    im = axes[i].contourf(era5_data.longitude, era5_data.latitude, era5_data)
                axes[i].set_title('ERA5')
            else:  # IFS members
                if binary:
                    im = axes[i].contourf(ifs_data.longitude, ifs_data.latitude, 
                                        ifs_data.isel(number=ifs_idx),
                                        levels=[0, 0.5, 1], colors=['white', 'blue'], extend='neither')
                else:
                    im = axes[i].contourf(ifs_data.longitude, ifs_data.latitude, 
                                        ifs_data.isel(number=ifs_idx))
                axes[i].set_title(f'IFS {ifs_idx+1}')
                ifs_idx += 1
        
        # Remove tick labels and ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.show()

def plot_ensemble_differences(aifs_filepath, ifs_filepath, era5_filepath, lon_slice=slice(-10, 30), 
                                lat_slice=slice(70, 35), time='2025-08-01', figsize=(12, 10), 
                                nrows=3, ncols=3):
    """
    Plot subplots showing differences between ensemble members and ERA5 (ensemble - ERA5).
    
    Parameters:
    aifs_filepath (str): Path to the AIFS ensemble netCDF file
    ifs_filepath (str): Path to the IFS ensemble netCDF file
    era5_filepath (str): Path to ERA5 netCDF file
    lon_slice (slice): Longitude slice for Europe (default: slice(-10, 30))
    lat_slice (slice): Latitude slice for Europe (default: slice(70, 35))
    time (str): Time for selection (default: '2025-08-01')
    figsize (tuple): Figure size for the plot
    nrows (int): Number of rows in subplot grid (default: 3)
    ncols (int): Number of columns in subplot grid (default: 3)
    """
    total_plots = nrows * ncols
    
    # Calculate rows for AIFS and IFS
    aifs_rows = nrows // 2
    ifs_rows = nrows - aifs_rows
    
    # Calculate members needed
    aifs_members = aifs_rows * ncols
    ifs_members = ifs_rows * ncols
    
    # Load ERA5 data
    era5_data = xr.open_dataarray(era5_filepath)
    era5_data = era5_data.sel(
        longitude=lon_slice,
        latitude=lat_slice).sel(time=time, method='nearest')
    
    # Load AIFS ensemble data
    aifs_data = xr.open_dataarray(aifs_filepath)
    if 'number' in aifs_data.dims:
        aifs_data = aifs_data.isel(number=slice(0, aifs_members))
    aifs_data = aifs_data.sel(
        longitude=lon_slice, 
        latitude=lat_slice).sel(time=time, method='nearest')
    
    # Load IFS ensemble data
    ifs_data = xr.open_dataarray(ifs_filepath)
    if 'number' in ifs_data.dims:
        ifs_data = ifs_data.isel(number=slice(0, ifs_members))
    ifs_data = ifs_data.sel(
        longitude=lon_slice, 
        latitude=lat_slice).sel(time=time, method='nearest')
    
    # Calculate differences
    aifs_diff = aifs_data - era5_data
    ifs_diff = ifs_data - era5_data
    
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if total_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot differences
    aifs_idx = 0
    ifs_idx = 0
    
    for i in range(total_plots):
        row = i // ncols
        
        # First rows (AIFS section)
        if row < aifs_rows:
            diff_data = aifs_diff.isel(number=aifs_idx)
            im = axes[i].contourf(diff_data.longitude, diff_data.latitude, diff_data, 
                                cmap='RdBu_r', levels=20)
            axes[i].set_title(f'AIFS {aifs_idx+1} - ERA5')
            aifs_idx += 1
        # Later rows (IFS section)
        else:
            diff_data = ifs_diff.isel(number=ifs_idx)
            im = axes[i].contourf(diff_data.longitude, diff_data.latitude, diff_data, 
                                cmap='RdBu_r', levels=20)
            axes[i].set_title(f'IFS {ifs_idx+1} - ERA5')
            ifs_idx += 1
        
        # Remove tick labels and ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    # Add colorbar
    plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
    plt.tight_layout()
    plt.show()

def plot_ensemble_timeseries(aifs_filepath, ifs_filepath, era5_filepath, lon=10.0, lat=50.0, 
                            start_time='2025-08-01', end_time='2025-08-10', n_members=5, 
                            figsize=(12, 8)):
    """
    Plot timeseries of N ensemble members from AIFS and IFS alongside ERA5 for a specific location.
    
    Parameters:
    aifs_filepath (str): Path to the AIFS ensemble netCDF file
    ifs_filepath (str): Path to the IFS ensemble netCDF file
    era5_filepath (str): Path to ERA5 netCDF file
    lon (float): Longitude for spatial location (default: 10.0)
    lat (float): Latitude for spatial location (default: 50.0)
    start_time (str): Start time for timeseries (default: '2025-08-01')
    end_time (str): End time for timeseries (default: '2025-08-10')
    n_members (int): Number of ensemble members to plot from each model (default: 5)
    figsize (tuple): Figure size for the plot (default: (12, 8))
    """
    # Load ERA5 data
    era5_data = xr.open_dataarray(era5_filepath)
    era5_ts = era5_data.sel(longitude=lon, latitude=lat, method='nearest').sel(
        time=slice(start_time, end_time))
    
    # Load AIFS ensemble data
    aifs_data = xr.open_dataarray(aifs_filepath)
    if 'number' in aifs_data.dims:
        aifs_data = aifs_data.isel(number=slice(0, n_members))
    aifs_ts = aifs_data.sel(longitude=lon, latitude=lat, method='nearest').sel(
        time=slice(start_time, end_time))
    
    # Load IFS ensemble data
    ifs_data = xr.open_dataarray(ifs_filepath)
    if 'number' in ifs_data.dims:
        ifs_data = ifs_data.isel(number=slice(0, n_members))
    ifs_ts = ifs_data.sel(longitude=lon, latitude=lat, method='nearest').sel(
        time=slice(start_time, end_time))
    
    ifs_ts = ifs_ts * 1000  # Convert from meters to mm if needed

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ERA5
    ax.plot(era5_ts.time, era5_ts, 'k-', linewidth=3, label='ERA5', zorder=10)
    
    # Plot AIFS ensemble members
    for i in range(n_members):
        if i < len(aifs_ts.number):
            ax.plot(aifs_ts.time, aifs_ts.isel(number=i), 'b-', alpha=0.6, 
                    label='AIFS' if i == 0 else '', linewidth=1.5)
    
    # Plot IFS ensemble members
    for i in range(n_members):
        if i < len(ifs_ts.number):
            ax.plot(ifs_ts.time, ifs_ts.isel(number=i), 'r-', alpha=0.6, 
                    label='IFS' if i == 0 else '', linewidth=1.5)
    
    # Formatting
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'Ensemble Timeseries at Lon={lon:.1f}, Lat={lat:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_ensemble_precipitation_boxplot(aifs_filepath, ifs_filepath, era5_filepath, lon=10.0, lat=50.0, 
                                        start_time='2025-08-01', end_time='2025-08-10', 
                                        figsize=(10, 6)):
    """
    Plot boxplot of total precipitation over a time period for AIFS and IFS ensemble members 
    compared to ERA5 at a specific location.
    
    Parameters:
    aifs_filepath (str): Path to the AIFS ensemble netCDF file
    ifs_filepath (str): Path to the IFS ensemble netCDF file
    era5_filepath (str): Path to ERA5 netCDF file
    lon (float): Longitude for spatial location (default: 10.0)
    lat (float): Latitude for spatial location (default: 50.0)
    start_time (str): Start time for period (default: '2025-08-01')
    end_time (str): End time for period (default: '2025-08-10')
    figsize (tuple): Figure size for the plot (default: (10, 6))
    """
    # Load ERA5 data and calculate total precipitation
    era5_data = xr.open_dataarray(era5_filepath)
    era5_ts = era5_data.sel(longitude=lon, latitude=lat, method='nearest').sel(
        time=slice(start_time, end_time))
    era5_total = float(era5_ts.sum())
    
    # Load AIFS ensemble data and calculate total precipitation for each member
    aifs_data = xr.open_dataarray(aifs_filepath)
    aifs_ts = aifs_data.sel(longitude=lon, latitude=lat, method='nearest').sel(
        time=slice(start_time, end_time))
    aifs_totals = aifs_ts.sum(dim='time').values
    
    # Load IFS ensemble data and calculate total precipitation for each member
    ifs_data = xr.open_dataarray(ifs_filepath)
    ifs_ts = ifs_data.sel(longitude=lon, latitude=lat, method='nearest').sel(
        time=slice(start_time, end_time))
    ifs_totals = ifs_ts.sum(dim='time').values * 1000  # Convert from meters to mm if needed
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create boxplot data
    box_data = [aifs_totals, ifs_totals]
    box_labels = ['AIFS', 'IFS']
    
    # Create boxplot
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    # Color the boxes
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    # Add ERA5 reference line
    ax.axhline(y=era5_total, color='black', linestyle='--', linewidth=2, label='ERA5')
    
    # Add individual points
    for i, totals in enumerate(box_data):
        x = np.random.normal(i+1, 0.04, size=len(totals))
        ax.scatter(x, totals, alpha=0.6, s=20)
    
    # Formatting
    ax.set_ylabel('Total Precipitation (mm)')
    ax.set_title(f'Total Precipitation ({start_time} to {end_time})\nLon={lon:.1f}, Lat={lat:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    aifs_mean = np.mean(aifs_totals)
    ifs_mean = np.mean(ifs_totals)
    ax.text(0.02, 0.98, f'AIFS mean: {aifs_mean:.1f} mm\nIFS mean: {ifs_mean:.1f} mm\nERA5: {era5_total:.1f} mm', 
            transform=ax.transAxes, verticalalignment='top', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

