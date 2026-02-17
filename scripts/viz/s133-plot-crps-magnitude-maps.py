import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path

def open_crps(path):
    """Opens CRPS Zarr and extracts the main variable."""
    try:
        ds = xr.open_zarr(path, consolidated=True)
    except Exception:
        ds = xr.open_zarr(path, consolidated=False)
    
    # Identify variable
    var_names = list(ds.data_vars)
    if "tp_crps" in var_names:
        da = ds["tp_crps"]
    elif "total_precipitation_cumulative_crps" in var_names:
        da = ds["total_precipitation_cumulative_crps"]
    elif "t2m_crps" in var_names:
        da = ds["t2m_crps"]
    elif "2m_temperature_crps" in var_names:
        da = ds["2m_temperature_crps"]
    elif "10m_wind_speed_crps" in var_names:
        da = ds["10m_wind_speed_crps"]
    elif "wspd10_crps" in var_names:
        da = ds["wspd10_crps"]
    elif len(var_names) == 1:
        da = ds[var_names[0]]
    else:
        raise ValueError(f"Cannot identify CRPS variable in {path}. Choices: {var_names}")
    
    if "time" in da.dims:
        da = da.sortby("time")
    
    return da

def main():
    # CONFIG
    VARIABLE = "t2m"  # "t2m", "winds", or "tp"
    LEAD = 72  
    
    if LEAD not in [24, 72]:  # 24 or 72
        raise ValueError("LEAD must be either 24 or 72")
    elif LEAD == 24:
        START_DATE = "2024-02-04"
    elif LEAD == 72:
        START_DATE = "2024-02-06"

    END_DATE = "2024-12-31"
    
    # Paths logic (matching s132-plot-crps-map-diff.py)
    if VARIABLE == "tp":
        paths = {
            "IFS": f"/scratch2/mg963/results/ecmwf/ifs_vs_ERA5/crps_{VARIABLE}_{LEAD}h.zarr",
            "GenCast": f"/scratch2/mg963/results/weathernext/gencast_vs_ERA5/crps_{VARIABLE}_{LEAD}h.zarr",
            "WN2": f"/scratch2/mg963/results/weathernext/wn2_vs_ERA5/crps_{VARIABLE}_{LEAD}h.zarr"
        }
    else:
        paths = {
            "IFS": f"/scratch2/mg963/results/ecmwf/ifs-ens/crps_{VARIABLE}_{LEAD}h.zarr",
            "GenCast": f"/scratch2/mg963/results/weathernext/gencast/crps_{VARIABLE}_{LEAD}h.zarr",
            "WN2": f"/scratch2/mg963/results/weathernext/WN2/crps_{VARIABLE}_{LEAD}h.zarr"
        }

    print(f"🔹 Loading {VARIABLE} {LEAD}h datasets...")
    das = {name: open_crps(p) for name, p in paths.items()}

    # Align and Slice
    print(f"🔹 Aligning and slicing from {START_DATE} to {END_DATE}...")
    da_ifs, da_gencast, da_wn2 = xr.align(das["IFS"], das["GenCast"], das["WN2"], join="inner")
    
    da_ifs = da_ifs.sel(time=slice(START_DATE, END_DATE))
    da_gencast = da_gencast.sel(time=slice(START_DATE, END_DATE))
    da_wn2 = da_wn2.sel(time=slice(START_DATE, END_DATE))

    # Compute time mean
    print("🔹 Computing time means...")
    means = {
        "IFS-ENS": da_ifs.mean(dim="time").compute(),
        "GenCast": da_gencast.mean(dim="time").compute(),
        "WeatherNext 2": da_wn2.mean(dim="time").compute()
    }

    # Plotting
    print("🔹 Plotting...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), subplot_kw={'projection': ccrs.PlateCarree()})

    cmap = "viridis" 
    unit = "K" if VARIABLE == "t2m" else ("m/s" if VARIABLE == "winds" else "m")
    
    # Determine robust vmax across all models for consistent comparison
    all_vals = np.concatenate([m.values.flatten() for m in means.values()])
    vmax = np.nanpercentile(all_vals, 98)
    vmin = 0
    
    for ax, (name, mean_da) in zip(axes, means.items()):
        # Add features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.1)
        
        im = mean_da.plot(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=vmin, vmax=vmax,
            add_colorbar=False
        )
        
        cb = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, extend='max', shrink=0.7)
        cb.set_label(f"CRPS ({unit})")
        
        ax.set_title(f"{name} ({LEAD}h {VARIABLE} Mean CRPS)\n{START_DATE} to {END_DATE}", fontsize=14)

    plt.tight_layout()
    out_path = f"/scratch2/mg963/results/diagnostics/maps/crps_{VARIABLE}_{LEAD}h_magnitude_maps.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"🎉 Magnitude maps saved to {out_path}")

if __name__ == "__main__":
    main()
