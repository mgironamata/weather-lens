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
    elif len(var_names) == 1:
        da = ds[var_names[0]]
    else:
        raise ValueError(f"Cannot identify CRPS variable in {path}. Choices: {var_names}")
    
    if "time" in da.dims:
        da = da.sortby("time")
    
    return da

def main():
    # CONFIG
    VARIABLE = "tp" # "t2m" or "tp" or "winds"
    LEAD = 72  # 24 or 72

    if LEAD not in [24, 72]:
        raise ValueError("LEAD must be either 24 or 72 hours.")
    elif LEAD == 24:
        START_DATE = "2024-02-04"
    elif LEAD == 72:
        START_DATE = "2024-02-06"
    
    END_DATE = "2024-12-31"
    
    # Paths
    if VARIABLE == "tp":
        # Precipitation CRPS files are in different folders (vs ERA5)
        paths = {
            "IFS": f"/scratch2/mg963/results/ecmwf/ifs_vs_ERA5/crps_{VARIABLE}_{LEAD}h.zarr",
            "GenCast": f"/scratch2/mg963/results/weathernext/gencast_vs_ERA5/crps_{VARIABLE}_{LEAD}h.zarr",
            "WN2": f"/scratch2/mg963/results/weathernext/wn2_vs_ERA5/crps_{VARIABLE}_{LEAD}h.zarr"
        }
    else:
        # Legacy paths for t2m and winds
        paths = {
            "IFS": f"/scratch2/mg963/results/ecmwf/ifs-ens/crps_{VARIABLE}_{LEAD}h.zarr",
            "GenCast": f"/scratch2/mg963/results/weathernext/gencast/crps_{VARIABLE}_{LEAD}h.zarr",
            "WN2": f"/scratch2/mg963/results/weathernext/WN2/crps_{VARIABLE}_{LEAD}h.zarr"
        }

    print(f"🔹 Loading datasets for {VARIABLE} {LEAD}h...")
    das = {name: open_crps(p) for name, p in paths.items()}

    # Align and Slice
    print(f"🔹 Aligning and slicing from {START_DATE} to {END_DATE}...")
    da_ifs, da_gencast, da_wn2 = xr.align(das["IFS"], das["GenCast"], das["WN2"], join="inner")
    
    da_ifs = da_ifs.sel(time=slice(START_DATE, END_DATE))
    da_gencast = da_gencast.sel(time=slice(START_DATE, END_DATE))
    da_wn2 = da_wn2.sel(time=slice(START_DATE, END_DATE))

    # Compute time mean
    print("🔹 Computing time means...")
    mean_ifs = da_ifs.mean(dim="time").compute()
    mean_gencast = da_gencast.mean(dim="time").compute()
    mean_wn2 = da_wn2.mean(dim="time").compute()

    # Compute differences (Model - Reference)
    # Negative = Model is better (lower CRPS)
    diffs = [
        (mean_gencast - mean_ifs, "GenCast - IFS"),
        (mean_wn2 - mean_ifs, "WN2 - IFS"),
        (mean_wn2 - mean_gencast, "WN2 - GenCast")
    ]

    # Plotting
    print("🔹 Plotting...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), subplot_kw={'projection': ccrs.PlateCarree()})

    cmap = "RdBu_r" 
    
    if VARIABLE == "tp":
        unit = "mm"
        # Convert from meters to mm for plotting
        diffs = [(d * 1000.0, t) for d, t in diffs]
    else:
        unit = "K" if VARIABLE == "t2m" else "m/s"
    
    # Determine robust vmin/vmax (98th percentile of absolute diff across all comparisons)
    all_diff_vals = np.concatenate([d[0].values.flatten() for d in diffs])
    vmax = np.nanpercentile(np.abs(all_diff_vals), 98)
    
    for ax, (diff, title) in zip(axes, diffs):
        # Add features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.1)
        
        im = diff.plot(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=-vmax, vmax=vmax,
            add_colorbar=False
        )
        
        cb = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, extend='both', shrink=0.7)
        
        # Label logic for the 3rd plot
        ref_name = title.split(" - ")[1]
        model_name = title.split(" - ")[0]
        cb.set_label(f"$\Delta$ CRPS ({unit}) [Blue = {model_name} Better]")
        
        ax.set_title(f"{title} ({LEAD}h {VARIABLE} CRPS Mean)\n{START_DATE} to {END_DATE}", fontsize=14)

    plt.tight_layout()
    out_path = f"/scratch2/mg963/results/diagnostics/maps/crps_{VARIABLE}_{LEAD}h_diff_maps.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"🎉 Map saved to {out_path}")

if __name__ == "__main__":
    main()
