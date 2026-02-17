import xarray as xr
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from dask import config as dask_config

# -----------------------------------------
# CONFIG & PATHS
# -----------------------------------------

os.environ["ZARR_V3_EXPERIMENTAL_API"] = "1"
dask_config.set(scheduler="threads", num_workers=8)

# Time period selection
START_DATE = "2024-02-04"
END_DATE = "2024-12-31"

MODEL_SPECS = {
    "WN2": {
        "t2m": "/scratch2/mg963/data/weathernext/wn2/wn2_2024_t2m.zarr",
        "winds": "/scratch2/mg963/data/weathernext/wn2/wn2_2024_10m_wind_speed.zarr",
        "tp": "/scratch2/mg963/data/weathernext/wn2/wn2_2024_precip_cumulative_24_72.zarr",
        "var_map": {"t2m": "2m_temperature", "winds": "10m_wind_speed", "tp": "total_precipitation_cumulative"},
        "member_dim": "sample",
        "rename_coords": {"lat": "latitude", "lon": "longitude"},
        "to_valid_time": True,
    },
    "GenCast": {
        "t2m": "/scratch2/mg963/data/weathernext/gencast/gencast_2024_t2m.zarr",
        "winds": "/scratch2/mg963/data/weathernext/gencast/gencast_2024_10m_wind_speed.zarr",
        "tp": "/scratch2/mg963/data/weathernext/gencast/gencast_2024_precip_cumulative_24_72.zarr",
        "var_map": {"t2m": "2m_temperature", "winds": "10m_wind_speed", "tp": "total_precipitation_cumulative"},
        "member_dim": "sample",
        "rename_coords": {"lat": "latitude", "lon": "longitude"},
        "to_valid_time": True,
    },
    "IFS-ENS": {
        "t2m": "/scratch2/mg963/data/ecmwf/ensembles/ifs/zarr_ens/t2m_f{lead}.zarr",
        "winds": "/scratch2/mg963/data/ecmwf/ensembles/ifs/zarr_ens/wspd10_f{lead}.zarr",
        "tp": "/scratch2/mg963/data/ecmwf/ensembles/ifs/zarr_ens/tp_f{lead}.zarr",
        "var_map": {"t2m": "t2m", "winds": "wspd10", "tp": "tp"},
        "member_dim": "number",
        "rename_coords": {},
        "to_valid_time": True,
    },
    "Deterministic": {
        "t2m": "/scratch2/mg963/data/ecmwf/analysis/ifs_ens_control_06z.zarr",
        "winds": "/scratch2/mg963/data/ecmwf/analysis/ifs_ens_control_06z.zarr",
        "tp": "/scratch2/mg963/data/ecmwf/era5/tp/era5/daily_utc6_2024/era5_tp_daily_utc6_2024_processed.nc",
        "var_map": {"t2m": "t2m", "winds": "ws10", "tp": "tp_daily_utc6_mm"},
        "member_dim": None,
        "rename_coords": {"valid_time": "time"}, 
        "to_valid_time": False,
    }
}

OUT_DIR = Path("/scratch2/mg963/results/diagnostics/correlation_maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# UTILS
# -----------------------------------------

def open_as_dataarray(zarr_path, variable=None, sel_dict=None, rename_coords=None, to_valid_time=False, fixed_lead_hours=None, standardize_lon=False):
    """Robust loader adapted from s139."""
    zarr_path = str(zarr_path)
    try:
        if zarr_path.endswith('.zarr'):
            ds = xr.open_dataset(zarr_path, engine='zarr', consolidated=False, chunks={})
        else:
            ds = xr.open_dataset(zarr_path, chunks={})
    except Exception as e:
        print(f"   ⚠️ Fallback opening for {zarr_path}: {e}")
        try:
            ds = xr.open_zarr(zarr_path, consolidated=True)
        except:
            ds = xr.open_zarr(zarr_path, consolidated=False)

    da = ds[variable] if variable else ds[list(ds.data_vars)[0]]
    if "time" in da.dims and not da.indexes["time"].is_unique:
        da = da.drop_duplicates(dim="time")
    if rename_coords:
        da = da.rename({k: v for k, v in rename_coords.items() if k in da.dims or k in da.coords})
    if sel_dict:
        effective_sel = {k: v for k, v in sel_dict.items() if k in da.coords or k in da.dims}
        if effective_sel:
            da = da.sel(effective_sel)
    if to_valid_time and "time" in da.coords:
        valid_time = None
        if "prediction_timedelta" in da.coords:
            valid_time = da["time"] + da["prediction_timedelta"]
        elif "step" in da.coords:
            dt = da["step"] if np.issubdtype(da["step"].dtype, np.timedelta64) else da["step"] * np.timedelta64(1, "h")
            valid_time = da["time"] + dt
        elif "lead_time" in da.coords:
            dt = da["lead_time"] if np.issubdtype(da["lead_time"].dtype, np.timedelta64) else da["lead_time"] * np.timedelta64(1, "h")
            valid_time = da["time"] + dt
        elif fixed_lead_hours is not None:
            valid_time = da["time"] + np.timedelta64(int(fixed_lead_hours), "h")
        if valid_time is not None:
            da = da.assign_coords(time=valid_time)
            if not da.indexes["time"].is_unique:
                da = da.drop_duplicates(dim="time")
            da = da.sortby("time")
    if standardize_lon:
        lon_name = "longitude" if "longitude" in da.coords else ("lon" if "lon" in da.coords else None)
        if lon_name:
            new_lon = (da[lon_name] + 180) % 360 - 180
            da = da.assign_coords({lon_name: new_lon}).sortby(lon_name)
    return da

# -----------------------------------------
# MAIN
# -----------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var1", type=str, default="t2m", choices=["t2m", "winds", "tp"])
    parser.add_argument("--var2", type=str, default="winds", choices=["t2m", "winds", "tp"])
    parser.add_argument("--lead", type=int, default=24)
    parser.add_argument("--num-members", type=int, default=None, help="Number of members to use")
    args = parser.parse_args()

    v1_name = args.var1
    v2_name = args.var2
    lead = args.lead
    
    print(f"🌍 Temporal Pixel Correlation Map: {v1_name} vs {v2_name} @ {lead}h")

    corr_maps = {}
    
    for model_name, spec in MODEL_SPECS.items():
        print(f"🔹 Processing {model_name}...")
        
        # Load Var 1
        path1 = spec[v1_name].format(lead=lead)
        sel_dict = None
        if model_name in ["WN2", "GenCast"]:
            lc = "prediction_timedelta" if v1_name in ["t2m", "winds"] else "lead_time"
            if model_name == "GenCast" and v1_name in ["t2m", "winds"]: lc = "step"
            sel_dict = {lc: np.timedelta64(lead, "h")}
        
        da1 = open_as_dataarray(path1, variable=spec["var_map"][v1_name], 
                               sel_dict=sel_dict, rename_coords=spec["rename_coords"],
                               to_valid_time=spec["to_valid_time"], 
                               fixed_lead_hours=lead if model_name=="IFS-ENS" else None,
                               standardize_lon=True)
                               
        # Load Var 2
        path2 = spec[v2_name].format(lead=lead)
        sel_dict = None
        if model_name in ["WN2", "GenCast"]:
            lc = "prediction_timedelta" if v2_name in ["t2m", "winds"] else "lead_time"
            if model_name == "GenCast" and v2_name in ["t2m", "winds"]: lc = "step"
            sel_dict = {lc: np.timedelta64(lead, "h")}

        da2 = open_as_dataarray(path2, variable=spec["var_map"][v2_name], 
                               sel_dict=sel_dict, rename_coords=spec["rename_coords"],
                               to_valid_time=spec["to_valid_time"],
                               fixed_lead_hours=lead if model_name=="IFS-ENS" else None,
                               standardize_lon=True)

        if model_name == "Deterministic":
            if v1_name == "tp":
                da1 = da1.assign_coords(time=da1.time + np.timedelta64(6, 'h'))
                if lead == 72: da1 = da1.rolling(time=3, min_periods=3).sum()
            if v2_name == "tp":
                da2 = da2.assign_coords(time=da2.time + np.timedelta64(6, 'h'))
                if lead == 72: da2 = da2.rolling(time=3, min_periods=3).sum()

        da1, da2 = xr.align(da1, da2, join="inner")
        
        member_dim = spec["member_dim"]
        if args.num_members and member_dim and member_dim in da1.dims:
            print(f"   Subsetting to {args.num_members} members...")
            da1 = da1.isel({member_dim: slice(0, args.num_members)})
            da2 = da2.isel({member_dim: slice(0, args.num_members)})

        da1 = da1.sel(time=slice(START_DATE, END_DATE))
        da2 = da2.sel(time=slice(START_DATE, END_DATE))

        print(f"   Computing temporal correlation for {da1.sizes['time']} steps...")
        # Chunk so time is one block for xr.corr
        da1 = da1.chunk({"time": -1, "latitude": 90, "longitude": 90})
        da2 = da2.chunk({"time": -1, "latitude": 90, "longitude": 90})
        
        pixel_corr = xr.corr(da1, da2, dim="time")
        
        if member_dim and member_dim in pixel_corr.dims:
            mean_map = pixel_corr.mean(member_dim).compute()
        else:
            mean_map = pixel_corr.compute()
            
        corr_maps[model_name] = mean_map
        print(f"   Global Mean Correlation: {mean_map.mean().values:.4f}")

    # Plotting
    print("🎨 Generating maps...")
    n_models = len(corr_maps)
    # We add 1 extra panel for the "Winner" map
    total_panels = n_models + (1 if "Deterministic" in corr_maps else 0)
    cols = 2
    rows = (total_panels + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows), 
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    # Standard Model Maps
    for i, (model_name, cmap) in enumerate(corr_maps.items()):
        ax = axes[i]
        im = cmap.plot(ax=ax, transform=ccrs.PlateCarree(),
                       cmap="RdBu_r", vmin=-1, vmax=1,
                       add_colorbar=False)
        ax.set_title(f"{model_name}", fontsize=14, fontweight='bold')
        ax.coastlines(color='0.3', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)

    # Winner Map (Closest to Deterministic)
    if "Deterministic" in corr_maps:
        ax = axes[len(corr_maps)]
        ref = corr_maps["Deterministic"]
        others = {k: v for k, v in corr_maps.items() if k != "Deterministic"}
        
        # Compute absolute distance to reference per model
        diffs = []
        model_names_others = list(others.keys())
        for m_name in model_names_others:
            # Align if grids differ slightly
            m_map = others[m_name].reindex_like(ref, method="nearest")
            diffs.append(np.abs(m_map - ref))
        
        # winner_idx will be 0, 1, 2... matching model_names_others
        diff_stack = xr.concat(diffs, dim="model")
        winner_idx = diff_stack.argmin(dim="model")
        
        # Setup discrete colormap
        from matplotlib.colors import BoundaryNorm, ListedColormap
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(model_names_others)]
        cmap_discrete = ListedColormap(colors)
        
        im_w = winner_idx.plot(ax=ax, transform=ccrs.PlateCarree(),
                              cmap=cmap_discrete, add_colorbar=False)
        
        ax.set_title("Closest Match to Deterministic", fontsize=14, fontweight='bold', color='darkred')
        ax.coastlines(color='0.3', linewidth=0.8)
        
        # Add a custom legend for the winner map
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], label=model_names_others[i]) for i in range(len(model_names_others))]
        ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, -0.2), 
                  ncol=len(model_names_others), fontsize=10, frameon=False)

    # Hide unused axes
    for j in range(total_panels, len(axes)):
        axes[j].axis('off')

    # Add shared colorbar for correlation maps
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.015])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Pearson Correlation Coefficient')
    
    title = f"Temporal Inter-variable Correlation Map: {v1_name} vs {v2_name} (@ {lead}h)"
    fig.suptitle(title, fontsize=16, y=0.95)
    
    mem_suffix = f"_m{args.num_members}" if args.num_members else ""
    out_img = OUT_DIR / f"map_corr_{v1_name}_{v2_name}_{lead}h{mem_suffix}.png"
    plt.savefig(out_img, bbox_inches="tight", dpi=150)
    print(f"📊 Plot saved to {out_img}")

if __name__ == "__main__":
    main()
