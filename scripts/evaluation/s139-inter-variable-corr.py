import xarray as xr
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
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

LSM_PATH = "/scratch2/mg963/data/ecmwf/era5/constants/era5_lsm.nc"
OUT_DIR = Path("/scratch2/mg963/results/diagnostics/correlation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# UTILS
# -----------------------------------------

def open_as_dataarray(zarr_path, variable=None, sel_dict=None, rename_coords=None, to_valid_time=False, fixed_lead_hours=None, standardize_lon=False):
    """Robust loader adapted from s134."""
    zarr_path = str(zarr_path)
    
    # Try opening with engine='zarr' for better v3 support
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
        # Check if lead_time or step is in da
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
            # Handle possible int lead_time
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
            da = da.assign_coords({lon_name: new_lon})
            # Handle sorting which can be slow with dask
            da = da.sortby(lon_name)
            
    return da

# -----------------------------------------
# MAIN
# -----------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var1", type=str, default="t2m", choices=["t2m", "winds", "tp"])
    parser.add_argument("--var2", type=str, default="winds", choices=["t2m", "winds", "tp"])
    parser.add_argument("--lead", type=int, default=24)
    parser.add_argument("--land-only", action="store_true", help="Restrict correlation to land points")
    parser.add_argument("--num-members", type=int, default=None, help="Number of ensemble members to use (default: all)")
    parser.add_argument("--mode", type=str, default="raw", choices=["raw", "abs", "squared", "fisher"], 
                        help="Averaging mode: raw, abs (coupling strength), squared (explained variance), or fisher (statistical Z-avg)")
    args = parser.parse_args()

    v1_name = args.var1
    v2_name = args.var2
    lead = args.lead
    
    print(f"🚀 Inter-variable Correlation: {v1_name} vs {v2_name} @ {lead}h")

    # Load Mask if requested
    lsm = None
    if args.land_only:
        print(f"🎭 Loading LSM from {LSM_PATH}...")
        ds_lsm = xr.open_dataset(LSM_PATH)
        lsm = ds_lsm["lsm"]
        if lsm.longitude.max() > 180:
            lsm = lsm.assign_coords(longitude=(((lsm.longitude + 180) % 360) - 180)).sortby("longitude")

    results = {}
    
    for model_name, spec in MODEL_SPECS.items():
        print(f"🔹 Processing {model_name}...")
        
        # Determine lead coordinate name for AI models
        # (This logic is slightly redundant with open_as_dataarray sel_dict logic but helps here)
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

        # TP specific handling for ERA5 (Deterministic)
        if model_name == "Deterministic":
            if v1_name == "tp":
                da1 = da1.assign_coords(time=da1.time + np.timedelta64(6, 'h'))
                if lead == 72:
                    print(f"   ⚠️  Rolling sum for ERA5 {v1_name} (72h)...")
                    da1 = da1.rolling(time=3, min_periods=3).sum()
            if v2_name == "tp":
                da2 = da2.assign_coords(time=da2.time + np.timedelta64(6, 'h'))
                if lead == 72:
                    print(f"   ⚠️  Rolling sum for ERA5 {v2_name} (72h)...")
                    da2 = da2.rolling(time=3, min_periods=3).sum()

        # Align vars
        da1, da2 = xr.align(da1, da2, join="inner")
        
        # Limit members if requested
        member_dim = spec["member_dim"]
        if args.num_members is not None and member_dim and member_dim in da1.dims:
            print(f"   Subsetting to first {args.num_members} members...")
            da1 = da1.isel({member_dim: slice(0, args.num_members)})
            da2 = da2.isel({member_dim: slice(0, args.num_members)})

        # Slices to period
        da1 = da1.sel(time=slice(START_DATE, END_DATE))
        da2 = da2.sel(time=slice(START_DATE, END_DATE))
        
        if da1.sizes["time"] == 0:
            print(f"   ⚠️  No overlapping time steps for {model_name}. Skipping.")
            continue

        # Spatial weights
        weights = np.cos(np.deg2rad(da1.latitude))
        if lsm is not None:
            m = lsm.reindex_like(da1, method="nearest")
            weights = weights.where(m > 0.5, 0.0)
        
        # Compute area-weighted spatial correlation per time step to save memory
        corr_list = []
        for t in range(da1.sizes["time"]):
            print(f"   Time {t+1}/{da1.sizes['time']}", end="\r")
            s1 = da1.isel(time=t).compute()
            s2 = da2.isel(time=t).compute()
            
            # Weighted operations on computed step
            s1_w = s1.weighted(weights)
            s2_w = s2.weighted(weights)
            
            m1 = s1_w.mean(("latitude", "longitude"))
            m2 = s2_w.mean(("latitude", "longitude"))
            
            s1_p = s1 - m1
            s2_p = s2 - m2
            
            cov = (s1_p * s2_p).weighted(weights).mean(("latitude", "longitude"))
            var1 = (s1_p**2).weighted(weights).mean(("latitude", "longitude"))
            var2 = (s2_p**2).weighted(weights).mean(("latitude", "longitude"))
            
            corr_step = cov / np.sqrt(var1 * var2)
            corr_list.append(corr_step)
        
        print(f"\n   Concatenating results for {model_name}...")
        corr = xr.concat(corr_list, dim=da1.time)
        
        member_dim = spec["member_dim"]
        
        # Apply transformation based on mode
        if args.mode == "abs":
            print("   Mode: Absolute Magnitude")
            corr = np.abs(corr)
        elif args.mode == "squared":
            print("   Mode: Explained Variance (R^2)")
            corr = corr**2
        elif args.mode == "fisher":
            print("   Mode: Fisher Z-transformation")
            # clip to avoid infinity at r=1/-1
            corr = np.clip(corr, -0.999, 0.999)
            corr = np.arctanh(corr)

        if member_dim and member_dim in corr.dims:
            mean_val = corr.mean(member_dim)
            std_val = corr.std(member_dim)
        else:
            mean_val = corr
            std_val = xr.zeros_like(corr)

        # Back-transform if Fisher
        if args.mode == "fisher":
            mean_val = np.tanh(mean_val)
            std_val = np.tanh(std_val) # Note: std in Z-space back-transformed is an approximation

        results[model_name] = {
            "mean": mean_val.values.squeeze(),
            "std": std_val.values.squeeze(),
            "time": corr.time.values
        }
        
        print(f"   Avg Correlation: {np.nanmean(results[model_name]['mean']):.4f}")

    # Plot
    if not results:
        print("❌ No results to plot.")
        return

    plt.figure(figsize=(14, 7))
    for model_name, data in results.items():
        if model_name == "Deterministic":
            plt.plot(data["time"], data["mean"], label=model_name, color='black', linewidth=2, linestyle='--')
        else:
            plt.plot(data["time"], data["mean"], label=model_name)
            if np.any(data["std"] > 0):
                plt.fill_between(data["time"], data["mean"] - data["std"], data["mean"] + data["std"], alpha=0.15)
    
    mode_titles = {
        "raw": "",
        "abs": " [Absolute Magnitude]",
        "squared": " [Explained Variance R^2]",
        "fisher": " [Fisher Z-Avg]"
    }
    title = f"Inter-variable Correlation: {v1_name} vs {v2_name} (@ {lead}h){mode_titles[args.mode]}"
    if args.land_only: title += " [Land Only]"
    plt.title(title, fontsize=14)
    plt.xlabel("Valid Time", fontsize=12)
    plt.ylabel(f"Correlation ({args.mode})", fontsize=12)
    plt.legend(frameon=True, loc='best')
    plt.grid(True, alpha=0.3)
    
    land_suffix = "_land" if args.land_only else ""
    mode_suffix = f"_{args.mode}" if args.mode != "raw" else ""
    mem_suffix = f"_m{args.num_members}" if args.num_members is not None else ""
    out_img = OUT_DIR / f"corr_{v1_name}_{v2_name}_{lead}h{land_suffix}{mode_suffix}{mem_suffix}.png"
    plt.savefig(out_img, bbox_inches="tight", dpi=150)
    print(f"📊 Plot saved to {out_img}")

if __name__ == "__main__":
    main()
