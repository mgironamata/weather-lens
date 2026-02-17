import xarray as xr
import numpy as np
import argparse
import time
from pathlib import Path
from dask.base import compute
from dask import config as dask_config

# -----------------------------------------
# CONFIG & PATHS
# -----------------------------------------

dask_config.set(scheduler="threads", num_workers=8)

# Variogram Score parameters
P_EXPONENT = 0.5  # Standard choice for Variogram Score (VS_p)
N_SAMPLES = 5000  # Number of random pairs of points to sample per time step

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
    }
}

OBS_PATHS = {
    "t2m": "/scratch2/mg963/data/ecmwf/analysis/ifs_ens_control_06z.zarr",
    "winds": "/scratch2/mg963/data/ecmwf/analysis/ifs_ens_control_06z.zarr",
    "tp": "/scratch2/mg963/data/ecmwf/era5/tp/era5/daily_utc6_2024/era5_tp_daily_utc6_2024_processed.nc",
}

OBS_VAR_MAP = {
    "t2m": "t2m",
    "winds": "ws10",
    "tp": "tp_daily_utc6_mm",
}

OUT_DIR = Path("/scratch2/mg963/results/diagnostics/variogram")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# UTILS
# -----------------------------------------

def open_as_dataarray(zarr_path, variable=None, sel_dict=None, rename_coords=None, to_valid_time=False, fixed_lead_hours=None, standardize_lon=False):
    """Robust loader adapted from s116."""
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except:
        ds = xr.open_zarr(zarr_path, consolidated=False) if zarr_path.endswith('.zarr') else xr.open_dataset(zarr_path)

    da = ds[variable] if variable else ds[list(ds.data_vars)[0]]
    
    if "time" in da.dims and not da.indexes["time"].is_unique:
        da = da.drop_duplicates(dim="time")

    if sel_dict:
        da = da.sel(sel_dict)

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
            da = da.assign_coords(time=valid_time).drop_duplicates(dim="time")

    if rename_coords:
        da = da.rename({k: v for k, v in rename_coords.items() if k in da.dims or k in da.coords})
    
    if standardize_lon:
        lon_name = "longitude" if "longitude" in da.coords else ("lon" if "lon" in da.coords else None)
        if lon_name:
            new_lon = (da[lon_name] + 180) % 360 - 180
            da = da.assign_coords({lon_name: new_lon}).sortby(lon_name)
            
    return da

def compute_vs_step(ens_step, obs_step, p=0.5, n_samples=1000):
    """
    Compute Variogram Score for a single time step using random sampling.

    ens_step: (member, lat, lon)
    obs_step: (lat, lon)
    """
    # Flatten spatial dims
    ens_flat = ens_step.reshape(ens_step.shape[0], -1)
    obs_flat = obs_step.flatten()
    
    # Mask NaNs
    mask = ~np.isnan(obs_flat) & ~np.any(np.isnan(ens_flat), axis=0)
    ens_flat = ens_flat[:, mask]
    obs_flat = obs_flat[mask]
    
    n_points = obs_flat.shape[0]
    if n_points < 2: return np.nan
    
    # Sample pairs
    idx1 = np.random.randint(0, n_points, n_samples)
    idx2 = np.random.randint(0, n_points, n_samples)
    
    # |y_i - y_j|^p
    obs_diff = np.abs(obs_flat[idx1] - obs_flat[idx2])**p
    # 1/M * sum |x_m,i - x_m,j|^p
    ens_diff = np.mean(np.abs(ens_flat[:, idx1] - ens_flat[:, idx2])**p, axis=0)
    
    return np.mean((obs_diff - ens_diff)**2)

# -----------------------------------------
# MAIN
# -----------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", type=str, default="t2m", choices=["t2m", "winds", "tp"])
    parser.add_argument("--lead", type=int, default=24)
    parser.add_argument("--mask-path", type=str, default=None, help="Path to land-sea mask file")
    parser.add_argument("--mask-var", type=str, default="lsm", help="Variable name for mask")
    parser.add_argument("--mask-type", type=str, default="all", choices=["all", "land", "ocean"], help="Mask type")
    args = parser.parse_args()

    variable = args.var
    lead = args.lead
    
    print(f"🚀 Variogram Score: {variable} @ {lead}h (Mask: {args.mask_type})")

    # 1. Load Obs
    obs_rename = {"valid_time": "time"} if variable == "tp" else None
    obs = open_as_dataarray(OBS_PATHS[variable], variable=OBS_VAR_MAP[variable], 
                           rename_coords=obs_rename, standardize_lon=True)
    
    if variable == "tp":
        obs = obs.assign_coords(time=obs.time + np.timedelta64(6, 'h')) * 0.001 # mm -> m
        if lead == 72:
            print("   ⚠️  Computing 3-day rolling sum for ERA5 (72h lead)...")
            obs = obs.rolling(time=3, min_periods=3).sum()
    
    # Slice obs to requested period
    obs = obs.sel(time=slice(START_DATE, END_DATE))
    print(f"   Obs period: {obs.time.values[0]} to {obs.time.values[-1]} ({len(obs.time)} steps)")
    
    # Load Mask if requested
    lsm = None
    if args.mask_path:
        print(f"🎭 Loading mask from {args.mask_path}...")
        try:
            ds_mask = xr.open_dataset(args.mask_path) if args.mask_path.endswith('.nc') else xr.open_zarr(args.mask_path)
            lsm = ds_mask[args.mask_var]
            # Standardize lon to [-180, 180]
            if lsm.longitude.max() > 180:
                lsm = lsm.assign_coords(longitude=(((lsm.longitude + 180) % 360) - 180)).sortby("longitude")
            # Align with obs (lat/lon only)
            lsm = lsm.reindex_like(obs, method="nearest")
        except Exception as e:
            print(f"   ⚠️  Failed to load mask: {e}. Proceeding without mask.")
            lsm = None

    results = {}
    for model_name, spec in MODEL_SPECS.items():
        print(f"🔹 Processing {model_name}...")
        path = spec[variable].format(lead=lead)
        
        sel_dict = None
        if model_name == "WN2":
            lead_coord = "prediction_timedelta" if variable in ["t2m", "winds"] else "lead_time"
            sel_dict = {lead_coord: np.timedelta64(lead, "h")}
        elif model_name == "GenCast":
            lead_coord = "step" if variable in ["t2m", "winds"] else "lead_time"
            sel_dict = {lead_coord: np.timedelta64(lead, "h")}
            
        da_ens = open_as_dataarray(path, variable=spec["var_map"][variable], 
                                  sel_dict=sel_dict, rename_coords=spec["rename_coords"],
                                  to_valid_time=True, fixed_lead_hours=lead if model_name=="IFS-ENS" else None,
                                  standardize_lon=True)
        
        # Align
        obs_aligned, ens_aligned = xr.align(obs, da_ens, join="inner")
        
        # Apply Mask
        if lsm is not None:
            m = lsm.reindex_like(obs_aligned, method="nearest")
            if args.mask_type == "land":
                obs_aligned = obs_aligned.where(m > 0.5)
                ens_aligned = ens_aligned.where(m > 0.5)
            elif args.mask_type == "ocean":
                obs_aligned = obs_aligned.where(m <= 0.5)
                ens_aligned = ens_aligned.where(m <= 0.5)

        print(f"   Aligned shape: {ens_aligned.sizes}")
        
        if ens_aligned.sizes["time"] == 0:
            print(f"   ⚠️  No overlapping time steps for {model_name}. Skipping.")
            continue

        # Compute VS per time step
        vs_times = []
        for t in range(ens_aligned.sizes["time"]):
            print(f"   Time {t+1}/{ens_aligned.sizes['time']}", end="\r")
            e_step = ens_aligned.isel(time=t).values # (member, lat, lon)
            o_step = obs_aligned.isel(time=t).values # (lat, lon)
            vs_times.append(compute_vs_step(e_step, o_step, p=P_EXPONENT, n_samples=N_SAMPLES))
        
        results[model_name] = xr.DataArray(
            vs_times, 
            coords={"time": obs_aligned.time}, 
            dims=["time"], 
            name=model_name
        )
        print(f"\n   Mean VS: {np.nanmean(vs_times):.6f}")

    # Save
    if not results:
        print("❌ No results to save.")
        return

    ds_vs = xr.Dataset(results)
    mask_suffix = f"_{args.mask_type}" if args.mask_type != "all" else ""
    out_path = OUT_DIR / f"vs_{variable}_{lead}h{mask_suffix}.nc"
    ds_vs.to_netcdf(out_path)
    print(f"💾 Saved to {out_path}")

if __name__ == "__main__":
    main()

# example: python s134-variogram-score.py --var t2m --lead 24