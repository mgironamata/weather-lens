import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# -----------------------------------------
# CONFIG & PATHS
# -----------------------------------------

# Time period selection
START_DATE = "2024-02-04"
END_DATE = "2024-12-31"

# Sampling parameters
N_SAMPLES_PER_STEP = 500  # Number of random pairs to sample per time step
MAX_DIST_KM = 5000          # Maximum distance to consider
BIN_SIZE_KM = 100           # Bin width for distance

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

OUT_DIR = Path("/scratch2/mg963/results/diagnostics/spatial_correlation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# UTILS
# -----------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute Great Circle distance in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def open_as_dataarray(zarr_path, variable=None, sel_dict=None, rename_coords=None, to_valid_time=False, fixed_lead_hours=None, standardize_lon=False):
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except:
        ds = xr.open_zarr(zarr_path, consolidated=False) if str(zarr_path).endswith('.zarr') else xr.open_dataset(zarr_path)

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

def compute_corr_decay_step(field, lats, lons, n_samples=10000, max_dist=5000, bin_size=100):
    """
    Compute spatial correlation decay for a single 2D field.
    field: (lat, lon)
    """
    # Flatten
    f_flat = field.flatten()
    lat_flat = lats.flatten()
    lon_flat = lons.flatten()
    
    # Mask NaNs
    mask = ~np.isnan(f_flat)
    f_flat = f_flat[mask]
    lat_flat = lat_flat[mask]
    lon_flat = lon_flat[mask]
    
    n_points = f_flat.shape[0]
    if n_points < 2: return None
    
    # Sample pairs
    idx1 = np.random.randint(0, n_points, n_samples)
    idx2 = np.random.randint(0, n_points, n_samples)
    
    # Distances
    dists = haversine_distance(lat_flat[idx1], lon_flat[idx1], lat_flat[idx2], lon_flat[idx2])
    
    # Anomalies (subtract spatial mean)
    f_mean = np.mean(f_flat)
    f_std = np.std(f_flat)
    if f_std == 0: return None
    
    # Correlation product: (x_i - mu)(x_j - mu) / sigma^2
    corr_prod = (f_flat[idx1] - f_mean) * (f_flat[idx2] - f_mean) / (f_std**2)
    
    # Binning
    bins = np.arange(0, max_dist + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2
    
    # Use digitize for fast binning
    bin_idx = np.digitize(dists, bins) - 1
    
    # Aggregate
    bin_sums = np.zeros(len(bin_centers))
    bin_counts = np.zeros(len(bin_centers))
    
    for i in range(len(bin_centers)):
        m = (bin_idx == i)
        if np.any(m):
            bin_sums[i] = np.sum(corr_prod[m])
            bin_counts[i] = np.sum(m)
            
    return bin_centers, bin_sums, bin_counts

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
    
    print(f"📏 Spatial Correlation Decay: {variable} @ {lead}h (Mask: {args.mask_type})")

    # 1. Load Obs (Ground Truth)
    obs_rename = {"valid_time": "time"} if variable == "tp" else None
    obs = open_as_dataarray(OBS_PATHS[variable], variable=OBS_VAR_MAP[variable], 
                           rename_coords=obs_rename, standardize_lon=True)
    
    if variable == "tp":
        obs = obs.assign_coords(time=obs.time + np.timedelta64(6, 'h')) * 0.001 # mm -> m
        if lead == 72:
            obs = obs.rolling(time=3, min_periods=3).sum()
    
    obs = obs.sel(time=slice(START_DATE, END_DATE))

    # Load Mask if requested
    lsm = None
    if args.mask_path:
        print(f"🎭 Loading mask from {args.mask_path}...")
        try:
            ds_mask = xr.open_dataset(args.mask_path) if args.mask_path.endswith('.nc') else xr.open_zarr(args.mask_path)
            lsm = ds_mask[args.mask_var]
            # Handle time if present
            if "time" in lsm.dims: lsm = lsm.isel(time=0, drop=True)
            elif "valid_time" in lsm.dims: lsm = lsm.isel(valid_time=0, drop=True)
            
            # Standardize lon to [-180, 180]
            if lsm.longitude.max() > 180:
                lsm = lsm.assign_coords(longitude=(((lsm.longitude + 180) % 360) - 180)).sortby("longitude")
            # Align with obs (lat/lon only)
            lsm = lsm.reindex_like(obs, method="nearest")
        except Exception as e:
            print(f"   ⚠️  Failed to load mask: {e}. Proceeding without mask.")
            lsm = None
    
    # Meshgrid for lats/lons
    lon_grid, lat_grid = np.meshgrid(obs.longitude.values, obs.latitude.values)
    
    results = {}
    
    # 2. Process Obs
    print("🔹 Processing Ground Truth (Obs)...")
    all_sums = None
    all_counts = None
    
    for t in tqdm(range(len(obs.time)), desc="Obs"):
        field = obs.isel(time=t)
        
        # Apply Mask
        if lsm is not None:
            if args.mask_type == "land":
                field = field.where(lsm > 0.5)
            elif args.mask_type == "ocean":
                field = field.where(lsm <= 0.5)
        
        res = compute_corr_decay_step(field.values, lat_grid, lon_grid, 
                                     n_samples=N_SAMPLES_PER_STEP, 
                                     max_dist=MAX_DIST_KM, bin_size=BIN_SIZE_KM)
        if res:
            centers, sums, counts = res
            if all_sums is None:
                all_sums = sums
                all_counts = counts
            else:
                all_sums += sums
                all_counts += counts
    
    results["Obs"] = all_sums / np.where(all_counts > 0, all_counts, 1)
    dist_centers = centers

    # 3. Process Models
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
        
        # Align with Obs
        _, ens_aligned = xr.align(obs, da_ens, join="inner")
        
        # We sample members to save time if needed, but let's try all or a subset
        n_members = ens_aligned.sizes[spec["member_dim"]]
        members_to_use = np.arange(min(n_members, 5)) # Use first 5 members to get a good estimate without taking forever
        
        m_sums = None
        m_counts = None
        
        for m_idx in members_to_use:
            for t in tqdm(range(len(ens_aligned.time)), desc=f"{model_name} M{m_idx}"):
                field = ens_aligned.isel({spec["member_dim"]: m_idx, "time": t})
                
                # Apply Mask
                if lsm is not None:
                    if args.mask_type == "land":
                        field = field.where(lsm > 0.5)
                    elif args.mask_type == "ocean":
                        field = field.where(lsm <= 0.5)

                res = compute_corr_decay_step(field.values, lat_grid, lon_grid, 
                                             n_samples=N_SAMPLES_PER_STEP, 
                                             max_dist=MAX_DIST_KM, bin_size=BIN_SIZE_KM)
                if res:
                    _, sums, counts = res
                    if m_sums is None:
                        m_sums = sums
                        m_counts = counts
                    else:
                        m_sums += sums
                        m_counts += counts
        
        results[model_name] = m_sums / np.where(m_counts > 0, m_counts, 1)

    # 4. Save & Plot
    df = pd.DataFrame(results, index=dist_centers)
    df.index.name = "distance_km"
    mask_suffix = f"_{args.mask_type}" if args.mask_type != "all" else ""
    out_csv = OUT_DIR / f"spatial_corr_{variable}_{lead}h{mask_suffix}.csv"
    df.to_csv(out_csv)
    print(f"✅ Data saved to {out_csv}")

    plt.figure(figsize=(10, 6))
    colors = {"Obs": "black", "WN2": "#1f77b4", "GenCast": "#ff7f0e", "IFS-ENS": "#2ca02c"}
    styles = {"Obs": "-", "WN2": "--", "GenCast": "--", "IFS-ENS": "--"}
    
    for col in df.columns:
        plt.plot(df.index, df[col], label=col, color=colors.get(col, None), linestyle=styles.get(col, "-"), linewidth=2)

    plt.title(f"Spatial Correlation Decay: {variable} @ {lead}h lead (Mask: {args.mask_type})", fontsize=14)
    plt.xlabel("Distance (km)", fontsize=12)
    plt.ylabel("Correlation", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.2, 1.1)
    
    out_plot = OUT_DIR / f"spatial_corr_{variable}_{lead}h{mask_suffix}.png"
    plt.savefig(out_plot, dpi=200, bbox_inches='tight')
    print(f"✅ Plot saved to {out_plot}")

if __name__ == "__main__":
    main()

# example: python s138-spatial-correlation-decay.py --var t2m --lead 24