# s122-obs-counts-vs-thresholds.py

import xarray as xr
import numpy as np
import dask
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------------
# CONFIG
# -----------------------------------------

dask.config.set(scheduler="threads", num_workers=8)

variable = "winds"      # "t2m" or "winds"

OBS_VAR_MAP = {
    "t2m":   "t2m",
    "winds": "ws10",   # analysis/control name
}

VAR_CONFIG = {
    "t2m": {
        "long_name": "2m Temperature",
        "short_name": "T2m",
        "obs_units": "K",           # units in observation file
        "display_units": "°C",      # units for display/plotting
        "convert_obs": lambda x: x - 273.15,              # K to °C (not actually needed here)
        "convert_threshold_display": lambda x: x - 273.15,  # K to °C for thresholds
        "threshold_label": "Temperature",
    },
    "winds": {
        "long_name": "10m Wind Speed",
        "short_name": "WS10",
        "obs_units": "m/s",
        "display_units": "m/s",
        "convert_obs": lambda x: x,
        "convert_threshold_display": lambda x: x,
        "threshold_label": "Wind Speed",
    },
}

var_cfg = VAR_CONFIG[variable]
OBS_PATH = "/scratch2/mg963/data/ecmwf/analysis/ifs_ens_control_06z.zarr"
OBS_VAR  = OBS_VAR_MAP[variable]

# Same quantiles as in your TWCRPS script
Q_LIST = [
    0.50, 0.90, 0.95, 0.99,
    0.995, 0.999, 0.9995, 0.9999,
    0.99995, 0.99999,
    0.999995,
    0.999999,
]

OUT_DIR = Path("/scratch2/mg963/results/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG  = OUT_DIR / f"obs_counts_{variable}_thresholds.png"
OUT_DATA = OUT_DIR / f"obs_counts_{variable}_thresholds.nc"


# -----------------------------------------
# 1. Load obs & compute thresholds
# -----------------------------------------

print("🔹 Loading observations...")
ds_obs = xr.open_zarr(OBS_PATH, consolidated=True)
obs = ds_obs[OBS_VAR]

# Drop duplicate times if present
if "time" in obs.dims and not obs.indexes["time"].is_unique:
    print("   ⚠️  WARNING: obs has duplicate times. Dropping duplicates along 'time'...")
    obs = obs.drop_duplicates(dim="time")

# Chunk reasonably for quantiles and counting
obs = obs.chunk({"time": 16, "latitude": 90, "longitude": 180})

print("   Computing quantile-based thresholds...")
quantiles = obs.quantile(Q_LIST, dim=("time", "latitude", "longitude"))
thresholds_native = quantiles.values  # native units (K or m/s)
thresholds_display = var_cfg["convert_threshold_display"](thresholds_native)

print(f"   Thresholds ({var_cfg['obs_units']}, {var_cfg['display_units']}):")
for q, native, display in zip(Q_LIST, thresholds_native, thresholds_display):
    print(f"      q={q:.6f} -> {native:.2f} {var_cfg['obs_units']} ({display:.2f} {var_cfg['display_units']})")

# Total number of valid obs (for fractions)
print("   Computing total valid obs count...")
valid_mask = obs.notnull()
total_valid = valid_mask.sum(dim=("time", "latitude", "longitude")).compute().item()
print(f"   Total valid obs: {total_valid}")


# -----------------------------------------
# 2. Count obs above each threshold (memory-safe loop)
# -----------------------------------------

def counts_for_thresholds(obs_da, thresholds_native):
    counts = []
    for i, thr in enumerate(thresholds_native):
        print(f"      Threshold {i+1}/{len(thresholds_native)}: {thr:.4f}", end="\r")
        mask = obs_da >= thr
        count = mask.sum(dim=("time", "latitude", "longitude"))
        count_val = int(count.compute().item())
        counts.append(count_val)
    print()
    return np.array(counts, dtype=np.int64)

print("\n🔹 Counting obs above thresholds...")
obs_counts = counts_for_thresholds(obs, thresholds_native)
obs_fractions = obs_counts / total_valid


# -----------------------------------------
# 3. Package results as Dataset and save
# -----------------------------------------

print("\n🔹 Packaging results...")

obs_units_clean = var_cfg["obs_units"].replace('/', '_per_')
display_units_clean = (
    var_cfg["display_units"]
    .replace('°', 'deg')
    .replace('/', '_per_')
)

ds_out = xr.Dataset(
    data_vars={
        "obs_count":    (["threshold"], obs_counts),
        "obs_fraction": (["threshold"], obs_fractions),
    },
    coords={
        f"threshold_{obs_units_clean}":   (["threshold"], thresholds_native),
        f"threshold_{display_units_clean}": (["threshold"], thresholds_display),
        "quantile":    (["threshold"], Q_LIST),
    },
)

ds_out.attrs["variable"] = variable
ds_out.attrs["variable_long_name"] = var_cfg["long_name"]
ds_out.attrs["obs_units"] = var_cfg["obs_units"]
ds_out.attrs["display_units"] = var_cfg["display_units"]
ds_out.attrs["description"] = (
    "Counts and fractions of analysis values above each quantile-defined threshold."
)

print(ds_out)

print(f"💾 Saving counts to {OUT_DATA}")
ds_out.to_netcdf(OUT_DATA)


# -----------------------------------------
# 4. Plot: threshold (display units) vs obs count
# -----------------------------------------

print(f"📈 Plotting to {OUT_FIG} ...")

plt.figure(figsize=(7, 5))

plt.plot(thresholds_display, obs_counts, marker="o")
plt.xlabel(f"Threshold {var_cfg['threshold_label']} ({var_cfg['display_units']})")
plt.ylabel("Number of observations above threshold")
plt.title(f"{var_cfg['long_name']} – Obs count vs threshold\n(IFS control)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150)

print("✅ Done.")