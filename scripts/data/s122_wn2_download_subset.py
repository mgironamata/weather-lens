# sXXX_wn2_download_subset.py

import xarray as xr
import gcsfs
import pandas as pd
import os
import shutil
import numpy as np

CONFIG = {
    "PERFORM_DOWNLOAD": True,
    "OVERWRITE_EXISTING_STORE": True,   # set False if you really want to append

    "project_id": "weathernextforecasts",
    "bucket_path": "gs://weathernext/weathernext_2_0_0/zarr/2024_to_2025/predictions.zarr",

    # NEW: precip-only (or multi-var) store for whatever steps you choose
    "zarr_store_path": "/scratch2/mg963/data/weathernext/wn2/wn2_2024_precip_extra_steps.zarr",

    "start_date": "2024-01-01",
    "end_date":   "2024-12-31",
    "init_hour":  6,

    # variables you want in this store
    "variables": [
        "total_precipitation_6hr",
        # add others if you want a multi-var store
    ],

    # WN2 uses prediction_timedelta (timedelta64) on a 6-hour grid
    # e.g. extra leads: 6, 12, 18, 30, 36, 42, 48, 54, 60, 66
    "steps_hours": [6, 12, 18, 30, 36, 42, 48, 54, 60, 66],

    # Chunking for the local store
    # (this matches the “GenCast-style” chunking and works fine for WN2: 360x720 grid)
    "chunk_plan": {
        "time": 8,
        "sample": -1,
        "lat": 90,
        "lon": 180,
    },

    "batch_days": 16,
}

def clean_attributes(ds: xr.Dataset) -> xr.Dataset:
    for k, v in list(ds.attrs.items()):
        if isinstance(v, (bool, np.bool_)):
            ds.attrs[k] = int(v)
    for vname in ds.variables:
        attrs = ds[vname].attrs
        for k, v in list(attrs.items()):
            if isinstance(v, (bool, np.bool_)):
                attrs[k] = int(v)
    return ds

def main():
    print("--- Starting WN2 subset download ---")
    print(f"Mode: {'DOWNLOAD ENABLED' if CONFIG['PERFORM_DOWNLOAD'] else 'SAFE MODE'}")

    out_path = CONFIG["zarr_store_path"]
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isdir(out_path) and CONFIG["OVERWRITE_EXISTING_STORE"]:
        print(f"Removing existing store {out_path}")
        shutil.rmtree(out_path)

    # auth + open remote Zarr
    fs = gcsfs.GCSFileSystem(project=CONFIG["project_id"], token="google_default")
    print("Opening remote WN2 predictions.zarr ...")
    ds = xr.open_zarr(CONFIG["bucket_path"], consolidated=True)

    # ---- subset variables & prediction_timedelta ----
    step_dim = "prediction_timedelta" if "prediction_timedelta" in ds.dims else "step"
    steps = [np.timedelta64(h, "h") for h in CONFIG["steps_hours"]]

    subset = ds[CONFIG["variables"]].sel({step_dim: steps})

    # ---- time filter ----
    time = subset["time"].to_index()
    mask = (
        (time >= pd.to_datetime(CONFIG["start_date"])) &
        (time <= pd.to_datetime(CONFIG["end_date"])) &
        (time.hour == CONFIG["init_hour"])
    )
    target_times = time[mask]
    print(f"Found {len(target_times)} valid init times at {CONFIG['init_hour']}Z")

    if not CONFIG["PERFORM_DOWNLOAD"]:
        print("SAFE MODE: stopping before writing.")
        return

    # batching by time
    first_write = True
    batch_size = CONFIG["batch_days"]
    times_list = list(target_times)

    for i in range(0, len(times_list), batch_size):
        batch_times = times_list[i:i + batch_size]
        print(f"Batch {i//batch_size + 1}: {len(batch_times)} init times")

        sub_batch = subset.sel(time=batch_times)
        sub_batch = clean_attributes(sub_batch)
        sub_batch = sub_batch.chunk(CONFIG["chunk_plan"])
        
        # Clear encoding to avoid chunk conflicts
        for var in sub_batch.data_vars:
            if 'chunks' in sub_batch[var].encoding:
                del sub_batch[var].encoding['chunks']

        mode = "w" if first_write else "a"
        append_dim = None if first_write else "time"

        sub_batch.to_zarr(
            out_path,
            mode=mode,
            append_dim=append_dim,
            consolidated=True,
            zarr_format=2,
            safe_chunks=False,
            compute=True,
        )
        first_write = False

    print("Job complete.")
    if os.path.exists(out_path):
        xr.open_zarr(out_path, consolidated=True)
        print("Verification: open_zarr succeeded.")

if __name__ == "__main__":
    main()