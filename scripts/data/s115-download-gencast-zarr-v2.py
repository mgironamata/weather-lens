import xarray as xr
import gcsfs
import pandas as pd
import os
import numpy as np
import shutil

# ==============================================================================
# 1. CONFIGURATION PARAMETERS
# ==============================================================================
CONFIG = {
    # --- SAFETY FLAGS ---
    "PERFORM_DOWNLOAD": True,           # If False: just check existence on GCS
    "OVERWRITE_EXISTING_STORE": False,  # True = delete & recreate, False = append

    # Google Cloud Project ID
    "project_id": "weathernextforecasts",

    # Input Bucket Base Path
    "bucket_base": "gs://weathernext/126478713_1_0/zarr/126478713_2024_to_present",

    # Output Zarr path (this will be Zarr v2)
    "zarr_store_path": "/scratch2/mg963/data/weathernext/gencast/gencast_2024_precip_12_36_48_60.zarr",

    # Filter Settings
    "start_date": "2024-01-01",  # <-- adjust for each run
    "end_date":   "2024-12-31",
    "init_hour":  6,  # 06:00 UTC

    # Variables
    "variables": [
        "total_precipitation_12hr",
        # "2m_temperature",
        # "10m_u_component_of_wind",
        # "10m_v_component_of_wind",
    ],

    # Forecast Steps (INTEGERS, in hours) up to 72h forecasts without 24 and 72 steps
    "steps": [12, 36, 48, 60],

    # Chunking Strategy (for final Zarr; good for CRPS over time & sample)
    # Roughly: 8 time steps × 64 samples × 90×180 tile ≈ 33 MB per var
    "chunk_plan": {
        "time": 8,      # multi-step chunks → efficient for CRPS over time
        "sample": -1,   # keep all ensemble members in one chunk
        "lat": 90,
        "lon": 180,
    },

    # Number of days to download and write in one batch
    "batch_days": 16,
}

# ==============================================================================
# HELPER: Sanitize Attributes
# ==============================================================================
def clean_attributes(ds: xr.Dataset) -> xr.Dataset:
    """Converts boolean attributes to integers to prevent NetCDF/Zarr errors."""
    for key, val in list(ds.attrs.items()):
        if isinstance(val, (bool, np.bool_)):
            ds.attrs[key] = int(val)

    for var in ds.variables:
        attrs = ds[var].attrs
        for key, val in list(attrs.items()):
            if isinstance(val, (bool, np.bool_)):
                attrs[key] = int(val)

    return ds

# ==============================================================================
# HELPER: Process a single day's Zarr into a subset Dataset
# ==============================================================================
def process_one_day(remote_zarr_path: str, init_time: pd.Timestamp) -> xr.Dataset:
    """
    Open a single daily GenCast predictions.zarr, subset required vars/steps,
    add 'time' dimension, clean attributes, and chunk according to CONFIG.
    """

    # IMPORTANT: disable timedelta decoding to keep 'time'/'step' numeric if possible
    ds = xr.open_zarr(
        remote_zarr_path,
        consolidated=True,
        decode_timedelta=False,
    )

    # Some GenCast layouts use 'time' where we actually want 'step'
    if "time" in ds.dims and "step" not in ds.dims:
        ds = ds.rename({"time": "step"})

    if "step" not in ds.dims:
        raise KeyError("No 'step' dimension found in dataset (after optional rename).")

    # ---- Robust handling of step coordinate ----
    step_coord = ds["step"]

    # If it's a timedelta, convert to integer hours
    if np.issubdtype(step_coord.dtype, np.timedelta64):
        step_hours = (step_coord / np.timedelta64(1, "h")).astype("int64")
        ds = ds.assign_coords(step=step_hours)
        step_coord = ds["step"]

    # If it's not numeric (e.g. strings), try a best-effort conversion to hours
    if not np.issubdtype(step_coord.dtype, np.number):
        try:
            step_values = pd.to_timedelta(step_coord.values)
            step_hours = (step_values / np.timedelta64(1, "h")).astype("int64")
            ds = ds.assign_coords(step=step_hours)
            step_coord = ds["step"]
        except Exception as e:
            raise KeyError(
                f"Cannot convert 'step' coordinate to numeric hours. dtype={step_coord.dtype}, "
                f"example values={step_coord.values[:5]}"
            ) from e

    # Now step_coord should be numeric hours; we can safely select CONFIG['steps']
    missing_steps = [s for s in CONFIG["steps"] if s not in step_coord.values]
    if missing_steps:
        raise KeyError(
            f"Not all requested steps found. Missing: {missing_steps}. "
            f"Available 'step' values (first 20): {step_coord.values[:20]}"
        )

    # Select variables and forecast steps
    subset = ds[CONFIG["variables"]]
    subset = subset.sel(step=CONFIG["steps"])

    # Add initialization time dimension
    subset = subset.expand_dims(time=[np.datetime64(init_time)])

    # Clean attributes & apply desired chunking
    subset = clean_attributes(subset)
    subset = subset.chunk(CONFIG["chunk_plan"])

    return subset

# ==============================================================================
# 2. MAIN SCRIPT
# ==============================================================================
def main():
    print("--- Starting GenCast Download Job ---")
    print(f"Mode: {'DOWNLOAD ENABLED' if CONFIG['PERFORM_DOWNLOAD'] else 'SAFE MODE'}")

    zarr_path = CONFIG["zarr_store_path"]
    zarr_dir = os.path.dirname(zarr_path)

    # Ensure output directory exists
    try:
        os.makedirs(zarr_dir, exist_ok=True)
        print(f"Output directory confirmed: {zarr_dir}")
    except Exception as e:
        print(f"\nCRITICAL ERROR: Could not create output directory {zarr_dir}")
        print(f"System reported: {e}")
        return

    # Handle overwrite / append logic
    first_write = True

    if os.path.isdir(zarr_path):
        if CONFIG["OVERWRITE_EXISTING_STORE"]:
            print(f"Existing directory at {zarr_path} found; removing it to start fresh.")
            shutil.rmtree(zarr_path)
            first_write = True
        else:
            # Try to open existing store; if it fails, tell user to enable overwrite
            try:
                _ = xr.open_zarr(zarr_path)
                first_write = False
                print("Existing Zarr store detected and readable; will append along 'time'.")
            except Exception as e:
                print(
                    f"Existing path found at {zarr_path} but could not be read as Zarr.\n"
                    f"Error: {e}\n"
                    "Set CONFIG['OVERWRITE_EXISTING_STORE'] = True if you want to recreate it."
                )
                return

    # Auth check
    try:
        fs = gcsfs.GCSFileSystem(project=CONFIG["project_id"], token="google_default")
        print("GCS authentication successful.")
    except Exception as e:
        print(f"Auth Error: {e}")
        return

    # Generate target dates
    target_dates = pd.date_range(
        start=CONFIG["start_date"],
        end=CONFIG["end_date"],
        freq="D",
    )
    n_dates = len(target_dates)
    print(f"Targeting {n_dates} daily forecasts at {CONFIG['init_hour']}Z.")

    batch = []
    batch_size = CONFIG["batch_days"]

    def flush_batch(batch_ds_list):
        nonlocal first_write
        if not batch_ds_list:
            return

        print(f"Flushing batch of {len(batch_ds_list)} day(s) to Zarr...")
        combined = xr.concat(batch_ds_list, dim="time")

        if first_write:
            mode = "w"
            append_dim = None
        else:
            mode = "a"
            append_dim = "time"

        combined.to_zarr(
            zarr_path,
            mode=mode,
            append_dim=append_dim,
            consolidated=True,
            zarr_format=2,  # force pure Zarr v2
        )

        first_write = False
        print("Batch write complete.")

    # --- DOWNLOAD + PROCESS LOOP ---
    for i, date in enumerate(target_dates):
        date_str = date.strftime("%Y%m%d")
        folder_name = f"{date_str}_{CONFIG['init_hour']:02d}hr_01_preds"
        remote_zarr_path = f"{CONFIG['bucket_base']}/{folder_name}/predictions.zarr"
        init_time_val = date + pd.Timedelta(hours=CONFIG["init_hour"])

        print(f"[{i+1}/{n_dates}] {folder_name}", end=" ", flush=True)

        # Safe mode: just check existence and skip
        if not CONFIG["PERFORM_DOWNLOAD"]:
            exists = fs.exists(remote_zarr_path)
            print("FOUND (Safe Mode)." if exists else "NOT FOUND (Safe Mode, skipping).")
            continue

        # Download / process mode
        try:
            subset = process_one_day(remote_zarr_path, init_time_val)
        except FileNotFoundError:
            print("Not found (skipping).")
            continue
        except KeyError as e:
            print(f"\nVariable/step Error on {date_str}: {e}")
            break
        except Exception as e:
            print(f"\nFAILED on {date_str}: {e}")
            continue

        batch.append(subset)
        print("OK (queued).")

        # Flush batch if full
        if len(batch) >= batch_size:
            flush_batch(batch)
            batch = []

    # Flush any remaining days
    if batch:
        flush_batch(batch)

    print("\nJob Complete.")

    # Optional sanity check
    if os.path.exists(zarr_path):
        try:
            ds_test = xr.open_zarr(zarr_path)
            print("Verification successful: Zarr store can be opened.")
            print(ds_test)
        except Exception as e:
            print(f"Verification failed: {e}")

if __name__ == "__main__":
    main()