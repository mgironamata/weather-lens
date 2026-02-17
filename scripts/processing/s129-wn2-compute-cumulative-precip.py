import os
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from functools import reduce
import operator

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    # Input zarr stores (WN2 incremental precipitation on a 6-hour grid)
    # One store contains only step/lead 24 and 72 (end-of-window 6-hour increments)
    "zarr_24_72": "/scratch2/mg963/data/weathernext/wn2/wn2_2024_precip.zarr",
    # The other store contains the remaining steps (6, 12, 18, 30, 36, 42, ..., 66)
    "zarr_rest": "/scratch2/mg963/data/weathernext/wn2/wn2_2024_precip_extra_steps.zarr",

    # Output zarr store for cumulative precipitation
    "output_zarr": "/scratch2/mg963/data/weathernext/wn2/wn2_2024_precip_cumulative_24_72.zarr",

    # Overwrite existing output
    "OVERWRITE_EXISTING": True,

    # Variable name in the WN2 zarrs
    "precip_var": "total_precipitation_6hr",

    # Chunking strategy for output
    "chunk_plan": {
        "time": 8,
        "sample": -1,
        "lat": 90,
        "lon": 180,
    },

    # Batch size for writing (number of time steps per batch)
    "write_batch_size": 16,
}


# ==============================================================================
# HELPERS
# ==============================================================================
def clean_attributes(ds: xr.Dataset) -> xr.Dataset:
    """Converts boolean attributes to integers to prevent NetCDF/Zarr encoding issues."""
    for key, val in list(ds.attrs.items()):
        if isinstance(val, (bool, np.bool_)):
            ds.attrs[key] = int(val)

    for var in ds.variables:
        attrs = ds[var].attrs
        for key, val in list(attrs.items()):
            if isinstance(val, (bool, np.bool_)):
                attrs[key] = int(val)

    return ds


def detect_step_dim(ds: xr.Dataset) -> str:
    # WN2 commonly uses prediction_timedelta (timedelta64), but some stores use step.
    if "prediction_timedelta" in ds.dims:
        return "prediction_timedelta"
    if "step" in ds.dims:
        return "step"
    raise ValueError("Could not find a step dimension (expected 'prediction_timedelta' or 'step').")


def hours_to_step_values(step_dim: str, hours: list[int]) -> list:
    # For WN2 prediction_timedelta, the coordinate values are timedelta64.
    if step_dim == "prediction_timedelta":
        return [np.timedelta64(int(h), "h") for h in hours]
    return [int(h) for h in hours]


def sum_dataarrays(arrays: tuple[xr.DataArray, ...]) -> xr.DataArray:
    if len(arrays) == 0:
        raise ValueError("No arrays to sum")
    return reduce(operator.add, arrays)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 80)
    print("WN2 CUMULATIVE PRECIPITATION COMPUTATION")
    print("Summing incremental 6-hour precipitation to get 24h and 72h cumulative totals")
    print("=" * 80)

    zarr_24_72 = CONFIG["zarr_24_72"]
    zarr_rest = CONFIG["zarr_rest"]
    output_zarr = CONFIG["output_zarr"]
    precip_var = CONFIG["precip_var"]

    print(f"\nInput zarr (24, 72 steps): {zarr_24_72}")
    print(f"Input zarr (other steps):  {zarr_rest}")
    print(f"Output zarr:               {output_zarr}")

    if not os.path.exists(zarr_24_72):
        raise FileNotFoundError(f"Endpoints zarr not found at {zarr_24_72}")
    if not os.path.exists(zarr_rest):
        raise FileNotFoundError(f"Rest-steps zarr not found at {zarr_rest}")

    # Handle output
    output_dir = os.path.dirname(output_zarr)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_zarr):
        if CONFIG["OVERWRITE_EXISTING"]:
            print(f"\nRemoving existing output zarr: {output_zarr}")
            shutil.rmtree(output_zarr)
        else:
            raise FileExistsError(
                f"Output zarr already exists at {output_zarr}. Set OVERWRITE_EXISTING=True to overwrite."
            )

    print("\nOpening input datasets...")
    ds_endpoints = xr.open_zarr(zarr_24_72, chunks="auto", consolidated=True)
    ds_rest = xr.open_zarr(zarr_rest, chunks="auto", consolidated=True)

    if precip_var not in ds_endpoints:
        raise ValueError(f"Variable {precip_var} not found in endpoints dataset")
    if precip_var not in ds_rest:
        raise ValueError(f"Variable {precip_var} not found in rest-steps dataset")

    step_dim_endpoints = detect_step_dim(ds_endpoints)
    step_dim_rest = detect_step_dim(ds_rest)
    if step_dim_endpoints != step_dim_rest:
        raise ValueError(
            f"Step dimension mismatch: endpoints uses '{step_dim_endpoints}', rest uses '{step_dim_rest}'."
        )
    step_dim = step_dim_endpoints

    precip_endpoints = ds_endpoints[precip_var]
    precip_rest = ds_rest[precip_var]

    # Ensure we have a common view of 'step' values. (We select using native dtype.)
    required_24h_steps = [6, 12, 18, 24]
    required_72h_steps = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]

    # ==========================================
    # COMPUTE 24HR CUMULATIVE
    # ==========================================
    print("\n" + "=" * 80)
    print("Computing 24hr cumulative precipitation...")
    print("=" * 80)

    increments_24 = []
    for h in required_24h_steps:
        step_val = hours_to_step_values(step_dim, [h])[0]
        if h in (24,):
            src = precip_endpoints
        else:
            src = precip_rest

        # Fall back if the expected source doesn't contain it
        if step_val in src[step_dim].values:
            increments_24.append(src.sel({step_dim: step_val}))
        else:
            # try the other dataset
            alt = precip_rest if src is precip_endpoints else precip_endpoints
            if step_val not in alt[step_dim].values:
                raise ValueError(f"Required step {h}h not found in either dataset")
            increments_24.append(alt.sel({step_dim: step_val}))

    aligned_24 = xr.align(*increments_24, join="inner")
    precip_24h = sum_dataarrays(aligned_24).drop_vars(step_dim, errors="ignore")
    print(f"24hr cumulative shape: {precip_24h.shape}")
    print(f"24hr time range: {precip_24h.time.values[0]} to {precip_24h.time.values[-1]}")

    # ==========================================
    # COMPUTE 72HR CUMULATIVE
    # ==========================================
    print("\n" + "=" * 80)
    print("Computing 72hr cumulative precipitation...")
    print("=" * 80)

    increments_72 = []
    for h in required_72h_steps:
        step_val = hours_to_step_values(step_dim, [h])[0]
        if h in (24, 72):
            src = precip_endpoints
        else:
            src = precip_rest

        if step_val in src[step_dim].values:
            increments_72.append(src.sel({step_dim: step_val}))
        else:
            alt = precip_rest if src is precip_endpoints else precip_endpoints
            if step_val not in alt[step_dim].values:
                raise ValueError(f"Required step {h}h not found in either dataset")
            increments_72.append(alt.sel({step_dim: step_val}))

    aligned_72 = xr.align(*increments_72, join="inner")
    precip_72h = sum_dataarrays(aligned_72).drop_vars(step_dim, errors="ignore")
    print(f"72hr cumulative shape: {precip_72h.shape}")
    print(f"72hr time range: {precip_72h.time.values[0]} to {precip_72h.time.values[-1]}")

    # ==========================================
    # CREATE OUTPUT DATASET (lead_time dimension)
    # ==========================================
    print("\n" + "=" * 80)
    print("Creating output dataset...")
    print("=" * 80)

    # Keep all times from both (NaN fill where one lead is missing)
    precip_24h, precip_72h = xr.align(precip_24h, precip_72h, join="outer")

    ds_cumulative = xr.Dataset(
        {
            "total_precipitation_cumulative": xr.concat(
                [precip_24h, precip_72h],
                dim=xr.DataArray(
                    [24, 72],
                    dims="lead_time",
                    name="lead_time",
                    attrs={"units": "hours", "long_name": "Forecast lead time"},
                ),
            )
        }
    )

    ds_cumulative["total_precipitation_cumulative"].attrs.update(
        {
            "long_name": "Cumulative precipitation",
            "description": "Total accumulated precipitation from forecast initialization to lead time",
            "units": precip_rest.attrs.get("units", precip_endpoints.attrs.get("units", "")),
            "note": "Computed by summing 6-hourly incremental precipitation",
        }
    )

    ds_cumulative = clean_attributes(ds_cumulative)

    chunk_dict = CONFIG["chunk_plan"].copy()
    chunk_dict["lead_time"] = -1
    ds_cumulative = ds_cumulative.chunk(chunk_dict)

    # ==========================================
    # WRITE TO ZARR IN BATCHES
    # ==========================================
    print(f"\nWriting cumulative precipitation to: {output_zarr}")
    print(f"Dataset:\n{ds_cumulative}")

    batch_size = CONFIG["write_batch_size"]
    n_times = len(ds_cumulative.time)
    n_batches = (n_times + batch_size - 1) // batch_size
    print(f"\nWriting in {n_batches} batches of {batch_size} time steps...")

    for i in range(0, n_times, batch_size):
        batch_num = i // batch_size + 1
        time_slice = slice(i, min(i + batch_size, n_times))
        subset = ds_cumulative.isel(time=time_slice)

        print(f"Batch {batch_num}/{n_batches}: writing times {i} to {min(i + batch_size, n_times) - 1}...")

        # Compute just this batch to keep memory bounded
        subset = subset.compute()

        if i == 0:
            subset.to_zarr(
                output_zarr,
                mode="w",
                consolidated=False,
                safe_chunks=False,
                zarr_format=2,
            )
        else:
            subset.to_zarr(
                output_zarr,
                mode="a",
                append_dim="time",
                consolidated=False,
                safe_chunks=False,
                zarr_format=2,
            )

        print(f"  ✓ Batch {batch_num} written")

    print("\nConsolidating metadata...")
    zarr.consolidate_metadata(output_zarr)

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)

    print("\nVerifying output (metadata only)...")
    ds_verify = xr.open_zarr(output_zarr, consolidated=True)
    print(ds_verify)
    print(f"\nDimensions: {dict(ds_verify.sizes)}")
    print(f"Lead times: {ds_verify.lead_time.values}")
    print(f"Time range: {ds_verify.time.values[0]} to {ds_verify.time.values[-1]}")


if __name__ == "__main__":
    main()
