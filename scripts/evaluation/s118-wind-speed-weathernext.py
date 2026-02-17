#!/usr/bin/env python
import xarray as xr
import numpy as np
import dask
from pathlib import Path
import os
import shutil

# ---------------------------------------------------------------------
# DASK CONFIG
# ---------------------------------------------------------------------
dask.config.set(scheduler="threads", num_workers=8)

# ---------------------------------------------------------------------
# MODEL-SPECIFIC PATHS / VAR NAMES
# ---------------------------------------------------------------------
MODEL_SPECS = {
    "WN2": {
        "input_path":  "/scratch2/mg963/data/weathernext/wn2/wn2_2024_global.zarr",
        "u_var":       "10m_u_component_of_wind",
        "v_var":       "10m_v_component_of_wind",
        "ws_name":     "10m_wind_speed",
        "output_path": "/scratch2/mg963/data/weathernext/wn2/wn2_2024_10m_wind_speed.zarr",
    },
    "gencast": {
        "input_path":  "/scratch2/mg963/data/weathernext/gencast/gencast_2024_global_optchunks.zarr",
        "u_var":       "10m_u_component_of_wind",
        "v_var":       "10m_v_component_of_wind",
        "ws_name":     "10m_wind_speed",
        "output_path": "/scratch2/mg963/data/weathernext/gencast/gencast_2024_10m_wind_speed.zarr",
    },
}

OVERWRITE = False  # set True to delete an existing wind_speed store

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def chunks_dict_like(da: xr.DataArray):
    """Convert da.chunks (tuple-of-tuples) into a dict dim -> chunk_sizes."""
    if da.chunks is None:
        return None
    return {dim: ch for dim, ch in zip(da.dims, da.chunks)}


def compute_wind_speed(u_da: xr.DataArray, v_da: xr.DataArray, name: str) -> xr.DataArray:
    """
    Compute 10m wind speed = hypot(u, v) lazily with dask.
    Returns DataArray with same dims/coords/chunks as u_da.
    """
    # Ensure same chunks
    if u_da.chunks != v_da.chunks:
        v_da = v_da.chunk(chunks_dict_like(u_da))

    ws = xr.apply_ufunc(
        np.hypot,
        u_da,
        v_da,
        dask="parallelized",
        output_dtypes=[u_da.dtype],
    )
    ws.name = name

    # Copy attrs and set some sensible metadata
    attrs = dict(u_da.attrs)
    attrs.setdefault("standard_name", "wind_speed")
    attrs.setdefault("long_name", "10 metre wind speed")
    if "units" in u_da.attrs:
        attrs["units"] = u_da.attrs["units"]
    ws.attrs = attrs

    # Match chunking
    cd = chunks_dict_like(u_da)
    if cd is not None:
        ws = ws.chunk(cd)

    return ws

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main(selected_model: str):
    if selected_model not in MODEL_SPECS:
        raise ValueError(f"Unknown model {selected_model!r}. Choose one of {list(MODEL_SPECS.keys())}")

    spec = MODEL_SPECS[selected_model]
    in_path = spec["input_path"]
    out_path = spec["output_path"]

    print(f"=== Wind-speed builder for {selected_model} ===")
    print(f"Input Zarr : {in_path}")
    print(f"Output Zarr: {out_path}")

    # Overwrite logic
    if os.path.isdir(out_path):
        if OVERWRITE:
            print("Output Zarr exists; removing it (OVERWRITE=True).")
            shutil.rmtree(out_path)
        else:
            print("Output Zarr already exists and OVERWRITE=False. Aborting.")
            return

    print("Opening input Zarr...")
    ds = xr.open_zarr(in_path, consolidated=True)

    u_var = spec["u_var"]
    v_var = spec["v_var"]
    ws_name = spec["ws_name"]

    if u_var not in ds.data_vars or v_var not in ds.data_vars:
        raise KeyError(
            f"Could not find required variables '{u_var}' and/or '{v_var}' "
            f"in {in_path}. Available: {list(ds.data_vars)}"
        )

    u = ds[u_var]
    v = ds[v_var]

    print(f"Found u10: {u_var} with dims {u.dims} and chunks {u.chunks}")
    print(f"Found v10: {v_var} with dims {v.dims} and chunks {v.chunks}")

    print("Computing 10m wind speed (lazy, dask graph)...")
    ws = compute_wind_speed(u, v, ws_name)

    ds_out = ws.to_dataset()

    #  Strip any inherited compressor/filters encoding to avoid v3 codec issues
    for name, var in ds_out.variables.items():
        var.encoding.clear()

    print("Writing wind speed to new Zarr store (v3 default)...")
    ds_out.to_zarr(
        out_path,
        mode="w",
        consolidated=True,   # OK with v3
        # no zarr_format argument → use default (v3)
    )

    print("Verifying output...")
    ds_check = xr.open_zarr(out_path, consolidated=True)
    print(ds_check)
    print("✅ Done.")


if __name__ == "__main__":
    import time
    start = time.time()
    SELECTED_MODEL = "gencast"   # or "gencast"
    main(SELECTED_MODEL)
    elapsed = time.time() - start
    print(f"Total elapsed time: {elapsed:.1f} seconds")