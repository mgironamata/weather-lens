#!/usr/bin/env python
"""
Build 10m wind-speed Zarr files for IFS ENS

Inputs (existing):
    /scratch2/mg963/data/ecmwf/ensembles/ifs/zarr_ens/u10_f24.zarr
    /scratch2/mg963/data/ecmwf/ensembles/ifs/zarr_ens/v10_f24.zarr
    /scratch2/mg963/data/ecmwf/ensembles/ifs/zarr_ens/u10_f72.zarr
    /scratch2/mg963/data/ecmwf/ensembles/ifs/zarr_ens/v10_f72.zarr

Outputs (new):
    .../wspd10_f24.zarr
    .../wspd10_f72.zarr
"""

import xarray as xr
import numpy as np
import dask
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BASE_DIR = Path("/scratch2/mg963/data/ecmwf/ensembles/ifs/zarr_ens")
LEADS = [24, 72]              # hours
OUT_VAR_NAME = "wspd10"       # name for wind-speed variable in output

# conservative threading
dask.config.set(scheduler="threads", num_workers=8)


def _pick_var(ds, preferred_names):
    """Pick a data variable by trying preferred_names then falling back to single var."""
    candidates = list(ds.data_vars)
    for name in preferred_names:
        if name in candidates:
            return name
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Could not identify variable in dataset. Candidates: {candidates}")


def build_wind_speed_for_lead(lead: int):
    """Compute wind speed for a single lead (e.g. 24 or 72)."""

    u_path = BASE_DIR / f"u10_f{lead}.zarr"
    v_path = BASE_DIR / f"v10_f{lead}.zarr"
    out_path = BASE_DIR / f"{OUT_VAR_NAME}_f{lead}.zarr"

    print("=" * 80)
    print(f"Lead {lead}h")
    print(f"   u10 Zarr : {u_path}")
    print(f"   v10 Zarr : {v_path}")
    print(f"   OUT Zarr : {out_path}")

    if not u_path.is_dir():
        raise FileNotFoundError(f"u10 file not found: {u_path}")
    if not v_path.is_dir():
        raise FileNotFoundError(f"v10 file not found: {v_path}")

    # Open lazily
    ds_u = xr.open_zarr(u_path, consolidated=True)
    ds_v = xr.open_zarr(v_path, consolidated=True)

    # Identify variable names (usually 'u10' and 'v10')
    var_u = _pick_var(ds_u, ["u10", "10u"])
    var_v = _pick_var(ds_v, ["v10", "10v"])

    u = ds_u[var_u]
    v = ds_v[var_v]

    print(f"   Found u10 var: {var_u} with dims {u.dims} and chunks {u.chunks}")
    print(f"   Found v10 var: {var_v} with dims {v.dims} and chunks {v.chunks}")

    # Basic sanity checks
    if u.dims != v.dims:
        raise ValueError(f"Dimension mismatch between u10 {u.dims} and v10 {v.dims}")
    for dim in u.dims:
        if u.sizes[dim] != v.sizes[dim]:
            raise ValueError(f"Size mismatch along dim '{dim}': u={u.sizes[dim]}, v={v.sizes[dim]}")

    # Ensure consistent chunking (they should already match, but be explicit)
    v = v.chunk(u.chunks)

    # Compute wind speed lazily
    print("   Computing 10m wind speed (lazy, dask graph)...")
    ws = np.sqrt(u ** 2 + v ** 2)
    ws = ws.chunk(u.chunks)  # keep same chunking as inputs
    ws.name = OUT_VAR_NAME

    # Copy units/attrs
    ws.attrs.update({
        "long_name": "10 metre wind speed",
        "units": u.attrs.get("units", "m s-1"),
        "description": f"sqrt({var_u}**2 + {var_v}**2)",
    })

    # Wrap into dataset and write
    ds_out = ws.to_dataset()
    print("   Writing wind speed to new Zarr store (v2)...")
    ds_out.to_zarr(
        str(out_path),
        mode="w",            # overwrite if exists
        consolidated=True,
    )
    print(f"   ✅ Done lead {lead}h -> {out_path}")


def main():
    for lead in LEADS:
        build_wind_speed_for_lead(lead)
    print("\nAll leads processed.")


if __name__ == "__main__":
    main()