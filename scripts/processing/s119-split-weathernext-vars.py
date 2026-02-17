#!/usr/bin/env python
import xarray as xr
import dask
from pathlib import Path
import os
import shutil

# -------------------------------------------
# Dask configuration (tune if you like)
# -------------------------------------------
dask.config.set(scheduler="threads", num_workers=8)

# -------------------------------------------
# Model-specific config
# -------------------------------------------
BASE = Path("/scratch2/mg963/data/weathernext")

MODEL_SPECS = {
    "WN2": {
        "input_path": BASE / "wn2" / "wn2_2024_global.zarr",
        "out_prefix": BASE / "wn2" / "wn2_2024",
        # Variable names in the big WN2 store
        "var_t2m": "2m_temperature",
        "var_tp": "total_precipitation_6hr",  # <- matches the dataset you printed
        "var_u10": "10m_u_component_of_wind",
        "var_v10": "10m_v_component_of_wind",
    },
    "gencast": {
        "input_path": BASE / "gencast" / "gencast_2024_global_optchunks.zarr",
        "out_prefix": BASE / "gencast" / "gencast_2024",
        "var_t2m": "2m_temperature",
        "var_tp": "total_precipitation_12hr",
        "var_u10": "10m_u_component_of_wind",
        "var_v10": "10m_v_component_of_wind",
    },
}

# Overwrite flag: if True, delete existing per-var stores before writing.
OVERWRITE = True


def write_subset(ds: xr.Dataset, vars_list, out_path: Path):
    """Write a subset of variables to a new Zarr store, streaming with dask."""
    vars_list = [v for v in vars_list if v in ds.data_vars]
    if not vars_list:
        print(f"   ⚠️ No requested variables found for {out_path.name}, skipping.")
        return

    print(f"   → Creating {out_path} with variables: {vars_list}")

    if out_path.is_dir():
        if OVERWRITE:
            print(f"      Existing store found; removing (OVERWRITE=True).")
            shutil.rmtree(out_path)
        else:
            print(f"      Store already exists and OVERWRITE=False, skipping.")
            return

    # Take only those vars, keep coords
    subset = ds[vars_list]        # ds[...] with list → Dataset already, no .to_dataset()

    # Clear encodings to avoid old Blosc/codec metadata issues
    for name, var in subset.variables.items():
        var.encoding.clear()

    # Write (lazy, via dask)
    subset.to_zarr(
        out_path,
        mode="w",
        consolidated=True,
    )

    # Quick sanity check
    check = xr.open_zarr(out_path, consolidated=True)
    print(f"      ✅ Wrote {out_path.name}: {list(check.data_vars)}")


def main(selected_model: str):
    if selected_model not in MODEL_SPECS:
        raise ValueError(f"Unknown model {selected_model!r}. Choose one of {list(MODEL_SPECS.keys())}")

    spec = MODEL_SPECS[selected_model]
    in_path = spec["input_path"]
    out_prefix = spec["out_prefix"]

    print(f"=== Splitting {selected_model} Zarr ===")
    print(f"Input Zarr : {in_path}")
    print(f"Output base: {out_prefix}")

    if not in_path.is_dir():
        raise FileNotFoundError(f"Input Zarr not found: {in_path}")

    print("Opening input Zarr (lazy)...")
    ds = xr.open_zarr(in_path, consolidated=True)
    print("Dataset variables:", list(ds.data_vars))

    # Build output paths
    out_t2m   = out_prefix.with_name(out_prefix.name + "_t2m.zarr")
    out_tp    = out_prefix.with_name(out_prefix.name + "_precip.zarr")
    out_winds = out_prefix.with_name(out_prefix.name + "_winds.zarr")

    # Variable names
    v_t2m = spec["var_t2m"]
    v_tp  = spec["var_tp"]
    v_u10 = spec["var_u10"]
    v_v10 = spec["var_v10"]

    # --- 2m temperature ---
    print("\n[1/3] Writing 2m temperature store...")
    write_subset(ds, [v_t2m], out_t2m)

    # --- Total precipitation ---
    print("\n[2/3] Writing total precipitation store...")
    write_subset(ds, [v_tp], out_tp)

    # --- Winds (u10 + v10 together) ---
    print("\n[3/3] Writing winds (u10, v10) store...")
    write_subset(ds, [v_u10, v_v10], out_winds)

    print("\n🎉 Done splitting.")
    print("When you're satisfied, you can safely delete the original big Zarr:")
    print(f"   rm -rf {in_path}")


if __name__ == "__main__":
    # First run for WN2, then switch to "gencast" and rerun
    SELECTED_MODEL = "gencast"
    main(SELECTED_MODEL)