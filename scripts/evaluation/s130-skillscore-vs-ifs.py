"""Compute simple CRPS skill score vs IFS.

Skill score definition (vs reference):
  skill = 1 - CRPS_model / CRPS_ref

This script computes global-mean CRPS over (time, lat, lon) from the already-
computed CRPS Zarr outputs and reports skill scores for GenCast and WN2 vs IFS.

Usage:
  python s130-skillscore-vs-ifs.py

Optional:
  python s130-skillscore-vs-ifs.py --out /scratch2/mg963/results/diagnostics/skill_tp_24h_vs_ifs.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


DEFAULT_INPUTS = {
    "wn2": Path("/scratch2/mg963/results/weathernext/wn2_vs_ERA5/crps_tp_24h.zarr"),
    "gencast": Path("/scratch2/mg963/results/weathernext/gencast_vs_ERA5/crps_tp_24h.zarr"),
    "ifs": Path("/scratch2/mg963/results/ecmwf/ifs_vs_ERA5/crps_tp_24h.zarr"),
}


def open_zarr_any(path: Path) -> xr.Dataset:
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:
        return xr.open_zarr(path, consolidated=False)


def mean_crps(path: Path) -> tuple[str, float]:
    ds = open_zarr_any(path)
    if len(ds.data_vars) != 1:
        var = list(ds.data_vars)[0]
    else:
        var = list(ds.data_vars)[0]
    da = ds[var].astype("float64")
    return var, float(da.mean().compute())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional NetCDF output path for the summary table",
    )
    args = parser.parse_args()

    vals: dict[str, float] = {}
    vars_used: dict[str, str] = {}

    for name, path in DEFAULT_INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing CRPS output for {name}: {path}")
        var, v = mean_crps(path)
        vars_used[name] = var
        vals[name] = v

    crps_ifs = vals["ifs"]
    skill_gencast = 1.0 - (vals["gencast"] / crps_ifs)
    skill_wn2 = 1.0 - (vals["wn2"] / crps_ifs)

    print("Mean CRPS (24h):")
    print(f"  IFS:     {vals['ifs']:.8g} (var={vars_used['ifs']})")
    print(f"  GenCast: {vals['gencast']:.8g} (var={vars_used['gencast']})")
    print(f"  WN2:     {vals['wn2']:.8g} (var={vars_used['wn2']})")

    print("\nSkill vs IFS (higher is better):")
    print(f"  GenCast vs IFS: {skill_gencast:.6g}")
    print(f"  WN2 vs IFS:     {skill_wn2:.6g}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        models = np.array(["ifs", "gencast", "wn2"], dtype=object)
        crps = np.array([vals[m] for m in models], dtype=float)
        skill_vs_ifs = np.array(
            [np.nan, skill_gencast, skill_wn2], dtype=float
        )

        ds = xr.Dataset(
            data_vars={
                "mean_crps": ("model", crps),
                "skill_vs_ifs": ("model", skill_vs_ifs),
            },
            coords={"model": models},
            attrs={
                "definition": "skill_vs_ifs = 1 - mean_crps(model)/mean_crps(ifs)",
                "lead_hours": 24,
                "variable": "tp",
                "reference_model": "ifs",
            },
        )

        ds.to_netcdf(out_path)
        print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
