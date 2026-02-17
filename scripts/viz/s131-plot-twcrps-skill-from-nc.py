"""Plot threshold-weighted CRPS skill vs IFS from an existing NetCDF.

This script does NOT recompute CRPS or twCRPS. It simply reads the NetCDF
produced by `s117-twcrps.py` and plots skill curves:

    skill = 1 - twCRPS_model / twCRPS_IFS

Outputs a PNG figure next to the NetCDF (or at a configured path).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


CONFIG = {
    # NetCDF created by s117-twcrps.py
    "twcrps_nc": "/scratch2/mg963/results/diagnostics/twcrps_tp_24h_thresholds.nc",
    # Output PNG
    "out_png": "/scratch2/mg963/results/diagnostics/twcrps_skill_vs_ifs_tp_24h_thresholds_from_nc.png",
    # Which models to plot skill for (must exist in the NetCDF)
    "models": ["GenCast", "WN2"],
    # Baseline model for skill
    "baseline": "IFS-ENS",
}


def _find_threshold_coords(ds: xr.Dataset) -> tuple[str, str | None]:
    """Return (threshold_native_coord, threshold_display_coord or None)."""
    # Prefer the precipitation naming we produce: threshold_m and threshold_mm
    if "threshold_m" in ds.coords:
        return "threshold_m", "threshold_mm" if "threshold_mm" in ds.coords else None

    # Otherwise, try generic patterns
    threshold_coords = [c for c in ds.coords if c.startswith("threshold_")]
    if not threshold_coords:
        raise ValueError(f"No threshold coordinate found in dataset. Coords: {list(ds.coords)}")

    # Pick a native coord (shortest name often native), and prefer mm-like as display
    threshold_display = None
    for c in threshold_coords:
        if "mm" in c.lower() or "deg" in c.lower():
            threshold_display = c
            break

    # Pick a native coord that is not the display coord
    native_candidates = [c for c in threshold_coords if c != threshold_display]
    threshold_native = native_candidates[0] if native_candidates else threshold_coords[0]

    return threshold_native, threshold_display


def _get_var(ds: xr.Dataset, prefix: str, name: str) -> str:
    """Find a variable like f"{prefix}_{name}" or fall back to contains match."""
    direct = f"{prefix}_{name}"
    if direct in ds.data_vars:
        return direct

    # Fallback: allow minor naming differences (e.g., hyphen/underscore)
    normalized_target = direct.replace("-", "_")
    for v in ds.data_vars:
        if v.replace("-", "_") == normalized_target:
            return v

    # Fallback: contains
    for v in ds.data_vars:
        if v.startswith(prefix + "_") and name.replace("-", "").lower() in v.replace("-", "").lower():
            return v

    raise KeyError(f"Could not find variable for {direct}. Available: {list(ds.data_vars)}")


def main() -> None:
    nc_path = Path(CONFIG["twcrps_nc"]).expanduser()
    out_png = Path(CONFIG["out_png"]).expanduser()

    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF not found: {nc_path}")

    ds = xr.open_dataset(nc_path)

    thr_native, thr_display = _find_threshold_coords(ds)
    x = ds[thr_display] if thr_display is not None else ds[thr_native]

    baseline = CONFIG["baseline"]
    v_ifs = _get_var(ds, "twcrps", baseline)
    ifs_vals = ds[v_ifs].astype("float64").values

    # Compute skill curves
    skills: dict[str, np.ndarray] = {}
    for model in CONFIG["models"]:
        v_model = _get_var(ds, "twcrps", model)
        vals = ds[v_model].astype("float64").values
        with np.errstate(divide="ignore", invalid="ignore"):
            skill = 1.0 - (vals / ifs_vals)
        skills[model] = skill

    # Plot
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to generate the plot in this script. "
            "Install it in your current environment and rerun."
        ) from e

    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    for model, skill in skills.items():
        plt.plot(x.values, skill, marker="o", label=model)

    plt.axhline(0.0, color="k", linewidth=1, alpha=0.4)
    plt.xlabel(x.name)
    plt.ylabel("Skill vs IFS-ENS (1 - twCRPS / twCRPS_IFS)")
    title_var = ds.attrs.get("variable_long_name", ds.attrs.get("variable", ""))
    lead = ds.attrs.get("lead_hours", "")
    plt.title(f"{lead}h {title_var} skill vs {baseline}\n(GenCast, WN2)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
