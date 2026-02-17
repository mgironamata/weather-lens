import xarray as xr
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot twCRPS and Skill from processed NC files")
    parser.add_argument("--var", type=str, default="t2m", choices=["t2m", "winds", "tp"], help="Variable name")
    parser.add_argument("--lead", type=int, default=24, help="Lead time in hours")
    parser.add_argument("--mask", type=str, default="land", help="Mask type")
    parser.add_argument("--model", type=str, required=True, choices=["wn2", "gencast", "WN2", "GenCast"], help="AI model to plot")
    
    args = parser.parse_args()
    
    # Configuration
    model_name = args.model.upper() if args.model.lower() == "wn2" else "GenCast"
    color = "orange" if model_name == "WN2" else "tab:blue"
    
    data_dir = Path("/scratch2/mg963/results/diagnostics")
    mask_suffix = f"_{args.mask}" if args.mask != "all" else ""
    
    # Locate file
    # Try both specific and general filenames
    potential_files = [
        data_dir / f"twcrps_{args.var}_{args.lead}h_thresholds{mask_suffix}.nc",
        data_dir / f"twcrps_{args.var}_{args.lead}h_thresholds{mask_suffix}_{args.model.lower()}.nc"
    ]
    
    ds = None
    for pf in potential_files:
        if pf.exists():
            print(f"📖 Loading {pf}...")
            ds = xr.open_dataset(pf)
            break
            
    if ds is None:
        print(f"❌ Error: Could not find data file for {args.var} {args.lead}h {args.mask}")
        return

    # Identify coordinates/attributes
    display_units = ds.attrs.get("display_units", "")
    var_long_name = ds.attrs.get("variable_long_name", args.var)
    
    # Clean the units string to match coordinate naming convention
    units_clean = display_units.replace('°', 'deg').replace('/', '_per_')
    
    threshold_coord = None
    # Prefer the coordinate that matches display units
    target_coord = f"threshold_{units_clean}"
    if target_coord in ds.coords:
        threshold_coord = target_coord
    else:
        # Fallback to any threshold_ coordinate
        for coord in ds.coords:
            if coord.startswith("threshold_"):
                threshold_coord = coord
                break
    
    if threshold_coord is None:
        threshold_coord = "threshold"

    x_vals = ds[threshold_coord].values
    
    # 1. Plot twCRPS
    plt.figure(figsize=(8, 6))
    
    # Plot IFS-ENS baseline
    if "twcrps_IFS-ENS" in ds:
        plt.plot(x_vals, ds["twcrps_IFS-ENS"], marker="o", color="black", label="IFS-ENS", linestyle="--")
    
    # Plot AI Model
    var_name = f"twcrps_{model_name}"
    if var_name in ds:
        plt.plot(x_vals, ds[var_name], marker="s", color=color, label=model_name)
    else:
        print(f"⚠️ Warning: {var_name} not found in dataset. Available: {list(ds.data_vars)}")

    plt.xlabel(f"Threshold ({display_units})")
    plt.ylabel(f"Threshold-weighted CRPS ({ds.attrs.get('crps_units', '')})")
    plt.title(f"{args.lead}h {var_long_name} twCRPS ({args.mask.capitalize()})\n{model_name} vs IFS-ENS")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_fig = data_dir / f"plot_twcrps_{args.var}_{args.lead}h_{args.mask}_{args.model.lower()}.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"📊 Saved twCRPS plot to {out_fig}")

    # 2. Plot Skill
    skill_var = f"skill_vs_ifs_{model_name}"
    if skill_var in ds:
        plt.figure(figsize=(8, 6))
        plt.axhline(0, color="black", linewidth=1, alpha=0.5)
        plt.plot(x_vals, ds[skill_var], marker="o", color=color, label=model_name)
        
        plt.xlabel(f"Threshold ({display_units})")
        plt.ylabel("Skill Score (1 - twCRPS_model / twCRPS_IFS)")
        plt.title(f"{args.lead}h {var_long_name} twCRPS Skill vs IFS-ENS ({args.mask.capitalize()})\n{model_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        out_skill = data_dir / f"plot_twcrps_skill_{args.var}_{args.lead}h_{args.mask}_{args.model.lower()}.png"
        plt.savefig(out_skill, dpi=150, bbox_inches="tight")
        print(f"📊 Saved Skill plot to {out_skill}")
    else:
        print(f"⚠️ Warning: {skill_var} not found in dataset.")

if __name__ == "__main__":
    main()
