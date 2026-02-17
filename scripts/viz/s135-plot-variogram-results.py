import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path

# -----------------------------------------
# CONFIG
# -----------------------------------------
RESULTS_DIR = Path("/scratch2/mg963/results/diagnostics/variogram")
PLOT_DIR = Path("/scratch2/mg963/results/plots/variogram")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "WN2": "#1f77b4",      # Blue
    "GenCast": "#ff7f0e",  # Orange
    "IFS-ENS": "#2ca02c"   # Green
}

VAR_NAMES = {
    "t2m": "2m Temperature (K)",
    "winds": "10m Wind Speed (m/s)",
    "tp": "Total Precipitation (m)"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", type=str, default="t2m", choices=["t2m", "winds", "tp"])
    parser.add_argument("--lead", type=int, default=24)
    parser.add_argument("--smooth", type=int, default=7, help="Days for rolling mean smoothing")
    parser.add_argument("--mask-type", type=str, default="all", choices=["all", "land", "ocean"], help="Mask type")
    args = parser.parse_args()

    variable = args.var
    lead = args.lead
    mask_suffix = f"_{args.mask_type}" if args.mask_type != "all" else ""
    
    file_path = RESULTS_DIR / f"vs_{variable}_{lead}h{mask_suffix}.nc"
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    print(f"📈 Plotting Variogram Score for {variable} @ {lead}h (Mask: {args.mask_type})")
    ds = xr.open_dataset(file_path)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(hspace=0.3)

    # 1. Time Series Plot
    for model in ds.data_vars:
        color = MODEL_COLORS.get(model, None)
        
        # Raw data (faded)
        ax1.plot(ds.time, ds[model], color=color, alpha=0.2, linewidth=0.5)
        
        # Smoothed data
        smoothed = ds[model].to_series().rolling(window=args.smooth, center=True).mean()
        ax1.plot(ds.time, smoothed, color=color, label=f"{model} ({args.smooth}d mean)", linewidth=2)

    ax1.set_title(f"Variogram Score Time Series: {VAR_NAMES[variable]} @ {lead}h Lead (Mask: {args.mask_type})", fontsize=14)
    ax1.set_ylabel("Variogram Score (lower is better)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Bar Chart (Mean Performance)
    means = ds.mean(dim="time").to_pandas()
    # Reorder to match MODEL_COLORS if possible
    models_present = [m for m in MODEL_COLORS if m in means.index] + [m for m in means.index if m not in MODEL_COLORS]
    mean_values = [means[m] for m in models_present]
    colors = [MODEL_COLORS.get(m, "gray") for m in models_present]

    bars = ax2.bar(models_present, mean_values, color=colors, alpha=0.8)
    ax2.set_title(f"Mean Variogram Score (Period: {pd.to_datetime(ds.time.values[0]).date()} to {pd.to_datetime(ds.time.values[-1]).date()})", fontsize=12)
    ax2.set_ylabel("Mean VS")
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    out_path = PLOT_DIR / f"vs_plot_{variable}_{lead}h{mask_suffix}.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"✅ Plot saved to {out_path}")

if __name__ == "__main__":
    main()

# example: python s135-plot-variogram-results.py --var t2m --lead 24