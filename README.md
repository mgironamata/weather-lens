# weather-lens
Weather forecasting model evaluation and diagnostics.

This repository focuses on evaluating atmospheric model performance using metrics such as CRPS (Continuous Ranked Probability Score) and its variants (Fair CRPS, Threshold-Weighted CRPS). It includes scripts for data acquisition, processing, scoring, and visualization for models like AIFS, IFS, WeatherNext (WN2), and GenCast.

## Repository Structure

- `notebooks/`: Interactive analysis and workflow developments.
    - `workflows/`: Pipeline walkthroughs (e.g., data download, regridding).
    - `analysis/`: High-level result summaries and comparisons.
- `scripts/`: Modular Python/Bash scripts categorized by functional stage.
    - `data/`: Downloading from Herbie (AIFS/IFS), GCS (WN2), and ERA5.
    - `processing/`: GRIB-to-Zarr conversion, alignment, and regridding.
    - `evaluation/`: Scientific metrics implementation (CRPS, Variogram, etc.).
    - `viz/`: Plotting utilities for skill scores and diagnostics.
    - `batch/`: Automation scripts for large-scale HPC runs.
- `src/weather_lens/`: Reusable Python modules and library code.
- `results/`: Output artifacts (metrics and figures).
- `models/`: Trained model weights (e.g., hail prediction models).

## Getting Started

1.  **Environment Setup**: Ensure you have the necessary meteorological libraries (`herbie`, `xarray`, `dask`, `cfgrib`).
2.  **Download Data**: Use scripts in `scripts/data/` to mirror required forecast and observational data.
3.  **Run Evaluation**: Use `scripts/evaluation/s117-twcrps.py` for standard threshold-weighted scoring.
4.  **Visualize Results**: Use `scripts/viz/` to generate skill-score plots.
