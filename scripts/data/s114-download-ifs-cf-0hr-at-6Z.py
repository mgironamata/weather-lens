import pandas as pd
from herbie import Herbie
from pathlib import Path

# 1. Configuration
# ----------------
# Explicitly start at 06:00
start_date = "2024-02-03 06:00"
end_date = "2024-12-31 06:00"  
output_dir = "/scratch2/mg963/data/ecmwf/analysis/raw_06z/" # Recommended: Keep separate or rename files

# Ensure output directory exists
save_path = Path(output_dir)
save_path.mkdir(parents=True, exist_ok=True)

# freq="D" will now step: Feb 1 06:00, Feb 2 06:00, etc.
dates = pd.date_range(start=start_date, end=end_date, freq="D")

# 2. Download Loop
# ----------------
print(f"📥 Starting 06Z download to: {save_path}")

for date in dates:
    # 1. Update filename to reflect 06z
    # e.g., ifs_ens_cf_20240201_06z.grib2
    filename = f"ifs_ens_cf_{date.strftime('%Y%m%d')}_06z.grib2"
    final_path = save_path / filename

    if final_path.exists():
        print(f"⏭️  Skipping {filename} (Already exists)")
        continue

    try:
        # 3. Initialize Herbie (Herbie reads the 06:00 from the date object automatically)
        H = Herbie(
            date,
            model="ifs",
            product="enfo", # Ensemble Forecast
            fxx=0,          # Analysis/Zero-hour
            save_dir=save_path,
            priority=['azure', 'aws'],
            verbose=False
        )

        # 4. Download Control Forecast (cf) variables
        temp_file_path = H.download(search=":(2t|10u|10v):.*:cf:")

        # 5. Rename
        temp_file_path.replace(final_path)

        print(f"✅ Downloaded: {filename}")

    except Exception as e:
        print(f"❌ Failed:     {date.strftime('%Y-%m-%d %H:%M')} - {e}")

print("🎉 Batch complete.")