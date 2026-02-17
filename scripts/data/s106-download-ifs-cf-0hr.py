import pandas as pd
from herbie import Herbie
from pathlib import Path
import shutil

# 1. Configuration
# ----------------
start_date = "2024-02-01"
end_date = "2024-02-02"  # Set your range
output_dir = "/scratch2/mg963/data/ecmwf/analysis/raw/" # Where you want the final clean files

# Ensure output directory existsimport pandas as pd
from herbie import Herbie
from pathlib import Path

# 1. Configuration
# ----------------
# Explicitly start at 06:00
start_date = "2024-02-01 06:00"
end_date = "2024-02-02 06:00"  
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
save_path = Path(output_dir)
save_path.mkdir(parents=True, exist_ok=True)

dates = pd.date_range(start=start_date, end=end_date, freq="D")

# 2. Download Loop
# ----------------
print(f"📥 Starting download to: {save_path}")

for date in dates:
    # 1. Define the clean filename you want
    # e.g., ifs_ens_cf_20240205.grib2
    filename = f"ifs_ens_cf_{date.strftime('%Y%m%d')}.grib2"
    final_path = save_path / filename

    # 2. Skip if already exists (Saves time/bandwidth)
    if final_path.exists():
        print(f"⏭️  Skipping {filename} (Already exists)")
        continue

    try:
        # 3. Initialize Herbie
        H = Herbie(
            date,
            model="ifs",
            product="enfo",
            fxx=0,
            save_dir=save_path, # Save temp file in same folder
            priority=['azure', 'aws'],
            verbose=False
        )

        # 4. Download
        # This downloads a file named something like "subset_e49a...grib2"
        # We capture the path in `temp_file_path`
        temp_file_path = H.download(search=":(2t|10u|10v):.*:cf:")

        # 5. Rename to your clean filename
        # This is instant (just a metadata change)
        temp_file_path.replace(final_path)

        print(f"✅ Downloaded: {filename}")

    except Exception as e:
        print(f"❌ Failed:     {date.strftime('%Y-%m-%d')} - {e}")

print("🎉 Batch complete.")