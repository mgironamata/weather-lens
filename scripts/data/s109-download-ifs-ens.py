import pandas as pd
from herbie import Herbie
from pathlib import Path
import warnings

# --- CONFIGURATION ---
start_date = "2024-02-01 06:00"
end_date = "2024-12-31 06:00"
save_dir = "/scratch2/mg963/data/ecmwf/ensembles/ifs/raw/"

# Define the "Rules of Engagement" for each step
# Key = Forecast Step (int)
# Value = The specific regex for variables you want at that step
step_requirements = {
    18: ":tp:",                     # Just Precip (for accumulation calc)
    24: ":(tp|2t|10u|10v):",        # Precip + Temp + Wind
    66: ":tp:",                     # Just Precip
    72: ":(tp|2t|10u|10v):"         # Precip + Temp + Wind
}

# Ensure directory exists
save_path = Path(save_dir)
save_path.mkdir(parents=True, exist_ok=True)

dates = pd.date_range(start=start_date, end=end_date, freq="D")

print(f"🚀 Starting Smart Download for {len(dates)} days...")
print(f"📂 Saving to: {save_path}")

for date in dates:
    # Real-time safety check
    if date > pd.Timestamp.now():
        break

    # Loop through our configured steps (18, 24, 66, 72)
    for step, var_regex in step_requirements.items():
        
        # 1. Define filename
        # e.g. ifs-ens_20240101_06z_f18.grib2
        filename = f"ifs-ens_{date.strftime('%Y%m%d')}_06z_f{step:02d}.grib2"
        final_path = save_path / filename

        # 2. Skip if exists
        if final_path.exists():
            # Optional: Print less frequently to reduce clutter
            # print(f"⏭️  Skipping {filename}")
            continue

        try:
            # 3. Initialize Herbie
            H = Herbie(
                date,
                model="ifs",
                product="enfo",
                fxx=step,
                save_dir=save_path,
                priority=['azure', 'aws'],
                verbose=False
            )

            # 4. Construct the Search String
            # Combine the Variable Regex (from dict) with the Member Regex (cf|pf)
            # Example for f24:  ":(tp|2t|10u|10v):.*:(cf|pf):"
            full_search = f"{var_regex}.*:(cf|pf):"

            # 5. Download
            downloaded_file = H.download(search=full_search)
            
            # 6. Rename
            downloaded_file.replace(final_path)
            
            print(f"✅ Saved:    {filename} (Vars: {var_regex})")

        except Exception as e:
            print(f"❌ Failed:   {filename} - {e}")

print("🎉 Custom Batch Complete.")