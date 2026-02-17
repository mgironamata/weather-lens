import xarray as xr
import gcsfs
import pandas as pd
from datetime import datetime
import os

# --- CONFIGURATION ---
PROJECT_ID = 'weathernextforecasts'  # Replace with your actual Project ID
TARGET_DATE = datetime(2025, 7, 2)  # 2nd July 2025
BASE_PATH = "gs://weathernext/weathernext_2_0_0/zarr/2025_to_present"

# 1. SETUP: Connect to Google Cloud
fs = gcsfs.GCSFileSystem(project=PROJECT_ID, token='google_default')

print(f"Scanning files in {BASE_PATH}...")

try:
    # 2. LIST & FILTER: Find files after the target date
    all_files = fs.ls(BASE_PATH)
    valid_forecasts = []

    for file_path in all_files:
        # Get the folder name (e.g., "20250705_12_00")
        folder_name = os.path.basename(file_path)
        
        # Simple check: Does it look like a date? (Starts with 8 digits)
        # We assume format is YYYYMMDD_...
        try:
            date_part = folder_name.split('_')[0] 
            file_date = datetime.strptime(date_part, "%Y%m%d")
            
            # THE FILTER LOGIC
            if file_date >= TARGET_DATE:
                valid_forecasts.append(file_path)
                
        except ValueError:
            # Skip files that aren't dates (like PDFs or hidden files)
            continue

    # 3. SELECT: Pick the latest one from the filtered list
    if not valid_forecasts:
        print(f"No forecasts found after {TARGET_DATE.strftime('%Y-%m-%d')}.")
    else:
        valid_forecasts.sort()  # Sorts by date because of the YYYYMMDD naming
        
        # This gives you the latest one. 
        # (If you want the earliest one after July 2nd, change index to [0])
        latest_valid_path = "gs://" + valid_forecasts[-1] 
        
        print(f"Found {len(valid_forecasts)} valid forecasts.")
        print(f"Opening the latest one: {latest_valid_path}")

        # 4. LOAD: Open the dataset (Lazy Load - No Cost yet)
        ds = xr.open_dataset(
            latest_valid_path, 
            engine='zarr', 
            chunks='auto', 
            backend_kwargs={'consolidated': True}
        )
        
        print("\nSuccess! Dataset Ready:")
        print(ds)

except Exception as e:
    print(f"\nError: {e}")