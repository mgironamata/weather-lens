import argparse
from herbie import Herbie
import os
from datetime import datetime, timedelta


def tp_download(DATE, fxx=24, dest_dir=None, variable=None):
    # RETRIEVE
    HAIFS = Herbie(DATE, model="aifs", product="enfo", fxx=fxx) #param=variable)
    HIFS = Herbie(DATE, model="ifs", product="enfo", fxx=fxx) #param=variable)

    HAIFS_NAME = f"AIFS_{DATE}_{variable}_FXX{fxx}.grib2"
    HIFS_NAME = f"IFS_{DATE}_{variable}_FXX{fxx}.grib2"

    HAIFS.LOCALFILE = HAIFS_NAME
    HIFS.LOCALFILE = HIFS_NAME
    
    # DOWNLOAD
    PATH_AIFS = HAIFS.download(search=f":{variable}:",verbose=True, overwrite=True, save_dir=dest_dir)
    PATH_IFS = HIFS.download(search=f":{variable}:",verbose=True, overwrite=True, save_dir=dest_dir)

def download_control_forecast(DATE, dest_dir=None, fxx=0):
    # RETRIEVE
    HIFS = Herbie(DATE, model="ifs", product="enfo", fxx=fxx) #param=variable)

    HIFS_NAME = f"CF_IFS_{DATE}_FXX{fxx}.grib2"
    HIFS.LOCALFILE = HIFS_NAME
    
    # DOWNLOAD
    PATH_IFS = HIFS.download(search=f":cf:",verbose=True, overwrite=True, save_dir=dest_dir)

if __name__ == "__main__":

    # argparse arguments for variable and fxx
    parser = argparse.ArgumentParser(description="Download AIFS and IFS data.")
    parser.add_argument("--variable", default="2t", help="Variable to download (default: 2t)")
    parser.add_argument("--fxx", default=72, type=int, help="Forecast hours (default: 72)")
    parser.add_argument("--dest_dir", default="/scratch2/mg963/data/ecmwf/ensembles", help="Destination directory for downloads (default: current directory)")
    parser.add_argument("--control", default=False, action='store_true', help="Download control forecast data only")
    args = parser.parse_args()
    # example: python run-data-download.py --variable tp --fxx 72

    variable = args.variable
    fxx = args.fxx
    dest_dir = args.dest_dir
    control = args.control

    if control:
        dest_path = f"{dest_dir}/control/"
    else:
        dest_path = f"{dest_dir}/{variable}/{fxx}/"

    start_date = datetime(2025, 7, 2)
    date_list = [start_date + timedelta(days=i) for i in range(90)]
    formatted_dates = [date.strftime("%Y-%m-%d-06") for date in date_list]

    for date_str in formatted_dates:
        # if date > 29 september 2025, stop
        if datetime.strptime(date_str, "%Y-%m-%d-%H") > datetime(2025, 9, 29, 6):
            break
        print("Starting with date:", date_str)
        if control:
            download_control_forecast(date_str, dest_dir=dest_path, fxx=0)
        else:
            tp_download(date_str, dest_dir=dest_path, fxx=fxx, variable=variable)
        print("Finished date:", date_str)