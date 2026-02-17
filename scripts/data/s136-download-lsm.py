import cdsapi
from pathlib import Path

out_dir = Path("/scratch2/mg963/data/ecmwf/era5/constants")
out_dir.mkdir(parents=True, exist_ok=True)
target = out_dir / "era5_lsm.nc"

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': 'land_sea_mask',
        'year': '2024',
        'month': '01',
        'day': '01',
        'time': '00:00',
        'format': 'netcdf',
    },
    str(target)
)

print(f"✅ Land-Sea Mask downloaded to: {target}")