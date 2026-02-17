import xarray as xr
import numpy as np
from pathlib import Path

# ==========================================================
# PATHS — WN2
# ==========================================================
BASE = Path("/scratch2/mg963/data/weathernext/gencast")
    
PATH_ORIG   = BASE / "gencast_2024_global_optchunks.zarr"
PATH_T2M    = BASE / "gencast_2024_t2m.zarr"
PATH_WIND   = BASE / "gencast_2024_winds.zarr"
PATH_PRECIP = BASE / "gencast_2024_precip.zarr"

# Variable names in split stores
V_T2M     = "2m_temperature"
V_U10     = "10m_u_component_of_wind"
V_V10     = "10m_v_component_of_wind"
V_PRECIP  = "total_precipitation_12hr"


# ==========================================================
# UTILS
# ==========================================================
def small_slice(da):
    """
    Very small slice to compare values cheaply.
    """
    # Select two timesteps & two ensemble samples,
    # all lead times, all spatial dims
    return da.isel(
        time=slice(0, 2),
        sample=slice(0, 2)
    )


def compare_var(orig, subset, varname):
    """
    Compare a variable in original vs a newly split store.
    """
    print(f"\n=== Checking variable: {varname} ===")

    da_orig = orig[varname]
    da_new  = subset[varname]

    # 1. dims
    print("Dims original:", da_orig.dims)
    print("Dims new     :", da_new.dims)

    # 2. shape
    print("Shape original:", da_orig.shape)
    print("Shape new     :", da_new.shape)

    # Check dimension match
    if da_orig.shape != da_new.shape:
        print("❌ Shape mismatch!")
    else:
        print("✔ Shape matches")

    # 3. small slice comparison
    so = small_slice(da_orig)
    sn = small_slice(da_new)

    # difference
    diff = (so - sn).compute()
    max_abs = float(np.nanmax(np.abs(diff)))

    if max_abs == 0.0:
        print("✔ Small-slice values are identical")
    else:
        print(f"❌ Mismatch in small slice: max abs diff = {max_abs:e}")

    # 4. NaN check (small slice only)
    nans_orig = int(np.isnan(so).sum())
    nans_new  = int(np.isnan(sn).sum())

    if nans_orig == nans_new:
        print("✔ NaNs consistent (small-slice)")
    else:
        print("❌ NaN count mismatch in small-slice")

    print("Done.")


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("=== Opening original & split datasets ===")

    ds_orig   = xr.open_zarr(PATH_ORIG,   consolidated=True)
    ds_t2m    = xr.open_zarr(PATH_T2M,    consolidated=True)
    ds_wind   = xr.open_zarr(PATH_WIND,   consolidated=True)
    ds_precip = xr.open_zarr(PATH_PRECIP, consolidated=False)  # might not be consolidated

    # ------------------------------------------------------
    # 1. Temperature
    # ------------------------------------------------------
    print("\n--- T2M CHECK ---")
    compare_var(ds_orig, ds_t2m, V_T2M)

    # ------------------------------------------------------
    # 2. Wind (both U and V)
    # ------------------------------------------------------
    print("\n--- WIND CHECK ---")
    compare_var(ds_orig, ds_wind, V_U10)
    compare_var(ds_orig, ds_wind, V_V10)

    # ------------------------------------------------------
    # 3. Precipitation
    # ------------------------------------------------------
    print("\n--- PRECIP CHECK ---")
    compare_var(ds_orig, ds_precip, V_PRECIP)


if __name__ == "__main__":
    main()