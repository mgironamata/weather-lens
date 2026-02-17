import xarray as xr
import numpy as np
import time
from dask.diagnostics import ProgressBar

def crps_fair(forecast: xr.DataArray, obs: xr.DataArray, member_dim="member"):
    """
    forecast: DataArray with a member dimension (…, member)
    obs     : DataArray broadcastable to forecast without the member dim (…)
    returns : CRPS_fair with the member dim removed
    """
    M = forecast.sizes[member_dim]
    # term1 = mean |X - y|
    term1 = (np.abs(forecast - obs)).mean(dim=member_dim)

    # term2 = 0.5 * mean_{i≠j} |X - X'|
    def _pairwise_mean_offdiag(a):
        # a: (..., M)
        # pairwise |a_i - a_j|
        diff = np.abs(a[..., None] - a[..., None, :])  # (..., M, M)
        # sum over i,j; diagonal is zero so no need to subtract explicitly
        s = diff.sum(axis=(-2, -1))
        denom = M * (M - 1)
        # handle M=1 safely
        return np.where(denom > 0, 0.5 * s / denom, 0.0)

    term2 = xr.apply_ufunc(
        _pairwise_mean_offdiag, forecast,
        input_core_dims=[[member_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[forecast.dtype],
        dask_gufunc_kwargs = {'allow_rechunk':True}
    )

    return term1 - term2


if __name__ == "__main__":

    uber_variable = "10uv"
    variable = "v10"  # 2-meter temperature
    analysis_variable = "10v"
    fxx = 72  # lead time
    ref_dataset = "cf" # era5 or cf
    
    ZONES = [
             "NH",
             "SH", 
             "Tropics"
            ]  # "NH" or "SH" or "Tropics"
    
    ERA_BASEPATH = f"/scratch2/mg963/data/ecmwf/era5/{uber_variable}/era5_daily_{variable}_july2sept2025"
    CF_BASEPATH = f"/scratch2/mg963/data/ecmwf/ensembles/CF/0/ifs/IFS_{analysis_variable}_0600UTC_FXX0.nc"
    AIFS_BASEPATH = f"/scratch2/mg963/data/ecmwf/ensembles/{uber_variable}/{fxx}/aifs/AIFS_{variable}_FXX{fxx}"
    IFS_BASEPATH = f"/scratch2/mg963/data/ecmwf/ensembles/{uber_variable}/{fxx}/ifs/IFS_{variable}_FXX{fxx}"

    for ZONE in ZONES[:]:
        print("Processing zone:", ZONE)
        ERA_PATH = f"{ERA_BASEPATH}_{ZONE}_aligned.nc"
        AIFS_PATH = f"{AIFS_BASEPATH}_{ZONE}.nc"
        IFS_PATH = f"{IFS_BASEPATH}_{ZONE}.nc"
        print("Data paths:", ERA_PATH, AIFS_PATH, IFS_PATH)

        if ref_dataset == "cf":
            if ZONE == "NH":
                latitude_slice = slice(90, 23.4394)
            elif ZONE == "SH":
                latitude_slice = slice(-23.4394, -90)
            else:  # Tropics
                latitude_slice = slice(23.4394, -23.4394)

            cf = xr.open_dataarray(CF_BASEPATH).sel(latitude=latitude_slice)
        else:
            era = xr.open_dataarray(ERA_PATH)

        aifs = xr.open_dataarray(AIFS_PATH)
        ifs = xr.open_dataarray(IFS_PATH)
        
        print("Datasets loaded.")

        start = time.time()
        
        obs=cf if ref_dataset == "cf" else era

        with ProgressBar():
            crps_aifs = crps_fair(forecast=aifs,
                        obs=obs,
                        member_dim='number')
        
        elapsed = time.time() - start
        print(f"{ZONE} CRPS computed in", round(elapsed, 2), "seconds.")
        if ref_dataset == "cf":
            with ProgressBar():
                crps_aifs.to_netcdf(f"../results/CRPS_AIFS_{ZONE}_{variable}_cf.nc")
        else:   
            crps_aifs.to_netcdf(f"../results/CRPS_AIFS_{ZONE}_{variable}_era5.nc")
        
        aifs.close()

        start = time.time()
        with ProgressBar():
            crps_ifs = crps_fair(forecast=ifs,
                            obs=obs,
                            member_dim='number')
        elapsed = time.time() - start
        print(f"{ZONE} IFS CRPS computed in", round(elapsed, 2), "seconds.")
        if ref_dataset == "cf":
            with ProgressBar():
                crps_ifs.to_netcdf(f"../results/CRPS_IFS_{ZONE}_{variable}_cf.nc")
        else:   
            crps_ifs.to_netcdf(f"../results/CRPS_IFS_{ZONE}_{variable}_era5.nc")

        # close datasets
        ifs.close()
        obs.close()