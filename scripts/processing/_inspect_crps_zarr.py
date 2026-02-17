import xarray as xr


def open_any(path: str) -> xr.Dataset:
    # consolidated=False works for both v2 and v3 and whether or not metadata is consolidated
    return xr.open_zarr(path, consolidated=False)


def main() -> None:
    paths = [
        ("ifs", "/scratch2/mg963/results/ecmwf/ifs_vs_ERA5/crps_tp_24h.zarr"),
        ("gencast", "/scratch2/mg963/results/weathernext/gencast_vs_ERA5/crps_tp_24h.zarr"),
        ("wn2", "/scratch2/mg963/results/weathernext/wn2_vs_ERA5/crps_tp_24h.zarr"),
    ]

    for name, path in paths:
        print("opening", name, path, flush=True)
        ds = open_any(path)
        print("==", name, "==")
        print("vars:", list(ds.data_vars))
        print("dims:", {k: int(v) for k, v in ds.dims.items()})
        print("coords:", list(ds.coords))
        for c in ("latitude", "longitude", "lat", "lon"):
            if c in ds.coords:
                coord = ds[c]
                print(f"{c}: shape={coord.shape} min={coord.min().item()} max={coord.max().item()}")
        print()


if __name__ == "__main__":
    main()
