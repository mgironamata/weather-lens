#!/usr/bin/env python3
import argparse
from pathlib import Path
import xarray as xr
from dask.diagnostics import ProgressBar

def parse_chunks(s: str):
    """Parse chunk spec string like "time=16,latitude=256,longitude=256" into a dict."""
    chunks = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split("=")
        chunks[k.strip()] = int(v)
    return chunks

def list_grib_files(model_dir: Path, model: str, 
                    # var: str, fxx: int
                    ):
    """List GRIB2 files in model_dir matching model pattern.
    
    arguments:
        model_dir: Path to folder containing GRIB2 files
        model: "AIFS" or "IFS"

    returns:
        sorted list of Path objects
    """

    out = []
    for p in model_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".grib2":
            continue
        name_u = p.name.upper()
        if not name_u.startswith(f"{model}_"):
            continue
        else:
            out.append(p)   
    return sorted(out)

def maybe_subset_bbox(ds, bbox):
    """Subset the dataset to the given bounding box.
    
    Args:
        ds: xarray.Dataset with latitude and longitude coordinates
        bbox: tuple of (lat_max, lat_min, lon_min, lon_max) or None
        
    Returns:
        Subsetted xarray.Dataset    
    """
    if not bbox:
        return ds
    lat_max, lat_min, lon_min, lon_max = bbox
    return ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

def nc_encoding_no_chunks(ds: xr.Dataset):
    """
    Build a netCDF encoding dict that disables chunking (contiguous layout).
    """
    enc = {}
    for v in ds.data_vars:
        enc[v] = {"contiguous": True}  # no chunks on disk
        # If cfgrib supplied chunks automatically, explicitly drop them:
        if "chunks" in ds[v].encoding:
            enc[v]["chunksizes"] = None
    return enc

def write_three_bands_nc(ds: xr.Dataset, out_base: Path, tropics_lat: float, engine: str = "netcdf4"):
    """
    Split by latitude into NH, Tropics, SH and write three .nc files with no chunking.
    Tropics: (-tropics_lat, +tropics_lat); NH: [tropics_lat, 90]; SH: [-90, -tropics_lat].
    """
    # Slicing is order-agnostic for latitude
    nh = ds.sel(latitude=slice(90.0, tropics_lat))
    tr = ds.sel(latitude=slice(tropics_lat, -tropics_lat))
    sh = ds.sel(latitude=slice(-tropics_lat, -90.0))

    enc_nh = nc_encoding_no_chunks(nh)
    enc_tr = nc_encoding_no_chunks(tr)
    enc_sh = nc_encoding_no_chunks(sh)

    out_nh = out_base.with_name(f"{out_base.stem}_NH.nc")
    out_tr = out_base.with_name(f"{out_base.stem}_Tropics.nc")
    out_sh = out_base.with_name(f"{out_base.stem}_SH.nc")

    # Compute+write with a progress bar; contiguous layout, no chunks
    with ProgressBar():
        nh.to_netcdf(out_nh.as_posix(), mode="w", engine=engine, encoding=enc_nh)
    with ProgressBar():
        tr.to_netcdf(out_tr.as_posix(), mode="w", engine=engine, encoding=enc_tr)
    with ProgressBar():
        sh.to_netcdf(out_sh.as_posix(), mode="w", engine=engine, encoding=enc_sh)

    print(f"Wrote: {out_nh.name}, {out_tr.name}, {out_sh.name}")

def build(
    base: Path,
    model: str,
    var: str,
    fxx: int,
    out_path: Path,
    fmt: str,
    chunks: dict,
    bbox,
    parallel: bool,
    tropics_lat: float,
    nc_engine: str,
    control: bool = False,
):
    model_dir = base / model.lower() / "raw" # expects aifs/ and ifs/
    if not model_dir.exists():
        raise SystemExit(f"Folder not found: {model_dir}")

    files = list_grib_files(model_dir, model)
    if not files:
        raise SystemExit(f"No files in {model_dir} matching '{model}_*.grib2'")

    print(f"[{model}] {len(files)} files (example: {files[0].name})")
    print(f"Chunks: {chunks}")
    print(f"Format: {fmt}")
    print(f"Output base: {out_path}")

    open_kwargs = dict(
        engine="cfgrib",
        combine="nested",
        concat_dim="time",
        parallel=parallel,
        chunks=chunks,
        indexpath="",  # disable automatic index file creation
    )

    if control:
        open_kwargs["filter_by_keys"] = {"dataType": f"{var}"}

    elif model == "IFS":
        # Ensemble perturbations; change/remove if using different streams
        open_kwargs["filter_by_keys"] = {"dataType": "pf"}


    ds = xr.open_mfdataset([str(p) for p in files], **open_kwargs)
    ds = maybe_subset_bbox(ds, bbox)

    if fmt == "zarr":
        with ProgressBar():
            ds.to_zarr(out_path.as_posix(), mode="w", consolidated=True)
        print("Done.")
    else:
        # NetCDF path: write three banded files without chunking
        # (out_path is used as the base name; we append _NH/_TROPICS/_SH)
        write_three_bands_nc(ds, out_path.with_suffix(".nc"), tropics_lat, engine=nc_engine)
        print("Done.")

def main():
    ap = argparse.ArgumentParser(description="Build Zarr or NetCDF from AIFS/IFS GRIB2 for a variable/lead time.")
    ap.add_argument("--base", "-b", required=True,
                    help="Path whose subfolders are 'aifs' and 'ifs' (e.g., /scratch2/aifs-data/tp/72)")
    ap.add_argument("--model", "-m", choices=["AIFS", "IFS"], required=True, help="Model family")
    ap.add_argument("--var", "-v", required=True, help="Variable token in filenames (e.g., TP, 2T, U10)")
    ap.add_argument("--fxx", "-x", type=int, required=True, help="Lead time number in filenames (e.g., 24, 72)")
    ap.add_argument("--format", "-F", choices=["zarr", "nc"], default="zarr", help="Output format")
    ap.add_argument("--output", "-o", default=None,
                    help="Output path base. For zarr: folder (.zarr). For nc: base name (we append _NH/_TROPICS/_SH).")
    ap.add_argument("--chunks", "-c", default="time=16,latitude=256,longitude=256",
                    help='Chunk spec for reading (ignored by .nc writing), e.g. "time=16,latitude=256,longitude=256"')
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("LAT_MAX","LAT_MIN","LON_MIN","LON_MAX"),
                    help="Optional geographic subset before output (e.g., 55 47 5 16)")
    ap.add_argument("--parallel", action="store_true", help="Pass parallel=True to open_mfdataset")
    ap.add_argument("--tropics-lat", type=float, default=23.4394,
                    help="Absolute latitude delimiting the tropics (default 23.4394 degrees)")
    ap.add_argument("--nc-engine", default="netcdf4",
                    help="Engine for NetCDF writing (default 'netcdf4'; use 'scipy' if netcdf4 not available)")
    ap.add_argument("--control", action="store_true", help="Process control forecast files instead of perturbed ensembles")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    chunks = parse_chunks(args.chunks)
    bbox = tuple(args.bbox) if args.bbox else None

    default_base = f"{args.model}_{args.var.upper()}_FXX{args.fxx}"
    if args.format == "zarr":
        out_path = Path(args.output) if args.output else Path(f"{default_base}.zarr")
    else:
        # For .nc we’ll append _NH/_TROPICS/_SH; here we just keep a clean base name
        out_path = Path(args.output) if args.output else Path(default_base + ".nc")

    build(
        base=base,
        model=args.model,
        var=args.var,
        fxx=args.fxx,
        out_path=out_path,
        fmt=args.format,
        chunks=chunks,
        bbox=bbox,
        parallel=args.parallel,
        tropics_lat=args.tropics_lat,
        nc_engine=args.nc_engine,
    )

if __name__ == "__main__":
    main()

# run example with nc: 
# python s003-run-build-agg-zarr-or-nc.py -b /scratch2/aifs-data/tp/72 -m IFS -v TP -x 72 -F nc -o /scratch2/aifs-data/tp/72/output.nc