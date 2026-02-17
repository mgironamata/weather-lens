#!/usr/bin/env python

import argparse
import shutil
from pathlib import Path

import dask
import xarray as xr
from numcodecs import Blosc
import zarr


def build_compressor(cname: str, clevel: int, shuffle: str) -> Blosc:
    shuffle = shuffle.lower()
    if shuffle.startswith("bit"):
        shuffle_code = Blosc.BITSHUFFLE
    elif shuffle.startswith("byte"):
        shuffle_code = Blosc.SHUFFLE
    else:
        raise ValueError(f"Unknown shuffle type: {shuffle!r} (use 'bitshuffle' or 'byteshuffle')")
    return Blosc(cname=cname, clevel=clevel, shuffle=shuffle_code)


def _open_v2_store(in_path: Path) -> xr.Dataset:
    """
    Try to open a Zarr v2 store (consolidated or not).
    If it looks like a Zarr v3 store or not a store at all, raise a clean RuntimeError.
    """
    # First, very cheap heuristic: must contain at least .zgroup or .zmetadata for v2
    entries = {p.name for p in in_path.iterdir()}
    has_zgroup = ".zgroup" in entries
    has_zmetadata = ".zmetadata" in entries

    if not (has_zgroup or has_zmetadata):
        raise RuntimeError(
            f"{in_path} does not look like a Zarr v2 store "
            "(missing .zgroup / .zmetadata). It may be Zarr v3."
        )

    # Try consolidated=True, then False
    try:
        return xr.open_zarr(in_path, consolidated=True)
    except Exception:
        try:
            return xr.open_zarr(in_path, consolidated=False)
        except zarr.errors.GroupNotFoundError as e:
            raise RuntimeError(
                f"Failed to open {in_path} as a Zarr v2 group "
                "(GroupNotFoundError). It may be Zarr v3."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to open {in_path} as Zarr v2 (both consolidated=True/False)."
            ) from e


def compress_zarr(
    in_path: Path,
    out_path: Path | None = None,
    cname: str = "zstd",
    clevel: int = 5,
    shuffle: str = "bitshuffle",
    num_workers: int = 8,
    overwrite: bool = False,
):
    if not in_path.exists():
        raise FileNotFoundError(f"Input Zarr store does not exist: {in_path}")

    if out_path is None:
        if in_path.name.endswith(".zarr"):
            out_name = in_path.name[:-5] + "_compressed.zarr"
        else:
            out_name = in_path.name + "_compressed.zarr"
        out_path = in_path.with_name(out_name)

    if out_path.exists():
        if overwrite:
            print(f"⚠️  Output path {out_path} exists; removing because --overwrite is set.")
            shutil.rmtree(out_path)
        else:
            raise FileExistsError(
                f"Output path {out_path} already exists. "
                "Use --overwrite if you want to replace it."
            )

    print(f"📂 Input Zarr : {in_path}")
    print(f"📦 Output Zarr: {out_path}")
    print(f"   Compressor : Blosc(cname='{cname}', clevel={clevel}, shuffle='{shuffle}')")
    print(f"   Workers    : {num_workers}")

    dask.config.set(scheduler="threads", num_workers=num_workers)

    print("🔹 Opening source dataset (expecting Zarr v2)...")
    ds = _open_v2_store(in_path)
    print("   Variables:", list(ds.data_vars))

    compressor = build_compressor(cname, clevel, shuffle)

    encoding = {
        var: {"compressor": compressor}
        for var in ds.data_vars
    }

    print("💾 Writing compressed Zarr (this may take a while)...")
    ds.to_zarr(
        str(out_path),
        mode="w",
        consolidated=True,  # create consolidated metadata in the *new* store
        encoding=encoding,
        zarr_format=2,
    )

    print("✅ Done.")
    print("   You can compare sizes with e.g.:")
    print(f"      du -sh {in_path}")
    print(f"      du -sh {out_path}")
    print("\nWhen happy, you can swap:")
    print(f"   mv {in_path} {in_path.with_name(in_path.name + '_uncompressed_backup')}")
    print(f"   mv {out_path} {in_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recompress a Zarr v2 store with Blosc (e.g. Zstd) to save disk space."
    )
    parser.add_argument("input", help="Path to existing Zarr store (directory ending in .zarr).")
    parser.add_argument(
        "-o", "--output",
        help="Optional output Zarr path. If omitted, '_compressed.zarr' is added next to input.",
    )
    parser.add_argument(
        "--cname", default="zstd",
        help="Blosc codec name (default: zstd; alternatives: lz4, zlib, etc.)",
    )
    parser.add_argument(
        "--clevel", type=int, default=5,
        help="Compression level 1–9 (default: 5 is a good tradeoff).",
    )
    parser.add_argument(
        "--shuffle", default="bitshuffle",
        help="Shuffle type: 'bitshuffle' (best for floats) or 'byteshuffle'.",
    )
    parser.add_argument(
        "-j", "--num-workers", type=int, default=8,
        help="Number of dask threads (default: 8).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="If set, overwrite existing output Zarr directory.",
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else None

    try:
        compress_zarr(
            in_path=in_path,
            out_path=out_path,
            cname=args.cname,
            clevel=args.clevel,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
        )
    except RuntimeError as e:
        # Nice user-facing message for v3 / non-v2 stores
        print(f"❌ Could not compress {in_path} as Zarr v2:")
        print(f"   {e}")
        print("   Hint: this store may be Zarr v3 (written from your 'regrid' env). "
              "This script only supports Zarr v2 stores in this environment.")


if __name__ == "__main__":
    main()