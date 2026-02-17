#!/usr/bin/env python

import argparse
import shutil
from pathlib import Path

import dask
import xarray as xr
import zarr


def build_compressor_dict(cname: str, clevel: int, shuffle: str) -> dict:
    """Build a Zarr v3 codec configuration dictionary."""
    shuffle = shuffle.lower()
    if shuffle.startswith("bit"):
        shuffle_mode = "bitshuffle"
    elif shuffle.startswith("byte"):
        shuffle_mode = "shuffle"
    elif shuffle == "noshuffle":
        shuffle_mode = "noshuffle"
    else:
        raise ValueError(f"Unknown shuffle type: {shuffle!r} (use 'bitshuffle', 'byteshuffle', or 'noshuffle')")
    
    return {
        "name": "blosc",
        "configuration": {
            "cname": cname,
            "clevel": clevel,
            "shuffle": shuffle_mode,
        }
    }


def compress_zarr_v3(
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

    # --- Open source store (v3) ---
    print("🔹 Opening source dataset (Zarr v3 via xarray)...")
    try:
        ds = xr.open_zarr(in_path, consolidated=True)
    except Exception:
        print("   ℹ️  consolidated=True failed; reopening with consolidated=False ...")
        ds = xr.open_zarr(in_path, consolidated=False)

    print("   Variables:", list(ds.data_vars))

    # Use zarr directly to copy with compression, parallel chunk processing
    print("💾 Writing compressed Zarr v3 (parallel chunk processing)...")
    
    from zarr.codecs import BloscCodec
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    codec_config = build_compressor_dict(cname, clevel, shuffle)
    blosc_codec = BloscCodec(
        cname=codec_config["configuration"]["cname"],
        clevel=codec_config["configuration"]["clevel"],
        shuffle=codec_config["configuration"]["shuffle"],
    )
    
    # Open source and destination
    src_store = zarr.open_group(in_path, mode="r")
    out_store = zarr.open_group(out_path, mode="w")
    
    def copy_chunk(src_array, dst_array, slices):
        """Copy a single chunk from source to destination."""
        dst_array[slices] = src_array[slices]
        return slices
    
    # Copy each data variable with compression, parallel chunks
    for var_name in ds.data_vars:
        print(f"   Compressing {var_name}...")
        src_array = src_store[var_name]
        
        # Create compressed array with same structure
        dst_array = out_store.create_array(
            name=var_name,
            shape=src_array.shape,
            chunks=src_array.chunks,
            dtype=src_array.dtype,
            compressor=blosc_codec,  # Changed from 'codecs' to 'compressor'
            fill_value=src_array.fill_value if hasattr(src_array, 'fill_value') else None,
        )
        
        # Generate all chunk slices
        chunk_grid_shape = tuple(
            (s + c - 1) // c  # ceiling division
            for s, c in zip(src_array.shape, src_array.chunks)
        )
        
        all_slices = []
        for chunk_coords in np.ndindex(chunk_grid_shape):
            slices = tuple(
                slice(i * c, min((i + 1) * c, s))
                for i, c, s in zip(chunk_coords, src_array.chunks, src_array.shape)
            )
            all_slices.append(slices)
        
        # Process chunks in parallel
        total_chunks = len(all_slices)
        completed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(copy_chunk, src_array, dst_array, slices): slices
                for slices in all_slices
            }
            
            for future in as_completed(futures):
                future.result()  # will raise if there was an error
                completed += 1
                if completed % max(1, total_chunks // 20) == 0:  # Progress every 5%
                    print(f"      {completed}/{total_chunks} chunks ({100*completed//total_chunks}%)")
        
        # Copy attributes
        dst_array.attrs.update(src_array.attrs)
    
    # Copy coordinates
    for coord_name in ds.coords:
        if coord_name not in ds.data_vars:
            print(f"   Copying coordinate {coord_name}...")
            src_array = src_store[coord_name]
            dst_array = out_store.create_array(
                name=coord_name,
                shape=src_array.shape,
                chunks=src_array.chunks if hasattr(src_array, 'chunks') else None,
                dtype=src_array.dtype,
                compressor=blosc_codec,  # Changed from 'codecs' to 'compressor'
            )
            dst_array[:] = src_array[:]
            dst_array.attrs.update(src_array.attrs)
    
    # Copy global attributes
    out_store.attrs.update(src_store.attrs)

    print("✅ Done.")
    print("   You can compare sizes with:")
    print(f"      du -sh {in_path}")
    print(f"      du -sh {out_path}")
    print("\nWhen happy, you can swap:")
    print(f"   mv {in_path} {in_path.with_name(in_path.name + '_uncompressed_backup')}")
    print(f"   mv {out_path} {in_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recompress a Zarr v3 store with Blosc (e.g. zstd) to save disk space."
    )
    parser.add_argument("input", help="Path to existing Zarr store (.zarr directory).")
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
        help="Compression level 1–9 (default: 5).",
    )
    parser.add_argument(
        "--shuffle", default="bitshuffle",
        help="Shuffle type: 'bitshuffle' (floats) or 'byteshuffle'.",
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

    compress_zarr_v3(
        in_path=in_path,
        out_path=out_path,
        cname=args.cname,
        clevel=args.clevel,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()