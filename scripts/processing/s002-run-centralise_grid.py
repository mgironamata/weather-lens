#!/usr/bin/env python3
import argparse, re, shutil
from pathlib import Path

PATTERN = re.compile(r"^subset_[^_]+__((CF_IFS|AIFS|IFS)_.+)$")  # capture the part starting at AIFS_/IFS_

def find_target_name(name: str) -> str | None:
    m = PATTERN.match(name)
    return m.group(1) if m else None

def unique_path(dest: Path) -> Path:
    """If dest exists, append _dupN before suffix."""
    if not dest.exists():
        return dest
    stem, suf = dest.stem, dest.suffix
    n = 1
    while True:
        cand = dest.with_name(f"{stem}_dup{n}{suf}")
        if not cand.exists():
            return cand
        n += 1

def process_root(root: Path, kind: str, apply: bool, variable: str = "tp", fxx: str = "72"):
    """
    root: path to the project root that contains tp/72/<kind>
    kind: 'aifs' or 'ifs'
    """
    base = root / f"{variable}" / f"{fxx}" / kind
    if not base.exists():
        print(f"Skip: {base} does not exist.")
        return

    # Destination is the base folder itself
    dest_dir = base

    # Walk subfolders one level (dates) + any nested just in case
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".grib2", ".grb2", ".grb"):
            continue

        new_name = find_target_name(p.name)
        print(p.name, new_name)
        if not new_name:
            # Either already clean or not matching the subset_... pattern
            # Only move if it already starts correctly (AIFS_/IFS_) and lives in a subfolder
            if (p.name.startswith("AIFS_") and kind == "aifs") or (p.name.startswith("IFS_") and kind == "ifs"):
                if p.parent != dest_dir:
                    target = unique_path(dest_dir / p.name)
                    action = f"MOVE only: {p} -> {target}"
                    print(action)
                    if apply:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(p), str(target))
            continue

        # Ensure we aren’t crossing kinds accidentally
        if new_name.startswith("AIFS_") and kind != "aifs":
            continue
        if new_name.startswith("IFS_") and kind != "ifs":
            continue

        target = unique_path(dest_dir / new_name)
        action = f"RENAME+MOVE: {p} -> {target}"
        print(action)
        if apply:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(target))

def main():
    ap = argparse.ArgumentParser(description="Centralise and rename AIFS/IFS GRIB files.")
    ap.add_argument("root", nargs="?", default=".", help="Project root containing tp/72/aifs and tp/72/ifs")
    ap.add_argument("--apply", action="store_true", help="Actually perform changes (default is dry-run)")
    ap.add_argument("--variable", default="tp", help="Variable name (default is 'tp')")
    ap.add_argument("--fxx", default="72", help="FXX value (default is '72')")
    # how to call the script: python run-centralise_grid.py /path/to/project/root --apply --variable tp --fxx 72
    args = ap.parse_args()

    root = Path(args.root).resolve()
    print(f"Root: {root}")
    print("Mode:", "APPLY" if args.apply else "DRY-RUN")

    if args.variable != 'CF':
        process_root(root, "aifs", args.apply, variable=args.variable, fxx=args.fxx)
    
    process_root(root, "ifs", args.apply, variable=args.variable, fxx=args.fxx)

if __name__ == "__main__":
    main()