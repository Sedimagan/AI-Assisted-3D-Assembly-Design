#!/usr/bin/env python3
"""
inspect_step_assembly.py

General-purpose check: given a folder of .step/.stp files (one "assembly"
worth), classify each file as a PART or a merged ASSEMBLY by scanning its
STEP entities -- no CAD library required, just text parsing of the STEP
exchange structure (ISO 10303-21).

Key entities checked:
  - MANIFOLD_SOLID_BREP        : one per solid body
  - PRODUCT_DEFINITION(        : one per distinct product (part or assembly)
  - NEXT_ASSEMBLY_USAGE_OCCURRENCE : only present in files that place child
                                      components into a parent -- this is
                                      the actual "this is an assembly" marker
                                      in STEP AP203/AP214/AP242.

Rule of thumb:
  - 0 NEXT_ASSEMBLY_USAGE_OCCURRENCE + 1 MANIFOLD_SOLID_BREP -> single PART
  - >0 NEXT_ASSEMBLY_USAGE_OCCURRENCE, or many PRODUCT_DEFINITIONs pointing
    at each other -> merged ASSEMBLY file (parts placed in world/parent space)

USAGE
-----
    python inspect_step_assembly.py /path/to/one/assembly/folder
    python inspect_step_assembly.py /path/to/assemblies_by_id --sample 20
"""

import argparse
import sys
from pathlib import Path


def classify_step(path: Path, max_bytes: int = 20_000_000):
    """Read up to max_bytes of a STEP file (they're plain text) and count
    the entities that indicate part vs. assembly structure. Follows
    symlinks explicitly (path.resolve()) since some dataset pipelines
    (e.g. the AutoMate downloader) store STEP files as symlinks."""
    try:
        real_path = path.resolve() if path.is_symlink() else path
        with open(real_path, "r", errors="ignore") as f:
            content = f.read(max_bytes)
    except Exception as e:
        return {"error": str(e)}

    solids = content.count("MANIFOLD_SOLID_BREP")
    product_defs = content.count("PRODUCT_DEFINITION(")
    nauo = content.count("NEXT_ASSEMBLY_USAGE_OCCURRENCE")
    shape_reps = content.count("SHAPE_REPRESENTATION_RELATIONSHIP")

    if nauo > 0:
        kind = "ASSEMBLY (has component placements)"
    elif solids > 1 or product_defs > 1:
        kind = "MULTI-BODY (several solids in one file, but no formal assembly links)"
    elif solids == 1:
        kind = "PART (single solid body)"
    else:
        kind = "UNKNOWN (no solids found -- may be surfaces/wireframe/empty)"

    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "solids": solids,
        "product_definitions": product_defs,
        "assembly_usage_occurrences": nauo,
        "shape_rep_relationships": shape_reps,
        "classification": kind,
    }


def inspect_folder(folder: Path):
    step_files = sorted(list(folder.glob("*.step")) + list(folder.glob("*.stp")))
    if not step_files:
        print(f"  (no .step/.stp files directly in {folder})")
        return
    any_assembly = False
    for f in step_files:
        info = classify_step(f)
        if "error" in info:
            print(f"  [!] {f.name}: could not read ({info['error']})")
            continue
        print(f"  {info['file']:<70} {info['classification']}  "
              f"(solids={info['solids']}, product_defs={info['product_definitions']}, "
              f"nauo={info['assembly_usage_occurrences']}, size={info['size_bytes']:,}B)")
        if info["assembly_usage_occurrences"] > 0:
            any_assembly = True

    print()
    if any_assembly:
        print("  => Found a merged assembly-level STEP file above.")
    else:
        print("  => No merged assembly-level STEP file found in this folder.")
        print("     All STEP files here are individual parts. If there is an")
        print("     assembly_info.json / metadata file alongside them, THAT is")
        print("     what defines the assembly (which parts belong together +")
        print("     how they mate) -- there is no single geometry file that")
        print("     represents the whole assembly.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="An assembly folder, OR a parent folder containing many assembly subfolders")
    ap.add_argument("--sample", type=int, default=0,
                     help="If PATH contains many assembly subfolders, only inspect this many of them (0 = treat PATH itself as one assembly folder)")
    args = ap.parse_args()

    if args.sample > 0:
        subfolders = [p for p in args.path.iterdir() if p.is_dir()]
        subfolders = subfolders[: args.sample]
        print(f"Inspecting {len(subfolders)} of the subfolders under {args.path} ...\n")
        for sub in subfolders:
            print(f"=== {sub.name} ===")
            inspect_folder(sub)
            print()
    else:
        print(f"=== {args.path} ===")
        inspect_folder(args.path)


if __name__ == "__main__":
    main()
