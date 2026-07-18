#!/usr/bin/env python3
"""
download_automate_assemblies.py

Download the AutoMate CAD Assembly Dataset (Jones et al., "AutoMate: A Dataset
and Learning Approach for Automatic Mating of CAD Assemblies", TOG 2021) from
Zenodo, and reconstruct it into a ready-to-use folder structure:

    <out>/assemblies_by_id/<assembly_id>/<part_file>.step
    <out>/assemblies_by_id/<assembly_id>/assembly_info.json   (mates + metadata)

Dataset page : https://zenodo.org/records/7776208
Paper        : https://doi.org/10.1145/3478513.3480562
License      : CC0 1.0 Universal (public domain) -- data itself is CC0,
               but you must credit the paper appropriately in your report.

------------------------------------------------------------------------
INSTALL
------------------------------------------------------------------------
    pip install requests tqdm pandas pyarrow

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
    # Full run: download everything (~14 GB), extract, and group by assembly
    python download_automate_assemblies.py --out ./automate_dataset

    # Just grab the small metadata files first to inspect the schema before
    # committing to the 13 GB step.zip download
    python download_automate_assemblies.py --out ./automate_dataset --skip-step

    # Already downloaded/extracted, just want to (re)run the grouping step
    python download_automate_assemblies.py --out ./automate_dataset --skip-download --skip-extract

------------------------------------------------------------------------
NOTE ON GROUPING
------------------------------------------------------------------------
parts.parquet has no assembly-id column, and part/assembly IDs cannot be
mapped by string manipulation (their shared-looking "hash1_hash2_hash3"
prefix does not actually correspond across assemblies and parts). Grouping
instead reads each assembly's own extracted JSON file (from assemblies.zip),
which has a "parts": [{"id": <part_id>, ...}, ...] list whose ids match the
extracted STEP filenames exactly.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests tqdm pandas pyarrow")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("Missing dependency: pip install requests tqdm pandas pyarrow")


ZENODO_RECORD = "7776208"
FILES_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{{}}?download=1"

# Published MD5 checksums (from the Zenodo record page) so large downloads
# can be verified instead of trusted blindly.
CHECKSUMS = {
    "assemblies.parquet": "dc47739e116ec168c984d589a6a8caad",
    "assemblies.zip": "a520a85a01d067a3506bb0d54382de7c",
    "config_encodings.json": "0fb1a05967e8e2d24f659f7c901aa12d",
    "mates.parquet": "7b9d36379077ad0e2a4b4ec2758f791b",
    "parts.parquet": "82bf5126ad9673a611c1f02d682fd496",
    "README.md": "53cead6fd88498e34976606c13d72dc3",
    "step.zip": "7eb29a422507cc881bd4b70416ec723b",
    # parasolid.zip deliberately omitted: 20.5 GB, proprietary format, not
    # needed since STEP is provided as the open alternative.
}

METADATA_FILES = [
    "assemblies.parquet",
    "parts.parquet",
    "mates.parquet",
    "config_encodings.json",
    "README.md",
]


# ============================================================================
# Download helpers
# ============================================================================

def md5_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: Path, expected: str) -> bool:
    print(f"  Verifying checksum for {path.name} ...")
    actual = md5_of(path)
    ok = actual == expected
    print(f"  {'OK' if ok else 'MISMATCH'}: expected {expected}, got {actual}")
    return ok


def download_file(name: str, dest_dir: Path, chunk_size: int = 1 << 20) -> Path:
    """Resumable download of one Zenodo file with a progress bar."""
    url = FILES_BASE_URL.format(name)
    dest = dest_dir / name
    tmp = dest_dir / (name + ".part")

    if dest.exists():
        expected = CHECKSUMS.get(name)
        if expected is None or verify_checksum(dest, expected):
            print(f"[skip] {name} already downloaded and verified.")
            return dest
        print(f"[redo] {name} exists but failed checksum, re-downloading.")
        dest.unlink()

    headers = {}
    mode = "wb"
    resume_pos = 0
    if tmp.exists():
        resume_pos = tmp.stat().st_size
        headers["Range"] = f"bytes={resume_pos}-"
        mode = "ab"

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:
            # Range not satisfiable -> file is already fully downloaded
            pass
        else:
            r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + resume_pos
        with open(tmp, mode) as f, tqdm(
            total=total if total else None,
            initial=resume_pos,
            unit="B",
            unit_scale=True,
            desc=name,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    tmp.rename(dest)

    expected = CHECKSUMS.get(name)
    if expected and not verify_checksum(dest, expected):
        raise RuntimeError(
            f"{name} failed checksum verification after download. "
            f"Delete {dest} and re-run to retry."
        )
    return dest


def extract_zip(zip_path: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / ".extracted_ok"
    if marker.exists():
        print(f"[skip] {zip_path.name} already extracted into {dest_dir}")
        return
    print(f"Extracting {zip_path.name} -> {dest_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        for member in tqdm(members, desc=f"extract {zip_path.name}"):
            zf.extract(member, dest_dir)
    marker.write_text("ok")


# ============================================================================
# Grouping STEP files by assembly
# ============================================================================
#
# Note: parts.parquet has no assembly-id column and part_id/STEP filenames
# cannot be mapped to an assembly by string manipulation (verified against
# the actual extracted files -- the shared "hash1_hash2_hash3_default"
# prefix pattern does NOT hold across assemblies/parts). The reliable source
# of truth is each assembly's own extracted JSON file (from assemblies.zip),
# which has a "parts": [{"id": <part_id>, ...}, ...] list whose ids match
# the STEP filenames (stem) exactly.

def index_step_files(step_dir: Path):
    """Map every extracted .step/.stp filename stem -> full path.
    Also keeps a lowercased-key index so lookups can fall back to a
    case-insensitive match if the exact-case match fails.
    """
    index = {}
    index_lower = {}
    for p in step_dir.rglob("*"):
        if p.suffix.lower() in (".step", ".stp") and p.is_file():
            index[p.stem] = p
            index_lower.setdefault(p.stem.lower(), p)
    print(f"Indexed {len(index)} STEP files under {step_dir}")
    return index, index_lower


def group_assemblies(assemblies_json_dir: Path, step_dir: Path, out_dir: Path, use_copy: bool):
    step_index, step_index_lower = index_step_files(step_dir)

    json_files = list(assemblies_json_dir.rglob("*.json"))
    print(f"Found {len(json_files)} assembly JSON files under {assemblies_json_dir}")

    grouped_dir = out_dir / "assemblies_by_id"
    grouped_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_missing = 0, 0

    for jf in tqdm(json_files, desc="grouping assemblies"):
        info = json.loads(jf.read_text())
        assembly_id = str(info.get("assemblyId") or jf.stem)

        assembly_folder = grouped_dir / assembly_id
        assembly_folder.mkdir(parents=True, exist_ok=True)

        found_any = False
        for part in info.get("parts", []):
            part_id = part.get("id")
            if not part_id:
                continue

            src = step_index.get(part_id) or step_index_lower.get(part_id.lower())
            if src is None:
                n_missing += 1
                continue

            dst = assembly_folder / src.name
            if not dst.exists():
                if use_copy:
                    shutil.copy2(src, dst)
                else:
                    try:
                        os.symlink(src.resolve(), dst)
                    except OSError:
                        shutil.copy2(src, dst)
            found_any = True

        if found_any:
            n_ok += 1

        (assembly_folder / "assembly_info.json").write_text(
            json.dumps(info, indent=2, default=str)
        )

    print(f"\nDone. {n_ok} assemblies populated with >=1 STEP part, "
          f"{n_missing} part references had no matching STEP file.")
    print(f"Output folder structure: {grouped_dir}/<assembly_id>/*.step")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/Users/mbp/Documents/MTECH/Sem4/Individual_project/AI_Assisted_3D_Assembly_Design/Assembly_files/Automate_3d_models"),
                     help="Output root directory (default: /Users/mbp/Documents/MTECH/Sem4/Individual_project/AI_Assisted_3D_Assembly_Design/Assembly_files/Automate_3d_models)")
    ap.add_argument("--skip-download", action="store_true",
                     help="Skip downloading, assume files already exist in <out>/raw")
    ap.add_argument("--skip-step", action="store_true",
                     help="Don't download/extract the 13 GB step.zip (metadata only)")
    ap.add_argument("--skip-extract", action="store_true",
                     help="Skip extraction step (assume already extracted)")
    ap.add_argument("--skip-group", action="store_true",
                     help="Skip the grouping-by-assembly step")
    ap.add_argument("--copy", action="store_true",
                     help="Copy STEP files into assembly folders instead of symlinking "
                          "(uses much more disk space, but is more portable)")
    args = ap.parse_args()

    raw_dir = args.out / "raw"
    extracted_dir = args.out / "extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)

    files_to_get = list(METADATA_FILES)
    if not args.skip_step:
        files_to_get.append("step.zip")
    files_to_get.append("assemblies.zip")

    if not args.skip_download:
        print(f"Downloading {len(files_to_get)} files into {raw_dir} ...")
        for name in files_to_get:
            download_file(name, raw_dir)
    else:
        print("Skipping download step (--skip-download).")

    if not args.skip_extract:
        if "assemblies.zip" in files_to_get:
            extract_zip(raw_dir / "assemblies.zip", extracted_dir / "assemblies")
        if "step.zip" in files_to_get:
            extract_zip(raw_dir / "step.zip", extracted_dir / "step")
    else:
        print("Skipping extraction step (--skip-extract).")

    if not args.skip_group:
        step_dir = extracted_dir / "step"
        if not step_dir.exists():
            print(f"\n[!] {step_dir} does not exist -- did you use --skip-step? "
                  "Grouping needs the extracted STEP files. Re-run without "
                  "--skip-step, or point step_dir manually.")
            sys.exit(1)
        group_assemblies(extracted_dir / "assemblies", step_dir, args.out, use_copy=args.copy)
    else:
        print("Skipping grouping step (--skip-group).")

    print("\nAll requested steps complete.")
    print(f"Root: {args.out.resolve()}")


if __name__ == "__main__":
    main()
