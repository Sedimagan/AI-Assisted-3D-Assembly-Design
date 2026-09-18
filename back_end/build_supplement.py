"""
build_supplement.py — library of parts the main part bank cannot hold.

part_bank.py only admits WATERTIGHT meshes (its task-18 quality gate). A helical
spring's swept BSpline surface tessellates with cracks along the seam, so springs
are rejected corpus-wide: a search of all 11,954 bank parts finds nothing within
4% of a spring's size. The raw corpus STEP files still contain them.

This scans the tool-post corpus for spring-like solids, tessellates each one at
wire-scale resolution, and stores them beside the bank. It is NOT a replacement
for the bank -- rebuild.py consults it only for families the bank cannot supply.

Usage:  python build_supplement.py          (rebuilds back_end/data/part_bank_supplement)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent / "Source_3d_models" / "Best_models_for_training" / "Tool_Post"
OUT = HERE / "data" / "part_bank_supplement"

# A spring here: 5-40mm long, 1.5-3.5x longer than wide, and helical (BSpline).
LONG_RANGE = (5.0, 40.0)
ASPECT_RANGE = (1.5, 3.5)
MESH_SIZE = 0.6             # wire diameter is ~1.2mm; 0.6 keeps the coil, 0.45 was 12.7k faces per spring


def corpus_files_naming(token: str):
    """Corpus STEP files whose product list mentions `token` (fast text scan)."""
    files = []
    for p in sorted(CORPUS.rglob("*")):
        if p.suffix.lower() not in (".step", ".stp"):
            continue
        try:
            if token.encode() in p.read_bytes():
                files.append(p)
        except Exception:
            continue
    return files


def extract_springs(path: Path):
    """Tessellate spring-like solids in one assembly. Returns list of dicts."""
    import gmsh
    import trimesh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    out = []
    try:
        gmsh.model.add("s")
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()

        keep, drop = [], []
        for _, t in gmsh.model.getEntities(3):
            try:
                bb = gmsh.model.getBoundingBox(3, t)
                ext = sorted([bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]], reverse=True)
                vol = abs(gmsh.model.occ.getMass(3, t))
                types = {gmsh.model.getType(2, abs(s))
                         for _, s in gmsh.model.getBoundary([(3, t)], oriented=False)}
            except Exception:
                drop.append((3, t)); continue
            aspect = ext[0] / max(ext[2], 1e-9)
            spring_like = (LONG_RANGE[0] <= ext[0] <= LONG_RANGE[1]
                           and ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]
                           and any("Spline" in x for x in types))
            (keep if spring_like else drop).append((3, t))
        if not keep:
            return []
        gmsh.model.occ.remove(drop, recursive=True)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE)
        gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE * 0.3)
        gmsh.model.mesh.generate(2)

        for _, t in gmsh.model.getEntities(3):
            verts, faces, vmap = [], [], {}
            for _, s in gmsh.model.getBoundary([(3, t)], oriented=False):
                s = abs(s)
                try:
                    nt, nc, _ = gmsh.model.mesh.getNodes(2, s, includeBoundary=True)
                    _, en = gmsh.model.mesh.getElementsByType(2, s)
                except Exception:
                    continue
                for ntag, xyz in zip(nt, np.array(nc).reshape(-1, 3)):
                    if ntag not in vmap:
                        vmap[ntag] = len(verts); verts.append(xyz)
                for tri in np.array(en, dtype=int).reshape(-1, 3):
                    if all(int(x) in vmap for x in tri):
                        faces.append([vmap[int(x)] for x in tri])
            if not faces:
                continue
            m = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=False)
            out.append({"vertices": np.asarray(m.vertices, np.float32),
                        "faces": np.asarray(m.faces, np.int64),
                        "volume": float(abs(gmsh.model.occ.getMass(3, t))),
                        "extents": [float(x) for x in m.extents]})
        return out
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = corpus_files_naming("Compress Spring")
    print(f"{len(files)} corpus assemblies name a compression spring")
    index = []
    for f in files:
        asm = f.parent.name if f.parent.name.startswith("Tool_Post") else f.stem
        try:
            springs = extract_springs(f)
        except Exception as exc:
            print(f"  {asm:<16} FAILED: {type(exc).__name__}: {str(exc)[:60]}")
            continue
        if not springs:
            print(f"  {asm:<16} no spring-like solid"); continue
        # one representative per assembly (its springs are copies of one another)
        s = springs[0]
        pid = f"spring_{asm}"
        v = s["vertices"]
        v = v - v.mean(axis=0)                      # centred; placed by the caller
        np.savez_compressed(OUT / f"{pid}.npz", vertices=v, faces=s["faces"],
                            extents=np.array(s["extents"], np.float32))
        index.append({"part_id": pid, "family": "compress_spring", "category": "Tool_Post",
                      "source_assembly": asm, "volume": round(s["volume"], 2),
                      "extents": [round(x, 3) for x in s["extents"]],
                      "n_in_assembly": len(springs)})
        print(f"  {asm:<16} {len(springs)} springs  vol={s['volume']:.1f}  "
              f"ext={[round(x,1) for x in s['extents']]}  tris={len(s['faces'])}")
    json.dump(index, open(OUT / "index.json", "w"), indent=1)
    print(f"\nwrote {len(index)} entries to {OUT}")


if __name__ == "__main__":
    main()
