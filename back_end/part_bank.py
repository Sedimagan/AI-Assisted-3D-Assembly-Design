"""
part_bank.py — Phase 3 part library: build + query.

Every parsed assembly's solid bodies are discarded after dataset.py extracts
their 22-dim node features. This module re-walks the same corpus, keeps the
per-body meshes, canonicalizes them (center, PCA-align, unit-scale) and
voxelizes them at a fixed resolution — persisting one .npz per part plus an
index.json so shape_generator.py can retrieve or condition-generate a
replacement part without re-parsing STEP files at inference time.

Run:
    cd back_end && python part_bank.py
    python part_bank.py --out-dir data/part_bank --timeout 300
"""

from __future__ import annotations

import json
import multiprocessing as _mp
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import trimesh

from dataset import (
    _build_trimesh,
    _compute_sdf_stats,
    _compute_shape_signals,
    _classify_component_type,
    COMP_TYPES,
)

VOXEL_RES = 32
_VOX_PAD = 1.2  # occupancy grid spans [-0.6, 0.6] for a unit-max-extent part


# ── Body extraction (own gmsh pass, mirrors dataset._parse_step's per-body loop) ──

def _extract_bodies_worker(step_path: str, result_queue: "_mp.Queue") -> None:
    status, payload = "error", "unknown error"
    try:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("partbank")
        bodies: list = []
        try:
            gmsh.merge(step_path)
            gmsh.model.occ.synchronize()
            volumes = gmsh.model.occ.getEntities(3)
            if len(volumes) < 1:
                status, payload = "ok", []
            else:
                gmsh.model.occ.fragment(volumes, [])
                gmsh.model.occ.synchronize()
                volumes = gmsh.model.occ.getEntities(3)
                gmsh.model.mesh.generate(2)

                # body_index = position in this enumeration — matches
                # dataset.py's node ordering exactly (same
                # merge/fragment/getEntities(3) sequence on the same file),
                # even though some bodies get skipped below.
                for body_index, (dim, tag) in enumerate(volumes):
                    bbox = gmsh.model.occ.getBoundingBox(dim, tag)
                    vol = gmsh.model.occ.getMass(dim, tag)
                    dx, dy, dz = bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]
                    center = np.array([(bbox[0] + bbox[3]) / 2,
                                       (bbox[1] + bbox[4]) / 2,
                                       (bbox[2] + bbox[5]) / 2])
                    try:
                        com = np.asarray(gmsh.model.occ.getCenterOfMass(dim, tag))
                    except Exception:
                        com = None
                    bnd = gmsh.model.getBoundary([(dim, tag)], oriented=False, combined=True)
                    surf_tags = [abs(s[1]) for s in bnd if s[0] == 2]
                    tm = _build_trimesh(surf_tags)
                    if tm is None or len(tm.faces) < 4:
                        continue
                    exact_sa = float(tm.area)
                    sdf_m, sdf_v = _compute_sdf_stats(tm)
                    signals = _compute_shape_signals(
                        tm, vol, (dx, dy, dz), center, com,
                        face_count=len(surf_tags),
                    )
                    type_idx, _hv, _headv, _notes = _classify_component_type(signals)
                    bodies.append({
                        "vertices": np.asarray(tm.vertices, dtype=np.float32),
                        "faces": np.asarray(tm.faces, dtype=np.int64),
                        "bbox": [float(dx), float(dy), float(dz)],
                        "volume": float(vol),
                        "comp_type_idx": int(type_idx),
                        "body_index": int(body_index),
                    })
                status, payload = "ok", bodies
        finally:
            gmsh.finalize()
    except Exception as exc:
        status, payload = "error", str(exc)

    # Ensure the feeder thread fully flushes into the OS pipe before this
    # (daemon) process exits — otherwise a large payload can be silently
    # lost/truncated and the parent misreads it as a timeout (see
    # extract_bodies_with_timeout's docstring).
    result_queue.put((status, payload))
    result_queue.close()
    result_queue.join_thread()


def extract_bodies_with_timeout(step_path: str, timeout_secs: int = 300) -> tuple:
    """Run body extraction in a child process with a time limit.
    Returns (list[dict], status) where status is "ok" | "timeout" | "error".

    Unlike dataset.py's _parse_step_with_timeout (tiny graph payloads), a
    body-extraction result carries full per-body mesh geometry which can
    exceed the OS pipe buffer — join()-then-get() would deadlock (the child
    blocks flushing the pipe while the parent isn't draining it). Poll the
    queue while the process may still be alive instead.
    """
    ctx = _mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_extract_bodies_worker, args=(step_path, q), daemon=True)
    p.start()

    status, payload = None, None
    deadline = time.time() + timeout_secs
    died_at = None
    grace = 5.0  # seconds to keep draining the queue after the child exits —
                 # its feeder thread may still be flushing a large payload

    while time.time() < deadline:
        try:
            status, payload = q.get(timeout=0.2)
            break
        except Exception:
            pass
        if not p.is_alive():
            if died_at is None:
                died_at = time.time()
            elif time.time() - died_at > grace:
                break

    # We already have a result from the queue — trust it regardless of
    # whether the process has finished its OS-level teardown yet (gmsh
    # finalize / thread cleanup can outlive the put() by a few hundred ms).
    if status is not None:
        if p.is_alive():
            p.join(2)  # brief grace for a clean exit; not fatal either way
        if status == "ok":
            return payload, "ok"
        return [], "error"

    # No result was ever queued — this is a genuine timeout or crash.
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            # Bounded, not unconditional: SIGKILL isn't always instant — a child
            # stuck in a long uninterruptible OpenCASCADE/gmsh C-extension call
            # can take minutes to actually die, and an unconditional join() here
            # would silently re-inflate this "timeout" into a multi-minute hang
            # (confirmed on a real corpus file via dataset.py's identical bug).
            p.join(10)
        return [], "timeout"

    p.join()
    return [], "error"


# ── Canonicalization + voxelization ────────────────────────────────────────────

def canonicalize(vertices: np.ndarray, faces: np.ndarray):
    """Center at centroid, align to OBB principal axes, scale to unit max-extent.
    Returns (verts_canon, faces, center, scale) — center/scale let a caller map
    a canonical part back to the original assembly's coordinate frame."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    center = mesh.centroid.copy()
    mesh.apply_translation(-center)

    try:
        obb = mesh.bounding_box_oriented
        align = np.linalg.inv(obb.primitive.transform)
        align[:3, 3] = 0.0  # rotation/reflection only — already centered above
        mesh.apply_transform(align)
    except Exception:
        pass  # fall back to axis-aligned (no rotation)

    extents = mesh.extents
    scale = float(max(extents.max(), 1e-6))
    mesh.apply_scale(1.0 / scale)
    return mesh.vertices.copy(), mesh.faces.copy(), center, scale


def _center_fit(occ: np.ndarray, res: int) -> np.ndarray:
    """Center-crop or zero-pad `occ` to exactly (res, res, res)."""
    out = np.zeros((res, res, res), dtype=bool)
    slices_out, slices_in = [], []
    for dim in range(3):
        s = occ.shape[dim]
        if s <= res:
            off = (res - s) // 2
            slices_out.append(slice(off, off + s))
            slices_in.append(slice(0, s))
        else:
            off = (s - res) // 2
            slices_out.append(slice(0, res))
            slices_in.append(slice(off, off + res))
    out[tuple(slices_out)] = occ[tuple(slices_in)]
    return out


def voxelize(verts_faces, res: int = VOXEL_RES) -> np.ndarray:
    """Occupancy grid for a unit-max-extent, origin-centered mesh (as produced
    by canonicalize()). Accepts (vertices, faces) or a trimesh.Trimesh."""
    if isinstance(verts_faces, trimesh.Trimesh):
        mesh = verts_faces
    else:
        verts, faces = verts_faces
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    pitch = _VOX_PAD / res
    try:
        vg = mesh.voxelized(pitch=pitch).fill()
        occ = np.asarray(vg.matrix, dtype=bool)
    except Exception:
        try:
            occ = np.asarray(mesh.voxelized(pitch=pitch).matrix, dtype=bool)
        except Exception:
            return np.zeros((res, res, res), dtype=bool)
    return _center_fit(occ, res)


def devoxelize(occ: np.ndarray, threshold: float = 0.5) -> Optional[trimesh.Trimesh]:
    """Marching cubes on an occupancy/probability grid, inverse of voxelize()'s
    coordinate convention (unit max-extent, centered at origin)."""
    from skimage import measure

    if occ.max() <= threshold:
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(occ.astype(np.float32), level=threshold)
    except Exception:
        return None
    res = occ.shape[0]
    verts = (verts / res - 0.5) * _VOX_PAD
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


# ── Bank builder ────────────────────────────────────────────────────────────────

def build_part_bank(
    source_dir: str,
    out_dir: str = "data/part_bank",
    categories: Optional[List[str]] = None,
    timeout_secs: int = 300,
) -> None:
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_exts = {".step", ".stp", ".STEP", ".STP"}
    files = [p for p in source_dir.rglob("*") if p.suffix in step_exts]
    if categories:
        cats = set(categories)
        files = [
            p for p in files
            if p.relative_to(source_dir).parts[0] in cats
        ]
    files = sorted(files)

    print(f"Building part bank from {len(files)} assemblies under {source_dir} …",
          flush=True)

    index: list = []
    part_id_counter = 0
    t0 = time.time()

    for i, sf in enumerate(files, 1):
        rel = sf.relative_to(source_dir)
        category = rel.parts[0]
        assembly = rel.parts[1] if len(rel.parts) > 1 else sf.stem

        ft = time.time()
        bodies, status = extract_bodies_with_timeout(str(sf), timeout_secs)
        elapsed = time.time() - ft
        print(f"[{i}/{len(files)}] {category}/{assembly}: {status} "
              f"({elapsed:.1f}s) — {len(bodies)} bodies", flush=True)

        for b in bodies:
            verts_c, faces_c, center, scale = canonicalize(b["vertices"], b["faces"])
            occ = voxelize((verts_c, faces_c), res=VOXEL_RES)

            part_id = f"part_{part_id_counter:06d}"
            part_id_counter += 1
            np.savez_compressed(
                out_dir / f"{part_id}.npz",
                vertices=verts_c.astype(np.float32),
                faces=faces_c.astype(np.int64),
                voxels=occ,
                bbox=np.array(b["bbox"], dtype=np.float32),
                scale=np.float32(scale),
                center=np.array(center, dtype=np.float32),
            )
            index.append({
                "part_id": part_id,
                "comp_type": COMP_TYPES[b["comp_type_idx"]],
                "comp_type_idx": b["comp_type_idx"],
                "category": category,
                "source_assembly": assembly,
                "body_index": b["body_index"],
                "bbox": b["bbox"],
                "scale": scale,
            })

    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=1)

    total_elapsed = time.time() - t0
    print(f"\nWrote {len(index)} parts from {len(files)} assemblies "
          f"in {total_elapsed:.0f}s", flush=True)
    print(f"Wrote {out_dir / 'index.json'}", flush=True)
    print("DONE", flush=True)


# ── Query interface ─────────────────────────────────────────────────────────────

class PartHit:
    def __init__(self, part_id: str, comp_type: str, category: str, fit_score: float):
        self.part_id = part_id
        self.comp_type = comp_type
        self.category = category
        self.fit_score = fit_score

    def __repr__(self):
        return (f"PartHit(part_id={self.part_id!r}, comp_type={self.comp_type!r}, "
                f"category={self.category!r}, fit_score={self.fit_score:.3f})")


class PartBank:
    """Loads data/part_bank/index.json lazily; query() ranks candidates by
    affine-invariant bbox-aspect-ratio similarity to a target shape."""

    def __init__(self, bank_dir: str = "data/part_bank"):
        self.bank_dir = Path(bank_dir)
        self._index: Optional[list] = None

    @property
    def index(self) -> list:
        if self._index is None:
            idx_path = self.bank_dir / "index.json"
            if idx_path.exists():
                with open(idx_path) as f:
                    self._index = json.load(f)
            else:
                self._index = []
        return self._index

    def __len__(self) -> int:
        return len(self.index)

    def query(
        self,
        comp_type: Optional[str],
        category: Optional[str],
        target_extents,
        top_k: int = 5,
        exclude_assemblies: Optional[set] = None,
    ) -> List[PartHit]:
        target_sorted = np.sort(np.asarray(target_extents, dtype=np.float64))
        target_ratio = target_sorted / (target_sorted[-1] + 1e-9)

        candidates = self.index
        if exclude_assemblies:
            candidates = [e for e in candidates
                          if e["source_assembly"] not in exclude_assemblies]
        if comp_type is not None:
            same_type = [e for e in candidates if e["comp_type"] == comp_type]
            if same_type:
                candidates = same_type
        if category is not None:
            same_cat = [e for e in candidates if e["category"] == category]
            if same_cat:
                candidates = same_cat

        scored = []
        for e in candidates:
            bbox = np.sort(np.asarray(e["bbox"], dtype=np.float64))
            ratio = bbox / (bbox[-1] + 1e-9)
            dist = float(np.linalg.norm(ratio - target_ratio))
            fit_score = float(np.exp(-2.0 * dist))  # 1.0 = identical shape ratios
            scored.append((fit_score, e))

        scored.sort(key=lambda t: -t[0])
        return [PartHit(e["part_id"], e["comp_type"], e["category"], fit_score)
                for fit_score, e in scored[:top_k]]

    def load_mesh(self, part_id: str) -> trimesh.Trimesh:
        data = np.load(self.bank_dir / f"{part_id}.npz")
        return trimesh.Trimesh(vertices=data["vertices"], faces=data["faces"], process=False)

    def load_voxels(self, part_id: str) -> np.ndarray:
        data = np.load(self.bank_dir / f"{part_id}.npz")
        return data["voxels"]

    def get_entry(self, part_id: str) -> dict:
        for e in self.index:
            if e["part_id"] == part_id:
                return e
        raise KeyError(part_id)

    def _by_source(self) -> dict:
        """Lazily built (category, source_assembly, body_index) -> entry map,
        for aligning a graph's node index back to its part-bank part during
        leave-one-out training (see train_shape_gen.py)."""
        if not hasattr(self, "_source_map") or self._source_map is None:
            self._source_map = {
                (e["category"], e["source_assembly"], e["body_index"]): e
                for e in self.index
            }
        return self._source_map

    def find_by_source(self, category: str, source_assembly: str,
                        body_index: int) -> Optional[dict]:
        return self._by_source().get((category, source_assembly, body_index))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir",
                    default="../Source_3d_models/Best_models_for_training")
    ap.add_argument("--out-dir", default="data/part_bank")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    build_part_bank(args.source_dir, args.out_dir, args.categories, args.timeout)
