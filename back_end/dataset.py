"""
dataset.py — AssemblyDataset
Loads STEP files from source_dir, converts each assembly to an attributed
graph (nodes = solid bodies, edges = shared surfaces / mate constraints),
and caches the processed PyG Data objects.

Pipeline per body
-----------------
  gmsh (OpenCASCADE)  → parse solids, fragment(), getBoundingBox(), getMass()
  trimesh             → exact surface area  (replaces bbox approximation)
  SDF ray-casting     → Shape Diameter Function mean + variance
                         used to infer geometry-driven component type
                         (approximates CGAL SDF via trimesh inward ray casting)

Node feature vector: 16-dim
  [0:8]  component-type one-hot  (geometry-driven, 8 classes)
  [8]    normalised volume
  [9]    exact surface area       (trimesh, normalised)
  [10]   bbox Δx / bbox_max
  [11]   bbox Δy / bbox_max
  [12]   bbox Δz / bbox_max
  [13]   SDF mean  (normalised)
  [14]   SDF variance (normalised)
  [15]   SA/V ratio (normalised)
"""

from __future__ import annotations

import json
import multiprocessing as _mp
import random
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import RandomLinkSplit

# 8-class component type vocabulary (index used for one-hot encoding)
COMP_TYPES = ["body", "fastener", "bearing", "shaft", "plate", "housing", "gear", "other"]
MATE_TYPES = ["coincident", "concentric", "parallel", "tangent", "fixed", "other"]


# ── Trimesh / SDF helpers ─────────────────────────────────────────────────────

def _build_trimesh(surf_tags: list) -> Optional["trimesh.Trimesh"]:
    """
    Extract a trimesh.Trimesh from the current gmsh surface mesh for the
    given list of surface (dim=2) tags.  Requires gmsh.model.mesh.generate(2)
    to have been called first.
    """
    try:
        import trimesh as _tm
        import gmsh as _gmsh
    except ImportError:
        return None

    vertices: list = []
    faces:    list = []
    node_map: dict = {}

    for stag in surf_tags:
        try:
            elem_types, _, ntags_per = _gmsh.model.mesh.getElements(2, stag)
        except Exception:
            continue
        for etype, ntags in zip(elem_types, ntags_per):
            if etype != 2:          # only 3-node triangles
                continue
            arr = np.array(ntags, dtype=np.int64).reshape(-1, 3)
            for tri in arr:
                fvids = []
                for nid in tri:
                    if nid not in node_map:
                        coords = _gmsh.model.mesh.getNode(nid)[0]
                        node_map[nid] = len(vertices)
                        vertices.append(coords[:3])
                    fvids.append(node_map[nid])
                faces.append(fvids)

    if not vertices or not faces:
        return None

    return _tm.Trimesh(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces,    dtype=np.int64),
        process=True,       # fix normals / degenerate faces for reliable ray casting
    )


def _compute_sdf_stats(mesh: "trimesh.Trimesh",
                       n_samples: int = 300) -> Tuple[float, float]:
    """
    Approximate the Shape Diameter Function (SDF) via inward ray casting —
    equivalent to CGAL's sdf() but implemented purely with trimesh.

    For each sampled surface point we shoot a ray along the *inward* normal
    and record the distance to the first back-face hit.  The distribution of
    those distances is the SDF:
      • mean  → average local thickness
      • variance → shape complexity (shafts: low; housings/gears: high)
    """
    try:
        import trimesh as _tm
    except ImportError:
        return 0.0, 0.0

    if mesh is None or len(mesh.faces) < 4:
        return 0.0, 0.0

    n_samples = min(n_samples, max(30, len(mesh.faces)))
    pts, face_ids = _tm.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_ids]

    # Nudge start points slightly *outside* the surface to avoid self-hit
    origins    = pts + normals * 1e-5
    directions = -normals               # pointing inward

    try:
        locs, ray_ids, _ = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False,
        )
    except Exception:
        return 0.0, 0.0

    if len(locs) == 0:
        return 0.0, 0.0

    dists = np.linalg.norm(locs - pts[ray_ids], axis=1)
    dists = dists[dists > 1e-9]        # discard degenerate near-zero hits

    if len(dists) < 3:
        return 0.0, 0.0

    return float(dists.mean()), float(dists.var())


def _infer_type_from_geometry(
    vol:      float,
    exact_sa: float,
    bbox:     Tuple[float, float, float],
    sdf_mean: float,
    sdf_var:  float,
) -> int:
    """
    Return a COMP_TYPES index using SDF statistics + bounding-box ratios.

    Rules (all comparisons are dimensionless ratios — unit-independent):
      high mean, low var   → shaft (index 3)
      small + elongated    → fastener (index 1)
      very flat            → plate (index 4)
      bimodal SDF (ring)   → bearing (index 2)
      high variance        → housing (index 5)
      default              → body (index 0)
    """
    dx, dy, dz = bbox
    ext        = sorted([dx, dy, dz])          # [shortest, middle, longest]
    elongation = ext[2] / (ext[1] + 1e-9)      # > 3.5 → shaft-like
    flatness   = ext[0] / (ext[2] + 1e-9)      # < 0.12 → plate-like

    # Shaft: strongly elongated, uniform cross-section, thick walls
    if elongation > 3.5 and flatness > 0.05 and sdf_mean > 0.05 * ext[2]:
        return 3   # shaft

    # Fastener: elongated but thin  (bolt, screw, pin)
    if elongation > 2.5 and sdf_mean < 0.08 * ext[2]:
        return 1   # fastener

    # Plate: very flat geometry
    if flatness < 0.12:
        return 4   # plate

    # Bearing: ring topology → bimodal SDF (std > 60 % of mean)
    if sdf_mean > 1e-9 and sdf_var ** 0.5 > 0.6 * sdf_mean:
        return 2   # bearing

    # Housing / gear: complex geometry → elevated variance
    if sdf_mean > 1e-9 and sdf_var > 0.2 * sdf_mean ** 2:
        return 5   # housing

    return 0  # body (generic fallback)


# ── Size-filter sentinels ─────────────────────────────────────────────────────

class _SkipTooManyNodes(Exception):
    """Raised when a STEP file has more bodies than the configured threshold."""

class _SkipTooManyEdges(Exception):
    """Raised when a STEP file has more contacts than the configured threshold."""


# ── STEP file parser ──────────────────────────────────────────────────────────

def _parse_step(
    step_path: str,
    max_nodes: int = 9999,
    max_edges: int = 9999,
) -> Optional[Data]:
    """
    Parse a single STEP file into a PyG Data object.

    Node features  (16-dim):
        [0:8]  component-type one-hot  (geometry-driven via SDF, 8 classes)
        [8]    normalised volume
        [9]    exact surface area       (trimesh, replaces bbox approximation)
        [10]   bbox Δx / bbox_max
        [11]   bbox Δy / bbox_max
        [12]   bbox Δz / bbox_max
        [13]   SDF mean  / sdf_mean_max
        [14]   SDF variance / sdf_var_max
        [15]   SA/V ratio / sav_max

    Edge features  (2-dim):
        [0]    mate type encoded  (0=coincident … 5=other, normalised to [0,1])
        [1]    weight             (1.0 for detected contacts)
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("assembly")

    try:
        gmsh.merge(step_path)
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.occ.getEntities(3)
        if len(volumes) < 2:
            return None

        # ── Pre-check: node count (fast — before expensive fragment()) ────
        if len(volumes) > max_nodes:
            raise _SkipTooManyNodes(
                f"{len(volumes)} bodies (threshold {max_nodes})"
            )

        # Fragment volumes to share boundary surface tags → contact detection
        gmsh.model.occ.fragment(volumes, [])
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.occ.getEntities(3)

        # ── Surface mesh for trimesh enrichment ───────────────────────────
        _trimesh_ok = False
        try:
            import trimesh as _tm        # noqa: F401
            gmsh.model.mesh.generate(2)  # triangulate all surfaces
            _trimesh_ok = True
        except Exception:
            pass

        # ── Raw geometry pass ─────────────────────────────────────────────
        raw_vols:   list = []
        bboxes:     list = []
        body_surfs: List[frozenset] = []

        for dim, tag in volumes:
            bbox = gmsh.model.occ.getBoundingBox(dim, tag)
            vol  = gmsh.model.occ.getMass(dim, tag)
            dx   = bbox[3] - bbox[0]
            dy   = bbox[4] - bbox[1]
            dz   = bbox[5] - bbox[2]
            raw_vols.append(vol)
            bboxes.append((dx, dy, dz))
            bnd = gmsh.model.getBoundary([(dim, tag)], oriented=False, combined=True)
            body_surfs.append(frozenset(abs(s[1]) for s in bnd if s[0] == 2))

        # ── Pre-check: edge count (before expensive trimesh+SDF) ─────────
        _n_dir_edges = sum(
            2 for i in range(len(volumes))
            for j in range(i + 1, len(volumes))
            if body_surfs[i] & body_surfs[j]
        )
        if _n_dir_edges > max_edges:
            raise _SkipTooManyEdges(
                f"{_n_dir_edges} directed edges / {_n_dir_edges // 2} contacts "
                f"(threshold {max_edges})"
            )

        # ── Trimesh enrichment: exact SA + SDF per body ───────────────────
        exact_sas: list = []
        sdf_means: list = []
        sdf_vars:  list = []

        for i, (dim, tag) in enumerate(volumes):
            dx, dy, dz = bboxes[i]
            bbox_sa = 2.0 * (dx * dy + dy * dz + dz * dx)   # fallback

            if _trimesh_ok:
                tm = _build_trimesh(list(body_surfs[i]))
                if tm is not None and len(tm.faces) >= 4:
                    exact_sa      = float(tm.area)
                    sdf_m, sdf_v  = _compute_sdf_stats(tm)
                else:
                    exact_sa, sdf_m, sdf_v = bbox_sa, 0.0, 0.0
            else:
                exact_sa, sdf_m, sdf_v = bbox_sa, 0.0, 0.0

            exact_sas.append(exact_sa)
            sdf_means.append(sdf_m)
            sdf_vars.append(sdf_v)

        # ── Normalisation constants ────────────────────────────────────────
        n         = len(volumes)
        vol_max   = max(raw_vols)  or 1.0
        sa_max    = max(exact_sas) or 1.0
        bbox_max  = max(max(b) for b in bboxes) or 1.0
        sdf_m_max = max(sdf_means) or 1.0
        sdf_v_max = max(sdf_vars)  or 1.0
        sav_vals  = [exact_sas[i] / (raw_vols[i] + 1e-9) for i in range(n)]
        sav_max   = max(sav_vals)  or 1.0

        # ── 16-dim node feature vectors ───────────────────────────────────
        node_feats: List[List[float]] = []

        for i in range(n):
            dx, dy, dz = bboxes[i]
            type_idx = _infer_type_from_geometry(
                raw_vols[i], exact_sas[i], bboxes[i], sdf_means[i], sdf_vars[i]
            )
            type_oh      = [0.0] * 8
            type_oh[type_idx] = 1.0

            feat = (
                type_oh                                     # [0:8]
                + [raw_vols[i]  / vol_max,                  # [8]   volume
                   exact_sas[i] / sa_max,                   # [9]   exact SA
                   dx / bbox_max,                           # [10]  bbox Δx
                   dy / bbox_max,                           # [11]  bbox Δy
                   dz / bbox_max,                           # [12]  bbox Δz
                   sdf_means[i] / sdf_m_max,                # [13]  SDF mean
                   sdf_vars[i]  / sdf_v_max,                # [14]  SDF variance
                   sav_vals[i]  / sav_max]                  # [15]  SA/V ratio
            )
            node_feats.append(feat)

        # ── Edge detection (shared bounding surface = mate contact) ────────
        src, dst, eattr = [], [], []

        for i in range(n):
            for j in range(i + 1, n):
                if body_surfs[i] & body_surfs[j]:
                    src   += [i, j];  dst   += [j, i]
                    eattr += [[0.0, 1.0], [0.0, 1.0]]   # coincident contact

        if not src:
            # No shared surfaces → fully connect as fallback
            for i in range(n):
                for j in range(n):
                    if i != j:
                        src.append(i);  dst.append(j)
                        eattr.append([1.0, 1.0])        # "other" contact

        x          = torch.tensor(node_feats, dtype=torch.float)
        edge_index = torch.tensor([src, dst],  dtype=torch.long)
        edge_attr  = torch.tensor(eattr,        dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    except (_SkipTooManyNodes, _SkipTooManyEdges):
        raise   # propagate up to worker — these are handled, not errors
    except Exception as exc:
        print(f"    [skip] {Path(step_path).name}: {exc}")
        return None

    finally:
        gmsh.finalize()


# ── Timeout wrapper for _parse_step ──────────────────────────────────────────

def _parse_step_worker(
    step_path: str,
    result_queue: "_mp.Queue",
    max_nodes: int,
    max_edges: int,
) -> None:
    """Worker target: parse one STEP file and push result onto the queue."""
    try:
        import io
        result = _parse_step(step_path, max_nodes=max_nodes, max_edges=max_edges)
        if result is not None:
            buf = io.BytesIO()
            torch.save(result, buf)
            result_queue.put(("ok", buf.getvalue()))
        else:
            result_queue.put(("ok", None))
    except _SkipTooManyNodes as exc:
        result_queue.put(("too_many_nodes", str(exc)))
    except _SkipTooManyEdges as exc:
        result_queue.put(("too_many_edges", str(exc)))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


def _parse_step_with_timeout(
    step_path: str,
    timeout_secs: int = 120,
    max_nodes:    int = 20,
    max_edges:    int = 60,
) -> tuple:
    """
    Run _parse_step in a child process with size and time limits.

    Returns
    -------
    (Data | None, status)
      status: "ok" | "timeout" | "too_many_nodes" | "too_many_edges" | "error"
    """
    ctx = _mp.get_context("spawn")   # fresh interpreter — avoids gmsh state leaks
    q   = ctx.Queue()
    p   = ctx.Process(
        target=_parse_step_worker,
        args=(step_path, q, max_nodes, max_edges),
        daemon=True,
    )
    p.start()
    p.join(timeout=timeout_secs)

    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        return None, "timeout"

    if not q.empty():
        status, payload = q.get_nowait()
        if status == "ok":
            if payload is None:
                return None, "ok"
            import io
            data = torch.load(io.BytesIO(payload), weights_only=False)
            return data, "ok"
        return None, status   # "too_many_nodes" | "too_many_edges" | "error"

    return None, "error"


# ── Synthetic data fallback ───────────────────────────────────────────────────

def _synthetic_graph(n_nodes: int = None) -> Data:
    """Generate one random 16-dim assembly graph for testing."""
    rng      = random.Random()
    n        = n_nodes or rng.randint(4, 20)
    node_dim = 16

    x = torch.zeros(n, node_dim)
    for i in range(n):
        t = rng.randint(0, 7)
        x[i, t] = 1.0           # type one-hot
        x[i, 8:] = torch.rand(8)  # geometry + SDF features

    src, dst, eattr = [], [], []
    perm = list(range(n)); rng.shuffle(perm)
    for k in range(n - 1):
        i, j = perm[k], perm[k + 1]
        mt = rng.randint(0, 5) / 5.0
        src += [i, j]; dst += [j, i]
        eattr += [[mt, 1.0], [mt, 1.0]]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(eattr,      dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _generate_synthetic(n: int = 300) -> List[Data]:
    print(f"  Generating {n} synthetic 16-dim assembly graphs for testing…")
    return [_synthetic_graph() for _ in range(n)]


# ── Main dataset class ────────────────────────────────────────────────────────

class AssemblyDataset(InMemoryDataset):
    """
    PyG InMemoryDataset wrapping either real STEP files or synthetic graphs.

    Usage
    -----
    ds = AssemblyDataset(source_dir="...", processed_dir="data/processed")
    """

    def __init__(
        self,
        source_dir: str,
        processed_dir: str,
        force_reload: bool = False,
        transform=None,
        pre_transform=None,
    ):
        self.source_dir = Path(source_dir)
        if force_reload:
            for fname in ["data.pt", "processed/data.pt"]:
                proc = Path(processed_dir) / fname
                if proc.exists():
                    proc.unlink()
        super().__init__(str(processed_dir), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0],
                                            weights_only=False)

    @property
    def raw_file_names(self): return []

    @property
    def processed_file_names(self): return ["data.pt"]

    def download(self): pass

    def process(self):
        import re as _re
        _UUID_RE = _re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            _re.IGNORECASE,
        )
        step_exts  = {".step", ".stp", ".STEP", ".STP"}
        step_files = [
            p for p in self.source_dir.rglob("*")
            if p.suffix in step_exts and not _UUID_RE.match(p.stem)
        ]

        graphs:       List[Data] = []
        source_paths: List[str]  = []
        skipped:      List[dict] = []

        # ── Thresholds ────────────────────────────────────────────────────
        _TIMEOUT   = 120   # seconds per file (was 300)
        _MAX_NODES = 20    # bodies — M1-friendly limit (74% of Fusion360 pass)
        _MAX_EDGES = 60    # directed edges (≈ 30 contacts) — 3× dataset mean

        _skipped_dir       = self.source_dir / "skipped_models"
        _skip_nodes_dir    = _skipped_dir / f"nodes_gt_{_MAX_NODES}"
        _skip_edges_dir    = _skipped_dir / f"edges_gt_{_MAX_EDGES}"
        _skip_timeout_dir  = _skipped_dir / "timeout"

        def _move_folder(sf: Path, dest_dir: Path) -> Optional[str]:
            """Move sf's parent folder into dest_dir; return dest path or None."""
            if sf.parent == self.source_dir:
                return None   # file sits directly in source_dir — don't move
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / sf.parent.name
            if not dest.exists():
                shutil.move(str(sf.parent), str(dest))
                return str(dest)
            return str(dest)

        if step_files:
            print(f"  Found {len(step_files)} STEP file(s) in {self.source_dir}",
                  flush=True)
            print(f"  Thresholds — nodes ≤ {_MAX_NODES}  edges ≤ {_MAX_EDGES}"
                  f"  timeout {_TIMEOUT}s", flush=True)

            for idx, sf in enumerate(sorted(step_files), 1):
                print(f"  [{idx}/{len(step_files)}] Parsing: {sf.name}", flush=True)
                t0 = time.time()
                g, status = _parse_step_with_timeout(
                    str(sf),
                    timeout_secs=_TIMEOUT,
                    max_nodes=_MAX_NODES,
                    max_edges=_MAX_EDGES,
                )
                elapsed = round(time.time() - t0, 1)

                if status == "too_many_nodes":
                    print(f"  [SKIP-NODES] {sf.name} — nodes > {_MAX_NODES} "
                          f"({elapsed}s)", flush=True)
                    entry = {"file": str(sf), "folder": str(sf.parent),
                             "reason": f"nodes > {_MAX_NODES}", "elapsed": elapsed}
                    dest = _move_folder(sf, _skip_nodes_dir)
                    if dest:
                        entry["moved_to"] = dest
                    skipped.append(entry)
                    continue

                if status == "too_many_edges":
                    print(f"  [SKIP-EDGES] {sf.name} — edges > {_MAX_EDGES} "
                          f"({elapsed}s)", flush=True)
                    entry = {"file": str(sf), "folder": str(sf.parent),
                             "reason": f"edges > {_MAX_EDGES}", "elapsed": elapsed}
                    dest = _move_folder(sf, _skip_edges_dir)
                    if dest:
                        entry["moved_to"] = dest
                    skipped.append(entry)
                    continue

                if status == "timeout":
                    print(f"  [TIMEOUT]    {sf.name} — exceeded {_TIMEOUT}s "
                          f"({elapsed}s)", flush=True)
                    entry = {"file": str(sf), "folder": str(sf.parent),
                             "reason": f"timeout > {_TIMEOUT}s", "elapsed": elapsed}
                    dest = _move_folder(sf, _skip_timeout_dir)
                    if dest:
                        entry["moved_to"] = dest
                    skipped.append(entry)
                    continue

                if status == "error":
                    print(f"  [ERROR]      {sf.name} — parse failed", flush=True)
                    continue

                if g is not None and g.num_nodes >= 2:
                    graphs.append(g)
                    source_paths.append(str(sf))
                    print(f"  [OK]  {sf.name} — {g.num_nodes} nodes  "
                          f"{g.edge_index.size(1)//2} edges  ({elapsed}s)", flush=True)

            n_nodes_skip = sum(1 for e in skipped if f"nodes >"  in e["reason"])
            n_edges_skip = sum(1 for e in skipped if f"edges >"  in e["reason"])
            n_time_skip  = sum(1 for e in skipped if "timeout"   in e["reason"])
            print(
                f"  Parsed {len(graphs)} valid  |  "
                f"Skipped {len(skipped)} total "
                f"(nodes>{_MAX_NODES}: {n_nodes_skip}  "
                f"edges>{_MAX_EDGES}: {n_edges_skip}  "
                f"timeout: {n_time_skip})",
                flush=True,
            )

            # ── Write skipped report ──────────────────────────────────────
            if skipped:
                report = {
                    "source_dir":        str(self.source_dir),
                    "thresholds": {
                        "max_nodes":    _MAX_NODES,
                        "max_edges":    _MAX_EDGES,
                        "timeout_secs": _TIMEOUT,
                    },
                    "total_found":       len(step_files),
                    "total_parsed":      len(graphs),
                    "total_skipped":     len(skipped),
                    "skipped_breakdown": {
                        f"nodes_gt_{_MAX_NODES}": n_nodes_skip,
                        f"edges_gt_{_MAX_EDGES}": n_edges_skip,
                        "timeout":                n_time_skip,
                    },
                    "skipped_folders": {
                        f"nodes_gt_{_MAX_NODES}": str(_skip_nodes_dir),
                        f"edges_gt_{_MAX_EDGES}": str(_skip_edges_dir),
                        "timeout":                str(_skip_timeout_dir),
                    },
                    "entries": skipped,
                }
                report_path = self.source_dir / "skipped_models_report.json"
                report_path.write_text(json.dumps(report, indent=2))
                print(f"  Skipped report → {report_path}", flush=True)
        else:
            print(f"  No STEP files found in {self.source_dir}.", flush=True)

        if len(graphs) == 0:
            raise RuntimeError(
                "No valid multi-body assembly graphs found in source directory. "
                "Check that Source_3d_models/ contains STEP files with ≥2 solid bodies."
            )
        print(f"  Using {len(graphs)} real assembly graph(s) — no synthetic data.")

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

        sources_file = Path(self.processed_paths[0]).parent / "sources.json"
        sources_file.write_text(json.dumps(source_paths, indent=2))
        print(f"  Saved {len(graphs)} graphs → {self.processed_paths[0]}")
        print(f"  Saved source paths    → {sources_file}")


# ── Split helper ──────────────────────────────────────────────────────────────

def get_splits(dataset: AssemblyDataset, cfg: dict):
    """
    Split the dataset into train / val / test lists, then apply
    RandomLinkSplit to each individual graph.

    RandomLinkSplit must be applied per-graph (it expects a Data object,
    not an InMemoryDataset).  Graphs with fewer than 8 directed edges are
    skipped to ensure the split can always allocate at least one val/test edge.
    """
    n       = len(dataset)
    n_test  = max(1, int(n * cfg["data"]["test_ratio"]))
    n_val   = max(1, int(n * cfg["data"]["val_ratio"]))
    n_train = max(1, n - n_val - n_test)

    torch.manual_seed(42)
    perm = torch.randperm(n).tolist()

    splitter = RandomLinkSplit(
        num_val                    = 0.1,
        num_test                   = 0.1,
        is_undirected              = True,
        add_negative_train_samples = True,
        neg_sampling_ratio         = cfg["training"]["neg_ratio"],
    )

    def _transform(indices: list, split_idx: int) -> List[Data]:
        result = []
        for i in indices:
            data = dataset[i]
            if data.edge_index.size(1) < 8:
                continue
            try:
                train_d, val_d, test_d = splitter(data)
                result.append([train_d, val_d, test_d][split_idx])
            except Exception:
                continue
        return result

    train_data = _transform(perm[:n_train],              0)
    val_data   = _transform(perm[n_train:n_train+n_val], 1)
    test_data  = _transform(perm[n_train+n_val:],        2)

    print(f"  Splits — train: {len(train_data)}  val: {len(val_data)}"
          f"  test: {len(test_data)}  (skipped graphs with <8 edges)")
    return train_data, val_data, test_data
