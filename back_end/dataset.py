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

Node feature vector: 22-dim
  [0:8]  component-type one-hot  (geometry-driven, 8 classes: long_shaft,
         short_shaft, thick_plate, thin_plate, bolt, washer, nut, body —
         see _classify_component_type())
  [8]    log1p(volume), clipped at 13.8
  [9]    log1p(surface_area), clipped at 11.5
  [10]   bbox Δx / bbox_max       (normalised width)
  [11]   bbox Δy / bbox_max       (normalised height)
  [12]   bbox Δz / bbox_max       (normalised depth)
  [13]   elongation               (longest / mid bbox dim — affine-invariant)
  [14]   flatness                 (shortest / longest bbox dim — affine-invariant)
  [15]   aspect x/y               (normalised — affine-invariant)
  [16]   aspect y/z               (normalised — affine-invariant)
  [17]   sphericity               (π^(1/3) (6V)^(2/3) / SA — affine-invariant)
  [18]   SDF mean  (normalised)
  [19]   SDF variance (normalised)
  [20]   SA/V ratio (normalised)
  [21]   log1p(n_holes)

Edge feature vector: 6-dim
  [0]    mate type encoded  (0=coincident … 5=other, normalised to [0,1])
  [1]    weight             (1.0 for detected contacts)
  [2:6]  joint type one-hot (rigid, revolute, slider, cylindrical)
"""

from __future__ import annotations

import json
import math
import multiprocessing as _mp
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import RandomLinkSplit

# 8-class component type vocabulary (index used for one-hot encoding).
# Proposed taxonomy (multi-signal geometric classifier, replaces the older
# body/fastener/bearing/shaft/plate/housing/gear/other SDF-rule scheme) —
# see _classify_component_type() below.
COMP_TYPES = ["long_shaft", "short_shaft", "thick_plate", "thin_plate",
              "bolt", "washer", "nut", "body"]
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


# ── Multi-signal component-type classifier (proposed taxonomy) ────────────────
#
# Ported from audit_component_types.classify_new() / its threshold dict `T` —
# dataset.py is the canonical home now; audit_component_types.py imports these
# back for its own threshold-tuning/regression reports. Thresholds are
# unvalidated starting points (see docs/ discussion) — a full-corpus audit
# (2026-07-31) showed ~39.5% of bodies as ambiguous (vote conflicts); revisit
# via audit_component_types.py if classification quality looks off in practice.
_TYPE_THRESHOLDS = {
    "hull_hole":        0.80,   # V/V_hull below this → hole vote
    "washer_flat":      0.20,   # s/l — washer must be flatter than this
    "washer_plan":      0.75,   # m/l — washer plan must be near-round
    "nut_elong":        1.80,   # l/m — nut must be compact
    "nut_flat_min":     0.12,   # s/l — thinner than this + ring → washer, not nut
    "round_min":        0.60,   # s/m — round cross-section gate (bolt/shaft)
    "bolt_elong_min":   2.0,
    "bolt_elong_max":   8.0,
    "head_fill":        0.65,   # bounding-cylinder fill below this → head vote
    "head_com":         0.06,   # |COM offset|/l above this → head vote
    "head_faces":       8,      # B-Rep face count at/above this → head vote
    "long_shaft_elong": 5.0,    # l/m at/above this → Long Shaft (else Short)
    "thin_flat":        0.08,   # s/l — Thin vs Thick Plate split
    "plate_flat":       0.30,   # s/l — Thick Plate upper bound
    "plate_plan":       0.30,   # m/l — plates must not be strongly elongated in plan
}


def _compute_shape_signals(
    tm:          Optional["trimesh.Trimesh"],
    vol:         float,
    bbox:        Tuple[float, float, float],
    center:      "np.ndarray",
    com:         Optional["np.ndarray"],
    face_count:  int,
) -> dict:
    """
    Compute the richer per-body signal set consumed by
    _classify_component_type(): OBB-aligned extents (AABB fallback), a
    convex-hull volume ratio and a central-axis ray-cast (through-hole
    votes), bounding-cylinder fill and COM offset along the long axis
    (bolt-head votes), and face count.

    `bbox` = (dx, dy, dz) AABB extents. `center` = AABB world-center [x,y,z]
    (used as ray/COM-offset origin if OBB alignment fails or isn't available).
    `com` = world center-of-mass, or None. `tm` may be None (AABB-only path).
    """
    dx, dy, dz = bbox
    ext    = sorted([dx, dy, dz])
    center = np.asarray(center, dtype=float)
    hull_ratio = None
    ray_hits   = None

    order  = np.argsort([dx, dy, dz])
    eye    = np.eye(3)
    axis_s = eye[order[0]]
    axis_l = eye[order[2]]

    if tm is not None and len(tm.faces) >= 4:
        try:
            obb   = tm.bounding_box_oriented
            e     = np.asarray(obb.primitive.extents, dtype=float)
            R     = np.asarray(obb.primitive.transform)[:3, :3]
            oorder = np.argsort(e)
            ext    = [float(e[oorder[0]]), float(e[oorder[1]]), float(e[oorder[2]])]
            axis_s = R[:, oorder[0]]
            axis_l = R[:, oorder[2]]
            center = np.asarray(obb.primitive.transform)[:3, 3]
        except Exception:
            pass
        try:
            hull_ratio = min(1.5, float(vol / (tm.convex_hull.volume + 1e-12)))
        except Exception:
            hull_ratio = None
        try:
            origin = center - axis_s * (ext[2] * 2.0)
            locs, _, _ = tm.ray.intersects_location(
                np.asarray([origin]), np.asarray([axis_s]))
            ray_hits = int(len(locs))
        except Exception:
            ray_hits = None

    s_, m_, l_ = ext
    cyl_fill = (float(vol / (math.pi * (m_ / 2.0) ** 2 * l_ + 1e-12))
                if l_ > 0 else None)
    com_offset = None
    if com is not None:
        com_offset = abs(float(np.dot(np.asarray(com) - center, axis_l))) / (l_ + 1e-9)

    return {
        "ext": ext, "hull_ratio": hull_ratio, "ray_hits": ray_hits,
        "cyl_fill": cyl_fill, "com_offset": com_offset, "face_count": face_count,
    }


def _classify_component_type(signals: dict) -> Tuple[int, int, int, list]:
    """
    Multi-signal geometric classifier over the proposed 8-class taxonomy
    (COMP_TYPES). Ported from audit_component_types.classify_new() — see
    that module's docstring for the full signal rationale.

    Returns (type_idx, hole_votes, head_votes, notes) — most callers only
    need type_idx; audit_component_types.py uses the vote/notes detail for
    its ambiguity reporting.
    """
    T = _TYPE_THRESHOLDS
    eps = 1e-9
    s, m, l = signals["ext"]
    elong = l / (m + eps)
    flat  = s / (l + eps)
    rnd   = s / (m + eps)
    plan  = m / (l + eps)
    notes: list = []

    hv = 0
    if signals["hull_ratio"] is not None and signals["hull_ratio"] < T["hull_hole"]:
        hv += 1
    if signals["ray_hits"] == 0:
        hv += 1
    has_hole = hv >= 2
    if hv == 1:
        notes.append("hole-votes-split")

    if has_hole and flat < T["washer_flat"] and plan > T["washer_plan"]:
        return COMP_TYPES.index("washer"), hv, 0, notes
    if has_hole and elong < T["nut_elong"] and flat >= T["nut_flat_min"]:
        return COMP_TYPES.index("nut"), hv, 0, notes

    headv = 0
    if signals["cyl_fill"] is not None and signals["cyl_fill"] < T["head_fill"]:
        headv += 1
    if signals["com_offset"] is not None and signals["com_offset"] > T["head_com"]:
        headv += 1
    if signals["face_count"] >= T["head_faces"]:
        headv += 1

    if rnd > T["round_min"] and elong >= T["bolt_elong_min"]:
        if elong <= T["bolt_elong_max"] and headv >= 2:
            return COMP_TYPES.index("bolt"), hv, headv, notes
        if headv == 1:
            notes.append("head-votes-split")
        if elong >= T["long_shaft_elong"]:
            return COMP_TYPES.index("long_shaft"), hv, headv, notes
        return COMP_TYPES.index("short_shaft"), hv, headv, notes

    if flat < T["thin_flat"] and plan > T["plate_plan"]:
        return COMP_TYPES.index("thin_plate"), hv, headv, notes
    if flat < T["plate_flat"] and plan > T["plate_plan"]:
        return COMP_TYPES.index("thick_plate"), hv, headv, notes
    return COMP_TYPES.index("body"), hv, headv, notes


# ── Log-normalisation helper ──────────────────────────────────────────────────

def _safe_log(x):
    """log1p transform for volume / surface-area features."""
    return math.log1p(max(0.0, x))


# ── STEP file parser ──────────────────────────────────────────────────────────

def _parse_step(step_path: str) -> Optional[Data]:
    """
    Parse a single STEP file into a PyG Data object.

    Node features  (22-dim):
        [0:8]  component-type one-hot  (geometry-driven via SDF, 8 classes)
        [8]    log1p(volume), clipped at 13.8
        [9]    log1p(surface_area), clipped at 11.5
        [10]   bbox Δx / bbox_max       (absolute width)
        [11]   bbox Δy / bbox_max       (absolute height)
        [12]   bbox Δz / bbox_max       (absolute depth)
        [13]   elongation               (longest/mid — affine-invariant)
        [14]   flatness                 (shortest/longest — affine-invariant)
        [15]   aspect x/y              (affine-invariant)
        [16]   aspect y/z              (affine-invariant)
        [17]   sphericity               (π^1/3 (6V)^2/3 / SA)
        [18]   SDF mean  / sdf_mean_max
        [19]   SDF variance / sdf_var_max
        [20]   SA/V ratio / sav_max
        [21]   log1p(n_holes)

    Edge features  (6-dim):
        [0]    mate type encoded  (0=coincident … 5=other, normalised to [0,1])
        [1]    weight             (1.0 for detected contacts)
        [2:6]  joint type one-hot (rigid, revolute, slider, cylindrical)
    """
    sp = Path(step_path)
    meta = {}
    json_path = sp.parent / "assembly.json"
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)

    # ── P4: Extract per-body hole counts from assembly.json ──────────────────
    holes = meta.get("holes", {}) or {}
    if not isinstance(holes, dict):
        holes = {}
    body_hole_counts = {}
    for hole_id, hole_data in holes.items():
        if isinstance(hole_data, dict):
            bid = hole_data.get("body") or hole_data.get("body_id", "")
            if bid:
                body_hole_counts[bid] = body_hole_counts.get(bid, 0) + 1

    # ── P6: Extract joint type per body-pair from assembly.json ──────────────
    joints_raw = meta.get("joints", {}) or {}
    if not isinstance(joints_raw, dict):
        joints_raw = {}
    JOINT_TYPE_MAP = {
        "RigidJointType":       [1, 0, 0, 0],
        "RevoluteJointType":    [0, 1, 0, 0],
        "SliderJointType":      [0, 0, 1, 0],
        "CylindricalJointType": [0, 0, 0, 1],
    }
    DEFAULT_JOINT = [0, 0, 0, 0]
    pair_joint = {}
    for jid, jdata in joints_raw.items():
        if isinstance(jdata, dict):
            jtype = jdata.get("jointType", jdata.get("type", ""))
            b1 = jdata.get("body1", jdata.get("occurrenceOne", ""))
            b2 = jdata.get("body2", jdata.get("occurrenceTwo", ""))
            if b1 and b2:
                pkey = tuple(sorted([str(b1), str(b2)]))
                pair_joint[pkey] = JOINT_TYPE_MAP.get(jtype, DEFAULT_JOINT)

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
        centers:    list = []
        coms:       list = []

        for dim, tag in volumes:
            bbox = gmsh.model.occ.getBoundingBox(dim, tag)
            vol  = gmsh.model.occ.getMass(dim, tag)
            dx   = bbox[3] - bbox[0]
            dy   = bbox[4] - bbox[1]
            dz   = bbox[5] - bbox[2]
            raw_vols.append(vol)
            bboxes.append((dx, dy, dz))
            centers.append(np.array([(bbox[0] + bbox[3]) / 2,
                                     (bbox[1] + bbox[4]) / 2,
                                     (bbox[2] + bbox[5]) / 2]))
            try:
                coms.append(np.asarray(gmsh.model.occ.getCenterOfMass(dim, tag)))
            except Exception:
                coms.append(None)
            bnd = gmsh.model.getBoundary([(dim, tag)], oriented=False, combined=True)
            body_surfs.append(frozenset(abs(s[1]) for s in bnd if s[0] == 2))

        # ── Trimesh enrichment: exact SA + SDF + component type per body ───
        exact_sas: list = []
        sdf_means: list = []
        sdf_vars:  list = []
        type_idxs: list = []

        for i, (dim, tag) in enumerate(volumes):
            dx, dy, dz = bboxes[i]
            bbox_sa = 2.0 * (dx * dy + dy * dz + dz * dx)   # fallback

            tm = _build_trimesh(list(body_surfs[i])) if _trimesh_ok else None
            if tm is not None and len(tm.faces) >= 4:
                exact_sa      = float(tm.area)
                sdf_m, sdf_v  = _compute_sdf_stats(tm)
            else:
                exact_sa, sdf_m, sdf_v = bbox_sa, 0.0, 0.0

            exact_sas.append(exact_sa)
            sdf_means.append(sdf_m)
            sdf_vars.append(sdf_v)

            signals = _compute_shape_signals(
                tm, raw_vols[i], bboxes[i], centers[i], coms[i],
                face_count=len(body_surfs[i]),
            )
            type_idx, _hv, _headv, _notes = _classify_component_type(signals)
            type_idxs.append(type_idx)

        # ── Normalisation constants ────────────────────────────────────────
        n         = len(volumes)
        vol_max   = max(raw_vols)  or 1.0
        sa_max    = max(exact_sas) or 1.0
        sdf_m_max = max(sdf_means) or 1.0
        sdf_v_max = max(sdf_vars)  or 1.0
        sav_vals  = [exact_sas[i] / (raw_vols[i] + 1e-9) for i in range(n)]
        sav_max   = max(sav_vals)  or 1.0
        bbox_max  = max(max(dx, dy, dz) for dx, dy, dz in bboxes) or 1.0

        # Affine-invariant normalisation
        elongations = []
        aspect_xys  = []
        aspect_yzs  = []
        for dx, dy, dz in bboxes:
            ext = sorted([dx, dy, dz])
            elongations.append(ext[2] / (ext[1] + 1e-9))
            aspect_xys.append(dx / (dy + 1e-9))
            aspect_yzs.append(dy / (dz + 1e-9))
        elong_max  = max(elongations) or 1.0
        asp_xy_max = max(aspect_xys)  or 1.0
        asp_yz_max = max(aspect_yzs)  or 1.0

        # ── 22-dim node feature vectors ───────────────────────────────────
        node_feats: List[List[float]] = []

        for i in range(n):
            dx, dy, dz = bboxes[i]
            ext = sorted([dx, dy, dz])
            type_oh      = [0.0] * 8
            type_oh[type_idxs[i]] = 1.0

            sphericity = min(1.0, (math.pi ** (1 / 3))
                            * ((6 * raw_vols[i]) ** (2 / 3))
                            / (exact_sas[i] + 1e-9))

            # P4: hole count for this body
            vtag_str = str(volumes[i][1])
            n_holes = body_hole_counts.get(vtag_str, 0)

            feat = (
                type_oh                                                     # [0:8]
                + [min(13.8, _safe_log(raw_vols[i])),                       # [8]   log1p(volume)
                   min(11.5, _safe_log(exact_sas[i])),                      # [9]   log1p(SA)
                   dx / bbox_max,                                           # [10]  bbox Δx
                   dy / bbox_max,                                           # [11]  bbox Δy
                   dz / bbox_max,                                           # [12]  bbox Δz
                   elongations[i] / elong_max,                              # [13]  elongation
                   ext[0] / (ext[2] + 1e-9),                               # [14]  flatness
                   aspect_xys[i]  / asp_xy_max,                            # [15]  aspect x/y
                   aspect_yzs[i]  / asp_yz_max,                            # [16]  aspect y/z
                   sphericity,                                              # [17]  sphericity
                   sdf_means[i] / sdf_m_max,                               # [18]  SDF mean
                   sdf_vars[i]  / sdf_v_max,                               # [19]  SDF variance
                   sav_vals[i]  / sav_max,                                  # [20]  SA/V ratio
                   math.log1p(n_holes)]                                     # [21]  log1p(n_holes)
            )
            node_feats.append(feat)

        # ── Edge detection (shared bounding surface = mate contact) ────────
        raw_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if body_surfs[i] & body_surfs[j]:
                    raw_edges.append((i, j))

        # P2: Deduplicate contacts to part-pair level
        seen = set()
        deduped_edges = []
        for u, v in raw_edges:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                deduped_edges.append((u, v))
        if len(deduped_edges) < len(raw_edges):
            print(f"    [dedup] {Path(step_path).name}: {len(raw_edges)} contacts → {len(deduped_edges)} part-pairs")

        # P6: Build bidirectional edges with joint type features
        src, dst, eattr = [], [], []
        for u, v in deduped_edges:
            vtag_u = str(volumes[u][1])
            vtag_v = str(volumes[v][1])
            pkey = tuple(sorted([vtag_u, vtag_v]))
            joint_oh = pair_joint.get(pkey, DEFAULT_JOINT)
            ea = [0.0, 1.0] + joint_oh   # 6-dim: mate + weight + joint type
            src += [u, v]; dst += [v, u]
            eattr += [ea, ea]

        if not src:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        vtag_i = str(volumes[i][1])
                        vtag_j = str(volumes[j][1])
                        pkey = tuple(sorted([vtag_i, vtag_j]))
                        joint_oh = pair_joint.get(pkey, DEFAULT_JOINT)
                        src.append(i); dst.append(j)
                        eattr.append([1.0, 1.0] + joint_oh)

        x          = torch.tensor(node_feats, dtype=torch.float)
        edge_index = torch.tensor([src, dst],  dtype=torch.long)
        edge_attr  = torch.tensor(eattr,        dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    except Exception as e:
        print(f"    [skip] {Path(step_path).name}: {e}")
        return None

    finally:
        gmsh.finalize()


# ── Timeout wrapper for _parse_step ──────────────────────────────────────────

def _parse_step_worker(step_path: str, result_queue: "_mp.Queue") -> None:
    """Worker target: parse one STEP file and push result onto the queue."""
    try:
        import io
        result = _parse_step(step_path)
        if result is not None:
            buf = io.BytesIO()
            torch.save(result, buf)
            result_queue.put(("ok", buf.getvalue()))
        else:
            result_queue.put(("ok", None))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


def _parse_step_with_timeout(step_path: str, timeout_secs: int = 300) -> tuple:
    """
    Run _parse_step in a child process with a time limit.

    Returns (Data | None, status) where status is "ok" | "timeout" | "error".
    """
    ctx = _mp.get_context("spawn")
    q   = ctx.Queue()
    p   = ctx.Process(target=_parse_step_worker, args=(step_path, q), daemon=True)
    p.start()

    # Poll in short increments against an explicit wall-clock deadline rather
    # than one big p.join(timeout_secs) call. On this platform a *single*
    # join() (even with a timeout argument) has been observed to block far
    # past its requested timeout when the child is stuck deep in an
    # uninterruptible OpenCASCADE/gmsh C-extension call — almost certainly
    # macOS's crash-reporter/diagnostic subsystem intercepting the killed
    # process (known to hold up reaping for large C++ processes), not a bug
    # in our own logic. A short-increment loop means each individual join()
    # call is small, so even if ONE of them ignores its bound, the outer
    # time.time() check still regains control close to the intended budget.
    t_start = time.time()
    while time.time() - t_start < timeout_secs and p.is_alive():
        p.join(1)

    if p.is_alive():
        p.terminate()
        t_term = time.time()
        while time.time() - t_term < 5 and p.is_alive():
            p.join(0.5)
        if p.is_alive():
            p.kill()
            t_kill = time.time()
            while time.time() - t_kill < 3 and p.is_alive():
                p.join(0.5)
        if p.is_alive():
            print(f"    [skip] {Path(step_path).name}: timeout after {timeout_secs}s "
                  f"(child unresponsive to SIGKILL — abandoning, daemon process)")
        else:
            print(f"    [skip] {Path(step_path).name}: timeout after {timeout_secs}s")
        return None, "timeout"

    if not q.empty():
        status, payload = q.get_nowait()
        if status == "ok":
            if payload is None:
                return None, "ok"
            import io
            data = torch.load(io.BytesIO(payload), weights_only=False)
            return data, "ok"
        if status == "error":
            print(f"    [detail] {Path(step_path).name}: {payload}", flush=True)
        return None, status

    print(f"    [detail] {Path(step_path).name}: worker process crashed (no output)",
          flush=True)
    return None, "error"


# ── Synthetic data fallback ───────────────────────────────────────────────────

def _synthetic_graph(n_nodes: int = None) -> Data:
    """Generate one structured 22-dim assembly graph using a random template."""
    import random as _rng
    r        = _rng.Random()
    node_dim = 22

    template = r.choice(["bolt", "shaft", "mixed"])
    nodes: List[tuple] = []   # (type_idx, geom_hint)
    edges: List[tuple] = []   # (i, j) undirected

    if template == "bolt":
        n_bolts   = r.randint(2, 4)
        nodes.append((4, "flat"))     # plate
        nodes.append((0, "body"))     # bracket
        idx = 2
        for _ in range(n_bolts):
            bolt_i   = idx;  nodes.append((1, "elongated")); idx += 1
            nut_i    = idx;  nodes.append((1, "small"));     idx += 1
            washer_i = idx;  nodes.append((4, "flat"));      idx += 1
            edges += [(bolt_i, nut_i), (bolt_i, washer_i),
                      (bolt_i, 0),    (bolt_i, 1),
                      (washer_i, 0)]

    elif template == "shaft":
        nodes += [(3, "elongated"), (2, "ring"), (2, "ring"),
                  (5, "body"),      (6, "round"), (1, "small")]
        edges += [(0, 1), (0, 2), (1, 3), (2, 3), (4, 0), (5, 0)]

    else:  # mixed
        nodes += [(4, "flat"), (1, "elongated"), (1, "small"),
                  (3, "elongated"), (2, "ring"), (5, "body")]
        edges += [(1, 2), (1, 0), (3, 4), (4, 5), (5, 0)]

    n = len(nodes)
    x = torch.zeros(n, node_dim)

    for i, (type_idx, hint) in enumerate(nodes):
        x[i, type_idx] = 1.0
        if hint == "elongated":
            x[i, 8]  = r.uniform(0.05, 0.2)    # volume
            x[i, 9]  = r.uniform(0.1,  0.3)    # exact SA
            x[i, 10] = r.uniform(0.1,  0.3)    # bbox Δx (narrow)
            x[i, 11] = r.uniform(0.1,  0.3)    # bbox Δy (narrow)
            x[i, 12] = r.uniform(0.6,  1.0)    # bbox Δz (long)
            x[i, 13] = r.uniform(0.6,  1.0)    # elongation (high)
            x[i, 14] = r.uniform(0.05, 0.15)   # flatness (low)
            x[i, 15] = r.uniform(0.8,  1.0)    # aspect x/y
            x[i, 16] = r.uniform(0.1,  0.3)    # aspect y/z
            x[i, 17] = r.uniform(0.3,  0.6)    # sphericity
        elif hint == "flat":
            x[i, 8]  = r.uniform(0.1,  0.4)
            x[i, 9]  = r.uniform(0.5,  0.9)
            x[i, 10] = r.uniform(0.5,  1.0)    # bbox Δx (wide)
            x[i, 11] = r.uniform(0.5,  1.0)    # bbox Δy (wide)
            x[i, 12] = r.uniform(0.02, 0.1)    # bbox Δz (thin)
            x[i, 13] = r.uniform(0.1,  0.3)    # elongation
            x[i, 14] = r.uniform(0.05, 0.12)   # flatness (very low)
            x[i, 15] = r.uniform(0.7,  1.0)    # aspect x/y
            x[i, 16] = r.uniform(0.7,  1.0)    # aspect y/z
            x[i, 17] = r.uniform(0.2,  0.5)    # sphericity
        elif hint == "ring":
            x[i, 8]  = r.uniform(0.15, 0.35)
            x[i, 9]  = r.uniform(0.4,  0.7)
            x[i, 10] = r.uniform(0.3,  0.7)    # bbox Δx
            x[i, 11] = r.uniform(0.3,  0.7)    # bbox Δy
            x[i, 12] = r.uniform(0.1,  0.3)    # bbox Δz (short)
            x[i, 13] = r.uniform(0.2,  0.5)    # elongation
            x[i, 14] = r.uniform(0.2,  0.5)    # flatness
            x[i, 15] = r.uniform(0.8,  1.0)    # aspect x/y
            x[i, 16] = r.uniform(0.8,  1.0)    # aspect y/z
            x[i, 17] = r.uniform(0.5,  0.8)    # sphericity
            x[i, 19] = r.uniform(0.5,  0.9)    # SDF variance (ring cavity)
        elif hint == "round":
            x[i, 8]  = r.uniform(0.2,  0.5)
            x[i, 9]  = r.uniform(0.5,  0.8)
            x[i, 10] = r.uniform(0.3,  0.6)    # bbox Δx
            x[i, 11] = r.uniform(0.3,  0.6)    # bbox Δy
            x[i, 12] = r.uniform(0.3,  0.6)    # bbox Δz
            x[i, 13] = r.uniform(0.15, 0.35)   # elongation
            x[i, 14] = r.uniform(0.15, 0.4)    # flatness
            x[i, 15] = r.uniform(0.85, 1.0)    # aspect x/y
            x[i, 16] = r.uniform(0.85, 1.0)    # aspect y/z
            x[i, 17] = r.uniform(0.4,  0.7)    # sphericity
            x[i, 19] = r.uniform(0.4,  0.8)    # SDF variance
        else:  # body / housing / small
            x[i, 8:] = torch.rand(node_dim - 8) * 0.6 + 0.2

        x[i, 18] = r.uniform(0.1, 0.8)   # SDF mean
        x[i, 20] = r.uniform(0.1, 0.9)   # SA/V ratio
        x[i, 21] = math.log1p(r.randint(0, 5))   # log1p(n_holes)

    joint_types = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,0,0,0]]
    src, dst, eattr = [], [], []
    for (ei, ej) in edges:
        mt = r.randint(0, 5) / 5.0
        jt = r.choice(joint_types)
        src += [ei, ej]; dst += [ej, ei]
        eattr += [[mt, 1.0] + jt, [mt, 1.0] + jt]

    if not src:
        return None

    return Data(
        x          = x,
        edge_index = torch.tensor([src, dst], dtype=torch.long),
        edge_attr  = torch.tensor(eattr,      dtype=torch.float),
    )


def _generate_synthetic(n: int = 500) -> List[Data]:
    print(f"  Generating {n} structured 22-dim assembly graphs…")
    graphs = []
    while len(graphs) < n:
        g = _synthetic_graph()
        if g is not None and g.num_nodes >= 4:
            graphs.append(g)
    return graphs


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
        categories: Optional[List[str]] = None,
        transform=None,
        pre_transform=None,
    ):
        self.source_dir = Path(source_dir)
        self.categories = categories  # if set, only scan files under these subdirs
        if force_reload:
            for fname in ["data.pt", "processed/data.pt"]:
                proc = Path(processed_dir) / fname
                if proc.exists():
                    proc.unlink()
        super().__init__(str(processed_dir), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0],
                                            weights_only=False)
        cats_file = Path(self.processed_paths[0]).parent / "categories.json"
        if cats_file.exists():
            with open(cats_file) as f:
                self.graph_categories = json.load(f)
        else:
            self.graph_categories = [''] * len(self)

        sources_file = Path(self.processed_paths[0]).parent / "sources.json"
        if sources_file.exists():
            with open(sources_file) as f:
                self.graph_sources = json.load(f)
        else:
            self.graph_sources = [''] * len(self)

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
        if self.categories:
            _cats_set = set(self.categories)
            step_files = [
                p for p in step_files
                if p.relative_to(self.source_dir).parts[0] in _cats_set
            ]
            print(f"  Category filter: {self.categories}")
            print(f"  {len(step_files)} STEP files after category filter")

        # Per-graph cache: each assembly saved immediately after parsing.
        # Survives force_reload and stall restarts — only data.pt is deleted,
        # not this directory.  Key = parent folder name (unique within a category).
        graph_cache_dir = Path(self.processed_paths[0]).parent / "graph_cache"
        graph_cache_dir.mkdir(parents=True, exist_ok=True)

        graphs:           List[Data] = []
        source_paths:     List[str]  = []
        graph_categories: List[str]  = []
        n_errors   = 0
        n_timeouts = 0
        n_cached   = 0

        _TIMEOUT = 300
        _category_dirs = {
            d.name for d in self.source_dir.iterdir() if d.is_dir()
        }

        if step_files:
            print(f"  Found {len(step_files)} STEP file(s) in {self.source_dir}",
                  flush=True)
            for idx, sf in enumerate(sorted(step_files), 1):
                label      = f"{sf.parent.name}/{sf.name}"
                cache_path = graph_cache_dir / f"{sf.parent.name}__{sf.stem}.pt"

                # ── Load from per-graph cache if available ────────────────
                if cache_path.exists():
                    try:
                        cached = torch.load(cache_path, weights_only=False)
                        g   = cached["data"]
                        cat = cached["category"]
                        src = cached["source"]
                        graphs.append(g)
                        graph_categories.append(cat)
                        source_paths.append(src)
                        n_cached += 1
                        print(f"  [{idx}/{len(step_files)}] [CACHED] {label}"
                              f" — {g.num_nodes} nodes", flush=True)
                        continue
                    except Exception as cache_err:
                        print(f"  [{idx}/{len(step_files)}] [CACHE-ERR] {label}:"
                              f" {cache_err} — re-parsing", flush=True)
                        cache_path.unlink(missing_ok=True)

                # ── Parse STEP file ───────────────────────────────────────
                print(f"  [{idx}/{len(step_files)}] Parsing: {label}", flush=True)
                t0 = time.time()
                g, status = _parse_step_with_timeout(str(sf), timeout_secs=_TIMEOUT)
                elapsed = round(time.time() - t0, 1)

                if status == "timeout":
                    print(f"  [TIMEOUT] {label} ({elapsed}s)", flush=True)
                    n_timeouts += 1
                    continue
                if status == "error":
                    print(f"  [ERROR]   {label} — parse failed", flush=True)
                    n_errors += 1
                    continue
                if g is not None and g.num_nodes >= 2:
                    n_dir = g.edge_index.size(1)
                    cat = ''
                    for part in sf.parts:
                        if part in _category_dirs:
                            cat = part
                            break
                    g.category = cat

                    # Save to per-graph cache immediately so restarts skip this file
                    torch.save({"data": g, "category": cat, "source": str(sf)},
                               cache_path)

                    graphs.append(g)
                    graph_categories.append(cat)
                    source_paths.append(str(sf))
                    print(f"  [OK]  {label} — {g.num_nodes} nodes  "
                          f"{n_dir//2} edges  ({elapsed}s)", flush=True)

            print(
                f"  Parsed {len(graphs) - n_cached} new  |  "
                f"cached: {n_cached}  errors: {n_errors}  timeouts: {n_timeouts}",
                flush=True,
            )
        else:
            print(f"  No STEP files found in {self.source_dir}.", flush=True)

        if len(graphs) == 0:
            raise RuntimeError(
                "No valid multi-body assembly graphs found in source directory."
            )
        print(f"  Using {len(graphs)} real assembly graph(s) — no synthetic data.")

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

        sources_file = Path(self.processed_paths[0]).parent / "sources.json"
        sources_file.write_text(json.dumps(source_paths, indent=2))
        cats_file = Path(self.processed_paths[0]).parent / "categories.json"
        cats_file.write_text(json.dumps(graph_categories, indent=2))
        print(f"  Saved {len(graphs)} graphs → {self.processed_paths[0]}")
        print(f"  Saved source paths    → {sources_file}")
        print(f"  Saved categories      → {cats_file}")


# ── Split helper ──────────────────────────────────────────────────────────────

def graph_level_indices(dataset: AssemblyDataset, cfg: dict, fold_idx: int = 0, n_folds: int = 5):
    """
    Seeded train/val/test graph-index partition — fixed 15% test set, then
    KFold cross-validation on the remaining 85%. Exposed standalone (not just
    via get_splits) so callers that need the raw, un-edge-masked graphs
    (e.g. NodeRanker training) can index the dataset directly without going
    through RandomLinkSplit.
    """
    n      = len(dataset)
    n_test = max(1, int(n * cfg["data"]["test_ratio"]))

    torch.manual_seed(42)
    perm     = torch.randperm(n).tolist()
    test_idx = perm[:n_test]
    train_val = perm[n_test:]

    kf    = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = list(kf.split(train_val))
    train_rel_idx, val_rel_idx = folds[fold_idx]
    train_idx = [train_val[i] for i in train_rel_idx]
    val_idx   = [train_val[i] for i in val_rel_idx]

    return train_idx, val_idx, test_idx


def get_splits(dataset: AssemblyDataset, cfg: dict, fold_idx: int = 0, n_folds: int = 5):
    """
    Fixed test set (15%) + KFold cross-validation on the remaining 85%.

    fold_idx selects which of the n_folds splits is used as validation;
    the rest form the training set.  RandomLinkSplit is applied per-graph.
    """
    train_idx, val_idx, test_idx = graph_level_indices(dataset, cfg, fold_idx, n_folds)

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
            cat = dataset.graph_categories[i] if i < len(dataset.graph_categories) else ''
            n_edges = data.edge_index.size(1)
            n_pos = n_edges // 2
            max_neg = n_pos * (data.num_nodes - 1) - n_pos
            req_neg = int(n_pos * cfg["training"]["neg_ratio"])
            if req_neg > max_neg:
                print(f"    [DIAG] graph {i}: {data.num_nodes} nodes, "
                      f"{n_edges} dir-edges, {n_pos} pos, "
                      f"requested {req_neg} neg but only {max_neg} possible")
            try:
                train_d, val_d, test_d = splitter(data)
                split_d = [train_d, val_d, test_d][split_idx]
                split_d.category = cat
                result.append(split_d)
            except Exception:
                continue
        return result

    train_data = _transform(train_idx, 0)
    val_data   = _transform(val_idx,   1)
    test_data  = _transform(test_idx,  2)

    print(f"  Fold {fold_idx + 1}/{n_folds} — train: {len(train_data)}"
          f"  val: {len(val_data)}  test: {len(test_data)}")
    return train_data, val_data, test_data
