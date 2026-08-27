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

Node feature vector: 34-dim
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
  [21]   log1p(n_holes)           (distinct hole locations — see _analyze_body_holes)
  [22]   has_holes                (1.0 if n_holes > 0, else 0.0)
  [23]   frac_through_holes       (fraction of this body's holes that go all the way through)
  [24]   frac_counterbore_holes   (fraction that are counterbore-style: two coaxial
         diameters, one for the bolt head to rest in, one narrower for the shaft)
  [25]   mean hole diameter / bbox_max
  [26]   max hole diameter / bbox_max
  [27]   has_counterbore_holes   (1.0 if frac_counterbore_holes > 0, else 0.0)
  [28]   frac_indentation_holes  (fraction of this body's non-through holes that are
         shallow (<= _INDENTATION_MAX_DEPTH) surface indentations/dimples rather
         than genuine blind holes meant to receive a fastener)
  [29]   has_indentation_holes   (1.0 if frac_indentation_holes > 0, else 0.0)
  [30]   frac_filled_holes       (fraction of this body's holes that already have
         another body's own centroid sitting on the bore axis -- a fastener (or
         other part) already occupies that hole, see _hole_is_occupied)
  [31]   has_empty_holes         (1.0 if this body has >=1 hole AND not all of
         them are filled, else 0.0 -- signals a genuine missing-component
         candidate for shape-gen to target)
  [32]   frac_curved_surface_holes  (fraction whose local surroundings are curved
         rather than flat, see _hole_end_is_on_flat_face -- fasteners almost
         never mount on a curved surface, e.g. a shaft's outer wall)
  [33]   has_curved_surface_holes   (1.0 if frac_curved_surface_holes > 0, else 0.0)

Edge feature vector: 6-dim
  [0]    mate type encoded  (0=coincident … 5=other, normalised to [0,1])
  [1]    weight             (1.0 for detected contacts)
  [2:6]  joint type one-hot (rigid, revolute, slider, cylindrical)
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as _mp
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import RandomLinkSplit

# 8-class component type vocabulary (index used for one-hot encoding).
# Proposed taxonomy (multi-signal geometric classifier, replaces the older
# body/fastener/bearing/shaft/plate/housing/gear/other SDF-rule scheme) —
# see _classify_component_type() below.
COMP_TYPES = ["long_shaft", "short_shaft", "thick_plate", "thin_plate",
              "bolt", "washer", "nut", "body"]
MATE_TYPES = ["coincident", "concentric", "parallel", "tangent", "fixed", "other"]

# Must match the node feature vector's actual width (see the module
# docstring above and _parse_step's own copy of the same list) -- used to
# invalidate the per-graph cache (see get_splits/AssemblyDataset's cache
# loading below) when it changes. Bump this every time a node feature is
# added/removed. Without this check, a stale cache built under an older
# feature set gets silently reused as-is (same filename, no other
# versioning), producing a dataset with graphs of MIXED, inconsistent
# widths -- confirmed as a real near-miss 2026-08-24: 235 graphs cached
# under the pre-hole-feature 22-dim vector were about to get mixed into a
# fresh 34-dim reparse before this check was added.
NODE_FEATURE_DIM = 34

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
    "bolt_elong_min":   1.7,    # was 2.0 — real stubby screws (e.g. BS 4183
                                 # cheese-head M16, elong~1.81-1.83) were
                                 # falling through to "body" just under the
                                 # old cutoff; 1.7 gives headroom below the
                                 # lowest observed real bolt while still
                                 # requiring round_min (0.60) to gate out
                                 # non-round chunky shapes. See
                                 # assembly_match_scoring_fix memory (2026-08-17).
    "bolt_elong_max":   8.0,
    "head_fill":        0.65,   # bounding-cylinder fill below this → head vote
    "head_com":         0.06,   # |COM offset|/l above this → head vote
    "head_faces":       8,      # B-Rep face count at/above this → head vote
    "long_shaft_elong": 5.0,    # l/m at/above this → Long Shaft (else Short)
    "thin_flat":        0.08,   # s/l — Thin vs Thick Plate split
    "plate_flat":       0.30,   # s/l — Thick Plate upper bound
    "plate_plan":       0.30,   # m/l — plates must not be strongly elongated in plan
    "max_fastener_extent": 100.0,  # mm — longest extent (l) cap for washer/nut
                                    # branches. Relative shape ratios alone can't
                                    # tell a small hex nut from a large multi-hole
                                    # plate (both can be "not elongated, flat,
                                    # has holes") — this absolute-size gate stops
                                    # a 150mm plate from landing on "nut". Set
                                    # generously above the largest real fastener
                                    # observed in this corpus (a 62mm clamping
                                    # nut) — revisit via audit_component_types.py
                                    # if a legitimately larger nut/washer starts
                                    # getting excluded.
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
    # A chunky, large-diameter nut (e.g. a big hex clamping nut) can have a
    # real threaded bore that doesn't dent the convex-hull volume ratio
    # enough to trigger that vote (bore is small relative to the nut's
    # bulk), even though the direct ray-cast probe correctly detects it
    # (ray_hits==0). For nut-proportioned candidates specifically, trust
    # that direct probe alone rather than requiring both votes — the
    # washer branch is untouched since washers are thin enough that their
    # bore reliably shows up in hull_ratio too. See assembly_match_scoring_fix
    # memory (2026-08-17): "Clamping Nut" (elong=1.01, flat=0.75) was falling
    # through to "body" on this gate alone.
    nut_has_hole = has_hole or signals["ray_hits"] == 0

    is_fastener_scale = l < T["max_fastener_extent"]
    if not is_fastener_scale and (
        (flat < T["washer_flat"] and plan > T["washer_plan"])
        or (elong < T["nut_elong"] and flat >= T["nut_flat_min"])
    ):
        notes.append("oversized-for-washer-or-nut")

    if has_hole and is_fastener_scale and flat < T["washer_flat"] and plan > T["washer_plan"]:
        return COMP_TYPES.index("washer"), hv, 0, notes
    if nut_has_hole and is_fastener_scale and elong < T["nut_elong"] and flat >= T["nut_flat_min"]:
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


# ── Geometry-derived joint/mate/hole signals ──────────────────────────────────
#
# These replace an assembly.json sidecar lookup (body_hole_counts, pair_joint
# below) that was the only source for edge dims [0:6] and node dim [21] — but
# no assembly.json has ever existed anywhere in the real training corpus (it
# was written for a Fusion360-exported metadata format that these STEP-only
# folders don't have), so every edge's joint-type one-hot was silently always
# [0,0,0,0] and mate-type always 0.0, and every node's hole-count was always
# 0.0. That meant RGATConv's per-relation weight matrices for 3 of its 4
# relations never received a training signal at all (edge_type always
# resolves to index 0 — see AssemblyGNN._edge_type in model.py), making the
# "heterogeneous" encoder functionally a plain GAT. The functions below
# derive real, always-available values straight from contact-surface geometry
# instead. assembly.json is still consulted first if present (e.g. a future
# Fusion360-derived corpus), so this is additive, not a removal.

JOINT_TYPE_MAP = {
    "RigidJointType":       [1, 0, 0, 0],
    "RevoluteJointType":    [0, 1, 0, 0],
    "SliderJointType":      [0, 0, 1, 0],
    "CylindricalJointType": [0, 0, 0, 1],
}
DEFAULT_JOINT = [0, 0, 0, 0]

_GEOM_JOINT_MAP = {
    "cylindrical": JOINT_TYPE_MAP["CylindricalJointType"],
    "revolute":    JOINT_TYPE_MAP["RevoluteJointType"],
    "rigid":       JOINT_TYPE_MAP["RigidJointType"],
    "slider":      JOINT_TYPE_MAP["SliderJointType"],
}
# MATE_TYPES = ["coincident", "concentric", "parallel", "tangent", "fixed", "other"]
_GEOM_MATE_IDX = {"concentric": 1, "coincident": 0, "other": 5}


def _classify_joint_from_geometry(shared_tags) -> Tuple[list, float]:
    """
    Classify a mate joint's type from the geometric type(s) of its shared
    contact surface(s) (gmsh.model.getType on each shared face tag) —
    real, always-available substitute for the never-present assembly.json
    joint-type lookup:
      - shared faces are cylinder(s) only, no planar face  -> revolute
        (pure shaft-in-bore fit — free rotation, e.g. a shaft in a bearing bore)
      - shared faces include both a cylinder and a plane    -> cylindrical
        (bore + shoulder — e.g. a bolt shank seated against a counterbore face)
      - shared faces are planar only                        -> rigid
        (flat mating face — most bolted/welded/stacked-plate contacts)
      - anything else (cone/sphere/torus/spline, or no
        resolvable face types)                               -> slider
        (catch-all — no strong rigid/rotational evidence either way)
    Returns (joint_one_hot, mate_type_normalised).
    """
    import gmsh
    types = []
    for tag in shared_tags:
        try:
            types.append(gmsh.model.getType(2, tag))
        except Exception:
            pass
    if not types:
        return DEFAULT_JOINT, _GEOM_MATE_IDX["other"] / (len(MATE_TYPES) - 1)

    has_plane = any(t == "Plane" for t in types)
    has_cyl   = any(t == "Cylinder" for t in types)

    if has_cyl and has_plane:
        joint_key, mate_key = "cylindrical", "concentric"
    elif has_cyl:
        joint_key, mate_key = "revolute", "concentric"
    elif has_plane:
        joint_key, mate_key = "rigid", "coincident"
    else:
        joint_key, mate_key = "slider", "other"

    mate_norm = _GEOM_MATE_IDX[mate_key] / (len(MATE_TYPES) - 1)
    return _GEOM_JOINT_MAP[joint_key], mate_norm


# Plausible bolt/screw hole diameter floor (mm) and minimum face depth (mm) —
# same calibration rationale as surface_analyzer.py's HOLE_DIAM_MIN/
# HOLE_MIN_DEPTH: without an absolute floor, tiny cylindrical B-Rep faces
# from thread-relief grooves, fillets, or chamfer facets get counted as
# "holes" purely because they're small relative to the body (observed:
# uncapped, a single nut's thread relief inflated its count to 30+).
_HOLE_DIAM_FLOOR = 1.0
_HOLE_MIN_DEPTH  = 0.5

# Coaxial faces within this XY-position tolerance (mm), on the same
# dominant axis, are treated as one physical hole rather than separate
# ones — a counterbore's wide recess face and its narrower clearance-bore
# face are two distinct Cylinder B-Rep faces for the same hole. Same value
# and rationale as surface_analyzer.py's HOLE_MERGE_TOLERANCE, duplicated
# locally (see this module's own no-cross-import convention below).
_HOLE_MERGE_TOLERANCE = 3.0

# A hole's combined axial span (across its merged coaxial faces), as a
# fraction of the body's own extent along the bore axis, needed to call it
# a through-hole rather than blind. Same value as surface_analyzer.py's
# HOLE_THROUGH_DEPTH_RATIO — calibrated there against real through/blind
# examples (through holes measured ~73.5% combined depth vs body
# thickness, blind ones ~17.8%); reused here since the same B-Rep-face-
# depth-vs-body-thickness relationship holds regardless of whether the
# hole belongs to a complete body (training time, here) or a partial
# assembly's open joint (inference time, surface_analyzer.py).
_HOLE_THROUGH_DEPTH_RATIO = 0.65

# A non-through hole this shallow (mm — STEP files in this corpus are
# consistently authored in mm, same assumption _HOLE_DIAM_FLOOR/
# _HOLE_MIN_DEPTH already make) isn't functioning as a real blind hole for
# a fastener to thread into -- it reads as a surface indentation/dimple
# (a countersink start, a locating dimple, a shallow machining mark)
# rather than a bore meant to receive a screw. Reclassified separately
# from genuine blind holes rather than lumped in with them.
_INDENTATION_MAX_DEPTH = 3.0


def _hole_axis(tag: int) -> List[float]:
    """Derive a cylindrical face's true bore axis via two parametric point
    samples (constant u, differing v). Sign-normalized so the
    largest-magnitude component is positive. Duplicated from
    surface_analyzer.py's identical helper — see _analyze_body_holes'
    docstring for why this module keeps its own copy instead of
    cross-importing."""
    import gmsh
    urange, vrange = gmsh.model.getParametrizationBounds(2, tag)
    p0 = gmsh.model.getValue(2, tag, [urange[0], vrange[0]])
    p1 = gmsh.model.getValue(2, tag, [urange[0], vrange[1]])
    axis = [p1[i] - p0[i] for i in range(3)]
    norm = sum(a * a for a in axis) ** 0.5
    if norm < 1e-9:
        return [0.0, 0.0, 1.0]
    axis = [a / norm for a in axis]
    dom = max(range(3), key=lambda i: abs(axis[i]))
    if axis[dom] < 0:
        axis = [-a for a in axis]
    return axis


# A flat face's own bbox extent along the bore axis, and how close its
# along-axis coordinate must sit to the hole's own entry point, to count
# as "this hole opens onto a flat face" (mm). Loose enough to tolerate a
# small chamfer/fillet between the flat face and the hole's own
# cylindrical wall (real parts almost always have one), tight enough that
# an unrelated flat face elsewhere on the body doesn't false-match.
_FLAT_FACE_THICKNESS_TOL = 2.0
_FLAT_FACE_LEVEL_TOL = 2.0


def _hole_end_is_on_flat_face(body_surf_tags, end_pt, dom_axis: int,
                               hole_radius: float) -> bool:
    """Does this hole end (either its entry or, for a through hole, its
    exit) open onto a flat (planar) face, rather than a curved one (a
    shaft's cylindrical outer wall, a curved housing, a fillet)? Real
    fasteners mount on flat surfaces -- a hole breaking through a curved
    one is more likely something else entirely (a lubrication port, a
    vent, a cross-drilled pin hole) even though it passes the diameter/
    depth filters same as a real fastener hole would.

    Heuristic, not exact B-Rep topology: several real parts have many
    small cylindrical faces near a hole (fillets, chamfers, thread
    reliefs), so picking "the one true adjacent face" via edge-adjacency
    queries is fragile. Instead, look for ANY planar boundary face on the
    same body whose own bbox is thin along the hole's bore axis and sits
    right at this end's own axis coordinate, with the hole's in-plane
    position falling within that face's own footprint -- consistent with
    the bbox-proximity style the rest of this module's hole detection
    already uses."""
    import gmsh
    other_axes = [i for i in range(3) if i != dom_axis]
    for tag in body_surf_tags:
        try:
            if gmsh.model.getType(2, tag) != "Plane":
                continue
            bb = gmsh.model.occ.getBoundingBox(2, tag)
        except Exception:
            continue
        if (bb[dom_axis + 3] - bb[dom_axis]) > _FLAT_FACE_THICKNESS_TOL:
            continue  # not thin along the bore axis -- not a face facing this way
        face_level = (bb[dom_axis] + bb[dom_axis + 3]) / 2
        if abs(face_level - end_pt[dom_axis]) > _FLAT_FACE_LEVEL_TOL:
            continue  # not at this end's own level
        if all(bb[a] - hole_radius <= end_pt[a] <= bb[a + 3] + hole_radius for a in other_axes):
            return True
    return False


def _hole_is_occupied(centroid, axis, diameter: float,
                       own_idx: int, all_coms: list) -> bool:
    """Is another body already sitting in this hole (a fastener, or
    anything else), so it isn't actually an empty/missing-component
    candidate? Same heuristic as surface_analyzer.py's identical helper
    (duplicated locally, see this module's no-cross-import convention):
    a part genuinely inserted through a hole is coaxial with it, so its
    own center of mass sits close to the hole's bore AXIS LINE regardless
    of how far along that axis it extends -- checked via perpendicular
    offset only. Deliberately axial-position-agnostic (not just "is the
    other body's bbox near the hole's own depth range") because a
    clearance-fit bolt (shaft modeled slightly narrower than the hole,
    common in real CAD) won't already be excluded as a shared/mated
    surface the way an exact-fit one would.

    Both bounds are relative to the HOLE's own diameter, not the parent
    body's overall size -- an earlier version bounded the axial check to
    the parent body's own extent (times 2), which on a real repro file
    falsely matched a washer sitting at one hole's position against a
    completely different, unrelated hole ~90mm away on the same body,
    just because their XY positions happened to coincide (a common
    occurrence in symmetric/grid mechanical layouts). A real inserted
    fastener doesn't extend many diameters past the hole it occupies, so
    bounding both checks to the hole's own diameter avoids that false
    positive while still catching genuine occupancy."""
    for idx, com in enumerate(all_coms):
        if idx == own_idx or com is None:
            continue
        v = com - centroid
        along = float(np.dot(v, axis))
        perp = v - along * np.asarray(axis)
        perp_dist = float(np.linalg.norm(perp))
        if perp_dist <= diameter * 0.6 and abs(along) <= diameter * 4.0:
            return True
    return False


def _analyze_body_holes(body_surf_tags, bbox_dxdydz: tuple, ext: list,
                         own_idx: int = -1, all_coms: Optional[list] = None) -> dict:
    """
    Per-body hole analysis, richer than a raw cylindrical-face count:
    groups coaxial cylindrical faces into distinct physical hole locations
    (a counterbore's wide + narrow faces are one hole, not two — same
    coaxial-merge concept as surface_analyzer.py's hole-candidate merge,
    duplicated locally rather than cross-imported since dataset.py is
    meant to stay import-independent of the inference-side modules that
    already import classifier logic *from* it), then classifies each
    location as through vs blind (combined depth vs body extent along the
    bore axis), simple vs counterbore (one distinct face diameter in the
    group vs two or more), flat vs curved surroundings (does either end
    open onto a planar face, see _hole_end_is_on_flat_face), and — if
    own_idx/all_coms are supplied — empty vs already occupied by another
    body (see _hole_is_occupied). A non-through hole shallower than
    _INDENTATION_MAX_DEPTH is reclassified as an indentation rather than
    counted as blind -- see that constant's docstring for why.

    Returns aggregate features for the whole body:
      n_holes           : int    distinct hole locations (not raw faces)
      frac_through      : float  fraction of those holes that are through
      frac_counterbore  : float  fraction that are counterbore-style
      frac_indentation  : float  fraction that are shallow non-through
                                  indentations rather than genuine blind
                                  holes (the remaining fraction,
                                  1 - frac_through - frac_indentation, is
                                  genuine blind holes)
      frac_filled       : float  fraction already occupied by another body
                                  (0.0 if all_coms wasn't supplied)
      frac_curved_surface : float fraction whose local surroundings are
                                  curved rather than flat -- fasteners
                                  almost never mount there even if the
                                  hole itself passes every other filter
      mean_diam         : float  mean of each hole's own representative
                                  (widest) diameter
      max_diam          : float  largest such representative diameter
    All fractions/diameters are 0.0 when the body has no qualifying holes.
    """
    import gmsh
    zero = {"n_holes": 0, "frac_through": 0.0, "frac_counterbore": 0.0,
            "frac_indentation": 0.0, "frac_filled": 0.0, "frac_curved_surface": 0.0,
            "mean_diam": 0.0, "max_diam": 0.0}
    _, m, _ = ext  # sorted [small, mid, long] body extents
    if m <= 0:
        return zero

    candidates = []
    for tag in body_surf_tags:
        try:
            if gmsh.model.getType(2, tag) != "Cylinder":
                continue
            bb = gmsh.model.occ.getBoundingBox(2, tag)
        except Exception:
            continue
        dims = sorted([bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]])
        diameter = dims[1]
        depth = dims[0] if (dims[1] - dims[0]) > (dims[2] - dims[1]) else dims[2]
        if not (_HOLE_DIAM_FLOOR <= diameter < 0.6 * m and depth >= _HOLE_MIN_DEPTH):
            continue
        try:
            axis = _hole_axis(tag)
        except Exception:
            continue
        candidates.append({"bbox": bb, "axis": axis, "diameter": diameter})

    if not candidates:
        return zero

    groups: dict = {}
    for c in candidates:
        dom = max(range(3), key=lambda i: abs(c["axis"][i]))
        bb = c["bbox"]
        centroid_perp = [(bb[i] + bb[i + 3]) / 2 for i in range(3) if i != dom]
        key = (dom, tuple(round(v / _HOLE_MERGE_TOLERANCE) for v in centroid_perp))
        groups.setdefault(key, []).append(c)

    n_through = 0
    n_counterbore = 0
    n_indentation = 0
    n_filled = 0
    n_curved = 0
    rep_diams = []
    for (dom, _), group in groups.items():
        distinct_diams = {round(g["diameter"], 1) for g in group}
        if len(distinct_diams) >= 2:
            n_counterbore += 1
        widest = max(group, key=lambda g: g["diameter"])
        rep_diams.append(widest["diameter"])

        near = min(g["bbox"][dom] for g in group)
        far = max(g["bbox"][dom + 3] for g in group)
        combined_depth = far - near
        body_extent = bbox_dxdydz[dom]
        is_through = body_extent > 1e-6 and (combined_depth / body_extent) >= _HOLE_THROUGH_DEPTH_RATIO
        if is_through:
            n_through += 1
        elif combined_depth <= _INDENTATION_MAX_DEPTH:
            n_indentation += 1

        wb = widest["bbox"]
        centroid_perp = [(wb[k] + wb[k + 3]) / 2 for k in range(3) if k != dom]
        other_axes = [k for k in range(3) if k != dom]
        near_pt = [0.0, 0.0, 0.0]
        far_pt = [0.0, 0.0, 0.0]
        near_pt[dom] = near
        far_pt[dom] = far
        for k, a in enumerate(other_axes):
            near_pt[a] = centroid_perp[k]
            far_pt[a] = centroid_perp[k]
        hole_radius = widest["diameter"] / 2.0
        on_flat = (_hole_end_is_on_flat_face(body_surf_tags, near_pt, dom, hole_radius)
                   or _hole_end_is_on_flat_face(body_surf_tags, far_pt, dom, hole_radius))
        if not on_flat:
            n_curved += 1

        if all_coms is not None:
            centroid = np.array([(wb[k] + wb[k + 3]) / 2 for k in range(3)])
            if _hole_is_occupied(centroid, np.asarray(widest["axis"]), widest["diameter"],
                                  own_idx, all_coms):
                n_filled += 1

    n_holes = len(groups)
    return {
        "n_holes": n_holes,
        "frac_filled": n_filled / n_holes,
        "frac_through": n_through / n_holes,
        "frac_counterbore": n_counterbore / n_holes,
        "frac_indentation": n_indentation / n_holes,
        "frac_curved_surface": n_curved / n_holes,
        "mean_diam": sum(rep_diams) / len(rep_diams),
        "max_diam": max(rep_diams),
    }


def _contact_area_weight(shared_tags, sa_u: float, sa_v: float) -> float:
    """
    Weight a detected contact by how much of the two bodies actually touch,
    relative to the smaller body's own total surface area — replaces a
    hardcoded 1.0 for every contact regardless of whether it's a flush
    mating face or a token touch. Clipped to [0,1] (numerically messy STEP
    faces can push the raw ratio slightly over 1; treat that as "fully
    mated" rather than propagating the noise as a feature value).
    """
    import gmsh
    total = 0.0
    for tag in shared_tags:
        try:
            total += abs(gmsh.model.occ.getMass(2, tag))
        except Exception:
            continue
    denom = min(sa_u, sa_v)
    if denom <= 0:
        return 1.0
    return max(0.0, min(1.0, total / denom))


# ── STEP file parser ──────────────────────────────────────────────────────────

def _parse_step(step_path: str) -> Optional[Data]:
    """
    Parse a single STEP file into a PyG Data object.

    Node features  (34-dim):
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
        [21]   log1p(n_holes)           (see _analyze_body_holes)
        [22]   has_holes
        [23]   frac_through_holes
        [24]   frac_counterbore_holes
        [25]   mean hole diameter / bbox_max
        [26]   max hole diameter / bbox_max
        [27]   has_counterbore_holes
        [28]   frac_indentation_holes
        [29]   has_indentation_holes
        [30]   frac_filled_holes
        [31]   has_empty_holes
        [32]   frac_curved_surface_holes
        [33]   has_curved_surface_holes

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

    # ── P4: Extract per-body hole counts from assembly.json, if present ──────
    # (never present in the real corpus — see the geometry-derived fallback
    # used below, _analyze_body_holes)
    holes = meta.get("holes", {}) or {}
    if not isinstance(holes, dict):
        holes = {}
    json_body_hole_counts = {}
    for hole_id, hole_data in holes.items():
        if isinstance(hole_data, dict):
            bid = hole_data.get("body") or hole_data.get("body_id", "")
            if bid:
                json_body_hole_counts[bid] = json_body_hole_counts.get(bid, 0) + 1

    # ── P6: Extract joint type per body-pair from assembly.json, if present ──
    # (never present in the real corpus — see the geometry-derived fallback
    # used below, _classify_joint_from_geometry)
    joints_raw = meta.get("joints", {}) or {}
    if not isinstance(joints_raw, dict):
        joints_raw = {}
    json_pair_joint = {}
    for jid, jdata in joints_raw.items():
        if isinstance(jdata, dict):
            jtype = jdata.get("jointType", jdata.get("type", ""))
            b1 = jdata.get("body1", jdata.get("occurrenceOne", ""))
            b2 = jdata.get("body2", jdata.get("occurrenceTwo", ""))
            if b1 and b2:
                pkey = tuple(sorted([str(b1), str(b2)]))
                json_pair_joint[pkey] = JOINT_TYPE_MAP.get(jtype, DEFAULT_JOINT)

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
            # getMass can return a negative volume for a solid whose B-Rep
            # shell has inconsistently-oriented faces (a real defect some
            # STEP files have, not junk input) -- left signed, sphericity's
            # (6*vol)**(2/3) silently promotes to a Python complex number
            # instead of raising, and the later min(1.0, ...) then blows up
            # with "'<' not supported between complex and float". abs() here
            # keeps every downstream use (sphericity, vol_max, sav ratios,
            # the log-volume feature) consistently unsigned.
            vol  = abs(gmsh.model.occ.getMass(dim, tag))
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
        ext_list:  list = []   # per-body sorted [small, mid, long] extents, reused for hole counting

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
            ext_list.append(signals["ext"])

        # ── Normalisation constants ────────────────────────────────────────
        n         = len(volumes)
        vol_max   = max(raw_vols)  or 1.0
        sa_max    = max(exact_sas) or 1.0
        sdf_m_max = max(sdf_means) or 1.0
        sdf_v_max = max(sdf_vars)  or 1.0
        sav_vals  = [exact_sas[i] / (raw_vols[i] + 1e-9) for i in range(n)]
        sav_max   = max(sav_vals)  or 1.0
        bbox_max  = max(max(dx, dy, dz) for dx, dy, dz in bboxes) or 1.0

        # ── Per-body hole analysis (through/blind, simple/counterbore) ─────
        hole_info_list = []
        for i in range(n):
            vtag_str = str(volumes[i][1])
            if vtag_str in json_body_hole_counts:
                # assembly.json only ever supplies a raw count (never
                # present in the real corpus, see P4 above) -- no
                # through/counterbore/diameter info available from it, so
                # those fields stay at their zero defaults.
                hole_info_list.append({
                    "n_holes": json_body_hole_counts[vtag_str],
                    "frac_through": 0.0, "frac_counterbore": 0.0,
                    "frac_indentation": 0.0, "frac_filled": 0.0,
                    "frac_curved_surface": 0.0, "mean_diam": 0.0, "max_diam": 0.0,
                })
            else:
                hole_info_list.append(_analyze_body_holes(
                    body_surfs[i], bboxes[i], ext_list[i], own_idx=i, all_coms=coms))

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

        # ── 34-dim node feature vectors ───────────────────────────────────
        node_feats: List[List[float]] = []

        for i in range(n):
            dx, dy, dz = bboxes[i]
            ext = sorted([dx, dy, dz])
            type_oh      = [0.0] * 8
            type_oh[type_idxs[i]] = 1.0

            sphericity = min(1.0, (math.pi ** (1 / 3))
                            * ((6 * raw_vols[i]) ** (2 / 3))
                            / (exact_sas[i] + 1e-9))

            # P4 / holes: per-body hole analysis — assembly.json count if
            # present (never is, in practice), else geometry-derived (see
            # _analyze_body_holes for the through/blind + simple/counterbore
            # classification this now includes beyond the original P4 count)
            hinfo = hole_info_list[i]
            n_holes = hinfo["n_holes"]

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
                   math.log1p(n_holes),                                     # [21]  log1p(n_holes)
                   1.0 if n_holes > 0 else 0.0,                             # [22]  has_holes
                   hinfo["frac_through"],                                   # [23]  frac_through_holes
                   hinfo["frac_counterbore"],                               # [24]  frac_counterbore_holes
                   hinfo["mean_diam"] / bbox_max,                          # [25]  mean hole diameter
                   hinfo["max_diam"]  / bbox_max,                          # [26]  max hole diameter
                   1.0 if hinfo["frac_counterbore"] > 0 else 0.0,           # [27]  has_counterbore_holes
                   hinfo["frac_indentation"],                               # [28]  frac_indentation_holes
                   1.0 if hinfo["frac_indentation"] > 0 else 0.0,           # [29]  has_indentation_holes
                   hinfo["frac_filled"],                                    # [30]  frac_filled_holes
                   1.0 if (n_holes > 0 and hinfo["frac_filled"] < 1.0) else 0.0,  # [31]  has_empty_holes
                   hinfo["frac_curved_surface"],                             # [32]  frac_curved_surface_holes
                   1.0 if hinfo["frac_curved_surface"] > 0 else 0.0]         # [33]  has_curved_surface_holes
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

        # P6: Build bidirectional edges with joint type features — assembly.json
        # pair lookup if present (never is, in practice), else geometry-derived
        # from the actual shared contact surface(s) between u and v.
        src, dst, eattr = [], [], []
        for u, v in deduped_edges:
            vtag_u = str(volumes[u][1])
            vtag_v = str(volumes[v][1])
            pkey = tuple(sorted([vtag_u, vtag_v]))
            shared = body_surfs[u] & body_surfs[v]
            if pkey in json_pair_joint:
                joint_oh, mate_norm = json_pair_joint[pkey], 0.0
            else:
                joint_oh, mate_norm = _classify_joint_from_geometry(shared)
            weight = _contact_area_weight(shared, exact_sas[u], exact_sas[v])
            ea = [mate_norm, weight] + joint_oh   # 6-dim: mate + weight + joint type
            src += [u, v]; dst += [v, u]
            eattr += [ea, ea]

        if not src:
            # Was a full pairwise mesh (O(n^2) edges) for assemblies where
            # the contact-detection heuristic found zero shared surfaces
            # between ANY body pair. Harmless for small n, but for a large
            # no-contact assembly this created thousands of edges in one
            # batch -- confirmed 2026-08-27: a ~70-body case needed RGATConv
            # to duplicate its per-relation weight matrix (torch.index_select
            # in message()) into a 9.26GiB buffer during real training,
            # crashing the run. Capped to each body's K nearest neighbors by
            # AABB-center distance instead: still gives every body *some*
            # fallback connectivity, but bounded to O(n*K) edges regardless
            # of n.
            _FALLBACK_K = 6
            pairs = set()
            for i in range(n):
                order = sorted((j for j in range(n) if j != i),
                               key=lambda j: np.linalg.norm(centers[i] - centers[j]))
                for j in order[:min(_FALLBACK_K, n - 1)]:
                    pairs.add((min(i, j), max(i, j)))
            for i, j in pairs:
                vtag_i = str(volumes[i][1])
                vtag_j = str(volumes[j][1])
                pkey = tuple(sorted([vtag_i, vtag_j]))
                joint_oh = json_pair_joint.get(pkey, DEFAULT_JOINT)
                ea = [1.0, 1.0] + joint_oh
                src += [i, j]; dst += [j, i]
                eattr += [ea, ea]

        x          = torch.tensor(node_feats, dtype=torch.float)
        edge_index = torch.tensor([src, dst],  dtype=torch.long)
        edge_attr  = torch.tensor(eattr,        dtype=torch.float)
        # Per-body AABB-center, scaled by this assembly's own bbox_max — NOT
        # centered on the assembly, since LinkPredictor only ever consumes
        # pos[u]-pos[v] differences, which cancel any constant origin offset
        # regardless of where the STEP file's arbitrary origin sits. Scaling
        # by bbox_max (the same constant used for the affine-invariant node
        # features above) keeps relative distances comparable in magnitude
        # across assemblies of very different absolute size. This is the
        # only source of geometric proximity signal in the whole feature
        # set — previously centers[i] was computed but only ever fed into
        # per-body signals (com_offset), never exposed to the model at all,
        # so "are these two bodies close enough to plausibly mate" had no
        # feature to be answered from.
        pos = torch.tensor(np.stack(centers) / bbox_max, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

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
    import queue as _queue_mod

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
    #
    # Drain the queue on EVERY iteration, not just after the child exits.
    # Queue.put() hands off to a background feeder thread that pickles the
    # payload and writes it to the underlying OS pipe -- for a large result
    # (a big graph's pickled bytes) that write can exceed the pipe's kernel
    # buffer and block until something reads it. With nothing reading until
    # is_alive() went False, a big-enough payload deadlocked BOTH sides: the
    # child's feeder thread stuck in write(), and this parent stuck forever
    # waiting on a child that could never finish exiting -- a documented
    # multiprocessing.Queue pitfall (stdlib "Programming guidelines": a
    # process that has put items on a queue won't terminate until they're
    # flushed, and joining it before consuming them can deadlock). Confirmed
    # live via `sample` on a real hang: the worker's feeder thread blocked in
    # write(), the parent blocked in read() even after the worker was
    # force-killed (2026-08-24, Bench_Vice_118). Reading here as results
    # arrive means the pipe never backs up in the first place.
    t_start = time.time()
    result = None
    while time.time() - t_start < timeout_secs and p.is_alive():
        p.join(1)
        try:
            while True:
                result = q.get_nowait()
        except _queue_mod.Empty:
            pass
        if result is not None:
            break

    if result is None and not q.empty():
        try:
            result = q.get_nowait()
        except _queue_mod.Empty:
            pass

    if result is None and p.is_alive():
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

    # Give an already-drained child a brief window to exit cleanly now that
    # nothing is blocking its feeder thread, so it doesn't linger as a
    # zombie -- not required for correctness (daemon process), just tidy.
    if result is not None and p.is_alive():
        p.join(2)

    if result is not None:
        status, payload = result
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
    """Generate one structured 34-dim assembly graph using a random template."""
    import random as _rng
    r        = _rng.Random()
    node_dim = 34

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
        _n_holes = r.randint(0, 5)
        x[i, 21] = math.log1p(_n_holes)           # log1p(n_holes)
        x[i, 22] = 1.0 if _n_holes > 0 else 0.0   # has_holes
        if _n_holes > 0:
            x[i, 23] = r.uniform(0.0, 1.0)        # frac_through_holes
            x[i, 24] = r.uniform(0.0, 0.4)        # frac_counterbore_holes (less common than simple)
            x[i, 25] = r.uniform(0.02, 0.3)       # mean hole diameter / bbox_max
            x[i, 26] = max(x[i, 25].item(), r.uniform(0.02, 0.3))  # max hole diameter / bbox_max
            x[i, 27] = 1.0 if x[i, 24].item() > 0 else 0.0          # has_counterbore_holes
            x[i, 28] = r.uniform(0.0, max(0.0, 1.0 - x[i, 23].item()))  # frac_indentation_holes (bounded by non-through fraction)
            x[i, 29] = 1.0 if x[i, 28].item() > 0 else 0.0          # has_indentation_holes
            x[i, 30] = r.uniform(0.0, 0.6)                          # frac_filled_holes
            x[i, 31] = 1.0 if x[i, 30].item() < 1.0 else 0.0        # has_empty_holes
            x[i, 32] = r.uniform(0.0, 0.2)                          # frac_curved_surface_holes (uncommon)
            x[i, 33] = 1.0 if x[i, 32].item() > 0 else 0.0          # has_curved_surface_holes

    joint_types = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,0,0,0]]
    src, dst, eattr = [], [], []
    for (ei, ej) in edges:
        mt = r.randint(0, 5) / 5.0
        jt = r.choice(joint_types)
        src += [ei, ej]; dst += [ej, ei]
        eattr += [[mt, 1.0] + jt, [mt, 1.0] + jt]

    if not src:
        return None

    # Random positions in a unit cube — LinkPredictor treats pos as required
    # (see model.py), so synthetic graphs need *some* value here even though
    # it carries no real geometric meaning for this fallback path.
    pos = torch.rand(n, 3)
    return Data(
        x          = x,
        edge_index = torch.tensor([src, dst], dtype=torch.long),
        edge_attr  = torch.tensor(eattr,      dtype=torch.float),
        pos        = pos,
    )


def _generate_synthetic(n: int = 500) -> List[Data]:
    print(f"  Generating {n} structured 34-dim assembly graphs…")
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
        timed_out_files: List[Path] = []

        # History: 300s -> 6h (too many real files abandoned) -> 60min
        # (still let genuinely-computing files run 2.5+ hours) -> 30min.
        # The large-file Gate_Valve batch mostly fails/times out anyway
        # regardless of budget (confirmed: 103 and 104 both ran the full
        # 60min just to time out), so a shorter cap loses little real data
        # there while roughly halving how long each dead-end file can hold
        # up the strictly-sequential pipeline. Nearly all real files finish
        # in well under 15 min regardless of the ceiling.
        _TIMEOUT = 30 * 60
        _category_dirs = {
            d.name for d in self.source_dir.iterdir() if d.is_dir()
        }

        def _category_of(sf: Path) -> str:
            for part in sf.parts:
                if part in _category_dirs:
                    return part
            return ''

        def _is_degenerate_full_mesh(g: Data) -> bool:
            # Signature of the pre-fix O(n^2) no-contact fallback: a near-
            # complete graph (almost every body pairs with almost every
            # other body) on an assembly large enough for that to matter.
            # A real physical assembly essentially never has every body
            # touching every other body, so this only fires on graphs built
            # under the old buggy fallback -- see the 2026-08-27 RGATConv
            # 9.26GiB crash this was root-caused from.
            n = g.num_nodes
            if n < 20:
                return False
            max_possible = n * (n - 1)
            return g.edge_index.size(1) >= max_possible * 0.9

        def _record_success(g: Data, sf: Path, cache_path: Path, elapsed: float,
                             tag: str = "OK") -> bool:
            if _is_degenerate_full_mesh(g):
                print(f"  [DEGENERATE] {sf.parent.name}/{sf.name} — "
                      f"{g.num_nodes} nodes, {g.edge_index.size(1)} edges "
                      f"(near-complete no-contact fallback graph) — excluded",
                      flush=True)
                return False
            cat = _category_of(sf)
            g.category = cat
            torch.save({"data": g, "category": cat, "source": str(sf)}, cache_path)
            graphs.append(g)
            graph_categories.append(cat)
            source_paths.append(str(sf))
            n_dir = g.edge_index.size(1)
            print(f"  [{tag}]  {sf.parent.name}/{sf.name} — {g.num_nodes} nodes  "
                  f"{n_dir//2} edges  ({elapsed}s)", flush=True)
            return True

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
                        if g.x.size(1) != NODE_FEATURE_DIM:
                            # Stale cache from an older feature set (see
                            # NODE_FEATURE_DIM's docstring) -- discard and
                            # re-parse rather than silently mixing widths.
                            raise ValueError(
                                f"cached node dim {g.x.size(1)} != current "
                                f"NODE_FEATURE_DIM {NODE_FEATURE_DIM}")
                        if _is_degenerate_full_mesh(g):
                            # Cached from before the fallback-edge fix (see
                            # _is_degenerate_full_mesh) -- invalidate so a
                            # future reload re-parses it with the bounded
                            # k-NN fallback instead.
                            raise ValueError(
                                f"cached graph is a near-complete no-contact "
                                f"fallback mesh ({g.num_nodes} nodes, "
                                f"{g.edge_index.size(1)} edges) — stale, "
                                f"re-parse with the fixed fallback")
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

                # CACHED_ONLY=1 skips attempting any file without an
                # existing cache entry outright -- for when the priority is
                # getting to model training on whatever's already parsed,
                # not spending more time on files that have already proven
                # difficult (used alongside SKIP_TIMEOUT_RETRY, 2026-08-27).
                import os as _os_env
                if _os_env.environ.get("CACHED_ONLY") == "1":
                    print(f"  [{idx}/{len(step_files)}] [SKIPPED] {label}"
                          f" (CACHED_ONLY=1, no cache entry)", flush=True)
                    n_errors += 1
                    continue

                # ── Parse STEP file ───────────────────────────────────────
                print(f"  [{idx}/{len(step_files)}] Parsing: {label}", flush=True)
                t0 = time.time()
                g, status = _parse_step_with_timeout(str(sf), timeout_secs=_TIMEOUT)
                elapsed = round(time.time() - t0, 1)

                if status == "timeout":
                    print(f"  [TIMEOUT] {label} ({elapsed}s)", flush=True)
                    n_timeouts += 1
                    timed_out_files.append(sf)
                    continue
                if status == "error":
                    print(f"  [ERROR]   {label} — parse failed", flush=True)
                    n_errors += 1
                    continue
                if g is not None and g.num_nodes >= 2:
                    if not _record_success(g, sf, cache_path, elapsed):
                        n_errors += 1

            print(
                f"  Parsed {len(graphs) - n_cached} new  |  "
                f"cached: {n_cached}  errors: {n_errors}  timeouts: {n_timeouts}",
                flush=True,
            )

            # ── Retry timeouts once, with a longer budget ──────────────────
            # Observed across runs (R35 vs R36 logs): the same file times out
            # in one run and parses in under a minute in another (e.g.
            # Gate_Valve_15: 34.1s in R35, 512s timeout in R36) -- since
            # parsing here is strictly sequential (one subprocess at a time,
            # no internal parallelism), that variance points to transient
            # system load (another process, swap pressure) at the moment
            # that file's turn came up, not a deterministically-too-complex
            # geometry. A single retry with 2x the timeout, after the main
            # pass has released whatever contention it created, recovers
            # most of these without needing to raise the budget for every
            # file up front.
            # SKIP_TIMEOUT_RETRY=1 bypasses this pass entirely -- useful when
            # the retry budget itself (2x an already-lowered ceiling) is
            # still eating hours with a near-zero recovery rate and the
            # priority is getting to model training with whatever's already
            # cached, not squeezing out a few more borderline files.
            import os as _os_env
            if timed_out_files and _os_env.environ.get("SKIP_TIMEOUT_RETRY") == "1":
                print(f"\n  Skipping retry pass for {len(timed_out_files)} "
                      f"timed-out file(s) (SKIP_TIMEOUT_RETRY=1)", flush=True)
            elif timed_out_files:
                retry_timeout = _TIMEOUT * 2
                print(f"\n  Retrying {len(timed_out_files)} timed-out file(s) "
                      f"with {retry_timeout}s budget …", flush=True)
                n_recovered = 0
                for sf in timed_out_files:
                    label = f"{sf.parent.name}/{sf.name}"
                    cache_path = graph_cache_dir / f"{sf.parent.name}__{sf.stem}.pt"
                    print(f"  [RETRY] Parsing: {label}", flush=True)
                    t0 = time.time()
                    g, status = _parse_step_with_timeout(str(sf), timeout_secs=retry_timeout)
                    elapsed = round(time.time() - t0, 1)
                    if status == "timeout":
                        print(f"  [RETRY-TIMEOUT] {label} ({elapsed}s) — giving up", flush=True)
                        continue
                    if status == "error":
                        print(f"  [RETRY-ERROR]   {label} — parse failed", flush=True)
                        continue
                    if g is not None and g.num_nodes >= 2:
                        if _record_success(g, sf, cache_path, elapsed, tag="RETRY-OK"):
                            n_timeouts -= 1
                            n_recovered += 1
                print(f"  Recovered {n_recovered}/{len(timed_out_files)} "
                      f"previously-timed-out file(s)  |  remaining timeouts: {n_timeouts}",
                      flush=True)
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

def _stable_bucket(key: str, n_buckets: int) -> int:
    """Deterministic hash bucket in [0, n_buckets) — same key always maps
    to the same bucket, independent of process/run/corpus size."""
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h, 16) % n_buckets


def graph_level_indices(dataset: AssemblyDataset, cfg: dict, fold_idx: int = 0, n_folds: int = 5,
                         true_5way: bool = False):
    """
    Train/val/test graph-index partition.

    true_5way (task 10, opt-in — default False, zero behavior change unless
    explicitly requested): every graph is used as test exactly once across
    the n_folds runs, instead of one fixed 15% test set reused for every
    fold's test evaluation. This is the textbook definition of k-fold CV,
    but it comes at a real cost flagged when this was scoped: each fold's
    test numbers are no longer measuring the *same* held-out graphs, so
    "does run B beat run A" stops being a clean apples-to-apples comparison
    the way it is with the fixed test set below (R37/R38's reported numbers
    all depend on that fixed-set property to mean what they say). Use this
    mode to characterize how much fold-to-fold test-set composition itself
    drives variance -- not as a drop-in replacement for the default split
    in an ongoing comparison series.

    Default (true_5way=False) split, unchanged, into two halves with
    different stability requirements:

    1. Test carve-out — hash-based on each graph's own (category, source
       path), corpus-size-independent. A plain torch.randperm(n)/KFold(n)
       split reshuffles entirely whenever n changes: confirmed, the
       fastener-only Phase 3 peak run (n=188) and the R34 resync (n=192)
       drew almost entirely different, non-overlapping 28-graph test sets
       purely because n changed, not because either model was evaluated
       differently on purpose — inflating an apparent "regression" that was
       partly just a different, harder-by-chance test slice. Since the test
       set is what gets compared *across* separate training runs, this is
       the half that actually needs corpus-size independence.
    2. CV fold split (within the train_val remainder) — category-stratified
       (StratifiedKFold, falls back to plain KFold if a category is too
       small). This part is only ever used *within* one training run (each
       run computes its own 5 folds fresh), so it doesn't need cross-run
       stability the way the test set does — and stratification matters
       more here: corpus categories range from Bench_vice (54 graphs) down
       to Tool_Post (8), and a hash-based per-graph fold assignment was
       tried and measured to leave Tool_Post with zero val examples in 3 of
       5 folds (worse than the plain-random baseline this whole split logic
       replaced) — reverted in favor of explicit stratification here, kept
       hash-based only where it's actually needed (the test set, above).

    Exposed standalone (not just via get_splits) so callers that need the
    raw, un-edge-masked graphs (e.g. NodeRanker training) can index the
    dataset directly without going through RandomLinkSplit.
    """
    n      = len(dataset)
    cats    = (dataset.graph_categories if len(dataset.graph_categories) == n
               else [''] * n)
    sources = (dataset.graph_sources if len(dataset.graph_sources) == n
               else [str(i) for i in range(n)])

    if true_5way:
        # Task 10: every graph is test exactly once. Outer stratified split
        # of ALL n graphs into n_folds -- fold[fold_idx] is this run's test
        # set, the other n_folds-1 folds are the train_val pool. Inner
        # split of that pool (fresh StratifiedKFold call, always taking its
        # first fold as val) gives train/val -- deterministic and simple
        # rather than trying to vary val selection with fold_idx too, since
        # the property actually being asked for is the rotating TEST set.
        all_idx = list(range(n))
        try:
            outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            outer_folds = list(outer.split(all_idx, cats))
        except ValueError as e:
            print(f"  [split] true_5way outer StratifiedKFold failed ({e}) — falling back to plain KFold")
            outer = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            outer_folds = list(outer.split(all_idx))

        train_val_rel, test_rel = outer_folds[fold_idx]
        test_idx  = [all_idx[i] for i in test_rel]
        train_val = [all_idx[i] for i in train_val_rel]

        tv_cats = [cats[i] for i in train_val]
        try:
            inner = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=43)
            inner_train_rel, inner_val_rel = list(inner.split(train_val, tv_cats))[0]
        except ValueError as e:
            print(f"  [split] true_5way inner StratifiedKFold failed ({e}) — falling back to plain KFold")
            inner = KFold(n_splits=n_folds, shuffle=True, random_state=43)
            inner_train_rel, inner_val_rel = list(inner.split(train_val))[0]

        train_idx = [train_val[i] for i in inner_train_rel]
        val_idx   = [train_val[i] for i in inner_val_rel]
    else:
        test_pct = max(1, min(99, round(cfg["data"]["test_ratio"] * 100)))
        test_idx:  List[int] = []
        train_val: List[int] = []
        for i in range(n):
            key = f"{cats[i]}::{sources[i]}"
            if _stable_bucket("test::" + key, 100) < test_pct:
                test_idx.append(i)
            else:
                train_val.append(i)

        tv_cats = [cats[i] for i in train_val]
        try:
            skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            folds = list(skf.split(train_val, tv_cats))
        except ValueError as e:
            print(f"  [split] StratifiedKFold failed ({e}) — falling back to plain KFold")
            kf    = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            folds = list(kf.split(train_val))

        train_rel_idx, val_rel_idx = folds[fold_idx]
        train_idx = [train_val[i] for i in train_rel_idx]
        val_idx   = [train_val[i] for i in val_rel_idx]

    cat_counts = {}
    for i in val_idx:
        cat_counts[cats[i]] = cat_counts.get(cats[i], 0) + 1
    missing_cats = sorted(set(cats) - set(cat_counts))
    if missing_cats:
        print(f"  [split] fold {fold_idx}: categories with 0 val examples: {missing_cats}")

    return train_idx, val_idx, test_idx


def get_splits(dataset: AssemblyDataset, cfg: dict, fold_idx: int = 0, n_folds: int = 5,
                true_5way: bool = False):
    """
    Fixed test set (15%) + KFold cross-validation on the remaining 85%
    (default), or a true 5-way rotating test partition when true_5way=True
    -- see graph_level_indices' docstring for the tradeoff.

    fold_idx selects which of the n_folds splits is used as validation;
    the rest form the training set.  RandomLinkSplit is applied per-graph.
    """
    train_idx, val_idx, test_idx = graph_level_indices(dataset, cfg, fold_idx, n_folds,
                                                        true_5way=true_5way)

    splitter = RandomLinkSplit(
        num_val                    = 0.1,
        num_test                   = 0.1,
        is_undirected              = True,
        add_negative_train_samples = True,
        neg_sampling_ratio         = cfg["training"]["neg_ratio"],
        # Without this, PyG defaults to 0.0 — meaning train supervision
        # edges are identical to train message-passing edges, so during
        # training the model scores edges it can already see in its own
        # attention/aggregation. At val/test time scored edges are held out
        # of message passing entirely (a genuinely inductive task), so the
        # default silently trains on an easier task than it's evaluated on.
        # 0.25 carves out a quarter of each graph's training edges to be
        # supervision-only, closing most of that train/eval task mismatch.
        disjoint_train_ratio       = 0.25,
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
            # RandomLinkSplit's edge permutation (torch.randperm) AND its
            # negative_sampling() call underneath (torch.randint plus
            # Python's own `random.sample` for the dense fallback path) all
            # draw from global, unseeded RNG state -- no generator param is
            # exposed anywhere in this chain. Left unseeded, re-evaluating
            # the SAME fold's SAME graphs (e.g. reloading a checkpoint to
            # re-check its reported AUC) redraws a different val/test edge
            # split every call, so the "same" fold gives different numbers
            # run to run. Seed both torch's and Python's global RNGs
            # deterministically per (fold, graph) using the same md5-hash
            # pattern as the test-set carve-out above, and restore both
            # states afterward so this doesn't perturb unrelated randomness
            # (model init, sample shuffling) that runs later in the process.
            source = dataset.graph_sources[i] if i < len(dataset.graph_sources) else str(i)
            seed = _stable_bucket(f"linksplit::{fold_idx}::{cat}::{source}", 2**31)
            torch_rng_state = torch.get_rng_state()
            py_rng_state = random.getstate()
            torch.manual_seed(seed)
            random.seed(seed)
            try:
                train_d, val_d, test_d = splitter(data)
                split_d = [train_d, val_d, test_d][split_idx]
                split_d.category = cat
                result.append(split_d)
            except Exception:
                continue
            finally:
                torch.set_rng_state(torch_rng_state)
                random.setstate(py_rng_state)
        return result

    train_data = _transform(train_idx, 0)
    val_data   = _transform(val_idx,   1)
    test_data  = _transform(test_idx,  2)

    print(f"  Fold {fold_idx + 1}/{n_folds} — train: {len(train_data)}"
          f"  val: {len(val_data)}  test: {len(test_data)}")
    return train_data, val_data, test_data
