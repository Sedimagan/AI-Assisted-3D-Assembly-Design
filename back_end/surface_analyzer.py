"""
surface_analyzer.py — Open Surface Detection via Octree Spatial Partitioning

Conceptually derived from the Octree-based regional analysis used in:
  Borah S. & Borah B., "Prediction Error Expansion (PEE) based Reversible
  polygon mesh watermarking scheme for regional tamper localization",
  Multimedia Tools and Applications, Vol. 79, pp. 11437-11458, 2020.
  DOI: 10.1007/s11042-019-08411-5

In that work, an Octree divides a 3-D mesh's bounding volume into independent
spatial sub-regions so that tampering can be localised to specific blocks.
Here we adapt the same partitioning idea for assembly analysis:

  • After gmsh boolean fragment(), surfaces shared by two volumes become
    *internal* (mated joints).  Surfaces belonging to only one volume are
    *free* boundary surfaces.
  • An Octree partitions the free-surface centroids into spatial clusters.
  • Within each cluster, the surface with the highest area-ratio (surface area
    relative to its parent body's total surface area) is selected as the most
    likely open assembly joint.
  • The triangulated mesh of that surface is extracted and returned so the
    front-end can render it in amber, annotated "Needs Assembly".

Usage
-----
  from surface_analyzer import analyze_open_surfaces
  records = analyze_open_surfaces("/path/to/assembly.step")
  # records: list of dicts with keys:
  #   centroid, area, area_ratio, body_idx, vertices, triangles, normal_hint, bbox
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Lightweight 3-D Octree ────────────────────────────────────────────────────

class OctreeNode:
    """
    Minimal 3-D octree for spatial grouping of surface centroids.

    Each leaf holds all items inserted into its sub-volume.  Splitting
    stops when the subtree reaches max_depth or the leaf holds ≤ max_items.
    This mirrors the block-based spatial decomposition used in the Borah &
    Borah (2020) PEE watermarking paper for localising mesh regions.
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float, float, float],
        depth: int = 0,
        max_depth: int = 3,
        max_items: int = 4,
    ):
        self.bounds    = bounds          # (xmin, ymin, zmin, xmax, ymax, zmax)
        self.depth     = depth
        self.max_depth = max_depth
        self.max_items = max_items
        self.items:    List                    = []   # [(centroid, payload), …]
        self.children: Optional[List[OctreeNode]] = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _contains(self, pt) -> bool:
        x, y, z = pt
        xmin, ymin, zmin, xmax, ymax, zmax = self.bounds
        return (xmin <= x <= xmax and
                ymin <= y <= ymax and
                zmin <= z <= zmax)

    def _mid(self):
        xmin, ymin, zmin, xmax, ymax, zmax = self.bounds
        return (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2

    def _child_bounds(self):
        xmin, ymin, zmin, xmax, ymax, zmax = self.bounds
        mx, my, mz = self._mid()
        return [
            (xmin, ymin, zmin, mx,   my,   mz  ),
            (mx,   ymin, zmin, xmax, my,   mz  ),
            (xmin, my,   zmin, mx,   ymax, mz  ),
            (mx,   my,   zmin, xmax, ymax, mz  ),
            (xmin, ymin, mz,   mx,   my,   zmax),
            (mx,   ymin, mz,   xmax, my,   zmax),
            (xmin, my,   mz,   mx,   ymax, zmax),
            (mx,   my,   mz,   xmax, ymax, zmax),
        ]

    def _split(self):
        self.children = [
            OctreeNode(b, self.depth + 1, self.max_depth, self.max_items)
            for b in self._child_bounds()
        ]
        for item in self.items:
            self._route(item)
        self.items = []

    def _route(self, item):
        for child in self.children:
            if child._contains(item[0]):
                child.insert(*item)
                return

    # ── Public API ────────────────────────────────────────────────────────────

    def insert(self, centroid, payload) -> None:
        if not self._contains(centroid):
            return
        if self.children is not None:
            self._route((centroid, payload))
        else:
            self.items.append((centroid, payload))
            if len(self.items) > self.max_items and self.depth < self.max_depth:
                self._split()

    def leaves(self) -> List[List]:
        """Return all non-empty leaf item lists."""
        if self.children is None:
            return [self.items] if self.items else []
        out = []
        for child in self.children:
            out.extend(child.leaves())
        return out


# ── Surface mesh extraction ───────────────────────────────────────────────────

def _extract_surface_mesh(
    surf_tag: int,
    max_triangles: int = 300,
) -> Tuple[List, List]:
    """
    Extract a triangulated mesh for one gmsh surface tag.

    Requires gmsh.model.mesh.generate(2) to have been called beforehand.
    Returns (vertices, triangles) where each vertex is [x, y, z] and each
    triangle is [i, j, k] (0-based indices into the vertex list).

    Triangle count is capped at max_triangles to keep JSON payload small.
    """
    import gmsh
    import random as _rng

    vertices: List   = []
    triangles: List  = []
    nmap: Dict       = {}

    try:
        elem_types, _, ntags_all = gmsh.model.mesh.getElements(2, surf_tag)
        for etype, ntags in zip(elem_types, ntags_all):
            # type 2 = 3-node triangle; type 9 = 6-node (second-order) triangle
            if etype == 2:
                stride = 3
            elif etype == 9:
                stride = 6   # use only the 3 corner nodes (first 3 per element)
            else:
                continue
            arr = list(ntags)
            for k in range(0, len(arr), stride):
                tri = arr[k : k + 3]   # always take first 3 node IDs
                fvids = []
                for nid in tri:
                    if nid not in nmap:
                        coords = gmsh.model.mesh.getNode(nid)[0]
                        nmap[nid] = len(vertices)
                        vertices.append([float(coords[0]),
                                         float(coords[1]),
                                         float(coords[2])])
                    fvids.append(nmap[nid])
                triangles.append(fvids)
    except Exception:
        pass

    # Cap triangles to keep the result lightweight
    if len(triangles) > max_triangles:
        _rng.shuffle(triangles)
        triangles = triangles[:max_triangles]

    return vertices, triangles


# ── Synthetic grid mesh fallback ─────────────────────────────────────────────

def _synthetic_surface_mesh(
    bbox: List,
    normal: List,
    grid_res: int = 10,
) -> Tuple[List, List]:
    """
    Generate a rectangular grid mesh from the surface bounding box and normal.

    Used when gmsh mesh generation produces no triangles so the 3D viewer
    always has a visible lime-green patch at each open joint location.
    The patch is a flat grid projected onto the plane perpendicular to the
    thinnest bounding-box dimension (the approximate surface normal axis).
    """
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = (zmin + zmax) / 2

    # Determine which axis is the normal
    try:
        normal_axis = normal.index(1.0)
    except ValueError:
        normal_axis = 2

    pts: List = []
    n = grid_res
    if normal_axis == 0:        # surface ⊥ X — span Y and Z
        us = [ymin + (ymax - ymin) * i / (n - 1) for i in range(n)]
        vs = [zmin + (zmax - zmin) * j / (n - 1) for j in range(n)]
        pts = [[cx, u, v] for u in us for v in vs]
    elif normal_axis == 1:      # surface ⊥ Y — span X and Z
        us = [xmin + (xmax - xmin) * i / (n - 1) for i in range(n)]
        vs = [zmin + (zmax - zmin) * j / (n - 1) for j in range(n)]
        pts = [[u, cy, v] for u in us for v in vs]
    else:                       # surface ⊥ Z — span X and Y
        us = [xmin + (xmax - xmin) * i / (n - 1) for i in range(n)]
        vs = [ymin + (ymax - ymin) * j / (n - 1) for j in range(n)]
        pts = [[u, v, cz] for u in us for v in vs]

    tris: List = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            tris.append([a, b, c])
            tris.append([b, d, c])

    return pts, tris


# ── Main analysis function ────────────────────────────────────────────────────

# ── Fastener-hole detection tuning ───────────────────────────────────────────
# Plausible bolt/screw hole diameter range (mm) — generous: covers small screws
# up to a large clamping-nut-scale bore, calibrated against real corpus parts
# (a 62mm hex clamping nut, ~6-30mm bolt-hole diameters observed in practice).
HOLE_DIAM_MIN = 2.0
HOLE_DIAM_MAX = 50.0
# Faces thinner than this (mm) along the bore axis are witness-mark / fragment
# slivers, not real functional holes (observed: 0.1mm-deep artifact faces with
# a "plausible" diameter that would otherwise false-positive).
HOLE_MIN_DEPTH = 1.0
# A real fastener-hole pattern repeats (a bolt circle, a 4-corner pattern, …).
# A one-off cylindrical bore of a "plausible" diameter (e.g. a central tool
# bore or shaft bore) is a single unmatched instance — excluded by requiring
# at least this many same-diameter, same-body candidates before trusting any
# of them as a fastener hole.
HOLE_MIN_PATTERN_COUNT = 2
# Coaxial holes (e.g. a counterbore + its narrower clearance bore) round-trip
# through this many mm of XY(-perpendicular) position tolerance to be merged
# into one hole candidate instead of double-counting the same physical hole.
HOLE_MERGE_TOLERANCE = 3.0

# Combined axial span (across a hole's coaxial B-Rep faces) needed, as a
# fraction of the parent body's own thickness along the bore axis, to call a
# hole "through" rather than blind. Calibrated against a real STEP file's
# corner mounting holes on a 34mm plate: their measurable combined depth
# only reaches ~73.5% of the plate thickness (the last few mm at the very
# entry face are apparently a non-cylindrical face -- a chamfer or flare --
# so they don't get picked up by the Cylinder-type face scan), even though
# they're unambiguously through-holes by design. A genuinely blind hole
# measured only ~18% on the same file, so there's wide margin either
# direction -- 0.65 isn't cutting it close for either case.
HOLE_THROUGH_DEPTH_RATIO = 0.65

# A non-through hole this shallow (mm) isn't a real blind hole for a
# fastener to thread into -- it reads as a surface indentation/dimple (a
# countersink start, a locating dimple, a shallow machining mark), and a
# generated bolt has no business sitting there. Same threshold and
# rationale as dataset.py's _INDENTATION_MAX_DEPTH (duplicated locally
# per this module's own no-cross-import relationship with dataset.py).
# Candidates at or under this depth are dropped entirely from hole_reps
# rather than returned as a fastener-worthy is_hole=True candidate.
HOLE_INDENTATION_MAX_DEPTH = 3.0

# A flat face's own bbox extent along the bore axis, and how close its
# along-axis coordinate must sit to a hole end's own coordinate, to count
# as "this hole opens onto a flat face" (mm). Same values and rationale as
# dataset.py's _FLAT_FACE_THICKNESS_TOL/_FLAT_FACE_LEVEL_TOL (duplicated
# locally per this module's own no-cross-import relationship with
# dataset.py).
HOLE_FLAT_FACE_THICKNESS_TOL = 2.0
HOLE_FLAT_FACE_LEVEL_TOL = 2.0


def _hole_diameter_depth(dims: List[float]) -> Tuple[float, float]:
    """Given a cylindrical face's 3 bbox extents, split into (diameter, depth).
    Two of the three extents are ~equal (the circular cross-section); the odd
    one out is the depth along the bore axis. Robust to depth being either
    larger (a deep through-hole) or smaller (a shallow counterbore) than the
    diameter — picks whichever extreme is farther from the middle value."""
    d = sorted(dims)
    diameter = d[1]
    depth = d[0] if (d[1] - d[0]) > (d[2] - d[1]) else d[2]
    return diameter, depth


def _hole_axis(tag: int) -> List[float]:
    """Derive a cylindrical face's true bore axis via two parametric point
    samples (constant u, differing v) — robust for shallow/wide holes where
    the bbox-thinness heuristic used for whole-face normal_hint breaks down.
    Sign-normalized so the largest-magnitude component is positive, matching
    the existing normal_hint convention."""
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


def analyze_open_surfaces(
    step_path: str,
    area_ratio_threshold: float = 0.04,
    max_surfaces: int = 4,
    max_hole_surfaces: int = 64,
) -> List[Dict]:
    """
    Identify open (unmated) surfaces in a STEP file: both whole-face regions
    (clustered via an Octree — adapting the block-based regional approach
    from Borah & Borah 2020) and individual fastener-hole candidates.

    After boolean fragment(), surfaces shared by ≥2 solid bodies are internal
    mating joints.  Remaining free surfaces that form a significant fraction
    of their parent body's area are flagged as potential open assembly joints
    (whole-face regions) — locations where a missing non-fastener component
    should be placed.

    Separately, every free *cylindrical* face is checked as an individual
    bolt/screw-hole candidate — these are collected independently of the
    whole-face path because a single hole's wall area is almost always far
    below area_ratio_threshold (calibrated for "big face vs. fillet noise",
    not "small hole vs. big body"), and because an octree keeping only one
    representative per spatial leaf would otherwise collapse an 8-hole bolt
    pattern down to 1. Hole candidates are filtered by plausible diameter/
    depth and by requiring the diameter to repeat elsewhere on the same body
    (real fastener patterns repeat; one-off functional bores like a central
    tool/shaft bore don't) — see HOLE_* constants above.

    Parameters
    ----------
    step_path            : absolute path to the STEP / STP file
    area_ratio_threshold : min (surface_area / body_total_SA) for whole-face candidates
    max_surfaces         : maximum number of whole-face regions returned
    max_hole_surfaces    : safety cap on returned per-hole candidates (not a
                            normal-case limit — real bolt patterns are far below this)

    Returns
    -------
    List[dict] — one entry per detected open-joint region (whole-face regions
    followed by individual hole candidates). Three categories of candidate
    are dropped entirely before this list is built, none should get a
    generated bolt/nut/washer: a non-through candidate whose combined depth
    is <= HOLE_INDENTATION_MAX_DEPTH (a shallow dimple/indentation, not a
    real blind hole); any candidate where another body's own center of
    mass already sits on the hole's bore axis (see _hole_is_occupied — a
    fastener, or any other part, already occupies it); and any candidate
    that doesn't open onto a flat face at either end (see
    _hole_end_is_on_flat_face — a chamfer/fillet along a curved bend can
    itself be a Cylinder-type B-Rep face and pass the diameter/depth
    filters purely by coincidence, a confirmed real prior bug):
      centroid     : [x, y, z]   centre of the surface bounding box
      area         : float        surface area (gmsh units)
      area_ratio   : float        fraction of parent body's total SA
      body_idx     : int          0-based index of the parent solid body
      vertices     : [[x,y,z]…]  triangle mesh vertices
      triangles    : [[i,j,k]…]  triangle mesh faces
      normal_hint  : [nx, ny, nz] approximate surface normal / bore-axis direction
      bbox         : [xmin,ymin,zmin,xmax,ymax,zmax] surface bounding box
      is_hole      : bool         True for individual fastener-hole candidates
      is_through   : bool         hole candidates only -- True if the hole's
                                   combined axial span (across its merged
                                   coaxial faces) covers >=HOLE_THROUGH_DEPTH_RATIO (65%) of the parent
                                   body's own thickness along the bore axis,
                                   i.e. it likely goes all the way through
                                   rather than dead-ending partway (a blind
                                   hole). Approximate -- no exit-face
                                   verification, just a depth-vs-thickness
                                   ratio. Always False for whole-face entries.
      exit_face_coord : float|None  hole candidates only -- the parent body's
                                   own coordinate (along the bore axis's
                                   dominant world component) at its far
                                   extreme, opposite the entry face. Lets a
                                   caller place something at the far side of
                                   a through-hole without needing the body's
                                   own bbox. None on error or for whole-face
                                   entries.
      shaft_diameter : float      hole candidates only -- diameter of the
                                   NARROWER coaxial face when this candidate
                                   is a counterbore (wide recess + narrow
                                   clearance bore merged into one entry, see
                                   HOLE_MERGE_TOLERANCE); equals the plain
                                   diameter when there's no separate
                                   narrower face. Use this, not the bbox's
                                   own in-plane extent, to size a bolt's
                                   shaft -- the merged bbox/diameter reflect
                                   the counterbore's wide opening, which a
                                   shaft must NOT be sized to.
      counterbore_depth : float|None  hole candidates only -- how deep the
                                   wide recess face itself goes (its own
                                   bbox extent along the bore axis), i.e.
                                   how far a bolt head should sink in to sit
                                   recessed within the counterbore rather
                                   than resting on the outer surface. None
                                   when this candidate isn't a counterbore.
      is_on_flat_surface : bool     hole candidates only -- always True in
                                   this returned list (a False candidate
                                   is dropped before it gets here, see
                                   above); kept as an explicit field
                                   rather than removed so a caller doesn't
                                   need to assume it from a candidate's
                                   mere presence.
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 8.0)   # coarse mesh = fast
    gmsh.model.add("open_surf_analysis")

    try:
        gmsh.merge(step_path)
        gmsh.model.occ.synchronize()
        vols = gmsh.model.occ.getEntities(3)
        if not vols:
            return []

        # ── Boolean fragment → shared surfaces become internal ────────────────
        gmsh.model.occ.fragment(vols, [])
        gmsh.model.occ.synchronize()
        vols = gmsh.model.occ.getEntities(3)     # re-fetch after fragment

        # ── Map surface tags → body indices (0-based) ─────────────────────────
        surf_to_body_idxs: Dict[int, List[int]] = {}
        body_to_surfs:     Dict[int, List[int]] = {}

        for idx, (dim, tag) in enumerate(vols):
            bnd = gmsh.model.getBoundary([(dim, tag)], oriented=False, combined=True)
            stags = [abs(s[1]) for s in bnd if s[0] == 2]
            body_to_surfs[idx] = stags
            for st in stags:
                surf_to_body_idxs.setdefault(st, []).append(idx)

        # Total surface area per body (for computing area_ratio)
        body_total_sa: Dict[int, float] = {}
        for idx in body_to_surfs:
            total = 0.0
            for st in body_to_surfs[idx]:
                try:
                    total += gmsh.model.occ.getMass(2, st)
                except Exception:
                    pass
            body_total_sa[idx] = total or 1.0

        # ── Identify free surfaces: whole-face candidates + hole candidates ────
        candidates: List[Dict] = []    # whole-face (existing octree/cap path)
        hole_raw:   List[Dict] = []    # individual fastener-hole candidates
        for st, body_idxs in surf_to_body_idxs.items():
            if len(body_idxs) != 1:
                continue                   # shared (mated) — skip
            body_idx = body_idxs[0]
            try:
                area = gmsh.model.occ.getMass(2, st)
                bb   = gmsh.model.occ.getBoundingBox(2, st)
                cx   = (bb[0] + bb[3]) / 2
                cy   = (bb[1] + bb[4]) / 2
                cz   = (bb[2] + bb[5]) / 2
                dims = [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]
                area_ratio = area / body_total_sa[body_idx]

                # ── Individual fastener-hole candidate (cylindrical faces) ────
                # Independent of area_ratio_threshold — a hole wall's own area
                # is almost never a significant fraction of the whole body's
                # surface area, so that threshold (calibrated for whole faces)
                # would always reject real holes.
                try:
                    is_cylinder = gmsh.model.getType(2, st) == "Cylinder"
                except Exception:
                    is_cylinder = False
                if is_cylinder:
                    diameter, depth = _hole_diameter_depth(dims)
                    if HOLE_DIAM_MIN <= diameter <= HOLE_DIAM_MAX and depth >= HOLE_MIN_DEPTH:
                        hole_raw.append({
                            "stag":       st,
                            "body_idx":   body_idx,
                            "centroid":   [cx, cy, cz],
                            "area":       area,
                            "area_ratio": area_ratio,
                            "bbox":       list(bb),
                            "diameter":   diameter,
                        })

                # ── Whole-face candidate (unchanged) ───────────────────────────
                if area_ratio < area_ratio_threshold:
                    continue              # fillet / chamfer — ignore

                # Approximate surface normal: direction of thinnest bbox dim
                min_ax = dims.index(min(dims))
                normal = [0.0, 0.0, 0.0]
                normal[min_ax] = 1.0

                candidates.append({
                    "stag":       st,
                    "body_idx":   body_idx,
                    "centroid":   [cx, cy, cz],
                    "area":       area,
                    "area_ratio": area_ratio,
                    "bbox":       list(bb),
                    "normal":     normal,
                })
            except Exception:
                pass

        # ── Whole-face Octree spatial grouping (unchanged) ─────────────────────
        # Partition the bounding volume of all candidate centroids so that
        # nearby open surfaces are clustered into a single "missing region".
        leaf_reps: List[Dict] = []
        if candidates:
            eps = 1e-6
            all_cx = [c["centroid"][0] for c in candidates]
            all_cy = [c["centroid"][1] for c in candidates]
            all_cz = [c["centroid"][2] for c in candidates]

            root = OctreeNode((
                min(all_cx) - eps, min(all_cy) - eps, min(all_cz) - eps,
                max(all_cx) + eps, max(all_cy) + eps, max(all_cz) + eps,
            ), max_depth=3)

            for cand in candidates:
                root.insert(cand["centroid"], cand)

            # From each Octree leaf, keep the representative with highest area_ratio
            for leaf_items in root.leaves():
                if not leaf_items:
                    continue
                _, best = max(leaf_items, key=lambda x: x[1]["area_ratio"])
                leaf_reps.append(best)

            # Sort by area_ratio descending and cap
            leaf_reps.sort(key=lambda x: x["area_ratio"], reverse=True)
            leaf_reps = leaf_reps[:max_surfaces]

        # ── Fastener-hole pattern filter + coaxial-duplicate merge ─────────────
        # Individual holes bypass octree reduction entirely (each is a genuine
        # separate joint, not a redundant duplicate of a neighboring region) and
        # aren't capped by max_surfaces — only require:
        #   (a) diameter/depth already filtered above, and
        #   (b) the diameter repeats elsewhere on the same body (a real bolt
        #       pattern has ≥2 instances; a one-off functional bore — e.g. a
        #       central tool/shaft bore — doesn't and gets excluded here).
        # Survivors are then merged if coaxial and co-located (a counterbore's
        # wide face + its narrower clearance-bore face are two topological
        # faces for the same physical hole — kept once, using the wider face
        # as the representative "opening").
        hole_reps: List[Dict] = []
        if hole_raw:
            # _hole_axis needed for every raw candidate now, not just the
            # pattern-filtered ones below -- the unfiltered set is reused for
            # the is_through depth check (see below), specifically because a
            # hole's true clearance bore can be a one-off, non-repeating
            # diameter that the pattern filter is *supposed* to reject when
            # picking bolt-hole candidates (that filter's job, unchanged
            # here), but which is still real geometry worth counting toward
            # how deep the hole actually goes.
            for h in hole_raw:
                try:
                    h["normal"] = _hole_axis(h["stag"])
                except Exception:
                    dims = [h["bbox"][3] - h["bbox"][0],
                            h["bbox"][4] - h["bbox"][1],
                            h["bbox"][5] - h["bbox"][2]]
                    min_ax = dims.index(min(dims))
                    h["normal"] = [1.0 if i == min_ax else 0.0 for i in range(3)]

            def _perp_key(h: Dict) -> Tuple:
                axis = h["normal"]
                perp = [round(v / HOLE_MERGE_TOLERANCE)
                        for i, v in enumerate(h["centroid"]) if abs(axis[i]) < 0.5]
                return (h["body_idx"], tuple(perp))

            # Unfiltered lookup: every raw candidate face (any diameter, any
            # repeat count) bucketed by body + coaxial position -- used only
            # for the depth/is_through measurement below, never for deciding
            # which holes are real bolt-pattern candidates.
            raw_by_perp: Dict[Tuple, List[Dict]] = {}
            for h in hole_raw:
                raw_by_perp.setdefault(_perp_key(h), []).append(h)

            by_diam: Dict[Tuple[int, float], List[Dict]] = {}
            for h in hole_raw:
                key = (h["body_idx"], round(h["diameter"] * 2) / 2)
                by_diam.setdefault(key, []).append(h)
            patterned = [h for group in by_diam.values()
                         if len(group) >= HOLE_MIN_PATTERN_COUNT
                         for h in group]

            # Merge coaxial duplicates (pattern-filtered set only -- this
            # selects which hole *candidates* get returned, unchanged from
            # before): group by body + position projected onto the plane
            # perpendicular to the bore axis (axis-aligned holes are the norm
            # in this corpus, matching the existing normal_hint convention
            # used for whole-face candidates above).
            merge_groups: Dict[Tuple, List[Dict]] = {}
            for h in patterned:
                merge_groups.setdefault(_perp_key(h), []).append(h)

            # Through vs blind: compare the COMBINED axial span across ALL
            # raw coaxial faces at this position (not just the pattern-
            # filtered ones -- see raw_by_perp above) against the parent
            # body's own extent along the bore axis -- see
            # HOLE_THROUGH_DEPTH_RATIO for the threshold and its calibration.
            # An approximation (no exit-face topology verification), since
            # this project doesn't have exact material-thickness data to
            # check against directly for a body that isn't itself
            # axis-aligned.
            body_bbox: Dict[int, list] = {}

            def _body_bbox(body_idx: int) -> list:
                if body_idx not in body_bbox:
                    dim, tag = vols[body_idx]
                    body_bbox[body_idx] = list(gmsh.model.occ.getBoundingBox(dim, tag))
                return body_bbox[body_idx]

            def _body_extent_along(body_idx: int, axis: List[float]) -> float:
                bb = _body_bbox(body_idx)
                dom = max(range(3), key=lambda i: abs(axis[i]))
                return bb[dom + 3] - bb[dom]

            body_com: Dict[int, List[float]] = {}

            def _body_com(body_idx: int) -> List[float]:
                if body_idx not in body_com:
                    dim, tag = vols[body_idx]
                    try:
                        body_com[body_idx] = list(gmsh.model.occ.getCenterOfMass(dim, tag))
                    except Exception:
                        bb = _body_bbox(body_idx)
                        body_com[body_idx] = [(bb[k] + bb[k + 3]) / 2 for k in range(3)]
                return body_com[body_idx]

            def _hole_is_occupied(centroid: List[float], axis: List[float],
                                   diameter: float, own_body_idx: int) -> bool:
                """Is a fastener (or any other part) already sitting in this
                hole, so it isn't actually an empty/missing-component
                candidate? A bolt/screw genuinely inserted through a hole is
                coaxial with it, so its own center of mass sits close to the
                hole's bore AXIS LINE regardless of how far along that axis
                the part extends (most of a bolt's head can sit proud above
                the surface, or only a short tip may reach into a blind hole
                -- axial position varies, but perpendicular offset from the
                axis stays small either way). Checking perpendicular offset
                only, not axial position, is deliberate: a clearance-fit
                bolt (shaft modeled slightly narrower than the hole, common
                in real CAD) doesn't share an exact mating surface with the
                hole wall, so it wouldn't already be excluded by this
                function's caller (free/unmated-surface detection) the way
                an exact-fit bolt would be.

                Both bounds are relative to the HOLE's own diameter, not the
                parent body's overall size -- an earlier version bounded the
                axial check to the parent body's own extent (times 2), which
                on a real repro file falsely matched a washer sitting at one
                hole's position against a completely different, unrelated
                hole ~90mm away on the same body, just because their XY
                positions happened to coincide (a common occurrence in
                symmetric/grid mechanical layouts). A real inserted fastener
                doesn't extend many diameters past the hole it occupies, so
                bounding both checks to the hole's own diameter avoids that
                false positive while still catching genuine occupancy."""
                n = axis
                for idx in range(len(vols)):
                    if idx == own_body_idx:
                        continue
                    com = _body_com(idx)
                    v = [com[k] - centroid[k] for k in range(3)]
                    along = sum(v[k] * n[k] for k in range(3))
                    perp = [v[k] - along * n[k] for k in range(3)]
                    perp_dist = sum(p * p for p in perp) ** 0.5
                    if perp_dist <= diameter * 0.6 and abs(along) <= diameter * 4.0:
                        return True
                return False

            def _hole_end_is_on_flat_face(body_idx: int, end_pt: List[float],
                                           dom_axis: int, hole_radius: float) -> bool:
                """Does this hole end open onto a flat (planar) face rather
                than a curved one (a shaft's cylindrical outer wall, a
                curved housing, a fillet)? Fasteners almost always mount on
                flat surfaces -- a hole breaking through a curved one is
                more likely something else (a lubrication port, a vent, a
                cross-drilled pin hole). Duplicated from dataset.py's
                identical helper (same no-cross-import relationship);
                bbox-proximity heuristic rather than exact B-Rep topology,
                consistent with the rest of this function's hole detection:
                look for a planar boundary face on the body whose own bbox
                is thin along the bore axis and sits at this end's own
                level, with the hole's in-plane position inside that
                face's footprint."""
                dim, tag = vols[body_idx]
                bnd = gmsh.model.getBoundary([(dim, tag)], oriented=False, combined=True)
                other_axes = [i for i in range(3) if i != dom_axis]
                for bd, bt in bnd:
                    if bd != 2:
                        continue
                    ftag = abs(bt)
                    try:
                        if gmsh.model.getType(2, ftag) != "Plane":
                            continue
                        bb = gmsh.model.occ.getBoundingBox(2, ftag)
                    except Exception:
                        continue
                    if (bb[dom_axis + 3] - bb[dom_axis]) > HOLE_FLAT_FACE_THICKNESS_TOL:
                        continue
                    face_level = (bb[dom_axis] + bb[dom_axis + 3]) / 2
                    if abs(face_level - end_pt[dom_axis]) > HOLE_FLAT_FACE_LEVEL_TOL:
                        continue
                    if all(bb[a] - hole_radius <= end_pt[a] <= bb[a + 3] + hole_radius
                           for a in other_axes):
                        return True
                return False

            for key, group in merge_groups.items():
                rep = max(group, key=lambda h: h["diameter"])
                axis = rep["normal"]
                dom = max(range(3), key=lambda i: abs(axis[i]))
                depth_group = raw_by_perp.get(key, group)  # unfiltered if available
                near = min(h["bbox"][dom] for h in depth_group)
                far  = max(h["bbox"][dom + 3] for h in depth_group)
                combined_depth = far - near
                try:
                    body_extent = _body_extent_along(rep["body_idx"], axis)
                except Exception:
                    body_extent = combined_depth  # can't compare -- assume blind
                rep["is_through"] = (body_extent > 1e-6
                                      and (combined_depth / body_extent) >= HOLE_THROUGH_DEPTH_RATIO)

                if not rep["is_through"] and combined_depth <= HOLE_INDENTATION_MAX_DEPTH:
                    continue  # indentation/dimple, not a real blind hole -- no fastener belongs here

                if _hole_is_occupied(rep["centroid"], axis, rep["diameter"], rep["body_idx"]):
                    continue  # a fastener already sits here -- not an empty/missing-component candidate

                # A round chamfer/fillet along a bend can itself be a
                # Cylinder-type B-Rep face (a partial-revolution surface,
                # same face type gmsh reports for a real drilled hole) and
                # pass the diameter/depth filters above purely by
                # coincidence -- confirmed as a real prior bug: a bolt got
                # generated on a chamfered bend that was never a hole at
                # all. A genuine fastener hole always opens onto a flat
                # face at (at least) one end; a chamfer/fillet along a
                # curved bend generally doesn't. Hard filter, not just
                # informational -- reported 2026-08-24 with a concrete
                # past-bug example, upgraded from the softer treatment
                # this started with. Falls back to True (don't drop) only
                # if the check itself errors, since an inconclusive result
                # shouldn't silently kill a real hole.
                near_pt = list(rep["centroid"])
                far_pt = list(rep["centroid"])
                near_pt[dom], far_pt[dom] = near, far
                hole_radius = rep["diameter"] / 2.0
                try:
                    rep["is_on_flat_surface"] = (
                        _hole_end_is_on_flat_face(rep["body_idx"], near_pt, dom, hole_radius)
                        or _hole_end_is_on_flat_face(rep["body_idx"], far_pt, dom, hole_radius)
                    )
                except Exception:
                    rep["is_on_flat_surface"] = True  # inconclusive -- don't drop a possibly-real hole

                if not rep["is_on_flat_surface"]:
                    continue  # curved surroundings (e.g. a chamfer/fillet along a bend) -- not a real fastener hole

                # Exit-face coordinate (parent body's own far extreme along
                # the bore axis, on the OPPOSITE side from the entry -- entry
                # sits at the body/hole's extreme in the +normal_hint
                # direction, matching app.py's own entry-centroid fix, so
                # exit is the body's extreme in the -normal_hint direction).
                # Only meaningful when is_through, but harmless to compute
                # either way. Lets a caller place a nut/washer at the far
                # side of a through-bolt without needing the body's own
                # bbox itself.
                try:
                    bb = _body_bbox(rep["body_idx"])
                    rep["exit_face_coord"] = bb[dom] if axis[dom] > 0 else bb[dom + 3]
                except Exception:
                    rep["exit_face_coord"] = None

                # A counterbore's wide recess face + its narrower clearance-
                # bore face merge into one candidate above (rep = the wide
                # one, kept as the representative "opening" for pattern-
                # matching/placement purposes -- unchanged). But a bolt's
                # SHAFT must be sized to the narrow bore it actually passes
                # through, not the counterbore's own width, and its head
                # should sit recessed within the counterbore specifically
                # rather than sized/placed off the merged (wide) opening as
                # if it were a plain uniform-diameter hole. Expose the
                # narrow face's diameter and the wide face's own depth
                # (i.e. how far the recess itself goes) separately so a
                # caller can size/place a bolt correctly through a
                # counterbore. Both fields equal the plain single-diameter
                # case's values when there's no separate narrower face.
                if len(group) > 1:
                    narrow = min(group, key=lambda h: h["diameter"])
                    rep["shaft_diameter"] = narrow["diameter"]
                    rep["counterbore_depth"] = rep["bbox"][dom + 3] - rep["bbox"][dom]
                else:
                    rep["shaft_diameter"] = rep["diameter"]
                    rep["counterbore_depth"] = None

                hole_reps.append(rep)

            hole_reps = hole_reps[:max_hole_surfaces]

        if not leaf_reps and not hole_reps:
            return []

        # ── Surface mesh generation for the flagged surfaces ──────────────────
        mesh_ok = False
        if leaf_reps or hole_reps:
            try:
                gmsh.model.mesh.generate(2)
                mesh_ok = True
            except Exception:
                pass

        # ── Build and return results (whole-face regions, then hole candidates) ─
        results: List[Dict] = []
        for rep in leaf_reps:
            verts, tris = [], []
            if mesh_ok:
                verts, tris = _extract_surface_mesh(rep["stag"])
            # Always fall back to synthetic grid so the 3D viewer has a visible patch
            if not verts or not tris:
                verts, tris = _synthetic_surface_mesh(rep["bbox"], rep["normal"])

            results.append({
                "centroid":    rep["centroid"],
                "area":        round(rep["area"], 4),
                "area_ratio":  round(rep["area_ratio"], 4),
                "body_idx":    rep["body_idx"],
                "vertices":    verts,
                "triangles":   tris,
                "normal_hint": rep["normal"],
                "bbox":        rep["bbox"],
                "is_hole":     False,
            })

        for rep in hole_reps:
            verts, tris = [], []
            if mesh_ok:
                verts, tris = _extract_surface_mesh(rep["stag"])
            if not verts or not tris:
                verts, tris = _synthetic_surface_mesh(rep["bbox"], rep["normal"])

            results.append({
                "centroid":    rep["centroid"],
                "area":        round(rep["area"], 4),
                "area_ratio":  round(rep["area_ratio"], 4),
                "body_idx":    rep["body_idx"],
                "vertices":    verts,
                "triangles":   tris,
                "normal_hint": rep["normal"],
                "bbox":        rep["bbox"],
                "is_hole":     True,
                "is_through":  bool(rep.get("is_through", False)),
                "exit_face_coord": rep.get("exit_face_coord"),
                "shaft_diameter": rep.get("shaft_diameter", rep["diameter"]),
                "counterbore_depth": rep.get("counterbore_depth"),
                "is_on_flat_surface": rep.get("is_on_flat_surface"),
            })

        return results

    except Exception:
        return []

    finally:
        gmsh.finalize()
