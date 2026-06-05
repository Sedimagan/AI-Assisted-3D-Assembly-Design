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
  #   centroid, area, area_ratio, body_idx, vertices, triangles, normal_hint
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

def analyze_open_surfaces(
    step_path: str,
    area_ratio_threshold: float = 0.04,
    max_surfaces: int = 4,
) -> List[Dict]:
    """
    Identify open (unmated) surfaces in a STEP file and cluster them into
    spatial regions using an Octree — adapting the block-based regional
    approach from Borah & Borah (2020) for assembly completeness analysis.

    After boolean fragment(), surfaces shared by ≥2 solid bodies are internal
    mating joints.  Remaining free surfaces that form a significant fraction
    of their parent body's area are flagged as potential open assembly joints:
    locations where a missing component should be placed.

    Parameters
    ----------
    step_path            : absolute path to the STEP / STP file
    area_ratio_threshold : min (surface_area / body_total_SA) to consider
    max_surfaces         : maximum number of regions returned

    Returns
    -------
    List[dict] — one entry per detected open-joint region:
      centroid     : [x, y, z]   centre of the surface bounding box
      area         : float        surface area (gmsh units)
      area_ratio   : float        fraction of parent body's total SA
      body_idx     : int          0-based index of the parent solid body
      vertices     : [[x,y,z]…]  triangle mesh vertices
      triangles    : [[i,j,k]…]  triangle mesh faces
      normal_hint  : [nx, ny, nz] approximate surface normal direction
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

        # ── Identify free surfaces above area threshold ───────────────────────
        candidates: List[Dict] = []
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
                area_ratio = area / body_total_sa[body_idx]

                if area_ratio < area_ratio_threshold:
                    continue              # fillet / chamfer — ignore

                # Approximate surface normal: direction of thinnest bbox dim
                dims = [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]
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

        if not candidates:
            return []

        # ── Octree spatial grouping ───────────────────────────────────────────
        # Partition the bounding volume of all candidate centroids so that
        # nearby open surfaces are clustered into a single "missing region".
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
        leaf_reps: List[Dict] = []
        for leaf_items in root.leaves():
            if not leaf_items:
                continue
            _, best = max(leaf_items, key=lambda x: x[1]["area_ratio"])
            leaf_reps.append(best)

        # Sort by area_ratio descending and cap
        leaf_reps.sort(key=lambda x: x["area_ratio"], reverse=True)
        leaf_reps = leaf_reps[:max_surfaces]

        # ── Surface mesh generation for the flagged surfaces ──────────────────
        mesh_ok = False
        if leaf_reps:
            try:
                gmsh.model.mesh.generate(2)
                mesh_ok = True
            except Exception:
                pass

        # ── Build and return results ──────────────────────────────────────────
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
            })

        return results

    except Exception:
        return []

    finally:
        gmsh.finalize()
