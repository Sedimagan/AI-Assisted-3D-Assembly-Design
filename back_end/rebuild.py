"""
rebuild.py — put the missing parts back, matching the original model.

slot_detector.py says WHERE a missing component goes. This module decides WHAT
it looks like and how it is oriented, and is the single source of geometry for
both viewer groups ("Suggested Shapes" and "Reconstructed"), so the two cannot
disagree.

Shape source, in order of preference:

  1. COPY a surviving instance from the uploaded file. If four of the eight
     machine screws are still fitted, the four missing ones are the same solid.
     Moving the real mesh to the empty seat reproduces shape AND orientation
     exactly, where approximating from a normalised library mesh does not.
     Measured against the original model, copies land within the noise floor of
     two independent triangulations (~0.2-0.6 mm mean chamfer).

  2. RETRIEVE from the part bank when nothing of that family survives. The bank
     is queried by the category's modal design (e.g. 11 of the 12 knob-sized
     bodies across the Tool_Post corpus are 40x40x40) rather than by a guessed
     bounding box.

The bank's stored `center` (a part's position in its ORIGINAL assembly frame)
is never used as an absolute position -- that would "match" any model drawn
from the corpus by reading back its own pose. It is used in exactly one way,
and only for parts with no geometric constraint in the upload (the central
shaft and the knob): as an OFFSET FROM A HOST BODY that the upload also
contains, taken from the most similar sibling assembly. Sibling tool posts
agree on that offset to ~0.05 mm, and with the test model's own source
excluded the siblings still agree with each other, so it is design-family
knowledge rather than memorisation.

Orientation: copies are translated only, which reproduces the survivor's own
orientation. Retrieved parts keep the bank's native (source-assembly) frame,
which is shared across the Tool_Post designs.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import gmsh
except Exception:                                    # pragma: no cover
    gmsh = None

try:
    import trimesh
except Exception:                                    # pragma: no cover
    trimesh = None

import slot_detector as sd

BANK_DIR = Path(__file__).resolve().parent / "data" / "part_bank"

# families for which a surviving instance is copied when one exists
INSTANCE_FAMILIES = ("machine_screw", "bs4183_screw", "din_washer",
                     "ball", "compress_spring")

# comp_type each family is reported as (the vocabulary the rest of the app uses)
TYPE_OF = {
    "machine_screw": "bolt", "bs4183_screw": "bolt", "din_washer": "washer",
    "ball": "body", "compress_spring": "body", "knob": "body",
    "shaft_central": "long_shaft", "shaft_side": "long_shaft",
}

COLOR_OF = {
    "machine_screw": (90, 200, 90),   "bs4183_screw": (70, 140, 235),
    "din_washer": (245, 190, 60),     "compress_spring": (235, 110, 200),
    "ball": (235, 85, 85),            "shaft_central": (80, 210, 210),
    "shaft_side": (255, 140, 40),     "knob": (175, 110, 240),
}

# per-family bank query: comp_types, longest-dimension window (mm)
BANK_QUERY = {
    "shaft_central": (("long_shaft",), (140.0, 180.0)),
    "shaft_side":    (("long_shaft",), (90.0, 125.0)),
    "knob":          (("body",),       (36.0, 44.0)),
}


# ── meshing ──────────────────────────────────────────────────────────────────

def load_part_meshes(step_path: str, mesh_size: float = 1.5,
                     host_min_volume: float = sd.HOST_MIN_VOLUME) -> List[Dict]:
    """Tessellate every solid SMALLER than a host body.

    Host bodies (holder, base plate, clamping nut) are removed from the model
    before meshing: they are large, slow to tessellate, and never copied.
    Returns [{tag, volume, centroid, extents, family, mesh}].
    """
    if gmsh is None or trimesh is None:
        raise RuntimeError("gmsh/trimesh unavailable")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("rebuild")
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()

        info = {}
        for _, tag in gmsh.model.getEntities(3):
            try:
                info[tag] = {
                    "volume": abs(gmsh.model.occ.getMass(3, tag)),
                    "centroid": np.array(gmsh.model.occ.getCenterOfMass(3, tag)),
                    "bb": gmsh.model.getBoundingBox(3, tag),
                }
            except Exception:
                continue

        hosts = [(3, t) for t, i in info.items() if i["volume"] >= host_min_volume]
        if hosts:
            gmsh.model.occ.remove(hosts, recursive=True)
            gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.2)
        gmsh.model.mesh.generate(2)

        out = []
        for _, tag in gmsh.model.getEntities(3):
            if tag not in info or info[tag]["volume"] >= host_min_volume:
                continue
            verts, faces, vmap = [], [], {}
            for _, st in gmsh.model.getBoundary([(3, tag)], oriented=False):
                st = abs(st)
                try:
                    nt, nc, _ = gmsh.model.mesh.getNodes(2, st, includeBoundary=True)
                    _, en = gmsh.model.mesh.getElementsByType(2, st)
                except Exception:
                    continue
                for ntag, xyz in zip(nt, np.array(nc).reshape(-1, 3)):
                    if ntag not in vmap:
                        vmap[ntag] = len(verts)
                        verts.append(xyz)
                for tri in np.array(en, dtype=int).reshape(-1, 3):
                    if all(int(t) in vmap for t in tri):
                        faces.append([vmap[int(t)] for t in tri])
            if not faces:
                continue
            m = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces),
                                process=False)
            bb = info[tag]["bb"]
            out.append({
                "tag": int(tag), "volume": float(info[tag]["volume"]),
                "centroid": info[tag]["centroid"],
                "extents": np.array([bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]),
                "family": sd.family_of_volume(info[tag]["volume"]),
                "mesh": m,
            })
        return out
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass


# ── part bank ────────────────────────────────────────────────────────────────

def _bank_index() -> List[Dict]:
    return json.load(open(BANK_DIR / "index.json"))


def _bank_mesh(entry: Dict):
    """Metric-sized mesh in the part's native source-assembly frame."""
    d = np.load(BANK_DIR / f"{entry['part_id']}.npz", allow_pickle=True)
    v = np.array(d["vertices"], dtype=float)
    ext = v.max(0) - v.min(0)
    ext[ext == 0] = 1.0
    v = (v - v.min(0)) * (np.array(entry["bbox"], dtype=float) / ext)   # true size
    m = trimesh.Trimesh(vertices=v, faces=np.array(d["faces"]), process=False)
    return m


def pick_modal(index: List[Dict], category: str, comp_types, window,
               cross_hint: Optional[float] = None, cross_tol: float = 3.0,
               exclude_assembly: Optional[str] = None,
               assemblies: Optional[set] = None) -> Optional[Dict]:
    """The category's most common design matching the query.

    Groups candidates by rounded size signature, ranks groups by how many
    DISTINCT assemblies contain them (a design present in many tool posts is
    the design, an outlier is not), and returns a representative.
    """
    lo, hi = window
    rows = []
    for p in index:
        if p["category"] != category or p["comp_type"] not in comp_types:
            continue
        if exclude_assembly and p["source_assembly"] == exclude_assembly:
            continue
        if assemblies is not None and p["source_assembly"] not in assemblies:
            continue
        b = sorted(p["bbox"], reverse=True)
        if not (lo <= b[0] <= hi):
            continue
        if cross_hint is not None and abs(float(np.mean(b[1:])) - cross_hint) > cross_tol:
            continue
        rows.append(p)
    if not rows:
        return None
    sig = lambda p: tuple(sorted((round(x / 2.0) * 2 for x in p["bbox"]), reverse=True))
    groups: Dict[tuple, List[Dict]] = {}
    for p in rows:
        groups.setdefault(sig(p), []).append(p)
    best = max(groups.items(),
               key=lambda kv: (len({p["source_assembly"] for p in kv[1]}), len(kv[1]),
                               kv[0]))
    members = best[1]
    mean_b = np.mean([sorted(p["bbox"], reverse=True) for p in members], axis=0)
    return min(members, key=lambda p: float(np.abs(np.array(sorted(p["bbox"], reverse=True))
                                                   - mean_b).sum()))


# ── exact geometry from the corpus STEP files ────────────────────────────────
#
# The part bank stores heavily DECIMATED proxy meshes: measured on the side
# lever, the bank copy has 336 faces enclosing 3,102 mm^3 against the real
# solid's 13,936 faces and 9,024 mm^3 -- it keeps roughly the right bounding
# box and loses two thirds of the material. Simple primitives (a plain shaft,
# a cube-ish knob) survive that; an L-shaped lever does not. So once the bank
# has told us WHICH design to use, the geometry itself is re-read from the
# corpus STEP file that design came from and tessellated at full fidelity.

CORPUS_DIR = Path(__file__).resolve().parent.parent / "Source_3d_models" / "Best_models_for_training"
EXACT_DIR = Path(__file__).resolve().parent / "data" / "part_bank_exact"


def _corpus_step(assembly: str, category: str = "Tool_Post") -> Optional[Path]:
    d = CORPUS_DIR / category / assembly
    if not d.exists():
        return None
    files = [f for f in d.rglob("*") if f.suffix.lower() in (".step", ".stp")]
    files.sort(key=lambda f: (f.stem != assembly, len(str(f))))
    return files[0] if files else None


def _extract_matching_solid(step_path: Path, target_extents, tol: float = 0.08,
                            mesh_size: Optional[float] = None):
    """Tessellate the ONE solid in `step_path` whose size matches `target_extents`.

    Tessellation scales with part size (longest dimension / 60, clamped to
    0.8-2.0 mm). A fixed fine size gave a plain 160 mm shaft 31k faces, and the
    viewer draws every part twice, so that detail cost render time without
    improving the match: the sagitta error at 2 mm on a 12.5 mm radius is 0.04 mm.
    """
    if gmsh is None or trimesh is None:
        return None
    tgt = np.sort(np.asarray(target_extents, float))[::-1]
    if mesh_size is None:
        mesh_size = float(np.clip(tgt[0] / 60.0, 0.8, 2.0))
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("exact")
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()
        best, best_dev = None, 1e9
        ents = gmsh.model.getEntities(3)
        for _, t in ents:
            try:
                bb = gmsh.model.getBoundingBox(3, t)
            except Exception:
                continue
            e = np.sort([bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]])[::-1]
            dev = float(np.max(np.abs(e - tgt) / np.maximum(tgt, 1e-9)))
            if dev < best_dev:
                best, best_dev = t, dev
        if best is None or best_dev > tol:
            return None
        drop = [(3, t) for _, t in ents if t != best]
        if drop:
            gmsh.model.occ.remove(drop, recursive=True)
            gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.2)
        gmsh.model.mesh.generate(2)
        verts, faces, vmap = [], [], {}
        for _, st in gmsh.model.getBoundary([(3, best)], oriented=False):
            st = abs(st)
            try:
                nt, nc, _ = gmsh.model.mesh.getNodes(2, st, includeBoundary=True)
                _, en = gmsh.model.mesh.getElementsByType(2, st)
            except Exception:
                continue
            for ntag, xyz in zip(nt, np.array(nc).reshape(-1, 3)):
                if ntag not in vmap:
                    vmap[ntag] = len(verts); verts.append(xyz)
            for tri in np.array(en, dtype=int).reshape(-1, 3):
                if all(int(x) in vmap for x in tri):
                    faces.append([vmap[int(x)] for x in tri])
        if not faces:
            return None
        return trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=False)
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass


def exact_mesh(entry: Dict, category: str = "Tool_Post"):
    """Full-fidelity mesh for a bank entry, in its native frame; None if unavailable.

    Cached under data/part_bank_exact/ so the STEP file is only read once per
    part. Callers fall back to the coarse bank mesh when this returns None.
    """
    EXACT_DIR.mkdir(parents=True, exist_ok=True)
    cache = EXACT_DIR / f"{entry['part_id']}.npz"
    if cache.exists():
        d = np.load(cache)
        return trimesh.Trimesh(vertices=np.array(d["vertices"], float),
                               faces=np.array(d["faces"]), process=False)
    step = _corpus_step(entry["source_assembly"], category)
    if step is None:
        return None
    m = _extract_matching_solid(step, entry["bbox"])
    if m is None:
        return None
    np.savez_compressed(cache, vertices=np.asarray(m.vertices, np.float32),
                        faces=np.asarray(m.faces, np.int64))
    return m


# ── supplementary library (parts the bank's watertight gate rejects) ──────────

SUPPLEMENT_DIR = Path(__file__).resolve().parent / "data" / "part_bank_supplement"


def _supplement_index() -> List[Dict]:
    f = SUPPLEMENT_DIR / "index.json"
    return json.load(open(f)) if f.exists() else []


def _supplement_mesh(entry: Dict):
    d = np.load(SUPPLEMENT_DIR / f"{entry['part_id']}.npz", allow_pickle=True)
    return trimesh.Trimesh(vertices=np.array(d["vertices"], float),
                           faces=np.array(d["faces"]), process=False)


def similar_assemblies(index: List[Dict], category: str, signatures,
                       exclude: Optional[str] = None, tol: float = 0.03,
                       min_score: float = 0.5) -> Optional[set]:
    """Bank assemblies that look most like the upload.

    Scores each assembly by the fraction of the upload's solid size-signatures
    that also occur among its parts, and returns the top scorers (ties kept).
    None when nothing scores >= min_score -- the caller then falls back to the
    whole category rather than trusting a weak match.

    Why this instead of a plain majority vote: the Tool_Post corpus holds two
    different side-lever designs, 3 assemblies each. The design that fits the
    upload lives in the assemblies scoring 0.86; the other lives in ones that
    score 0.0. A vote tie between them was being broken arbitrarily -- and
    wrongly, giving a 10mm shape error.
    """
    boxes: Dict[str, List[np.ndarray]] = {}
    for p in index:
        if p["category"] != category or p["source_assembly"] == exclude:
            continue
        boxes.setdefault(p["source_assembly"], []).append(
            np.array(sorted(p["bbox"], reverse=True), float))
    sigs = [np.array(sorted(s, reverse=True), float) for s in signatures]
    if not boxes or not sigs:
        return None
    scores = {}
    for asm, bxs in boxes.items():
        hit = 0
        for t in sigs:
            den = np.maximum(t, 1e-9)
            if any(np.all(np.abs(b - t) / den < tol) for b in bxs):
                hit += 1
        scores[asm] = hit / len(sigs)
    best = max(scores.values())
    if best < min_score:
        return None
    return {a for a, sc in scores.items() if sc >= best - 1e-9}


# ── template transfer (parts with no geometric constraint in the upload) ─────

def _bank_center(part_id: str) -> np.ndarray:
    return np.array(np.load(BANK_DIR / f"{part_id}.npz", allow_pickle=True)["center"], float)


def template_center(index: List[Dict], pick: Dict, host_boxes: List[Dict],
                    tol: float = 0.03) -> Optional[np.ndarray]:
    """Where `pick` belongs in THIS upload, from its offset to a shared host.

    In the assembly `pick` came from, find the body matching the upload's
    largest host (same size signature) and take part_center - host_center.
    Apply that offset to the upload's own copy of the host. None when the
    sibling has no matching host, so the caller falls back to geometry.
    """
    if not host_boxes:
        return None
    ref = host_boxes[0]
    tgt = np.array(ref["extents"], float)
    cands = []
    for p in index:
        if p["source_assembly"] != pick["source_assembly"] or p["part_id"] == pick["part_id"]:
            continue
        b = np.array(sorted(p["bbox"], reverse=True), float)
        if np.all(np.abs(b - tgt) / np.maximum(tgt, 1e-9) < tol):
            cands.append(p)
    if not cands:
        return None
    host = min(cands, key=lambda p: float(np.abs(np.array(sorted(p["bbox"], reverse=True)) - tgt).sum()))
    offset = _bank_center(pick["part_id"]) - _bank_center(host["part_id"])
    return np.array(ref["center"], float) + offset


# ── placement helpers ────────────────────────────────────────────────────────

def _long_axis(mesh) -> int:
    return int(np.argmax(mesh.extents))


def _rotate_axis_to(mesh, src_axis: int, dst_axis: int):
    """Rotate `mesh` so its `src_axis` lies along `dst_axis` (about its centre)."""
    if src_axis == dst_axis:
        return mesh
    a, b = np.eye(3)[src_axis], np.eye(3)[dst_axis]
    m = mesh.copy()
    c = m.bounds.mean(axis=0)
    m.apply_translation(-c)
    m.apply_transform(trimesh.geometry.align_vectors(a, b))
    m.apply_translation(c)
    return m


def _chamfer_to_bbox_fit(mesh, target_extents) -> float:
    """1.0 = identical bounding-box signature."""
    a = np.sort(np.asarray(mesh.extents, float))[::-1]
    b = np.sort(np.asarray(target_extents, float))[::-1]
    den = np.maximum(a, b)
    den[den == 0] = 1e-9
    return float(1.0 - np.mean(np.abs(a - b) / den))


# ── the entry point ──────────────────────────────────────────────────────────

def rebuild_missing(step_path: str, category: str = "Tool_Post",
                    exclude_assembly: Optional[str] = None,
                    mesh_size: float = 1.5) -> Dict:
    """Locate every missing part and produce a placed mesh for each.

    Returns {"recognised": bool, "holes": n, "empty": n, "placed": [...]} where
    each placed entry is
      {family, type, source, part_id, fit_score, confidence, centroid, mesh}.
    `mesh` is a trimesh.Trimesh in world coordinates.
    """
    res = sd.find_missing_slots(step_path)
    out = {"recognised": res["recognised"], "holes": res["holes"],
           "empty": res["empty"], "placed": []}
    if not res["recognised"]:
        return out

    parts = load_part_meshes(step_path, mesh_size=mesh_size)
    survivors: Dict[str, List[Dict]] = {}
    for p in parts:
        survivors.setdefault(p["family"], []).append(p)

    index = _bank_index()
    sim = similar_assemblies(index, category, res.get("solid_extents", []),
                             exclude=exclude_assembly)
    out["similar_assemblies"] = sorted(sim) if sim else []
    placed: List[Dict] = []

    # ---- (1) copy surviving instances into every empty seat ----
    for sl in res["slots"]:
        fam = sl.get("family")
        if fam not in INSTANCE_FAMILIES or not survivors.get(fam):
            continue
        src = survivors[fam][0]
        m = src["mesh"].copy()
        m.apply_translation(np.asarray(sl["centroid"], float) - src["centroid"])
        placed.append({"family": fam, "type": TYPE_OF[fam], "source": "copied",
                       "part_id": f"upload:{src['tag']}", "fit_score": 1.0,
                       "confidence": 1.0, "centroid": list(map(float, sl["centroid"])),
                       "seat": sl.get("source", ""), "mesh": m})

    # ---- (1b) springs: none survive, and the bank cannot hold them ----
    #      part_bank.py rejects non-watertight meshes and a helical spring's
    #      swept surface never is, so springs come from the supplementary
    #      library (back_end/build_supplement.py), restricted to the tool posts
    #      most like this upload and to a coil that fits the seat's bore.
    spring_slots = [sl for sl in res["slots"] if sl.get("family") == "compress_spring"]
    if spring_slots and not survivors.get("compress_spring"):
        bore = spring_slots[0].get("diam") or 10.0
        sup = [e for e in _supplement_index()
               if e["category"] == category and e["source_assembly"] != exclude_assembly
               and abs(float(np.mean(sorted(e["extents"], reverse=True)[1:])) - (bore - 1.0)) <= 2.5]
        pool = [e for e in sup if sim and e["source_assembly"] in sim] or sup
        if pool:
            sig = lambda e: (round(e["volume"] / 5.0), tuple(round(x) for x in sorted(e["extents"], reverse=True)))
            groups: Dict[tuple, List[Dict]] = {}
            for e in pool:
                groups.setdefault(sig(e), []).append(e)
            members = max(groups.values(),
                          key=lambda g: (len({e["source_assembly"] for e in g}), len(g)))
            entry = members[0]
            base = _supplement_mesh(entry)
            base = _rotate_axis_to(base, _long_axis(base), 2)       # coil axis on Z
            for sl in spring_slots:
                m = base.copy()
                m.apply_translation(np.asarray(sl["centroid"], float) - m.bounds.mean(axis=0))
                placed.append({"family": "compress_spring", "type": "body",
                               "source": "retrieved", "part_id": entry["part_id"],
                               "fit_score": 0.9, "confidence": 0.9,
                               "centroid": list(map(float, sl["centroid"])),
                               "seat": sl.get("source", ""), "mesh": m})

    # ---- (2) central shaft: nothing of it survives -> modal bank design ----
    for sl in res["slots"]:
        if sl.get("family") != "shaft_central":
            continue
        comp, win = BANK_QUERY["shaft_central"]
        cross = (sl["diam"] - 1.0) if sl.get("diam") else None
        pick = pick_modal(index, category, comp, win, cross_hint=cross,
                          exclude_assembly=exclude_assembly, assemblies=sim)
        if pick is None:
            pick = pick_modal(index, category, comp, win, cross_hint=cross,
                              exclude_assembly=exclude_assembly)
        if pick is None:
            continue
        m = exact_mesh(pick, category) or _bank_mesh(pick)
        m = _rotate_axis_to(m, _long_axis(m), 2)                # shaft stands on Z
        tc = template_center(index, pick, res.get("host_boxes", []))
        target = tc if tc is not None else np.asarray(sl["centroid"], float)
        m.apply_translation(target - m.bounds.mean(axis=0))
        placed.append({"family": "shaft_central", "type": "long_shaft",
                       "source": "retrieved", "part_id": pick["part_id"],
                       "fit_score": 0.9, "confidence": 0.9,
                       "centroid": list(map(float, m.bounds.mean(axis=0))),
                       "seat": "sibling template" if tc is not None else "bore", "mesh": m})

    # ---- (3) side shaft + knob, anchored on the empty mount hole ----
    mount = next((sl for sl in res["slots"] if sl.get("family") == "shaft_side"), None)
    if mount is not None:
        comp, win = BANK_QUERY["shaft_side"]
        pick = (pick_modal(index, category, comp, win,
                           exclude_assembly=exclude_assembly, assemblies=sim)
                or pick_modal(index, category, comp, win,
                              exclude_assembly=exclude_assembly))
        if pick is not None:
            m = exact_mesh(pick, category) or _bank_mesh(pick)
            m = _rotate_axis_to(m, _long_axis(m), mount["axis"])   # along the hole axis
            hc = np.asarray(mount["centroid"], float)
            r = (mount.get("diam") or 10.0) / 2.0
            half_len = (mount.get("length") or 12.0) / 2.0
            # insertion tip: flush with the hole's entry face along the axis,
            # one hole-radius clear of the axis in the two other directions
            corner = hc.copy()
            corner[mount["axis"]] = hc[mount["axis"]] - half_len
            for k in range(3):
                if k != mount["axis"]:
                    corner[k] = hc[k] - r
            m.apply_translation(corner - m.bounds[0])
            placed.append({"family": "shaft_side", "type": "long_shaft",
                           "source": "retrieved", "part_id": pick["part_id"],
                           "fit_score": 0.85, "confidence": 0.85,
                           "centroid": list(map(float, m.bounds.mean(axis=0))),
                           "seat": "mount hole", "mesh": m})

            # knob: on the lever's free tip = shaft vertex farthest from the mount
            comp_k, win_k = BANK_QUERY["knob"]
            kpick = (pick_modal(index, category, comp_k, win_k,
                                exclude_assembly=exclude_assembly, assemblies=sim)
                     or pick_modal(index, category, comp_k, win_k,
                                   exclude_assembly=exclude_assembly))
            if kpick is not None:
                km = exact_mesh(kpick, category) or _bank_mesh(kpick)
                v = np.asarray(m.vertices)
                tip = v[int(np.argmax(np.linalg.norm(v - hc, axis=1)))]
                ktc = template_center(index, kpick, res.get("host_boxes", []))
                km.apply_translation((ktc if ktc is not None else tip) - km.bounds.mean(axis=0))
                placed.append({"family": "knob", "type": "body",
                               "source": "retrieved", "part_id": kpick["part_id"],
                               "fit_score": 0.85, "confidence": 0.85,
                               "centroid": list(map(float, km.bounds.mean(axis=0))),
                               "seat": "sibling template" if ktc is not None else "shaft tip",
                               "mesh": km})

    for p in placed:
        p["color"] = COLOR_OF.get(p["family"], (150, 150, 150))
        rgba = list(p["color"]) + [255]
        p["mesh"].visual.face_colors = rgba
    out["placed"] = placed
    return out


def export_glb(placed: List[Dict], path: str) -> None:
    scene = trimesh.Scene()
    for i, p in enumerate(placed):
        name = f"recon_{p['family']}_{i}"
        scene.add_geometry(p["mesh"], node_name=name, geom_name=name)
    scene.export(path)
