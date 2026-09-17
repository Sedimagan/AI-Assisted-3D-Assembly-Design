"""
slot_detector.py — where does a missing component actually go?

Phase 1 tells us a component is missing and Phase 2 tells us what type it is,
but neither produces a 6-DOF pose, so a generated part has nowhere to sit. This
module supplies the placement.

Two mechanisms, both reading ONLY the uploaded partial assembly:

  1. Empty-hole detection. Cylindrical faces on the large "host" bodies are
     grouped into distinct holes and tested for occupancy. An unoccupied hole is
     a mounting site. This is the primary mechanism and is what recovers seats
     the surviving parts cannot point at -- on the reference tool post the
     missing screws sit at radius 84.85 while the survivors sit at 60, so
     rotating the survivors would never have found them.

  2. N-fold rotational symmetry of surviving fasteners, for seats whose own hole
     is concealed underneath a part that is still fitted (washers and the
     screws they sit under, typically).

SCOPE (2026-09-17): the diameter bands in TOOLPOST_FAMILIES are calibrated
against the Tool_Post corpus. They are applied only when classify_family()
recognises a tool-post-like assembly; every other category falls through to
generic slot reporting with no family assignment, which is the pre-existing
behaviour. Deriving these bands per-assembly is the generalisation step and is
deliberately not attempted here.
"""
from __future__ import annotations

import math
from typing import List, Dict, Optional

import numpy as np

try:
    import gmsh
except Exception:                                    # pragma: no cover
    gmsh = None


# Host bodies are the things holes are drilled INTO. Everything smaller is a
# candidate occupant. 100k mm^3 separates the tool holder / base plate /
# clamping nut from every fastener in the reference corpus.
HOST_MIN_VOLUME = 100_000.0


# ── family model (tool-post calibrated) ──────────────────────────────────────
# name -> (diam_lo, diam_hi, axis, z_lo, z_hi, comp_type, colour)
TOOLPOST_FAMILIES = {
    "machine_screw":   (13.0, 14.5, 2,  80.0, 120.0, "bolt",        (90, 200, 90)),
    "ball":            ( 9.0, 11.0, 2,  25.0,  45.0, None,          (235, 85, 85)),
    "compress_spring": ( 9.0, 11.0, 2,  25.0,  45.0, None,          (235, 110, 200)),
    "shaft_central":   (24.0, 28.0, 2,  50.0, 100.0, "long_shaft",  (80, 210, 210)),
}

# Families whose seats are hidden under surviving parts -> symmetry instead.
SYMMETRY_FAMILIES = {
    "bs4183_screw": (70, 140, 235),
    "din_washer":   (245, 190, 60),
}

# Volume signature -> family, for classifying what is ALREADY in the file.
FAMILY_BY_VOLUME = [
    (1_263_852.4, "host_tool_holder"),
    (518_376.2,   "host_base_plate"),
    (102_981.1,   "host_clamping_nut"),
    (15_498.0,    "machine_screw"),
    (9_677.4,     "bs4183_screw"),
    (1_439.6,     "din_washer"),
    (494.1,       "compress_spring"),
    (179.6,       "ball"),
]


def family_of_volume(vol: float, tol: float = 0.01) -> str:
    for v, name in FAMILY_BY_VOLUME:
        if abs(vol - v) / max(v, 1e-9) < tol:
            return name
    return "unknown"


def looks_like_toolpost(solids: List[Dict]) -> bool:
    """True when the assembly matches the tool-post signature the family bands
    were calibrated on. Gate for applying those bands at all."""
    fams = {family_of_volume(s["volume"]) for s in solids}
    return "host_tool_holder" in fams and "host_base_plate" in fams


# ── hole detection ───────────────────────────────────────────────────────────

def detect_holes(step_path: str, host_min_volume: float = HOST_MIN_VOLUME) -> List[Dict]:
    """Distinct cylindrical holes on host bodies, each flagged occupied or not.

    Caller is responsible for gmsh not already being initialised.
    """
    if gmsh is None:
        raise RuntimeError("gmsh unavailable")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("slotdet")
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()

        solids = []
        for _, tag in gmsh.model.getEntities(3):
            try:
                vol = abs(gmsh.model.occ.getMass(3, tag))
                com = np.array(gmsh.model.occ.getCenterOfMass(3, tag))
                bb = gmsh.model.getBoundingBox(3, tag)
            except Exception:
                continue
            solids.append({
                "tag": int(tag), "volume": float(vol), "centroid": com,
                "extents": np.array([bb[3]-bb[0], bb[4]-bb[1], bb[5]-bb[2]]),
                "bbox_min": np.array(bb[:3]), "bbox_max": np.array(bb[3:]),
            })
        hosts = [s for s in solids if s["volume"] >= host_min_volume]
        parts = [s for s in solids if s["volume"] < host_min_volume]

        raw = []
        for h in hosts:
            try:
                bnd = gmsh.model.getBoundary([(3, h["tag"])], oriented=False)
            except Exception:
                continue
            for _, st in bnd:
                st = abs(st)
                try:
                    if "ylinder" not in gmsh.model.getType(2, st):
                        continue
                    c = np.array(gmsh.model.occ.getCenterOfMass(2, st))
                    fbb = gmsh.model.getBoundingBox(2, st)
                except Exception:
                    continue
                ext = np.array([fbb[3]-fbb[0], fbb[4]-fbb[1], fbb[5]-fbb[2]])
                order = np.argsort(ext)
                axis = int(order[-1])
                diam = float(np.mean([ext[order[0]], ext[order[1]]]))
                length = float(ext[axis])
                if diam <= 0.5 or length <= 0.5:
                    continue
                raw.append({"centroid": c, "diam": diam,
                            "length": length, "axis": axis})

        # a physical hole shows up as several coaxial cylindrical faces
        merged: List[Dict] = []
        for r in raw:
            hit = None
            for m in merged:
                if m["axis"] == r["axis"] and abs(m["diam"] - r["diam"]) < 0.75:
                    lateral = np.delete(m["centroid"] - r["centroid"], r["axis"])
                    if np.linalg.norm(lateral) < 1.5:
                        hit = m
                        break
            if hit:
                hit["n"] += 1
                hit["centroid"] = (hit["centroid"] * (hit["n"] - 1)
                                   + r["centroid"]) / hit["n"]
                hit["length"] = max(hit["length"], r["length"])
            else:
                r = dict(r); r["n"] = 1
                merged.append(r)

        for m in merged:
            m["occupied"] = False
            for p in parts:
                d = p["centroid"] - m["centroid"]
                lateral = np.delete(d, m["axis"])
                if (np.linalg.norm(lateral) < max(m["diam"] * 0.75, 4.0)
                        and abs(d[m["axis"]]) < max(m["length"], 40.0)):
                    m["occupied"] = True
                    break

        for m in merged:
            m["centroid"] = [float(x) for x in m["centroid"]]
        return merged, solids
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass


# ── symmetry ─────────────────────────────────────────────────────────────────

def _rot_z(p, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])


def infer_by_symmetry(present: List[np.ndarray], n_fold: int = 4,
                      tol: float = 1.5) -> List[np.ndarray]:
    """Seats generated by rotating surviving parts that are not themselves
    occupied. Recovers fastener seats concealed under fitted parts."""
    out: List[np.ndarray] = []
    for c in present:
        c = np.asarray(c, dtype=float)
        for k in range(1, n_fold):
            q = _rot_z(c, 360.0 * k / n_fold)
            if any(np.linalg.norm(q - np.asarray(p, float)) < tol for p in present):
                continue
            if any(np.linalg.norm(q - e) < tol for e in out):
                continue
            out.append(q)
    return out


# ── the thing the app calls ──────────────────────────────────────────────────

def find_missing_slots(step_path: str) -> Dict:
    """Locate every seat in `step_path` that should hold a part but does not.

    Returns {"recognised": bool, "slots": [...], "holes": n, "empty": n}.
    Each slot: {family, centroid[3], axis, diam, source, comp_type, color}.

    `recognised` is False for assemblies outside the calibrated tool-post
    family model; slots are then reported without family labels.
    """
    merged, solids = detect_holes(step_path)
    empty = [m for m in merged if not m["occupied"]]
    result = {"holes": len(merged), "empty": len(empty),
              "recognised": False, "slots": []}

    if not looks_like_toolpost(solids):
        result["slots"] = [
            {"family": None, "centroid": m["centroid"], "axis": m["axis"],
             "diam": m["diam"], "source": "hole", "comp_type": None,
             "color": (150, 150, 150)}
            for m in empty
        ]
        return result

    result["recognised"] = True
    by_fam: Dict[str, List[Dict]] = {}
    for s in solids:
        by_fam.setdefault(family_of_volume(s["volume"]), []).append(s)

    # Real extents of each family that still has a surviving instance. Callers
    # retrieving from the part bank should size against THIS rather than
    # guessing from hole diameter -- guessing dropped washer retrieval fit from
    # 1.00 to 0.65 because a washer's thickness bears no relation to its bore.
    result["family_extents"] = {
        fam: sorted((float(x) for x in lst[0]["extents"]), reverse=True)
        for fam, lst in by_fam.items()
        if lst and not fam.startswith("host") and fam != "unknown"
    }

    def present_z(fam, default):
        lst = by_fam.get(fam, [])
        return float(np.mean([s["centroid"][2] for s in lst])) if lst else default

    slots = []

    # (1) hole-derived seats
    for fam, (dlo, dhi, axis, zlo, zhi, ctype, col) in TOOLPOST_FAMILIES.items():
        if fam == "compress_spring":
            continue                       # handled below: every seat, not just empty
        for m in empty:
            c = m["centroid"]
            if (m["axis"] == axis and dlo <= m["diam"] <= dhi
                    and zlo <= c[2] <= zhi):
                z = present_z(fam, c[2]) if fam in ("machine_screw", "ball") else c[2]
                slots.append({"family": fam, "centroid": [c[0], c[1], z],
                              "axis": axis, "diam": m["diam"], "source": "hole",
                              "comp_type": ctype, "color": col})

    # (2) springs ride above the balls, so every seat of that bore gets one --
    #     including bores whose ball is still fitted.
    dlo, dhi, axis, zlo, zhi, ctype, col = TOOLPOST_FAMILIES["compress_spring"]
    for m in merged:
        c = m["centroid"]
        if m["axis"] == axis and dlo <= m["diam"] <= dhi and zlo <= c[2] <= zhi:
            slots.append({"family": "compress_spring", "centroid": list(c),
                          "axis": axis, "diam": m["diam"], "source": "hole",
                          "comp_type": ctype, "color": col})

    # (3) seats hidden under surviving parts
    for fam, col in SYMMETRY_FAMILIES.items():
        present = [s["centroid"] for s in by_fam.get(fam, [])]
        for q in infer_by_symmetry(present, n_fold=4):
            slots.append({"family": fam, "centroid": [float(x) for x in q],
                          "axis": 2, "diam": None, "source": "symmetry",
                          "comp_type": "bolt" if "screw" in fam else "washer",
                          "color": col})

    result["slots"] = slots
    return result


def summarise(result: Dict) -> str:
    if not result["slots"]:
        return "No unoccupied mounting slots detected."
    from collections import Counter
    c = Counter(s["family"] or "unclassified" for s in result["slots"])
    parts = ", ".join(f"{n}x {f.replace('_', ' ')}" for f, n in c.most_common())
    tag = "" if result["recognised"] else " (uncalibrated category)"
    return f"{len(result['slots'])} missing-component slots{tag}: {parts}"


# ── orientation ──────────────────────────────────────────────────────────────

def head_end_sign(mesh, axis: int = 2, nbins: int = 12, frac: float = 0.25) -> int:
    """Which end of a part is the HEAD, i.e. physically wider?

    +1 when the wide end faces +axis, -1 when it faces -axis, 0 when the part
    is axially symmetric and orientation does not matter.

    Slices along `axis` and compares mean lateral spread of the outermost bins.
    A volume-centroid test was tried first and got the tool-post screws
    backwards: the bank screw and the fitted screw have different mass
    distributions, so their centroid offsets agreed in sign while the parts
    were visually inverted. Cross-sectional width reads the head directly.
    """
    v = np.asarray(mesh.vertices, dtype=float)
    a = v[:, axis]
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-9:
        return 0
    lat = np.delete(v, axis, axis=1)
    edges = np.linspace(lo, hi, nbins + 1)
    widths = []
    for i in range(nbins):
        m = (a >= edges[i]) & (a <= edges[i + 1])
        if m.sum() < 3:
            widths.append(np.nan)
            continue
        w = lat[m]
        widths.append(float(np.linalg.norm(w.max(axis=0) - w.min(axis=0))))
    widths = np.array(widths, dtype=float)
    k = max(1, int(round(nbins * frac)))
    top, bot = np.nanmean(widths[-k:]), np.nanmean(widths[:k])
    if not np.isfinite(top) or not np.isfinite(bot):
        return 0
    if abs(top - bot) < 0.02 * max(top, bot):
        return 0
    return 1 if top > bot else -1


def orient_like(mesh, reference_sign: int, axis: int = 2):
    """Flip `mesh` 180 deg if its head points opposite to `reference_sign`."""
    import trimesh
    if not reference_sign:
        return mesh
    got = head_end_sign(mesh, axis)
    if not got or got == reference_sign:
        return mesh
    m = mesh.copy()
    c = m.bounds.mean(axis=0)
    m.apply_translation(-c)
    about = (1.0, 0.0, 0.0) if axis == 2 else (0.0, 0.0, 1.0)
    m.apply_transform(trimesh.transformations.rotation_matrix(math.pi, about))
    m.apply_translation(c)
    return m
