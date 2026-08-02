"""
audit_component_types.py — READ-ONLY corpus audit for the PROPOSED 8-class
component taxonomy:

    Long Shaft · Short Shaft · Thick Plate · Thin Plate · Bolt · Washer · Nut · Body

Classifies every solid body in Source_3d_models/Best_models_for_training using
multi-signal voting (convex-hull ratio + central-axis ray probe for through-holes;
bounding-cylinder fill + center-of-mass offset + B-Rep face count for bolt heads),
side-by-side with the CURRENT pipeline taxonomy (dataset._infer_type_from_geometry)
so threshold tuning can see exactly what moves where.

Touches nothing: no cache writes, no checkpoint reads, no dataset.py changes.

Usage:
    cd back_end
    ../.venv/bin/python audit_component_types.py            # full corpus
    ../.venv/bin/python audit_component_types.py --limit 3  # smoke test

Outputs (written to the corpus root, next to gallery.html):
    audit_classification.json   — full per-body feature + vote + class dump
    audit_classification.html   — visual report (thumbnails, class chips, crosstab)
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import math
import multiprocessing as _mp
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

CORPUS     = _HERE.parent / "Source_3d_models" / "Best_models_for_training"
CATEGORIES = ["Bench_vice", "C_Clamps", "Pipe_vice", "Gate_Valve", "Press_Tool",
              "Tool_Post", "Crane_hook"]
NEW_CLASSES = ["Long Shaft", "Short Shaft", "Thick Plate", "Thin Plate",
               "Bolt", "Washer", "Nut", "Body"]
IMG_EXTS     = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"}
TIMEOUT_SECS = 300

# ── Tunable thresholds (dimensionless; starting points, calibrate via this audit) ──
T = {
    "hull_hole":       0.80,   # V/V_hull below this → hole vote
    "washer_flat":     0.20,   # s/l — washer must be flatter than this
    "washer_plan":     0.75,   # m/l — washer plan must be near-round
    "nut_elong":       1.80,   # l/m — nut must be compact
    "nut_flat_min":    0.12,   # s/l — thinner than this + ring → washer, not nut
    "round_min":       0.60,   # s/m — round cross-section gate (bolt/shaft)
    "bolt_elong_min":  2.0,
    "bolt_elong_max":  8.0,
    "head_fill":       0.65,   # bounding-cylinder fill below this → head vote
    "head_com":        0.06,   # |COM offset|/l above this → head vote
    "head_faces":      8,      # B-Rep face count at/above this → head vote
    "long_shaft_elong": 5.0,   # l/m at/above this → Long Shaft (else Short)
    "thin_flat":       0.08,   # s/l — Thin vs Thick Plate split
    "plate_flat":      0.30,   # s/l — Thick Plate upper bound
    "plate_plan":      0.30,   # m/l — plates must not be strongly elongated in plan
}


def classify_new(b: dict) -> tuple[str, int, int, list]:
    """Proposed-taxonomy classifier. Returns (class, hole_votes, head_votes, notes)."""
    eps = 1e-9
    s, m, l = b["ext"]
    elong = l / (m + eps)
    flat  = s / (l + eps)
    rnd   = s / (m + eps)
    plan  = m / (l + eps)
    notes: list = []

    # Through-hole votes (weak spot 1: two independent signals, require agreement)
    hv = 0
    if b["hull_ratio"] is not None and b["hull_ratio"] < T["hull_hole"]:
        hv += 1
    if b["ray_hits"] == 0:
        hv += 1
    has_hole = hv >= 2
    if hv == 1:
        notes.append("hole-votes-split")

    if has_hole and flat < T["washer_flat"] and plan > T["washer_plan"]:
        return "Washer", hv, 0, notes
    if has_hole and elong < T["nut_elong"] and flat >= T["nut_flat_min"]:
        return "Nut", hv, 0, notes

    # Head votes (weak spot 2: three independent signals, require 2)
    headv = 0
    if b["cyl_fill"] is not None and b["cyl_fill"] < T["head_fill"]:
        headv += 1
    if b["com_offset"] is not None and b["com_offset"] > T["head_com"]:
        headv += 1
    if b["face_count"] >= T["head_faces"]:
        headv += 1

    if rnd > T["round_min"] and elong >= T["bolt_elong_min"]:
        if elong <= T["bolt_elong_max"] and headv >= 2:
            return "Bolt", hv, headv, notes
        if headv == 1:
            notes.append("head-votes-split")
        if elong >= T["long_shaft_elong"]:
            return "Long Shaft", hv, headv, notes
        return "Short Shaft", hv, headv, notes

    if flat < T["thin_flat"] and plan > T["plate_plan"]:
        return "Thin Plate", hv, headv, notes
    if flat < T["plate_flat"] and plan > T["plate_plan"]:
        return "Thick Plate", hv, headv, notes
    return "Body", hv, headv, notes


# ── Per-file worker (child process, mirrors dataset._parse_step's gmsh flow) ─────

def _audit_parse(step_path: str) -> list:
    import gmsh
    import numpy as np
    from dataset import (_build_trimesh, _compute_sdf_stats,
                         _infer_type_from_geometry, COMP_TYPES)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("audit")
    bodies: list = []
    try:
        gmsh.merge(step_path)
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.occ.getEntities(3)
        if not volumes:
            return bodies
        if len(volumes) >= 2:
            gmsh.model.occ.fragment(volumes, [])
            gmsh.model.occ.synchronize()
            volumes = gmsh.model.occ.getEntities(3)

        mesh_ok = True
        try:
            gmsh.model.mesh.generate(2)
        except Exception:
            mesh_ok = False

        for dim, tag in volumes:
            bbox = gmsh.model.occ.getBoundingBox(dim, tag)
            vol  = gmsh.model.occ.getMass(dim, tag)
            try:
                com = np.asarray(gmsh.model.occ.getCenterOfMass(dim, tag))
            except Exception:
                com = None
            dx, dy, dz = bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]
            center = np.array([(bbox[0] + bbox[3]) / 2,
                               (bbox[1] + bbox[4]) / 2,
                               (bbox[2] + bbox[5]) / 2])
            bnd = gmsh.model.getBoundary([(dim, tag)], oriented=False, combined=True)
            surf_tags  = [abs(sf[1]) for sf in bnd if sf[0] == 2]
            face_count = len(surf_tags)

            ext      = sorted([dx, dy, dz])          # AABB fallback
            obb_used = False
            axis_s = axis_l = None
            hull_ratio = None
            ray_hits   = None
            exact_sa   = 2.0 * (dx * dy + dy * dz + dz * dx)
            sdf_m = sdf_v = 0.0

            tm = _build_trimesh(surf_tags) if mesh_ok else None
            if tm is not None and len(tm.faces) >= 4:
                exact_sa = float(tm.area)
                sdf_m, sdf_v = _compute_sdf_stats(tm)
                try:  # oriented bbox → true extents/axes regardless of part rotation
                    obb   = tm.bounding_box_oriented
                    e     = np.asarray(obb.primitive.extents, dtype=float)
                    R     = np.asarray(obb.primitive.transform)[:3, :3]
                    order = np.argsort(e)
                    ext    = [float(e[order[0]]), float(e[order[1]]), float(e[order[2]])]
                    axis_s = R[:, order[0]]
                    axis_l = R[:, order[2]]
                    center = np.asarray(obb.primitive.transform)[:3, 3]
                    obb_used = True
                except Exception:
                    pass
                try:  # hole vote 1: hull fills the hole, solid parts don't lose volume
                    hull_ratio = min(1.5, float(vol / (tm.convex_hull.volume + 1e-12)))
                except Exception:
                    hull_ratio = None

            if axis_s is None:  # AABB fallback axes (world-aligned)
                order  = np.argsort([dx, dy, dz])
                eye    = np.eye(3)
                axis_s = eye[order[0]]
                axis_l = eye[order[2]]

            if tm is not None and len(tm.faces) >= 4:
                try:  # hole vote 2: ray along shortest axis through center → 0 hits = hole
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
                com_offset = abs(float(np.dot(com - center, axis_l))) / (l_ + 1e-9)

            old_idx = _infer_type_from_geometry(vol, exact_sa, (dx, dy, dz), sdf_m, sdf_v)

            bodies.append({
                "tag": int(tag), "vol": float(vol), "sa": float(exact_sa),
                "ext": [round(s_, 4), round(m_, 4), round(l_, 4)],
                "obb": obb_used,
                "hull_ratio": None if hull_ratio is None else round(hull_ratio, 3),
                "ray_hits": ray_hits,
                "cyl_fill": None if cyl_fill is None else round(cyl_fill, 3),
                "com_offset": None if com_offset is None else round(com_offset, 3),
                "face_count": face_count,
                "sdf_mean": round(float(sdf_m), 5), "sdf_var": round(float(sdf_v), 6),
                "old_class": COMP_TYPES[old_idx],
            })
    finally:
        gmsh.finalize()
    return bodies


def _worker(step_path: str, q: "_mp.Queue") -> None:
    try:
        q.put(("ok", _audit_parse(step_path)))
    except Exception as exc:
        q.put(("error", f"{type(exc).__name__}: {exc}"))


def parse_with_timeout(step_path: str, timeout_secs: int = TIMEOUT_SECS):
    """Returns (bodies | None, status) — status: ok | timeout | error."""
    ctx = _mp.get_context("spawn")
    q   = ctx.Queue()
    p   = ctx.Process(target=_worker, args=(step_path, q), daemon=True)
    p.start()
    p.join(timeout=timeout_secs)
    if p.is_alive():
        p.terminate(); p.join(5)
        if p.is_alive():
            p.kill(); p.join()
        return None, "timeout"
    if not q.empty():
        status, payload = q.get_nowait()
        if status == "ok":
            return payload, "ok"
        return None, f"error: {payload}"
    return None, "error: no result (worker died)"


# ── Corpus walk ──────────────────────────────────────────────────────────────────

def collect_folders() -> list:
    out = []
    for cat in CATEGORIES:
        catdir = CORPUS / cat
        if not catdir.is_dir():
            continue
        for folder in sorted(p for p in catdir.iterdir()
                             if p.is_dir() and not p.name.startswith(".")):
            stepf = next((f for f in sorted(folder.iterdir())
                          if f.is_file() and f.suffix.lower() in (".step", ".stp")),
                         None)
            if stepf:
                out.append((cat, folder.name, stepf))
    return out


def first_thumb(cat: str, folder: str) -> str | None:
    imgdir = CORPUS / cat / folder / "Images"
    if imgdir.is_dir():
        for f in sorted(imgdir.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                return f"{quote(cat)}/{quote(folder)}/Images/{quote(f.name)}"
    return None


# ── HTML report ──────────────────────────────────────────────────────────────────

def write_html(records: list, overall: Counter, per_cat: dict,
               crosstab: Counter, n_ambiguous: int, elapsed: float) -> Path:
    esc = html_lib.escape
    total_bodies = sum(overall.values())

    summary_rows = "".join(
        f"<tr><td>{esc(c)}</td><td>{overall.get(c, 0)}</td>"
        f"<td>{100.0 * overall.get(c, 0) / max(1, total_bodies):.1f}%</td></tr>"
        for c in NEW_CLASSES)

    old_classes = sorted({o for (o, _n) in crosstab})
    xhead = "".join(f"<th>{esc(c)}</th>" for c in NEW_CLASSES)
    xrows = ""
    for oc in old_classes:
        cells = "".join(f"<td>{crosstab.get((oc, nc), 0) or ''}</td>"
                        for nc in NEW_CLASSES)
        xrows += f"<tr><th>{esc(oc)}</th>{cells}</tr>"

    sections = ""
    for cat in CATEGORIES:
        recs = [r for r in records if r["category"] == cat]
        if not recs:
            continue
        cards = ""
        for r in recs:
            thumb = first_thumb(r["category"], r["folder"])
            timg = (f'<img src="{thumb}" loading="lazy" alt="">' if thumb
                    else '<div class="noimg">no img</div>')
            if r["status"] != "ok":
                chips = f'<span class="chip bad">{esc(r["status"])}</span>'
                detail = ""
            else:
                cc = Counter(b["new_class"] for b in r["bodies"])
                chips = "".join(
                    f'<span class="chip{" hero" if cls in ("Bolt", "Washer", "Nut") else ""}">'
                    f'{n}× {esc(cls)}</span>'
                    for cls, n in cc.most_common())
                amb = sum(1 for b in r["bodies"] if b["notes"])
                if amb:
                    chips += f'<span class="chip warn">{amb} ambiguous</span>'
                def fmt(v, nd=2):
                    return "" if v is None else f"{v:.{nd}f}"
                rows = "".join(
                    f'<tr{" class=amb" if b["notes"] else ""}>'
                    f'<td>{b["tag"]}</td><td><b>{esc(b["new_class"])}</b></td>'
                    f'<td>{esc(b["old_class"])}</td>'
                    f'<td>{b["ext"][0]:.1f} / {b["ext"][1]:.1f} / {b["ext"][2]:.1f}</td>'
                    f'<td>{b["ext"][2] / (b["ext"][1] + 1e-9):.2f}</td>'
                    f'<td>{b["ext"][0] / (b["ext"][2] + 1e-9):.2f}</td>'
                    f'<td>{fmt(b["hull_ratio"])}</td>'
                    f'<td>{"" if b["ray_hits"] is None else b["ray_hits"]}</td>'
                    f'<td>{fmt(b["cyl_fill"])}</td>'
                    f'<td>{fmt(b["com_offset"])}</td>'
                    f'<td>{b["face_count"]}</td>'
                    f'<td>{esc(", ".join(b["notes"]))}</td></tr>'
                    for b in r["bodies"])
                detail = (
                    '<details><summary>bodies</summary><div class="tw"><table>'
                    '<tr><th>tag</th><th>new</th><th>old</th><th>s/m/l</th>'
                    '<th>elong</th><th>flat</th><th>hull</th><th>ray</th>'
                    '<th>fill</th><th>com</th><th>faces</th><th>notes</th></tr>'
                    f'{rows}</table></div></details>')
            cards += (f'<div class="card">{timg}<div class="meta">'
                      f'<div class="fname">{esc(r["folder"])}</div>'
                      f'<div class="chips">{chips}</div>{detail}</div></div>')
        sections += (f'<section><h2>{esc(cat)} '
                     f'<span class="muted">({len(recs)} folders)</span></h2>'
                     f'<div class="cards">{cards}</div></section>')

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Component Type Audit — Proposed 8-Class Taxonomy</title>
<style>
:root {{ --bg:#f5f7fa; --panel:#fff; --border:#dbe2ea; --text:#1c2733; --muted:#64748b;
        --accent:#3b82f6; --accent-bg:#eaf1fe; --warn:#b45309; --warn-bg:#fef3c7;
        --bad:#b91c1c; --bad-bg:#fee2e2; --hero-bg:#ecfdf5; --hero:#047857; }}
@media (prefers-color-scheme: dark) {{
:root {{ --bg:#0f1520; --panel:#161f2e; --border:#2a3648; --text:#e6edf5; --muted:#8aa0b8;
        --accent:#5b9bd5; --accent-bg:#1b2a3f; --warn:#f0a83c; --warn-bg:#3a2c10;
        --bad:#ef5a5a; --bad-bg:#3a1414; --hero-bg:#0f2e22; --hero:#4caf82; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
       background:var(--bg); color:var(--text); }}
header {{ padding:18px 28px; background:var(--panel); border-bottom:1px solid var(--border); }}
h1 {{ font-size:1.2rem; margin:0 0 4px; }}
.sub {{ color:var(--muted); font-size:0.85rem; }}
main {{ max-width:1400px; margin:0 auto; padding:24px; }}
h2 {{ font-size:1.15rem; margin:26px 0 12px; }}
.muted {{ color:var(--muted); font-weight:400; }}
table {{ border-collapse:collapse; font-size:0.8rem; }}
th,td {{ padding:4px 9px; border-bottom:1px solid var(--border); text-align:left; }}
th {{ color:var(--muted); font-size:0.72rem; text-transform:uppercase; }}
.top {{ display:flex; gap:32px; flex-wrap:wrap; align-items:flex-start; }}
.panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px 18px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:12px; }}
.card {{ background:var(--panel); border:1px solid var(--border); border-radius:10px;
        padding:10px; display:flex; gap:10px; }}
.card img {{ width:64px; height:64px; object-fit:cover; border-radius:6px;
            border:1px solid var(--border); background:#fff; flex-shrink:0; }}
.noimg {{ width:64px; height:64px; border-radius:6px; border:1px dashed var(--border);
         color:var(--muted); font-size:0.65rem; display:flex; align-items:center;
         justify-content:center; flex-shrink:0; }}
.meta {{ min-width:0; flex:1; }}
.fname {{ font-weight:700; font-size:0.85rem; margin-bottom:5px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:4px; margin-bottom:4px; }}
.chip {{ font-size:0.7rem; padding:1px 7px; border-radius:9px;
        background:var(--accent-bg); color:var(--accent); }}
.chip.hero {{ background:var(--hero-bg); color:var(--hero); }}
.chip.warn {{ background:var(--warn-bg); color:var(--warn); }}
.chip.bad {{ background:var(--bad-bg); color:var(--bad); }}
details summary {{ cursor:pointer; font-size:0.75rem; color:var(--muted); }}
.tw {{ overflow-x:auto; max-width:100%; }}
tr.amb td {{ background:var(--warn-bg); }}
</style></head><body>
<header><h1>Component Type Audit — proposed 8-class taxonomy</h1>
<div class="sub">{len(records)} folders · {total_bodies} bodies · {n_ambiguous} ambiguous
(vote conflicts) · generated in {elapsed:.0f}s · thresholds in
back_end/audit_component_types.py</div></header>
<main>
<div class="top">
<div class="panel"><h2 style="margin-top:0;">Class balance (new taxonomy)</h2>
<table><tr><th>Class</th><th>Bodies</th><th>%</th></tr>{summary_rows}</table></div>
<div class="panel"><h2 style="margin-top:0;">Old → New crosstab</h2>
<div class="tw"><table><tr><th>old \\ new</th>{xhead}</tr>{xrows}</table></div></div>
</div>
{sections}
</main></body></html>"""

    out = CORPUS / "audit_classification.html"
    out.write_text(doc)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="audit only the first N folders (smoke test)")
    args = ap.parse_args()

    folders = collect_folders()
    if args.limit:
        folders = folders[:args.limit]
    print(f"Auditing {len(folders)} folders under {CORPUS}", flush=True)

    records = []
    overall  = Counter()
    per_cat  = {c: Counter() for c in CATEGORIES}
    crosstab = Counter()
    n_ambiguous = 0
    t0 = time.time()

    for i, (cat, folder, stepf) in enumerate(folders, 1):
        ft = time.time()
        bodies, status = parse_with_timeout(str(stepf))
        elapsed = time.time() - ft
        rec = {"category": cat, "folder": folder, "status": status,
               "bodies": [], "n_bodies": 0}
        if status == "ok" and bodies:
            for b in bodies:
                cls, hv, headv, notes = classify_new(b)
                b["new_class"]  = cls
                b["hole_votes"] = hv
                b["head_votes"] = headv
                b["notes"]      = notes
                overall[cls] += 1
                per_cat[cat][cls] += 1
                crosstab[(b["old_class"], cls)] += 1
                if notes:
                    n_ambiguous += 1
            rec["bodies"]   = bodies
            rec["n_bodies"] = len(bodies)
            top = ", ".join(f"{n}×{c}" for c, n in
                            Counter(b["new_class"] for b in bodies).most_common(4))
            print(f"[{i}/{len(folders)}] {cat}/{folder}: {len(bodies)} bodies "
                  f"({elapsed:.1f}s) → {top}", flush=True)
        else:
            print(f"[{i}/{len(folders)}] {cat}/{folder}: {status} ({elapsed:.1f}s)",
                  flush=True)
        records.append(rec)

    total_elapsed = time.time() - t0
    total_bodies  = sum(overall.values())

    out_json = CORPUS / "audit_classification.json"
    out_json.write_text(json.dumps({
        "generated":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "thresholds":  T,
        "n_folders":   len(records),
        "n_bodies":    total_bodies,
        "n_ambiguous": n_ambiguous,
        "class_counts": dict(overall),
        "per_category": {c: dict(per_cat[c]) for c in CATEGORIES},
        "crosstab":     {f"{o} -> {n}": v for (o, n), v in sorted(crosstab.items())},
        "folders":      records,
    }, indent=1))
    out_html = write_html(records, overall, per_cat, crosstab,
                          n_ambiguous, total_elapsed)

    print("\n" + "=" * 70, flush=True)
    print(f"TOTAL: {len(records)} folders, {total_bodies} bodies, "
          f"{n_ambiguous} ambiguous, {total_elapsed:.0f}s", flush=True)
    print("\nClass balance (new taxonomy):", flush=True)
    for c in NEW_CLASSES:
        n = overall.get(c, 0)
        print(f"  {c:12s} {n:5d}  {100.0 * n / max(1, total_bodies):5.1f}%", flush=True)
    print(f"\nWrote {out_json}", flush=True)
    print(f"Wrote {out_html}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
