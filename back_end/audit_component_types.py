"""
audit_component_types.py — READ-ONLY corpus audit for the live 8-class
component taxonomy (dataset.COMP_TYPES):

    Long Shaft · Short Shaft · Thick Plate · Thin Plate · Bolt · Washer · Nut · Body

Classifies every solid body in Source_3d_models/Best_models_for_training using
the same multi-signal voting classifier the live pipeline uses
(dataset._compute_shape_signals + dataset._classify_component_type: convex-hull
ratio + central-axis ray probe for through-holes; bounding-cylinder fill +
center-of-mass offset + B-Rep face count for bolt heads) — a standalone,
full-corpus class-balance/ambiguity report and threshold-tuning sandbox
(edit dataset._TYPE_THRESHOLDS, re-run, see what shifts) without needing a
full training run.

Touches nothing: no cache writes, no checkpoint reads, no dataset.py changes.

Usage:
    cd back_end
    ../.venv/bin/python audit_component_types.py            # full corpus
    ../.venv/bin/python audit_component_types.py --limit 3  # smoke test

Outputs (written to the corpus root, next to gallery.html):
    audit_classification.json   — full per-body feature + vote + class dump
    audit_classification.html   — visual report (thumbnails, class chips)
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import multiprocessing as _mp
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dataset import COMP_TYPES, _TYPE_THRESHOLDS  # noqa: E402

CORPUS     = _HERE.parent / "Source_3d_models" / "Best_models_for_training"
CATEGORIES = ["Bench_vice", "C_Clamps", "Pipe_vice", "Gate_Valve", "Press_Tool",
              "Tool_Post", "Crane_hook"]
# Human-readable display labels, in dataset.COMP_TYPES index order.
NEW_CLASSES = [t.replace("_", " ").title() for t in COMP_TYPES]
IMG_EXTS     = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"}
TIMEOUT_SECS = 300
T = _TYPE_THRESHOLDS  # local alias — kept for quick inline threshold experiments

# ── Per-file worker (child process, mirrors dataset._parse_step's gmsh flow) ─────
#
# Classification itself (_compute_shape_signals + _classify_component_type) now
# lives in dataset.py — this worker only re-derives the raw per-body signals for
# reporting/display and calls the shared classifier, so the audit always reflects
# exactly what the live pipeline would produce.

def _audit_parse(step_path: str) -> list:
    import gmsh
    import numpy as np
    from dataset import (_build_trimesh, _compute_sdf_stats,
                         _compute_shape_signals, _classify_component_type)

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

            exact_sa = 2.0 * (dx * dy + dy * dz + dz * dx)  # AABB fallback
            sdf_m = sdf_v = 0.0

            tm = _build_trimesh(surf_tags) if mesh_ok else None
            if tm is not None and len(tm.faces) >= 4:
                exact_sa = float(tm.area)
                sdf_m, sdf_v = _compute_sdf_stats(tm)

            signals = _compute_shape_signals(
                tm, vol, (dx, dy, dz), center, com, face_count)
            type_idx, hv, headv, notes = _classify_component_type(signals)
            s_, m_, l_ = signals["ext"]

            bodies.append({
                "tag": int(tag), "vol": float(vol), "sa": float(exact_sa),
                "ext": [round(s_, 4), round(m_, 4), round(l_, 4)],
                "obb": tm is not None and len(tm.faces) >= 4,
                "hull_ratio": signals["hull_ratio"] and round(signals["hull_ratio"], 3),
                "ray_hits": signals["ray_hits"],
                "cyl_fill": signals["cyl_fill"] and round(signals["cyl_fill"], 3),
                "com_offset": signals["com_offset"] and round(signals["com_offset"], 3),
                "face_count": face_count,
                "sdf_mean": round(float(sdf_m), 5), "sdf_var": round(float(sdf_v), 6),
                "new_class": NEW_CLASSES[type_idx],
                "hole_votes": hv, "head_votes": headv, "notes": notes,
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
               n_ambiguous: int, elapsed: float) -> Path:
    esc = html_lib.escape
    total_bodies = sum(overall.values())

    summary_rows = "".join(
        f"<tr><td>{esc(c)}</td><td>{overall.get(c, 0)}</td>"
        f"<td>{100.0 * overall.get(c, 0) / max(1, total_bodies):.1f}%</td></tr>"
        for c in NEW_CLASSES)

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
                    '<tr><th>tag</th><th>class</th><th>s/m/l</th>'
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
<title>Component Type Audit — 8-Class Taxonomy</title>
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
<header><h1>Component Type Audit — 8-class taxonomy</h1>
<div class="sub">{len(records)} folders · {total_bodies} bodies · {n_ambiguous} ambiguous
(vote conflicts) · generated in {elapsed:.0f}s · thresholds in
back_end/dataset.py (_TYPE_THRESHOLDS)</div></header>
<main>
<div class="top">
<div class="panel"><h2 style="margin-top:0;">Class balance</h2>
<table><tr><th>Class</th><th>Bodies</th><th>%</th></tr>{summary_rows}</table></div>
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
    n_ambiguous = 0
    t0 = time.time()

    for i, (cat, folder, stepf) in enumerate(folders, 1):
        ft = time.time()
        bodies, status = parse_with_timeout(str(stepf))
        elapsed = time.time() - ft
        rec = {"category": cat, "folder": folder, "status": status,
               "bodies": [], "n_bodies": 0}
        if status == "ok" and bodies:
            # _audit_parse() already classified each body (new_class/hole_votes/
            # head_votes/notes set) via dataset.py's shared classifier.
            for b in bodies:
                overall[b["new_class"]] += 1
                per_cat[cat][b["new_class"]] += 1
                if b["notes"]:
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
        "folders":      records,
    }, indent=1))
    out_html = write_html(records, overall, per_cat, n_ambiguous, total_elapsed)

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
