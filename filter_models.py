#!/usr/bin/env python3
"""
filter_models.py — Move assembly model folders that exceed node/edge thresholds.

Thresholds (matching dataset.py):
  max_nodes = 20   (solid bodies / volumes)
  max_edges = 60   (directed contact edges = unique body-pairs × 2)

Strategy:
  - Fusion360 Gallery models (have assembly.json):
      nodes  = len(bodies dict in assembly.json)
      edges  = unique body-pairs from contacts × 2
  - Hand-crafted models (no assembly.json, only STEP file):
      run gmsh with 120s timeout — same logic as dataset._parse_step_with_timeout

Destinations:
  nodes > 20  →  skipped_models/nodes_gt_20/<folder>
  edges > 60  →  skipped_models/edges_gt_60/<folder>  (only if nodes ≤ 20)
"""

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

SOURCE_DIR   = Path("/Users/mbp/Documents/MTECH/Sem4/Individual_project/AI_Assisted_3D_Assembly_Design/AI-Assisted-3D-Assembly-Design/Source_3d_models")
SKIP_BASE    = SOURCE_DIR / "skipped_models"
SKIP_NODES   = SKIP_BASE / "nodes_gt_20"
SKIP_EDGES   = SKIP_BASE / "edges_gt_60"
SKIP_TIMEOUT = SKIP_BASE / "timeout"
REPORT_PATH  = SOURCE_DIR / "skipped_models_report.json"
LOG_PATH     = SOURCE_DIR / "filter_models.log"

MAX_NODES    = 20
MAX_EDGES    = 60
TIMEOUT_SECS = 120

UUID_RE   = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)
STEP_EXTS = {".step", ".stp", ".STEP", ".STP"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    print(msg, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def move_folder(folder: Path, dest_dir: Path) -> Optional[str]:
    """Move folder into dest_dir. Return destination path or None if already there."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / folder.name
    if dest.exists():
        return str(dest)   # already moved (e.g. previous run)
    shutil.move(str(folder), str(dest))
    return str(dest)


# ── Fast JSON-based filtering (Fusion360 Gallery format) ─────────────────────

def classify_json_model(json_path: Path):
    """
    Returns ("nodes_gt_20", n_nodes) | ("edges_gt_60", n_edges) | ("ok", n_nodes) | ("error", msg)
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return "error", str(e)

    bodies   = data.get("bodies") or {}
    n_nodes  = len(bodies) or (data.get("properties") or {}).get("body_count") or 0

    if n_nodes > MAX_NODES:
        return "nodes_gt_20", n_nodes

    contacts = data.get("contacts") or []
    body_pairs: set = set()
    for c in contacts:
        b1 = (c.get("entity_one") or {}).get("body", "")
        b2 = (c.get("entity_two") or {}).get("body", "")
        if b1 and b2 and b1 != b2:
            body_pairs.add(tuple(sorted([b1, b2])))
    n_dir_edges = len(body_pairs) * 2

    if n_dir_edges > MAX_EDGES:
        return "edges_gt_60", n_dir_edges

    return "ok", n_nodes


# ── gmsh-based filtering (for STEP files without assembly.json) ───────────────

def _gmsh_worker(step_path: str, result_queue, max_nodes: int, max_edges: int):
    """Subprocess worker: parse STEP with gmsh and put result on queue."""
    import gmsh

    class _SkipNodes(Exception): pass
    class _SkipEdges(Exception): pass

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("assy")
    try:
        gmsh.merge(step_path)
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.occ.getEntities(3)
        if len(volumes) < 2:
            result_queue.put(("ok_skip", 0, 0))   # not a multi-body assembly
            return
        if len(volumes) > max_nodes:
            raise _SkipNodes(len(volumes))

        gmsh.model.occ.fragment(volumes, [])
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.occ.getEntities(3)

        body_surfs = []
        for dim, tag in volumes:
            bnd = gmsh.model.getBoundary([(dim, tag)], oriented=False, combined=True)
            body_surfs.append(frozenset(abs(s[1]) for s in bnd if s[0] == 2))

        n_dir = sum(
            2 for i in range(len(volumes))
            for j in range(i + 1, len(volumes))
            if body_surfs[i] & body_surfs[j]
        )
        if n_dir > max_edges:
            raise _SkipEdges(n_dir)

        result_queue.put(("ok", len(volumes), n_dir))

    except _SkipNodes as e:
        result_queue.put(("nodes_gt_20", int(str(e)), 0))
    except _SkipEdges as e:
        result_queue.put(("edges_gt_60", 0, int(str(e))))
    except Exception as exc:
        result_queue.put(("error", 0, str(exc)))
    finally:
        gmsh.finalize()


def classify_step_model(step_path: Path):
    """
    Returns ("nodes_gt_20", n) | ("edges_gt_60", n) | ("ok", n) | ("timeout", 0) | ("error", msg)
    """
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q   = ctx.Queue()
    p   = ctx.Process(
        target=_gmsh_worker,
        args=(str(step_path), q, MAX_NODES, MAX_EDGES),
        daemon=True,
    )
    p.start()
    p.join(timeout=TIMEOUT_SECS)

    if p.is_alive():
        p.terminate(); p.join(5)
        if p.is_alive(): p.kill(); p.join()
        return "timeout", 0

    if not q.empty():
        result = q.get_nowait()
        status = result[0]
        if status == "ok":
            return "ok", result[1]
        if status == "ok_skip":
            return "ok", 0
        if status == "nodes_gt_20":
            return "nodes_gt_20", result[1]
        if status == "edges_gt_60":
            return "edges_gt_60", result[2]
        if status == "error":
            return "error", result[2]

    return "error", "no result"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    SKIP_NODES.mkdir(parents=True, exist_ok=True)
    SKIP_EDGES.mkdir(parents=True, exist_ok=True)
    SKIP_TIMEOUT.mkdir(parents=True, exist_ok=True)

    log(f"\n{'='*60}")
    log(f"filter_models.py — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Source: {SOURCE_DIR}")
    log(f"Thresholds: nodes > {MAX_NODES}  edges > {MAX_EDGES}  timeout {TIMEOUT_SECS}s")
    log(f"{'='*60}\n")

    # ── 1. Collect all assembly.json paths (Fusion360 Gallery models) ─────────
    json_paths = [
        p for p in SOURCE_DIR.rglob("assembly.json")
        if SKIP_BASE not in p.parents
    ]
    log(f"Found {len(json_paths)} assembly.json model(s).")

    # ── 2. Collect non-assembly STEP files (hand-crafted models) ─────────────
    step_paths = [
        p for p in SOURCE_DIR.rglob("*")
        if p.suffix in STEP_EXTS
        and not UUID_RE.match(p.stem)
        and SKIP_BASE not in p.parents
        and p.name not in ("assembly.step", "Assembly.step")
        and not (p.parent / "assembly.json").exists()
    ]
    log(f"Found {len(step_paths)} hand-crafted STEP model(s) (no assembly.json).\n")

    skipped_entries = []
    counts = {"nodes_gt_20": 0, "edges_gt_60": 0, "timeout": 0, "ok": 0, "error": 0}

    # ── Phase 1: Fast JSON-based filtering ────────────────────────────────────
    log(f"--- Phase 1: JSON-based filtering ({len(json_paths)} models) ---")
    for i, jpath in enumerate(sorted(json_paths), 1):
        folder = jpath.parent
        status, value = classify_json_model(jpath)

        if status == "nodes_gt_20":
            dest = move_folder(folder, SKIP_NODES)
            log(f"  [{i}/{len(json_paths)}] NODES>{MAX_NODES} ({value} bodies): {folder.name} → nodes_gt_20/")
            skipped_entries.append({
                "file": str(jpath), "folder": str(folder),
                "reason": f"nodes > {MAX_NODES}", "nodes": value, "moved_to": dest,
            })
            counts["nodes_gt_20"] += 1

        elif status == "edges_gt_60":
            dest = move_folder(folder, SKIP_EDGES)
            log(f"  [{i}/{len(json_paths)}] EDGES>{MAX_EDGES} ({value} dir-edges): {folder.name} → edges_gt_60/")
            skipped_entries.append({
                "file": str(jpath), "folder": str(folder),
                "reason": f"edges > {MAX_EDGES}", "edges": value, "moved_to": dest,
            })
            counts["edges_gt_60"] += 1

        elif status == "error":
            log(f"  [{i}/{len(json_paths)}] ERROR: {folder.name}: {value}")
            counts["error"] += 1

        else:
            counts["ok"] += 1

        if i % 500 == 0:
            log(f"  Progress: {i}/{len(json_paths)} — nodes_mv: {counts['nodes_gt_20']}  edges_mv: {counts['edges_gt_60']}")

    log(f"\nPhase 1 done: {counts['nodes_gt_20']} nodes_gt_20  {counts['edges_gt_60']} edges_gt_60  {counts['ok']} ok  {counts['error']} errors\n")

    # ── Phase 2: gmsh-based filtering for hand-crafted STEP files ─────────────
    log(f"--- Phase 2: gmsh-based filtering ({len(step_paths)} STEP models) ---")
    for i, spath in enumerate(sorted(step_paths), 1):
        folder = spath.parent
        t0 = time.time()
        status, value = classify_step_model(spath)
        elapsed = round(time.time() - t0, 1)

        if status == "nodes_gt_20":
            # Only move if folder is NOT a top-level source dir child containing multiple models
            if folder == SOURCE_DIR:
                log(f"  [{i}/{len(step_paths)}] NODES>{MAX_NODES} — skipping move (file is in source root)")
            else:
                dest = move_folder(folder, SKIP_NODES)
                log(f"  [{i}/{len(step_paths)}] NODES>{MAX_NODES} ({value}): {spath.name} [{elapsed}s] → nodes_gt_20/")
                skipped_entries.append({
                    "file": str(spath), "folder": str(folder),
                    "reason": f"nodes > {MAX_NODES}", "nodes": value, "elapsed": elapsed, "moved_to": dest,
                })
                counts["nodes_gt_20"] += 1

        elif status == "edges_gt_60":
            if folder == SOURCE_DIR:
                log(f"  [{i}/{len(step_paths)}] EDGES>{MAX_EDGES} — skipping move (file is in source root)")
            else:
                dest = move_folder(folder, SKIP_EDGES)
                log(f"  [{i}/{len(step_paths)}] EDGES>{MAX_EDGES} ({value}): {spath.name} [{elapsed}s] → edges_gt_60/")
                skipped_entries.append({
                    "file": str(spath), "folder": str(folder),
                    "reason": f"edges > {MAX_EDGES}", "edges": value, "elapsed": elapsed, "moved_to": dest,
                })
                counts["edges_gt_60"] += 1

        elif status == "timeout":
            if folder != SOURCE_DIR:
                dest = move_folder(folder, SKIP_TIMEOUT)
                log(f"  [{i}/{len(step_paths)}] TIMEOUT: {spath.name} [{elapsed}s] → timeout/")
                skipped_entries.append({
                    "file": str(spath), "folder": str(folder),
                    "reason": f"timeout > {TIMEOUT_SECS}s", "elapsed": elapsed, "moved_to": dest,
                })
                counts["timeout"] += 1
            else:
                log(f"  [{i}/{len(step_paths)}] TIMEOUT — {spath.name} (in source root, not moved)")

        elif status == "error":
            log(f"  [{i}/{len(step_paths)}] ERROR: {spath.name}: {value}")
            counts["error"] += 1

        else:
            counts["ok"] += 1
            log(f"  [{i}/{len(step_paths)}] OK  {spath.name} [{elapsed}s]")

    log(f"\nPhase 2 done: {counts['nodes_gt_20']} total-nodes_mv  {counts['edges_gt_60']} total-edges_mv  {counts['timeout']} timeout")

    # ── Write report ──────────────────────────────────────────────────────────
    report = {
        "source_dir": str(SOURCE_DIR),
        "thresholds": {"max_nodes": MAX_NODES, "max_edges": MAX_EDGES, "timeout_secs": TIMEOUT_SECS},
        "total_found": len(json_paths) + len(step_paths),
        "total_skipped": len(skipped_entries),
        "skipped_breakdown": {
            f"nodes_gt_{MAX_NODES}": counts["nodes_gt_20"],
            f"edges_gt_{MAX_EDGES}": counts["edges_gt_60"],
            "timeout": counts["timeout"],
        },
        "skipped_folders": {
            f"nodes_gt_{MAX_NODES}": str(SKIP_NODES),
            f"edges_gt_{MAX_EDGES}": str(SKIP_EDGES),
            "timeout": str(SKIP_TIMEOUT),
        },
        "entries": skipped_entries,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    log(f"\nReport written → {REPORT_PATH}")

    log(f"\n{'='*60}")
    log(f"DONE — {len(skipped_entries)} folders moved total")
    log(f"  nodes_gt_20 : {counts['nodes_gt_20']}")
    log(f"  edges_gt_60 : {counts['edges_gt_60']}")
    log(f"  timeout     : {counts['timeout']}")
    log(f"  ok (kept)   : {counts['ok']}")
    log(f"  errors      : {counts['error']}")
    log(f"{'='*60}\n")


if __name__ == "__main__":
    main()
