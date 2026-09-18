#!/usr/bin/env python3
"""Score the rebuild against the original Tool_Post_10 model.

Per placed part:
  pos_err   : centroid distance to the matching original solid (mm)
  shape_cd  : chamfer distance after CENTRING both meshes -> pure shape/orientation
  pose_cd   : chamfer distance as placed -> shape + orientation + position
The noise floor of two independent triangulations of the SAME solid is ~0.2-0.6mm.
"""
import sys, json, time
import numpy as np, trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, "/Users/mbp/Documents/MTECH/Sem4/Individual_project/AI_Assisted_3D_Assembly_Design/AI-Assisted-3D-Assembly-Design/back_end")
import rebuild as RB
import slot_detector as sd

PROJ = "/Users/mbp/Documents/MTECH/Sem4/Individual_project/AI_Assisted_3D_Assembly_Design/AI-Assisted-3D-Assembly-Design"
PARTIAL = f"{PROJ}/Test_3D_models/Partial_tool_post.step"
SRC = f"{PROJ}/Source_3d_models/Best_models_for_training/Tool_Post/Tool_Post_10/Tool_Post_10.stp"

GT_EXTRA = {74985.6: "shaft_central", 9024.5: "shaft_side", 32670.1: "knob"}
def gt_family(vol):
    f = sd.family_of_volume(vol)
    if f != "unknown": return f
    for v, n in GT_EXTRA.items():
        if abs(vol - v) / v < 0.01: return n
    return "unknown"

def pts(m, n=5000): return trimesh.sample.sample_surface(m, n, seed=0)[0]
def cd(a, b): return (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()) / 2

t0 = time.time()
gt = RB.load_part_meshes(SRC, mesh_size=1.5)
for g in gt: g["gfam"] = gt_family(g["volume"])
print(f"loaded {len(gt)} original parts in {time.time()-t0:.0f}s")

def score(placed, label):
    print(f"\n=== {label} ===")
    print(f"  {'family':<16}{'source':<10}{'part':<16}{'pos_err':>8}{'shape_cd':>10}{'pose_cd':>9}")
    pool = {}
    for g in gt: pool.setdefault(g["gfam"], []).append(g)
    rows = []
    for p in placed:
        cands = pool.get(p["family"], [])
        if not cands: continue
        pc = np.array(p["centroid"])
        g = min(cands, key=lambda g: np.linalg.norm(g["centroid"] - pc))
        cands.remove(g)
        gp = pts(g["mesh"]); mp = pts(p["mesh"])
        posr = float(np.linalg.norm(g["centroid"] - pc))
        shape = cd(mp - mp.mean(0), gp - gp.mean(0))
        pose = cd(mp, gp)
        rows.append((p["family"], posr, shape, pose))
        print(f"  {p['family']:<16}{p['source']:<10}{str(p['part_id']):<16}{posr:>8.2f}{shape:>10.2f}{pose:>9.2f}")
    if rows:
        print(f"  {'MEAN':<42}{np.mean([r[1] for r in rows]):>8.2f}"
              f"{np.mean([r[2] for r in rows]):>10.2f}{np.mean([r[3] for r in rows]):>9.2f}")
    return rows

for label, ex in [("as the app runs (bank contains Tool_Post_10 itself)", None),
                  ("LEAVE-ONE-OUT (Tool_Post_10's own parts excluded from retrieval)", "Tool_Post_10")]:
    t0 = time.time()
    res = RB.rebuild_missing(PARTIAL, exclude_assembly=ex)
    print(f"\nrebuilt {len(res['placed'])} parts in {time.time()-t0:.0f}s   recognised={res['recognised']}")
    score(res["placed"], label)
    if ex is None:
        RB.export_glb(res["placed"], "/tmp/rebuilt_test.glb")
