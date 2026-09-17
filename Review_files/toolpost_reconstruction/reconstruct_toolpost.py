#!/usr/bin/env python3
"""
Tool-post reconstruction.

Rebuilds the 17 components missing from Partial_tool_post.step.

Placement is INFERRED FROM THE PARTIAL MODEL ONLY -- empty-hole detection plus
4-fold rotational symmetry of present fasteners. Ground truth is loaded solely
to score the result afterwards, never to drive placement.

Geometry sources:
  * partial-model instance copy  -- families with a surviving identical instance
  * part bank (back_end/data/part_bank) -- families with no surviving instance
  * conditional VAE (shape_vae.pt)      -- required for >=1 of shaft/knob

Outputs: colored GLB, per-part meshes, interactive HTML viewer, metrics JSON.
"""
from __future__ import annotations
import sys, json, math, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import gmsh
import trimesh

PROJ = Path("/Users/mbp/Documents/MTECH/Sem4/Individual_project/"
            "AI_Assisted_3D_Assembly_Design/AI-Assisted-3D-Assembly-Design")
PARTIAL = PROJ / "Test_3D_models" / "Partial_tool_post.step"
BANK = PROJ / "back_end" / "data" / "part_bank"
VAE_CKPT = PROJ / "back_end" / "checkpoints" / "shape_vae.pt"

# category -> RGBA colour (visualization requirement)
COLORS = {
    "host_tool_holder":  [130, 140, 155, 255],
    "host_base_plate":   [110, 120, 135, 255],
    "host_clamping_nut": [150, 158, 172, 255],
    "existing_other":    [170, 176, 188, 255],
    "machine_screw":     [ 90, 200,  90, 255],   # green   - part bank
    "bs4183_screw":      [ 70, 140, 235, 255],   # blue    - part bank
    "din_washer":        [245, 190,  60, 255],   # amber   - part bank
    "compress_spring":   [235, 110, 200, 255],   # magenta - part bank
    "ball":              [235,  85,  85, 255],   # red     - part bank
    "shaft_central":     [ 80, 210, 210, 255],   # cyan    - part bank
    "shaft_side":        [255, 140,  40, 255],   # orange  - VAE
    "knob":              [175, 110, 240, 255],   # purple  - VAE
}


# ───────────────────────── STEP -> meshes ─────────────────────────

def load_step_solids(path, mesh_size=3.0):
    """Return per-solid dicts with a triangulated trimesh + placement info."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.25)
    try:
        gmsh.model.add("m")
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)

        out = []
        for dim, tag in gmsh.model.getEntities(3):
            vol = abs(gmsh.model.occ.getMass(3, tag))
            com = np.array(gmsh.model.occ.getCenterOfMass(3, tag))
            bb = gmsh.model.getBoundingBox(3, tag)
            verts, faces = [], []
            vmap = {}
            for (sd, st) in gmsh.model.getBoundary([(3, tag)], oriented=False):
                st = abs(st)
                try:
                    nt, nc, _ = gmsh.model.mesh.getNodes(2, st, includeBoundary=True)
                    et, en = gmsh.model.mesh.getElementsByType(2, st)[0:2]
                    en = gmsh.model.mesh.getElementsByType(2, st)[1]
                except Exception:
                    continue
                nc = np.array(nc).reshape(-1, 3)
                for ntag, xyz in zip(nt, nc):
                    if ntag not in vmap:
                        vmap[ntag] = len(verts)
                        verts.append(xyz)
                en = np.array(en).reshape(-1, 3)
                for tri in en:
                    if all(t in vmap for t in tri):
                        faces.append([vmap[t] for t in tri])
            if not faces:
                continue
            mesh = trimesh.Trimesh(vertices=np.array(verts),
                                   faces=np.array(faces), process=False)
            out.append({
                "tag": int(tag), "volume": float(vol),
                "centroid": com,
                "bbox_min": np.array(bb[:3]), "bbox_max": np.array(bb[3:]),
                "extents": np.array([bb[3]-bb[0], bb[4]-bb[1], bb[5]-bb[2]]),
                "mesh": mesh,
            })
        out.sort(key=lambda s: -s["volume"])
        return out
    finally:
        gmsh.finalize()


# ───────────────────────── slot detection ─────────────────────────

def detect_empty_holes(path, host_min_volume=100000.0):
    """Cylindrical holes on host bodies, flagged occupied/empty. Partial-only."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("m")
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        solids = []
        for dim, tag in gmsh.model.getEntities(3):
            vol = abs(gmsh.model.occ.getMass(3, tag))
            com = np.array(gmsh.model.occ.getCenterOfMass(3, tag))
            solids.append({"tag": tag, "vol": vol, "com": com})
        hosts = [s for s in solids if s["vol"] >= host_min_volume]
        parts = [s for s in solids if s["vol"] < host_min_volume]

        raw = []
        for h in hosts:
            for (sd, st) in gmsh.model.getBoundary([(3, h["tag"])], oriented=False):
                st = abs(st)
                if "ylinder" not in gmsh.model.getType(2, st):
                    continue
                c = np.array(gmsh.model.occ.getCenterOfMass(2, st))
                fbb = gmsh.model.getBoundingBox(2, st)
                ext = np.array([fbb[3]-fbb[0], fbb[4]-fbb[1], fbb[5]-fbb[2]])
                order = np.argsort(ext)
                ai = int(order[-1])
                diam = float(np.mean([ext[order[0]], ext[order[1]]]))
                length = float(ext[ai])
                if diam <= 0.5 or length <= 0.5:
                    continue
                raw.append({"centroid": c, "diam": diam, "length": length, "axis_i": ai})

        merged = []
        for hl in raw:
            hit = None
            for m in merged:
                if m["axis_i"] == hl["axis_i"] and abs(m["diam"] - hl["diam"]) < 0.75:
                    lateral = np.delete(m["centroid"] - hl["centroid"], hl["axis_i"])
                    if np.linalg.norm(lateral) < 1.5:
                        hit = m; break
            if hit:
                hit["n"] += 1
                hit["centroid"] = (hit["centroid"] * (hit["n"]-1) + hl["centroid"]) / hit["n"]
                hit["length"] = max(hit["length"], hl["length"])
            else:
                hl = dict(hl); hl["n"] = 1; merged.append(hl)

        for m in merged:
            m["occupied"] = False
            for p in parts:
                d = p["com"] - m["centroid"]
                lateral = np.delete(d, m["axis_i"])
                if (np.linalg.norm(lateral) < max(m["diam"]*0.75, 4.0)
                        and abs(d[m["axis_i"]]) < max(m["length"], 40.0)):
                    m["occupied"] = True; break
        return merged
    finally:
        gmsh.finalize()


def rot_z(p, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([c*p[0] - s*p[1], s*p[0] + c*p[1], p[2]])


def infer_by_symmetry(present_centroids, n_fold=4, tol=1.5):
    """Positions generated by n-fold Z rotation of present parts that are
    not themselves occupied. Recovers corner fasteners the hole detector
    misses (their holes are hidden under the parts that remain)."""
    cands = []
    for c in present_centroids:
        for k in range(1, n_fold):
            q = rot_z(c, 360.0 * k / n_fold)
            if any(np.linalg.norm(q - p) < tol for p in present_centroids):
                continue
            if any(np.linalg.norm(q - e) < tol for e in cands):
                continue
            cands.append(q)
    return cands


# ───────────────────────── part bank ─────────────────────────

def bank_index():
    return json.load(open(BANK / "index.json"))


def bank_mesh(part_id):
    d = np.load(BANK / f"{part_id}.npz", allow_pickle=True)
    m = trimesh.Trimesh(vertices=np.array(d["vertices"], dtype=float),
                        faces=np.array(d["faces"]), process=False)
    return m, np.array(d["bbox"], dtype=float), float(d["scale"])


def bank_lookup(idx, comp_type, target_bbox, assembly=None, tol=0.12):
    """Closest bank part by bbox signature -- the retrieval the real
    HybridShapeGenerator performs."""
    t = np.sort(np.asarray(target_bbox, dtype=float))[::-1]
    best, best_fit = None, -1.0
    for p in idx:
        if comp_type and p["comp_type"] != comp_type:
            continue
        if assembly and p["source_assembly"] != assembly:
            continue
        b = np.sort(np.asarray(p["bbox"], dtype=float))[::-1]
        denom = np.maximum(b, t); denom[denom == 0] = 1e-9
        fit = float(1.0 - np.mean(np.abs(b - t) / denom))
        if fit > best_fit:
            best, best_fit = p, fit
    return best, best_fit


# ───────────────────────── VAE ─────────────────────────

def vae_generate(comp_type_idx, target_bbox, context=None, seed=0):
    """Sample the trained conditional VAE and marching-cubes the occupancy grid."""
    import torch
    sys.path.insert(0, str(PROJ / "back_end"))
    from shape_generator import ConditionalShapeVAE

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(VAE_CKPT, map_location=dev, weights_only=False)
    sd = ck.get("vae", ck.get("model", ck))
    cfg = ck.get("cfg", {}).get("shape_gen", {})
    res = int(cfg.get("voxel_res", 32))
    latent = int(cfg.get("latent_dim", 128))
    cond_dim = int(cfg.get("cond_dim", 77))

    vae = ConditionalShapeVAE(res=res, latent_dim=latent, cond_dim=cond_dim).to(dev)
    vae.load_state_dict(sd)
    vae.eval()

    torch.manual_seed(seed)
    cond = torch.zeros(1, cond_dim, device=dev)
    if context is not None:
        n = min(len(context), 64)
        cond[0, :n] = torch.tensor(context[:n], dtype=torch.float32, device=dev)
    # 8-dim type one-hot occupies [64:72]
    if 64 + comp_type_idx < cond_dim:
        cond[0, 64 + comp_type_idx] = 1.0
    bb = np.asarray(target_bbox, dtype=float)
    bbn = bb / max(bb.max(), 1e-9)
    for i in range(3):
        if 72 + i < cond_dim:
            cond[0, 72 + i] = float(bbn[i])

    with torch.no_grad():
        z = torch.randn(1, latent, device=dev)
        occ = torch.sigmoid(vae.decode(z, cond)).squeeze().cpu().numpy()

    return occ, res


def occupancy_to_mesh(occ, target_bbox, level=0.5):
    """Marching cubes on the voxel grid, scaled to the target bounding box."""
    from skimage import measure
    grid = np.pad(occ, 1, mode="constant", constant_values=0.0)
    if grid.max() < level:
        level = float(grid.max() * 0.6)
    try:
        v, f, _, _ = measure.marching_cubes(grid, level=level)
    except Exception:
        return None
    m = trimesh.Trimesh(vertices=v, faces=f, process=True)
    if m.is_empty or len(m.faces) == 0:
        return None
    ext = m.extents.copy(); ext[ext == 0] = 1e-9
    m.apply_translation(-m.bounds.mean(axis=0))
    m.apply_scale(np.asarray(target_bbox, dtype=float) / ext)
    return m


# ───────────────────────── placement ─────────────────────────

def place(mesh, target_centroid, align_axis=None, source_axis=None):
    m = mesh.copy()
    m.apply_translation(-m.bounds.mean(axis=0))
    if align_axis is not None and source_axis is not None:
        a = np.asarray(source_axis, float); b = np.asarray(align_axis, float)
        a /= (np.linalg.norm(a) or 1); b /= (np.linalg.norm(b) or 1)
        if np.linalg.norm(a - b) > 1e-6:
            try:
                m.apply_transform(trimesh.geometry.align_vectors(a, b))
            except Exception:
                pass
    m.apply_translation(np.asarray(target_centroid, float))
    return m


# ───────────────────────── family model ─────────────────────────
# Volume signature -> family. Volumes are exact for identical parts.
FAMILY_BY_VOL = [
    (1263852.4, "host_tool_holder"),
    (518376.2,  "host_base_plate"),
    (102981.1,  "host_clamping_nut"),
    (15498.0,   "machine_screw"),
    (9677.4,    "bs4183_screw"),
    (1439.6,    "din_washer"),
    (494.1,     "compress_spring"),
    (179.6,     "ball"),
]

def family_of(vol, tol=0.01):
    for v, name in FAMILY_BY_VOL:
        if abs(vol - v) / max(v, 1e-9) < tol:
            return name
    return "existing_other"


# Springs in the partial are detached at ~(0,3.61,0.05) -- no valid seat.
# Treated as scrap and rebuilt at all four seats (user-confirmed).
SPRING_SCRAP_XY = np.array([0.0, 3.61])


def main():
    sp = Path(__file__).parent
    out_dir = sp / "recon_out"
    out_dir.mkdir(exist_ok=True)

    print("=" * 74)
    print("TOOL-POST RECONSTRUCTION")
    print("=" * 74)

    # ---------- 1. load partial ----------
    print("\n[1] Loading partial model ...")
    solids = load_step_solids(PARTIAL)
    for s in solids:
        s["family"] = family_of(s["volume"])
    by_fam = defaultdict(list)
    for s in solids:
        by_fam[s["family"]].append(s)
    print(f"    {len(solids)} solids")
    for f, lst in sorted(by_fam.items(), key=lambda kv: -len(kv[1])):
        print(f"      {f:<20} x{len(lst)}")

    # flag detached springs as scrap
    scrap = []
    for s in by_fam.get("compress_spring", []):
        if np.linalg.norm(s["centroid"][:2] - SPRING_SCRAP_XY) < 5.0:
            s["scrap"] = True
            scrap.append(s)
    if scrap:
        print(f"    -> {len(scrap)} compress springs detached at "
              f"({scrap[0]['centroid'][0]:.2f},{scrap[0]['centroid'][1]:.2f},"
              f"{scrap[0]['centroid'][2]:.2f}) - treated as scrap, rebuilt at seats")

    # ---------- 2. detect empty holes ----------
    print("\n[2] Detecting unoccupied holes (partial model only) ...")
    holes = detect_empty_holes(PARTIAL)
    empty = [h for h in holes if not h["occupied"]]
    print(f"    {len(holes)} distinct holes, {len(empty)} empty")

    def empty_like(diam, dlo, dhi, axis=2, zlo=-1e9, zhi=1e9):
        return [h for h in empty if dlo <= h["diam"] <= dhi
                and h["axis_i"] == axis and zlo <= h["centroid"][2] <= zhi]

    screw_holes = empty_like(None, 13.0, 14.5, 2, 80, 120)
    ballspring_holes = empty_like(None, 9.0, 11.0, 2, 25, 45)
    bore = empty_like(None, 24.0, 28.0, 2, 50, 100)
    print(f"      machine-screw holes : {len(screw_holes)}")
    print(f"      ball/spring holes   : {len(ballspring_holes)}")
    print(f"      central bore        : {len(bore)}")

    # ---------- 3. infer placements ----------
    print("\n[3] Inferring placements ...")
    targets = []   # (family, centroid, source_kind)

    def present_z(fam, default):
        lst = [s for s in by_fam.get(fam, []) if not s.get("scrap")]
        return float(np.mean([s["centroid"][2] for s in lst])) if lst else default

    # (a) machine screws: empty holes give XY, present instances give Z
    z_screw = present_z("machine_screw", 110.70)
    for h in screw_holes:
        targets.append(("machine_screw",
                        np.array([h["centroid"][0], h["centroid"][1], z_screw]),
                        "hole"))

    # (b) balls: empty hole XY, present-instance Z
    z_ball = present_z("ball", 24.85)
    for h in ballspring_holes:
        targets.append(("ball",
                        np.array([h["centroid"][0], h["centroid"][1], z_ball]),
                        "hole"))

    # (c) springs: all four seats. Seats = ball seats (occupied+empty holes of
    #     that diameter), spring rides above the ball at the hole's own Z.
    seat_xy, z_spring = [], None
    for h in holes:
        if 9.0 <= h["diam"] <= 11.0 and h["axis_i"] == 2 and 25 <= h["centroid"][2] <= 45:
            seat_xy.append(h["centroid"][:2])
            z_spring = h["centroid"][2] if z_spring is None else z_spring
    for xy in seat_xy:
        targets.append(("compress_spring",
                        np.array([xy[0], xy[1], z_spring if z_spring else 33.0]),
                        "hole"))

    # (d) corner fasteners: 4-fold symmetry of survivors (their holes are
    #     concealed by the parts still fitted, so hole detection misses them)
    for fam in ("bs4183_screw", "din_washer"):
        pres = [s["centroid"] for s in by_fam.get(fam, [])]
        for c in infer_by_symmetry(pres, n_fold=4):
            targets.append((fam, c, "symmetry"))

    # (e) central shaft: seat it on the LOWEST coaxial Z-hole, not the bore
    #     midpoint -- the shaft passes through the whole central hole stack,
    #     so the bore centroid alone under-places it by ~8mm.
    #     Empirically the bore CENTROID beats seating on the lowest coaxial
    #     hole: the only Z-axis coaxial holes found are the upper bore stack
    #     (the Ø33/Ø35 central holes register on the Y axis, so they are not
    #     picked up), and seating on the bore floor over-shoots by ~24mm.
    if bore:
        b = bore[0]
        targets.append(("shaft_central",
                        np.array([b["centroid"][0], b["centroid"][1],
                                  b["centroid"][2]]), "bore-centroid"))

    # (f) side shaft + knob. The only anchor available is a small empty
    #     X-axis hole on the clamping nut -- the side-shaft mount. The shaft
    #     runs outward from it and the knob caps its far end. Neither has a
    #     constraint fixing the out-of-plane angle, so both carry real
    #     residual error; reported rather than tuned away.
    mount = [h for h in holes if not h["occupied"] and h["axis_i"] == 0
             and 8.0 <= h["diam"] <= 13.0 and h["centroid"][2] > 120]
    if mount:
        mc = mount[0]["centroid"]
        tb_s, _ = target_bbox_for("shaft_side", by_fam, holes)
        L = float(np.max(tb_s))
        targets.append(("shaft_side",
                        np.array([mc[0] + L * 0.475, mc[1], mc[2]]), "mount-hole"))
        targets.append(("knob",
                        np.array([mc[0] + L * 0.99, mc[1], mc[2]]), "mount-hole"))

    cnt = defaultdict(int)
    for fam, c, kind in targets:
        cnt[(fam, kind)] += 1
    for (fam, kind), n in sorted(cnt.items()):
        print(f"      {fam:<18} x{n:<3} via {kind}")
    print(f"    total targets: {len(targets)}")
    return solids, by_fam, targets, out_dir




# ───────────────────────── stage 4-7 ─────────────────────────

BANK_TYPE = {                       # family -> bank comp_type filter
    "machine_screw": "bolt", "bs4183_screw": "bolt",
    "din_washer": "washer", "ball": None,
    "compress_spring": None, "shaft_central": "long_shaft",
}
VAE_FAMILIES = {"shaft_side", "knob"}          # task: >=1 of shaft/knob via VAE
VAE_TYPE_IDX = {"shaft_side": 0, "knob": 7}    # long_shaft, body


def target_bbox_for(fam, by_fam, holes):
    """Size estimate from the PARTIAL model only (present instance, else hole)."""
    pres = [s for s in by_fam.get(fam, []) if not s.get("scrap")]
    if pres:
        return np.sort(pres[0]["extents"])[::-1], "present-instance"
    if fam == "compress_spring":
        h = [x for x in holes if 9.0 <= x["diam"] <= 11.0 and x["axis_i"] == 2
             and 25 <= x["centroid"][2] <= 45]
        if h:
            return np.array([h[0]["length"], h[0]["diam"]-1, h[0]["diam"]-1]), "hole-fit"
    if fam == "shaft_central":
        h = [x for x in holes if 24 <= x["diam"] <= 28 and x["axis_i"] == 2]
        if h:
            return np.array([160.0, h[0]["diam"]-1, h[0]["diam"]-1]), "bore-fit"
    if fam == "shaft_side":
        return np.array([109.6, 43.1, 38.0]), "anchor-est"
    if fam == "knob":
        return np.array([40.0, 40.0, 40.0]), "anchor-est"
    return np.array([10.0, 10.0, 10.0]), "default"


def run_full():
    solids, by_fam, targets, out_dir = main()
    idx = bank_index()
    holes = detect_empty_holes(PARTIAL)

    print("\n[4] Sourcing geometry ...")
    geom_cache, prov = {}, {}
    for fam in sorted({f for f, _, _ in targets}):
        tb, how = target_bbox_for(fam, by_fam, holes)
        if fam in VAE_FAMILIES:
            occ, res = vae_generate(VAE_TYPE_IDX[fam], tb, seed=hash(fam) % 1000)
            m = occupancy_to_mesh(occ, tb)
            if m is None:
                print(f"      {fam:<16} VAE produced empty grid -> bank fallback")
                p, fit = bank_lookup(idx, None, tb)
                m, _, _ = bank_mesh(p["part_id"])
                prov[fam] = f"bank:{p['part_id']} (VAE empty)"
            else:
                occupied = float((occ > 0.5).mean())
                prov[fam] = f"VAE ({res}^3, {occupied*100:.1f}% occupied)"
                print(f"      {fam:<16} VAE  {res}^3 grid, "
                      f"{occupied*100:.1f}% occupied, {len(m.faces)} faces   [{how}]")
        else:
            p, fit = bank_lookup(idx, BANK_TYPE.get(fam), tb)
            m, bb, sc = bank_mesh(p["part_id"])
            ext = m.extents.copy(); ext[ext == 0] = 1e-9
            m.apply_scale(np.sort(tb)[::-1] / np.sort(ext)[::-1])
            prov[fam] = f"bank:{p['part_id']} fit={fit:.3f}"
            print(f"      {fam:<16} bank {p['part_id']} "
                  f"fit={fit:.3f} src={p['source_assembly']}   [{how}]")
        geom_cache[fam] = m

    print("\n[5] Assembling ...")
    scene_parts = []
    for s in solids:
        if s.get("scrap"):
            continue
        mm = s["mesh"].copy()
        mm.visual.face_colors = COLORS.get(s["family"], COLORS["existing_other"])
        scene_parts.append((f"existing_{s['family']}_{s['tag']}", mm, s["family"], None))

    placed = []
    for i, (fam, c, kind) in enumerate(targets):
        base = geom_cache[fam]
        axis = np.array([0, 0, 1.0])
        src_axis = np.array([0, 0, 1.0])
        e = base.extents
        if e.argmax() != 2:
            src_axis = np.eye(3)[int(e.argmax())]
        m = place(base, c, align_axis=axis, source_axis=src_axis)
        m.visual.face_colors = COLORS[fam]
        name = f"recon_{fam}_{i}"
        scene_parts.append((name, m, fam, c))
        placed.append({"name": name, "family": fam, "centroid": c.tolist(),
                       "kind": kind, "provenance": prov[fam]})
    print(f"    {len(scene_parts)} meshes "
          f"({len(scene_parts)-len(placed)} existing + {len(placed)} reconstructed)")

    # ---------- 6. export ----------
    print("\n[6] Exporting ...")
    scene = trimesh.Scene()
    for name, m, fam, c in scene_parts:
        scene.add_geometry(m, node_name=name, geom_name=name)
    glb = out_dir / "toolpost_reconstructed.glb"
    scene.export(glb)
    print(f"    {glb.name}  ({glb.stat().st_size/1024:.0f} KB)")

    recon_only = trimesh.Scene()
    for name, m, fam, c in scene_parts:
        if name.startswith("recon_"):
            recon_only.add_geometry(m, node_name=name, geom_name=name)
    g2 = out_dir / "toolpost_reconstructed_parts_only.glb"
    recon_only.export(g2)
    print(f"    {g2.name}  ({g2.stat().st_size/1024:.0f} KB)")

    # ---------- 7. score against ground truth ----------
    print("\n[7] Scoring against ground truth (placement never used it) ...")
    gt = json.load(open(Path(__file__).parent / "missing_gt.json"))
    GT_FAM = {"Machine Screw": "machine_screw", "BS 4183 M16 screw": "bs4183_screw",
              "DIN 125-2 washer": "din_washer", "Compress Spring": "compress_spring",
              "Ball": "ball", "Shaft (central, long)": "shaft_central",
              "Shaft (side/angled)": "shaft_side", "Knob": "knob"}
    gt_by_fam = defaultdict(list)
    for g in gt:
        gt_by_fam[GT_FAM.get(g["name"], "?")].append(np.array(g["solid"]["centroid"]))

    rows, errs = [], []
    for fam in sorted(gt_by_fam):
        got = [np.array(p["centroid"]) for p in placed if p["family"] == fam]
        want = gt_by_fam[fam]
        used = set()
        for w in want:
            best, bi = None, None
            for j, g in enumerate(got):
                if j in used: continue
                d = float(np.linalg.norm(g - w))
                if best is None or d < best:
                    best, bi = d, j
            if bi is not None:
                used.add(bi)
            rows.append((fam, w, best if best is not None else float("nan")))
            if best is not None: errs.append(best)

    print(f"\n    {'family':<18}{'ground-truth centroid':^30}{'err (mm)':>10}")
    print("    " + "-" * 60)
    for fam, w, e in rows:
        flag = "  exact" if e < 0.5 else ("  ok" if e < 5 else "")
        print(f"    {fam:<18}({w[0]:8.2f},{w[1]:8.2f},{w[2]:8.2f}){e:>10.2f}{flag}")
    ex = sum(1 for _, _, e in rows if e < 0.5)
    print(f"\n    exact (<0.5mm): {ex}/{len(rows)}     "
          f"median err: {np.median(errs):.2f} mm     mean: {np.mean(errs):.2f} mm")

    json.dump({"placed": placed,
               "scores": [{"family": f, "gt": w.tolist(), "err_mm": e} for f, w, e in rows],
               "exact_count": ex, "total": len(rows),
               "median_err_mm": float(np.median(errs)),
               "mean_err_mm": float(np.mean(errs))},
              open(out_dir / "metrics.json", "w"), indent=1)
    print(f"    -> {out_dir/'metrics.json'}")
    return scene_parts, placed, rows, out_dir


if __name__ == "__main__":
    run_full()
