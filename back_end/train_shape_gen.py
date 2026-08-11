"""
train_shape_gen.py — Phase 3 ConditionalShapeVAE training.
Leave-one-part-out training loop, mirrors train_ranker.py's structure:
reuses the frozen Phase-1 encoder from a serving checkpoint (never touched),
trains only shape_generator.ConditionalShapeVAE, saves checkpoints/shape_vae.pt
with the same encoder-staleness guard as node_ranker.pt.

Run:
    cd back_end && python train_shape_gen.py
    python train_shape_gen.py --config config.yaml --serving-ckpt checkpoints/best_serving.pt \
        --fold-idx 0 --epochs 60 --n-per-graph 8

Requires a part bank (data/part_bank/index.json) — built automatically on
first run if missing (can take a while; reparses the corpus once).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.data import Data

from dataset import AssemblyDataset, graph_level_indices, COMP_TYPES
from model import build_model
from part_bank import PartBank, build_part_bank, VOXEL_RES
from shape_generator import (ConditionalShapeVAE, vae_loss, build_conditioning_vector,
                              COND_DIM, FASTENER_TYPES)

# 18 model folders with zero bolt/nut/washer bodies (from audit_classification.json),
# concentrated in Press_Tool (8/21 folders) and C_Clamps (5/31) — excluded from
# shape-gen training samples only, since generation is now fastener-scoped. Phase 1/2
# training and config.yaml's categories/source_dir are untouched: they still need
# every folder across all 8 component types.
_FASTENER_TRAIN_EXCLUDE = {
    "Bench_Vice_46",
    "C_Clamps_08", "C_Clamps_15", "C_Clamps_28", "C_Clamps_37", "C_Clamps_40",
    "Pipe_Vice_21",
    "Press_Tool_06", "Press_Tool_13", "Press_Tool_14", "Press_Tool_15",
    "Press_Tool_16", "Press_Tool_17", "Press_Tool_23", "Press_Tool_24",
    "Crane_Hook_07", "Crane_Hook_13", "Crane_Hook_21",
}


# ── Leave-one-node-out sampling (mirrors train_ranker.py's remove_node) ─────────

def remove_node(data: Data, idx: int) -> Data:
    n = data.num_nodes
    keep = torch.tensor([i for i in range(n) if i != idx], dtype=torch.long)
    old_to_new = -torch.ones(n, dtype=torch.long)
    old_to_new[keep] = torch.arange(keep.size(0))

    src, dst = data.edge_index
    edge_mask = (src != idx) & (dst != idx)
    new_edge_index = torch.stack([
        old_to_new[src[edge_mask]],
        old_to_new[dst[edge_mask]],
    ])
    new_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else None

    return Data(x=data.x[keep], edge_index=new_edge_index, edge_attr=new_edge_attr)


def build_samples(graphs, sources, n_per_graph, part_bank: PartBank, rng: random.Random,
                   fastener_only: bool = True, exclude_folders=_FASTENER_TRAIN_EXCLUDE):
    """For each graph with >=2 nodes, sample up to n_per_graph nodes that have
    a matching part-bank entry (some bodies fail mesh extraction and have no
    entry — skipped). Returns [(partial_graph, target_vox, comp_type_idx,
    target_bbox_norm, source_assembly, category, target_bbox_raw), ...].
    category and target_bbox_raw are carried alongside the already-used
    fields specifically so a downstream retrieval-gated evaluation can
    re-query the part bank per sample without re-deriving body indices.

    fastener_only restricts leave-one-out targets to bolt/nut/washer (shape
    *generation* is now scoped to fasteners — detection of all 8 types is
    unaffected, this only narrows what the VAE is trained to reconstruct) and
    skips assemblies with zero fastener bodies entirely, so every remaining
    graph can actually contribute up to n_per_graph fastener samples."""
    samples = []
    for g, src_path in zip(graphs, sources):
        n = g.num_nodes
        if n < 2:
            continue
        category = getattr(g, "category", "") or ""
        assembly = Path(src_path).parent.name if src_path else ""
        if fastener_only and assembly in exclude_folders:
            continue
        idxs = list(range(n))
        rng.shuffle(idxs)
        taken = 0
        for i in idxs:
            if taken >= n_per_graph:
                break
            comp_type_idx = int(g.x[i, :len(COMP_TYPES)].argmax().item())
            if fastener_only and COMP_TYPES[comp_type_idx] not in FASTENER_TYPES:
                continue
            entry = part_bank.find_by_source(category, assembly, i)
            if entry is None:
                continue
            vox = part_bank.load_voxels(entry["part_id"]).astype(np.float32)
            bbox = np.asarray(entry["bbox"], dtype=np.float32)
            bbox_norm = bbox / (bbox.max() + 1e-9)
            samples.append((remove_node(g, i), vox, comp_type_idx, bbox_norm,
                             assembly, category, bbox))
            taken += 1
    return samples


def retrieval_gated_samples(samples, part_bank: PartBank, retrieval_tau: float):
    """Filter `samples` (from build_samples) down to the subset where part-bank
    retrieval's own fit_score falls below retrieval_tau — the harder minority
    of cases that actually reach the VAE in production (HybridShapeGenerator
    only falls back to the VAE when retrieval doesn't find a good enough
    match). Without this, the VAE's reported IoU/chamfer are averaged over
    every held-out fastener regardless of whether retrieval would have
    handled it in practice, which doesn't answer "how good is the VAE at the
    job it's actually given." exclude_assemblies={assembly} so a sample
    can't trivially retrieve its own held-out part from the bank."""
    gated = []
    for sample in samples:
        partial, vox, comp_type_idx, bbox_norm, assembly, category, bbox_raw = sample
        hits = part_bank.query(COMP_TYPES[comp_type_idx], category, bbox_raw,
                                top_k=1, exclude_assemblies={assembly})
        fit_score = hits[0].fit_score if hits else 0.0
        if fit_score < retrieval_tau:
            gated.append(sample)
    return gated


def _center_crop_or_pad(vox: np.ndarray, target_res: int) -> np.ndarray:
    """Center-crop or zero-pad a cubic voxel grid back to target_res along
    every axis — needed after scale-jitter changes the array shape."""
    out = vox
    for axis in range(3):
        cur = out.shape[axis]
        if cur > target_res:
            start = (cur - target_res) // 2
            out = np.take(out, range(start, start + target_res), axis=axis)
        elif cur < target_res:
            pad_before = (target_res - cur) // 2
            pad_after = target_res - cur - pad_before
            pad_width = [(0, 0)] * 3
            pad_width[axis] = (pad_before, pad_after)
            out = np.pad(out, pad_width, mode="constant", constant_values=0.0)
    return out


def augment_voxel(vox: np.ndarray, rng: random.Random, max_angle: float = 20.0,
                   scale_range=(0.92, 1.08), jitter_prob: float = 0.02,
                   jitter_band: int = 2) -> np.ndarray:
    """Continuous-angle 3D rotation (composed over all three axis pairs, not
    just 90-degree steps) plus mild scale and occupancy jitter on a fastener's
    voxel grid. Widens the orientation/size variety the VAE trains on using
    only geometry already in the corpus — pairs with the normal_hint-based
    rotate_to_target_axis() fix at inference time, which now needs the model
    to have seen more than four fixed orientations per part.

    jitter_band restricts occupancy jitter to a band within jitter_band
    voxels of the (pre-jitter) occupied region, instead of the full 32^3
    grid. A canonicalized fastener occupies roughly 1-5% of the padded grid,
    so a uniform per-voxel flip probability lands ~98% of its flips on empty
    background — for a sparse target that's mostly false-positive noise
    injected into the reconstruction target the model can never learn to
    predict, capping achievable IoU/Dice independent of model quality.
    Restricting to a dilated near-surface band keeps jitter meaningful
    (perturbing the boundary, which is where real shape variation lives)
    without flooding empty space."""
    from scipy.ndimage import rotate as ndi_rotate, zoom as ndi_zoom, binary_dilation

    res = vox.shape[-1]
    out = vox.astype(np.float32)
    for axes in ((0, 1), (0, 2), (1, 2)):
        angle = rng.uniform(-max_angle, max_angle)
        out = ndi_rotate(out, angle, axes=axes, reshape=False, order=1,
                          mode="constant", cval=0.0)

    scale = rng.uniform(*scale_range)
    if abs(scale - 1.0) > 1e-3:
        out = _center_crop_or_pad(ndi_zoom(out, scale, order=1, mode="constant", cval=0.0), res)

    if jitter_prob > 0:
        occ_band = binary_dilation(out > 0.5, iterations=jitter_band)
        np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
        flip = (np_rng.random(out.shape) < jitter_prob) & occ_band
        out = np.where(flip, 1.0 - np.round(out), out)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# System swap-pressure check, same mitigation as train.py's swap_guard() --
# see that function's comment for the full rationale (R36/R37 crash pattern).
_SWAP_WARN_PCT = 90.0
_SWAP_THROTTLE_SECS = 3.0


def swap_guard(device, context: str = "") -> None:
    try:
        pct = psutil.swap_memory().percent
    except Exception:
        return
    if pct < _SWAP_WARN_PCT:
        return
    print(f"    [swap-guard] {pct:.1f}% swap used{' (' + context + ')' if context else ''} "
          f"-- pausing {_SWAP_THROTTLE_SECS:.0f}s for extra cleanup", flush=True)
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    time.sleep(_SWAP_THROTTLE_SECS)
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


# ── Conditioning + context ───────────────────────────────────────────────────────

@torch.no_grad()
def compute_ctx(gnn, partial: Data, device) -> torch.Tensor:
    if partial.num_nodes == 0:
        return torch.zeros(gnn.out_dim if hasattr(gnn, "out_dim") else 64, device=device)
    z = gnn(partial.x.to(device), partial.edge_index.to(device),
            partial.edge_attr.to(device) if partial.edge_attr is not None else None)
    return z.mean(0)


def neighbor_scale_from(partial: Data) -> np.ndarray:
    if partial.num_nodes == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    log_vol = partial.x[:, 8].mean().item()
    max_extent = partial.x[:, 10:13].max(dim=1).values.mean().item()
    return np.array([log_vol, max_extent], dtype=np.float32)


# ── Metrics ──────────────────────────────────────────────────────────────────────

def voxel_iou(pred_prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (pred_prob > threshold)
    tgt = (target > 0.5)
    inter = (pred & tgt).sum().item()
    union = (pred | tgt).sum().item()
    return inter / union if union > 0 else 0.0



# Fixed, module-level seed for chamfer's point subsampling — without this,
# torch.randperm() below draws from the global RNG state, so re-evaluating
# the SAME checkpoint on the SAME voxel grid gives a DIFFERENT chamfer every
# call once a shape exceeds n_points occupied voxels. That's pure
# measurement noise stacked on top of real model variance (observed: ~110%
# coefficient-of-variation across epochs vs. ~20% for the deterministic,
# full-grid IoU metric) — enough to make single-epoch chamfer comparisons
# (e.g. checkpoint selection) unreliable. A fixed generator makes repeated
# evaluations of one checkpoint reproducible while still genuinely
# subsampling (not cherry-picked) for shapes that need it.
_CHAMFER_GEN = torch.Generator().manual_seed(1234)


def chamfer_from_voxels(pred_prob: torch.Tensor, target: torch.Tensor,
                         threshold: float = 0.5, n_points: int = 300) -> float:
    """Lightweight unit-normalized Chamfer distance between occupied-voxel
    centroid clouds (cheap proxy — avoids a full marching-cubes + sampling
    pass during training)."""
    res = target.shape[-1]
    pred_pts = torch.nonzero(pred_prob > threshold, as_tuple=False).float()
    tgt_pts = torch.nonzero(target > 0.5, as_tuple=False).float()
    if pred_pts.numel() == 0 or tgt_pts.numel() == 0:
        return 1.0  # worst case, one side empty
    if pred_pts.size(0) > n_points:
        pred_pts = pred_pts[torch.randperm(pred_pts.size(0), generator=_CHAMFER_GEN)[:n_points]]
    if tgt_pts.size(0) > n_points:
        tgt_pts = tgt_pts[torch.randperm(tgt_pts.size(0), generator=_CHAMFER_GEN)[:n_points]]
    pred_pts = pred_pts / res
    tgt_pts = tgt_pts / res
    d = torch.cdist(pred_pts, tgt_pts)
    d1 = d.min(dim=1).values.mean().item()
    d2 = d.min(dim=0).values.mean().item()
    return 0.5 * (d1 + d2)


# ── Epoch pass ─────────────────────────────────────────────────────────────────

def epoch_pass(vae, gnn, samples, device, opt=None, augment=False, rng=None,
               beta_kl=0.05, lambda_dice=0.5):
    is_train = opt is not None
    vae.train(is_train)

    total_loss = 0.0
    loss_parts_sum = {"bce": 0.0, "dice": 0.0, "kl": 0.0}
    ious, chamfers = [], []
    n_seen = 0

    for _i, (partial, vox, comp_type_idx, bbox_norm, _assembly, _category, _bbox_raw) in enumerate(samples):
        if augment and rng is not None:
            vox = augment_voxel(vox, rng)

        ctx = compute_ctx(gnn, partial, device)
        nbr = neighbor_scale_from(partial)
        cond = build_conditioning_vector(ctx, comp_type_idx, bbox_norm, nbr).unsqueeze(0)
        vox_t = torch.from_numpy(vox).float().unsqueeze(0).to(device)  # (1, res, res, res)

        if is_train:
            opt.zero_grad()
            recon, mu, logvar = vae(vox_t.unsqueeze(1), cond)
            loss, parts = vae_loss(recon, vox_t, mu, logvar, beta_kl, lambda_dice)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            for k in loss_parts_sum:
                if k in parts:
                    loss_parts_sum[k] += float(parts[k])
        else:
            with torch.no_grad():
                recon, mu, logvar = vae(vox_t.unsqueeze(1), cond)
                loss, parts = vae_loss(recon, vox_t, mu, logvar, beta_kl, lambda_dice)
                total_loss += loss.item()
                for k in loss_parts_sum:
                    if k in parts:
                        loss_parts_sum[k] += float(parts[k])
                probs = torch.sigmoid(recon.squeeze(1).squeeze(0))
                ious.append(voxel_iou(probs, vox_t.squeeze(0)))
                chamfers.append(chamfer_from_voxels(probs, vox_t.squeeze(0)))
        n_seen += 1

        # Unbatched per-sample loop: `partial`'s node count varies sample to
        # sample, and MPS caches a compiled graph executable per distinct
        # tensor shape it sees with no automatic eviction (same root cause
        # diagnosed in train.py's train_epoch — see its comments). Left
        # unchecked this grows unbounded over hundreds of samples/epoch and
        # thrashes swap (observed: this exact leak, 19GB+ RSS, mid-run).
        # Clear periodically, same proven mitigation as train.py.
        if device.type == "mps" and (_i + 1) % 10 == 0:
            gc.collect()
            torch.mps.empty_cache()
            swap_guard(device, context=f"sample {_i+1}")

    n_seen = max(1, n_seen)
    if device.type == "mps":
        gc.collect()
        torch.mps.empty_cache()
        swap_guard(device, context="pass end")
    loss_parts_avg = {k: v / n_seen for k, v in loss_parts_sum.items()}
    if is_train:
        return total_loss / n_seen, loss_parts_avg
    return {
        "loss": total_loss / n_seen,
        "iou": float(np.mean(ious)) if ious else 0.0,
        "chamfer": float(np.mean(chamfers)) if chamfers else 1.0,
        **{f"loss_{k}": v for k, v in loss_parts_avg.items()},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--serving-ckpt", default="checkpoints/best_serving.pt")
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-per-graph", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sc = cfg.get("shape_gen", {})
    epochs = args.epochs or sc.get("epochs", 60)
    n_per_graph = args.n_per_graph or sc.get("n_per_graph", 8)
    lr = args.lr or sc.get("lr", 1.0e-3)
    beta_kl = sc.get("beta_kl", 0.05)
    lambda_dice = sc.get("lambda_dice", 0.5)
    part_bank_dir = sc.get("part_bank_dir", "data/part_bank")
    latent_dim = sc.get("latent_dim", 128)
    voxel_res = sc.get("voxel_res", VOXEL_RES)
    fold_idx = args.fold_idx

    print("\n[1/6] Loading dataset …")
    dataset = AssemblyDataset(
        source_dir=cfg["data"]["source_dir"],
        processed_dir=cfg["data"]["processed_dir"],
        categories=cfg["data"].get("categories") or None,
    )
    print(f"      {len(dataset)} graphs loaded.")

    train_idx, val_idx, test_idx = graph_level_indices(dataset, cfg, fold_idx=fold_idx)
    train_graphs = [dataset[i] for i in train_idx]
    val_graphs = [dataset[i] for i in val_idx]
    test_graphs = [dataset[i] for i in test_idx]
    train_src = [dataset.graph_sources[i] for i in train_idx]
    val_src = [dataset.graph_sources[i] for i in val_idx]
    test_src = [dataset.graph_sources[i] for i in test_idx]
    print(f"      train: {len(train_graphs)}  val: {len(val_graphs)}  test: {len(test_graphs)}")

    print("\n[2/6] Loading frozen Phase-1 encoder …")
    ckpt_path = Path(args.serving_ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = ckpt["cfg"]["model"]
    gnn, _lp, device = build_model(
        in_dim=mc["in_dim"], out_dim=mc["out_dim"], hidden=mc["hidden_dim"],
        heads=mc["heads"], dropout=mc["dropout"], edge_dim=mc["edge_dim"],
    )
    gnn.load_state_dict(ckpt["gnn"])
    gnn.eval()
    for p in gnn.parameters():
        p.requires_grad_(False)
    print(f"      Encoder frozen (val AUC={ckpt['auc']:.4f}, trained_at={ckpt.get('trained_at')})")

    print("\n[3/6] Loading / building part bank …")
    bank = PartBank(part_bank_dir)
    if not (Path(part_bank_dir) / "index.json").exists():
        print(f"      No part bank found at {part_bank_dir} — building it now "
              f"(this reparses the corpus, can take a while) …")
        build_part_bank(cfg["data"]["source_dir"], part_bank_dir,
                         categories=cfg["data"].get("categories") or None)
        bank = PartBank(part_bank_dir)
    print(f"      Part bank: {len(bank)} parts")

    print("\n[4/6] Building leave-one-out samples …")
    rng = random.Random(42)
    train_samples = build_samples(train_graphs, train_src, n_per_graph, bank, rng)
    val_samples = build_samples(val_graphs, val_src, n_per_graph, bank, rng)
    test_samples = build_samples(test_graphs, test_src, n_per_graph, bank, rng)
    print(f"      Samples — train: {len(train_samples)}  val: {len(val_samples)}  "
          f"test: {len(test_samples)}")
    if not train_samples:
        raise RuntimeError(
            "No training samples with matching part-bank entries — check that "
            "the part bank was built from the same corpus as the dataset cache."
        )

    print("\n[5/6] Building ConditionalShapeVAE …")
    vae = ConditionalShapeVAE(res=voxel_res, latent_dim=latent_dim, cond_dim=COND_DIM).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1.0e-5)
    # Mirrors train.py's ReduceLROnPlateau pattern: steps on the same
    # IoU/chamfer composite `score` used for checkpoint selection below, so
    # the LR backs off exactly when the metric that actually matters plateaus
    # rather than on raw val loss (which can keep falling from KL/BCE terms
    # after IoU has already stalled).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=8, min_lr=1e-5,
    )
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"      VAE params: {n_params:,}")

    print(f"\n[6/6] Training ({epochs} epochs) …")
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(cfg["paths"]["results"])
    res_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint selection used to be IoU-only, which let ties (or near-ties)
    # get broken purely by chamfer noise it never looked at — observed:
    # epoch 32 (IoU=0.5718, chamfer=0.0424) was passed over for epoch 59
    # (IoU=0.5898, chamfer=0.0513 — ~20% worse) purely because IoU alone
    # ticked up. CHAMFER_WEIGHT=2.0 makes chamfer meaningfully load-bearing
    # in the selection score without letting it override a real IoU gap —
    # a calibration choice, not a rigorously derived constant; revisit if
    # selected checkpoints keep favoring poor-chamfer epochs in practice.
    CHAMFER_WEIGHT = 2.0
    best_score = -1e9
    best_iou = -1.0
    best_state = None
    best_metrics = None

    for epoch in range(1, epochs + 1):
        rng.shuffle(train_samples)
        train_loss, train_loss_parts = epoch_pass(
            vae, gnn, train_samples, device, opt=opt,
            augment=True, rng=rng, beta_kl=beta_kl, lambda_dice=lambda_dice,
        )
        val_metrics = epoch_pass(vae, gnn, val_samples, device, opt=None,
                                  beta_kl=beta_kl, lambda_dice=lambda_dice)
        score = val_metrics["iou"] - CHAMFER_WEIGHT * val_metrics["chamfer"]
        scheduler.step(score)
        cur_lr = opt.param_groups[0]["lr"]
        print(f"  Ep {epoch:3d}/{epochs}  loss={train_loss:.4f} "
              f"(bce={train_loss_parts['bce']:.4f} dice={train_loss_parts['dice']:.4f} "
              f"kl={train_loss_parts['kl']:.4f})  "
              f"val_loss={val_metrics['loss']:.4f}  IoU={val_metrics['iou']:.4f}  "
              f"chamfer={val_metrics['chamfer']:.4f}  score={score:.4f}  lr={cur_lr:.2e}",
              flush=True)
        if score > best_score:
            best_score = score
            best_iou = val_metrics["iou"]
            best_state = {k: v.clone() for k, v in vae.state_dict().items()}
            best_metrics = val_metrics

    vae.load_state_dict(best_state)
    test_metrics = epoch_pass(vae, gnn, test_samples, device, opt=None,
                               beta_kl=beta_kl, lambda_dice=lambda_dice)

    # Shape-gen is fastener-scoped (see build_samples), so the production
    # threshold that actually gates VAE fallback for these samples is
    # retrieval_tau_fastener, not the generic retrieval_tau (config.yaml) —
    # using the wrong one would report a metric HybridShapeGenerator never
    # actually uses for this component family.
    retrieval_tau = sc.get("retrieval_tau_fastener", sc.get("retrieval_tau", 0.6))
    gated_test_samples = retrieval_gated_samples(test_samples, bank, retrieval_tau)
    if gated_test_samples:
        gated_test_metrics = epoch_pass(vae, gnn, gated_test_samples, device, opt=None,
                                         beta_kl=beta_kl, lambda_dice=lambda_dice)
    else:
        gated_test_metrics = None

    print(f"\n{'='*55}")
    print("  ConditionalShapeVAE — Test Results")
    print(f"{'='*55}")
    for k, v in test_metrics.items():
        print(f"     {k:10s}: {v:.4f}")
    print(f"  (selected epoch: val IoU={best_iou:.4f}, "
          f"chamfer={best_metrics['chamfer']:.4f}, score={best_score:.4f})")
    if gated_test_metrics is not None:
        print(f"\n  Retrieval-gated (fit_score < {retrieval_tau}, "
              f"{len(gated_test_samples)}/{len(test_samples)} of test samples "
              f"— the subset that would actually reach the VAE in production):")
        for k, v in gated_test_metrics.items():
            print(f"     {k:10s}: {v:.4f}")
    else:
        print(f"\n  Retrieval-gated: no test samples had fit_score < {retrieval_tau} "
              f"— retrieval alone covers this test set at the current threshold.")

    with open(res_dir / "shape_gen_test_metrics.json", "w") as f:
        json.dump({
            "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
            "val_metrics": {k: round(v, 4) for k, v in best_metrics.items()},
            "retrieval_gated": {
                "tau": retrieval_tau,
                "n_gated": len(gated_test_samples),
                "n_total": len(test_samples),
                "metrics": ({k: round(v, 4) for k, v in gated_test_metrics.items()}
                            if gated_test_metrics is not None else None),
            },
        }, f, indent=2)

    torch.save({
        "epoch": epochs,
        "val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "trained_at": datetime.now().isoformat(),
        "encoder_ckpt": str(ckpt_path),
        "encoder_trained_at": ckpt.get("trained_at"),
        "encoder_auc": ckpt["auc"],
        "vae": vae.state_dict(),
        "res": voxel_res,
        "latent_dim": latent_dim,
        "cond_dim": COND_DIM,
        "part_bank_dir": part_bank_dir,
    }, ckpt_dir / "shape_vae.pt")

    print(f"\n  Saved → {ckpt_dir / 'shape_vae.pt'}")
    print(f"  Saved → {res_dir / 'shape_gen_test_metrics.json'}")


if __name__ == "__main__":
    main()
