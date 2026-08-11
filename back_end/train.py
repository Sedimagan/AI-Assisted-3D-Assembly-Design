"""
train.py — Training pipeline
Run: python train.py [--config config.yaml] [--force-reload]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

from dataset  import AssemblyDataset, get_splits
from model    import build_model
from evaluate import evaluate


# ── Category sample weights ──────────────────────────────────────────────────

CATEGORY_WEIGHTS = {
    # Weights are inverse-frequency (relative to the largest category,
    # Bench_vice=54 graphs), capped at 3.0. Real counts from the R32
    # corpus overhaul (193 folders total, 2026-07-31, after moving
    # rejected/non_compatible_formats/slow_or_unstable folders out of
    # Best_models_for_training) — re-derive if the corpus changes again.
    # Names must match the directory names under
    # Source_3d_models/Best_models_for_training/.
    "Bench_vice":  1.0,   # 54 graphs
    "Pipe_vice":   1.3,   # 42 graphs
    "C_Clamps":    1.7,   # 31 graphs
    "Gate_Valve":  2.6,   # 21 graphs
    "Press_Tool":  2.6,   # 21 graphs
    "Crane_hook":  3.0,   # 16 graphs (capped, uncapped 3.4)
    "Tool_Post":   3.0,   # 8 graphs (capped, uncapped 6.75)
}


# ── Loss ─────────────────────────────────────────────────────────────────────

def hard_negative_pairs(z, pos_edge_index, n_hard):
    """For each positive-edge source, find the most similar non-neighbour node."""
    if pos_edge_index.size(1) == 0:
        return None
    src, _ = pos_edge_index
    connected = set(zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()))
    z_norm = F.normalize(z.detach(), dim=-1)
    sim    = z_norm @ z_norm.T
    hard_src, hard_dst = [], []
    for u in src.tolist()[:n_hard]:
        scores = sim[u].clone()
        scores[u] = -1.0
        for v in range(z.size(0)):
            if (u, v) in connected or (v, u) in connected:
                scores[v] = -1.0
        w = int(scores.argmax().item())
        if w != u:
            hard_src.append(u)
            hard_dst.append(w)
    if not hard_src:
        return None
    return torch.tensor([hard_src, hard_dst], dtype=torch.long, device=z.device)


def link_loss(lp, z, batch, device, sample_weight=1.0):
    """BCE on pos+neg edges plus a weighted hard-negative term."""
    ei    = batch.edge_label_index.to(device)
    label = batch.edge_label.float().to(device)
    pos   = batch.pos.to(device) if getattr(batch, "pos", None) is not None else None
    logit = lp(z, ei, pos)
    base_loss = F.binary_cross_entropy_with_logits(logit, label)
    pos_mask = label > 0.5
    if pos_mask.sum() > 2:
        hard_ei = hard_negative_pairs(
            z, ei[:, pos_mask],
            n_hard=min(20, int(pos_mask.sum().item()))
        )
        if hard_ei is not None:
            hard_logits = lp(z, hard_ei, pos)
            hard_labels = torch.zeros(hard_ei.size(1), device=device)
            hard_loss   = F.binary_cross_entropy_with_logits(hard_logits, hard_labels)
            return (base_loss + 0.3 * hard_loss) * sample_weight
    return base_loss * sample_weight


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(gnn, lp, loader, opt, device):
    gnn.train(); lp.train()
    total_loss = 0.0
    n_batches = len(loader)
    for i, batch in enumerate(loader):
        # Heartbeat print so train_monitor.sh's log-growth stall-detector
        # doesn't false-trigger on a long-running-but-healthy epoch (it
        # kills the process after 300s of zero new log output).
        if i % 5 == 0:
            print(f"    batch {i+1}/{n_batches}", flush=True)
        batch = batch.to(device)
        opt.zero_grad()
        z    = gnn(batch.x, batch.edge_index,
                   getattr(batch, "edge_attr", None))
        # P5: category-based sample weighting
        weight = 1.0
        cat = getattr(batch, 'category', None)
        if cat is not None:
            if isinstance(cat, str):
                weight = CATEGORY_WEIGHTS.get(cat, 1.0)
            elif isinstance(cat, (list, tuple)):
                weights = [CATEGORY_WEIGHTS.get(c, 1.0) for c in cat]
                weight = sum(weights) / len(weights)
        loss = link_loss(lp, z, batch, device, sample_weight=weight)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        # Autograd graphs commonly contain reference cycles (a tensor's
        # grad_fn holding saved tensors that, through the graph, end up
        # referencing back to it) — CPython's refcounting alone can't free
        # those, only the cyclic GC can, and it doesn't run on a fixed
        # schedule tied to how fast MPS-backed tensors accumulate. Until
        # gc.collect() actually runs, those tensors look "still referenced"
        # from empty_cache()'s point of view and it has nothing to free.
        # Drop our own references first so the cycle has nothing external
        # keeping it alive either.
        del z, loss, batch
        # See below on why this is cleared this often — even with an
        # end-of-epoch clear, swap pressure was still building within a
        # single fold (observed 92%+ swap usage) because each batch's
        # distinct shape can pin real memory until the *next* clear, and on
        # an 8-batch epoch that's a long time to hold onto several
        # multi-GB compiled graphs at once. Every 3 batches bounds how many
        # distinct shapes are ever resident simultaneously, at the cost of
        # some recompilation overhead — worth it given the alternative is
        # the system running out of swap entirely.
        if device.type == "mps" and (i + 1) % 3 == 0:
            gc.collect()
            torch.mps.empty_cache()
    # MPS caches a compiled graph executable per distinct tensor shape it
    # sees, with no automatic eviction. PyG batches variable-sized graphs
    # and `shuffle=True` reshuffles batch composition every epoch, so on a
    # corpus with wide graph-size variance (this one has assemblies from a
    # few nodes up to 170+) nearly every batch triggers a new compiled
    # shape — the cache then grows unbounded over a long run (observed:
    # process RSS climbing from ~17GB to 33GB over ~28 epochs, unrelated to
    # any Python-level tensor reference). Clearing it every epoch is cheap
    # (next epoch just recompiles shapes it re-encounters) and bounds
    # memory to what a single epoch's shape diversity actually needs.
    if device.type == "mps":
        gc.collect()
        torch.mps.empty_cache()
    return total_loss / len(loader)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--force-reload", action="store_true",
                        help="Re-process STEP files even if cache exists")
    parser.add_argument("--start-fold",  type=int, default=0,
                        help="Resume from this fold (0-indexed, loads prior checkpoints)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Dataset ───────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset …")
    dataset = AssemblyDataset(
        source_dir    = cfg["data"]["source_dir"],
        processed_dir = cfg["data"]["processed_dir"],
        force_reload  = args.force_reload,
        categories    = cfg["data"].get("categories") or None,
    )
    print(f"      {len(dataset)} graphs loaded.")

    # ── Dataset summary ──────────────────────────────────────────────────
    n_graphs = len(dataset)
    total_nodes = sum(dataset[i].num_nodes for i in range(n_graphs))
    total_edges = sum(dataset[i].edge_index.size(1) for i in range(n_graphs))
    cat_counts = {}
    for i in range(n_graphs):
        c = dataset.graph_categories[i] if i < len(dataset.graph_categories) else ''
        cat_counts[c] = cat_counts.get(c, 0) + 1
    cat_summary = " ".join(f"{k}={v}" for k, v in sorted(cat_counts.items()) if k)
    print(f"      Dataset: {n_graphs} graphs | "
          f"mean nodes={total_nodes / n_graphs:.1f} | "
          f"mean edges={total_edges / n_graphs:.1f} | "
          f"categories: {cat_summary}")

    # ── Cross-validation setup ────────────────────────────────────────────
    tc       = cfg["training"]
    mc       = cfg["model"]
    N_FOLDS  = tc.get("n_folds", 5)
    bs       = tc["batch_size"]

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir  = Path(cfg["paths"]["results"])
    res_dir.mkdir(parents=True, exist_ok=True)

    fold_aucs = []
    fold_aps  = []
    fold_random_aps = []
    best_overall_auc  = 0.0
    best_overall_fold = -1
    all_log_rows      = []

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\n[2/4] Building model …  ({N_FOLDS}-fold CV)")
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    # ── Fold loop ─────────────────────────────────────────────────────────
    start_fold = args.start_fold
    if start_fold > 0:
        print(f"\n  Resuming from fold {start_fold} — loading prior checkpoints …")
        for prev in range(start_fold):
            prev_ckpt = ckpt_dir / f"best_fold{prev}.pt"
            if not prev_ckpt.exists():
                raise FileNotFoundError(f"Cannot resume: {prev_ckpt} not found")
            ck = torch.load(prev_ckpt, map_location="cpu", weights_only=False)
            gnn_tmp, lp_tmp, dev_tmp = build_model(
                in_dim=mc["in_dim"], out_dim=mc["out_dim"],
                hidden=mc["hidden_dim"], heads=mc["heads"],
                dropout=mc["dropout"], edge_dim=mc["edge_dim"],
            )
            gnn_tmp.load_state_dict(ck["gnn"])
            lp_tmp.load_state_dict(ck["lp"])
            _, _, test_data_prev = get_splits(dataset, cfg,
                                              fold_idx=prev, n_folds=N_FOLDS)
            test_loader_prev = DataLoader(test_data_prev, batch_size=bs)
            prev_metrics = evaluate(gnn_tmp, lp_tmp, test_loader_prev, dev_tmp)
            fold_aucs.append(prev_metrics["auc"])
            fold_aps.append(prev_metrics["ap"])
            fold_random_aps.append(prev_metrics["random_ap"])
            if ck["auc"] > best_overall_auc:
                best_overall_auc = ck["auc"]
                best_overall_fold = prev
            print(f"  Fold {prev+1}: AUC={prev_metrics['auc']:.4f}  "
                  f"AP={prev_metrics['ap']:.4f}  (from checkpoint, "
                  f"random={prev_metrics['random_ap']:.4f})")

    print(f"\n[3/4] Training ({N_FOLDS} folds) …")

    for fold in range(start_fold, N_FOLDS):
        print(f"\n{'='*55}")
        print(f"  FOLD {fold + 1} / {N_FOLDS}")
        print(f"{'='*55}")

        train_data, val_data, test_data = get_splits(dataset, cfg,
                                                      fold_idx=fold,
                                                      n_folds=N_FOLDS)

        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
        val_loader   = DataLoader(val_data,   batch_size=bs)
        test_loader  = DataLoader(test_data,  batch_size=bs)

        # fresh model for each fold
        gnn, lp, device = build_model(
            in_dim   = mc["in_dim"],
            out_dim  = mc["out_dim"],
            hidden   = mc["hidden_dim"],
            heads    = mc["heads"],
            dropout  = mc["dropout"],
            edge_dim = mc["edge_dim"],
        )

        params = list(gnn.parameters()) + list(lp.parameters())
        opt    = torch.optim.Adam(params, lr=tc["lr"],
                                  weight_decay=tc["weight_decay"])
        sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=tc["lr_factor"],
            patience=tc["lr_patience"],
        )

        best_auc       = 0.0
        patience_left  = tc["patience"]
        fold_ckpt      = ckpt_dir / f"best_fold{fold}.pt"
        log_rows       = []
        # Raw single-epoch val AUC on a ~33-graph val set is noisy enough
        # that early stopping can latch onto a lucky epoch (observed: R34
        # fold 3 hit val AUC=0.7364 mid-training, saved that checkpoint, and
        # its test AUC came back at 0.4899 — a 0.25 gap). Comparing/saving
        # against a trailing-window average instead damps single-epoch
        # noise without needing a bigger val set.
        SMOOTH_WINDOW  = 3
        val_auc_history: list = []

        for epoch in range(1, tc["epochs"] + 1):
            t0          = time.time()
            train_loss  = train_epoch(gnn, lp, train_loader, opt, device)
            val_metrics = evaluate(gnn, lp, val_loader, device)
            # evaluate()'s forward passes populate the same MPS compiled-
            # shape cache as training (it's a per-shape cache regardless of
            # grad state) — clear again here so the val set's shapes don't
            # linger resident until the next epoch's train_epoch() call.
            if device.type == "mps":
                gc.collect()
                torch.mps.empty_cache()

            val_auc_history.append(val_metrics["auc"])
            smoothed_auc = (sum(val_auc_history[-SMOOTH_WINDOW:])
                             / len(val_auc_history[-SMOOTH_WINDOW:]))
            sched.step(smoothed_auc)

            row = {"fold": fold, "epoch": epoch,
                   "train_loss": round(train_loss, 4),
                   "smoothed_auc": round(smoothed_auc, 4), **{
                       k: round(v, 4) for k, v in val_metrics.items()
                   }}
            log_rows.append(row)

            elapsed = time.time() - t0
            print(f"  Ep {epoch:3d}/{tc['epochs']}  "
                  f"loss={train_loss:.4f}  "
                  f"AUC={val_metrics['auc']:.4f}  "
                  f"AP={val_metrics['ap']:.4f}  "
                  f"smoothAUC={smoothed_auc:.4f}  "
                  f"({elapsed:.1f}s)")

            if smoothed_auc > best_auc:
                best_auc      = smoothed_auc
                patience_left = tc["patience"]
                torch.save({
                    "fold":  fold,
                    "epoch": epoch,
                    "auc":   best_auc,
                    "gnn":   gnn.state_dict(),
                    "lp":    lp.state_dict(),
                    "cfg":   cfg,
                }, fold_ckpt)
                print(f"  ✓ Fold {fold+1} new best smoothed AUC={best_auc:.4f}  saved.")
            else:
                patience_left -= 1
                if patience_left == 0:
                    print(f"\n  Early stopping at epoch {epoch}.")
                    break

        # ── Per-fold test evaluation ──────────────────────────────────────
        ckpt = torch.load(fold_ckpt, map_location=device, weights_only=False)
        gnn.load_state_dict(ckpt["gnn"])
        lp.load_state_dict(ckpt["lp"])

        test_metrics = evaluate(gnn, lp, test_loader, device)
        fold_aucs.append(test_metrics["auc"])
        fold_aps.append(test_metrics["ap"])
        fold_random_aps.append(test_metrics["random_ap"])
        all_log_rows.extend(log_rows)

        print(f"\n  Fold {fold+1} test — AUC={test_metrics['auc']:.4f}"
              f"  AP={test_metrics['ap']:.4f}"
              f"  (random={test_metrics['random_ap']:.4f})")

        if best_auc > best_overall_auc:
            best_overall_auc  = best_auc
            best_overall_fold = fold
            torch.save(torch.load(fold_ckpt, map_location="cpu",
                                  weights_only=False),
                       ckpt_dir / "best_overall.pt")

        # Per-epoch clearing (inside train_epoch) bounds memory *within* a
        # fold, but each fold builds a brand-new gnn/lp/opt/sched — observed
        # memory climbing to new session-highs specifically partway through
        # fold 3 (after folds 1-2 completed cleanly) suggests something
        # from the *previous* fold's model/optimizer/loaders wasn't fully
        # released before the next fold's objects were built on top of it.
        # Drop our references explicitly and collect before moving on.
        del gnn, lp, opt, sched, train_loader, val_loader, test_loader, ckpt
        if device.type == "mps":
            gc.collect()
            torch.mps.empty_cache()

    # ── CV summary ────────────────────────────────────────────────────────
    import statistics
    mean_auc = statistics.mean(fold_aucs)
    std_auc  = statistics.stdev(fold_aucs) if len(fold_aucs) > 1 else 0.0
    mean_ap  = statistics.mean(fold_aps)
    std_ap   = statistics.stdev(fold_aps)  if len(fold_aps)  > 1 else 0.0
    mean_random_ap = statistics.mean(fold_random_aps)
    std_random_ap  = statistics.stdev(fold_random_aps) if len(fold_random_aps) > 1 else 0.0

    print(f"\n{'='*55}")
    print(f"  {N_FOLDS}-Fold CV Summary")
    print(f"{'='*55}")
    for i, (a, p, r) in enumerate(zip(fold_aucs, fold_aps, fold_random_aps)):
        print(f"  Fold {i+1}: AUC={a:.4f}  AP={p:.4f}  (random={r:.4f})")
    print(f"  ─────────────────────────────────────────")
    print(f"  Mean AUC        = {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Mean AP         = {mean_ap:.4f} ± {std_ap:.4f}")
    print(f"  Mean random AP  = {mean_random_ap:.4f} ± {std_random_ap:.4f}"
          f"  (chance baseline — AP lift over this is the real signal)")
    print(f"  Best overall fold: {best_overall_fold + 1}"
          f"  (val AUC={best_overall_auc:.4f})")

    # ── [4/4] Final test evaluation using best overall model ──────────────
    print("\n[4/4] Final test evaluation (best_overall.pt) …")
    ckpt   = torch.load(ckpt_dir / "best_overall.pt", map_location=device,
                        weights_only=False)
    gnn, lp, device = build_model(
        in_dim   = mc["in_dim"],
        out_dim  = mc["out_dim"],
        hidden   = mc["hidden_dim"],
        heads    = mc["heads"],
        dropout  = mc["dropout"],
        edge_dim = mc["edge_dim"],
    )
    gnn.load_state_dict(ckpt["gnn"])
    lp.load_state_dict(ckpt["lp"])

    # Use fixed test set from the best fold
    _, _, test_data = get_splits(dataset, cfg,
                                 fold_idx=best_overall_fold,
                                 n_folds=N_FOLDS)
    test_loader = DataLoader(test_data, batch_size=bs)
    test_metrics = evaluate(gnn, lp, test_loader, device)
    print("\n  ── Test results (best overall model) ─────")
    for k, v in test_metrics.items():
        print(f"     {k:12s}: {v:.4f}")

    # ── Save logs and metrics ─────────────────────────────────────────────
    cv_summary = {
        "n_folds":        N_FOLDS,
        "fold_aucs":      [round(a, 4) for a in fold_aucs],
        "fold_aps":       [round(p, 4) for p in fold_aps],
        "fold_random_aps": [round(r, 4) for r in fold_random_aps],
        "mean_auc":       round(mean_auc, 4),
        "std_auc":        round(std_auc,  4),
        "mean_ap":        round(mean_ap,  4),
        "std_ap":         round(std_ap,   4),
        "mean_random_ap": round(mean_random_ap, 4),
        "std_random_ap":  round(std_random_ap,  4),
        "best_fold":      best_overall_fold,
    }
    with open(res_dir / "train_log.json", "w") as f:
        json.dump(all_log_rows, f, indent=2)
    with open(res_dir / "test_metrics.json", "w") as f:
        json.dump({k: round(v, 4) for k, v in test_metrics.items()}, f, indent=2)
    with open(res_dir / "cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    # ── Save timestamped trained model ────────────────────────────────────
    tm_dir = Path(cfg["paths"]["trained_models"])
    tm_dir.mkdir(parents=True, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    auc_tag = f"auc{best_overall_auc:.4f}".replace(".", "")
    tm_path = tm_dir / f"assembly_gnn_{ts}_{auc_tag}.pt"

    torch.save({
        "epoch":        ckpt["epoch"],
        "auc":          best_overall_auc,
        "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
        "cv_summary":   cv_summary,
        "trained_at":   datetime.now().isoformat(),
        "source_dir":   cfg["data"]["source_dir"],
        "gnn":          gnn.state_dict(),
        "lp":           lp.state_dict(),
        "cfg":          cfg,
    }, tm_path)

    print(f"\n  Results saved        → {res_dir}")
    print(f"  Best overall ckpt    → {ckpt_dir / 'best_overall.pt'}")
    print(f"  Trained model export → {tm_path}")

    # ── Promote to best_serving.pt only if this run beats the incumbent on
    # BOTH mean AUC and mean AP-lift-over-chance ──
    #
    # This used to compare `(mean_auc, mean_ap) <= (prev_auc, prev_ap)` --
    # Python tuple comparison is LEXICOGRAPHIC, so it only ever looked at AP
    # when AUC was an exact tie; any AUC improvement promoted regardless of
    # what AP did, silently contradicting the "beats on both" behavior
    # documented in docs/pipeline_architecture.html. Caught when R37 (mean
    # AUC=0.6765, mean AP=0.6696) promoted over R36 (mean AUC=0.6277, mean
    # AP=0.7747) despite a lower raw AP -- the promotion was substantively
    # right (see below) but the gate got there by luck, not by actually
    # checking.
    #
    # Raw AP also isn't safely comparable across a neg_ratio change (a
    # random scorer's expected AP shifts with the pos:neg ratio -- see
    # evaluate.py's random_ap) -- R36 was measured at neg_ratio=0.5
    # (chance≈0.667), R37 at neg_ratio=1.0 (chance=0.5), so comparing raw
    # mean_ap across them is apples-to-oranges regardless of the tuple bug.
    # Compare AP-lift-over-chance instead, which is what's actually
    # comparable. Older checkpoints saved before mean_random_ap existed
    # default to 0.5 (the current standard ratio) rather than guessing their
    # true historical chance level.
    serving_path = ckpt_dir / "best_serving.pt"
    promote = True
    mean_ap_lift = mean_ap - mean_random_ap
    if serving_path.exists():
        prev = torch.load(serving_path, map_location="cpu", weights_only=False)
        prev_summary  = prev.get("cv_summary", {})
        prev_auc      = prev_summary.get("mean_auc", 0.0)
        prev_ap       = prev_summary.get("mean_ap", 0.0)
        prev_ap_lift  = prev_ap - prev_summary.get("mean_random_ap", 0.5)
        if mean_auc <= prev_auc or mean_ap_lift <= prev_ap_lift:
            promote = False
            print(f"\n  ⊘ Serving model NOT updated — incumbent "
                  f"(AUC={prev_auc:.4f}, AP-lift={prev_ap_lift:.4f}) is at least as good as "
                  f"this run (AUC={mean_auc:.4f}, AP-lift={mean_ap_lift:.4f})")
    if promote:
        torch.save({
            "epoch":        ckpt["epoch"],
            "auc":          best_overall_auc,
            "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
            "cv_summary":   cv_summary,
            "trained_at":   datetime.now().isoformat(),
            "gnn":          gnn.state_dict(),
            "lp":           lp.state_dict(),
            "cfg":          cfg,
        }, serving_path)
        print(f"\n  ✓ best_serving.pt updated — AUC={mean_auc:.4f}, AP={mean_ap:.4f}")

    # ── Assembly template DB ──────────────────────────────────────────────
    print("\n  Building assembly template cache …")
    try:
        from assembly_templates import AssemblyTemplateDB
        db = AssemblyTemplateDB("data/assembly_templates.json")
        db.build(cfg["data"]["processed_dir"])
        db.save()
    except Exception as e:
        print(f"  [TemplateDB] Warning: could not build templates — {e}")


if __name__ == "__main__":
    main()
