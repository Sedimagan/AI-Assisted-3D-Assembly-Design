"""
train.py — Training pipeline
Run: python train.py [--config config.yaml] [--force-reload]
"""

from __future__ import annotations

import argparse
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


def link_loss(lp, z, batch, device):
    """BCE on pos+neg edges plus a weighted hard-negative term."""
    ei    = batch.edge_label_index.to(device)
    label = batch.edge_label.float().to(device)
    logit = lp(z, ei)
    base_loss = F.binary_cross_entropy_with_logits(logit, label)
    pos_mask = label > 0.5
    if pos_mask.sum() > 2:
        hard_ei = hard_negative_pairs(
            z, ei[:, pos_mask],
            n_hard=min(20, int(pos_mask.sum().item()))
        )
        if hard_ei is not None:
            hard_logits = lp(z, hard_ei)
            hard_labels = torch.zeros(hard_ei.size(1), device=device)
            hard_loss   = F.binary_cross_entropy_with_logits(hard_logits, hard_labels)
            return base_loss + 0.3 * hard_loss
    return base_loss


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(gnn, lp, loader, opt, device):
    gnn.train(); lp.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        z    = gnn(batch.x, batch.edge_index,
                   getattr(batch, "edge_attr", None))
        loss = link_loss(lp, z, batch, device)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--force-reload", action="store_true",
                        help="Re-process STEP files even if cache exists")
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
    best_overall_auc  = 0.0
    best_overall_fold = -1
    all_log_rows      = []

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\n[2/4] Building model …  ({N_FOLDS}-fold CV)")

    # ── Fold loop ─────────────────────────────────────────────────────────
    print(f"\n[3/4] Training ({N_FOLDS} folds) …")

    for fold in range(N_FOLDS):
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

        best_auc      = 0.0
        patience_left = tc["patience"]
        fold_ckpt     = ckpt_dir / f"best_fold{fold}.pt"
        log_rows      = []

        for epoch in range(1, tc["epochs"] + 1):
            t0          = time.time()
            train_loss  = train_epoch(gnn, lp, train_loader, opt, device)
            val_metrics = evaluate(gnn, lp, val_loader, device)
            sched.step(val_metrics["auc"])

            row = {"fold": fold, "epoch": epoch,
                   "train_loss": round(train_loss, 4), **{
                       k: round(v, 4) for k, v in val_metrics.items()
                   }}
            log_rows.append(row)

            elapsed = time.time() - t0
            print(f"  Ep {epoch:3d}/{tc['epochs']}  "
                  f"loss={train_loss:.4f}  "
                  f"AUC={val_metrics['auc']:.4f}  "
                  f"AP={val_metrics['ap']:.4f}  "
                  f"({elapsed:.1f}s)")

            if val_metrics["auc"] > best_auc:
                best_auc      = val_metrics["auc"]
                patience_left = tc["patience"]
                torch.save({
                    "fold":  fold,
                    "epoch": epoch,
                    "auc":   best_auc,
                    "gnn":   gnn.state_dict(),
                    "lp":    lp.state_dict(),
                    "cfg":   cfg,
                }, fold_ckpt)
                print(f"  ✓ Fold {fold+1} new best AUC={best_auc:.4f}  saved.")
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
        all_log_rows.extend(log_rows)

        print(f"\n  Fold {fold+1} test — AUC={test_metrics['auc']:.4f}"
              f"  AP={test_metrics['ap']:.4f}")

        if best_auc > best_overall_auc:
            best_overall_auc  = best_auc
            best_overall_fold = fold
            torch.save(torch.load(fold_ckpt, map_location="cpu",
                                  weights_only=False),
                       ckpt_dir / "best_overall.pt")

    # ── CV summary ────────────────────────────────────────────────────────
    import statistics
    mean_auc = statistics.mean(fold_aucs)
    std_auc  = statistics.stdev(fold_aucs) if len(fold_aucs) > 1 else 0.0
    mean_ap  = statistics.mean(fold_aps)
    std_ap   = statistics.stdev(fold_aps)  if len(fold_aps)  > 1 else 0.0

    print(f"\n{'='*55}")
    print(f"  {N_FOLDS}-Fold CV Summary")
    print(f"{'='*55}")
    for i, (a, p) in enumerate(zip(fold_aucs, fold_aps)):
        print(f"  Fold {i+1}: AUC={a:.4f}  AP={p:.4f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Mean AP  = {mean_ap:.4f} ± {std_ap:.4f}")
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
        "n_folds":   N_FOLDS,
        "fold_aucs": [round(a, 4) for a in fold_aucs],
        "fold_aps":  [round(p, 4) for p in fold_aps],
        "mean_auc":  round(mean_auc, 4),
        "std_auc":   round(std_auc,  4),
        "mean_ap":   round(mean_ap,  4),
        "std_ap":    round(std_ap,   4),
        "best_fold": best_overall_fold,
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
