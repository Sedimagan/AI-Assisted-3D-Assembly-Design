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

def link_loss(lp, z, batch, device):
    """Binary cross-entropy on positive + negative edge pairs."""
    ei    = batch.edge_label_index.to(device)
    label = batch.edge_label.float().to(device)
    logit = lp(z, ei)
    return F.binary_cross_entropy_with_logits(logit, label)


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
    )
    print(f"      {len(dataset)} graphs loaded.")

    train_data, val_data, test_data = get_splits(dataset, cfg)

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=bs)
    test_loader  = DataLoader(test_data,  batch_size=bs)

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n[2/4] Building model …")
    mc  = cfg["model"]
    tc  = cfg["training"]
    gnn, lp, device = build_model(
        in_dim   = mc["in_dim"],
        out_dim  = mc["out_dim"],
        hidden   = mc["hidden_dim"],
        heads    = mc["heads"],
        dropout  = mc["dropout"],
        edge_dim = mc["edge_dim"],
    )

    params = list(gnn.parameters()) + list(lp.parameters())
    opt    = torch.optim.Adam(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=tc["lr_factor"],
        patience=tc["lr_patience"],
    )

    # ── Training ──────────────────────────────────────────────────────────
    print("\n[3/4] Training …")
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir  = Path(cfg["paths"]["results"])
    res_dir.mkdir(parents=True, exist_ok=True)

    best_auc      = 0.0
    patience_left = tc["patience"]
    log_rows      = []

    for epoch in range(1, tc["epochs"] + 1):
        t0         = time.time()
        train_loss = train_epoch(gnn, lp, train_loader, opt, device)
        val_metrics = evaluate(gnn, lp, val_loader, device)
        sched.step(val_metrics["auc"])

        row = {"epoch": epoch, "train_loss": round(train_loss, 4), **{
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
                "epoch": epoch,
                "auc":   best_auc,
                "gnn":   gnn.state_dict(),
                "lp":    lp.state_dict(),
                "cfg":   cfg,
            }, ckpt_dir / "best.pt")
            print(f"  ✓ New best AUC={best_auc:.4f}  checkpoint saved.")
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    # ── Test evaluation ───────────────────────────────────────────────────
    print("\n[4/4] Test evaluation …")
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    gnn.load_state_dict(ckpt["gnn"])
    lp.load_state_dict(ckpt["lp"])

    test_metrics = evaluate(gnn, lp, test_loader, device)
    print("\n  ── Test results ──────────────────────────")
    for k, v in test_metrics.items():
        print(f"     {k:12s}: {v:.4f}")

    # Save logs
    with open(res_dir / "train_log.json", "w") as f:
        json.dump(log_rows, f, indent=2)
    with open(res_dir / "test_metrics.json", "w") as f:
        json.dump({k: round(v, 4) for k, v in test_metrics.items()}, f, indent=2)

    # ── Save timestamped trained model ────────────────────────────────────
    tm_dir = Path(cfg["paths"]["trained_models"])
    tm_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    auc_tag = f"auc{best_auc:.4f}".replace(".", "")
    tm_path = tm_dir / f"assembly_gnn_{ts}_{auc_tag}.pt"

    torch.save({
        "epoch":        ckpt["epoch"],
        "auc":          best_auc,
        "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
        "trained_at":   datetime.now().isoformat(),
        "source_dir":   cfg["data"]["source_dir"],
        "gnn":          gnn.state_dict(),
        "lp":           lp.state_dict(),
        "cfg":          cfg,
    }, tm_path)

    print(f"\n  Results saved        → {res_dir}")
    print(f"  Best checkpoint      → {ckpt_dir / 'best.pt'}")
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
