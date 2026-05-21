"""
infer.py — Inference on partial assemblies  (Phase 1)
Predicts missing component connections in a partial CAD assembly.

Usage:
    python infer.py --checkpoint checkpoints/best.pt --step path/to/file.step
    python infer.py --checkpoint checkpoints/best.pt --demo

Phase 2 (planned): next-component ranking via NodeRanker.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import yaml

from dataset import _parse_step, _synthetic_graph
from model   import build_model


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    mc   = cfg["model"]
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
    gnn.eval(); lp.eval()
    print(f"Loaded checkpoint (val AUC={ckpt['auc']:.4f}, epoch={ckpt['epoch']})")
    return gnn, lp, device, cfg


# ── Missing component detection ───────────────────────────────────────────────

@torch.no_grad()
def predict_missing(
    gnn, lp, graph, device, top_k: int = 5
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Score all absent node pairs and return the top-k most likely missing links.
    Each result: ((src_node, dst_node), confidence_score)
    """
    g  = graph.to(device)
    z  = gnn(g.x, g.edge_index, getattr(g, "edge_attr", None))
    n  = g.num_nodes

    existing = set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
    cands    = [(u, v) for u in range(n) for v in range(n)
                if u != v and (u, v) not in existing]
    if not cands:
        return []

    ei     = torch.tensor(cands, dtype=torch.long).T.to(device)
    scores = torch.sigmoid(lp(z, ei)).cpu()
    top_i  = scores.topk(min(top_k, len(cands))).indices

    return [((cands[i][0], cands[i][1]), round(scores[i].item(), 4))
            for i in top_i.tolist()]


# ── Pretty print ──────────────────────────────────────────────────────────────

def _bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def print_missing(results):
    print(f"\n{'═'*52}")
    print(" Missing Component — Link Predictions")
    print(f"{'═'*52}")
    if not results:
        print("  (no missing links predicted)")
        return
    for rank, ((u, v), score) in enumerate(results, 1):
        print(f"  {rank}. node {u:2d} ↔ node {v:2d}  |{_bar(score)}| {score:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Missing component inference")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--step",  default=None, help="Path to STEP file")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--demo",  action="store_true",
                        help="Run on a synthetic demo assembly")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run  python train.py  first.")
        return

    gnn, lp, device, cfg = load_checkpoint(args.checkpoint)

    if args.step:
        print(f"\nParsing STEP: {args.step}")
        graph = _parse_step(args.step)
        if graph is None:
            print("Could not parse STEP file."); return
    else:
        print("\nUsing synthetic demo assembly …")
        graph = _synthetic_graph(n_nodes=8)

    print(f"Assembly: {graph.num_nodes} components, "
          f"{graph.edge_index.size(1)} mate edges")

    missing = predict_missing(gnn, lp, graph, device, top_k=args.top_k)
    print_missing(missing)


if __name__ == "__main__":
    main()
