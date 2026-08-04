"""
infer.py — Inference on partial assemblies
Predicts missing component connections (Phase 1, LinkPredictor) and ranks
candidate next-component types (Phase 2, NodeRanker) for a partial CAD assembly.

Usage:
    python infer.py --checkpoint checkpoints/best.pt --step path/to/file.step
    python infer.py --checkpoint checkpoints/best.pt --demo
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import yaml

from dataset import _parse_step, _synthetic_graph, COMP_TYPES
from model   import build_model
from train_ranker import embed_prototypes


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


def load_ranker(ranker_path: str, gnn, device):
    """
    Load NodeRanker weights and re-embed the type-prototype table fresh
    through the already-loaded frozen encoder (never trust a cached
    embedding table — it goes stale the moment the encoder is retrained).
    Returns (nr, type_prototypes, comp_types, raw_ckpt).
    """
    from model import build_ranker

    ckpt = torch.load(ranker_path, map_location="cpu", weights_only=False)
    mc = ckpt["cfg"]["model"]
    nr = build_ranker(out_dim=mc["out_dim"], device=device)
    nr.load_state_dict(ckpt["nr"])
    nr.eval()

    type_prototypes = embed_prototypes(gnn, ckpt["type_prototypes_raw"], device)
    print(f"Loaded NodeRanker (encoder_auc={ckpt['encoder_auc']:.4f}, "
          f"trained_at={ckpt['trained_at']})")
    return nr, type_prototypes, ckpt["comp_types"], ckpt


def load_shape_generator(bank_dir: str, vae_path: str, device, retrieval_tau: float = 0.6):
    """
    Load the Phase 3 hybrid shape generator. Returns None if the part bank or
    VAE checkpoint is missing — callers degrade gracefully (same pattern as
    NodeRanker: everything upstream still works without this).
    """
    from part_bank import PartBank
    from shape_generator import ConditionalShapeVAE, ShapeRetriever, HybridShapeGenerator

    bank = PartBank(bank_dir)
    if len(bank) == 0:
        print(f"No part bank found at {bank_dir} — shape generation disabled.")
        return None
    retriever = ShapeRetriever(bank)

    vae = None
    vae_path = Path(vae_path)
    if vae_path.exists():
        ckpt = torch.load(vae_path, map_location="cpu", weights_only=False)
        vae = ConditionalShapeVAE(res=ckpt["res"], latent_dim=ckpt["latent_dim"],
                                   cond_dim=ckpt["cond_dim"]).to(device)
        vae.load_state_dict(ckpt["vae"])
        vae.eval()
        print(f"Loaded ConditionalShapeVAE (encoder_auc={ckpt['encoder_auc']:.4f}, "
              f"trained_at={ckpt['trained_at']})")
    else:
        print(f"No shape_vae.pt at {vae_path} — retrieval-only mode (no VAE fallback).")

    return HybridShapeGenerator(retriever, vae, device, retrieval_tau=retrieval_tau)


@torch.no_grad()
def generate_missing_shape(
    hsg, gnn, graph, comp_type: str, open_joint_extents=None,
    open_joint_centroid=None, category: str | None = None, device=None,
    mode: str = "auto",
):
    """
    Produce a ShapeResult for an explicitly-given missing-component type
    (e.g. from AssemblyTemplateDB.get_missing(), or a NodeRanker top-1 pick
    the caller has already selected). Returns None if hsg is None (artifacts
    missing), comp_type is falsy, or nothing could be retrieved/generated.
    """
    if hsg is None or not comp_type:
        return None
    from part_bank import PartBank
    from shape_generator import build_conditioning_vector, estimate_target_bbox

    comp_type_idx = COMP_TYPES.index(comp_type)

    g = graph.to(device) if device is not None else graph
    z = gnn(g.x, g.edge_index, getattr(g, "edge_attr", None))
    ctx = z.mean(0) if z.size(0) > 0 else torch.zeros(z.size(-1), device=z.device)

    bank: PartBank = hsg.retriever.bank
    target_bbox = estimate_target_bbox(open_joint_extents, bank, comp_type, category)
    bbox_norm = target_bbox / (target_bbox.max() + 1e-9)

    if g.num_nodes > 0:
        log_vol = g.x[:, 8].mean().item()
        max_extent = g.x[:, 10:13].max(dim=1).values.mean().item()
    else:
        log_vol, max_extent = 0.0, 0.0
    neighbor_scale = [log_vol, max_extent]

    cond_vec = build_conditioning_vector(ctx, comp_type_idx, bbox_norm, neighbor_scale)
    centroid = open_joint_centroid if open_joint_centroid is not None else [0.0, 0.0, 0.0]

    return hsg.generate(comp_type, category, target_bbox, cond_vec, centroid, mode=mode)


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


# ── Next-component ranking (Phase 2, NodeRanker) ──────────────────────────────

@torch.no_grad()
def predict_next_component(
    gnn, nr, type_prototypes, graph, device, top_k: int = len(COMP_TYPES)
) -> List[Tuple[str, float]]:
    """
    Rank the fixed candidate component types by cosine similarity to the
    partial assembly's context vector. Works even for single-node graphs
    (unlike predict_missing, which needs >=2 nodes).
    Each result: (type_name, cosine_similarity) sorted best-first.
    """
    g = graph.to(device)
    z = gnn(g.x, g.edge_index, getattr(g, "edge_attr", None))
    if z.size(0) == 0:
        return []

    scores = nr(z, type_prototypes).cpu()
    top_i  = scores.topk(min(top_k, scores.size(0))).indices

    return [(COMP_TYPES[i], round(scores[i].item(), 4)) for i in top_i.tolist()]


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


def print_next_component(results):
    print(f"\n{'═'*52}")
    print(" Next Component — Ranked Suggestions (NodeRanker)")
    print(f"{'═'*52}")
    if not results:
        print("  (no ranking available)")
        return
    for rank, (type_name, score) in enumerate(results, 1):
        pct = max(0.0, (score + 1) / 2)  # cosine sim [-1,1] -> bar [0,1]
        print(f"  {rank}. {type_name:10s}  |{_bar(pct)}| {score:+.3f}")


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
