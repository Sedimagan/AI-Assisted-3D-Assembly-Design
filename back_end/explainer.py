"""
explainer.py — GNNExplainer interpretability layer

Post-hoc explanations for the two trained prediction heads (Phase 1
LinkPredictor, Phase 2 NodeRanker), built on PyG's Explainer/GNNExplainer.
No retraining: GNNExplainer optimises a small edge mask (+ per-node
per-feature mask) against the already-trained, frozen AssemblyGNN encoder
and task head by gradient descent — a few hundred steps, seconds per
explanation, same "frozen encoder" pattern already used by train_ranker.py
and train_shape_gen.py.

Approach: rather than ask GNNExplainer to explain an entire batched output
(PyG's edge/node task-level conventions expect edge_label_index/batching
that don't map cleanly onto this project's single-graph, single-candidate
inference calls), each explanation wraps the encoder + head in a tiny
nn.Module whose forward pass bakes in the one candidate being explained
(a specific (u, v) edge, or a specific candidate component type) and
returns a single scalar. GNNExplainer is then run in "graph regression"
mode against that scalar — task_level="graph" because the whole input
graph produces one number, mode="regression" because we want to preserve
the model's own raw score under masking (explanation_type="model"), not
match a ground-truth class label.

Usage:
    python explainer.py --checkpoint checkpoints/best_serving.pt --demo
    python explainer.py --checkpoint checkpoints/best_serving.pt --step path/to/file.step
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch_geometric.explain import Explainer, GNNExplainer

from dataset import _parse_step, _synthetic_graph, COMP_TYPES
from model import AssemblyGNN, LinkPredictor, NodeRanker
from infer import load_checkpoint, load_ranker, predict_missing, predict_next_component

# Node feature layout — must match _parse_step()'s docstring in dataset.py
# exactly (22-dim: 8 type one-hot + 14 geometric/SDF scalars).
NODE_FEATURE_NAMES = [
    *[f"type={t}" for t in COMP_TYPES],   # [0:8]
    "log_volume",                          # [8]
    "log_surface_area",                    # [9]
    "bbox_dx",                             # [10]
    "bbox_dy",                             # [11]
    "bbox_dz",                             # [12]
    "elongation",                          # [13]
    "flatness",                            # [14]
    "aspect_xy",                           # [15]
    "aspect_yz",                           # [16]
    "sphericity",                          # [17]
    "sdf_mean",                            # [18]
    "sdf_variance",                        # [19]
    "sa_over_v",                           # [20]
    "log_n_holes",                         # [21]
]
assert len(NODE_FEATURE_NAMES) == 22


# ── Explainable wrappers ────────────────────────────────────────────────────
# Each bakes in the one candidate prediction being explained, so the model's
# forward signature reduces to the plain (x, edge_index, edge_attr) shape
# GNNExplainer expects, returning a single scalar it can regress against.

class _LinkExplainWrapper(nn.Module):
    """Explains LinkPredictor's raw logit for one fixed candidate edge (u, v)."""

    def __init__(self, gnn: AssemblyGNN, lp: LinkPredictor, u: int, v: int,
                 pos: torch.Tensor | None):
        super().__init__()
        self.gnn, self.lp = gnn, lp
        self.u, self.v, self.pos = u, v, pos

    def forward(self, x, edge_index, edge_attr=None):
        z = self.gnn(x, edge_index, edge_attr)
        ei = torch.tensor([[self.u], [self.v]], dtype=torch.long, device=x.device)
        return self.lp(z, ei, self.pos)  # (1,) raw logit


class _RankExplainWrapper(nn.Module):
    """Explains NodeRanker's raw score for one fixed candidate component type."""

    def __init__(self, gnn: AssemblyGNN, nr: NodeRanker,
                 type_prototypes: torch.Tensor, candidate_idx: int):
        super().__init__()
        self.gnn, self.nr = gnn, nr
        self.type_prototypes, self.candidate_idx = type_prototypes, candidate_idx

    def forward(self, x, edge_index, edge_attr=None):
        z = self.gnn(x, edge_index, edge_attr)
        scores = self.nr(z, self.type_prototypes)          # (n_types,)
        return scores[self.candidate_idx:self.candidate_idx + 1]  # (1,)


def _make_explainer(model: nn.Module, epochs: int = 200) -> Explainer:
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="regression", task_level="graph", return_type="raw"),
    )


def _summarize_edges(graph, edge_mask: torch.Tensor, top_k: int = 5) -> List[dict]:
    if edge_mask is None or edge_mask.numel() == 0:
        return []
    order = edge_mask.argsort(descending=True)[:top_k]
    node_type = graph.x[:, :len(COMP_TYPES)].argmax(dim=1)
    out = []
    for i in order.tolist():
        if edge_mask[i].item() <= 0:
            continue
        s, d = graph.edge_index[0, i].item(), graph.edge_index[1, i].item()
        out.append({
            "edge": (s, d),
            "src_type": COMP_TYPES[node_type[s].item()],
            "dst_type": COMP_TYPES[node_type[d].item()],
            "importance": round(edge_mask[i].item(), 4),
        })
    return out


def _summarize_node_features(node_mask: torch.Tensor, node_idx: int, top_k: int = 5) -> List[dict]:
    row = node_mask[node_idx]
    order = row.argsort(descending=True)[:top_k]
    out = []
    for j in order.tolist():
        if row[j].item() <= 0:
            continue
        out.append({"feature": NODE_FEATURE_NAMES[j], "importance": round(row[j].item(), 4)})
    return out


@torch.no_grad()
def _current_score(model: nn.Module, graph, device) -> float:
    g = graph.to(device)
    return model(g.x, g.edge_index, getattr(g, "edge_attr", None)).item()


# ── Public API ───────────────────────────────────────────────────────────────

def explain_missing_link(gnn, lp, graph, u: int, v: int, device,
                          epochs: int = 200, top_k: int = 5) -> dict:
    """
    Explain why LinkPredictor scored the (u, v) candidate the way it did.
    Returns which EXISTING edges (message-passing paths) and which node
    features on u and v most influenced that score.
    """
    g = graph.to(device)
    wrapper = _LinkExplainWrapper(gnn, lp, u, v, getattr(g, "pos", None)).to(device)
    explainer = _make_explainer(wrapper, epochs=epochs)
    explanation = explainer(g.x, g.edge_index, edge_attr=getattr(g, "edge_attr", None))

    confidence = torch.sigmoid(torch.tensor(_current_score(wrapper, g, device))).item()
    return {
        "target": {"edge": (u, v), "confidence": round(confidence, 4)},
        "contributing_edges": _summarize_edges(g, explanation.edge_mask, top_k),
        "contributing_features": {
            "u": _summarize_node_features(explanation.node_mask, u, top_k),
            "v": _summarize_node_features(explanation.node_mask, v, top_k),
        },
    }


def explain_next_component(gnn, nr, type_prototypes, graph, comp_type: str, device,
                            epochs: int = 200, top_k: int = 5) -> dict:
    """
    Explain why NodeRanker ranked `comp_type` the way it did for this
    partial assembly. Returns which existing connections and which
    existing components most influenced that ranking.
    """
    candidate_idx = COMP_TYPES.index(comp_type)
    g = graph.to(device)
    wrapper = _RankExplainWrapper(gnn, nr, type_prototypes, candidate_idx).to(device)
    explainer = _make_explainer(wrapper, epochs=epochs)
    explanation = explainer(g.x, g.edge_index, edge_attr=getattr(g, "edge_attr", None))

    score = _current_score(wrapper, g, device)
    node_type = g.x[:, :len(COMP_TYPES)].argmax(dim=1)
    node_importance = explanation.node_mask.sum(dim=-1)
    order = node_importance.argsort(descending=True)[:top_k]
    top_nodes = [
        {"node": i, "type": COMP_TYPES[node_type[i].item()],
         "importance": round(node_importance[i].item(), 4)}
        for i in order.tolist() if node_importance[i].item() > 0
    ]
    return {
        "target": {"type": comp_type, "score": round(score, 4)},
        "contributing_edges": _summarize_edges(g, explanation.edge_mask, top_k),
        "contributing_nodes": top_nodes,
    }


# ── Pretty print ─────────────────────────────────────────────────────────────

def print_link_explanation(result: dict) -> None:
    u, v = result["target"]["edge"]
    print(f"\nExplaining missing link ({u} → {v})  confidence={result['target']['confidence']}")
    print("  Contributing existing edges (message-passing paths):")
    for e in result["contributing_edges"]:
        s, d = e["edge"]
        print(f"    {s}({e['src_type']}) — {d}({e['dst_type']})   importance={e['importance']}")
    for node_key in ("u", "v"):
        print(f"  Contributing features on node {result['target']['edge'][0 if node_key=='u' else 1]} ({node_key}):")
        for f in result["contributing_features"][node_key]:
            print(f"    {f['feature']:<20s} importance={f['importance']}")


def print_rank_explanation(result: dict) -> None:
    print(f"\nExplaining next-component ranking: {result['target']['type']}  score={result['target']['score']}")
    print("  Contributing existing edges:")
    for e in result["contributing_edges"]:
        s, d = e["edge"]
        print(f"    {s}({e['src_type']}) — {d}({e['dst_type']})   importance={e['importance']}")
    print("  Contributing existing components:")
    for n in result["contributing_nodes"]:
        print(f"    node {n['node']} ({n['type']})   importance={n['importance']}")


# ── CLI demo ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GNNExplainer interpretability demo")
    parser.add_argument("--checkpoint", default="checkpoints/best_serving.pt")
    parser.add_argument("--ranker",     default="checkpoints/node_ranker.pt")
    parser.add_argument("--step",  default=None, help="Path to STEP file")
    parser.add_argument("--demo",  action="store_true", help="Use a synthetic demo assembly")
    parser.add_argument("--epochs", type=int, default=200, help="GNNExplainer optimisation steps")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    gnn, lp, device, _ = load_checkpoint(args.checkpoint)

    if args.step:
        print(f"\nParsing STEP: {args.step}")
        graph = _parse_step(args.step)
        if graph is None:
            print("Could not parse STEP file."); return
    else:
        print("\nUsing synthetic demo assembly …")
        graph = _synthetic_graph(n_nodes=8)

    print(f"Assembly: {graph.num_nodes} components, {graph.edge_index.size(1)} mate edges")

    if graph.num_nodes >= 2:
        missing = predict_missing(gnn, lp, graph, device, top_k=1)
        if missing:
            (u, v), _conf = missing[0]
            result = explain_missing_link(gnn, lp, graph, u, v, device, epochs=args.epochs)
            print_link_explanation(result)

    if Path(args.ranker).exists():
        nr, type_prototypes, _comp_types, _ = load_ranker(args.ranker, gnn, device)
        ranked = predict_next_component(gnn, nr, type_prototypes, graph, device, top_k=1)
        if ranked:
            comp_type, _score = ranked[0]
            result = explain_next_component(gnn, nr, type_prototypes, graph, comp_type, device,
                                             epochs=args.epochs)
            print_rank_explanation(result)
    else:
        print(f"\nNo NodeRanker checkpoint at {args.ranker} — skipping Phase 2 explanation.")


if __name__ == "__main__":
    main()
