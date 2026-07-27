"""
model.py — AssemblyGNN
3-layer Relational GAT encoder with type-specific parameters for both node
types (component type one-hot) and edge types (joint type one-hot), plus one
Phase 1 task head:
  - LinkPredictor  : detects missing component connections (binary edge scoring)

Phase 2 (planned):
  - NodeRanker     : ranks candidate next components (cosine similarity)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv, BatchNorm


# ── Type-conditioned projection ───────────────────────────────────────────────

class TypedLinear(nn.Module):
    """Per-node-type Linear transform, selected by node_type (dispatched via masking)."""

    def __init__(self, in_dim: int, out_dim: int, n_types: int = 8):
        super().__init__()
        self.n_types = n_types
        self.layers = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(n_types)])

    def forward(self, x: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        out = x.new_zeros(x.size(0), self.layers[0].out_features)
        for t in range(self.n_types):
            mask = node_type == t
            if mask.any():
                out[mask] = self.layers[t](x[mask])
        return out


# ── Shared encoder ────────────────────────────────────────────────────────────

class AssemblyGNN(nn.Module):
    """
    3-layer Relational GAT encoder with type-aware message passing.

    Node type (from the type one-hot in x) gets its own input/output projection.
    Edge type (from the joint-type one-hot in edge_attr) gets its own attention
    weights at every layer via RGATConv.

    Layer 1: in_dim  → hidden * heads[0]   (edge features injected here)
    Layer 2: hidden*h[0] → hidden * heads[1]
    Layer 3: hidden*h[1] → out_dim          (single head, no concat)
    """

    def __init__(
        self,
        in_dim:  int   = 21,
        out_dim: int   = 64,
        hidden:  int   = 128,
        heads:   list  = None,
        dropout: float = 0.2,
        edge_dim: int  = 2,
        n_node_types: int = 8,   # len(COMP_TYPES) in dataset.py
        n_edge_types: int = 4,   # joint-type one-hot width in dataset.py
    ):
        super().__init__()
        heads = heads or [4, 2, 1]
        self.dropout = dropout
        self.n_node_types = n_node_types
        self.n_edge_types = n_edge_types

        self.type_in  = TypedLinear(in_dim, in_dim, n_node_types)
        self.type_out = TypedLinear(out_dim, out_dim, n_node_types)

        self.conv1 = RGATConv(in_dim,              hidden,  num_relations=n_edge_types,
                              heads=heads[0],      edge_dim=edge_dim, dropout=dropout)
        self.bn1   = BatchNorm(hidden * heads[0])

        self.conv2 = RGATConv(hidden * heads[0],   hidden,  num_relations=n_edge_types,
                              heads=heads[1],      dropout=dropout)
        self.bn2   = BatchNorm(hidden * heads[1])

        self.conv3 = RGATConv(hidden * heads[1],   out_dim, num_relations=n_edge_types,
                              heads=heads[2],      concat=False, dropout=dropout)

    def _edge_type(self, edge_index: torch.Tensor, edge_attr: torch.Tensor | None) -> torch.Tensor:
        if edge_attr is not None and edge_attr.size(0) > 0:
            return edge_attr[:, 2:2 + self.n_edge_types].argmax(dim=1)
        return torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_type = x[:, :self.n_node_types].argmax(dim=1)
        edge_type = self._edge_type(edge_index, edge_attr)

        x = self.type_in(x, node_type)

        # Layer 1 — edge features injected at this layer only
        x = self.conv1(x, edge_index, edge_type, edge_attr=edge_attr)
        x = F.elu(self.bn1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index, edge_type)
        x = F.elu(self.bn2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index, edge_type)
        x = self.type_out(x, node_type)
        return x                # (N, out_dim)


# ── Task head A: Link Prediction ──────────────────────────────────────────────

class LinkPredictor(nn.Module):
    """
    MLP that scores a candidate edge (u, v) given node embeddings.
    score(u,v) = MLP( z_u ‖ z_v )  →  scalar logit
    """

    def __init__(self, in_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        h = torch.cat([z[src], z[dst]], dim=-1)   # (E, 2*D)
        return self.mlp(h).squeeze(-1)             # (E,)


# ── Task head B: Node Ranking ─────────────────────────────────────────────────

class NodeRanker(nn.Module):
    """
    Ranks candidate components by cosine similarity to the partial-assembly
    context vector (mean-pooled node embeddings).
    """

    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(in_dim, in_dim)

    def forward(
        self,
        z_partial:    torch.Tensor,   # (N_partial, D) — embeddings of known nodes
        z_candidates: torch.Tensor,   # (N_cand, D)   — embeddings of candidate nodes
    ) -> torch.Tensor:
        ctx    = self.proj(z_partial.mean(0, keepdim=True))  # (1, D)
        scores = F.cosine_similarity(ctx, z_candidates, dim=-1)  # (N_cand,)
        return scores


# ── Convenience builder ───────────────────────────────────────────────────────

def build_model(
    in_dim:   int    = 21,
    out_dim:  int    = 64,
    hidden:   int    = 128,
    heads:    list   = None,
    dropout:  float  = 0.2,
    edge_dim: int    = 2,
    device:   torch.device | None = None,
):
    heads  = heads or [4, 2, 1]
    device = device or torch.device("cuda" if torch.cuda.is_available() else
                                    "mps"  if torch.backends.mps.is_available() else "cpu")

    gnn = AssemblyGNN(in_dim, out_dim, hidden, heads, dropout, edge_dim).to(device)
    lp  = LinkPredictor(out_dim).to(device)

    total = sum(p.numel() for p in list(gnn.parameters()) + list(lp.parameters()))
    print(f"Model built on {device}  |  total params: {total:,}")
    return gnn, lp, device


def build_ranker(out_dim: int = 64, device: torch.device | None = None) -> NodeRanker:
    device = device or torch.device("cuda" if torch.cuda.is_available() else
                                    "mps"  if torch.backends.mps.is_available() else "cpu")
    return NodeRanker(out_dim).to(device)
