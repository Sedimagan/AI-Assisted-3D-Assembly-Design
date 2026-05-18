"""
model.py — AssemblyGNN
3-layer Graph Attention Network with a shared encoder and two task heads:
  - LinkPredictor  : detects missing component connections (binary edge scoring)
  - NodeRanker     : ranks candidate next components (cosine similarity)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm


# ── Shared encoder ────────────────────────────────────────────────────────────

class AssemblyGNN(nn.Module):
    """
    3-layer GAT encoder.

    Layer 1: in_dim  → hidden * heads[0]   (edge features injected here)
    Layer 2: hidden*h[0] → hidden * heads[1]
    Layer 3: hidden*h[1] → out_dim          (single head, no concat)
    """

    def __init__(
        self,
        in_dim:  int   = 13,
        out_dim: int   = 64,
        hidden:  int   = 128,
        heads:   list  = None,
        dropout: float = 0.2,
        edge_dim: int  = 2,
    ):
        super().__init__()
        heads = heads or [4, 2, 1]
        self.dropout = dropout

        self.conv1 = GATConv(in_dim,              hidden,  heads=heads[0],
                             edge_dim=edge_dim,   dropout=dropout)
        self.bn1   = BatchNorm(hidden * heads[0])

        self.conv2 = GATConv(hidden * heads[0],   hidden,  heads=heads[1],
                             dropout=dropout)
        self.bn2   = BatchNorm(hidden * heads[1])

        self.conv3 = GATConv(hidden * heads[1],   out_dim, heads=heads[2],
                             concat=False,         dropout=dropout)

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Layer 1 — edge features injected at this layer only
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(self.bn1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.elu(self.bn2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index)
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
    in_dim:   int    = 13,
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

    gnn    = AssemblyGNN(in_dim, out_dim, hidden, heads, dropout, edge_dim).to(device)
    lp     = LinkPredictor(out_dim).to(device)
    ranker = NodeRanker(out_dim).to(device)

    total = sum(p.numel() for p in list(gnn.parameters())
                             + list(lp.parameters())
                             + list(ranker.parameters()))
    print(f"Model built on {device}  |  total params: {total:,}")
    return gnn, lp, ranker, device
