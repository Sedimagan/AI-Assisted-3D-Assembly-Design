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
from torch_geometric.nn import RGATConv, GATv2Conv, SAGEConv, GINConv, BatchNorm


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
    3-layer GNN encoder with type-aware message passing. The conv layer
    class is swappable via `encoder_type` (task 12: re-benchmark RGATConv
    against alternatives) -- default "rgat" is the unchanged, deployed
    architecture; every other codepath (train.py, train_ranker.py,
    train_shape_gen.py, infer.py) calls build_model() without specifying
    this, so behavior there is byte-for-byte identical to before this
    parameter existed.

    Node type (from the type one-hot in x) gets its own input/output
    projection regardless of encoder_type.

    Per-stage width is kept constant across all encoder_type choices
    (hidden*heads[0] after layer 1, hidden*heads[1] after layer 2, out_dim
    after layer 3) so the comparison is about the aggregation/attention
    mechanism itself, not a confound from different layer widths:

      "rgat"  (default): RGATConv, relation-typed (edge_type argmax) at
               every layer, continuous edge_attr injected at layer 1 only.
      "gatv2": GATv2Conv, continuous edge_attr injected at layer 1 only
               (via edge_dim), no relation typing -- GATv2Conv has no
               num_relations concept.
      "sage":  SAGEConv (GraphSAGE), topology + node features only, no
               edge features at any layer -- vanilla SAGEConv has no
               edge_dim param.
      "gin":   GINConv (2-layer MLP per hop), topology + node features
               only, same edge-feature limitation as "sage".

    "sage"/"gin" losing edge information entirely is a real asymmetry
    against "rgat"/"gatv2", not an oversight -- part of what this
    benchmark is meant to surface is whether the joint-type/contact-area
    edge features (added in R35/R36) are actually load-bearing, and
    dropping them for two of the four candidates is the cleanest way to
    see that.
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
        encoder_type: str = "rgat",
    ):
        super().__init__()
        heads = heads or [4, 2, 1]
        self.dropout = dropout
        self.n_node_types = n_node_types
        self.n_edge_types = n_edge_types
        self.encoder_type = encoder_type

        self.type_in  = TypedLinear(in_dim, in_dim, n_node_types)
        self.type_out = TypedLinear(out_dim, out_dim, n_node_types)

        w1, w2 = hidden * heads[0], hidden * heads[1]

        if encoder_type == "rgat":
            self.conv1 = RGATConv(in_dim, hidden, num_relations=n_edge_types,
                                  heads=heads[0], edge_dim=edge_dim, dropout=dropout)
            self.conv2 = RGATConv(w1,     hidden, num_relations=n_edge_types,
                                  heads=heads[1], dropout=dropout)
            self.conv3 = RGATConv(w2,     out_dim, num_relations=n_edge_types,
                                  heads=heads[2], concat=False, dropout=dropout)
        elif encoder_type == "gatv2":
            self.conv1 = GATv2Conv(in_dim, hidden, heads=heads[0],
                                   edge_dim=edge_dim, dropout=dropout, add_self_loops=False)
            self.conv2 = GATv2Conv(w1,     hidden, heads=heads[1],
                                   dropout=dropout, add_self_loops=False)
            self.conv3 = GATv2Conv(w2,     out_dim, heads=heads[2],
                                   concat=False, dropout=dropout, add_self_loops=False)
        elif encoder_type == "sage":
            self.conv1 = SAGEConv(in_dim, w1)
            self.conv2 = SAGEConv(w1,     w2)
            self.conv3 = SAGEConv(w2,     out_dim)
        elif encoder_type == "gin":
            self.conv1 = GINConv(nn.Sequential(nn.Linear(in_dim, w1), nn.ReLU(), nn.Linear(w1, w1)))
            self.conv2 = GINConv(nn.Sequential(nn.Linear(w1, w2),     nn.ReLU(), nn.Linear(w2, w2)))
            self.conv3 = GINConv(nn.Sequential(nn.Linear(w2, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type!r} "
                              f"(expected one of: rgat, gatv2, sage, gin)")

        self.bn1 = BatchNorm(w1)
        self.bn2 = BatchNorm(w2)

    def _edge_type(self, edge_index: torch.Tensor, edge_attr: torch.Tensor | None) -> torch.Tensor:
        if edge_attr is not None and edge_attr.size(0) > 0:
            return edge_attr[:, 2:2 + self.n_edge_types].argmax(dim=1)
        return torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)

    def _conv1(self, x, edge_index, edge_type, edge_attr):
        if self.encoder_type == "rgat":
            return self.conv1(x, edge_index, edge_type, edge_attr=edge_attr)
        if self.encoder_type == "gatv2":
            return self.conv1(x, edge_index, edge_attr=edge_attr)
        return self.conv1(x, edge_index)  # sage, gin: no edge features

    def _conv_next(self, conv, x, edge_index, edge_type):
        if self.encoder_type == "rgat":
            return conv(x, edge_index, edge_type)
        return conv(x, edge_index)  # gatv2 (layers 2-3), sage, gin: no edge features

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_type = x[:, :self.n_node_types].argmax(dim=1)
        edge_type = self._edge_type(edge_index, edge_attr)

        x = self.type_in(x, node_type)

        # Layer 1 — edge features injected at this layer only (where the
        # encoder_type supports them at all — see class docstring)
        x = self._conv1(x, edge_index, edge_type, edge_attr)
        x = F.elu(self.bn1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self._conv_next(self.conv2, x, edge_index, edge_type)
        x = F.elu(self.bn2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3
        x = self._conv_next(self.conv3, x, edge_index, edge_type)
        x = self.type_out(x, node_type)
        return x                # (N, out_dim)


# ── Task head A: Link Prediction ──────────────────────────────────────────────

class LinkPredictor(nn.Module):
    """
    MLP that scores a candidate edge (u, v) given node embeddings.
    score(u,v) = MLP( z_u ‖ z_v ‖ dist(u,v) )  →  scalar logit

    dist(u,v) is the Euclidean distance between the two bodies' AABB
    centers (dataset.py's `pos`, already scale-normalised per assembly),
    appended as one extra scalar. Before this, the model had zero spatial/
    proximity signal anywhere — node features are all affine-invariant
    shape ratios, and message passing only ever sees REAL edges, so a
    candidate (non-edge) pair being scored had no way to express "these two
    bodies are nowhere near each other." `pos=None` keeps the old
    embeddings-only behaviour (zero-filled distance) for callers that don't
    have positions (e.g. synthetic fallback graphs, or a checkpoint
    predating this change being loaded into a stale caller — the shape
    itself won't retroactively match, but the forward pass degrades to
    "no distance info" rather than crashing on a None).
    """

    def __init__(self, in_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor,
                pos: torch.Tensor | None = None) -> torch.Tensor:
        src, dst = edge_index
        if pos is not None:
            dist = (pos[src] - pos[dst]).norm(dim=-1, keepdim=True)  # (E, 1)
        else:
            dist = z.new_zeros(src.size(0), 1)
        h = torch.cat([z[src], z[dst], dist], dim=-1)   # (E, 2*D + 1)
        return self.mlp(h).squeeze(-1)                   # (E,)


# ── Task head B: Node Ranking ─────────────────────────────────────────────────

class NodeRanker(nn.Module):
    """
    Ranks candidate components by cosine similarity to the partial-assembly
    context vector (pooled node embeddings).

    logit_scale is a learnable temperature (CLIP-style: scores = cosine_sim *
    exp(logit_scale)) — plain cosine similarity is bounded in [-1, 1], so raw
    score differences fed into the BPR loss are small and the logsigmoid
    gradient sits in a fairly flat region for a head this thin (~4.2K params
    in `proj` alone). Initialised to exp(0)=1.0, i.e. identical to the old
    unscaled behaviour at the start of training — it only starts sharpening
    or softening the ranking once gradients say to.

    Task 16 — `self.proj` is now applied to BOTH sides of the cosine
    comparison (context AND candidates/prototypes), not just the context.
    Previously the prototype table sat exactly where the frozen encoder put
    it while only the context vector was free to move via `proj` — an
    asymmetric comparison space where cosine similarity was measuring
    "does the learned context land near the encoder's raw, untouched
    prototype," not a jointly-learned metric space. Reusing the same
    `proj` for both sides keeps this a single tiny linear map (no new
    params) while letting training actually shape where prototypes sit
    relative to context, not just the reverse.

    Task 30 — optional `anchor_weight` lets pooling emphasize specific
    nodes instead of a flat mean over every remaining node. During
    leave-one-out training this is set from the removed node's actual
    former neighbours (see train_ranker.py's remove_node) — the intuition
    being "what usually sits next to a bolt hole" is a much sharper signal
    than "the average of this whole assembly," which is what plain
    mean-pooling was diluting the majority class (Body, by far the largest
    and most numerous type) into. Defaults to uniform (identical to the
    old flat mean) when not supplied, which is what every current
    inference call site still does — see the module-level note in
    train_ranker.py for the train/inference distribution-mismatch this
    leaves open.
    """

    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(in_dim, in_dim)
        self.logit_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        z_partial:     torch.Tensor,           # (N_partial, D) — embeddings of known nodes
        z_candidates:  torch.Tensor,            # (N_cand, D)   — embeddings of candidate nodes
        anchor_weight: torch.Tensor | None = None,  # (N_partial,) optional pooling weights
    ) -> torch.Tensor:
        if anchor_weight is None:
            pooled = z_partial.mean(0, keepdim=True)              # (1, D) -- old behaviour
        else:
            w = anchor_weight.to(z_partial.device).unsqueeze(-1)  # (N_partial, 1)
            pooled = (z_partial * w).sum(0, keepdim=True) / w.sum().clamp(min=1e-9)
        ctx    = self.proj(pooled)             # (1, D)
        protos = self.proj(z_candidates)       # (N_cand, D) -- task 16
        cos    = F.cosine_similarity(ctx, protos, dim=-1)  # (N_cand,)
        scores = cos * self.logit_scale.exp()
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
    encoder_type: str = "rgat",
):
    heads  = heads or [4, 2, 1]
    device = device or torch.device("cuda" if torch.cuda.is_available() else
                                    "mps"  if torch.backends.mps.is_available() else "cpu")

    gnn = AssemblyGNN(in_dim, out_dim, hidden, heads, dropout, edge_dim,
                      encoder_type=encoder_type).to(device)
    lp  = LinkPredictor(out_dim).to(device)

    total = sum(p.numel() for p in list(gnn.parameters()) + list(lp.parameters()))
    print(f"Model built on {device}  |  total params: {total:,}")
    return gnn, lp, device


def build_ranker(out_dim: int = 64, device: torch.device | None = None) -> NodeRanker:
    device = device or torch.device("cuda" if torch.cuda.is_available() else
                                    "mps"  if torch.backends.mps.is_available() else "cpu")
    return NodeRanker(out_dim).to(device)
