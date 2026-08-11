"""
train_ranker.py — NodeRanker training  (Phase 2)
Trains the NodeRanker head to rank candidate next-component types by cosine
similarity to a partial-assembly context vector, using a BPR ranking loss.

Reuses the already-trained AssemblyGNN encoder from a Phase-1 serving
checkpoint (frozen — only NodeRanker's small projection layer is trained),
so this is a lightweight one-time pass, not a full multi-hour retrain.

Run:
    cd back_end && python train_ranker.py
    python train_ranker.py --config config.yaml --serving-ckpt checkpoints/best_serving.pt \
        --fold-idx 0 --epochs 30 --n-per-graph 5 --lr 1e-3

Note: node_ranker.pt is fit to the *current* best_serving.pt's latent space.
If Phase 1 is retrained/re-promoted, re-run this script.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch_geometric.data import Data

from dataset  import AssemblyDataset, graph_level_indices, COMP_TYPES
from model    import build_model, build_ranker
from evaluate import (ranking_metrics, majority_baseline_hit1, per_class_ranking_metrics,
                       corpus_majority_baseline_hit1, corpus_majority_type)


# ── Type prototypes ────────────────────────────────────────────────────────────

def compute_type_prototypes(graphs: list) -> torch.Tensor:
    """Mean 22-dim raw feature vector per COMP_TYPE, over the given graphs."""
    n_types = len(COMP_TYPES)
    sums   = torch.zeros(n_types, graphs[0].x.size(1))
    counts = torch.zeros(n_types)
    for g in graphs:
        type_idx = g.x[:, :n_types].argmax(dim=1)
        for t in range(n_types):
            mask = type_idx == t
            if mask.any():
                sums[t]   += g.x[mask].sum(dim=0)
                counts[t] += mask.sum()
    counts = counts.clamp(min=1)
    return sums / counts.unsqueeze(1)


@torch.no_grad()
def embed_prototypes(gnn, prototypes_raw: torch.Tensor, device) -> torch.Tensor:
    """Pass the (8,22) raw prototype features through the frozen encoder as an
    isolated zero-edge batch, returning (8, out_dim) embeddings."""
    x = prototypes_raw.to(device)
    edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
    edge_attr  = torch.zeros(0, 6, dtype=torch.float, device=device)
    return gnn(x, edge_index, edge_attr)


# ── Leave-one-node-out sampling ────────────────────────────────────────────────

def remove_node(data: Data, idx: int) -> Data:
    """Return a copy of data with node idx (and its incident edges) removed."""
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

    return Data(
        x=data.x[keep],
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
    )


def sample_leave_one_out(graphs: list, n_per_graph: int, rng: random.Random):
    """For each graph with >=2 nodes, sample up to n_per_graph node indices
    and return [(partial_graph, true_type_idx), ...]."""
    samples = []
    for g in graphs:
        n = g.num_nodes
        if n < 2:
            continue
        idxs = list(range(n))
        rng.shuffle(idxs)
        for i in idxs[:min(n_per_graph, n)]:
            true_type = int(g.x[i, :len(COMP_TYPES)].argmax().item())
            samples.append((remove_node(g, i), true_type))
    return samples


# ── Epoch pass ─────────────────────────────────────────────────────────────────

def compute_class_weights(samples, n_types: int, cap: float = 5.0) -> torch.Tensor:
    """
    Inverse-frequency class weight per COMP_TYPES index, computed from the
    REALIZED leave-one-out sample distribution (not the raw corpus) since
    that's what's actually driving gradients -- sklearn's 'balanced' formula:
    weight[c] = n_samples / (n_types * count[c]), so a perfectly balanced
    class gets weight 1.0, an over-represented class (Body, ~40% of samples)
    gets pulled below 1.0, and an under-represented class (Washer, Short
    Shaft) gets pulled above 1.0. Capped to avoid one near-absent class (a
    handful of leave-one-out draws) blowing up into a wildly dominant
    gradient -- same "capped inverse-frequency" pattern already used for
    Phase 1's per-category CATEGORY_WEIGHTS in train.py.
    """
    counts = torch.zeros(n_types)
    for _, t in samples:
        counts[t] += 1
    n = max(1, len(samples))
    counts = counts.clamp(min=1)  # a class with 0 samples just gets the cap, not div-by-zero
    weights = n / (n_types * counts)
    return weights.clamp(max=cap)


def epoch_pass(nr, gnn, prototypes_raw, samples, device, opt=None, return_raw=False,
                class_weights: torch.Tensor | None = None):
    """One pass over `samples`. Trains nr if opt is given, else runs eval-only
    and returns ranking_metrics() -- or, if return_raw, a
    (metrics_dict, true_idx_all, scores_all) tuple so callers can compute
    additional breakdowns (e.g. per_class_ranking_metrics) without a second
    pass over the encoder. gnn is always frozen (no_grad context).

    class_weights, if given, scales each sample's BPR loss by its true
    type's inverse-frequency weight (see compute_class_weights) -- without
    this, samples of the majority type dominate the gradient simply by
    outnumbering everything else, which is what caused the majority-class
    collapse confirmed via per_class_ranking_metrics (body predicted 94.3%
    of the time vs. its true 42.9% frequency).

    Prototype embeddings depend only on the frozen encoder + the fixed
    prototype table, so they're computed once per pass, not per sample."""
    is_train = opt is not None
    nr.train(is_train)

    protos = embed_prototypes(gnn, prototypes_raw, device)

    total_loss = 0.0
    true_idx_all, scores_all = [], []

    for partial, true_type in samples:
        with torch.no_grad():
            z = gnn(partial.x.to(device), partial.edge_index.to(device),
                    partial.edge_attr.to(device) if partial.edge_attr is not None else None)
            if z.size(0) == 0:
                continue

        if is_train:
            opt.zero_grad()
            scores = nr(z, protos)  # (8,) cosine similarities, uses nr.proj internally
            pos = scores[true_type]
            neg_mask = torch.ones(len(COMP_TYPES), dtype=torch.bool)
            neg_mask[true_type] = False
            negs = scores[neg_mask]
            loss = -F.logsigmoid(pos - negs).sum()
            if class_weights is not None:
                loss = loss * class_weights[true_type]
            loss.backward()
            opt.step()
            total_loss += loss.item()
        else:
            with torch.no_grad():
                scores = nr(z, protos)
            true_idx_all.append(true_type)
            scores_all.append(scores.cpu().numpy())

    if is_train:
        return total_loss / max(1, len(samples))
    metrics = ranking_metrics(true_idx_all, scores_all)
    if return_raw:
        return metrics, true_idx_all, scores_all
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--serving-ckpt", default="checkpoints/best_serving.pt")
    parser.add_argument("--fold-idx",     type=int, default=0)
    parser.add_argument("--epochs",       type=int, default=None)
    parser.add_argument("--n-per-graph",  type=int, default=None)
    parser.add_argument("--lr",           type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rc = cfg.get("ranker", {})
    epochs      = args.epochs      or rc.get("epochs", 30)
    n_per_graph = args.n_per_graph or rc.get("n_per_graph", 5)
    lr          = args.lr          or rc.get("lr", 1.0e-3)
    weight_decay = rc.get("weight_decay", 1.0e-5)
    fold_idx    = args.fold_idx if args.fold_idx is not None else rc.get("fold_idx", 0)

    print("\n[1/5] Loading dataset …")
    dataset = AssemblyDataset(
        source_dir    = cfg["data"]["source_dir"],
        processed_dir = cfg["data"]["processed_dir"],
        categories    = cfg["data"].get("categories") or None,
    )
    print(f"      {len(dataset)} graphs loaded.")

    train_idx, val_idx, test_idx = graph_level_indices(dataset, cfg, fold_idx=fold_idx)
    train_graphs = [dataset[i] for i in train_idx]
    val_graphs   = [dataset[i] for i in val_idx]
    test_graphs  = [dataset[i] for i in test_idx]
    print(f"      train: {len(train_graphs)}  val: {len(val_graphs)}  test: {len(test_graphs)}")

    print("\n[2/5] Loading frozen Phase-1 encoder …")
    ckpt_path = Path(args.serving_ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = ckpt["cfg"]["model"]
    gnn, lp, device = build_model(
        in_dim=mc["in_dim"], out_dim=mc["out_dim"], hidden=mc["hidden_dim"],
        heads=mc["heads"], dropout=mc["dropout"], edge_dim=mc["edge_dim"],
        encoder_type=mc.get("encoder_type", "rgat"),
    )
    gnn.load_state_dict(ckpt["gnn"])
    gnn.eval()
    for p in gnn.parameters():
        p.requires_grad_(False)
    print(f"      Encoder frozen (val AUC={ckpt['auc']:.4f}, trained_at={ckpt.get('trained_at')})")

    print("\n[3/5] Computing type prototypes …")
    prototypes_raw = compute_type_prototypes(train_graphs + val_graphs)
    print(f"      Prototype table: {tuple(prototypes_raw.shape)}")

    print("\n[4/5] Building NodeRanker …")
    nr = build_ranker(out_dim=mc["out_dim"], device=device)
    opt = torch.optim.Adam(nr.parameters(), lr=lr, weight_decay=weight_decay)

    rng = random.Random(42)
    train_samples = sample_leave_one_out(train_graphs, n_per_graph, rng)
    val_samples   = sample_leave_one_out(val_graphs,   n_per_graph, rng)
    test_samples  = sample_leave_one_out(test_graphs,  n_per_graph, rng)
    train_true_idx = [t for _, t in train_samples]
    print(f"      Leave-one-out samples — train: {len(train_samples)}  "
          f"val: {len(val_samples)}  test: {len(test_samples)}")

    class_weights = compute_class_weights(train_samples, len(COMP_TYPES))
    print(f"      Class weights (inverse-frequency, capped at 5.0): "
          + ", ".join(f"{t}={w:.2f}" for t, w in zip(COMP_TYPES, class_weights.tolist())))

    print(f"\n[5/5] Training ({epochs} epochs) …")
    best_mrr = -1.0
    best_state = None
    best_metrics = None

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(cfg["paths"]["results"])
    res_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        rng.shuffle(train_samples)
        train_loss = epoch_pass(nr, gnn, prototypes_raw, train_samples, device, opt=opt,
                                 class_weights=class_weights)
        val_metrics = epoch_pass(nr, gnn, prototypes_raw, val_samples, device, opt=None)
        print(f"  Ep {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
              f"Hit@1={val_metrics['hit@1']:.4f}  Hit@5={val_metrics['hit@5']:.4f}  "
              f"MRR={val_metrics['mrr']:.4f}  NDCG@5={val_metrics['ndcg@5']:.4f}")
        if val_metrics["mrr"] > best_mrr:
            best_mrr = val_metrics["mrr"]
            best_state = {k: v.clone() for k, v in nr.state_dict().items()}
            best_metrics = val_metrics

    nr.load_state_dict(best_state)
    test_metrics, test_true_idx, test_scores = epoch_pass(
        nr, gnn, prototypes_raw, test_samples, device, opt=None, return_raw=True)
    # Two baselines, deliberately both reported: majority_baseline_hit1 is
    # computed from the leave-one-out SAMPLE pool (capped at n_per_graph per
    # graph) -- kept for continuity with every prior run's saved numbers, but
    # it moves whenever n_per_graph/seed/corpus changes, undermining "did we
    # beat baseline" comparisons across runs. corpus_majority_baseline_hit1
    # uses the TRUE corpus-wide node-type frequency instead -- stable, and
    # the one that should actually be compared run-to-run.
    baseline_hit1 = majority_baseline_hit1(train_true_idx, [t for _, t in test_samples])
    corpus_baseline_hit1 = corpus_majority_baseline_hit1(
        train_graphs, [t for _, t in test_samples], len(COMP_TYPES))
    corpus_majority = COMP_TYPES[corpus_majority_type(train_graphs, len(COMP_TYPES))]
    per_class = per_class_ranking_metrics(test_true_idx, test_scores, COMP_TYPES)

    print(f"\n{'='*55}")
    print("  NodeRanker — Test Results")
    print(f"{'='*55}")
    for k, v in test_metrics.items():
        print(f"     {k:10s}: {v:.4f}")
    print(f"     majority-class baseline Hit@1 (sample-pool, legacy):  {baseline_hit1:.4f}")
    print(f"     majority-class baseline Hit@1 (corpus-wide, stable):  {corpus_baseline_hit1:.4f}  "
          f"(majority type: {corpus_majority})")
    print(f"  (best val MRR={best_mrr:.4f} at selection time)")

    print(f"\n  Per-class breakdown (test set, n={len(test_true_idx)}):")
    print(f"  {'type':12s} {'n':>4s} {'hit@1':>7s} {'true%':>7s} {'pred%':>7s}")
    n_test = max(1, len(test_true_idx))
    for t in COMP_TYPES:
        c = per_class["per_class"][t]
        true_pct = 100 * per_class["true_distribution"][t] / n_test
        pred_pct = 100 * per_class["predicted_distribution"][t] / n_test
        hit1_str = f"{c['hit@1']:.3f}" if c["hit@1"] is not None else "  n/a"
        print(f"  {t:12s} {c['n']:>4d} {hit1_str:>7s} {true_pct:>6.1f}% {pred_pct:>6.1f}%")
    print("  (true% vs pred% far apart for a type = the model is over/under-"
          "predicting it relative to how often it's actually the answer — "
          "a sign of majority-class collapse if e.g. 'body' pred% >> true%)")

    with open(res_dir / "ranker_test_metrics.json", "w") as f:
        json.dump({
            "test_metrics":  {k: round(v, 4) for k, v in test_metrics.items()},
            "val_metrics":   {k: round(v, 4) for k, v in best_metrics.items()},
            "baseline_hit1":                round(baseline_hit1, 4),
            "baseline_hit1_corpus_wide":     round(corpus_baseline_hit1, 4),
            "corpus_majority_type":          corpus_majority,
            "per_class":     per_class,
        }, f, indent=2)

    torch.save({
        "epoch":              epochs,
        "val_metrics":        best_metrics,
        "test_metrics":       test_metrics,
        "baseline_hit1":      baseline_hit1,
        "baseline_hit1_corpus_wide": corpus_baseline_hit1,
        "trained_at":         datetime.now().isoformat(),
        "encoder_ckpt":       str(ckpt_path),
        "encoder_trained_at": ckpt.get("trained_at"),
        "encoder_auc":        ckpt["auc"],
        "nr":                 nr.state_dict(),
        "type_prototypes_raw": prototypes_raw,
        "comp_types":         COMP_TYPES,
        "cfg":                ckpt["cfg"],
    }, ckpt_dir / "node_ranker.pt")

    print(f"\n  Saved → {ckpt_dir / 'node_ranker.pt'}")
    print(f"  Saved → {res_dir / 'ranker_test_metrics.json'}")


if __name__ == "__main__":
    main()
