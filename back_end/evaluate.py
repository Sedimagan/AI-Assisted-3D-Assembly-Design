"""
evaluate.py — Evaluation metrics
AUC-ROC · Average Precision for missing component detection (Phase 1).
Hit@K · MRR · NDCG@K for next-component ranking (Phase 2, NodeRanker).
"""

from __future__ import annotations

from collections import Counter
from typing import List, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score


def link_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """AUC-ROC and Average Precision for binary edge labels."""
    if len(np.unique(y_true)) < 2:
        return {"auc": 0.0, "ap": 0.0}
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ap":  float(average_precision_score(y_true, y_score)),
    }


@torch.no_grad()
def evaluate(gnn, lp, loader, device) -> dict:
    """
    Full evaluation pass — returns AUC-ROC and Average Precision.
    """
    gnn.eval(); lp.eval()
    all_true, all_score = [], []

    for batch in loader:
        batch  = batch.to(device)
        z      = gnn(batch.x, batch.edge_index, batch.edge_attr)
        pos_ei = batch.edge_label_index[:, batch.edge_label == 1]
        neg_ei = batch.edge_label_index[:, batch.edge_label == 0]

        # Fallback negative sampling if none were allocated by PyG due to small/dense graphs
        if neg_ei.size(1) == 0 and pos_ei.size(1) > 0:
            import random
            num_nodes = batch.x.size(0)
            existing = set(zip(batch.edge_index[0].tolist(), batch.edge_index[1].tolist()))
            existing.update(zip(pos_ei[0].tolist(), pos_ei[1].tolist()))
            cands = []
            for u in range(num_nodes):
                for v in range(num_nodes):
                    if u != v and (u, v) not in existing and (v, u) not in existing:
                        cands.append([u, v])
            if cands:
                sampled = random.sample(cands, min(len(cands), pos_ei.size(1)))
                neg_ei = torch.tensor(sampled, dtype=torch.long, device=device).t()

        ei_all = torch.cat([pos_ei, neg_ei], dim=1)
        y_true = torch.cat([
            torch.ones(pos_ei.size(1)),
            torch.zeros(neg_ei.size(1)),
        ]).numpy()
        pos = getattr(batch, "pos", None)
        scores = torch.sigmoid(lp(z, ei_all, pos)).cpu().numpy()
        all_true.append(y_true)
        all_score.append(scores)

    return link_metrics(np.concatenate(all_true), np.concatenate(all_score))


# ── Phase 2: next-component ranking metrics ───────────────────────────────────

def ranking_metrics(
    true_idx: Sequence[int],
    scores:   Sequence[np.ndarray],
    k_list:   Sequence[int] = (1, 5),
) -> dict:
    """
    true_idx: ground-truth COMP_TYPES index per validation sample.
    scores:   one (n_candidates,) cosine-similarity array per sample.
    Returns Hit@k for each k in k_list, MRR, and NDCG@5 aggregated over samples.
    """
    n = len(true_idx)
    if n == 0:
        return {f"hit@{k}": 0.0 for k in k_list} | {"mrr": 0.0, "ndcg@5": 0.0}

    hits = {k: 0 for k in k_list}
    reciprocal_ranks = []
    ndcgs = []

    for t, s in zip(true_idx, scores):
        s = np.asarray(s)
        order = np.argsort(-s)
        rank = int(np.where(order == t)[0][0]) + 1  # 1-indexed
        for k in k_list:
            if rank <= k:
                hits[k] += 1
        reciprocal_ranks.append(1.0 / rank)

        relevance = np.zeros(len(s))
        relevance[t] = 1.0
        k5 = min(5, len(s))
        ndcgs.append(ndcg_score(relevance.reshape(1, -1), s.reshape(1, -1), k=k5))

    result = {f"hit@{k}": hits[k] / n for k in k_list}
    result["mrr"]    = float(np.mean(reciprocal_ranks))
    result["ndcg@5"] = float(np.mean(ndcgs))
    return result


def majority_baseline_hit1(train_true_idx: Sequence[int], eval_true_idx: Sequence[int]) -> float:
    """Hit@1 if we always predict the most frequent type seen in training."""
    if not train_true_idx or not eval_true_idx:
        return 0.0
    majority = Counter(train_true_idx).most_common(1)[0][0]
    return sum(1 for t in eval_true_idx if t == majority) / len(eval_true_idx)


def per_class_ranking_metrics(
    true_idx: Sequence[int],
    scores:   Sequence[np.ndarray],
    comp_types: Sequence[str],
) -> dict:
    """
    Per-type breakdown of Hit@1, plus the predicted-vs-true type distribution —
    the diagnostic for "is the ranker actually distinguishing types, or has it
    collapsed to always guessing whichever type is most common overall
    (typically Body)?" A model with reasonable aggregate Hit@1 can still have
    collapsed onto one class if that class also happens to be the true label
    often enough — this makes that failure mode visible per-class rather than
    hidden inside one aggregate number.

    true_idx: ground-truth COMP_TYPES index per sample.
    scores:   one (n_candidates,) cosine-similarity array per sample.
    Returns {
      "per_class": {type_name: {"n": int, "hit@1": float}},
      "true_distribution":      {type_name: count},
      "predicted_distribution": {type_name: count},
    }
    """
    n_types = len(comp_types)
    n_correct  = [0] * n_types
    n_total    = [0] * n_types
    true_dist  = [0] * n_types
    pred_dist  = [0] * n_types

    for t, s in zip(true_idx, scores):
        s = np.asarray(s)
        pred = int(np.argmax(s))
        n_total[t] += 1
        true_dist[t] += 1
        pred_dist[pred] += 1
        if pred == t:
            n_correct[t] += 1

    per_class = {
        comp_types[i]: {
            "n": n_total[i],
            "hit@1": (n_correct[i] / n_total[i]) if n_total[i] > 0 else None,
        }
        for i in range(n_types)
    }
    return {
        "per_class":              per_class,
        "true_distribution":      dict(zip(comp_types, true_dist)),
        "predicted_distribution": dict(zip(comp_types, pred_dist)),
    }
