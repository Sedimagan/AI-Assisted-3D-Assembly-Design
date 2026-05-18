"""
evaluate.py — Evaluation metrics
AUC-ROC, Average Precision, Hit@K, MRR, NDCG@K
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


# ── Link prediction metrics ───────────────────────────────────────────────────

def link_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    """AUC-ROC and Average Precision for binary edge labels."""
    if len(np.unique(y_true)) < 2:
        return {"auc": 0.0, "ap": 0.0}
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ap":  float(average_precision_score(y_true, y_score)),
    }


# ── Ranking metrics ───────────────────────────────────────────────────────────

def hit_at_k(ranked_labels: np.ndarray, k: int) -> float:
    """1 if the ground-truth positive appears in the top-k, else 0."""
    return float(1 in ranked_labels[:k])


def mrr(ranked_labels: np.ndarray) -> float:
    """Mean Reciprocal Rank for a single query."""
    hits = np.where(ranked_labels == 1)[0]
    return float(1.0 / (hits[0] + 1)) if len(hits) else 0.0


def ndcg_at_k(ranked_labels: np.ndarray, k: int) -> float:
    """Normalised Discounted Cumulative Gain @ K."""
    gains = ranked_labels[:k] / np.log2(np.arange(2, k + 2))
    dcg   = gains.sum()
    ideal = float(1 / np.log2(2))         # best possible: positive at rank 1
    return float(dcg / ideal) if ideal else 0.0


def ranking_metrics(
    scores:     torch.Tensor,
    pos_mask:   torch.Tensor,
    top_k: list = None,
) -> dict:
    """
    Given per-candidate scores and a boolean mask of true positives,
    return Hit@K, MRR, NDCG@K averaged over all queries.

    Parameters
    ----------
    scores   : (N,) float tensor — model scores for each candidate
    pos_mask : (N,) bool tensor  — True for ground-truth positive candidates
    top_k    : list of K values to evaluate
    """
    top_k  = top_k or [1, 3, 5, 10]
    order  = scores.argsort(descending=True).cpu().numpy()
    labels = pos_mask.cpu().numpy().astype(int)[order]

    result = {"mrr": mrr(labels)}
    for k in top_k:
        result[f"hit@{k}"]  = hit_at_k(labels, k)
        result[f"ndcg@{k}"] = ndcg_at_k(labels, k)
    return result


# ── Full evaluation pass ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(gnn, lp, loader, device, top_k=None):
    """
    Run a full evaluation pass over a DataLoader.
    Returns a dict of averaged metrics.
    """
    top_k = top_k or [1, 3, 5, 10]
    gnn.eval(); lp.eval()

    all_true, all_score = [], []
    hit_sums  = {k: 0.0 for k in top_k}
    mrr_sum   = 0.0
    ndcg_sums = {k: 0.0 for k in top_k}
    n_batch   = 0

    for batch in loader:
        batch = batch.to(device)
        z     = gnn(batch.x, batch.edge_index, batch.edge_attr)

        # Link prediction
        pos_ei  = batch.edge_label_index[:, batch.edge_label == 1]
        neg_ei  = batch.edge_label_index[:, batch.edge_label == 0]
        ei_all  = torch.cat([pos_ei, neg_ei], dim=1)
        y_true  = torch.cat([
            torch.ones(pos_ei.size(1)),
            torch.zeros(neg_ei.size(1)),
        ]).numpy()
        scores  = torch.sigmoid(lp(z, ei_all)).cpu().numpy()
        all_true.append(y_true); all_score.append(scores)

        # Ranking: treat each batch graph as one query
        rm      = ranking_metrics(
            torch.tensor(scores),
            torch.tensor(y_true, dtype=torch.bool),
            top_k,
        )
        mrr_sum += rm["mrr"]
        for k in top_k:
            hit_sums[k]  += rm[f"hit@{k}"]
            ndcg_sums[k] += rm[f"ndcg@{k}"]
        n_batch += 1

    y_true_all  = np.concatenate(all_true)
    y_score_all = np.concatenate(all_score)
    lm          = link_metrics(y_true_all, y_score_all)

    result = {**lm, "mrr": mrr_sum / max(n_batch, 1)}
    for k in top_k:
        result[f"hit@{k}"]  = hit_sums[k]  / max(n_batch, 1)
        result[f"ndcg@{k}"] = ndcg_sums[k] / max(n_batch, 1)
    return result
