"""
evaluate.py — Evaluation metrics  (Phase 1)
AUC-ROC · Average Precision for missing component detection.

Phase 2 (planned): Hit@K · MRR · NDCG@K for next-component ranking.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


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
        ei_all = torch.cat([pos_ei, neg_ei], dim=1)
        y_true = torch.cat([
            torch.ones(pos_ei.size(1)),
            torch.zeros(neg_ei.size(1)),
        ]).numpy()
        scores = torch.sigmoid(lp(z, ei_all)).cpu().numpy()
        all_true.append(y_true)
        all_score.append(scores)

    return link_metrics(np.concatenate(all_true), np.concatenate(all_score))
