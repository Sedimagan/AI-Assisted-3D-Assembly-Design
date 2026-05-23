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
        scores = torch.sigmoid(lp(z, ei_all)).cpu().numpy()
        all_true.append(y_true)
        all_score.append(scores)

    return link_metrics(np.concatenate(all_true), np.concatenate(all_score))
