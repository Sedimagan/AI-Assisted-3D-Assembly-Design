"""Generates docs/phase2_fold_variance.png — per-fold AUC for R29/R30/R31,
showing fold-to-fold variance driven by small/heterogeneous corpus size."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

folds = [1, 2, 3, 4, 5]
r29 = [0.4956, 0.7711, 0.5669, 0.5334, 0.4615]
r30 = [0.4949, 0.5254, 0.6461, 0.7733, 0.5574]
r31 = [0.5953, 0.6382, 0.5494, 0.6099, 0.5770]

fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
fig.patch.set_facecolor("#0a121f")
ax.set_facecolor("#0a121f")

ax.plot(folds, r29, marker="o", color="#8aa0b8", linewidth=1.6, label="R29 (248g, homog. GAT)")
ax.plot(folds, r30, marker="o", color="#5b9bd5", linewidth=2.0, label="R30 (248g, RGATConv — current serving)")
ax.plot(folds, r31, marker="o", color="#8b5cf6", linewidth=1.6, label="R31 (484g, RGATConv — not promoted)")

ax.set_xticks(folds)
ax.set_xlabel("CV Fold", color="#8aa0b8", fontsize=10)
ax.set_ylabel("Test AUC-ROC", color="#8aa0b8", fontsize=10)
ax.set_ylim(0.4, 0.85)
ax.tick_params(colors="#8aa0b8")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#24405f")
ax.grid(axis="y", color="#24405f", linewidth=0.5, alpha=0.5)
ax.legend(facecolor="#101c30", edgecolor="#24405f", labelcolor="#e6edf5", loc="upper right", fontsize=8.5)
ax.set_title("Fold-to-fold AUC variance — small, heterogeneous corpus, not a hard architecture ceiling",
             color="#e6edf5", fontsize=10.5, pad=12)

plt.tight_layout()
plt.savefig("docs/phase2_fold_variance.png", facecolor=fig.get_facecolor())
print("Saved docs/phase2_fold_variance.png")
