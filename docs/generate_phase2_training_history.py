"""Generates docs/phase2_training_history.png — R29/R30/R31 mean AUC & AP trend
for the Phase 2 First Review slide deck. One-off chart generator, run manually."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

runs = ["R29\n(248g, homog. GAT)", "R30\n(248g, RGATConv+TypedLinear)", "R31\n(484g, RGATConv+TypedLinear)"]
mean_auc = [0.566, 0.599, 0.594]
std_auc  = [0.121, 0.112, 0.0335]
mean_ap  = [0.843, 0.869, 0.7617]
std_ap   = [0.055, 0.047, 0.0323]

fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
fig.patch.set_facecolor("#0a121f")
ax.set_facecolor("#0a121f")

x = np.arange(len(runs))
w = 0.32

b1 = ax.bar(x - w/2, mean_auc, w, yerr=std_auc, capsize=4, label="Mean AUC-ROC",
            color="#5b9bd5", edgecolor="none")
b2 = ax.bar(x + w/2, mean_ap, w, yerr=std_ap, capsize=4, label="Mean AP",
            color="#8b5cf6", edgecolor="none")

ax.axhline(0.85, color="#f0a83c", linestyle="--", linewidth=1.2, alpha=0.8)
ax.text(len(runs) - 0.55, 0.858, "Phase 1 AUC target (0.85)", color="#f0a83c", fontsize=8.5)

for bars, vals in [(b1, mean_auc), (b2, mean_ap)]:
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.03,
                 f"{v:.3f}", ha="center", va="bottom", color="#e6edf5", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(runs, color="#cfd9e6", fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score", color="#8aa0b8", fontsize=10)
ax.tick_params(colors="#8aa0b8")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#24405f")
ax.legend(facecolor="#101c30", edgecolor="#24405f", labelcolor="#e6edf5", loc="upper left", fontsize=9)
ax.set_title("Node/edge-typed encoder (R30) beats homogeneous GAT (R29); larger corpus (R31) did not beat R30",
             color="#e6edf5", fontsize=10.5, pad=14)

plt.tight_layout()
plt.savefig("docs/phase2_training_history.png", facecolor=fig.get_facecolor())
print("Saved docs/phase2_training_history.png")
