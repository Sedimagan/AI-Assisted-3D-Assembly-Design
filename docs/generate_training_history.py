"""
Generate docs/training_history.png + docs/training_changelog.png
Run: python docs/generate_training_history.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Training run data ─────────────────────────────────────────────────────────
runs = [
    dict(id="R1",  date="23 May\n14:12", graphs="synth+real",
         note="13-dim · synthetic+real\nEarly stop ep 1\nUnderfitting on mixed corpus",
         val_auc=0.440, test_auc=0.415, test_ap=0.513, synthetic=True),
    dict(id="R3",  date="23 May\n16:24", graphs="synth+real",
         note="⚠ Inflated by 300 synthetic graphs\n13-dim · neg:1 (→ 21)\nSynthetic test leakage",
         val_auc=0.927, test_auc=0.985, test_ap=0.993, synthetic=True),
    dict(id="R6",  date="26 May\n10:42", graphs="25 real",
         note="Removed 300 synthetic graphs\n13-dim · 25 real assemblies\nHonest real-geometry baseline",
         val_auc=0.813, test_auc=0.636, test_ap=0.584, synthetic=False),
    dict(id="R7",  date="26 May\n11:13", graphs="4 real",
         note="16-dim node features introduced\nBug: getNode() 4 outputs → 11 files skipped\nOnly 4 graphs parsed — weak split",
         val_auc=0.719, test_auc=0.342, test_ap=0.427, synthetic=False),
    dict(id="R8",  date="26 May\n11:39", graphs="25 real",
         note="getNode() fix applied\n16-dim · 25 real assemblies\nExact SA + SDF mean+var + SA/V ratio",
         val_auc=0.750, test_auc=0.585, test_ap=0.559, synthetic=False),
    dict(id="R9",  date="27 May\n10:06", graphs="138 real",
         note="Hinge assemblies added\n93 STEP files → 138 graphs\nTrain 98 Val 12 Test 11\nDiverse graph = honest baseline",
         val_auc=0.623, test_auc=0.512, test_ap=0.533, synthetic=False),
    dict(id="R10", date="10 Jun\n12:00", graphs="780 real",
         note="+Fusion360 Gallery dataset\n938 STEP → 780 graphs (45 skipped)\nEarly stop ep 58 · loss 0.401\n5-min timeout per file",
         val_auc=0.585, test_auc=0.604, test_ap=0.577, synthetic=False),
    dict(id="R11", date="13 Jun\n12:53", graphs="416 real",
         note="Size filters: nodes≤20, edges≤60, 120s timeout\n753 STEP → 416 graphs\n162 nodes skip · 67 edges skip · 4 timeout\nEarly stop ep 26 (~4× faster than R10)",
         val_auc=0.781, test_auc=0.538, test_ap=0.592, synthetic=False),
    dict(id="R12", date="13 Jun\n18:04", graphs="416 real",
         note="5-Fold Cross-Validation introduced\nMean AUC=0.697±0.024 · Mean AP=0.827±0.038\nBest fold test AUC=0.663 · AP=0.782\nMore robust generalisation estimate",
         val_auc=0.659, test_auc=0.663, test_ap=0.782, synthetic=False),
    dict(id="R13", date="14 Jun\n05:00", graphs="807 real",
         note="+New STEP files · 1404 STEP → 807 graphs\n597 skipped (nodes:426 edges:136 timeout:35)\n5-Fold CV · Mean AUC=0.574±0.059\nMean AP=0.650±0.053 · Best fold val AUC=0.699",
         val_auc=0.699, test_auc=0.597, test_ap=0.683, synthetic=False),
    dict(id="R14", date="14 Jun\n10:08", graphs="270 real",
         note="Category filter: Mech.Eng + Mach.Design\n+ Automotive + Tools only\n306 STEP → 270 graphs · 0 skipped\n5-Fold CV · Mean AUC=0.624±0.036 · Mean AP=0.734±0.015",
         val_auc=0.715, test_auc=0.641, test_ap=0.738, synthetic=False),
    dict(id="R15", date="20 Jun\n10:30", graphs="270 real",
         note="P1–P6 improvements · 18-dim features\nheads [8,4,1] · hard negatives (0.3×)\naffine-invariant shape · structured synthetics\n5-Fold CV · Mean AUC=0.726±0.041 · Mean AP=0.917±0.012",
         val_auc=0.902, test_auc=0.614, test_ap=0.879, synthetic=False),
    dict(id="R16", date="20 Jun\n12:16", graphs="270 real",
         note="21-dim: kept bbox [10–12] + affine [13–17]\nSDF/SA shifted to [18–20]\n5-Fold CV · Mean AUC=0.659±0.083\nMean AP=0.894±0.031 · Fold1 outlier (0.524)",
         val_auc=0.902, test_auc=0.688, test_ap=0.897, synthetic=False),
    dict(id="R17", date="22 Jun\n08:16", graphs="1760 real",
         note="6.5× more data: 1,760 graphs (was 270)\nDeduped 144 cross-cat folders · no-contact pre-filter\nhidden_dim 128 · edges≥10 filter\n5-Fold CV · Mean AUC=0.625±0.017 · Mean AP=0.878±0.005",
         val_auc=0.712, test_auc=0.622, test_ap=0.868, synthetic=False),
    dict(id="R18", date="24 Jun\n05:08", graphs="995 real",
         note="22-dim nodes + 6-dim edges · curated 996→995\nlog1p vol/SA · holes · joint types · MIN_EDGES 10→6\nZero parse failures · dataset curation study\n5-Fold CV · Mean AUC=0.483±0.056 · Mean AP=0.833±0.025",
         val_auc=0.628, test_auc=0.444, test_ap=0.807, synthetic=False),
    dict(id="R19", date="24 Jun\n10:11", graphs="995 real",
         note="best_serving.pt promotion gate added\ndevice bug fix for --start-fold resume\nOnly better models deployed for inference\n5-Fold CV · Mean AUC=0.504±0.069 · Mean AP=0.835±0.028",
         val_auc=0.641, test_auc=0.472, test_ap=0.795, synthetic=False),
]

n   = len(runs)
x   = np.arange(n)
w   = 0.22
gap = 0.04

# ── Colours ───────────────────────────────────────────────────────────────────
C_VAL  = "#90caf9"   # light blue
C_TEST = "#1565c0"   # dark blue
C_AP   = "#42a5f5"   # mid blue (translucent via alpha)
C_SYN  = "#b0bec5"   # grey for synthetic-inflated runs
C_TGT_AUC = "#ef5350"
C_TGT_AP  = "#ffa726"
BG      = "#1a1a2e"
PANEL   = "#16213e"
TEXT    = "#e0e0e0"
SUBTEXT = "#9e9e9e"

log_colors = {
    "R1": "#90caf9", "R3": "#ffa726", "R6": "#4caf50",
    "R7": "#ef9a9a", "R8": "#80cbc4", "R9": "#ce93d8",
    "R10": "#86efac", "R11": "#fde68a", "R12": "#c084fc", "R13": "#f9a8d4", "R14": "#67e8f9",
    "R15": "#4ade80", "R16": "#a78bfa", "R17": "#f59e0b",
    "R18": "#38bdf8", "R19": "#4f8ef7",
}

# ═════════════════════════════════════════════════════════════════════════════
# IMAGE 1: Bar chart — full width, no side panel
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 8), facecolor=BG)
ax = fig.add_axes([0.04, 0.18, 0.92, 0.74], facecolor=PANEL)

# Era separator
ax.axvspan(-0.5, 1.5, alpha=0.07, color="#ffa726", label="_nolegend_")
ax.axvspan(1.5, n - 0.5, alpha=0.04, color="#42a5f5", label="_nolegend_")
ax.text(0.08, 1.02, "← Synthetic era", transform=ax.transAxes,
        ha="center", color="#ffa726", fontsize=8, style="italic",
        bbox=dict(facecolor=PANEL, edgecolor="none", alpha=0.5))
ax.text(0.30, 1.02, "Real data only →", transform=ax.transAxes,
        ha="center", color="#42a5f5", fontsize=8, style="italic",
        bbox=dict(facecolor=PANEL, edgecolor="none", alpha=0.5))

for i, r in enumerate(runs):
    col = C_SYN if r["synthetic"] else None
    c_val  = col or C_VAL
    c_test = col or C_TEST
    c_ap   = col or C_AP

    b1 = ax.bar(x[i] - w - gap/2, r["val_auc"],  w, color=c_val,  alpha=0.85, zorder=3)
    b2 = ax.bar(x[i],              r["test_auc"], w, color=c_test, alpha=0.95, zorder=3)
    b3 = ax.bar(x[i] + w + gap/2, r["test_ap"],  w, color=c_ap,   alpha=0.55, zorder=3)

    for b, v in [(b1, r["val_auc"]), (b2, r["test_auc"]), (b3, r["test_ap"])]:
        ax.text(b[0].get_x() + b[0].get_width()/2, v + 0.012,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                color=TEXT, fontweight="bold", zorder=4)

# Phase 1 targets
ax.axhline(0.85, color=C_TGT_AUC, linewidth=1.2, linestyle="--", zorder=2, alpha=0.8)
ax.axhline(0.82, color=C_TGT_AP,  linewidth=1.0, linestyle=":",  zorder=2, alpha=0.8)
ax.text(n - 0.48, 0.855, "AUC target 0.85", color=C_TGT_AUC, fontsize=8, va="bottom")
ax.text(n - 0.48, 0.825, "AP target 0.82",  color=C_TGT_AP,  fontsize=8, va="bottom")

ax.set_ylim(0, 1.12)
ax.set_xlim(-0.55, n - 0.45)
ax.set_xticks(x)
ax.set_xticklabels([f"{r['id']}\n{r['date']}" for r in runs],
                   color=TEXT, fontsize=8.5)
ax.set_ylabel("Score", color=TEXT, fontsize=10)
ax.set_title("Training Progression — AUC-ROC & Average Precision",
             color=TEXT, fontsize=14, fontweight="bold", pad=14)
ax.tick_params(colors=TEXT)
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# Legend
legend_patches = [
    mpatches.Patch(color=C_VAL,  alpha=0.85, label="Val AUC"),
    mpatches.Patch(color=C_TEST, alpha=0.95, label="Test AUC"),
    mpatches.Patch(color=C_AP,   alpha=0.55, label="Test AP"),
]
ax.legend(handles=legend_patches, loc="upper left", facecolor=PANEL,
          edgecolor="#555", labelcolor=TEXT, fontsize=9)

# Summary table below chart
col_labels = ["Run", "Date", "Node Dims", "Graphs", "Val AUC", "Test AUC", "Test AP"]
table_data = [
    [r["id"], r["date"].replace("\n", " "),
     "22+6-dim · 995g" if r["id"] in ("R18","R19") else "13-dim" if r["id"] in ("R1","R3","R6") else "21-dim · 1760g" if r["id"] == "R17" else "21-dim · bbox+affine" if r["id"] == "R16" else "18-dim · P1-P6" if r["id"] == "R15" else "16-dim (5-fold CV)" if r["id"] in ("R12","R13","R14") else "16-dim",
     r["graphs"],
     f"{r['val_auc']:.3f}", f"{r['test_auc']:.3f}", f"{r['test_ap']:.3f}"]
    for r in runs
]

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               bbox=[0, -0.38, 1, 0.32], cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor("#0f3460" if row == 0 else PANEL)
    cell.set_text_props(color=TEXT)
    cell.set_edgecolor("#444")
    if row > 0 and runs[row-1]["synthetic"]:
        cell.set_facecolor("#2a2a1a")
    if row > 0 and runs[row-1]["id"] == "R10":
        cell.set_facecolor("#0d2b0d")
        cell.set_text_props(color="#86efac", fontweight="bold")
    if row > 0 and runs[row-1]["id"] == "R11":
        cell.set_facecolor("#2b2800")
        cell.set_text_props(color="#fde68a", fontweight="bold")
    if row > 0 and runs[row-1]["id"] == "R12":
        cell.set_facecolor("#1a0b2e")
        cell.set_text_props(color="#c084fc", fontweight="bold")
    if row > 0 and runs[row-1]["id"] == "R13":
        cell.set_facecolor("#2e0f1a")
        cell.set_text_props(color="#f9a8d4", fontweight="bold")
    if row > 0 and runs[row-1]["id"] == "R14":
        cell.set_facecolor("#082030")
        cell.set_text_props(color="#67e8f9", fontweight="bold")
    if row > 0 and runs[row-1]["id"] == "R15":
        cell.set_facecolor("#0a2e14")
        cell.set_text_props(color="#4ade80", fontweight="bold")
    if row > 0 and runs[row-1]["id"] == "R16":
        cell.set_facecolor("#1a0b2e")
        cell.set_text_props(color="#a78bfa", fontweight="bold")
    if row > 0 and runs[row-1]["id"] == "R17":
        cell.set_facecolor("#2e1a00")
        cell.set_text_props(color="#f59e0b", fontweight="bold")
    if row > 0 and runs[row-1]["id"] == "R18":
        cell.set_facecolor("#0a1e30")
        cell.set_text_props(color="#38bdf8", fontweight="bold")
    if row > 0 and runs[row-1]["id"] == "R19":
        cell.set_facecolor("#0d1a3a")
        cell.set_text_props(color="#4f8ef7", fontweight="bold")

ax.text(0.5, -0.42, "* Inflated: synthetic test graphs trivially match synthetic training patterns.",
        transform=ax.transAxes, ha="center", color=SUBTEXT, fontsize=7.5, style="italic")

out1 = "docs/training_history.png"
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved: {out1}")


# ═════════════════════════════════════════════════════════════════════════════
# IMAGE 2: Change Log — separate full-width image
# ═════════════════════════════════════════════════════════════════════════════
n_runs = len(runs)
row_h = 0.58       # inches per run
fig_h = 1.2 + n_runs * row_h
fig2 = plt.figure(figsize=(18, fig_h), facecolor=BG)
ax2 = fig2.add_axes([0.02, 0.02, 0.96, 0.96], facecolor=PANEL)
ax2.axis("off")

ax2.text(0.5, 0.99, "Change Log — R1 to R19", transform=ax2.transAxes,
         ha="center", va="top", color=TEXT, fontsize=14, fontweight="bold")

# Two-column layout: runs split left/right
mid = (n_runs + 1) // 2  # 7 left, 7 right
cols = [runs[:mid], runs[mid:]]
col_x_offsets = [0.02, 0.52]

for ci, (col_runs, x_off) in enumerate(zip(cols, col_x_offsets)):
    y = 0.93
    step = 0.88 / mid
    for r in col_runs:
        c = log_colors[r["id"]]
        # Run ID + date
        ax2.text(x_off, y, r["id"], transform=ax2.transAxes,
                 color=c, fontsize=11, fontweight="bold", va="top")
        ax2.text(x_off + 0.06, y, r["date"].replace("\n", " "), transform=ax2.transAxes,
                 color=SUBTEXT, fontsize=9, va="top")
        ax2.text(x_off + 0.18, y, f"Graphs: {r['graphs']}", transform=ax2.transAxes,
                 color=SUBTEXT, fontsize=9, va="top")
        # Note lines
        for j, line in enumerate(r["note"].split("\n")):
            ax2.text(x_off + 0.005, y - 0.030 - j * 0.028, line,
                     transform=ax2.transAxes,
                     color=TEXT if j == 0 else SUBTEXT, fontsize=9.5, va="top")
        y -= step
        # Separator
        ax2.plot([x_off, x_off + 0.46], [y + 0.008, y + 0.008],
                 transform=ax2.transAxes, color="#333", linewidth=0.5)

# Vertical divider
ax2.plot([0.50, 0.50], [0.02, 0.94], transform=ax2.transAxes,
         color="#333", linewidth=1)

out2 = "docs/training_changelog.png"
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print(f"Saved: {out2}")
