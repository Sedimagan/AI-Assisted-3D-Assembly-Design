"""Generate system_architecture.png for the dissertation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(16, 20))
ax.set_xlim(0, 16)
ax.set_ylim(0, 22)
ax.axis("off")

# Colours
C_INPUT   = "#3b82f6"   # blue
C_PARSE   = "#8b5cf6"   # purple
C_GRAPH   = "#06b6d4"   # cyan
C_MODEL   = "#ef4444"   # red
C_LINK    = "#f97316"   # orange
C_TEMPL   = "#14b8a6"   # teal
C_OCTREE  = "#84cc16"   # lime
C_AIDA    = "#eab308"   # yellow
C_FRONT   = "#ec4899"   # pink
C_WHITE   = "#ffffff"
C_DARK    = "#1e293b"

def draw_box(x, y, w, h, color, label, fontsize=10, sublabel=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.92)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=C_WHITE)
        ax.text(x + w/2, y + h/2 - 0.25, sublabel, ha="center", va="center",
                fontsize=7.5, color=C_WHITE, style="italic")
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=C_WHITE)

def draw_arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2))

# Background
fig.patch.set_facecolor(C_DARK)

# Title
ax.text(8, 21.3, "System Architecture", ha="center", va="center",
        fontsize=18, fontweight="bold", color=C_WHITE)
ax.text(8, 20.8, "AI-Assisted 3D Assembly Design", ha="center", va="center",
        fontsize=12, color="#94a3b8")

# ── Row 1: Input ─────────────────────────────────────────────────────────────
draw_box(5.5, 19.2, 5, 0.9, C_INPUT, "STEP / STP File Input",
         fontsize=11, sublabel="ISO 10303 CAD Assembly")

# Arrow down
draw_arrow(8, 19.2, 8, 18.5)

# ── Row 2: Parsing Pipeline ──────────────────────────────────────────────────
draw_box(1.5, 17.0, 4.0, 1.3, C_PARSE, "gmsh + OpenCASCADE",
         sublabel="STEP parse · fragment()\ncontact detect · bbox · volume")

draw_box(6.0, 17.0, 4.0, 1.3, C_PARSE, "trimesh",
         sublabel="Exact surface area\nSDF ray-casting (mean+var)")

draw_box(10.5, 17.0, 4.0, 1.3, C_PARSE, "Type Inference",
         sublabel="SDF + bbox heuristics\n8-class component typing")

# Arrows between parsing stages
draw_arrow(5.5, 17.65, 6.0, 17.65)
draw_arrow(10.0, 17.65, 10.5, 17.65)

# Arrow from input to parsing
draw_arrow(8, 18.5, 3.5, 18.3)
draw_arrow(8, 18.5, 8, 18.3)
draw_arrow(8, 18.5, 12.5, 18.3)

# Section label
ax.text(0.5, 17.65, "Stage 1\nParsing", ha="center", va="center",
        fontsize=8, color="#94a3b8", style="italic")

# Arrow down from parsing to graph
draw_arrow(8, 17.0, 8, 16.3)

# ── Row 3: Graph Construction ────────────────────────────────────────────────
draw_box(3.5, 15.0, 9, 1.1, C_GRAPH, "PyG Assembly Graph Construction",
         fontsize=11, sublabel="Nodes: 22-dim (type OH + vol + SA + bbox + shape + SDF + SA/V + holes)  ·  Edges: 6-dim (mate + weight + joint type)")

ax.text(0.5, 15.55, "Stage 2\nGraph", ha="center", va="center",
        fontsize=8, color="#94a3b8", style="italic")

# Arrow down
draw_arrow(8, 15.0, 8, 14.3)

# ── Row 4: GNN Model ─────────────────────────────────────────────────────────
draw_box(4.0, 12.8, 8, 1.3, C_MODEL, "AssemblyGNN — 3-Layer GAT Encoder",
         fontsize=11, sublabel="22 → 1024 (8 heads) → 512 (4 heads) → 64 (1 head)  ·  ELU + BN + Dropout")

ax.text(0.5, 13.45, "Stage 3\nEncoder", ha="center", va="center",
        fontsize=8, color="#94a3b8", style="italic")

# Arrow down — split into 3 branches
draw_arrow(8, 12.8, 3.5, 12.0)
draw_arrow(8, 12.8, 8, 12.0)
draw_arrow(8, 12.8, 12.5, 12.0)

# ── Row 5: Three Analysis Modules (parallel) ─────────────────────────────────
# Left: LinkPredictor
draw_box(1.0, 10.2, 4.5, 1.6, C_LINK, "LinkPredictor MLP",
         fontsize=10, sublabel="MLP(z_u || z_v) → score\nBCE + hard negatives\nMissing component detection")

# Center: AssemblyTemplateDB
draw_box(5.8, 10.2, 4.5, 1.6, C_TEMPL, "AssemblyTemplateDB",
         fontsize=10, sublabel="Per-category templates\nJaccard + subset scoring\nGap analysis")

# Right: Octree Surface Detector
draw_box(10.5, 10.2, 4.5, 1.6, C_OCTREE, "Octree Open Surface",
         fontsize=10, sublabel="Borah & Borah (2020)\nFree-surface clustering\nJoint localisation")

ax.text(0.3, 11.0, "Stage 4\nAnalysis", ha="center", va="center",
        fontsize=8, color="#94a3b8", style="italic")

# Arrows down from 3 modules
draw_arrow(3.25, 10.2, 3.25, 9.5)
draw_arrow(8.05, 10.2, 8.05, 9.5)
draw_arrow(12.75, 10.2, 12.75, 9.5)

# ── Row 5b: Metrics under LinkPredictor ───────────────────────────────────────
draw_box(1.0, 8.7, 4.5, 0.7, "#b91c1c", "AUC-ROC · Avg Precision",
         fontsize=9, sublabel=None)
ax.text(3.25, 8.5, "Phase 1 metrics", ha="center", va="top",
        fontsize=7, color="#94a3b8")

draw_box(5.8, 8.7, 4.5, 0.7, "#0f766e", "Assembly Type · Missing Types",
         fontsize=9, sublabel=None)

draw_box(10.5, 8.7, 4.5, 0.7, "#4d7c0f", "Open Joint Surfaces",
         fontsize=9, sublabel=None)

# Arrows converge down to AIDA
draw_arrow(3.25, 8.7, 8, 7.9)
draw_arrow(8.05, 8.7, 8, 7.9)
draw_arrow(12.75, 8.7, 8, 7.9)

# ── Row 6: AIDA Skills Agent ─────────────────────────────────────────────────
draw_box(4.0, 6.5, 8, 1.2, C_AIDA, "AIDA — Gemini Skills AI Agent",
         fontsize=11, sublabel="6 domain skills · gemini-2.0-flash · Structured 4-section explanation")

ax.text(0.5, 7.1, "Stage 5\nExplain", ha="center", va="center",
        fontsize=8, color="#94a3b8", style="italic")

# Arrow down
draw_arrow(8, 6.5, 8, 5.8)

# ── Row 7: Frontend ──────────────────────────────────────────────────────────
draw_box(2.5, 3.8, 11, 1.8, C_FRONT, "Streamlit Dual-Panel UI",
         fontsize=12, sublabel=None)

# Sub-boxes inside frontend
draw_box(3.0, 4.1, 4.5, 1.0, "#be185d",
         "Left: AI Analysis Panel",
         fontsize=9,
         sublabel="6 sections · colour-coded")

draw_box(8.5, 4.1, 4.5, 1.0, "#be185d",
         "Right: 3D Viewer",
         fontsize=9,
         sublabel="Plotly Mesh3d · overlays")

ax.text(0.5, 4.7, "Stage 6\nUI", ha="center", va="center",
        fontsize=8, color="#94a3b8", style="italic")

# ── Row 8: Output legend ─────────────────────────────────────────────────────
ax.text(8, 3.2, "3D Overlay Colours", ha="center", va="center",
        fontsize=9, fontweight="bold", color=C_WHITE)

legend_items = [
    ("#ef4444", "Red — Not assembled"),
    ("#f97316", "Orange — Auto-reference missing"),
    ("#84cc16", "Lime — Open joints"),
    ("#5b9bd5", "Blue — GNN predicted links"),
    ("#14b8a6", "Teal — Template missing"),
    ("#a78bfa", "Purple — Assembly type"),
]

for i, (color, label) in enumerate(legend_items):
    col = i % 3
    row = i // 3
    x = 2.5 + col * 4.0
    y = 2.6 - row * 0.45
    rect = FancyBboxPatch((x, y - 0.12), 0.3, 0.24, boxstyle="round,pad=0.02",
                           facecolor=color, edgecolor="white", linewidth=0.5)
    ax.add_patch(rect)
    ax.text(x + 0.45, y, label, ha="left", va="center",
            fontsize=7.5, color=C_WHITE)

# ── Side annotation: Training config ─────────────────────────────────────────
config_text = (
    "Training Config\n"
    "─────────────\n"
    "Optimizer: Adam\n"
    "LR: 1e-3\n"
    "Batch: 32\n"
    "Epochs: 200\n"
    "Early stop: 20\n"
    "5-fold CV\n"
    "Neg ratio: 0.5\n"
    "Hard neg: 0.3×"
)

ax.text(15.5, 13.0, config_text, ha="center", va="center",
        fontsize=7, color="#94a3b8", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#334155", edgecolor="#475569", alpha=0.8))

# ── Side annotation: Dataset ─────────────────────────────────────────────────
data_text = (
    "Dataset\n"
    "────────\n"
    "1,760 graphs\n"
    "24,811 nodes\n"
    "33,166 edges\n"
    "643 Fusion360\n"
    "137 local\n"
    "5-fold CV"
)

ax.text(15.5, 17.0, data_text, ha="center", va="center",
        fontsize=7, color="#94a3b8", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#334155", edgecolor="#475569", alpha=0.8))

plt.tight_layout(pad=0.5)
out = "/Users/mbp/Documents/MTECH/Sem4/Individual_project/AI_Assisted_3D_Assembly_Design/AI-Assisted-3D-Assembly-Design/docs/system_architecture.png"
fig.savefig(out, dpi=200, facecolor=C_DARK, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
