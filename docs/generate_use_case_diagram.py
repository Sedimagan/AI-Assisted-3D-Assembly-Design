"""Generate use_case_diagram.png for the dissertation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 16))
ax.set_xlim(0, 14)
ax.set_ylim(0, 16)
ax.axis("off")

C_DARK   = "#1e293b"
C_WHITE  = "#ffffff"
C_BORDER = "#475569"
C_ACTOR  = "#3b82f6"
C_UC     = "#1e40af"
C_SYS    = "#0f172a"
C_LINE   = "#64748b"
C_INCLUDE = "#f59e0b"
C_EXTEND  = "#ef4444"

fig.patch.set_facecolor(C_DARK)

# Title
ax.text(7, 15.5, "Use Case Diagram", ha="center", va="center",
        fontsize=18, fontweight="bold", color=C_WHITE)
ax.text(7, 15.0, "AI-Assisted 3D Assembly Design", ha="center", va="center",
        fontsize=11, color="#94a3b8")

# ── System boundary ──────────────────────────────────────────────────────────
sys_rect = FancyBboxPatch((3.0, 1.0), 8.5, 13.2, boxstyle="round,pad=0.3",
                           facecolor=C_SYS, edgecolor="#3b82f6", linewidth=2, alpha=0.5)
ax.add_patch(sys_rect)
ax.text(7.25, 14.0, "AI-Assisted 3D Assembly Design System",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#93c5fd")


# ── Actor: Engineer (left) ───────────────────────────────────────────────────
def draw_stick_figure(ax, cx, cy, label, color=C_ACTOR):
    head_r = 0.2
    ax.add_patch(plt.Circle((cx, cy + 0.6), head_r, facecolor=color,
                             edgecolor=C_WHITE, linewidth=1.5))
    # body
    ax.plot([cx, cx], [cy + 0.4, cy - 0.1], color=C_WHITE, lw=1.5)
    # arms
    ax.plot([cx - 0.3, cx + 0.3], [cy + 0.25, cy + 0.25], color=C_WHITE, lw=1.5)
    # legs
    ax.plot([cx, cx - 0.25], [cy - 0.1, cy - 0.5], color=C_WHITE, lw=1.5)
    ax.plot([cx, cx + 0.25], [cy - 0.1, cy - 0.5], color=C_WHITE, lw=1.5)
    ax.text(cx, cy - 0.75, label, ha="center", va="top",
            fontsize=10, fontweight="bold", color=C_WHITE)

draw_stick_figure(ax, 1.3, 10.0, "Engineer")

# ── Actor: Gemini AI (right) ─────────────────────────────────────────────────
draw_stick_figure(ax, 12.7, 5.5, "Gemini AI", color="#eab308")

# ── Use cases (ellipses) ─────────────────────────────────────────────────────
use_cases = [
    # (cx, cy, label, color)
    (7.25, 13.0, "Upload STEP File", "#2563eb"),
    (7.25, 11.5, "Train GNN Model", "#7c3aed"),
    (7.25, 10.0, "Detect Missing\nComponents", "#dc2626"),
    (7.25, 8.5,  "Identify Assembly\nType", "#0891b2"),
    (7.25, 7.0,  "Detect Open\nSurfaces", "#65a30d"),
    (7.25, 5.5,  "Generate AI\nExplanation", "#d97706"),
    (7.25, 4.0,  "View 3D Model\nwith Overlays", "#db2777"),
    (7.25, 2.5,  "Review Analysis\nResults", "#6366f1"),
]

for cx, cy, label, color in use_cases:
    ellipse = Ellipse((cx, cy), 4.2, 1.1, facecolor=color, edgecolor=C_WHITE,
                       linewidth=1.5, alpha=0.75)
    ax.add_patch(ellipse)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color=C_WHITE)

# ── Association lines: Engineer → use cases ──────────────────────────────────
engineer_x = 1.8
for _, cy, _, _ in use_cases:
    ax.plot([engineer_x, 5.15], [10.0, cy], color=C_LINE, lw=1.2, alpha=0.6)

# ── Association lines: Gemini AI → Generate AI Explanation ───────────────────
ax.plot([12.2, 9.35], [5.5, 5.5], color=C_LINE, lw=1.5, alpha=0.8)

# ── Include relationships (dashed) ───────────────────────────────────────────
def draw_dependency(ax, x1, y1, x2, y2, label, color=C_INCLUDE):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                                linestyle="dashed"))
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mx + 0.6, my, label, ha="center", va="center",
            fontsize=7, color=color, style="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=C_DARK, edgecolor=color,
                      alpha=0.9, linewidth=0.8))

# Upload STEP → Detect Missing (include)
draw_dependency(ax, 7.25, 12.45, 7.25, 10.55, "<<include>>")

# Detect Missing → Identify Assembly Type (include)
draw_dependency(ax, 7.25, 9.45, 7.25, 9.05, "<<include>>")

# Detect Missing → Detect Open Surfaces (include)
draw_dependency(ax, 8.5, 9.5, 8.5, 7.5, "<<include>>")

# Detect Missing → Generate AI Explanation (include)
draw_dependency(ax, 6.0, 9.5, 6.0, 6.0, "<<include>>")

# ── Extend relationships ────────────────────────────────────────────────────
# Train GNN → Detect Missing (extend)
draw_dependency(ax, 8.8, 11.0, 8.8, 10.5, "<<extend>>", color=C_EXTEND)

# View 3D → Review Analysis (extend)
draw_dependency(ax, 7.25, 3.45, 7.25, 3.05, "<<extend>>", color=C_EXTEND)

# ── Legend ───────────────────────────────────────────────────────────────────
legend_y = 0.5
ax.plot([1.0, 2.0], [legend_y, legend_y], color=C_LINE, lw=1.5)
ax.text(2.2, legend_y, "Association", ha="left", va="center",
        fontsize=8, color=C_WHITE)

ax.plot([4.5, 5.5], [legend_y, legend_y], color=C_INCLUDE, lw=1.3, linestyle="dashed")
ax.annotate("", xy=(5.5, legend_y), xytext=(5.3, legend_y),
            arrowprops=dict(arrowstyle="-|>", color=C_INCLUDE, lw=1.3))
ax.text(5.7, legend_y, "<<include>>", ha="left", va="center",
        fontsize=8, color=C_INCLUDE)

ax.plot([8.5, 9.5], [legend_y, legend_y], color=C_EXTEND, lw=1.3, linestyle="dashed")
ax.annotate("", xy=(9.5, legend_y), xytext=(9.3, legend_y),
            arrowprops=dict(arrowstyle="-|>", color=C_EXTEND, lw=1.3))
ax.text(9.7, legend_y, "<<extend>>", ha="left", va="center",
        fontsize=8, color=C_EXTEND)

plt.tight_layout(pad=0.5)
out = "/Users/mbp/Documents/MTECH/Sem4/Individual_project/AI_Assisted_3D_Assembly_Design/AI-Assisted-3D-Assembly-Design/docs/use_case_diagram.png"
fig.savefig(out, dpi=200, facecolor=C_DARK, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
