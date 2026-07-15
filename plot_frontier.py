#!/usr/bin/env python3
"""plot_frontier.py - clean, non-overlapping tradeoff figure for the new (burst)
scenario, seed 555. Scatter (energy saved vs throughput) with short tags + leader
lines, and a full data table alongside (so all numbers, incl. RLF, are legible and
nothing collides). Heuristic and PPO are the SAME point (a tie) and are drawn once.
NOTE: single seed 555 (illustrative); the 4-seed averaged truth is summary_tradeoff.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# (label, throughput Mbps, energy saved %, RLF, gNBs on, group)   -- seed 555
ROWS = [
    ("all cells on",      6.10,  0.0, 0.00, 7.00, "allon"),
    ("heuristic = PPO",   5.71, 35.5, 1.40, 4.34, "heur"),
    ("pruning g=8",       5.04, 49.0, 2.12, 3.32, "prune"),
    ("pruning g=5",       5.24, 51.5, 1.53, 3.07, "prune"),
    ("pruning g=3",       5.57, 48.0, 1.05, 3.27, "prune"),
    ("pruning g=2",       5.25, 55.4, 2.71, 2.86, "prune"),
    ("pruning g=1",       5.08, 57.0, 2.62, 2.68, "prune"),
]
C = {"allon": "#E69F00", "heur": "#0072B2", "prune": "#009E73"}
INK, MUTE = "#222222", "#666666"

fig = plt.figure(figsize=(14, 6.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.16)
ax = fig.add_subplot(gs[0, 0])
axt = fig.add_subplot(gs[0, 1]); axt.axis("off")

# leader-line label offsets (points), tuned so nothing overlaps
OFF = {"all cells on": (8, -22), "heuristic = PPO": (44, 20),
       "pruning g=8": (-14, -34), "pruning g=5": (-52, -6),
       "pruning g=3": (16, 8), "pruning g=2": (12, 22), "pruning g=1": (-12, 26)}
short = {"all cells on": "all-on", "heuristic = PPO": "heuristic=PPO",
         "pruning g=8": "g8", "pruning g=5": "g5", "pruning g=3": "g3",
         "pruning g=2": "g2", "pruning g=1": "g1"}

for name, thr, saved, rlf, on, grp in ROWS:
    ax.scatter(thr, saved, s=230, color=C[grp], edgecolor="white", lw=1.6, zorder=3)
    dx, dy = OFF[name]
    ax.annotate(short[name], (thr, saved), textcoords="offset points", xytext=(dx, dy),
                fontsize=10, fontweight="bold", color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C[grp], alpha=0.95))

ax.set_xlabel("DL throughput (Mbps)  —  higher = better QoS", fontsize=10)
ax.set_ylabel("↑ Energy saved vs all cells ON (%)", fontsize=11)
ax.set_title("Seed 555: energy vs throughput\n(pruning saves more energy but to the LEFT = less throughput)",
             fontsize=11, fontweight="bold")
ax.grid(alpha=0.3); ax.set_ylim(-6, 64); ax.set_xlim(4.85, 6.3)
ax.legend(handles=[Line2D([0],[0], marker="o", ls="", ms=11, mfc=c, mec="white",
                          label={"allon":"all cells on","heur":"heuristic = PPO (tie)",
                                 "prune":"pruning (grace g)"}[k])
                   for k, c in C.items()],
          loc="lower left", framealpha=0.95, fontsize=9)

# ---- data table (holds every number so the scatter stays uncluttered) ----
col_labels = ["policy", "thrpt\n(Mbps)", "energy\nsaved %", "RLF\n(drops)", "gNBs\non"]
cells, colors = [], []
for name, thr, saved, rlf, on, grp in ROWS:
    cells.append([name, f"{thr:.2f}", f"{saved:.1f}", f"{rlf:.2f}", f"{on:.2f}"])
    tint = {"allon": "#fdf1dc", "heur": "#dcecf8", "prune": "#dcf3ea"}[grp]
    colors.append([tint] * 5)
tbl = axt.table(cellText=cells, colLabels=col_labels, cellColours=colors,
                cellLoc="center", loc="center", bbox=[0.0, 0.12, 1.0, 0.72],
                colWidths=[0.34, 0.16, 0.18, 0.16, 0.16])
tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 1.9)
for j in range(5):
    c = tbl[0, j]; c.set_facecolor("#333333"); c.set_text_props(color="white", fontweight="bold")
for i in range(1, len(ROWS) + 1):                    # left-align + pad the policy column
    c = tbl[i, 0]; c.set_text_props(ha="left"); c.PAD = 0.04
axt.set_title("All numbers (seed 555)", fontsize=11, fontweight="bold", y=0.88)
axt.text(0.5, 0.06,
         "PPO ties the heuristic exactly. Pruning saves more energy but costs throughput + RLF.\n"
         "Single seed shown; 4-seed averages (the honest picture) are in summary_tradeoff.png.",
         ha="center", va="top", fontsize=8.5, color=MUTE, transform=axt.transAxes, style="italic")

fig.suptitle("New burst scenario @6 UEs — energy-vs-QoS tradeoff", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("tradeoff_new.png", dpi=130)
print("saved: tradeoff_new.png")
