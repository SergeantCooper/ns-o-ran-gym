#!/usr/bin/env python3
"""plot_summary.py - honest averaged summary of the new (burst) scenario results,
over seeds 555/777/999/1234. Two panels: energy saved vs throughput, and energy
saved vs RLF (reliability). Shows the heuristic (= the tied PPO policy) and the
aggressive pruning controller as points on the SAME energy-vs-QoS frontier."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# averaged over 4 seeds
pts = {  # name: (throughput Mbps, energy saved %, RLF, color)
    "all cells on":       (6.10, 0.0,  0.00, "#E69F00"),
    "heuristic (= PPO)":  (5.68, 33.8, 1.20, "#0072B2"),
    "pruning g=3":        (5.25, 52.9, 2.36, "#009E73"),
}
INK, MUTE = "#222222", "#666666"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

for ax, xi, xlabel, xinv in [
    (ax1, 0, "DL throughput (Mbps)  —  higher = better QoS", False),
    (ax2, 2, "Radio-link failures (RLF)  —  lower = better QoS", True)]:
    for name, v in pts.items():
        ax.scatter(v[xi], v[1], s=260, color=v[3], edgecolor="white", lw=1.5, zorder=3)
        ax.annotate(f"{name}\n{v[1]:.0f}% saved\n{v[0]:.2f} Mbps, RLF {v[2]:.2f}",
                    (v[xi], v[1]), textcoords="offset points", xytext=(10, 10),
                    fontsize=9, color=INK,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=v[3], alpha=0.9))
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("↑ Energy saved vs all-on (%)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 65)
    if xinv:
        ax.invert_xaxis()  # so "better QoS" (low RLF) is on the right in both panels

ax1.set_title("More energy needs less throughput", fontsize=11, fontweight="bold")
ax2.set_title("More energy costs more dropped calls", fontsize=11, fontweight="bold")
fig.suptitle("New burst scenario @6 UEs (avg of 4 seeds): heuristic is near the efficient "
             "energy-vs-QoS frontier\nPPO ties it; pruning trades QoS for more energy — no point "
             "strictly beats the heuristic",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("summary_tradeoff.png", dpi=130)
print("saved: summary_tradeoff.png")
