#!/usr/bin/env python3
"""plot_summary.py - single-panel honest summary of the new (burst) scenario,
averaged over 4 seeds (555/777/999/1234). One scatter: energy saved (y) vs
throughput (x); each point also carries its RLF (dropped calls) in the label, and
an arrow shows the heuristic->pruning trade. All-on is the reference at 0% saved."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 4-seed averages: (label, throughput Mbps, energy saved %, RLF, color, label-offset)
P = {
    "allon": ("all cells on\n(reference)",         6.10,  0.0, 0.00, "#E69F00", (-140, 10)),
    "heur":  ("heuristic  =  PPO\n(RL ties it)",    5.68, 33.8, 1.20, "#0072B2", (18, -30)),
    "prune": ("pruning (aggressive)",               5.25, 52.9, 2.36, "#009E73", (14, 6)),
}
INK, MUTE = "#222222", "#666666"
fig, ax = plt.subplots(figsize=(10.5, 7))

for name, thr, saved, rlf, col, off in P.values():
    ax.scatter(thr, saved, s=320, color=col, edgecolor="white", lw=2, zorder=4)
    ax.annotate(f"{name}\n{saved:.0f}% energy saved  |  {thr:.2f} Mbps  |  RLF {rlf:.2f}",
                (thr, saved), textcoords="offset points", xytext=off,
                fontsize=10, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=col, lw=1.5, alpha=0.97))

# the trade: heuristic -> pruning
h, p = P["heur"], P["prune"]
ax.annotate("", xy=(p[1], p[2]), xytext=(h[1], h[2]),
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=2.4,
                            connectionstyle="arc3,rad=-0.15"), zorder=2)
ax.annotate("the trade:\n+19 pts energy saved\nBUT −0.4 Mbps  &  ~2× dropped calls",
            xy=(5.44, 45), fontsize=9.5, color="#666666", ha="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f4f4f4", ec="#bbbbbb"))

ax.set_xlabel("DL throughput carried (Mbps)   —   → right = better QoS", fontsize=11)
ax.set_ylabel("↑ Energy saved vs 'all cells on' (%)", fontsize=11)
ax.set_title("New burst scenario @6 UEs — averaged over 4 seeds\n"
             "The heuristic is near the efficient energy-vs-QoS frontier: PPO ties it, "
             "and\nsaving more energy (pruning) costs throughput AND reliability — no clean win",
             fontsize=11.5, fontweight="bold")
ax.grid(alpha=0.3)
ax.set_xlim(5.05, 6.35); ax.set_ylim(-6, 62)
# "better" corner cue
ax.annotate("BETTER\n(more energy, same QoS)\n— no policy reaches here",
            xy=(0.985, 0.97), xycoords="axes fraction", ha="right", va="top",
            fontsize=9.5, fontweight="bold", color="#00795c",
            bbox=dict(boxstyle="round", fc="#eafaf1", ec="#009E73"))
fig.tight_layout()
fig.savefig("summary_tradeoff.png", dpi=130)
print("saved: summary_tradeoff.png")
