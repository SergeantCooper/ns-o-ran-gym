#!/usr/bin/env python3
"""plot_shaped.py - the shaped-reward RL vs balanced-heuristic result (RESULTS.md §7).
Reads rl/shaped_runs.tsv (policy<TAB>seed<TAB>folder), computes every metric LIVE from the
run folders via plot_energy.summarize() (no hard-coded numbers), and draws a 3-panel bar
chart (energy saved / throughput / dropped calls) with per-seed error bars. -> shaped_vs_heuristic.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from plot_energy import summarize

MANIFEST = "rl/shaped_runs.tsv"
runs = defaultdict(list)
for line in open(MANIFEST):
    line = line.strip()
    if line and not line.startswith("#"):
        pol, _seed, folder = line.split("\t")
        runs[pol].append(folder)

def metrics(folder):
    s = summarize(folder, None, 600.0, 400.0, 0.0)
    saved = (s["e_base"] - s["e_ctrl"]) / s["e_base"] * 100 if s["e_base"] else 0.0
    return saved, s["thr"], s["rlf"]

# per-policy arrays: rows = seeds, cols = [saved%, thr, rlf]
data = {pol: np.array([metrics(f) for f in fs]) for pol, fs in runs.items()}
POL = [("heur", "balanced heuristic", "#0072B2"), ("rl", "RL (reward-shaped)", "#009E73")]

# (title, col index, unit, higher_is_better)
PANELS = [("Energy saved", 0, "%", True),
          ("DL throughput", 1, "Mbps", True),
          ("Dropped calls (RLF)", 2, "", False)]
INK, MUTE = "#222222", "#666666"
fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))

for ax, (title, ci, unit, hib) in zip(axes, PANELS):
    means = [data[p][:, ci].mean() for p, _, _ in POL]
    stds  = [data[p][:, ci].std()  for p, _, _ in POL]
    cols  = [c for _, _, c in POL]
    x = np.arange(len(POL))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=cols, edgecolor="white",
                  lw=1.5, width=0.62, error_kw=dict(ecolor=MUTE, lw=1.5))
    ax.set_xticks(x); ax.set_xticklabels(["heuristic", "RL\n(shaped)"], fontsize=10)
    ax.set_title(f"{title}\n({'higher' if hib else 'lower'} = better)", fontsize=11, fontweight="bold")
    ax.set_ylabel(unit, fontsize=10)
    ymax = max(m + s for m, s in zip(means, stds))
    for xi, m, sd in zip(x, means, stds):
        ax.text(xi, m + sd + ymax * 0.02, f"{m:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=INK)
    # mark the winner, above its bar (clear of the x-axis labels)
    rl_i, heur_i = 1, 0
    rl_better = (means[rl_i] > means[heur_i]) if hib else (means[rl_i] < means[heur_i])
    win_i = rl_i if rl_better else heur_i
    ax.text(x[win_i], means[win_i] + stds[win_i] + ymax * 0.13, "✓ better", ha="center",
            va="bottom", fontsize=10, fontweight="bold",
            color="#00795c" if win_i == rl_i else "#0059a0")
    ax.margins(y=0.30); ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

fig.suptitle("Reward-shaped RL vs balanced heuristic  (uniform-burst scenario, avg of 4 seeds; error bars = per-seed spread)\n"
             "RL is greener (+energy) and more reliable (fewer drops) but carries a little less throughput — a favourable tradeoff, not strict domination",
             fontsize=11.5, fontweight="bold", y=1.02)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in POL],
           labels=[lbl for _, lbl, _ in POL], loc="lower center", ncol=2, framealpha=0.95,
           bbox_to_anchor=(0.5, -0.04), fontsize=10)
fig.tight_layout(rect=[0, 0.03, 1, 0.93])
fig.savefig("shaped_vs_heuristic.png", dpi=130, bbox_inches="tight")
print("saved: shaped_vs_heuristic.png")
for pol, lbl, _ in POL:
    m = data[pol].mean(0); print(f"  {lbl:22s} saved={m[0]:.1f}%  thr={m[1]:.2f}  rlf={m[2]:.2f}")
