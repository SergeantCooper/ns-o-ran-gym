#!/usr/bin/env python3
"""plot_report.py - regenerate the FULL figure suite as SEPARATE files under figures/,
each focused on one thing, all computed live from the run folders (no hard-coded numbers).
Does NOT touch the older top-level PNGs (tradeoff.png, summary_tradeoff.png, energy_new.png,
shaped_vs_heuristic.png). Reads the committed manifests rl/*.tsv + rl/allon_baseline_folder.txt.

Outputs:
  figures/1_tradeoff_energy_vs_throughput.png   headline: all-on vs heuristic vs shaped-RL (4-seed avg)
  figures/2_rl_vs_heuristic_by_metric.png       3 panels: energy / throughput / RLF, with error bars
  figures/3_pruning_frontier.png                the tunable pruning frontier (grace sweep, seed 555)
  figures/4_controller_behavior_over_time.png   power / cells-on / sleep-timeline for a shaped-RL run
"""
import os, subprocess
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_energy import summarize

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
INK, MUTE = "#222222", "#666666"
C = {"allon": "#E69F00", "heur": "#0072B2", "rl": "#009E73", "prune": "#CC79A7"}

def metrics(folder):
    s = summarize(folder, None, 600.0, 400.0, 0.0)
    saved = (s["e_base"] - s["e_ctrl"]) / s["e_base"] * 100 if s["e_base"] else 0.0
    return np.array([saved, s["thr"] if s["thr"] else np.nan, s["rlf"] if s["rlf"] is not None else np.nan])

def groups(path, folder_col=-1):
    g = defaultdict(list)
    for ln in open(path):
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            p = ln.split("\t"); g[p[0]].append(p[folder_col])
    return g

shaped = groups("rl/shaped_runs.tsv")                 # rl, heur  (4 seeds each)
allon_folder = open("rl/allon_baseline_folder.txt").read().strip()
rl = np.array([metrics(f) for f in shaped["rl"]]);   rl_m, rl_s = np.nanmean(rl, 0), np.nanstd(rl, 0)
hu = np.array([metrics(f) for f in shaped["heur"]]); hu_m, hu_s = np.nanmean(hu, 0), np.nanstd(hu, 0)
al = metrics(allon_folder)

# ---------- FIG 1: headline energy-vs-throughput scatter (4-seed averages) ----------
fig, ax = plt.subplots(figsize=(9, 6.5))
pts = [("all cells on", al, "allon"), ("balanced heuristic", hu_m, "heur"), ("RL (reward-shaped)", rl_m, "rl")]
for name, m, k in pts:
    ax.scatter(m[1], m[0], s=300, color=C[k], edgecolor="white", lw=2, zorder=4)
    ax.annotate(f"{name}\n{m[0]:.0f}% saved | {m[1]:.2f} Mbps | RLF {m[2]:.2f}",
                (m[1], m[0]), textcoords="offset points", xytext=(12, 10), fontsize=9.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C[k], lw=1.4, alpha=0.96))
ax.annotate("", xy=(rl_m[1], rl_m[0]), xytext=(hu_m[1], hu_m[0]),
            arrowprops=dict(arrowstyle="-|>", color="#888", lw=2, connectionstyle="arc3,rad=-0.2"))
ax.set_xlabel("DL throughput (Mbps)  —  higher = better QoS", fontsize=11)
ax.set_ylabel("↑ Energy saved vs all cells ON (%)", fontsize=11)
ax.set_title("Energy vs throughput (avg of 4 seeds)\nRL: +energy, +reliability (RLF), −throughput vs the heuristic",
             fontsize=12, fontweight="bold")
ax.grid(alpha=0.3); ax.margins(0.25)
fig.tight_layout(); fig.savefig(f"{OUT}/1_tradeoff_energy_vs_throughput.png", dpi=130); plt.close(fig)

# ---------- FIG 2: 3-panel bars (energy / throughput / RLF) with per-seed error bars ----------
PAN = [("Energy saved", 0, "%", True), ("DL throughput", 1, "Mbps", True), ("Dropped calls (RLF)", 2, "", False)]
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
for ax, (title, ci, unit, hib) in zip(axes, PAN):
    means = [hu_m[ci], rl_m[ci]]; stds = [hu_s[ci], rl_s[ci]]
    ax.bar([0, 1], means, yerr=stds, capsize=6, color=[C["heur"], C["rl"]], edgecolor="white",
           lw=1.5, width=0.62, error_kw=dict(ecolor=MUTE, lw=1.5))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["heuristic", "RL\n(shaped)"], fontsize=10)
    ax.set_title(f"{title}\n({'higher' if hib else 'lower'} = better)", fontsize=11, fontweight="bold")
    ax.set_ylabel(unit, fontsize=10)
    ymax = max(m + s for m, s in zip(means, stds))
    for xi, m, sd in zip([0, 1], means, stds):
        ax.text(xi, m + sd + ymax * 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    win = (1 if (means[1] > means[0]) == hib else 0)
    ax.text(win, means[win] + stds[win] + ymax * 0.13, "✓ better", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#00795c" if win == 1 else "#0059a0")
    ax.margins(y=0.30); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
fig.suptitle("RL (reward-shaped) vs balanced heuristic — per metric (avg of 4 seeds; error bars = per-seed spread)",
             fontsize=12, fontweight="bold", y=1.0)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C[k]) for k in ("heur", "rl")],
           labels=["balanced heuristic", "RL (reward-shaped)"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=[0, 0.03, 1, 0.95]); fig.savefig(f"{OUT}/2_rl_vs_heuristic_by_metric.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ---------- FIG 3: pruning frontier (grace sweep, seed 555) ----------
fr = defaultdict(list)
for ln in open("rl/frontier_runs.tsv"):
    p = ln.strip().split("\t")
    if len(p) == 3: fr[p[0]].append((p[1], p[2]))
fig, ax = plt.subplots(figsize=(9, 6.5))
for grp, items in fr.items():
    xs, ys, labs = [], [], []
    for tag, folder in items:
        m = metrics(folder); xs.append(m[1]); ys.append(m[0]); labs.append(tag)
    ax.scatter(xs, ys, s=160, color=C.get(grp, "#999"), edgecolor="white", lw=1.4, zorder=3,
               label={"allon": "all-on", "heur": "heuristic", "prune": "pruning (grace g)"}.get(grp, grp))
    for x, y, t in zip(xs, ys, labs):
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8.5, color=INK)
ax.set_xlabel("DL throughput (Mbps)", fontsize=11); ax.set_ylabel("↑ Energy saved vs all-on (%)", fontsize=11)
ax.set_title("Pruning controller — tunable energy↔QoS frontier (seed 555)\n"
             "grace dial trades throughput for more energy", fontsize=12, fontweight="bold")
ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9); ax.margins(0.18)
fig.tight_layout(); fig.savefig(f"{OUT}/3_pruning_frontier.png", dpi=130); plt.close(fig)

# ---------- FIG 4: controller behaviour over time (reuse plot_energy --plot on a shaped-RL run) ----------
rl_run = shaped["rl"][0]
subprocess.run(["/workspace/.venv/bin/python", "plot_energy.py", rl_run, "--plot",
                f"{OUT}/4_controller_behavior_over_time.png"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("wrote figures/ :"); [print("  ", f) for f in sorted(os.listdir(OUT))]
