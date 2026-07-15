#!/usr/bin/env python3
"""plot_summary.py - single-panel honest summary, averaged over seeds. ALL numbers
are computed live from the run folders via plot_energy.summarize() - nothing is
hard-coded. Energy saved (y) vs throughput (x); each point carries its RLF in the
label; an arrow shows the heuristic->pruning trade.

Runs are given on the command line or (default) read from a manifest:

    python3 plot_summary.py allon=output/<uuid> heur=output/<a> heur=output/<b> \
                            prune=output/<c> prune=output/<d> ...   --out summary_tradeoff.png
    python3 plot_summary.py                       # reads rl/summary_runs.tsv  (group<TAB>folder)

A group may appear multiple times (one folder per seed); its metrics are averaged.
"""
import argparse
import os
import sys
from collections import defaultdict

from plot_energy import summarize

DEFAULT_MANIFEST = "rl/summary_runs.tsv"
STYLE = {  # group: (display label, colour, label-offset)
    "allon": ("all cells on\n(reference)",      "#E69F00", (-150, 10)),
    "heur":  ("heuristic  =  PPO\n(RL ties it)", "#0072B2", (18, -34)),
    "prune": ("pruning (aggressive)",           "#009E73", (14, 6)),
}


def parse_runs(args):
    if args:
        for item in args:
            group, _, folder = item.partition("=")
            yield group, folder
    else:
        with open(DEFAULT_MANIFEST) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    g, folder = line.split("\t")
                    yield g, folder


def metrics(folder):
    s = summarize(folder, None, 600.0, 400.0, 0.0)
    saved = (s["e_base"] - s["e_ctrl"]) / s["e_base"] * 100 if s["e_base"] else 0.0
    return s["thr"], saved, s["rlf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="group=folder (repeatable; else rl/summary_runs.tsv)")
    ap.add_argument("--out", default="summary_tradeoff.png")
    a = ap.parse_args()

    by_group = defaultdict(list)
    for group, folder in parse_runs(a.runs):
        by_group[group].append(metrics(folder))

    avg = {}  # group -> (thr, saved, rlf, n)
    for g, ms in by_group.items():
        ms = [m for m in ms if m[0] is not None]
        if not ms:
            continue
        n = len(ms)
        avg[g] = (sum(m[0] for m in ms) / n, sum(m[1] for m in ms) / n,
                  sum(m[2] for m in ms) / n, n)
        print(f"{g:6s} n={n}  thr={avg[g][0]:.2f}  saved={avg[g][1]:.1f}%  rlf={avg[g][2]:.2f}")
    if not avg:
        sys.exit("no runs to plot")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTE = "#222222", "#666666"
    fig, ax = plt.subplots(figsize=(10.5, 7))

    for g, (thr, saved, rlf, n) in avg.items():
        label, col, off = STYLE.get(g, (g, "#999999", (12, 10)))
        ax.scatter(thr, saved, s=320, color=col, edgecolor="white", lw=2, zorder=4)
        ax.annotate(f"{label}\n{saved:.0f}% energy saved  |  {thr:.2f} Mbps  |  RLF {rlf:.2f}"
                    + (f"   (avg of {n} seeds)" if n > 1 else ""),
                    (thr, saved), textcoords="offset points", xytext=off,
                    fontsize=10, color=INK, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=col, lw=1.5, alpha=0.97))

    if "heur" in avg and "prune" in avg:
        h, p = avg["heur"], avg["prune"]
        ax.annotate("", xy=(p[0], p[1]), xytext=(h[0], h[1]),
                    arrowprops=dict(arrowstyle="-|>", color="#888888", lw=2.4,
                                    connectionstyle="arc3,rad=-0.15"), zorder=2)
        ax.annotate(f"the trade:\n+{p[1]-h[1]:.0f} pts energy saved\n"
                    f"BUT {p[0]-h[0]:+.2f} Mbps  &  ~{p[2]/h[2]:.1f}× dropped calls",
                    xy=((h[0]+p[0])/2, (h[1]+p[1])/2 + 3), fontsize=9.5, color="#666666",
                    ha="center", style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#f4f4f4", ec="#bbbbbb"))

    ax.set_xlabel("DL throughput carried (Mbps)   —   → right = better QoS", fontsize=11)
    ax.set_ylabel("↑ Energy saved vs 'all cells on' (%)", fontsize=11)
    ax.set_title("New burst scenario @6 UEs — averaged over seeds (computed live)\n"
                 "The heuristic is near the efficient energy-vs-QoS frontier: PPO ties it, "
                 "and\nsaving more energy (pruning) costs throughput AND reliability — no clean win",
                 fontsize=11.5, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.annotate("BETTER\n(more energy, same QoS)\n— no policy reaches here",
                xy=(0.985, 0.97), xycoords="axes fraction", ha="right", va="top",
                fontsize=9.5, fontweight="bold", color="#00795c",
                bbox=dict(boxstyle="round", fc="#eafaf1", ec="#009E73"))
    ax.margins(0.22)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"saved: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
