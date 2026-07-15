#!/usr/bin/env python3
"""plot_frontier.py - per-seed energy-vs-QoS tradeoff figure (clean scatter + data
table). ALL numbers are computed live from each run's output folder via
plot_energy.summarize() - nothing is hard-coded.

Runs are given either on the command line or (default) read from a manifest file:

    # explicit:
    python3 plot_frontier.py allon@all-on=output/<uuid> heur@heuristic=PPO=output/<uuid> \
                             prune@g3=output/<uuid> ...  --out tradeoff.png
    # or from a manifest (TSV: 'group<TAB>tag<TAB>folder' per line):
    python3 plot_frontier.py                          # reads rl/frontier_runs.tsv

group in {allon, heur, prune, ppo} selects the colour; tag is the short point label.
"""
import argparse
import os
import sys

from plot_energy import summarize          # live power-model metrics from a folder

C   = {"allon": "#E69F00", "heur": "#0072B2", "ppo": "#0072B2", "prune": "#009E73"}
LEG = {"allon": "all cells on", "heur": "heuristic = PPO (tie)",
       "ppo": "PPO", "prune": "pruning (grace g)"}
DEFAULT_MANIFEST = "rl/frontier_runs.tsv"


def parse_runs(args):
    """Yield (group, tag, folder) from CLI 'group@tag=folder' items or the manifest."""
    if args:
        for item in args:
            spec, _, folder = item.partition("=")
            group, _, tag = spec.partition("@")
            yield group, tag, folder
    else:
        with open(DEFAULT_MANIFEST) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                group, tag, folder = line.split("\t")
                yield group, tag, folder


def metrics(folder):
    s = summarize(folder, None, 600.0, 400.0, 0.0)
    saved = (s["e_base"] - s["e_ctrl"]) / s["e_base"] * 100 if s["e_base"] else 0.0
    return s["thr"], saved, s["rlf"], s["mean_on"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="group@tag=folder (else read rl/frontier_runs.tsv)")
    ap.add_argument("--out", default="tradeoff.png")
    a = ap.parse_args()

    rows = []  # (tag, thr, saved, rlf, on, group)
    for group, tag, folder in parse_runs(a.runs):
        thr, saved, rlf, on = metrics(folder)
        if thr is None:
            print(f"  (skip {folder}: no throughput in grafana db)"); continue
        rows.append((tag, thr, saved, rlf, on, group))
        print(f"{tag:16s} thr={thr:.2f}  saved={saved:.1f}%  rlf={rlf:.2f}  on={on:.2f}  [{folder}]")
    if not rows:
        sys.exit("no runs to plot")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    INK, MUTE = "#222222", "#666666"

    fig = plt.figure(figsize=(14, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.16)
    ax = fig.add_subplot(gs[0, 0]); axt = fig.add_subplot(gs[0, 1]); axt.axis("off")

    # leader-line offsets keyed by tag (tags are stable across re-runs)
    OFF = {"all-on": (8, -22), "heuristic=PPO": (44, 20), "g8": (-16, -30),
           "g5": (-52, -6), "g3": (16, 8), "g2": (26, 12), "g1": (-4, 15)}
    for tag, thr, saved, rlf, on, group in rows:
        ax.scatter(thr, saved, s=230, color=C.get(group, "#999999"),
                   edgecolor="white", lw=1.6, zorder=3)
        dx, dy = OFF.get(tag, (12, 10))
        ax.annotate(tag, (thr, saved), textcoords="offset points", xytext=(dx, dy),
                    fontsize=10, fontweight="bold", color=INK, ha="center",
                    arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=C.get(group, "#999999"), alpha=0.95))
    ax.set_xlabel("DL throughput (Mbps)  —  higher = better QoS  "
                  "(pruning saves more energy but sits to the LEFT = less throughput)", fontsize=9)
    ax.set_ylabel("↑ Energy saved vs all cells ON (%)", fontsize=11)
    ax.set_title("Seed 555 (single seed; 4-seed averages in summary_tradeoff.png)",
                 fontsize=10, fontweight="bold", pad=12)
    ax.grid(alpha=0.3)
    ax.set_xlim(4.88, 6.32); ax.set_ylim(-4, 66)   # headroom so top labels clear the title
    seen = []
    for _, _, _, _, _, g in rows:
        if g not in seen:
            seen.append(g)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=11, mfc=C[g], mec="white",
                              label=LEG[g]) for g in seen if g in C],
              loc="lower left", framealpha=0.95, fontsize=9)

    # data table (holds every number; scatter stays uncluttered)
    order = {"allon": 0, "heur": 1, "ppo": 1, "prune": 2}
    rows_sorted = sorted(rows, key=lambda r: (order.get(r[5], 9), -r[2]))
    tint = {"allon": "#fdf1dc", "heur": "#dcecf8", "ppo": "#dcecf8", "prune": "#dcf3ea"}
    cells = [[t, f"{thr:.2f}", f"{sv:.1f}", f"{rlf:.2f}", f"{on:.2f}"]
             for t, thr, sv, rlf, on, g in rows_sorted]
    colcol = [[tint.get(g, "#eee")] * 5 for *_, g in rows_sorted]
    tbl = axt.table(cellText=cells, colLabels=["policy", "thrpt\n(Mbps)", "energy\nsaved %",
                    "RLF\n(drops)", "gNBs\non"], cellColours=colcol, cellLoc="center",
                    loc="center", bbox=[0.0, 0.12, 1.0, 0.72],
                    colWidths=[0.26, 0.18, 0.20, 0.18, 0.18])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 1.9)
    for j in range(5):
        c = tbl[0, j]; c.set_facecolor("#333333"); c.set_text_props(color="white", fontweight="bold")
    for i in range(1, len(cells) + 1):
        tbl[i, 0].set_text_props(ha="left")
    axt.set_title("All numbers (computed live from run folders)", fontsize=11,
                  fontweight="bold", y=0.88)
    axt.text(0.5, 0.06, "PPO ties the heuristic exactly. Pruning saves more energy but "
             "costs throughput + RLF.\nSingle seed shown; 4-seed averages are in "
             "summary_tradeoff.png.", ha="center", va="top", fontsize=8.5,
             color=MUTE, transform=axt.transAxes, style="italic")

    fig.suptitle("New burst scenario @6 UEs — energy-vs-QoS tradeoff", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(a.out, dpi=130)
    print(f"saved: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
