#!/usr/bin/env python3
"""plot_tradeoff.py - energy-vs-throughput (QoS) tradeoff across runs.

Each run is one point: x = mean DL throughput (Mbps, higher = better QoS),
y = % energy saved vs same-cells-all-on (higher = greener). Points that share a
label are joined into a curve. A policy "wins" if its point sits UP-and-to-the-
RIGHT of another (more energy saved at equal-or-better throughput) - so this is
the scoreboard for "does the RL beat the heuristic".

Each run is given as   label@tag=folder   (tag is the point caption, e.g. load):
    python3 plot_tradeoff.py \
        all-on@6UEs=output/<base> heuristic@3UEs=output/<a> \
        heuristic@6UEs=output/<b> heuristic@9UEs=output/<c> --out tradeoff.png
    # later: rl@6UEs=output/<d> to overlay the learned policy
"""
import argparse
import os

from plot_energy import summarize   # reuse the power-model energy computation

# colorblind-safe (Okabe-Ito) colors, assigned by policy identity
COLORS = {"heuristic": "#0072B2", "all-on": "#E69F00", "rl": "#009E73",
          "baseline": "#E69F00", "_default": "#CC79A7"}


def point(folder, ps, al, sl):
    s = summarize(folder, None, ps, al, sl)
    saved = (s["e_base"] - s["e_ctrl"]) / s["e_base"] * 100 if s["e_base"] else 0.0
    return s["thr"], saved, s["mean_on"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label@tag=folder per run (tag optional)")
    ap.add_argument("--out", default="tradeoff.png")
    ap.add_argument("--title", default="Energy saved vs throughput (QoS)")
    ap.add_argument("--p-static", type=float, default=600.0)
    ap.add_argument("--alpha", type=float, default=400.0)
    ap.add_argument("--p-sleep", type=float, default=0.0)
    args = ap.parse_args()

    groups = {}
    for item in args.runs:
        spec, _, folder = item.partition("=")
        label, _, tag = spec.partition("@")
        if not folder:
            label, folder = "policy", spec
        groups.setdefault(label, []).append((folder, tag))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTE = "#222222", "#666666"

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    print(f"{'label':<12}{'tag':<8}{'throughput':>12}{'saved%':>9}{'gNBs_on':>9}")
    thr_vals = []
    for label, items in groups.items():
        color = COLORS.get(label, COLORS["_default"])
        pts = []
        for folder, tag in items:
            thr, saved, mon = point(folder, args.p_static, args.alpha, args.p_sleep)
            if thr is None:
                print(f"  (skip {folder}: no throughput)")
                continue
            pts.append((thr, saved, mon, tag)); thr_vals.append(thr)
            print(f"{label:<12}{tag:<8}{thr:>12.2f}{saved:>9.1f}{mon:>9.2f}")
        pts.sort()
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                color=color, lw=2.2, ms=11, label=label, zorder=3)
        for x, y, mon, tag in pts:
            cap = (tag + "\n" if tag else "") + f"{y:.0f}% saved\n{x:.1f} Mbps\n{mon:.1f} gNBs on"
            ax.annotate(cap, (x, y), textcoords="offset points", xytext=(10, 8),
                        fontsize=8.5, color=INK,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9))

    # "better" direction cue (up-and-right)
    ax.annotate("BETTER\nmore energy saved\n(compared at the SAME load)",
                xy=(0.985, 0.96), xycoords="axes fraction", ha="right", va="top",
                fontsize=10, fontweight="bold", color="#00795c",
                bbox=dict(boxstyle="round", fc="#eafaf1", ec="#009E73"))
    ax.annotate("", xy=(0.965, 0.78), xytext=(0.80, 0.60), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#009E73", lw=2.5))

    ax.axhline(0, ls="--", lw=1.5, color=MUTE)
    ax.set_xlabel("DL throughput carried (Mbps)   —   higher = more users / more traffic, NOT 'better service'",
                  fontsize=10)
    ax.set_ylabel("↑ Energy saved vs 'all cells ON'  (%)", fontsize=11)
    ax.set_title(args.title + "\neach dot = one run at a DIFFERENT load (# users);  y=0 = all cells on (no saving)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(loc="lower left", framealpha=0.95, title="policy")
    ax.margins(0.20)
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"\nsaved: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
