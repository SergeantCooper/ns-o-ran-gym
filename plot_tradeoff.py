#!/usr/bin/env python3
"""plot_tradeoff.py - energy-vs-throughput tradeoff scatter across runs.

Each run folder becomes one point: x = mean DL throughput (Mbps), y = % energy
saved vs same-cells-all-on (both from plot_energy's power model). Points sharing
a policy label are joined into a curve, sorted by throughput. The all-on
baseline is the y=0 line (it saves nothing by definition).

This is the scoreboard for "does policy X beat the heuristic": a better policy
sits UP and to the RIGHT (more energy saved at equal-or-higher throughput).

Usage:
    python3 plot_tradeoff.py heuristic=output/<a> heuristic=output/<b> \
        --out tradeoff.png --title "ES tradeoff: scenario-three"
    # (later) add rl=output/<c> to overlay the learned policy
"""
import argparse
import os

from plot_energy import summarize   # reuse the power-model energy computation


def point(folder, p_static, alpha, p_sleep):
    s = summarize(folder, None, p_static, alpha, p_sleep)
    saved = (s["e_base"] - s["e_ctrl"]) / s["e_base"] * 100 if s["e_base"] else 0.0
    return s["thr"], saved, s["mean_on"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=folder (or just folder) per run")
    ap.add_argument("--out", default="tradeoff.png")
    ap.add_argument("--title", default="Energy vs throughput tradeoff")
    ap.add_argument("--p-static", type=float, default=600.0)
    ap.add_argument("--alpha", type=float, default=400.0)
    ap.add_argument("--p-sleep", type=float, default=0.0)
    args = ap.parse_args()

    # group folders by label
    groups = {}
    for item in args.runs:
        label, _, folder = item.partition("=")
        if not folder:
            label, folder = "policy", label
        groups.setdefault(label, []).append(folder)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    print(f"{'label':<12}{'throughput(Mbps)':>18}{'energy saved(%)':>18}{'mean_on':>10}")
    for label, folders in groups.items():
        pts = []
        for f in folders:
            thr, saved, mon = point(f, args.p_static, args.alpha, args.p_sleep)
            if thr is None:
                print(f"  (skip {f}: no throughput/grafana)")
                continue
            pts.append((thr, saved, mon))
            print(f"{label:<12}{thr:>18.2f}{saved:>18.1f}{mon:>10.2f}")
        pts.sort()
        if pts:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", label=label)
            for x, y, mon in pts:
                ax.annotate(f"{mon:.1f} on", (x, y), textcoords="offset points",
                            xytext=(6, 6), fontsize=8)

    ax.axhline(0, ls="--", color="gray", lw=1, label="all-on baseline (0%)")
    ax.set_xlabel("mean DL throughput (Mbps)")
    ax.set_ylabel("energy saved vs all-on (%)")
    ax.set_title(args.title)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print(f"\nsaved: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
