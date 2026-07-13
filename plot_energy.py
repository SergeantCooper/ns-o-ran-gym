#!/usr/bin/env python3
"""plot_energy.py - quantify the energy an ES controller actually saves.

Turns a run's bsState.txt (per-gNB ON/OFF over time) + per-cell DL PRB
utilisation into Watts and "% energy saved", using the power model from the
heuristic spec (3GPP-style RU model, refs R1/R2):

    P_on(util)  = P_static + alpha * util        # util in [0,1]
    P_off(sleep)= P_sleep                         # RU/carrier asleep

Defaults (macro RU, spec section 2): P_static=600 W, alpha=400 W, P_sleep=0 W.

Energy for a run  = sum over (cell, control-step) of P(state, util) * dt.
Baseline (all-on) = the SAME cells but never slept (each still carries the load
it actually served; a slept interval would just draw P_static idle).
    energy_saved_% = (E_baseline - E_controlled) / E_baseline * 100
Because static power dominates (spec: "even at low load most power is static"),
the saving is essentially the static power removed by sleeping cells.

Utilisation note: we use each cell's MEAN DL PRB utilisation (RRU.PrbUsedDl /
dlAvailablePrbs) from du-cell-*.txt as a representative load; the alpha*util
term is small next to P_static, so this is a fair first-order estimate. Pass a
real all-on run with --baseline to compare two runs directly instead.

Usage:
    python3 plot_energy.py output/<uuid>                      # self-contained % saved
    python3 plot_energy.py output/<ctrl> --baseline output/<allon>
    python3 plot_energy.py output/<uuid> --plot energy.png
"""
import argparse
import csv
import glob
import os
import sqlite3
from collections import defaultdict


def read_bsstate(folder):
    """Return {cell: [(t, state), ...]} and the control-step dt (seconds)."""
    path = os.path.join(folder, "bsState.txt")
    per_cell = defaultdict(list)
    times = set()
    with open(path) as f:
        reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
        header = next(reader)  # Timestamp UNIX Id State
        for row in reader:
            row = [c for c in row if c != ""]
            if len(row) < 4:
                continue
            t = float(row[0]); cell = int(row[2]); state = int(row[3])
            per_cell[cell].append((t, state))
            times.add(round(t, 6))
    times = sorted(times)
    # dt = smallest positive gap between logged timestamps (the control period)
    dt = 0.1
    diffs = [b - a for a, b in zip(times, times[1:]) if b - a > 1e-9]
    if diffs:
        dt = min(diffs)
    for cell in per_cell:
        per_cell[cell].sort()
    return per_cell, dt


def read_mean_util(folder):
    """Return {cell: mean DL PRB utilisation in [0,1]} from du-cell-*.txt."""
    util = {}
    for path in glob.glob(os.path.join(folder, "du-cell-*.txt")):
        cell = int(path.split("du-cell-")[1].split(".")[0])
        vals = []
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    used = float(r["RRU.PrbUsedDl"])
                    tot = float(r.get("dlAvailablePrbs", 139) or 139)
                    if tot > 0:
                        vals.append(min(used / tot, 1.0))
                except (KeyError, TypeError, ValueError):
                    pass
        util[cell] = sum(vals) / len(vals) if vals else 0.0
    return util


def read_grafana_means(folder):
    """Mean (throughput Mbps, rlf) from the grafana table, or (None, None).
    throughput is the primary QoS proxy; rlf (radio link failures) is reliability."""
    db = os.path.join(folder, "database.db")
    if not os.path.exists(db):
        return None, None
    try:
        con = sqlite3.connect(db)
        row = con.execute("SELECT AVG(throughput), AVG(rlf) FROM grafana").fetchone()
        con.close()
        thr = float(row[0]) if row and row[0] is not None else None
        rlf = float(row[1]) if row and row[1] is not None else None
        return thr, rlf
    except sqlite3.Error:
        return None, None


def compute_energy(per_cell, dt, util, p_static, alpha, p_sleep):
    """Return (E_controlled_Wh, E_baseline_Wh, per-cell dict)."""
    e_ctrl = e_base = 0.0
    per = {}
    for cell, timeline in sorted(per_cell.items()):
        p_on = p_static + alpha * util.get(cell, 0.0)   # Watts when ON
        on_steps = sum(1 for _, s in timeline if s == 1)
        off_steps = sum(1 for _, s in timeline if s == 0)
        hours = dt / 3600.0
        ctrl = (on_steps * p_on + off_steps * p_sleep) * hours
        base = (len(timeline) * p_on) * hours
        e_ctrl += ctrl
        e_base += base
        per[cell] = dict(p_on_W=p_on, on=on_steps, off=off_steps,
                         ctrl_Wh=ctrl, base_Wh=base)
    return e_ctrl, e_base, per


def summarize(folder, dt, p_static, alpha, p_sleep):
    per_cell, dt_run = read_bsstate(folder)
    dt = dt_run if dt is None else dt
    util = read_mean_util(folder)
    e_ctrl, e_base, per = compute_energy(per_cell, dt, util, p_static, alpha, p_sleep)
    thr, rlf = read_grafana_means(folder)
    n_steps = max((len(t) for t in per_cell.values()), default=0)
    mean_on = (sum(v["on"] for v in per.values()) / n_steps) if n_steps else 0.0
    return dict(folder=folder, dt=dt, per=per, util=util, e_ctrl=e_ctrl,
                e_base=e_base, thr=thr, rlf=rlf, n_steps=n_steps, mean_on=mean_on)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="run output folder (has bsState.txt)")
    ap.add_argument("--baseline", help="an all-on run folder to compare against")
    ap.add_argument("--p-static", type=float, default=600.0, help="static RU power (W)")
    ap.add_argument("--alpha", type=float, default=400.0, help="load-dependent power (W)")
    ap.add_argument("--p-sleep", type=float, default=0.0, help="sleeping RU power (W)")
    ap.add_argument("--plot", help="save a cells-on + power PNG to this path")
    args = ap.parse_args()

    s = summarize(args.folder, None, args.p_static, args.alpha, args.p_sleep)
    print(f"Run: {args.folder}")
    print(f"  control step dt   : {s['dt']:.3f} s over {s['n_steps']} steps")
    print(f"  mean gNBs ON      : {s['mean_on']:.2f} / {len(s['per'])}")
    print("  per-cell (P_on W | on/off steps):")
    for cell, v in s["per"].items():
        print(f"    cell {cell}: {v['p_on_W']:6.1f} W | {v['on']:3d}/{v['off']:3d}  "
              f"util={s['util'].get(cell,0.0)*100:4.1f}%")
    print(f"  energy (controlled): {s['e_ctrl']*1000:8.2f} mWh")
    print(f"  energy (all-on)    : {s['e_base']*1000:8.2f} mWh")
    if s["e_base"] > 0:
        saved = (s["e_base"] - s["e_ctrl"]) / s["e_base"] * 100
        print(f"  >> energy saved vs same-cells-all-on: {saved:.1f}%")
    if s["thr"] is not None:
        print(f"  mean throughput   : {s['thr']:.2f} Mbps  (QoS)")
    if s.get("rlf") is not None:
        print(f"  mean RLF          : {s['rlf']:.3f}  (QoS reliability; lower is better)")

    if args.baseline:
        b = summarize(args.baseline, None, args.p_static, args.alpha, args.p_sleep)
        print(f"\nBaseline run: {args.baseline}")
        print(f"  energy (as-run)   : {b['e_ctrl']*1000:8.2f} mWh")
        if b["e_ctrl"] > 0:
            saved = (b["e_ctrl"] - s["e_ctrl"]) / b["e_ctrl"] * 100
            print(f"  >> controlled vs baseline energy saved: {saved:.1f}%")
        if s["thr"] is not None and b["thr"]:
            dthr = (s["thr"] - b["thr"]) / b["thr"] * 100
            print(f"  >> throughput change vs baseline: {dthr:+.1f}%")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        per_cell, _ = read_bsstate(args.folder)
        util = read_mean_util(args.folder)
        p_on = {c: args.p_static + args.alpha * util.get(c, 0.0) for c in per_cell}
        base_power = sum(p_on.values())                  # every cell ON (validated baseline)
        state_at = {c: dict(tl) for c, tl in per_cell.items()}
        ts = sorted({t for tl in per_cell.values() for t, _ in tl})
        power = [sum(p_on[c] if state_at[c].get(t, 0) == 1 else args.p_sleep
                     for c in per_cell) for t in ts]
        saved = (base_power * len(ts) - sum(power)) / (base_power * len(ts)) * 100 if ts else 0.0
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.step(ts, power, where="post", color="tab:blue", lw=1.8, label="controller power")
        ax.axhline(base_power, ls="--", color="tab:red", label=f"all-on baseline ({base_power:.0f} W)")
        ax.fill_between(ts, power, base_power, step="post", alpha=0.15, color="tab:green")
        ax.set_xlabel("sim time (s)"); ax.set_ylabel("total RU power (W)")
        ax.set_title(f"Energy over time - {os.path.basename(args.folder.rstrip('/'))}  "
                     f"(~{saved:.0f}% saved; green = energy saved)")
        ax.set_ylim(0, base_power * 1.12); ax.grid(alpha=0.3); ax.legend(loc="lower right")
        fig.tight_layout(); fig.savefig(args.plot, dpi=120)
        print(f"  plot saved: {args.plot}  (~{saved:.0f}% energy saved vs all-on)")


if __name__ == "__main__":
    main()
