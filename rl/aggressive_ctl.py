#!/usr/bin/env python3
"""aggressive_ctl.py - an energy-saving controller that targets the ONE thing
naive PPO could not isolate: the heuristic's habit of keeping traffic-less cells
powered (measured at 43% of on-cell-time; ~30% wasted energy).

Two modes:
  * prune    - run the TwinHeuristic, then turn OFF any non-anchor cell it keeps
               ON but that has carried no traffic (PRB<thr, no RLF) for `grace`
               consecutive steps. Keeps the heuristic's (good) WAKE decisions and
               only removes the waste. This is the safe, high-confidence variant.
  * reactive - pure feedback rule (no heuristic): a non-anchor cell is ON iff it
               is loaded, or failing (RLF>0), or within `grace` steps of its last
               activity. Tests whether waking can be driven by observations alone.

Reports its output folder so it can be scored with plot_energy / plot_tradeoff.
"""
import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from es_wrapper import EnergySavingRLEnv          # noqa: E402
from heuristic_twin import TwinHeuristic          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns3_path", default="/workspace/ns-3-mmwave-oran")
    ap.add_argument("--config", default="src/environments/scenario_configurations/es_use_case.json")
    ap.add_argument("--num_steps", type=int, default=58)
    ap.add_argument("--mode", choices=["prune", "reactive"], default="prune")
    ap.add_argument("--thr", type=float, default=3.0, help="PRB%% below this = idle")
    ap.add_argument("--grace", type=int, default=2, help="idle steps tolerated before sleeping")
    ap.add_argument("--anchor", type=int, nargs="+", default=[2])
    a = ap.parse_args()

    env = EnergySavingRLEnv(ns3_path=a.ns3_path, config=a.config)
    cs, cells = env.columns_state, env.cell_list
    n = len(cells)
    iprb = [cs.index(f"RRU_PRBTOTDL_{c}") for c in cells]
    irlf = [cs.index(f"RLF_VALUE_{c}") for c in cells]
    anchor_set = set(a.anchor)
    heur = TwinHeuristic(cell_list=cells, anchor_cells=tuple(a.anchor)) if a.mode == "prune" else None
    idle_streak = [0] * n

    obs, _ = env.reset()
    pruned_total = 0
    for t in range(a.num_steps):
        v = np.asarray(obs, dtype=np.float32)
        base = heur.act(env.env.observations) if heur is not None else [1] * n
        act = list(base)
        for i, c in enumerate(cells):
            if c in anchor_set:
                act[i] = 1
                continue
            idle = (v[iprb[i]] < a.thr) and (v[irlf[i]] <= 0)
            idle_streak[i] = idle_streak[i] + 1 if idle else 0
            if a.mode == "prune":
                # only ever turn a heuristic-ON cell OFF; never add cells
                if base[i] == 1 and idle and idle_streak[i] > a.grace:
                    act[i] = 0; pruned_total += 1
            else:  # reactive
                act[i] = 0 if (idle and idle_streak[i] > a.grace) else 1
        obs, r, term, trunc, _ = env.step(act)
        print(f"step {t:3d}  base_on={sum(base)}  act_on={sum(act)}  action={act}  reward={r:.1f}")
        if term or trunc:
            break

    folder = getattr(env.env, "sim_path", None) or sorted(glob.glob("output/*/"), key=os.path.getmtime)[-1]
    env.close()
    print(f"\nmode={a.mode} thr={a.thr} grace={a.grace}  cells pruned/slept total={pruned_total}")
    print(f"Run folder: {folder}")


if __name__ == "__main__":
    main()
