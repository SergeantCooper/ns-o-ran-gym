#!/usr/bin/env python3
"""eval_rl.py - run a trained SB3 policy on the twin (deterministic) for one
episode and report where its output landed, so it can be scored with plot_energy
and overlaid on the tradeoff figure (plot_frontier.py / plot_summary.py).
"""
import argparse
import glob
import os
import sys

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from es_wrapper import EnergySavingRLEnv       # noqa: E402
from train_ppo import NormObs                  # noqa: E402


def newest_output():
    fs = sorted(glob.glob("output/*/"), key=os.path.getmtime)
    return fs[-1] if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rl/ppo_final.zip")
    ap.add_argument("--stats", default="rl/obs_stats.npz")
    ap.add_argument("--ns3_path", default="/workspace/ns-3-mmwave-oran")
    ap.add_argument("--config", default="src/environments/scenario_configurations/es_use_case.json")
    ap.add_argument("--num_steps", type=int, default=58)
    a = ap.parse_args()

    s = np.load(a.stats)
    rl = EnergySavingRLEnv(ns3_path=a.ns3_path, config=a.config, reward_mode="power")
    env = NormObs(rl, s["mean"], s["std"])
    model = PPO.load(a.model, device="cpu")

    obs, _ = env.reset()
    for t in range(a.num_steps):
        act, _ = model.predict(obs, deterministic=True)
        bits = [int(x) for x in np.asarray(act).reshape(-1)]
        bits[0] = 1                              # anchor forced ON (wrapper applies this)
        obs, r, terminated, truncated, _ = env.step(act)
        print(f"step {t:3d}  action={bits}  cells_on={sum(bits)}  reward={r:.2f}")
        if terminated or truncated:
            break

    folder = getattr(rl.env, "sim_path", None) or newest_output()
    env.close()
    print(f"\nRL run folder: {folder}")
    print(f"Score it:  /workspace/.venv/bin/python plot_energy.py {folder}")


if __name__ == "__main__":
    main()
