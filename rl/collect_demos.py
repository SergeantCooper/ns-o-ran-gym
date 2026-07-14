#!/usr/bin/env python3
"""collect_demos.py - run the TwinHeuristic expert on the twin and log
(observation, action) pairs for Behavior Cloning.

This is the one-time (simulator-bound, ~10-12 s/step) data collection: the expert
drives the environment; at each step we record the observation vector (the 61
state features the RL agent will see) and the expert's 7-bit ON/OFF action
(1 = ON), which becomes the BC target. Saved as a compressed .npz.

Run several loads for coverage, then combine, e.g.:
    PYTHONPATH=src python rl/collect_demos.py --config <cfg_ues3> --out rl/demos_ues3.npz
    PYTHONPATH=src python rl/collect_demos.py --config <cfg_ues6> --out rl/demos_ues6.npz
    python rl/combine_demos.py rl/demos_ues*.npz --out rl/demos.npz   # (trivial np.concatenate)
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))   # environments / nsoran
sys.path.insert(0, os.path.join(_HERE, ".."))          # heuristic_twin

from es_wrapper import EnergySavingRLEnv          # noqa: E402
from heuristic_twin import TwinHeuristic          # noqa: E402


def collect(ns3_path, config, num_steps, out, anchor):
    env = EnergySavingRLEnv(ns3_path=ns3_path, config=config)
    policy = TwinHeuristic(cell_list=env.cell_list, anchor_cells=tuple(anchor))
    obs_vec, _ = env.reset()
    OBS, ACT = [], []
    for t in range(num_steps):
        df = env.env.observations                 # wide DataFrame the expert reads
        action = policy.act(df)                   # 7-bit list, 1 = ON
        OBS.append(np.asarray(obs_vec, dtype=np.float32).copy())
        ACT.append(np.asarray(action, dtype=np.int8))
        obs_vec, reward, terminated, truncated, _ = env.step(action)
        print(f"step {t:3d}  action={action}  cells_on={sum(action)}  reward={reward:.0f}")
        if terminated or truncated:
            break
    env.close()
    OBS = np.asarray(OBS, dtype=np.float32)
    ACT = np.asarray(ACT, dtype=np.int8)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, observations=OBS, actions=ACT,
                        columns=np.array(env.columns_state))
    print(f"\nsaved {len(OBS)} transitions -> {out}   obs={OBS.shape}  act={ACT.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ns3_path", default="/workspace/ns-3-mmwave-oran")
    p.add_argument("--config", default="src/environments/scenario_configurations/es_use_case.json")
    p.add_argument("--num_steps", type=int, default=58)
    p.add_argument("--out", default="rl/demos.npz")
    p.add_argument("--anchor", type=int, nargs="+", default=[2])
    a = p.parse_args()
    collect(a.ns3_path, a.config, a.num_steps, a.out, a.anchor)
