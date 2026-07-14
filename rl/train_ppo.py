#!/usr/bin/env python3
"""train_ppo.py - PPO fine-tuning on the true (power-model) reward, warm-started
from the BC policy. This is the step that can SURPASS the heuristic.

Live ns-3 rollouts => slow (~10-12 s per step). Uses a SMALL n_steps so each PPO
update is affordable, and reuses BC's fixed observation normalization so the
warm-started policy sees observations on the scale it was trained on.
"""
import argparse
import os
import sys

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from es_wrapper import EnergySavingRLEnv  # noqa: E402


class NormObs(gym.ObservationWrapper):
    """Fixed (BC-derived) observation normalization, shared with training."""
    def __init__(self, env, mean, std):
        super().__init__(env)
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    def observation(self, obs):
        return np.clip((obs - self.mean) / self.std, -10, 10).astype(np.float32)


def build_env(ns3_path, config, mean, std, reward_kw):
    return DummyVecEnv([lambda: NormObs(
        EnergySavingRLEnv(ns3_path=ns3_path, config=config, reward_mode="power", **reward_kw),
        mean, std)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns3_path", default="/workspace/ns-3-mmwave-oran")
    ap.add_argument("--config", default="src/environments/scenario_configurations/es_use_case.json")
    ap.add_argument("--bc", default="rl/bc_ppo.zip")
    ap.add_argument("--stats", default="rl/obs_stats.npz")
    ap.add_argument("--out", default="rl/ppo_final.zip")
    ap.add_argument("--timesteps", type=int, default=512)
    ap.add_argument("--w_energy", type=float, default=1.0)
    ap.add_argument("--w_rlf", type=float, default=2.0)
    ap.add_argument("--ent_coef", type=float, default=0.0, help="entropy bonus (exploration)")
    args = ap.parse_args()

    s = np.load(args.stats)
    env = build_env(args.ns3_path, args.config, s["mean"], s["std"],
                    dict(w_energy=args.w_energy, w_rlf=args.w_rlf))
    # warm-start: load the BC-pretrained policy + hyperparams, attach the live env
    model = PPO.load(args.bc, env=env, device="cpu")
    model.verbose = 1                        # show per-rollout reward trend
    model.ent_coef = args.ent_coef           # exploration nudge to escape the BC/heuristic behavior
    ckpt = CheckpointCallback(save_freq=max(1, model.n_steps),
                              save_path="rl/ckpt", name_prefix="ppo")
    print(f"PPO fine-tune: n_steps={model.n_steps}, total_timesteps={args.timesteps} "
          f"(~{args.timesteps} live sim steps; slow)")
    model.learn(total_timesteps=args.timesteps, callback=ckpt, progress_bar=False)
    model.save(args.out)
    env.close()
    print(f"saved fine-tuned PPO -> {args.out}")


if __name__ == "__main__":
    main()
