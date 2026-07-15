#!/usr/bin/env python3
"""train_ppo.py - PPO fine-tuning on the true (power-model) reward, warm-started
from the BC policy. Live ns-3 rollouts (slow), so training is bounded by wall-clock
(--max_hours) rather than #steps. Logs episode reward (Monitor -> ep_rew_mean) so
progress is observable, checkpoints periodically, and reuses BC's fixed observation
normalization for consistency.
"""
import argparse
import os
import sys
import time

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback

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


class TimeLimit(BaseCallback):
    """Stop training after a wall-clock budget (the simulator, not #steps, is our cost)."""
    def __init__(self, max_seconds):
        super().__init__()
        self.max_seconds = max_seconds
        self._t0 = None

    def _on_training_start(self):
        self._t0 = time.time()

    def _on_step(self):
        if self.max_seconds > 0 and (time.time() - self._t0) >= self.max_seconds:
            print(f"[TimeLimit] reached {self.max_seconds / 3600:.2f} h budget -> stopping")
            return False
        return True


def build_env(ns3_path, config, mean, std, reward_kw):
    # Monitor -> episode-reward logging (ep_rew_mean); NormObs -> fixed obs normalization.
    return DummyVecEnv([lambda: Monitor(NormObs(
        EnergySavingRLEnv(ns3_path=ns3_path, config=config, **reward_kw),
        mean, std))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns3_path", default="/workspace/ns-3-mmwave-oran")
    ap.add_argument("--config", default="src/environments/scenario_configurations/es_use_case.json")
    ap.add_argument("--bc", default="rl/bc_ppo.zip")
    ap.add_argument("--stats", default="rl/obs_stats.npz")
    ap.add_argument("--out", default="rl/ppo_final.zip")
    ap.add_argument("--timesteps", type=int, default=640, help="step budget ~= 4 h at ~23 s/step (10 rollouts of 64)")
    ap.add_argument("--max_hours", type=float, default=0.0, help="wall-clock cap; 0 = NO time-cancel (run to --timesteps)")
    ap.add_argument("--w_energy", type=float, default=1.0)
    ap.add_argument("--w_rlf", type=float, default=2.0)
    ap.add_argument("--reward_mode", default="power_shaped",
                    help="'power' (global) or 'power_shaped' (adds potential-based idle-sleep credit)")
    ap.add_argument("--shape_coef", type=float, default=0.3, help="reward-shaping strength")
    ap.add_argument("--ent_coef", type=float, default=0.0, help="entropy bonus (exploration)")
    args = ap.parse_args()

    s = np.load(args.stats)
    env = build_env(args.ns3_path, args.config, s["mean"], s["std"],
                    dict(w_energy=args.w_energy, w_rlf=args.w_rlf,
                         reward_mode=args.reward_mode, shape_coef=args.shape_coef))
    model = PPO.load(args.bc, env=env, device="cpu")
    model.verbose = 1                        # print rollout table incl. ep_rew_mean
    model.ent_coef = args.ent_coef           # exploration nudge
    cbs = CallbackList([
        CheckpointCallback(save_freq=max(1, model.n_steps), save_path="rl/ckpt", name_prefix="ppo"),
        TimeLimit(args.max_hours * 3600),
    ])
    print(f"PPO fine-tune: n_steps={model.n_steps}, ent_coef={args.ent_coef}, "
          f"cap={args.max_hours} h (time-limited; live sim, slow)")
    model.learn(total_timesteps=args.timesteps, callback=cbs, progress_bar=False)
    model.save(args.out)
    env.close()
    print(f"saved fine-tuned PPO -> {args.out}")


if __name__ == "__main__":
    main()
