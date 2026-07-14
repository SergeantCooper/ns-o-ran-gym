#!/usr/bin/env python3
"""train_bc.py - Behavior Cloning: supervised pre-training of an SB3 PPO policy
on the expert demonstrations. Offline (no simulator), GPU/CPU-fast.

Approach: construct a PPO("MlpPolicy") over the same spaces (via a spaces-only
stub env so no ns-3 is launched), then maximize the log-likelihood the policy
assigns to the expert's actions (= behavior cloning). Saves:
  * the BC-pretrained PPO model (rl/bc_ppo.zip) - PPO fine-tuning warm-starts from it
  * fixed observation normalization stats (rl/obs_stats.npz) - the PPO env MUST reuse
    these so BC and PPO see observations on the same scale.
"""
import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class _StubEnv(gym.Env):
    """Spaces-only env: lets us build the PPO policy without launching ns-3."""
    def __init__(self, n_obs, n_act):
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = spaces.MultiBinary(n_act)

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, a):
        return np.zeros(self.observation_space.shape, np.float32), 0.0, True, False, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="rl/demos.npz")
    ap.add_argument("--out", default="rl/bc_ppo.zip")
    ap.add_argument("--stats", default="rl/obs_stats.npz")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_steps", type=int, default=64, help="carried into the PPO model for fine-tuning")
    args = ap.parse_args()

    d = np.load(args.demos, allow_pickle=True)
    obs = d["observations"].astype(np.float32)          # (N, 61)
    act = d["actions"].astype(np.float32)               # (N, 7) in {0,1}
    n_obs, n_act = obs.shape[1], act.shape[1]
    print(f"demos: observations {obs.shape}, actions {act.shape}")

    # fixed observation normalization (PPO must reuse these exact stats)
    mean = obs.mean(0)
    std = obs.std(0) + 1e-6
    np.savez(args.stats, mean=mean, std=std)
    obs_n = np.clip((obs - mean) / std, -10, 10).astype(np.float32)

    model = PPO("MlpPolicy", DummyVecEnv([lambda: _StubEnv(n_obs, n_act)]),
                n_steps=args.n_steps, batch_size=args.batch, device="cpu", verbose=0)
    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    Xo = torch.as_tensor(obs_n)
    Xa = torch.as_tensor(act)
    N = len(Xo)
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        running = 0.0
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch]
            _, log_prob, _ = policy.evaluate_actions(Xo[idx], Xa[idx])
            loss = -log_prob.mean()                     # maximize expert-action likelihood
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * len(idx)
        if ep == 0 or (ep + 1) % 50 == 0:
            with torch.no_grad():
                pred, _ = model.predict(obs_n, deterministic=True)
                pred = np.asarray(pred).astype(int)
                pred[:, 0] = 1                          # anchor forced ON (as the wrapper does)
                bit_acc = float((pred == act.astype(int)).mean())
                exact = float((pred == act.astype(int)).all(1).mean())
            print(f"epoch {ep+1:3d}  bc_loss {running/N:7.4f}  bit_acc {bit_acc:.3f}  exact_match {exact:.3f}")
    model.save(args.out)
    print(f"\nsaved BC-pretrained PPO -> {args.out}   (+ obs stats -> {args.stats})")


if __name__ == "__main__":
    main()
