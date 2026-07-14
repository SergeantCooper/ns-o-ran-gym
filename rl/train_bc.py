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
    # critic warm-up (must match the PPO run's reward, else advantages start wrong)
    ap.add_argument("--ep_len", type=int, default=58, help="steps per demo episode (for return computation)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--w_energy", type=float, default=1.0)
    ap.add_argument("--w_rlf", type=float, default=2.0)
    ap.add_argument("--value_epochs", type=int, default=300)
    args = ap.parse_args()

    d = np.load(args.demos, allow_pickle=True)
    obs = d["observations"].astype(np.float32)          # (N, 61)
    act = d["actions"].astype(np.float32)               # (N, 7) in {0,1}
    cols = list(d["columns"])
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

    # ---- critic warm-up: fit the value net to demo returns -------------------
    # Why: PPO warm-started from BC has a RANDOM critic -> first updates use
    # garbage advantages and destroy the good policy (we observed exactly this
    # collapse: ep_rew_mean -6.4 -> -30). Pre-fitting V(s) to the demos' returns
    # (under the SAME reward PPO optimizes) makes the first advantages sane.
    # Policy and value nets are separate (MlpPolicy default), so this does NOT
    # touch the cloned policy - we optimize only value-side params.
    cells = [int(c.split("_")[-1]) for c in cols if c.startswith("RRU_PRBTOTDL_")]
    i_thr = cols.index("SUM_QosFlow.PdcpPduVolumeDL_Filter")
    i_rlf = cols.index("SUM_RLF_VALUE")
    i_prb = [cols.index(f"RRU_PRBTOTDL_{c}") for c in cells]

    def step_reward(res_obs, a):                        # mirrors es_wrapper._power_reward
        thr = float(res_obs[i_thr]) * 10.0 / 1e6
        p = sum(600.0 + 400.0 * max(float(res_obs[i_prb[i]]), 0.0) / 100.0
                for i in range(n_act) if a[i] >= 0.5)
        return thr - args.w_energy * (p / 1000.0) - args.w_rlf * float(res_obs[i_rlf])

    returns = np.zeros(N, np.float32)
    for s in range(0, N, args.ep_len):                  # per contiguous episode
        e = min(s + args.ep_len, N)
        r = [step_reward(obs[t + 1] if (t + 1) < e else obs[t], act[t]) for t in range(s, e)]
        G = 0.0
        for j in range(len(r) - 1, -1, -1):
            G = r[j] + args.gamma * G
            returns[s + j] = G
    print(f"\ncritic warm-up: demo returns  mean={returns.mean():.2f}  "
          f"min={returns.min():.2f}  max={returns.max():.2f}")

    vparams = (list(policy.mlp_extractor.value_net.parameters())
               + list(policy.value_net.parameters()))
    vopt = torch.optim.Adam(vparams, lr=1e-3)
    Xr = torch.as_tensor(returns)
    for ep in range(args.value_epochs):
        perm = torch.randperm(N)
        vrun = 0.0
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch]
            v = policy.predict_values(Xo[idx]).squeeze(-1)
            vloss = torch.nn.functional.mse_loss(v, Xr[idx])
            vopt.zero_grad(); vloss.backward(); vopt.step()
            vrun += vloss.item() * len(idx)
        if ep == 0 or (ep + 1) % 100 == 0:
            with torch.no_grad():
                vpred = policy.predict_values(Xo).squeeze(-1).numpy()
            print(f"  value epoch {ep+1:3d}  mse {vrun/N:9.3f}  "
                  f"corr {np.corrcoef(vpred, returns)[0,1]:.3f}")
    # confirm the BC policy is still intact after the value-only fit
    with torch.no_grad():
        pred, _ = model.predict(obs_n, deterministic=True)
        pred = np.asarray(pred).astype(int); pred[:, 0] = 1
        print(f"BC policy exact_match after critic warm-up: "
              f"{float((pred == act.astype(int)).all(1).mean()):.3f}")

    model.save(args.out)
    print(f"\nsaved BC-pretrained PPO (+ warm critic) -> {args.out}   (+ obs stats -> {args.stats})")


if __name__ == "__main__":
    main()
