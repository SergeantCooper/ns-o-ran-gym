#!/usr/bin/env python3
"""es_wrapper.py - a Gymnasium wrapper exposing EnergySavingEnv to RL (SB3).

Design choices:
  * Action = MultiBinary(7): one on/off bit per mmWave gNB (cells 2..8), 1 = ON.
    The anchor cell (index 0 = cell 2) is FORCED ON every step, matching the
    heuristic expert (coverage anchor never sleeps). We drive the underlying env
    through its do_heuristic=True path, which applies a raw 7-bit action directly
    - the same, already-verified path the TwinHeuristic uses. This avoids the
    Discrete(64) index + ON/OFF-inversion mapping.
  * Observation = Box over the env's `columns_state` (61 features), float32,
    NaN/inf-scrubbed. Normalize at train time with SB3 VecNormalize.
  * Reward (reward_mode='power', the default for RL): a power-model tradeoff
        reward = throughput_Mbps - w_energy * power_kW - w_rlf * RLF
    computed from the observation using the same power model as plot_energy
    (P_on = P_static + alpha*util; slept cells draw 0). This makes the agent
    optimize REAL energy-vs-QoS - i.e. the exact axes of the tradeoff scoreboard,
    so "beating the heuristic" means beating it on the plotted metric.
    reward_mode='env' falls back to the environment's native reward.

EnergySavingEnv itself defines no gym spaces, so this wrapper supplies them.
"""
import json
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# make `environments`/`nsoran` importable regardless of CWD
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from environments.es_env import EnergySavingEnv  # noqa: E402


class EnergySavingRLEnv(gym.Env):
    """Gymnasium env: MultiBinary(7) actions, Box(len(columns_state)) observations."""

    metadata = {"render_modes": []}

    def __init__(self, ns3_path, config, output_folder="output", optimized=True,
                 anchor_idx=0, reward_mode="power",
                 p_static=600.0, alpha=400.0, w_energy=1.0, w_rlf=2.0):
        super().__init__()
        with open(config) as f:
            cfg = json.load(f)
        # do_heuristic=True => env applies a raw 7-bit [s2..s8] action (1=ON) directly.
        self.env = EnergySavingEnv(
            ns3_path=ns3_path, scenario_configuration=cfg,
            output_folder=output_folder, optimized=optimized, do_heuristic=True,
        )
        self.columns_state = list(self.env.columns_state)
        self.cell_list = list(self.env.cellList)          # [2..8]
        self.n_cells = len(self.cell_list)                # 7
        self.anchor_idx = anchor_idx                      # cell 2 -> index 0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.columns_state),), dtype=np.float32)
        self.action_space = spaces.MultiBinary(self.n_cells)
        self._last_vec = np.zeros(len(self.columns_state), dtype=np.float32)

        # --- reward model ---
        self.reward_mode = reward_mode
        self.p_static, self.alpha = p_static, alpha
        self.w_energy, self.w_rlf = w_energy, w_rlf
        cs = self.columns_state
        self._i_thr = cs.index("SUM_QosFlow.PdcpPduVolumeDL_Filter")
        self._i_rlf = cs.index("SUM_RLF_VALUE")
        self._i_prb = [cs.index(f"RRU_PRBTOTDL_{c}") for c in self.cell_list]

    def _vec(self, raw):
        """env._get_obs() returns [tuple(columns_state values)] -> clean float32 vector."""
        try:
            arr = np.asarray(raw[0], dtype=np.float32)
        except Exception:
            arr = self._last_vec
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        self._last_vec = arr
        return arr

    def _power_reward(self, vec, action_bits):
        """throughput_Mbps - w_energy*power_kW - w_rlf*RLF (power model = plot_energy)."""
        thr_mbps = float(vec[self._i_thr]) * 10.0 / 1e6      # SUM_QosFlow -> Mbps (grafana convention)
        power_w = 0.0
        for i in range(self.n_cells):
            if action_bits[i] == 1:                          # cell ON draws static + load power
                util = max(float(vec[self._i_prb[i]]), 0.0) / 100.0   # RRU_PRBTOTDL is a percentage
                power_w += self.p_static + self.alpha * util
        rlf = float(vec[self._i_rlf])
        return thr_mbps - self.w_energy * (power_w / 1000.0) - self.w_rlf * rlf

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset()
        return self._vec(obs), (info or {})

    def step(self, action):
        a = [int(x) for x in np.asarray(action).reshape(-1)[: self.n_cells]]
        a[self.anchor_idx] = 1                            # anchor never sleeps
        obs, env_reward, terminated, truncated, info = self.env.step(a)
        vec = self._vec(obs)
        reward = self._power_reward(vec, a) if self.reward_mode == "power" else float(env_reward)
        if isinstance(info, dict):
            info = {**info, "env_reward": float(env_reward)}
        return vec, float(reward), bool(terminated), bool(truncated), (info or {})

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def make_env(ns3_path, config, **kw):
    """Factory (handy for SB3 DummyVecEnv([lambda: make_env(...)]))."""
    return EnergySavingRLEnv(ns3_path=ns3_path, config=config, **kw)
