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
    NaN/inf-scrubbed. Normalize at train time (see rl/train_bc.py obs stats).
  * Reward (reward_mode='power', the RL default): a power-model tradeoff
        reward = throughput_Mbps - w_energy * power_kW - w_rlf * RLF
    computed from the observation with the same power model as plot_energy, so
    the agent optimizes the same energy-vs-QoS axes the scoreboard plots.
  * Episode termination: the base env never sets terminated/truncated, so we
    truncate after `max_episode_steps` (kept safely below the sim length) OR when
    is_simulation_over() -> this gives PPO clean fixed-length episodes with proper
    resets (a fresh ns-3 launch each episode) instead of stepping a dead sim.

EnergySavingEnv itself defines no gym spaces, so this wrapper supplies them.
"""
import json
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from environments.es_env import EnergySavingEnv  # noqa: E402


class EnergySavingRLEnv(gym.Env):
    """Gymnasium env: MultiBinary(7) actions, Box(len(columns_state)) observations."""

    metadata = {"render_modes": []}

    def __init__(self, ns3_path, config, output_folder="output", optimized=True,
                 anchor_idx=0, reward_mode="power", max_episode_steps=58,
                 p_static=600.0, alpha=400.0, w_energy=1.0, w_rlf=2.0,
                 idle_thr=3.0, shape_coef=0.3, gamma=0.99):
        super().__init__()
        with open(config) as f:
            cfg = json.load(f)
        self.env = EnergySavingEnv(
            ns3_path=ns3_path, scenario_configuration=cfg,
            output_folder=output_folder, optimized=optimized, do_heuristic=True,
        )
        self.columns_state = list(self.env.columns_state)
        self.cell_list = list(self.env.cellList)
        self.n_cells = len(self.cell_list)
        self.anchor_idx = anchor_idx
        self.max_episode_steps = int(max_episode_steps)
        self._t = 0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.columns_state),), dtype=np.float32)
        self.action_space = spaces.MultiBinary(self.n_cells)
        self._last_vec = np.zeros(len(self.columns_state), dtype=np.float32)

        self.reward_mode = reward_mode
        self.p_static, self.alpha = p_static, alpha
        self.w_energy, self.w_rlf = w_energy, w_rlf
        self.idle_thr, self.shape_coef, self.gamma = idle_thr, shape_coef, gamma
        self._prev_phi = 0.0   # potential-based shaping state (reward_mode='power_shaped')
        cs = self.columns_state
        self._i_thr = cs.index("SUM_QosFlow.PdcpPduVolumeDL_Filter")
        self._i_rlf = cs.index("SUM_RLF_VALUE")
        self._i_prb = [cs.index(f"RRU_PRBTOTDL_{c}") for c in self.cell_list]

    def _vec(self, raw):
        try:
            arr = np.asarray(raw[0], dtype=np.float32)
        except Exception:
            arr = self._last_vec
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        self._last_vec = arr
        return arr

    def _power_reward(self, vec, action_bits):
        thr_mbps = float(vec[self._i_thr]) * 10.0 / 1e6
        power_w = 0.0
        for i in range(self.n_cells):
            if action_bits[i] == 1:
                util = max(float(vec[self._i_prb[i]]), 0.0) / 100.0
                power_w += self.p_static + self.alpha * util
        rlf = float(vec[self._i_rlf])
        return thr_mbps - self.w_energy * (power_w / 1000.0) - self.w_rlf * rlf

    def _potential(self, vec, action_bits):
        """Potential Phi(s) = -(# ON non-anchor cells carrying ~no traffic). Higher (less
        negative) when idle cells are asleep. Potential-based shaping term gamma*Phi(s')-Phi(s)
        gives an immediate, per-cell reward for sleeping an idle cell WITHOUT changing the
        optimal policy (Ng et al. 1999) - it just makes the credit dense instead of buried in
        the noisy global energy/RLF signal, which is what stalled learning."""
        idle_on = sum(1 for i in range(self.n_cells)
                      if action_bits[i] == 1 and i != self.anchor_idx
                      and float(vec[self._i_prb[i]]) < self.idle_thr)
        return -float(idle_on)

    def _sim_over(self):
        try:
            return bool(self.env.is_simulation_over())
        except Exception:
            return False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._prev_phi = 0.0
        obs, info = self.env.reset()
        return self._vec(obs), (info or {})

    def step(self, action):
        a = [int(x) for x in np.asarray(action).reshape(-1)[: self.n_cells]]
        a[self.anchor_idx] = 1                            # anchor never sleeps
        obs, env_reward, terminated, truncated, info = self.env.step(a)
        self._t += 1
        vec = self._vec(obs)
        if self.reward_mode in ("power", "power_shaped"):
            reward = self._power_reward(vec, a)
            if self.reward_mode == "power_shaped":       # + potential-based idle-sleep credit
                phi = self._potential(vec, a)
                reward += self.shape_coef * (self.gamma * phi - self._prev_phi)
                self._prev_phi = phi
        else:
            reward = float(env_reward)
        truncated = bool(truncated or self._t >= self.max_episode_steps or self._sim_over())
        info = {**(info or {}), "env_reward": float(env_reward)}
        return vec, float(reward), bool(terminated), truncated, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def make_env(ns3_path, config, **kw):
    return EnergySavingRLEnv(ns3_path=ns3_path, config=config, **kw)
