"""Regression test for the empty-KPM-window crash in EnergySavingEnv.

When a 100 ms control window has no matching NR+LTE UE records, datalake.read_kpms
returns None. Previously _get_obs iterated over None (raising TypeError) and left
self.observations as its initial [] list, so _compute_reward then raised
`TypeError: list indices must be integers or slices, not list`. The fix keeps
self.observations a valid 1-row DataFrame (cached last-good, else zero-filled).
"""
from unittest.mock import patch

import pandas as pd

from nsoran.ns_env import NsOranEnv
from environments.es_env import EnergySavingEnv


def _make_env():
    cfg = {"simTime": [10], "ues": [3]}
    # Skip the heavy ns-3 configure/build the real __init__ triggers via setup_sim().
    with patch.object(NsOranEnv, "setup_sim", lambda self: None):
        env = EnergySavingEnv(
            ns3_path="/tmp",
            scenario_configuration=cfg,
            output_folder="/tmp",
            optimized=True,
        )
    # _update_cell_states reads live sim state we don't have in a unit test.
    env._update_cell_states = lambda: None

    class _FakeDatalake:
        def read_kpms(self, timestamp, cols):
            return None  # simulate an empty window

        def insert_data(self, table, row):
            return None

    env.datalake = _FakeDatalake()
    env.last_timestamp = 0
    env.num_steps = 0
    return env


def test_empty_window_first_step_zero_filled():
    env = _make_env()
    assert env._last_obs_df is None

    obs = env._get_obs()
    assert isinstance(obs, list) and len(obs) == 1
    assert len(obs[0]) == len(env.columns_state)
    assert all(v == 0.0 for v in obs[0])
    # self.observations must be a valid DataFrame now, not the initial [] list.
    assert isinstance(env.observations, pd.DataFrame)

    # _compute_reward must not raise and must return a numeric reward (0 here).
    reward = env._compute_reward()
    assert isinstance(reward, float)
    assert reward == 0.0


def test_empty_window_reuses_last_good_obs():
    env = _make_env()
    # Seed a "good" observation as if a previous non-empty step had produced it.
    good = pd.DataFrame(
        [{c: (5.0 if c in env.columns_reward else 1.0) for c in env.columns_state}]
    )
    env._last_obs_df = good

    obs = env._get_obs()
    assert len(obs[0]) == len(env.columns_state)
    # The cached frame is reused rather than zero-filled.
    assert env.observations is good
    assert obs[0] == tuple(good[env.columns_state].iloc[0].values)

    reward = env._compute_reward()
    assert isinstance(reward, float)
