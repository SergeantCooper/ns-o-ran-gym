#!/usr/bin/env python3
"""
heuristic_twin.py - Path B: run the energy-saving heuristic ON the ns-O-RAN twin.

This reuses the heuristic's decision logic but adapts it to what scenario-three
actually provides: only DL PRB, throughput, a 64QAM-ratio channel proxy and RLF
per cell; on/off actuation only (no path/carrier sub-states); "interval" = the
100 ms control step; and hard-assigned roles (one anchor gNB that never sleeps,
the rest are capacity gNBs eligible to sleep). Guardrails are REAL here: we watch
the next step's throughput/RLF and revert a sleep that hurt service.

    pip3 install gymnasium posix-ipc sem pandas numpy
    PYTHONPATH=src python3 examples/heuristic_twin.py \
        --ns3_path ~/dev/ns-o-ran-ns3-mmwave \
        --config src/environments/scenario_configurations/es_use_case.json \
        --num_steps 120 --optimized

Tip: set "bsOn":[7],"bsOff":[0] in the config so the network starts fully on and
the heuristic decides what to sleep. Afterwards, plot with:
    python3 plot_energy.py output/<uuid> --labels heuristic --p-on 200 --p-off 0
(<uuid> = newest folder: ls -dt output/*/ | head -1)

Self-test the decision logic without ns-3:
    python3 -c "import heuristic_twin as h; h._selftest()"
"""
import argparse
import json
import os

CELL_LIST = [2, 3, 4, 5, 6, 7, 8]

DEFAULT_CFG = dict(
    w_prb=0.6, w_thr=0.4,              # LoadScore weights: DL-PRB + throughput, per-cell normalized.
                                       # active-UE and UL-PRB deliberately excluded: at the twin's 100 ms
                                       # granularity active-UE counts camped/idle UEs (idle cells show
                                       # 1-3 UEs with 0 traffic), so per-cell normalization makes it spike
                                       # and reset the sleep hysteresis; UL-PRB is identically 0 here.
    blend_avg=0.6, blend_peak=0.4,     # Step 5 blend
    thr_capacity=20.0, thr_coverage=15.0,   # Step 5 sleep thresholds (on normalized LoadScore)
    prb_low_abs=20.0,                  # absolute PRB% floor required to sleep (robust to warmup)
    T_sleep=4, T_cooldown=8, T_guardrail=4, # Steps 6/12/14, counted in CONTROL STEPS
    wake_neigh_prb=90.0,               # Step 9: wake a slept cell if an ON neighbor is genuinely
                                       # congested. Spec says 75%, but in this scenario a normally-busy
                                       # anchor-adjacent cell sits at ~75-84% PRB, which would wake idle
                                       # neighbours every step; 90% reserves wake-up for real overload.
    neigh_future_max=70.0, anchor_spare=50.0,   # Step 7 feasibility
    g_thr_loss=5.0, g_rlf_rise=5.0,    # Step 14 guardrails
)


class TwinHeuristic:
    def __init__(self, cell_list=CELL_LIST, anchor_cells=(2,), cfg=None):
        self.cells = list(cell_list)
        self.anchor = set(anchor_cells)
        self.role = {c: ("coverage" if c in self.anchor else "capacity") for c in self.cells}
        self.cfg = dict(DEFAULT_CFG); self.cfg.update(cfg or {})
        inf = float("inf")
        self.mn = {c: {"prb": inf, "thr": inf} for c in self.cells}
        self.mx = {c: {"prb": -inf, "thr": -inf} for c in self.cells}
        self.low_streak = {c: 0 for c in self.cells}
        self.cooldown = {c: 0 for c in self.cells}
        self.state = {c: 1 for c in self.cells}         # 1 = ON, 0 = OFF (sleep)
        self.last_slept = set()
        self.pre_sleep_thr = 0.0    # aggregate throughput just BEFORE the last new sleep (spec 14 baseline)
        self.slept_thr = {}         # per-cell throughput captured when each cell was slept
        self.rlf_ref = None
        self.warmup = self.cfg["T_sleep"]

    def _v(self, obs, col):
        try:
            return float(obs[col].iloc[0])
        except Exception:
            return 0.0

    def _norm(self, c, k, v):
        lo, hi = self.mn[c][k], self.mx[c][k]
        return 100.0 * (v - lo) / (hi - lo) if hi > lo else 0.0

    def _neighbor_feasible(self, c, prb, live_state):
        """Step 7: offload c's PRB onto currently-ON neighbors; each must stay < 70%,
        and at least one must be a capacity cell (or an anchor with clear spare)."""
        on = [n for n in self.cells if n != c and live_state[n] == 1]
        if not on:
            return False
        offload = prb[c] / len(on)
        if any(prb[n] + offload >= self.cfg["neigh_future_max"] for n in on):
            return False
        for n in on:
            if self.role[n] == "capacity":
                return True
            if self.role[n] == "coverage" and prb[n] + offload < self.cfg["anchor_spare"]:
                return True
        return False

    def act(self, obs):
        # No usable observation yet -> keep everything on (safe).
        if not hasattr(obs, "columns"):
            return [1] * len(self.cells)

        prb, thr, rlf = {}, {}, {}
        for c in self.cells:
            # RRU_PRBTOTDL_{c} is ALREADY DL PRB utilisation % ((PrbUsedDl/139)*100),
            # produced by es_env.offline_training_preprocessing - NOT a PRB total. Use it
            # directly: the old 100*used/tot cancelled to a constant ~139, so every cell
            # (even fully idle ones) looked busy and the policy never slept anything.
            prb[c] = self._v(obs, f"RRU_PRBTOTDL_{c}")
            thr[c] = self._v(obs, f"QosFlow.PdcpPduVolumeDL_Filter_{c}")
            rlf[c] = self._v(obs, f"RLF_VALUE_{c}")
        sum_thr = self._v(obs, "SUM_QosFlow.PdcpPduVolumeDL_Filter")
        sum_rlf = self._v(obs, "SUM_RLF_VALUE")

        # Step 14 guardrail: this observation reflects last step's action. Compare service
        # against the throughput we had JUST BEFORE the last new sleep (spec: "next interval
        # after applying") - NOT the all-time peak, which any bursty dip would breach and so
        # revert every sleep. And only wake cells whose sleep could plausibly have hurt
        # service: a cell carrying ~0 traffic before sleeping cannot have caused a drop, so
        # it stays asleep (prevents a genuinely-idle cell being reverted as collateral).
        if self.last_slept:
            thr_drop = (self.pre_sleep_thr > 0
                        and sum_thr < (1 - self.cfg["g_thr_loss"] / 100.0) * self.pre_sleep_thr)
            rlf_rise = (self.rlf_ref is not None and sum_rlf > self.rlf_ref + self.cfg["g_rlf_rise"])
            if thr_drop or rlf_rise:
                for c in self.last_slept:
                    if rlf_rise or self.slept_thr.get(c, 0.0) > 0.0:
                        self.state[c] = 1
                        self.cooldown[c] = self.cfg["T_guardrail"]
        self.rlf_ref = sum_rlf if self.rlf_ref is None else min(self.rlf_ref, sum_rlf)

        for c in self.cells:                                           # update per-cell min/max
            self.mn[c]["prb"] = min(self.mn[c]["prb"], prb[c]); self.mx[c]["prb"] = max(self.mx[c]["prb"], prb[c])
            self.mn[c]["thr"] = min(self.mn[c]["thr"], thr[c]); self.mx[c]["thr"] = max(self.mx[c]["thr"], thr[c])

        self.warmup = max(0, self.warmup - 1)
        live = dict(self.state)                                        # decisions ripple within the step
        for c in self.cells:
            if self.cooldown[c] > 0:
                self.cooldown[c] -= 1

            pn = self._norm(c, "prb", prb[c]); tn = self._norm(c, "thr", thr[c])
            weighted = self.cfg["w_prb"] * pn + self.cfg["w_thr"] * tn
            load_score = self.cfg["blend_avg"] * weighted + self.cfg["blend_peak"] * max(pn, tn)  # Step 5
            thr_th = self.cfg["thr_coverage"] if self.role[c] == "coverage" else self.cfg["thr_capacity"]
            low = (load_score < thr_th) and (prb[c] < self.cfg["prb_low_abs"])
            self.low_streak[c] = self.low_streak[c] + 1 if low else 0

            # Step 9 wake-up: a sleeping cell comes back if a neighbor gets congested.
            if self.state[c] == 0:
                neigh_hot = any(prb[n] > self.cfg["wake_neigh_prb"]
                                for n in self.cells if n != c and live[n] == 1)
                if neigh_hot:
                    live[c] = 1; self.cooldown[c] = self.cfg["T_cooldown"]; continue

            s = self.state[c]
            can_sleep = (self.role[c] == "capacity" and self.warmup == 0
                         and self.cooldown[c] == 0 and self.low_streak[c] >= self.cfg["T_sleep"])
            if can_sleep and self._neighbor_feasible(c, prb, live):    # Step 8
                s = 0
            elif self.state[c] == 0 and self.low_streak[c] >= self.cfg["T_sleep"]:
                s = 0                                                  # stay asleep while still idle
            else:
                s = 1
            if self.role[c] == "coverage":                            # Step 4: anchor never sleeps
                s = 1
            live[c] = s

        prev_off = self.last_slept
        self.state = live
        self.last_slept = {c for c in self.cells if self.state[c] == 0}
        newly_off = self.last_slept - prev_off
        if newly_off:
            # snapshot service level right before these cells go dark (guardrail baseline)
            self.pre_sleep_thr = sum_thr
            for c in newly_off:
                self.slept_thr[c] = thr[c]
        return [self.state[c] for c in self.cells]


def run(args):
    from environments.es_env import EnergySavingEnv          # lazy import (needs ns-3 stack)

    with open(args.config) as f:
        scenario_configuration = json.load(f)
    # The empty-window crash fix now lives upstream in EnergySavingEnv._get_obs, so no
    # subclass/monkey-patch is needed here anymore.
    env = EnergySavingEnv(ns3_path=args.ns3_path,
                          scenario_configuration=scenario_configuration,
                          output_folder=args.output_folder,
                          optimized=args.optimized,
                          do_heuristic=True)
    policy = TwinHeuristic(cell_list=env.cellList, anchor_cells=tuple(args.anchor))
    obs, info = env.reset()
    for step in range(args.num_steps):
        seen = env.observations                               # wide df the policy decides on
        action = policy.act(seen)
        obs, reward, terminated, truncated, info = env.step(action)
        on = sum(policy.state.values())
        print(f"step {step:3d}  action={action}  cells_on={on}  reward={reward:.0f}")
        if os.environ.get("HEUR_DEBUG"):                      # per-cell decision trace for tuning
            prb = {c: round(policy._v(seen, f"RRU_PRBTOTDL_{c}")) for c in policy.cells}
            print(f"          prb%={prb}  low_streak={dict(policy.low_streak)}  cooldown={dict(policy.cooldown)}")
        if terminated or truncated:
            break
    env.close()
    print("\nDone. Find the run folder with:  ls -dt output/*/ | head -1")
    print("Then plot:  python3 plot_energy.py <that_folder> --labels heuristic --p-on 200 --p-off 0")


def _selftest():
    """Exercise the policy with fake observations - no ns-3 needed."""
    import pandas as pd

    def make_obs(prb_by_cell, thr_scale=1.0, rlf=0.0):
        row = {}
        for c in CELL_LIST:
            p = prb_by_cell.get(c, 0.0)
            row[f"RRU_PRBTOTDL_{c}"] = p  # DL PRB utilisation % (matches es_env semantics)
            row[f"RRU.PrbUsedDl_{c}"] = p
            row[f"QosFlow.PdcpPduVolumeDL_Filter_{c}"] = p * 1000 * thr_scale
            row[f"TB_TOTNBRDLINITIAL_64QAM_RATIO_{c}"] = 0.6
            row[f"RLF_VALUE_{c}"] = rlf
        row["SUM_QosFlow.PdcpPduVolumeDL_Filter"] = sum(
            row[f"QosFlow.PdcpPduVolumeDL_Filter_{c}"] for c in CELL_LIST)
        row["SUM_RLF_VALUE"] = rlf * len(CELL_LIST)
        return pd.DataFrame([row])

    pol = TwinHeuristic()
    print("roles:", pol.role)
    print("--- capacity cells idle (anchor=2 busy) -> expect 3..8 to sleep after hysteresis ---")
    for t in range(10):
        obs = make_obs({2: 60.0, 3: 5.0, 4: 5.0, 5: 5.0, 6: 5.0, 7: 5.0, 8: 5.0})
        a = pol.act(obs)
        assert len(a) == 7 and all(x in (0, 1) for x in a), a
        assert a[0] == 1, "anchor must stay ON"
        print(f"  t={t} action={a} cells_on={sum(a)}")
    print("--- load returns on all cells -> expect everything ON ---")
    for t in range(4):
        obs = make_obs({c: 80.0 for c in CELL_LIST})
        a = pol.act(obs)
        print(f"  t={t} action={a} cells_on={sum(a)}")
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ns3_path", default="/workspace/ns-3-mmwave-oran")
    p.add_argument("--config", default="src/environments/scenario_configurations/es_use_case.json")
    p.add_argument("--output_folder", default="output")
    p.add_argument("--num_steps", type=int, default=120)
    p.add_argument("--anchor", type=int, nargs="+", default=[2], help="cell id(s) that never sleep")
    p.add_argument("--optimized", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    _selftest() if args.selftest else run(args)
