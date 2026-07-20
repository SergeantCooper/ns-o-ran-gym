# Project Context Primer — ns-O-RAN Energy-Saving Twin + RL

*Paste this whole file into a fresh Claude chat to give it complete context on my project so it can
help me discuss/continue it. It captures the full journey, the results, and the reasoning behind the
decisions. Last updated: 2026-07-17.*

---

## 0. How to use this
I'm working on an energy-saving controller for a simulated 5G network (my company assigned it).
The full docs live in the repo (`OVERVIEW.md` = how it works, `RESULTS.md` = findings, `REPRODUCE.md`
= build & run); this file is the narrative + current-state summary for quick context.

---

## 1. The goal
Base-station radios ("gNBs") waste power staying ON when few people use them. I have a **digital
twin** (an ns-3 simulator of a 5G network) and I want a **controller** that decides every 100 ms
which gNBs to put to **sleep** to save energy **without hurting service quality** (throughput /
dropped calls). The company's framing: **an RL model that beats the hand-coded heuristic** on the
energy-vs-quality tradeoff.

**Where we landed:** the heuristic is a **strong baseline**; **plain-reward RL tied it**; and with an
**improved (shaped) reward, RL reached a favourable tradeoff** — it saves **more energy and drops
fewer calls, at slightly less throughput**. So RL is better on energy + reliability, slightly worse
on throughput: a favourable tradeoff, **not** a strict win on all three axes — and we can explain
exactly why (a structural sample-budget limit of RL on a slow simulator).

---

## 2. The system (how it works)
- **Twin:** ns-3 mmWave O-RAN sim, `scenario-three.cc`. 1 LTE anchor cell (always on) + **7 mmWave
  gNBs** (cells 2–8, can sleep) + **6 UEs** (users). Runs ~15–20 s per 100 ms step (CPU-bound; the
  bottleneck). One 58-step episode ≈ 12–20 min; full RL training ≈ 4 h.
- **Control loop:** ns-3 runs with `heuristicType=-1` (external controller) + `useSemaphores=1`.
  Each step: ns-3 writes per-cell KPMs to CSV files → Python gym reads them into a **61-number
  observation** → the controller picks a **7-bit ON/OFF action** → gym writes it to a file → ns-3
  applies it. Coordinated by two POSIX semaphores in `/dev/shm`. Real-time closed loop.
- **Observation (61 features):** per cell — PRB utilisation (how busy), throughput, RLF (dropped
  calls), efficiency KPI, modulation — plus 5 network aggregates.
- **Action:** `MultiBinary(7)`, one ON/OFF bit per gNB; the anchor is forced ON.
- **Custom RL reward (I wrote this):** `reward = throughput_Mbps − 1.0·power_kW − 2.0·RLF` per step,
  where power = Σ over ON cells of `(600 W + 400 W·utilisation)`. In `rl/es_wrapper.py`. A
  **shaped** variant (`reward_mode=power_shaped`) adds dense per-cell credit for sleeping idle cells
  (potential-based; doesn't change the optimum) — this is what produced the final RL result.
- **Energy model / metrics:** `plot_energy.py` — each ON gNB = 600 W static + 400 W·util, sleeping
  ≈ 0 W. "% saved" is vs. all-cells-on. QoS = throughput (Mbps) + RLF (dropped calls, lower better).

---

## 3. What we did (the journey + decisions)
1. **Fixed the environment:** crash on empty KPM windows, a SINR string-vs-int bug, a PRB-utilisation
   bug; corrected the scenario config (`e2nrEnabled`, `bsOn=7`, `heuristicType=-1`).
2. **Understood the built-in heuristic:** ns-3's own `heuristicType=2` is **quota-driven** (sleeps a
   fixed *count* of cells regardless of load) → can't adapt. So the project uses a custom
   **load-adaptive** expert, `heuristic_twin.py` (`TwinHeuristic`).
3. **Deferred GAIL (company's original ask):** GAIL only *imitates* the heuristic (can't beat it) and
   is expensive on a slow sim. The "beating" must come from **RL on the true reward** → plan became
   **BC (imitate) → PPO (surpass)**, GAIL optional. Company accepted.
4. **Built the RL pipeline** (`rl/`): expert demos → **Behaviour Cloning** (100 % match) → **PPO** on
   the reward → evaluate. Stack: Stable-Baselines3 + PyTorch (CPU; the sim, not the NN, is the bottleneck).
5. **Old scenario (smooth day/night traffic) → RL TIED the heuristic** — because the *scenario* had no
   headroom (heuristic already near-optimal on smooth load), not because RL failed.
6. **Redesigned the traffic to be bursty** (3 sharp bursts/episode + always-on baseline user) so the
   controller must adapt in real time (cells-on now swings 2–7).
7. **PPO run #1 COLLAPSED** (reward −6.4 → −30). Diagnosed: warm-starting PPO from the clone leaves its
   **value estimator (critic) random** → early updates wreck the good policy. Classic BC→PPO trap.
8. **Fixed it:** BC now also **pre-trains the critic** on the demos' returns (`train_bc.py`) →
   critic↔returns corr **0.99**, cloned policy stays a perfect copy.
9. **PPO run #2 (warm critic) = TIE** — no collapse, but the greedy policy converged back to the
   heuristic. **Why (structural):** random ON/OFF exploration sleeps idle *and* busy cells together;
   busy-cell failures dominate the reward; with only ~640 affordable trial-steps (slow sim), RL can't
   isolate "sleeping *this idle* cell was good." Plain RL can't capture the prize at this sim speed.
10. **Quantified the headroom:** the heuristic keeps a cell powered with **no traffic ~43 % of the
    time**; a *hindsight* oracle would save 49–63 % at ~no cost — **but** a real controller decides
    *before* seeing the traffic, so sleeping a soon-needed cell drops a call. Headroom is real in
    hindsight, **not causally free**.
11. **Built a pruning controller** (`rl/aggressive_ctl.py`) with a `grace` dial: robustly saves ~19
    pts more energy but at ~8 % less throughput and ~2× dropped calls — a **tunable** energy↔QoS
    frontier, no setting strictly beating the heuristic.
12. **Improved the reward (potential-based shaping)** → this moved RL **off the tie**: the shaped-RL
    policy reaches a **greener + more-reliable** operating point — **+6 pts energy AND ~26 % fewer
    dropped calls, at ~9 % less throughput** (4-seed avg). A **favourable tradeoff**, better on 2 of 3
    axes, but still not a strict all-axis win (the heuristic stays ahead on throughput).
13. **Conclusion:** the heuristic is strong; shaped-RL is a favourable, honest tradeoff; the one real
    blocker to a *strict* win is the simulator's tiny RL sample budget.

---

## 4. Key results (averaged over 4 seeds: 555/777/999/1234)

| Policy | Energy saved | Throughput | RLF (dropped calls) |
|---|---|---|---|
| all cells on | 0 % | 6.10 Mbps | 0.00 |
| **heuristic** (baseline) | ~34 % | ~5.68 Mbps | ~1.20 |
| RL — plain reward | ~34 % | ~5.68 Mbps | ~1.20  ← ties the heuristic |
| **RL — shaped reward** (final) | **~40 %** | ~5.18 Mbps | **~0.90**  ← +energy, +reliability, −throughput |
| pruning (aggressive) | ~53 % | ~5.25 Mbps | ~2.36  ← more energy, worse QoS |

**Bottom line:** plain RL ties; shaped RL is a **favourable tradeoff** (better on energy + reliability,
slightly worse on throughput) — not a strict all-axis win. Per-seed RL energy varies 25–46 %.

---

## 5. Current state (files, artifacts, git)
- **Two git repos** (siblings under `/workspace`): `ns-3-mmwave-oran` (twin, C++, branch
  `energy-saving-twin`) and `ns-o-ran-gym` (gym/RL/analysis, Python, branch `energy-saving-fixes`).
  (`oran-e2sim` = E2 interface library, built once.) A few commits may be unpushed — `git push` both.
- **Trained artifacts (committed):** `rl/ppo_final.zip` (plain reward, ties), `rl/ppo_shaped.zip`
  (shaped reward, the **final** favourable-tradeoff model), `rl/bc_ppo.zip` (BC + warm critic),
  `rl/obs_stats.npz`, `rl/demos.npz`.
- **Figures (computed live from run folders):** `figures/` suite (1 tradeoff scatter, 2 per-metric
  bars, 3 pruning frontier, 4 controller-over-time), plus `summary_tradeoff.png`, `shaped_vs_heuristic.png`.
- **Presentation:** `energy_saving_RL_project.pptx` (built by `build_ppt.py`).
- **Docs:** `OVERVIEW.md`, `RESULTS.md`, `REPRODUCE.md`, `CONTEXT.md` (this file). Older phase notes
  are archived in `work.zip`.
- **Key code:** `heuristic_twin.py` (expert), `rl/es_wrapper.py` (wrapper + reward), `rl/train_bc.py`,
  `rl/train_ppo.py`, `rl/eval_rl.py`, `rl/aggressive_ctl.py`, `plot_energy.py`, `plot_report.py`.
  Env: `src/environments/es_env.py`; base loop `src/nsoran/ns_env.py`; config
  `src/environments/scenario_configurations/es_use_case.json` (6 UEs, seed 555, burst traffic).

---

## 6. Gotchas / constraints (important for reasoning)
- **ns-3 is slow (~15–20 s/step) and single-threaded; a GPU does not help** — *the* binding constraint
  (why RL is sample-starved; training ≈ 4 h).
- Compare energy **per-step**, not per-run total (runs can differ in step count).
- Clean stale semaphores between runs: `rm -f /dev/shm/sem.*`.
- **Never rebuild ns-3 while a live gym/PPO run is going** (it relaunches the binary each episode).
- Each run writes a **new** `output/<uuid>/` folder; newest = `ls -dt output/*/ | head -1`.
- The RL reward is my **custom** power-model reward (`rl/es_wrapper.py`), not the env's built-in proxy
  (which is what prints during a plain heuristic run — a common source of confusion).
- Figures/models are seed-averaged (555/777/999/1234); single-seed numbers can look very different.

---

## 7. Open decisions / possible next steps
1. **Sample-efficient RL** — offline RL on logged rollouts, a fast learned surrogate of the sim, or
   model-based RL — to escape the ~640-trial-step budget that blocks a strict win.
2. **Richer / larger network scenarios** — more cells and spatial structure a fixed heuristic can't
   fully exploit.
3. **Ship the tunable controller as-is** — greener + more reliable, with an operator-selectable
   energy↔QoS dial: a solid, honest deliverable.

---

## 8. Glossary
**gNB** 5G base station · **UE** user device · **PRB** radio-capacity unit (utilisation = how busy) ·
**RLF** radio link failure (dropped call) · **KPM** key performance metric · **BC** behaviour cloning
(supervised imitation) · **PPO** an RL algorithm · **critic** the value-estimator half of an RL agent ·
**shaped reward** the potential-based reward that adds dense idle-sleep credit · **heuristic** the
hand-coded expert (`TwinHeuristic`) · **twin** the simulator standing in for a real network.
