# Project Context Primer — ns-O-RAN Energy-Saving Twin + RL

*Paste this whole file into a fresh Claude chat to give it complete context on my project so it can
help me discuss/continue it. It captures everything done **up to (not including) the "hotspot"
step**, including the decisions and the reasoning behind them. Date: 2026-07-15.*

---

## 0. How to use this
I'm working on an energy-saving controller for a simulated 5G network (my company assigned it).
I can't run the code from my phone — I just need you to understand the project so we can reason
about it. The full docs live in the repo (`OVERVIEW.md`, `RESULTS.md`, `REPRODUCE.md`); this file
is the narrative + current-state summary.

---

## 1. The goal
Base-station radios ("gNBs") waste power staying ON when few people use them. I have a **digital
twin** (an ns-3 simulator of a 5G network) and I want a **controller** that decides every 100 ms
which gNBs to put to **sleep** to save energy **without hurting service quality** (throughput /
dropped calls). The company's framing: **an RL model that beats the hand-coded heuristic** on the
energy-vs-quality tradeoff.

**Where we landed (before the hotspot step): the heuristic is already near-optimal on our
scenario; RL *ties* it and we can explain exactly why; more energy is only obtainable by trading
some quality.** We're now testing whether a more realistic "hotspot" scenario makes a genuine clean
win reachable.

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
  calls), efficiency KPI, modulation — plus 5 network aggregates. **No per-cell demand forecast**
  exists (this matters — see §4).
- **Action:** `MultiBinary(7)`, one ON/OFF bit per gNB; the anchor is forced ON.
- **Custom RL reward (I wrote this):** `reward = throughput_Mbps − 1.0·power_kW − 2.0·RLF` per step,
  where power = Σ over ON cells of `(600 W + 400 W·utilisation)`. It's in `rl/es_wrapper.py`
  (`_power_reward`). I replaced the environment's built-in opaque proxy reward with this so the
  agent optimises **the same power model the analysis scores** (in real units: Mbps, kW, drops).
- **Energy model / metrics:** `plot_energy.py` — each ON gNB = 600 W static + 400 W·util, sleeping
  ≈ 0 W. "% saved" is vs. all-cells-on. QoS = throughput (Mbps) + RLF (dropped calls, lower better).

---

## 3. What we did (the journey + decisions)
1. **Fixed the environment (M1):** crash on empty KPM windows, a SINR string-vs-int bug, a PRB
   utilisation bug; corrected the scenario config (`e2nrEnabled`, `bsOn=7`, `heuristicType=-1`).
2. **Understood the built-in heuristic:** ns-3's own `heuristicType=2` is **quota-driven** (sleeps a
   fixed *count* of cells regardless of load) → it can't adapt. That's why the project uses a custom
   **load-adaptive** expert, `heuristic_twin.py` (`TwinHeuristic`): sleeps idle cells, wakes on
   neighbour load, with hysteresis/guardrails.
3. **Deferred GAIL (company's original ask):** I argued (in a memo) that GAIL only *imitates* the
   heuristic — it can't beat it — and is expensive on a slow simulator. The "beating" must come from
   **RL on the true reward**. So the plan became **BC (imitate) → PPO (surpass)**, GAIL optional.
   The company accepted deferring GAIL.
4. **Built the RL pipeline** (`rl/`): collect expert demos → **Behaviour Cloning** (BC) to clone the
   heuristic into a neural net (reached **100 % match**) → **PPO** fine-tune on the custom reward →
   evaluate. Stack: Stable-Baselines3, PyTorch (CPU — the sim, not the NN, is the bottleneck).
5. **Old scenario (smooth day/night traffic) → RL TIED the heuristic.** Root cause was the
   *scenario*, not the RL: on smooth load the heuristic was already near-optimal, so there was no
   waste to reclaim.
6. **Redesigned the traffic to be bursty** (`scenario-three.cc`): 3 sharp traffic bursts per episode
   with an always-on baseline user, so the controller must adapt in real time. Verified the load
   oscillates and the heuristic became dynamic (cells-on swings 2–7).
7. **PPO run #1 on the burst scenario COLLAPSED** (reward −6.4 → −30). Diagnosed: warm-starting PPO
   from the cloned policy leaves its **value estimator (critic) random**, so early updates use
   garbage feedback and wreck the good policy — a classic BC→PPO failure.
8. **Fixed it:** BC now also **pre-trains the critic** on the demos' returns (`train_bc.py`). Verified
   the critic then matches true returns at **0.99 correlation** while the cloned policy stays a
   perfect copy.
9. **PPO run #2 (with warm critic) = TIE.** No collapse this time, but the final deterministic
   policy came out **identical to the heuristic** (its probability of sleeping any cell converged to
   ~0.4 %). **Why (structural, not a bug):** with random ON/OFF exploration over 6 cells, each trial
   sleeps idle *and* busy cells together; busy-cell failures dominate the single reward, so the
   algorithm learns "sleeping is risky." With only ~640 affordable trial-steps (slow sim) it can't
   do the fine-grained credit assignment to learn "sleeping *this specific idle* cell was good."
   **Plain RL exploration can't capture the prize at this simulator's speed.**
10. **Quantified the headroom:** the heuristic keeps a cell powered with **no traffic 43 % of the
    time**. A *hindsight* oracle (that knows which cells had no traffic) would save 49–63 % at almost
    no cost. **But** a real controller must decide *before* seeing the traffic — sleeping a
    momentarily-idle cell that then gets used causes dropped calls. So the headroom is **real in
    hindsight but not causally free**.
11. **Built a pruning controller** (`rl/aggressive_ctl.py`) that keeps the heuristic's wake logic but
    switches OFF cells it powers with no traffic (a `grace` dial sets aggressiveness). Swept it and
    **validated on 4 seeds**: it robustly saves **~19 points more energy** but costs **~8 %
    throughput and ~2× the dropped calls**. No setting *strictly* beats the heuristic.
12. **Conclusion:** the heuristic sits **near the efficient energy-vs-QoS frontier**; every energy
    gain buys a quality loss. That's the honest result. To get a genuine *clean* win we need a
    scenario with **structurally idle cells** — which is the hotspot step now in progress.

---

## 4. Key results (averaged over 4 seeds: 555/777/999/1234)

| Policy | Energy saved | Throughput | RLF (dropped calls) |
|---|---|---|---|
| all cells on | 0 % | 6.10 Mbps | 0.00 |
| **heuristic** (baseline) | ~34 % | ~5.68 Mbps | ~1.20 |
| **PPO (RL)** | ~34 % | ~5.68 Mbps | ~1.20  ← ties the heuristic exactly |
| pruning (aggressive) | ~53 % | ~5.25 Mbps | ~2.36  ← more energy, worse QoS |

**Bottom line:** PPO ties; nothing strictly beats the heuristic on this (uniform-load) scenario;
the limitation is structural (RL sample-starved on a slow sim + no free energy to reclaim).

---

## 5. Current state (files, artifacts, git)
- **Two git repos** (siblings under `/workspace`):
  - `ns-3-mmwave-oran` (the twin, C++), branch **`energy-saving-twin`** — **pushed** ✓.
  - `ns-o-ran-gym` (gym/RL/analysis, Python), branch **`energy-saving-fixes`** — **2 commits still
    unpushed** as of writing (the OVERVIEW.md doc + a doc-map fix); I need to `git push` once more.
  - (`oran-e2sim` = E2 interface library, built once; ns-3 links it.)
- **Trained artifacts (committed):** `rl/ppo_final.zip` (the RL model), `rl/bc_ppo.zip` (BC + warm
  critic), `rl/obs_stats.npz` (obs normalization), `rl/demos.npz` (expert demos).
- **Figures:** `tradeoff.png` (per-seed scatter + table), `summary_tradeoff.png` (4-seed averaged),
  `energy_new.png` (power/cells-on/sleep-timeline over time). All computed **live from run folders**
  (no hard-coded numbers).
- **Docs (current):** `OVERVIEW.md` (how it all works), `RESULTS.md` (findings), `REPRODUCE.md`
  (build & run from scratch), `CONTEXT.md` (this file). Older docs (`HANDOFF.md`,
  `MILESTONE1_CHECKS.txt`, `PROJECT_HANDOFF.md`, `RL_APPROACH_RECOMMENDATION.md`) are **superseded**
  and archived in `work.zip`.
- **Key code:** `heuristic_twin.py` (expert), `rl/es_wrapper.py` (gym wrapper + custom reward),
  `rl/train_bc.py` (BC + critic warm-up), `rl/train_ppo.py` (PPO), `rl/eval_rl.py`,
  `rl/aggressive_ctl.py` (pruning), `plot_energy.py` / `plot_frontier.py` / `plot_summary.py`.
  Env: `src/environments/es_env.py`, base loop `src/nsoran/ns_env.py`. Config
  `src/environments/scenario_configurations/es_use_case.json` (6 UEs, seed 555, burst traffic).

---

## 6. Gotchas / constraints (important for reasoning)
- **ns-3 is slow (~15–20 s/step) and single-threaded; a GPU does not help.** This is *the* binding
  constraint — it's why RL is sample-starved and full training takes ~4 h.
- Compare energy **per-step**, not per-run total (runs can differ in step count).
- Clean stale semaphores between runs: `rm -f /dev/shm/sem.*`.
- **Never rebuild ns-3 while a live gym/PPO run is going** — it relaunches the binary each episode.
- Each run writes a **new** `output/<uuid>/` folder (nothing overwritten); newest = `ls -dt output/*/ | head -1`.
- The RL reward is my **custom** power-model reward (`rl/es_wrapper.py`), *not* the env's built-in
  proxy (which is what gets printed during a plain heuristic run — a common source of confusion).

---

## 7. What's running now / immediate next step (the hotspot step — just started)
I'm testing whether a **spatial-hotspot** scenario yields a genuine clean win. It clusters the 6
users around only **4 of the 7 cells** (config: `positionAllocator=1, nBsNoUesAlloc=3`), leaving
**3 cells structurally idle** — more realistic (busy vs quiet areas) and potentially winnable,
because a controller could sleep the never-used cells **for free**. A ~15-min feasibility probe is
running: does the heuristic wastefully keep those idle cells ON? If yes → headroom → full pipeline
(after fixing a non-deterministic cell-shuffle in the scenario). If no → the tie result stands.
*(This is the one part NOT yet resolved as of this doc.)*

---

## 8. Open decisions / possible next steps
1. **Hotspot scenario** (in progress) — the realistic route to a clean beat.
2. **Fix the deeper RL limitation** — give each cell its own credit (factored reward / action
   masking) so RL can isolate "sleep this idle cell." Would let RL win even on the hard scenario.
3. **More rigor** — more seeds + reward-weight sensitivity to make the final claim bulletproof.
4. **Stakeholder call** — accept the honest tie + tunable tradeoff frontier as the deliverable, or
   invest in #1/#2 for a clean "RL beats heuristic" story.

---

## 9. Glossary
**gNB** 5G base station · **UE** user device · **PRB** radio-capacity unit (utilisation = how busy) ·
**RLF** radio link failure (dropped call) · **KPM** key performance metric · **BC** behaviour cloning
(supervised imitation) · **PPO** an RL algorithm · **critic** the value-estimator half of an RL agent ·
**heuristic** the hand-coded expert controller (`TwinHeuristic`) · **twin** the simulator standing in
for a real network.
