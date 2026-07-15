# Project Overview — ns-O-RAN Energy-Saving Digital Twin + Learning Controller

*The single "read me first" document: what the project is, how every piece fits together, how it
runs, and where everything lives. Self-contained — no chat history needed. Date: 2026-07-15.*

**Companion docs:** `RESULTS.md` (findings & numbers) · `REPRODUCE.md` (build & run from scratch).

---

## 1. What this project is (in one paragraph)

Mobile networks waste energy because base-station radios ("gNBs") stay powered even when almost no
one is using them. This project builds a **digital twin** — a realistic 5G network simulator — and
an intelligent **controller** that decides, every 100 ms, *which gNBs to put to sleep* to save
energy without hurting service quality. We compare three controllers on the same twin: the
network's existing **hand-coded heuristic**, a **reinforcement-learning (RL) agent** trained to
beat it, and an **aggressive pruning controller**. The twin, the control loop, the learning
pipeline, and the analysis are all reproducible from code.

**Goal:** an RL model with a better energy-vs-quality tradeoff than the heuristic.
**Outcome (see `RESULTS.md`):** the heuristic is already near-optimal here; RL *ties* it, and more
energy is only obtainable by trading some quality — a rigorously-explained result.

---

## 2. Background — the O-RAN concepts you need (plain language)

- **gNB** — a 5G base station (radio + antenna). Powering one costs ~600 W even when idle; sleeping
  it saves almost all of that. Our twin has **1 LTE anchor** cell (always-on coverage) + **7 mmWave
  gNBs** (cells 2–8) that can sleep.
- **UE** — "User Equipment," i.e. a phone/device. Ours: **6 UEs** moving and generating traffic.
- **PRB** — "Physical Resource Block," the unit of radio capacity. A cell's **PRB utilisation** = how
  busy it is (0 % = idle, 100 % = full). This is our main "is this cell being used?" signal.
- **RLF** — "Radio Link Failure," i.e. a dropped connection. Our main **reliability / QoS** metric
  (lower is better). Sleeping a cell that's still needed causes RLFs.
- **KPM** — "Key Performance Metric." Every 100 ms each cell reports KPMs (PRB, throughput, RLF, …).
- **O-RAN / RIC / E2 / xApp** — O-RAN is an open 5G architecture where a **RIC** (RAN Intelligent
  Controller) runs control apps ("**xApps**") that read KPMs and send control commands over the
  **E2** interface. In this project **our controller plays the role of the xApp**, and instead of a
  live RIC we drive the simulator directly through shared files + semaphores (details in §5).

---

## 3. The big picture — how the pieces fit

```
        ┌─────────────────────────────────────────────────────────────────┐
        │                     ns-3 twin  (C++ simulator)                    │
        │   scenario-three.cc: 1 LTE anchor + 7 mmWave gNBs, 6 UEs,          │
        │   bursty traffic. Every 100 ms it emits KPMs and applies the       │
        │   ON/OFF decision it is given.                                     │
        └───────▲───────────────────────────────────────────────┬──────────┘
                │  KPMs (per-cell CSV files)          ON/OFF action (CSV file)
                │  + "metrics ready" semaphore        + "control" semaphore
        ┌───────┴───────────────────────────────────────────────▼──────────┐
        │              Gym environment  (Python: EnergySavingEnv)            │
        │   reads KPMs → builds a 61-number observation → asks the           │
        │   controller for an action → writes the action back to ns-3.       │
        └───────▲───────────────────────────────────────────────┬──────────┘
                │  observation (61 features)              action (7 bits)
        ┌───────┴───────────────────────────────────────────────▼──────────┐
        │                        The controller (one of):                   │
        │   • TwinHeuristic  (hand-coded expert)                            │
        │   • PPO agent      (reinforcement learning)                       │
        │   • pruning controller (aggressive energy saver)                  │
        └───────────────────────────────────────────────────────────────────┘
```

The twin is the slow part (~15–20 s per 100 ms step, CPU-bound). Everything else is fast.

---

## 4. The digital twin (what is being simulated)

- **Topology:** cell 1 = LTE anchor (coverage, never sleeps); cells 2–8 = 7 mmWave gNBs (can sleep).
- **Users:** 6 UEs moving in the area, attaching to whichever cell serves them best.
- **Traffic (the important design choice):** a **3-burst-per-episode** pattern with one always-on
  baseline user — load rises and falls sharply over the ~6 s episode. This forces the controller to
  *adapt in real time* (sleep during lulls, wake for bursts), which a flat-traffic scenario would
  not. Defined in `scratch/scenario-three.cc`.
- **Energy model** (`plot_energy.py`): each **ON** gNB draws `P = 600 W (static) + 400 W × utilisation`;
  a **sleeping** gNB ≈ 0 W. Energy = Σ over (cell, 100 ms step) of that power. "% saved" is versus
  the same cells never sleeping (all-on).

---

## 5. The runtime control loop (how the controller drives the sim)

ns-3 runs with `heuristicType=-1` ("an external controller decides") and `useSemaphores=1`. Each
100 ms step is a handshake over `/dev/shm` (shared memory), coordinated by `NsOranEnv`
(`src/nsoran/ns_env.py`):

1. **ns-3 → gym:** ns-3 writes per-cell KPMs to CSV files (`du-cell-*.txt`, …) and releases the
   **metrics-ready** semaphore.
2. **gym reads:** `EnergySavingEnv._get_obs()` loads the KPMs (via the `Datalake`, an SQLite cache)
   into a **61-number observation**.
3. **controller decides:** the heuristic / PPO model / pruning rule picks a **7-bit ON/OFF action**.
4. **gym → ns-3:** `ActionController.create_control_action()` writes the action to a control file
   (`es_actions_for_ns3.csv`) and releases the **control** semaphore.
5. **ns-3 applies it:** sleeps/wakes the gNBs, advances 100 ms, and the loop repeats.

This is true real-time closed-loop control — during both training and evaluation the model drives a
live ns-3 process step by step.

### Observation, action, reward
- **Observation (61 features):** per cell (2–8) — PRB utilisation, throughput, RLF, energy-efficiency
  KPI, modulation ratio — plus 5 network-wide aggregates. (No per-cell *demand forecast* exists,
  which is why "when to wake a sleeping cell" must be *learned*, not hand-rules — see `RESULTS.md`.)
- **Action:** `MultiBinary(7)` — one ON/OFF bit per gNB; the anchor is forced ON.
- **Reward (for RL):** `throughput_Mbps − 1.0 × power_kW − 2.0 × RLF` per step — i.e. reward carried
  traffic, penalise energy and dropped calls. Same power model as the analysis.

---

## 6. The three controllers

| Controller | What it is | Where |
|---|---|---|
| **TwinHeuristic** | Hand-coded expert: sleeps idle cells, wakes on neighbour load, with hysteresis/guardrails. The baseline to beat. | `heuristic_twin.py` |
| **PPO agent** | RL policy (neural net). Warm-started by cloning the heuristic, then fine-tuned on the reward. | `rl/train_bc.py` → `rl/train_ppo.py` → `rl/ppo_final.zip` |
| **Pruning controller** | Keeps the heuristic's wake logic but switches OFF cells it powers with *no traffic*; a `grace` dial sets aggressiveness. | `rl/aggressive_ctl.py` |

*(There is also ns-3's own built-in heuristic in `src/mmwave/helper/energy-heuristic.cc`, but it is
quota-driven — it sleeps a fixed *count* of cells regardless of load — so it cannot adapt. That is
why the project uses the custom load-adaptive `TwinHeuristic` instead.)*

---

## 7. The learning pipeline (end to end)

```
 collect demos           behaviour cloning        PPO fine-tune          evaluate & compare
 (heuristic drives  ──►   (+ critic warm-up)  ──►  (~4 h, live ns-3) ──►  (heuristic vs PPO vs
  the sim, log obs        clone heuristic          improve on reward       pruning; make figures)
  + action pairs)         into a neural net
```

1. **Collect demos** (`rl/collect_demos.py`, `rl/collect_new.sh`): run the heuristic on the twin,
   log (observation, action) pairs → `rl/demos.npz`. Simulator-bound (slow), done once.
2. **Behaviour Cloning + critic warm-up** (`rl/train_bc.py`): supervised-train the PPO network to
   copy the heuristic (reaches **100 % match**), *and* pre-train its value estimator on the demos'
   returns. The warm-up is essential — without it, PPO's first updates use garbage feedback and
   destroy the good policy (we observed and fixed exactly that). → `rl/bc_ppo.zip`, `rl/obs_stats.npz`.
3. **PPO fine-tune** (`rl/train_ppo.py`): reinforcement-learn on the true reward via live ns-3
   rollouts. ~4 h (the sim is the bottleneck). → `rl/ppo_final.zip`.
4. **Evaluate & analyse** (`rl/eval_rl.py`, `rl/compare_new.sh`, `rl/sweep_grace.sh`,
   `rl/validate_g3.sh`): run each controller, score with `plot_energy.py`, and render the figures.

---

## 8. Repository structure (three repos, siblings)

```
/workspace/
├── ns-3-mmwave-oran/        # THE TWIN (C++/ns-3). branch: energy-saving-twin
│   ├── scratch/scenario-three.cc          # the scenario (topology, traffic, control hook)
│   ├── src/mmwave/helper/energy-heuristic.{cc,h}   # ns-3's built-in (quota) heuristic
│   └── build/optimized/…/scenario-three   # the compiled binary the gym launches
│
├── ns-o-ran-gym/            # THE GYM + RL + ANALYSIS (Python). branch: energy-saving-fixes
│   ├── src/nsoran/          # base framework (upstream + our fixes)
│   │   ├── ns_env.py            #   NsOranEnv: the ns-3 <-> gym control loop + semaphores
│   │   ├── action_controller.py #   writes the action file ns-3 reads
│   │   └── datalake.py          #   SQLite cache of the KPMs
│   ├── src/environments/
│   │   └── es_env.py            # EnergySavingEnv: 61-feature obs + reward (our use case)
│   ├── heuristic_twin.py    # TwinHeuristic — the load-adaptive expert
│   ├── rl/                  # THE RL PIPELINE
│   │   ├── es_wrapper.py        #   Gym wrapper: MultiBinary(7) action, power reward
│   │   ├── collect_demos.py     #   log expert (obs, action) demos
│   │   ├── train_bc.py          #   behaviour cloning + critic warm-up
│   │   ├── train_ppo.py         #   PPO fine-tune
│   │   ├── eval_rl.py           #   run a trained model on the twin
│   │   ├── aggressive_ctl.py    #   the pruning / reactive controller
│   │   ├── *.sh                 #   batch runners (collect/compare/sweep/validate)
│   │   ├── *.tsv                #   figure manifests (run-folder pointers)
│   │   └── ppo_final.zip, bc_ppo.zip, obs_stats.npz, demos.npz   # trained artifacts
│   ├── plot_energy.py       # score ONE run (energy/throughput/RLF) + power-over-time PNG
│   ├── plot_frontier.py     # per-seed tradeoff figure  -> tradeoff.png
│   ├── plot_summary.py      # 4-seed averaged figure     -> summary_tradeoff.png
│   ├── src/environments/scenario_configurations/es_use_case.json   # scenario config
│   ├── RESULTS.md, REPRODUCE.md, requirements-lock.txt
│   └── output/<uuid>/       # one folder per simulation run (raw traces; see §9)
│
└── oran-e2sim/              # E2 interface library (built once; ns-3 links against it)
```

---

## 9. Outputs & how metrics are computed

Every simulation run creates a new **`output/<uuid>/`** folder (nothing is overwritten). The three
files the analysis uses:
- `bsState.txt` — which gNBs were ON/OFF at each step (→ energy).
- `du-cell-*.txt` — per-cell KPMs incl. PRB utilisation (→ energy detail, load).
- `database.db` — SQLite `grafana` table with throughput & RLF per step (→ QoS).

`plot_energy.py`'s `summarize()` reads these and applies the power model → energy, % saved,
throughput, RLF, mean gNBs-on. The figure scripts (`plot_frontier.py`, `plot_summary.py`) read a
**manifest of run folders** (`rl/frontier_runs.tsv`, `rl/summary_runs.tsv`) and compute every number
live via `summarize()` — **no numbers are hard-coded**.

Find a run's folder: each run script prints it, or `ls -dt output/*/ | head -1` gives the newest.

---

## 10. Results (headline)

Averaged over 4 random seeds — full discussion in `RESULTS.md`:

| Policy | Energy saved | Throughput | RLF (dropped calls) |
|---|---|---|---|
| all cells on | 0 % | 6.10 Mbps | 0.00 |
| **heuristic** (baseline) | ~34 % | ~5.68 Mbps | ~1.20 |
| **PPO (RL)** | ~34 % | ~5.68 Mbps | ~1.20  ← ties the heuristic |
| pruning (aggressive) | ~53 % | ~5.25 Mbps | ~2.36  ← more energy, worse QoS |

**Bottom line:** the heuristic sits near the efficient energy-vs-QoS frontier; PPO matches it (a
structural limit of RL at this simulator's sample budget, not a tuning bug); saving more energy
costs reliability. A genuine clean win would need a *spatial-hotspot* scenario (recommended next
step, `RESULTS.md` §4). Figures: `tradeoff.png`, `summary_tradeoff.png`, `energy_new.png`.

---

## 11. How to run it

- **Rebuild & run from scratch on a new machine:** follow `REPRODUCE.md`.
- **Quick evaluate the trained model** (after building):
  ```bash
  cd ns-o-ran-gym
  PYTHONPATH=src /workspace/.venv/bin/python rl/eval_rl.py --model rl/ppo_final.zip \
      --stats rl/obs_stats.npz --config src/environments/scenario_configurations/es_use_case.json
  ```
- **Regenerate the figures:** `python plot_frontier.py` and `python plot_summary.py`.

---

## 12. Documentation map

| Doc | Purpose | Status |
|---|---|---|
| **`OVERVIEW.md`** (this file) | whole-project map + how it works | current |
| **`RESULTS.md`** | findings, numbers, why, next step | current |
| **`REPRODUCE.md`** | build + run from zero | current |
| `HANDOFF.md`, `MILESTONE1_CHECKS.txt`, `PROJECT_HANDOFF.md`, `RL_APPROACH_RECOMMENDATION.md` | earlier-phase notes | **superseded** — archived in `work.zip`, not in the tree |

---

## 13. Glossary

**gNB** 5G base station · **UE** user device/phone · **PRB** unit of radio capacity (utilisation =
how busy) · **RLF** radio link failure (dropped call) · **KPM** key performance metric · **RIC**
RAN intelligent controller · **xApp** control app on the RIC · **E2** RIC↔RAN control interface ·
**RU** radio unit · **BC** behaviour cloning (supervised imitation) · **PPO** Proximal Policy
Optimization (an RL algorithm) · **critic** the value-estimator half of an RL agent · **twin**
the simulator standing in for a real network · **grace** how long the pruning controller keeps an
idle cell on before sleeping it.
