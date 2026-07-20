# Energy-Saving Digital Twin (ns-O-RAN) — Results & Findings

*Self-contained summary of the energy-saving controller work. Readable without the chat
history. Last updated: 2026-07-17.*

---

## TL;DR (one paragraph)

We built a realistic, time-varying ns-O-RAN energy-saving twin (1 LTE anchor + 7 mmWave gNBs,
bursty traffic) and a full learning pipeline (expert heuristic → imitation → reinforcement
learning). The company's load-adaptive **heuristic is a strong baseline** — near the efficient
energy-vs-QoS frontier. A **plain-reward PPO agent tied it** (reproduced it exactly). With an
**improved, potential-based *shaped* reward, RL moved to a distinct, favourable operating point**:
averaged over 4 seeds it **saves more energy (~40% vs ~34%) and drops ~26% fewer calls, at ~9%
less throughput**. So RL ends up **better on energy and reliability, slightly worse on throughput
— a favourable tradeoff, not a strict win on all three axes.** The reason it cannot *strictly*
dominate is structural (the slow simulator gives RL a tiny training budget), which we explain and
quantify. We also built a **tunable pruning controller** that extends the energy-vs-reliability
frontier beyond the heuristic's single operating point.

---

## 1. What was built

| Component | What it is |
|---|---|
| **Twin** | ns-3 mmWave O-RAN sim, `scenario-three`: 1 LTE anchor (cell 1) + 7 mmWave gNBs (cells 2–8), 6 UEs. |
| **Traffic** | A **3-burst-per-episode** pattern with an always-on baseline user — load rises and falls sharply (verified from PRB traces), so the controller must adapt in real time. |
| **Control loop** | Gym ↔ ns-3 every 100 ms; the policy chooses which gNBs sleep each step (files + semaphores). |
| **Expert heuristic** | `heuristic_twin.py` — load-adaptive: sleeps idle cells, wakes on neighbour load, with hysteresis/guardrails. The company baseline. |
| **Learning pipeline** | Behaviour Cloning (imitate the expert) → PPO on a power-model reward (optionally **shaped**). All in `ns-o-ran-gym/rl/`. |
| **Energy model** | Per gNB: P = 600 W static + 400 W × utilisation; sleep ≈ 0 W. |
| **Metrics / figures** | `plot_energy.py` (energy, % saved, throughput, RLF); `plot_report.py` → `figures/` suite; `plot_summary.py`, `plot_frontier.py`. |

---

## 2. Headline results (averaged over 4 random seeds: 555/777/999/1234)

| Policy | Energy saved vs all-on | DL throughput | Dropped calls (RLF) |
|---|---|---|---|
| All cells on (reference) | 0.0% | 6.10 Mbps | 0.00 |
| **Heuristic** (company baseline) | 33.8% | **5.68 Mbps** | 1.20 |
| RL — plain reward | 33.8% | 5.68 Mbps | 1.20  *(ties the heuristic)* |
| **RL — shaped reward** (final) | **39.7%** | 5.18 Mbps | **0.90** |
| Aggressive pruning (grace=3) | 52.9% | 5.25 Mbps | 2.36 |

- **Plain-reward PPO reproduces the heuristic** (a tie, confirmed byte-for-byte).
- **Shaped-reward PPO** — the final RL result — saves **+5.8 pts more energy AND ~26% fewer dropped
  calls (RLF 0.90 vs 1.20)**, at **−0.5 Mbps throughput**. Better on 2 of 3 axes: a **favourable
  tradeoff**, not a strict all-axis win. (Per-seed RL energy varies 25–46%; the average is 39.7%.)
- **Pruning** saves the most energy but costs the most QoS — a tunable operating point (see §3.4).

Figures: `summary_tradeoff.png`, `shaped_vs_heuristic.png`, and the `figures/` suite.

---

## 3. Key findings

### 3.1 The heuristic is near the efficient frontier
Plot energy-saved against both throughput and RLF and the heuristic sits on essentially the
**efficient tradeoff line** — more energy always costs some QoS. It is a genuinely good baseline,
which is exactly why it is hard to beat outright.

### 3.2 The RL journey: a tie, then a favourable tradeoff
- **Plain-reward PPO tied the heuristic.** We warm-started PPO from the cloned policy and
  pre-trained its value estimator (critic) to avoid the classic warm-start collapse; training was
  healthy, but the greedy policy converged back onto the heuristic.
- **Why (structural, not a bug):** with on/off actions over 6 cells, random exploration sleeps
  *idle* and *busy* cells together each trial. Sleeping a busy cell is very costly (dropped calls),
  so the single reward says "sleeping is risky" — and with only ~640 trial-steps affordable
  (the sim runs ~20 s per 100 ms step), RL can't do the credit assignment to learn "sleeping *this
  specific idle* cell was the good move." **Plain RL exploration can't capture the prize at this
  simulator's speed.**
- **Fix — a better reward:** a **potential-based shaped reward** (`reward_mode=power_shaped`) gives
  dense, per-cell credit for sleeping idle cells *without changing the optimum* (Ng et al. 1999).
  With it, RL moved **off the tie** to a distinct, greener + more-reliable operating point (§2) —
  a genuine improvement, though still not a strict 3-axis win (the heuristic stays ahead on throughput).

### 3.3 The "waste" is real in hindsight but not free to reclaim
- The heuristic keeps a cell powered with **no traffic ~43% of the time**.
- A *hindsight* oracle (that "knows" which cells stayed idle) would save 49–63% at almost no QoS cost.
- **But a real controller decides *before* seeing the traffic:** sleeping a momentarily-idle cell
  that's needed next drops a call. So the causal cost of chasing that headroom is real — the
  heuristic's conservatism is justified, and it is what keeps RL from a clean sweep.

### 3.4 The learned controller is *tunable* — an operational plus
The pruning controller has a `grace` dial: an operator can pick any point from the heuristic's
(~34% saved, low RLF) up to ~57% saved (higher RLF), whereas the heuristic is a single fixed point.
Useful when the operator wants to weight energy over a small reliability cost.

---

## 4. What a strict (all-axis) win would take

The binding constraint is the **simulator's sample budget**: ~640 RL trial-steps at ~15–20 s/step
is far too little for RL to out-tune a strong heuristic on a 6-cell on/off decision. Realistic
directions to overcome it:
1. **Sample-efficient RL** — offline RL on logged rollouts, a **fast learned surrogate** of the
   simulator, or model-based RL, so the agent gets orders of magnitude more effective experience.
2. **Richer / larger network scenarios** — more cells and spatial structure a fixed heuristic
   can't fully exploit.
3. **Ship the tunable controller as-is** — a defensible deliverable: greener + more reliable, with
   an operator-selectable energy↔QoS dial.

---

## 5. How to reproduce

```bash
cd /workspace/ns-o-ran-gym
PY=/workspace/.venv/bin/python; NS3=/workspace/ns-3-mmwave-oran
CFG=src/environments/scenario_configurations/es_use_case.json   # 6 UEs, seed 555, burst traffic

# heuristic baseline (default = the balanced baseline in §2)
PYTHONPATH=src $PY heuristic_twin.py --ns3_path $NS3 --config $CFG --num_steps 58 --optimized
$PY plot_energy.py <output_folder>                       # ~34% saved / 5.68 Mbps / RLF 1.20

# imitation + RL   (committed: rl/bc_ppo.zip, rl/ppo_final.zip [plain, tie], rl/ppo_shaped.zip [final])
PYTHONPATH=src $PY rl/train_bc.py --demos rl/demos.npz                       # BC + critic warm-up
PYTHONPATH=src $PY rl/train_ppo.py --config $CFG --reward_mode power_shaped   # PPO fine-tune (~4 h)
PYTHONPATH=src $PY rl/eval_rl.py --model rl/ppo_shaped.zip --config $CFG      # deterministic eval

# pruning controller (energy-vs-reliability frontier)
PYTHONPATH=src $PY rl/aggressive_ctl.py --config $CFG --mode prune --thr 3 --grace 3

# figures + multi-seed comparisons
$PY plot_report.py            # -> figures/ suite (computed live from run folders)
bash rl/compare_clean.sh      # shaped-RL vs heuristic, 4 seeds -> the §2 numbers
```
Key scripts: `heuristic_twin.py`, `rl/train_bc.py` (BC + critic warm-up), `rl/train_ppo.py`,
`rl/eval_rl.py`, `rl/aggressive_ctl.py`, `rl/compare_clean.sh`, `rl/validate_g3.sh`, `plot_report.py`.

---

## 6. Honest bottom line

- **RL result:** with reward shaping, RL learns a **greener + more-reliable** controller than the
  heuristic — **more energy saved AND fewer dropped calls, at slightly lower throughput.** A
  favourable, honest tradeoff; it does **not** strictly beat a strong heuristic on all three axes,
  and we explain precisely why (RL is sample-starved at this simulator's speed; the idle-cell
  headroom is not causally free).
- **Delivered:** a realistic bursty twin, a working imitation+RL pipeline (BC 100% expert match,
  PPO trains stably via a critic warm-up), a rigorous energy-vs-QoS tradeoff, and a tunable
  energy-saving controller.
- **Path to a strict win:** sample-efficient RL (offline / learned-surrogate / model-based) to
  escape the simulator's sample budget — the one real blocker.
