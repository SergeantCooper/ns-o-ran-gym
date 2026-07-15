# Energy-Saving Digital Twin (ns-O-RAN) — Results & Findings

*Self-contained summary of the energy-saving controller work. Written to be readable
without the chat history. Date: 2026-07-15.*

---

## TL;DR (one paragraph)

We built a realistic, time-varying ns-O-RAN energy-saving twin (1 LTE anchor + 7 mmWave
gNBs, bursty traffic) and a full learning pipeline (expert heuristic → imitation →
reinforcement learning). **The hand-coded heuristic turns out to sit very near the
efficient energy-vs-QoS frontier for this scenario.** A reinforcement-learning agent (PPO),
trained correctly, **matches the heuristic but does not beat it** — and we identified *why*
(a structural limit of RL at a slow simulator's sample budget, not a tuning bug). A more
aggressive controller can save substantially more energy (~53% vs the heuristic's ~34%),
but **only by trading reliability** (roughly 2× the dropped calls). No approach *strictly*
beats the heuristic on every axis at once. We explain the cause and give the concrete,
realistic route to a genuine clean win (a spatial-hotspot scenario).

---

## 1. What was built

| Component | What it is |
|---|---|
| **Twin** | ns-3 mmWave O-RAN sim, `scenario-three`: 1 LTE anchor (cell 1) + 7 mmWave gNBs (cells 2–8), 6 UEs. |
| **Traffic** | Redesigned to a **3-burst-per-episode** pattern with an always-on baseline user — load rises and falls sharply (verified from PRB traces), so a controller must adapt in real time. |
| **Control loop** | Gym ↔ ns-3 every 100 ms; a policy chooses which gNBs sleep each step. |
| **Expert heuristic** | `heuristic_twin.py` — load-adaptive: sleeps idle cells, wakes on neighbour load, with hysteresis/guardrails. |
| **Learning pipeline** | Behaviour Cloning (imitate the expert) → PPO (improve on a power-model reward). All in `ns-o-ran-gym/rl/`. |
| **Energy model** | Per gNB: P = 600 W static + 400 W × utilisation; sleep ≈ 0 W. |
| **Metrics** | `plot_energy.py` (energy, % saved, throughput, RLF), `plot_frontier.py` / `plot_summary.py` (energy-vs-QoS scatter). |

---

## 2. Headline results (averaged over 4 random seeds: 555/777/999/1234)

| Policy | Energy saved vs all-on | DL throughput | Dropped calls (RLF) | Mean gNBs on |
|---|---|---|---|---|
| All cells on | 0.0% | 6.10 Mbps | 0.00 | 7.00 |
| **Heuristic** (company baseline) | **33.8%** | **5.68 Mbps** | **1.20** | 4.34 |
| **PPO (reinforcement learning)** | 33.8% | 5.68 Mbps | 1.20 | 4.34 |
| Aggressive pruning (grace=3) | 52.9% | 5.25 Mbps | 2.36 | ~3.3 |

- The PPO row is **identical** to the heuristic row — the trained deterministic policy
  reproduces the heuristic exactly. This is a **tie**, confirmed byte-for-byte.
- Pruning saves ~19 points more energy (robust: it beat the heuristic on energy on *every*
  seed) **but** costs ~8% throughput and roughly **doubles** dropped calls.

See `ns-o-ran-gym/summary_tradeoff.png` for the averaged tradeoff picture.

---

## 3. Key findings (the important part)

### 3.1 The heuristic is near the efficient frontier
Plotting energy-saved against both throughput and RLF, the heuristic and the pruning
controller lie on essentially the **same tradeoff line** — more energy always costs more
dropped calls. The heuristic is simply a well-chosen point on that line. That is a genuine
(if modest-sounding) validation of the company's heuristic: it is hard to beat because it is
already good.

### 3.2 RL ties the heuristic — and *why* (structural, not a bug)
- We warm-started PPO from the cloned heuristic and pre-trained its value estimator (critic)
  to avoid the classic warm-start collapse. Training was healthy (no divergence).
- **Yet the final policy learned to keep cells on**, matching the heuristic (its probability
  of sleeping any cell converged to ~0.4%).
- **Cause:** with on/off actions over 6 cells, random exploration sleeps *idle* and *busy*
  cells together in each trial. Sleeping a busy cell is very costly (dropped calls), so the
  single reward signal says "sleeping is risky" — and with only ~640 trial-steps affordable
  (the simulator runs ~20 s per 100 ms step), the algorithm cannot do the fine-grained credit
  assignment to learn "sleeping *this specific idle* cell was the good part." **Plain RL
  exploration cannot capture the prize at this simulator's speed.** This is a real, citable
  limitation, not a hyperparameter mistake.

### 3.3 The "waste" is real in hindsight but not free to reclaim
- Measured directly: the heuristic keeps a cell powered with **no traffic 43% of the time**.
- A *hindsight* oracle (which "knows" which cells had no traffic) would save 49–63% at almost
  no QoS cost — that is where the ~30% headroom claim came from.
- **But a real controller must decide *before* seeing the traffic.** When it sleeps a
  momentarily-idle cell that then gets used, a call drops. So the causal cost of chasing that
  headroom is the ~2× RLF we measured. The heuristic's conservatism is *justified*.

### 3.4 The learned controller is *tunable* — an operational plus
The pruning controller has a "grace" dial (how long a cell stays on after going idle). This
gives an operator a **family** of operating points from the heuristic's (34% saved, low RLF)
up to ~57% saved (higher RLF) — whereas the heuristic is a single fixed point. If an operator
values energy over a small reliability cost, that flexibility is useful.

---

## 4. What "beating the heuristic" would take (recommended next step)

The reason there is no free lunch here: this scenario spreads users across **all** cells, so
every cell is eventually used and sleeping any of them risks a dropped call. Real networks
have **spatial hotspots** — busy areas and quiet areas. In a hotspot scenario some cells are
*genuinely, persistently* idle; the heuristic wastes power keeping them on "just in case,"
and a smart controller can sleep them **for free** (no traffic → no dropped calls). That is:
- the **honest route to a genuine clean win** over the heuristic, and
- a **more realistic** twin.

This was scoped but not executed (it needs a scenario redesign + a re-run of the pipeline,
several more hours). It is the recommended direction if a strict "RL beats heuristic" result
is required.

---

## 5. How to reproduce

```bash
cd /workspace/ns-o-ran-gym
PY=/workspace/.venv/bin/python; NS3=/workspace/ns-3-mmwave-oran
CFG=src/environments/scenario_configurations/es_use_case.json   # 6 UEs, seed 555, burst traffic

# heuristic baseline
PYTHONPATH=src $PY heuristic_twin.py --ns3_path $NS3 --config $CFG --num_steps 58 --optimized
$PY plot_energy.py <output_folder>

# imitation + RL  (already trained: rl/bc_ppo.zip, rl/ppo_final.zip, rl/obs_stats.npz)
PYTHONPATH=src $PY rl/train_bc.py --demos rl/demos.npz          # BC + critic warm-up
PYTHONPATH=src $PY rl/train_ppo.py --config $CFG                 # PPO fine-tune (~4 h)
PYTHONPATH=src $PY rl/eval_rl.py  --model rl/ppo_final.zip --config $CFG   # deterministic eval

# aggressive pruning controller (energy-vs-reliability frontier)
PYTHONPATH=src $PY rl/aggressive_ctl.py --config $CFG --mode prune --thr 3 --grace 3

# figures
$PY plot_summary.py            # honest averaged tradeoff  -> summary_tradeoff.png
bash rl/validate_g3.sh         # multi-seed heuristic-vs-pruning validation
```

Key scripts: `heuristic_twin.py`, `rl/train_bc.py` (BC + critic warm-up), `rl/train_ppo.py`,
`rl/aggressive_ctl.py` (pruning/reactive controller), `rl/compare_new.sh`,
`rl/sweep_grace.sh`, `rl/validate_g3.sh`, `plot_summary.py`.

---

## 6. Honest bottom line

- We did **not** get an RL model that strictly beats the heuristic on this scenario, and we
  can explain precisely why (the heuristic is near-optimal; RL is sample-starved at this
  simulator speed; the headroom isn't causally free).
- We **did** deliver: a realistic bursty twin, a working imitation+RL pipeline (BC reaches
  100% expert match, PPO trains stably with a critic warm-up fix), a rigorous quantification
  of the energy-vs-QoS tradeoff, and a tunable controller that extends the energy frontier
  beyond the heuristic at a measured QoS cost.
- The concrete, realistic path to a genuine clean win is a **spatial-hotspot scenario**.
