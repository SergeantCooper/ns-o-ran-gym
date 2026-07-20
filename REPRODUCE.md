# Reproducing the ns-O-RAN Energy-Saving Project from Scratch

*Step-by-step to rebuild and re-run the whole project on another (Linux) machine.
Self-contained — no chat history needed. Date: 2026-07-15.*

> **Platform note.** This stack (ns-O-RAN + E2 simulator) builds and runs on **Linux**
> (tested on Ubuntu-like). It will *not* build natively on Windows/macOS — on a Windows work
> laptop use **WSL2 (Ubuntu)** or a Linux VM. Expect the ns-3 build to take ~30–60 min and the
> simulator to run at **~15–20 s per 100 ms control step** (it is CPU-bound and single-threaded;
> a GPU does not help). One 58-step run ≈ 12–20 min; the full PPO training ≈ 4 h.

---

## 0. Two paths

- **Path A — reuse the trained artifacts (fast).** The trained model and demos are committed, so
  you can skip the 4 h PPO training and just build + evaluate. Do sections 1–5, then 6, then 7A.
- **Path B — full retrain from zero.** Everything including PPO. Do 1–5, then 6, then 7B.

Either way, sections 1–5 (build + environment) are required.

---

## 1. System prerequisites

```bash
sudo apt update
sudo apt install -y build-essential cmake g++ git pkg-config \
     libsctp-dev lksctp-tools autoconf automake libtool bison flex \
     python3 python3-dev python3-venv
```
Python **3.12** is what this was developed on. (The exact system-dependency list for ns-O-RAN /
e2sim is maintained upstream: <https://openrangym.com/tutorials/ns-o-ran> — consult it if the
build complains about a missing library.)

---

## 2. Get the code

Three repos are involved. **Important:** the energy-saving work lives on specific *branches* and
in commits that may only exist on the machine this was developed on — if a fresh `git clone` does
not contain the RL pipeline / `scenario-three` changes, the branches must first be **pushed** from
the dev machine (`git push origin <branch>`) or the folders copied directly (e.g. the `work.zip`
snapshot in the workspace).

```bash
mkdir -p ~/oran && cd ~/oran

# 2a. E2 simulator (E2 interface library)
git clone https://github.com/wineslab/ns-o-ran-e2-sim oran-e2sim
git -C oran-e2sim checkout develop            # HEAD used: 2209995

# 2b. ns-3 mmWave O-RAN (the twin)
git clone https://github.com/SergeantCooper/ns-o-ran-ns3-mmwave.git ns-3-mmwave-oran
git -C ns-3-mmwave-oran checkout energy-saving-twin     # HEAD used: 5c231413

# 2c. Gym / RL side
git clone https://github.com/SergeantCooper/ns-o-ran-gym.git ns-o-ran-gym
git -C ns-o-ran-gym checkout energy-saving-fixes        # branch with all the RL work
```

Final layout must be **siblings** (paths are hard-wired to this):
```
~/oran/oran-e2sim
~/oran/ns-3-mmwave-oran
~/oran/ns-o-ran-gym
```
> The scripts below assume the base is `/workspace`. If you use `~/oran`, either symlink
> (`sudo ln -s ~/oran /workspace`) or pass `--ns3_path ~/oran/ns-3-mmwave-oran` to the Python
> tools and adjust paths.

---

## 3. Build the E2 simulator (must come first — ns-3 links against it)

```bash
cd ~/oran/oran-e2sim/e2sim
./build_e2sim.sh 2          # builds libe2sim and installs it to /usr/local/lib
# if it needs root to install:  sudo ./root_build_e2sim.sh 2
```
Verify: `ls /usr/local/lib/libe2sim.a` should exist. (ns-3's `oran-interface` module finds e2sim
via `/usr/local/include/e2sim` + `/usr/local/lib`.)

---

## 4. Build ns-3 (the twin)

```bash
cd ~/oran/ns-3-mmwave-oran
./ns3 configure -d optimized --enable-examples --disable-python
./ns3 build
```
This produces `build/optimized/…/scenario-three`. (The repo's `e2term-ns3-build.sh` uses the old
`./waf`; the current tree uses the CMake-based `./ns3` shown above — that is what produced the
working `build/optimized`.) A successful build ends with no errors (a pre-existing
`random_shuffle` deprecation *warning* is harmless).

Quick check the scenario binary runs:
```bash
./ns3 run "scratch/scenario-three --simTime=2 --ues=3 --heuristicType=0 --useSemaphores=0"
```

---

## 5. Python environment

```bash
cd ~/oran/ns-o-ran-gym
python3 -m venv ~/oran/.venv
source ~/oran/.venv/bin/activate

# core deps (CPU torch — we don't need CUDA; ns-3 is the bottleneck, not the NN)
pip install numpy pandas gymnasium matplotlib scipy posix-ipc "sem>=0.3.9" \
            stable-baselines3 pytest
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .          # installs the local 'nsoran' package (the gym env)
```
Exact versions this was run with are pinned in `ns-o-ran-gym/requirements-lock.txt`
(`pip install -r requirements-lock.txt` reproduces them — note that file pins a CUDA torch build;
the CPU install above is preferred on a laptop).

---

## 6. Smoke test (verify the whole loop works)

```bash
cd ~/oran/ns-o-ran-gym
export PY=~/oran/.venv/bin/python NS3=~/oran/ns-3-mmwave-oran
CFG=src/environments/scenario_configurations/es_use_case.json   # 6 UEs, seed 555, burst traffic

rm -f /dev/shm/sem.*                    # clean any stale semaphores first
PYTHONPATH=src $PY heuristic_twin.py --ns3_path $NS3 --config $CFG --num_steps 58 --optimized
```
Expected: it steps ~58 times printing `cells_on=…` (values vary 2–7, not stuck at 7), then finishes.
Score the run it produced:
```bash
$PY plot_energy.py $(ls -dt output/*/ | head -1)     # ~35% saved, ~5.7 Mbps, RLF ~1.4
```

---

## 7A. Path A — evaluate the committed model (fast)

The trained models — `rl/ppo_final.zip` (plain reward, ties the heuristic) and `rl/ppo_shaped.zip`
(shaped reward, the final favourable-tradeoff model) — plus the BC start (`rl/bc_ppo.zip`),
normalization (`rl/obs_stats.npz`) and demos (`rl/demos.npz`) are committed. Evaluate the final model:
```bash
rm -f /dev/shm/sem.*
PYTHONPATH=src $PY rl/eval_rl.py --model rl/ppo_shaped.zip --stats rl/obs_stats.npz \
    --ns3_path $NS3 --config $CFG --num_steps 58
# -> prints per-step actions + 'RL run folder: output/<uuid>'; score it with plot_energy.py
```

## 7B. Path B — full pipeline from zero

```bash
cd ~/oran/ns-o-ran-gym                          # $PY, $NS3, $CFG as above
# 1) collect expert demos + all-on baseline (a few ~12-min runs)
bash rl/collect_new.sh
# 2) behaviour cloning + critic warm-up (fast, offline)  -> rl/bc_ppo.zip, rl/obs_stats.npz
PYTHONPATH=src $PY rl/train_bc.py --demos rl/demos.npz
# 3) PPO fine-tune (~4 h; live ns-3 rollouts). Do NOT rebuild ns-3 while this runs.
PYTHONPATH=src $PY rl/train_ppo.py --config $CFG --reward_mode power_shaped --ent_coef 0.01  # -> rl/ppo_shaped.zip
# 4) evaluate + compare + frontier + multi-seed validation
bash rl/compare_new.sh        # heuristic vs PPO vs pruning (seed 555)
bash rl/sweep_grace.sh        # pruning frontier -> writes rl/frontier_runs.tsv + tradeoff.png
bash rl/validate_g3.sh        # multi-seed -> writes rl/summary_runs.tsv + summary_tradeoff.png
```

---

## 8. Expected results (so you can confirm)

Averaged over seeds 555/777/999/1234 (see `RESULTS.md` for the full write-up):

| Policy | Energy saved | Throughput | RLF (dropped calls) |
|---|---|---|---|
| all cells on | 0% | 6.10 Mbps | 0.00 |
| heuristic | ~34% | ~5.68 Mbps | ~1.20 |
| RL — plain reward | ~34% | ~5.68 Mbps | ~1.20  ← ties the heuristic |
| **RL — shaped reward** (final) | **~40%** | ~5.18 Mbps | **~0.90**  ← +energy, +reliability, −throughput |
| pruning (aggressive) | ~53% | ~5.25 Mbps | ~2.36  ← more energy, worse QoS |

Headline: plain-reward PPO **ties** the heuristic; the **shaped-reward** RL is a **favourable tradeoff**
— more energy saved AND fewer dropped calls, at slightly less throughput (better on 2 of 3 axes), not a
strict all-axis win. Full explanation in `RESULTS.md`.

---

## 9. Regenerate the figures (numbers computed live from run folders)

```bash
$PY plot_frontier.py     # reads rl/frontier_runs.tsv -> tradeoff.png
$PY plot_summary.py      # reads rl/summary_runs.tsv  -> summary_tradeoff.png
```
Both read a **manifest of run folders** (not hard-coded numbers) and recompute every metric via
`plot_energy.summarize()`. To point them at your own runs, edit the `.tsv` manifests (or pass
`group@tag=folder` / `group=folder` on the command line) — see each script's header.

---

## 10. Gotchas (learned the hard way)

- **Clean semaphores** between runs: `rm -f /dev/shm/sem.*` (a crashed run leaves them and the next
  run hangs).
- **Never rebuild ns-3 while a gym/PPO run is live** — it relaunches the binary each episode and a
  mid-run rebuild corrupts it.
- **Each run writes a NEW `output/<uuid>/` folder** (nothing is overwritten). Find the newest with
  `ls -dt output/*/ | head -1`; the run scripts also print their folder.
- **Compare energy per-step, not per-run total** — runs can differ in step count.
- **`heuristicType=-1`** in the config = "external controller decides" (the gym drives it via shared
  files + semaphores; no external RIC is required for these runs).
- ns-3 is the speed bottleneck (~15–20 s/step); a GPU does not help.
