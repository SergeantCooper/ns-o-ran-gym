#!/bin/bash
# Post-PPO evaluation on the new (burst) scenario. Runs the learned policy and the
# heuristic on the SAME seeds (identical traffic), on the training seed (555) and a
# HELD-OUT seed (1234), then scores each and overlays RL vs heuristic on the tradeoff.
# Launch ONLY after the PPO run has finished (needs the CPU + rl/ppo_final.zip).
set -u
cd /workspace/ns-o-ran-gym
PY=/workspace/.venv/bin/python
NS3=/workspace/ns-3-mmwave-oran
BASE=src/environments/scenario_configurations/es_use_case.json
LOG=/tmp/claude-0/-workspace/f27d95ac-61f5-49b8-b643-88876af515ab/scratchpad
mkdir -p "$LOG" rl/cfgs
FLT='cuda\|torch._c\|userwarning\|warn('

run_rl () {  # $1=config  $2=logfile -> echoes folder
  rm -f /dev/shm/sem.* 2>/dev/null
  PYTHONPATH=src $PY rl/eval_rl.py --model rl/ppo_final.zip --stats rl/obs_stats.npz \
      --ns3_path "$NS3" --config "$1" --num_steps 58 2>&1 | grep -vi "$FLT" | tee "$2"
}
run_heu () { # $1=config -> newest folder
  rm -f /dev/shm/sem.* 2>/dev/null
  PYTHONPATH=src $PY heuristic_twin.py --ns3_path "$NS3" --config "$1" \
      --num_steps 58 --optimized 2>&1 | grep -vi "$FLT" | tail -3
}

echo "=== RL eval, seed 555 (training) ===";   run_rl "$BASE" "$LOG/rl555.log"
RL555=$(grep "RL run folder" "$LOG/rl555.log" | awk '{print $NF}')
echo "=== heuristic, seed 555 ===";            run_heu "$BASE"
HEU555=$(ls -dt output/*/ | head -1)

CFG=rl/cfgs/new_ues6_rng1234.json
$PY -c "import json;c=json.load(open('$BASE'));c['RngRun']=[1234];json.dump(c,open('$CFG','w'))"
echo "=== RL eval, seed 1234 (held-out) ===";  run_rl "$CFG" "$LOG/rl1234.log"
RL1234=$(grep "RL run folder" "$LOG/rl1234.log" | awk '{print $NF}')
echo "=== heuristic, seed 1234 ===";           run_heu "$CFG"
HEU1234=$(ls -dt output/*/ | head -1)

echo; echo "RL555=$RL555"; echo "HEU555=$HEU555"; echo "RL1234=$RL1234"; echo "HEU1234=$HEU1234"

echo; echo "=== per-run metrics (throughput / energy / RLF) ==="
for f in "$RL555" "$HEU555" "$RL1234" "$HEU1234"; do
  echo "-- $f"; $PY plot_energy.py "$f" 2>&1 | grep -vi "$FLT" | tail -8
done

echo; echo "Folders + metrics printed above. Figures via plot_frontier.py / plot_summary.py."
echo "=== EVAL DONE ==="
