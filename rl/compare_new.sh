#!/bin/bash
# Seed-555 comparison: heuristic vs the (deterministic) PPO policy vs the targeted
# pruning controller at two grace levels. Scores each and overlays on a tradeoff.
set -u
cd /workspace/ns-o-ran-gym
PY=/workspace/.venv/bin/python
NS3=/workspace/ns-3-mmwave-oran
BASE=src/environments/scenario_configurations/es_use_case.json
LOG=/tmp/claude-0/-workspace/f27d95ac-61f5-49b8-b643-88876af515ab/scratchpad
mkdir -p "$LOG"
FLT='cuda\|torch._c\|userwarning\|warn('

echo "=== heuristic (555) ==="
rm -f /dev/shm/sem.* 2>/dev/null
PYTHONPATH=src $PY heuristic_twin.py --ns3_path "$NS3" --config "$BASE" --num_steps 58 --optimized 2>&1 | grep -vi "$FLT" | tail -2
HEU=$(ls -dt output/*/ | head -1)

echo "=== RL deterministic ppo_final (555) ==="
rm -f /dev/shm/sem.* 2>/dev/null
PYTHONPATH=src $PY rl/eval_rl.py --model rl/ppo_final.zip --stats rl/obs_stats.npz --ns3_path "$NS3" --config "$BASE" --num_steps 58 2>&1 | grep -vi "$FLT" | tee "$LOG/rl.log" | tail -2
RL=$(grep "RL run folder" "$LOG/rl.log" | awk '{print $NF}')

echo "=== prune grace=2 (555) ==="
rm -f /dev/shm/sem.* 2>/dev/null
PYTHONPATH=src $PY rl/aggressive_ctl.py --ns3_path "$NS3" --config "$BASE" --num_steps 58 --mode prune --thr 3 --grace 2 2>&1 | grep -vi "$FLT" | tee "$LOG/p2.log" | tail -2
P2=$(grep "Run folder" "$LOG/p2.log" | awk '{print $NF}')

echo "=== prune grace=1 (555) ==="
rm -f /dev/shm/sem.* 2>/dev/null
PYTHONPATH=src $PY rl/aggressive_ctl.py --ns3_path "$NS3" --config "$BASE" --num_steps 58 --mode prune --thr 3 --grace 1 2>&1 | grep -vi "$FLT" | tee "$LOG/p1.log" | tail -2
P1=$(grep "Run folder" "$LOG/p1.log" | awk '{print $NF}')

echo; echo "FOLDERS: HEU=$HEU  RL=$RL  P2=$P2  P1=$P1"
echo; echo "=== per-run metrics (throughput / % saved / RLF / gNBs) ==="
for f in "$HEU" "$RL" "$P2" "$P1"; do echo "-- $f"; $PY plot_energy.py "$f" 2>&1 | grep -vi "$FLT" | tail -7; done

echo; echo "=== tradeoff overlay ==="
ALLON=$(cat rl/allon_baseline_folder.txt)
$PY plot_tradeoff.py "all-on@6UEs=$ALLON" "heuristic@6UEs=$HEU" "rl@6UEs=$RL" \
    "prune-g2@6UEs=$P2" "prune-g1@6UEs=$P1" \
    --out tradeoff_new.png --title "New scenario @6 UEs: heuristic vs PPO vs targeted pruning" \
    2>&1 | grep -vi "$FLT"
echo "=== COMPARE DONE -> tradeoff_new.png ==="
