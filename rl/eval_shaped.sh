#!/bin/bash
# Definitive eval: shaped-reward PPO vs the FIXED heuristic on the uniform-burst scenario
# (seed 555 + held-out 1234), scored on energy / throughput / RLF.
set -u
cd /workspace/ns-o-ran-gym
PY=/workspace/.venv/bin/python
NS3=/workspace/ns-3-mmwave-oran
BASE=src/environments/scenario_configurations/es_use_case.json
LOG=/tmp/claude-0/-workspace/f27d95ac-61f5-49b8-b643-88876af515ab/scratchpad
FLT='cuda\|torch._c\|userwarning\|warn('

score() { $PY plot_energy.py "$1" 2>&1 | grep -iE "saved vs|throughput|mean RLF|mean gNBs"; }

for seed in 555 1234; do
  cfg=rl/cfgs/fixed_${seed}.json
  $PY -c "import json;c=json.load(open('$BASE'));c['ues']=[6];c['RngRun']=[$seed];json.dump(c,open('$cfg','w'))"
  echo "############ seed $seed ############"
  rm -f /dev/shm/sem.* 2>/dev/null
  echo "-- RL (shaped) --"
  PYTHONPATH=src $PY rl/eval_rl.py --model rl/ppo_shaped.zip --stats rl/obs_stats.npz \
      --ns3_path "$NS3" --config "$cfg" --num_steps 58 2>&1 | grep -vi "$FLT" | tee "$LOG/es_rl_$seed.log" | tail -1
  RL=$(grep "RL run folder" "$LOG/es_rl_$seed.log" | awk '{print $NF}')
  rm -f /dev/shm/sem.* 2>/dev/null
  echo "-- fixed heuristic --"
  PYTHONPATH=src $PY heuristic_twin.py --ns3_path "$NS3" --config "$cfg" --num_steps 58 --optimized \
      2>&1 | grep -vi "$FLT" | tail -1
  HEU=$(ls -dt output/*/ | head -1)
  echo "  [RL shaped]   $RL"; score "$RL"
  echo "  [heuristic]   $HEU"; score "$HEU"
done
echo "=== EVAL SHAPED DONE ==="
