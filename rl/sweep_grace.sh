#!/bin/bash
# Map the pruning energy-vs-reliability frontier on seed 555: sweep grace (how long
# an idle cell is kept on before sleeping). High grace = conservative (only sleep
# long-idle cells) = closer to the heuristic; low grace = aggressive. Overlays the
# whole frontier against the heuristic point + the (tied) PPO point.
set -u
cd /workspace/ns-o-ran-gym
PY=/workspace/.venv/bin/python
NS3=/workspace/ns-3-mmwave-oran
BASE=src/environments/scenario_configurations/es_use_case.json
LOG=/tmp/claude-0/-workspace/f27d95ac-61f5-49b8-b643-88876af515ab/scratchpad
FLT='cuda\|torch._c\|userwarning\|warn('

# already have from compare_new.sh:
HEU=output/c5ae5296-5d0b-4dc5-8ba8-8422cbb5b017/
RL=output/fb6b1773-2876-4182-918f-83cf12f87cf8
G2=output/ff6f2015-6bdf-4060-9b15-5cdc12a9ce18
G1=output/5ceab7d8-2eb1-4e4b-b3a6-5e8d01217c44

declare -A GF
for g in 3 5 8; do
  rm -f /dev/shm/sem.* 2>/dev/null
  echo "=== prune grace=$g (555) ==="
  PYTHONPATH=src $PY rl/aggressive_ctl.py --ns3_path "$NS3" --config "$BASE" \
      --num_steps 58 --mode prune --thr 3 --grace $g 2>&1 | grep -vi "$FLT" | tee "$LOG/g$g.log" | tail -2
  GF[$g]=$(grep "Run folder" "$LOG/g$g.log" | awk '{print $NF}')
done

echo; echo "=== metrics (grace 8/5/3) ==="
for g in 8 5 3; do echo "-- grace=$g ${GF[$g]}"; $PY plot_energy.py "${GF[$g]}" 2>&1 | grep -vi "$FLT" | tail -4; done

echo; echo "=== full frontier tradeoff ==="
ALLON=$(cat rl/allon_baseline_folder.txt)
$PY plot_tradeoff.py "all-on@=$ALLON" "heuristic@=$HEU" "rl(PPO)@=$RL" \
    "prune@g8=${GF[8]}" "prune@g5=${GF[5]}" "prune@g3=${GF[3]}" "prune@g2=$G2" "prune@g1=$G1" \
    --out tradeoff_new.png --title "New scenario @6 UEs: pruning frontier vs heuristic vs PPO" \
    2>&1 | grep -vi "$FLT"
echo "=== SWEEP DONE -> tradeoff_new.png ==="
