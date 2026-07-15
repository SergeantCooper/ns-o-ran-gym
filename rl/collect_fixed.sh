#!/bin/bash
# Fresh expert demos from the FIXED heuristic (uniform burst scenario, seeds 555/777/999)
# + an all-on baseline, combined into rl/demos.npz. The old demos were from the pre-fix
# heuristic, so they must be regenerated before BC/PPO on the new reward.
set -u
cd /workspace/ns-o-ran-gym
PY=/workspace/.venv/bin/python
NS3=/workspace/ns-3-mmwave-oran
BASE=src/environments/scenario_configurations/es_use_case.json
FLT='cuda\|torch._c\|userwarning\|warn('
mkdir -p rl/cfgs rl/demos

for seed in 555 777 999; do
  cfg=rl/cfgs/fixed_${seed}.json
  $PY -c "import json;c=json.load(open('$BASE'));c['ues']=[6];c['RngRun']=[$seed];json.dump(c,open('$cfg','w'))"
  rm -f /dev/shm/sem.* 2>/dev/null
  echo "=== demos (fixed heuristic) seed=$seed ==="
  PYTHONPATH=src $PY rl/collect_demos.py --ns3_path "$NS3" --config "$cfg" \
      --num_steps 58 --out rl/demos/fixed_${seed}.npz 2>&1 | grep -vi "$FLT" | tail -1
done

rm -f /dev/shm/sem.* 2>/dev/null
echo "=== all-on baseline (fixed heuristic irrelevant here; --anchor all) ==="
PYTHONPATH=src $PY heuristic_twin.py --ns3_path "$NS3" --config "$BASE" \
    --num_steps 58 --optimized --anchor 2 3 4 5 6 7 8 2>&1 | grep -vi "$FLT" | tail -1
echo "$(ls -dt output/*/ | head -1)" > rl/allon_baseline_folder.txt
echo "ALLON_FOLDER=$(cat rl/allon_baseline_folder.txt)"

$PY - <<'PYEOF'
import glob, numpy as np
files = sorted(glob.glob("rl/demos/fixed_*.npz"))
O, A, cols = [], [], None
for f in files:
    d = np.load(f, allow_pickle=True); O.append(d["observations"]); A.append(d["actions"]); cols = d["columns"]
O = np.concatenate(O); A = np.concatenate(A)
np.savez_compressed("rl/demos.npz", observations=O, actions=A, columns=cols)
print(f"combined {len(files)} -> demos.npz {O.shape}")
PYEOF
echo "=== FIXED DEMOS DONE ==="
