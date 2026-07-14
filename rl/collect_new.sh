#!/bin/bash
# New (burst) scenario: collect a few more demo seeds at ues=6 + an all-on baseline,
# then combine into rl/demos.npz. (seed 555 already collected.)
set -u
cd /workspace/ns-o-ran-gym
PY=/workspace/.venv/bin/python
NS3=/workspace/ns-3-mmwave-oran
BASE=src/environments/scenario_configurations/es_use_case.json
mkdir -p rl/cfgs rl/demos

for seed in 777 999; do
  cfg=rl/cfgs/new_ues6_rng${seed}.json
  $PY -c "import json;c=json.load(open('$BASE'));c['ues']=[6];c['RngRun']=[$seed];json.dump(c,open('$cfg','w'))"
  rm -f /dev/shm/sem.* 2>/dev/null
  echo "=== demos ues=6 seed=$seed ==="
  PYTHONPATH=src $PY rl/collect_demos.py --ns3_path "$NS3" --config "$cfg" \
      --num_steps 58 --out rl/demos/new_ues6_rng${seed}.npz \
      2>&1 | grep -vi "cuda\|torch._c\|userwarning\|warn(" | tail -2
done

# all-on baseline (ues=6, seed 555) = tradeoff reference on the new scenario
rm -f /dev/shm/sem.* 2>/dev/null
echo "=== all-on baseline ues=6 ==="
PYTHONPATH=src $PY heuristic_twin.py --ns3_path "$NS3" --config "$BASE" \
    --num_steps 58 --optimized --anchor 2 3 4 5 6 7 8 \
    2>&1 | grep -vi "cuda\|torch._c\|userwarning\|warn(" | tail -2
echo "ALLON_FOLDER=$(ls -dt output/*/ | head -1)"

# combine all new-scenario ues=6 demos into rl/demos.npz
$PY - <<'EOF'
import glob, numpy as np
files = sorted(glob.glob("rl/demos/new_ues6_rng*.npz"))
O, A, cols = [], [], None
for f in files:
    d = np.load(f, allow_pickle=True)
    O.append(d["observations"]); A.append(d["actions"]); cols = d["columns"]
O = np.concatenate(O); A = np.concatenate(A)
np.savez_compressed("rl/demos.npz", observations=O, actions=A, columns=cols)
print(f"combined {len(files)} files -> observations {O.shape}, actions {A.shape}")
EOF
echo "=== NEW-SCENARIO DEMOS + BASELINE DONE ==="
