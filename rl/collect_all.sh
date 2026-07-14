#!/bin/bash
# Collect expert demonstrations across loads (ues) and seeds (RngRun) for BC.
# Each run is simulator-bound (~10-12 min). Combines into rl/demos.npz.
set -u
cd /workspace/ns-o-ran-gym
PY=/workspace/.venv/bin/python
NS3=/workspace/ns-3-mmwave-oran
BASE=src/environments/scenario_configurations/es_use_case.json
mkdir -p rl/cfgs rl/demos

for ues in 3 6 9; do
  for seed in 555 777; do
    cfg=rl/cfgs/ues${ues}_rng${seed}.json
    $PY -c "import json;c=json.load(open('$BASE'));c['ues']=[$ues];c['RngRun']=[$seed];json.dump(c,open('$cfg','w'))"
    rm -f /dev/shm/sem.* 2>/dev/null
    echo "=== collecting  ues=$ues  seed=$seed ==="
    PYTHONPATH=src $PY rl/collect_demos.py --ns3_path "$NS3" --config "$cfg" \
        --num_steps 58 --out rl/demos/ues${ues}_rng${seed}.npz \
        2>&1 | grep -vi "cuda\|torch._c\|userwarning" | tail -2
  done
done

# combine all per-run files into one dataset
$PY - <<'EOF'
import glob, numpy as np
files = sorted(glob.glob("rl/demos/ues*_rng*.npz"))
O, A, cols = [], [], None
for f in files:
    d = np.load(f, allow_pickle=True)
    O.append(d["observations"]); A.append(d["actions"]); cols = d["columns"]
O = np.concatenate(O); A = np.concatenate(A)
np.savez_compressed("rl/demos.npz", observations=O, actions=A, columns=cols)
print(f"combined {len(files)} files -> observations {O.shape}, actions {A.shape}")
EOF
echo "=== DEMOS DONE ==="
