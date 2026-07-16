#!/bin/bash
# CLEAN comparison: shaped-reward PPO vs the ORIGINAL (balanced) heuristic (min_offload=0,
# i.e. the idle-sleep fix disabled) on the uniform-burst scenario, seeds 555/777/999/1234.
# Averages energy / throughput / RLF per policy. This is the controlled "does RL beat the
# heuristic" test (the earlier eval used the over-aggressive FIXED heuristic).
set -u
cd /workspace/ns-o-ran-gym
PY=/workspace/.venv/bin/python
NS3=/workspace/ns-3-mmwave-oran
BASE=src/environments/scenario_configurations/es_use_case.json
S=/tmp/claude-0/-workspace/f27d95ac-61f5-49b8-b643-88876af515ab/scratchpad
FLT='cuda\|torch._c\|userwarning\|warn('
: > "$S/clean_rl.txt"; : > "$S/clean_heu.txt"

for seed in 555 777 999 1234; do
  cfg=rl/cfgs/fixed_${seed}.json
  $PY -c "import json;c=json.load(open('$BASE'));c['ues']=[6];c['RngRun']=[$seed];json.dump(c,open('$cfg','w'))"
  echo "############ seed $seed ############"
  rm -f /dev/shm/sem.* 2>/dev/null
  echo "-- RL (shaped) --"
  PYTHONPATH=src $PY rl/eval_rl.py --model rl/ppo_shaped.zip --stats rl/obs_stats.npz \
      --ns3_path "$NS3" --config "$cfg" --num_steps 58 2>&1 | grep -vi "$FLT" | tee "$S/cr.log" | tail -1
  grep "RL run folder" "$S/cr.log" | awk '{print $NF}' >> "$S/clean_rl.txt"
  rm -f /dev/shm/sem.* 2>/dev/null
  echo "-- ORIGINAL heuristic (min_offload=0) --"
  PYTHONPATH=src $PY heuristic_twin.py --ns3_path "$NS3" --config "$cfg" --num_steps 58 \
      --optimized --min_offload 0 2>&1 | grep -vi "$FLT" | tail -1
  ls -dt output/*/ | head -1 >> "$S/clean_heu.txt"
done

echo; echo "=== AGGREGATE (mean over 4 seeds) ==="
$PY - <<'PYEOF'
import re, subprocess, numpy as np
S="/tmp/claude-0/-workspace/f27d95ac-61f5-49b8-b643-88876af515ab/scratchpad"
def metrics(folder):
    o=subprocess.run(["/workspace/.venv/bin/python","plot_energy.py",folder.strip()],
                     capture_output=True,text=True).stdout
    return (float(re.search(r"saved vs.*?:\s*([\d.]+)%",o).group(1)),
            float(re.search(r"throughput\s*:\s*([\d.]+)",o).group(1)),
            float(re.search(r"mean RLF\s*:\s*([\d.]+)",o).group(1)))
res={}
for name,ff in [("RL (shaped)",f"{S}/clean_rl.txt"),("orig heuristic",f"{S}/clean_heu.txt")]:
    rows=np.array([metrics(l) for l in open(ff) if l.strip()])
    res[name]=rows
    print(f"{name:16s} n={len(rows)}  saved%={rows[:,0].mean():5.1f}  thr={rows[:,1].mean():.2f} Mbps  RLF={rows[:,2].mean():.2f}"
          f"   (per-seed saved%: {[round(x,1) for x in rows[:,0]]})")
if "RL (shaped)" in res and "orig heuristic" in res:
    rl,hu=res["RL (shaped)"].mean(0),res["orig heuristic"].mean(0)
    print(f"\nRL vs ORIGINAL heuristic:  dEnergy={rl[0]-hu[0]:+.1f} pts  dThroughput={rl[1]-hu[1]:+.2f} Mbps  dRLF={rl[2]-hu[2]:+.2f}")
    win = rl[0]>=hu[0] and rl[1]>=hu[1] and rl[2]<=hu[2]
    print("VERDICT:", "RL DOMINATES (>= energy, >= throughput, <= RLF) - a clean win" if win
          else "mixed / tradeoff - inspect the axes above")
PYEOF
echo "=== CLEAN COMPARE DONE ==="
