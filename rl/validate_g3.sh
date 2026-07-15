#!/bin/bash
# Validate the grace=3 pruning win on fresh seeds (555 already done). For each seed,
# run heuristic and prune-g3 on the SAME load, then average energy/throughput/RLF
# per policy across all seeds. A robust win = g3 saves more energy at RLF no worse
# and throughput within ~noise, on average (not just the lucky seed 555).
set -u
cd /workspace/ns-o-ran-gym
PY=/workspace/.venv/bin/python
NS3=/workspace/ns-3-mmwave-oran
BASE=src/environments/scenario_configurations/es_use_case.json
LOG=/tmp/claude-0/-workspace/f27d95ac-61f5-49b8-b643-88876af515ab/scratchpad
FLT='cuda\|torch._c\|userwarning\|warn('

# seed 555 already computed:
echo "output/c5ae5296-5d0b-4dc5-8ba8-8422cbb5b017/" > "$LOG/heu_folders.txt"
echo "output/8b5aac85-c425-4319-839b-d26901681cf5" > "$LOG/g3_folders.txt"

for seed in 777 999 1234; do
  cfg=rl/cfgs/new_ues6_rng${seed}.json
  $PY -c "import json;c=json.load(open('$BASE'));c['ues']=[6];c['RngRun']=[$seed];json.dump(c,open('$cfg','w'))"
  rm -f /dev/shm/sem.* 2>/dev/null
  echo "=== heuristic seed=$seed ==="
  PYTHONPATH=src $PY heuristic_twin.py --ns3_path "$NS3" --config "$cfg" --num_steps 58 --optimized 2>&1 | grep -vi "$FLT" | tail -1
  ls -dt output/*/ | head -1 >> "$LOG/heu_folders.txt"
  rm -f /dev/shm/sem.* 2>/dev/null
  echo "=== prune-g3 seed=$seed ==="
  PYTHONPATH=src $PY rl/aggressive_ctl.py --ns3_path "$NS3" --config "$cfg" --num_steps 58 --mode prune --thr 3 --grace 3 2>&1 | grep -vi "$FLT" | tee "$LOG/vg3_$seed.log" | tail -1
  grep "Run folder" "$LOG/vg3_$seed.log" | awk '{print $NF}' >> "$LOG/g3_folders.txt"
done

echo; echo "=== AGGREGATE (mean over seeds 555/777/999/1234) ==="
$PY - <<PYEOF
import re, subprocess
def metrics(folder):
    out = subprocess.run(["$PY","plot_energy.py",folder.strip()],capture_output=True,text=True).stdout
    saved = float(re.search(r"energy saved vs.*?:\s*([\d.]+)%",out).group(1))
    thr   = float(re.search(r"mean throughput\s*:\s*([\d.]+)",out).group(1))
    rlf   = float(re.search(r"mean RLF\s*:\s*([\d.]+)",out).group(1))
    return saved,thr,rlf
import numpy as np
for name,ff in [("heuristic","$LOG/heu_folders.txt"),("prune-g3","$LOG/g3_folders.txt")]:
    rows=[metrics(l) for l in open(ff) if l.strip()]
    a=np.array(rows)
    print(f"{name:10s}  n={len(rows)}  saved%={a[:,0].mean():5.1f}  thr={a[:,1].mean():.2f} Mbps  RLF={a[:,2].mean():.2f}   (per-seed saved%: {[round(x,1) for x in a[:,0]]})")
PYEOF
# write the summary manifest (folder pointers only) + render live
ALLON=$(cat rl/allon_baseline_folder.txt)
{ printf 'allon\t%s\n' "$ALLON"
  awk 'NF{print "heur\t"$0}'  "$LOG/heu_folders.txt"
  awk 'NF{print "prune\t"$0}' "$LOG/g3_folders.txt"
} > rl/summary_runs.tsv
$PY plot_summary.py 2>&1 | grep -vi "$FLT"
echo "=== VALIDATE DONE -> summary_tradeoff.png (numbers computed live from folders) ==="
