#!/bin/bash
# DA learning-rate re-tuning for finished ablation jobs (2026-10-24). If a run's tuned lr sits at
# the edge of its grid (tuning on gt_extrap), the grid is extended outward by factors of 3-10 and
# the tune + the four df9 sweeps (da_abl) and test_score's df9 problem are re-run; repeated until
# the optimum is interior (max 3 rounds). Safe to re-run: interior rows are skipped.
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
PY=/home/rg625/anaconda3/envs/samudra/bin/python
S=../audit_report/scripts
wait_gpu () { until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge 4500 ]; do sleep 60; done; }
for f in model_outputs_ctkae/ablation_state/*.done; do
  r=$(cat $f); lab=ctkae_$(basename $r); T=model_outputs_ctkae/da_abl/$lab/df9/tuning.json
  for round in 1 2 3; do
    [ -f $T ] || break
    G=$($PY - "$T" <<'PYEOF'
import json, sys
t = json.load(open(sys.argv[1]))["KAE"]
lrs = sorted(r["lr"] for r in t["sweep"]); lr = t["lr"]
if lr == lrs[-1]:
    print(" ".join(f"{lr * k:g}" for k in (1, 3, 10, 30)))
elif lr == lrs[0]:
    print(" ".join(f"{lr / k:g}" for k in (30, 10, 3, 1)))
PYEOF
)
    [ -z "$G" ] && break
    echo "=== $(date) $lab round $round: edge -> grid $G"
    wait_gpu
    DF9_ONLY=1 DF9_SECTIONS="delta_l n_obs sparsity noise" LRS="$G" DA_ROOT=da_abl bash $S/run_da_v2.sh $lab ctkae $r last.pth
    $PY -W ignore $S/test_score.py $r --no-lsim --lrs $G --json $r/test_score_da.json 2>&1 | grep -v INFO | tail -1 | cut -c1-120
    $PY - "$r" <<'PYEOF'
import json, sys
from pathlib import Path
r = Path(sys.argv[1]); ts = json.loads((r / "test_score.json").read_text())
new = json.loads((r / "test_score_da.json").read_text())
ts["da_tune_first"] = ts.get("da_tune_first", ts["da_tune"])
ts.update(da_df9=new["da_df9"], da_lr=new["da_lr"], da_tune=new["da_tune"])
(r / "test_score.json").write_text(json.dumps(ts, indent=1))
PYEOF
  done
done
echo "=== $(date) RETUNE DONE"
