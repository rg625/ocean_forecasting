#!/bin/bash
# Time-scale mixture readout for every finished ablation job (2026-10-26). Per .done run with a
# checkpoint: sigma calibrated on train.nc rollouts (calibrate_mix_sigma.py -> RUN/mix_sigma.json),
# test-case evaluation with that sigma (eval_ctkae.py --mix_sigma -> RUN/eval/last_mix<s>.json),
# forecasting scores incl. LSiM (test_score.py --mix-sigma --no-da -> RUN/test_score_mix.json; the
# DA does not depend on the readout and stays in da_abl). Each step is skipped if its output
# exists, so the script is safe to re-run. Same serialised GPU start as the ablation runner.
# The table: ablation_table.py --mix.
#   usage: mix_postprocess.sh [TAG ...]   (default: every tag; e.g. ref back003 back001)
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
PY=/home/rg625/anaconda3/envs/samudra/bin/python
S=../audit_report/scripts
LOCK=model_outputs_ctkae/ablation_state
wait_gpu () { until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge 4500 ]; do sleep 60; done; }
gpu_start () { exec 9>$LOCK/gpu.flock; flock 9; wait_gpu; "$@" & local pid=$!; sleep 60; flock -u 9; exec 9>&-; wait $pid; }
if [ $# -gt 0 ]; then FILES=$(for t in "$@"; do ls $LOCK/${t}_s*.done 2>/dev/null; done); else FILES=$(ls $LOCK/*.done); fi
for f in $FILES; do
  r=$(cat $f); [ -f $r/last.pth ] || continue
  [ -f $r/mix_sigma.json ] || gpu_start $PY -W ignore $S/calibrate_mix_sigma.py $r > /dev/null 2>&1
  [ -f $r/mix_sigma.json ] || { echo "!!! $(basename $f): calibration failed"; continue; }
  sg=$($PY -c "import json;print('%g' % json.load(open('$r/mix_sigma.json'))['sigma'])")
  if [ "$sg" = 0 ]; then
    # sigma 0 is the single decode: its scores are test_score.json
    [ -f $r/test_score_mix.json ] || cp $r/test_score.json $r/test_score_mix.json
    continue
  fi
  [ -f $r/eval/last_mix$sg.json ] || gpu_start $PY -W ignore eval_ctkae.py $r --ckpt last.pth --mix_sigma $sg > /dev/null 2>&1
  [ -f $r/test_score_mix.json ] || gpu_start $PY -W ignore $S/test_score.py $r --mix-sigma $sg --no-da --json $r/test_score_mix.json > /dev/null 2>&1
  echo "--- $(date) $(basename $f .done) sigma $sg $([ -f $r/test_score_mix.json ] && echo ok || echo FAILED)"
done
echo "=== mix_postprocess done $(date)"
