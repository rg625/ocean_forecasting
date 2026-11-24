#!/bin/bash
# Full evaluation of the back001 candidate (w_back 0.01 + mixture readout, 2026-10-27), the
# same set the reference H8_d256 got, plus a long-horizon mixture calibration for both:
#  0. check: the chunked calibrate_mix_sigma.py reproduces ref s0's stored train MSEs
#  1. sigma calibrated on 238-lead TRAIN rollouts (RUN/mix_sigma_long.json) for ref and
#     back001, seeds 0-2; test scores with that sigma -> RUN/test_score_mixlong.json
#  2. eval --dt_test --ma_sweep for ref and back001, seeds 0-2 (eval/last.json is copied to
#     eval/last_pre_dt.json first; the forecast scores are recomputed identically)
#  3. DA in da_v3 as the reference: seed 0 the full suite (df1 + df9), seeds 1-2 all six
#     df9 sections; lr grid 0.3-10 (da_v3)
#  4. DA conditioning diagnostic (da_conditioning.py) for back001 seeds 0-2 and ref seeds 1-2
#     (ref seed 0 is in da_v3/conditioning already)
# Every step is skipped if its output exists. Same serialised GPU start as the ablation runner.
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
PY=/home/rg625/anaconda3/envs/samudra/bin/python
S=../audit_report/scripts
ST=model_outputs_ctkae/ablation_state
wait_gpu () { until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge $1 ]; do sleep 60; done; }
gpu_start () { local need=$1; shift; exec 9>$ST/gpu.flock; flock 9; wait_gpu $need; "$@" & local pid=$!; sleep 90; flock -u 9; exec 9>&-; wait $pid; }
run_of () { cat $ST/$1.done; }
lr_of () { $PY -c "import json;print(json.load(open('model_outputs_ctkae/da_v3/$1/df9/tuning.json'))['KAE']['lr'])"; }
sigma_of () { $PY -c "import json;print('%g' % json.load(open('$1'))['sigma'])"; }

echo "=== $(date) 0. calibration check"
R0=$(run_of ref_s0); CHK=$R0/mix_sigma_check.json
[ -f $CHK ] || gpu_start 4500 $PY -W ignore $S/calibrate_mix_sigma.py $R0 --out mix_sigma_check.json > /dev/null 2>&1
$PY -c "
import json; a=json.load(open('$R0/mix_sigma.json'))['train_mse']; b=json.load(open('$CHK'))['train_mse']
d=max(abs(x-y)/x for x,y in zip(a,b)); print('max rel diff vs stored', d); assert d < 1e-3"
[ $? -eq 0 ] || { echo "!!! chunked calibration differs from the stored one; stopping"; exit 1; }

for t in ref back001; do for s in 0 1 2; do
  r=$(run_of ${t}_s$s)
  echo "=== $(date) 1. long-horizon sigma ${t}_s$s"
  [ -f $r/mix_sigma_long.json ] || gpu_start 4500 $PY -W ignore $S/calibrate_mix_sigma.py $r --leads 238 --starts 300 500 --out mix_sigma_long.json > /dev/null 2>&1
  [ -f $r/mix_sigma_long.json ] || { echo "!!! calibration failed"; continue; }
  sg=$(sigma_of $r/mix_sigma_long.json); echo "    sigma_long $sg (58-lead sigma $(sigma_of $r/mix_sigma.json))"
  if [ "$sg" = 0 ]; then
    [ -f $r/test_score_mixlong.json ] || cp $r/test_score.json $r/test_score_mixlong.json
  else
    [ -f $r/eval/last_mix$sg.json ] || gpu_start 4500 $PY -W ignore eval_ctkae.py $r --ckpt last.pth --mix_sigma $sg > /dev/null 2>&1
    [ -f $r/test_score_mixlong.json ] || gpu_start 4500 $PY -W ignore $S/test_score.py $r --mix-sigma $sg --no-da --json $r/test_score_mixlong.json > /dev/null 2>&1
  fi
done; done

for t in back001 ref; do for s in 0 1 2; do
  r=$(run_of ${t}_s$s)
  $PY -c "import json,sys;sys.exit(0 if 'dt_test' in json.load(open('$r/eval/last.json')) else 1)" && continue
  echo "=== $(date) 2. dt_test + ma_sweep ${t}_s$s"
  [ -f $r/eval/last_pre_dt.json ] || cp $r/eval/last.json $r/eval/last_pre_dt.json
  gpu_start 4500 $PY -W ignore eval_ctkae.py $r --ckpt last.pth --dt_test --ma_sweep > /dev/null 2>&1
done; done

for s in 0 1 2; do
  r=$(run_of back001_s$s); lab=ctkae_back001_s$s
  grep -q "=== DONE" model_outputs_ctkae/da_v3/$lab/run.log 2>/dev/null && continue
  echo "=== $(date) 3. DA da_v3 $lab"
  if [ $s = 0 ]; then DF9=0; else DF9=1; fi
  DF9_ONLY=$DF9 LRS="0.3 1.0 3.0 10.0" DA_ROOT=da_v3 gpu_start 8000 bash $S/run_da_v2.sh $lab ctkae $r last.pth
done

COND=model_outputs_ctkae/da_v3/conditioning
for k in back001_s0 back001_s1 back001_s2 ref_s1 ref_s2; do
  r=$(run_of $k); s=${k##*_s}
  case $k in back001*) lab=ctkae_back001_s$s ;; ref*) lab=ctkae_H8_d256_s$s ;; esac
  [ -f $COND/$lab.json ] && continue
  echo "=== $(date) 4. conditioning $lab"
  gpu_start 8000 $PY -W ignore -m data_assimilation.tra.da_conditioning --kae-kind ctkae --kae-run $r --kae-ckpt last.pth --lr $(lr_of $lab) --json $COND/$lab.json > /dev/null 2>&1
done
echo "=== $(date) ALL DONE"
