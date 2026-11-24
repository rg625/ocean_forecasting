#!/bin/bash
# Evaluate the matched runs that eval_matched_queue.sh skipped: its wait matched the first,
# OOM-failed attempts' "exit=1" lines in lanes.log. Same steps as the queue; two lanes.
PY=/home/rg625/anaconda3/envs/samudra/bin/python
D=/home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts/ks_ctkae/ks_driver.py
F=/home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts/ks_matched/eval_fc_matched.py
M=/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder/model_outputs_ks_matched
R=/home/rg625/mnt/ocean_forecasting_impl/audit_report/results/ks_matched
L=$R/logs/lanes.log
ev () {
  n=$1; RD=$(ls -d $M/$n/*/ | head -n 1); RD=${RD%/}; O=$R/$n; mkdir -p $O; G=$O/run.log
  echo "=== eval $n $(date -Is)" >> $L
  step () { echo "=== $1 $(date -Is)" >> $G; shift; "$@" >> $G 2>&1; echo "--- exit=$? $(date -Is)" >> $G; }
  step fc_matched $PY -W ignore $F $O/fc_matched.npz $n=$RD
  step init_scales $PY -W ignore $D init_scales $RD $O/init_scales.json
  step tune $PY -W ignore $D tune $RD $O/init_scales.json $O/tuning.json
  C="--kae-run $RD --init-scales $O/init_scales.json --tuning $O/tuning.json"
  step long env KS_ONLY_KAE=1 $PY -W ignore $D run exp_long_horizon $C --test data/ks/da_test_long.nc --out-dir $O/long --sections FC HL LB PF
  step geometry $PY -W ignore $D run exp_geometry $C --methods KAE-expm --out-dir $O/geometry --sections S1 C1 C2 C3 NL JS --pin-delta-f 1 --c2-delta-l 5 10 25 50 100 200 400 700 --c3-n 2 3 5 9 17 25
  echo "--- eval $n done $(date -Is)" >> $L
}
( ev kae_128_s0; ev ct_ref_256_s0 ) &
( ev ct_ref_128_s0 ) &
wait
echo "=== RERUN EVAL DONE $(date -Is)" >> $L
