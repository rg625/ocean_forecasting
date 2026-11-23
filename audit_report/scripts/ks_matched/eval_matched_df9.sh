#!/bin/bash
# Matched KS runs, evaluation reordered by priority (replaces eval_matched_rerun.sh):
#   1. forecasting (fc_matched; FC HL LB PF) and the delta_f = 9 DA campaign (exp_geometry's
#      default pin; C2 C3 NL JS, as the main repository's da_results_geometry_df9) for all five;
#   2. the delta_f = 1 campaign (S1 C1 C2 C3 NL JS) for the three not yet run.
# Steps whose output already exists are skipped. Two lanes (shared GPU: at most two jobs).
PY=/home/rg625/anaconda3/envs/samudra/bin/python
D=/home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts/ks_ctkae/ks_driver.py
F=/home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts/ks_matched/eval_fc_matched.py
M=/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder/model_outputs_ks_matched
R=/home/rg625/mnt/ocean_forecasting_impl/audit_report/results/ks_matched
L=$R/logs/lanes.log
DF1="--pin-delta-f 1 --c2-delta-l 5 10 25 50 100 200 400 700 --c3-n 2 3 5 9 17 25"
step () { # name done-marker cmd...
  [ -e "$2" ] && { echo "=== $1 skipped (exists) $(date -Is)" >> $G; return; }
  echo "=== $1 $(date -Is)" >> $G; local nm=$1; shift 2; "$@" >> $G 2>&1; echo "--- exit=$? $(date -Is)" >> $G; }
setup () { n=$1; RD=$(ls -d $M/$n/*/ | head -n 1); RD=${RD%/}; O=$R/$n; mkdir -p $O; G=$O/run.log
  C="--kae-run $RD --init-scales $O/init_scales.json --tuning $O/tuning.json"; }
first () {
  setup $1; echo "=== eval9 $n $(date -Is)" >> $L
  step fc_matched $O/fc_matched.json $PY -W ignore $F $O/fc_matched.npz $n=$RD
  step init_scales $O/init_scales.json $PY -W ignore $D init_scales $RD $O/init_scales.json
  step tune $O/tuning.json $PY -W ignore $D tune $RD $O/init_scales.json $O/tuning.json
  step long $O/long/PF_post_da_forecast.npz env KS_ONLY_KAE=1 $PY -W ignore $D run exp_long_horizon $C --test data/ks/da_test_long.nc --out-dir $O/long --sections FC HL LB PF
  step geometry_df9 $O/geometry_df9/joint_sparsity.json $PY -W ignore $D run exp_geometry $C --methods KAE-expm --out-dir $O/geometry_df9 --sections C2 C3 NL JS
  echo "--- eval9 $n done $(date -Is)" >> $L; }
second () {
  setup $1; echo "=== eval1 $n $(date -Is)" >> $L
  step geometry $O/geometry/joint_sparsity.json $PY -W ignore $D run exp_geometry $C --methods KAE-expm --out-dir $O/geometry --sections S1 C1 C2 C3 NL JS $DF1
  echo "--- eval1 $n done $(date -Is)" >> $L; }
( first kae_128_s0; first ct_ref_256_s0; touch $R/logs/.lane1_df9_done; second kae_128_s0 ) &
( first ct_ref_128_s0; first ct_back001_128_s0; first ct_back001_256_s0; echo "=== DF9 ALL DONE (lane 2) $(date -Is)" >> $L; second ct_ref_128_s0; second ct_ref_256_s0 ) &
wait
echo "=== MATCHED EVAL DONE $(date -Is)" >> $L
