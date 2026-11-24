#!/bin/bash
# delta_f = 9 DA with a warm start for every gradient-based method (replaces the rest of
# eval_matched_df9.sh): back001-256 with its best init='obs' point (tuning_warm.json; its free
# choice was 'zero'), and the U-Net with the nearest-observation warm start (ks_driver.py
# patch; lr from da_results_geometry_df9/unet_tuning.json), on the campaign's problems.
# Waits for the two geometry_df9 runs eval_matched_df9.sh left running. Two lanes.
PY=/home/rg625/anaconda3/envs/samudra/bin/python
D=/home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts/ks_ctkae/ks_driver.py
M=/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder/model_outputs_ks_matched
R=/home/rg625/mnt/ocean_forecasting_impl/audit_report/results/ks_matched
L=$R/logs/lanes.log
P_REF256=$1; P_B128=$2
waitpid () { while kill -0 $1 2>/dev/null; do sleep 60; done; echo "--- exit=? (left running by eval_matched_df9.sh) $(date -Is)" >> $2; }
step () { G=$1; echo "=== $2 $(date -Is)" >> $G; shift 2; "$@" >> $G 2>&1; echo "--- exit=$? $(date -Is)" >> $G; }
rd () { RD=$(ls -d $M/$1/*/ | head -n 1); echo ${RD%/}; }
KRD=$(rd kae_128_s0); U=$R/unet_warm; mkdir -p $U
UC="--kae-run $KRD --init-scales $R/kae_128_s0/init_scales.json --tuning $U/tuning.json --methods UNet"
(
  waitpid $P_REF256 $R/ct_ref_256_s0/run.log; echo "--- eval9 ct_ref_256_s0 done $(date -Is)" >> $L
  step $U/run.log unet_df9_C3_NL $PY -W ignore $D run exp_geometry $UC --out-dir $U/geometry_df9_a --sections C3 NL
  step $U/run.log unet_df9_C2 $PY -W ignore $D run exp_geometry $UC --out-dir $U/geometry_df9_c --sections C2
) &
(
  waitpid $P_B128 $R/ct_back001_128_s0/run.log; echo "--- eval9 ct_back001_128_s0 done $(date -Is)" >> $L
  O=$R/ct_back001_256_s0; RD=$(rd ct_back001_256_s0)
  step $O/run.log geometry_df9_warm env KS_ONLY_KAE=1 $PY -W ignore $D run exp_geometry --kae-run $RD --init-scales $O/init_scales.json --tuning $O/tuning_warm.json --methods KAE-expm --out-dir $O/geometry_df9 --sections C2 C3 NL JS
  echo "--- eval9 ct_back001_256_s0 done (warm) $(date -Is)" >> $L
  step $U/run.log unet_df9_JS $PY -W ignore $D run exp_geometry $UC --out-dir $U/geometry_df9_b --sections JS
) &
wait
echo "=== WARM DF9 DONE $(date -Is)" >> $L
