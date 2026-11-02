#!/bin/bash
# The five matched KS runs (paper KAE settings; train_matched.py), seed 0, two lanes.
#   lane A: paper KAE (control), CT-KAE ref 128, CT-KAE ref 256
#   lane B: CT-KAE back001 128, CT-KAE back001 256
PY=/home/rg625/anaconda3/envs/samudra/bin/python
T=/home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts/ks_matched/train_matched.py
O=/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder/model_outputs_ks_matched
L=/home/rg625/mnt/ocean_forecasting_impl/audit_report/results/ks_matched/logs
mkdir -p $O $L
run () { n=$1; shift; echo "=== $n $(date -Is)" >> $L/lanes.log; $PY -W ignore $T "$@" > $L/$n.log 2>&1; echo "--- $n exit=$? $(date -Is)" >> $L/lanes.log; }
( run kae_128_s0 kae $O/kae_128_s0
  run ct_ref_128_s0 ct $O/ct_ref_128_s0 --latent 128
  run ct_ref_256_s0 ct $O/ct_ref_256_s0 --latent 256 ) &
( run ct_back001_128_s0 ct $O/ct_back001_128_s0 --latent 128 --w-back 0.01
  run ct_back001_256_s0 ct $O/ct_back001_256_s0 --latent 256 --w-back 0.01 ) &
wait
echo "=== ALL DONE $(date -Is)" >> $L/lanes.log
