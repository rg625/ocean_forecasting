#!/bin/bash
# PF (forecast from each method's own analysis) for seed 0 of both candidates; re-run after the
# problem fallback for the informed first guess (ks_driver.run)
K=/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder/model_outputs_ctkae_ks
O=/home/rg625/mnt/ocean_forecasting_impl/audit_report/results/ks_ctkae
PY=/home/rg625/anaconda3/envs/samudra/bin/python
D=/home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts/ks_ctkae/ks_driver.py
for m in ref back001; do
  R=$(ls -d $K/ks_${m}_s0_*); L=$O/${m}_s0/run.log
  echo "=== pf $(date -Is)" >> $L
  KS_ONLY_KAE=1 $PY -W ignore $D run exp_long_horizon --kae-run $R --init-scales $O/${m}_s0/init_scales.json --tuning $O/${m}_s0/tuning.json --test data/ks/da_test_long.nc --out-dir $O/${m}_s0/long --sections PF >> $O/${m}_s0/pf.log 2>&1
  echo "--- pf exit=$? $(date -Is)" >> $L
done
echo "=== KS PF DONE $(date)"
