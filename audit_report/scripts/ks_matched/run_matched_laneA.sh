#!/bin/bash
# Lane A of run_matched.sh re-queued: a CT-KAE run at the paper's batch (1024) takes ~17.5 GB,
# so the runs go one at a time, after lane B ("ALL DONE" in lanes.log).
PY=/home/rg625/anaconda3/envs/samudra/bin/python
T=/home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts/ks_matched/train_matched.py
O=/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder/model_outputs_ks_matched
L=/home/rg625/mnt/ocean_forecasting_impl/audit_report/results/ks_matched/logs
until grep -q "ALL DONE" $L/lanes.log; do sleep 120; done
run () { n=$1; shift; echo "=== $n $(date -Is)" >> $L/lanes.log; $PY -W ignore $T "$@" > $L/$n.log 2>&1; echo "--- $n exit=$? $(date -Is)" >> $L/lanes.log; }
run kae_128_s0 kae $O/kae_128_s0
run ct_ref_128_s0 ct $O/ct_ref_128_s0 --latent 128
run ct_ref_256_s0 ct $O/ct_ref_256_s0 --latent 256
echo "=== LANE A DONE $(date -Is)" >> $L/lanes.log
