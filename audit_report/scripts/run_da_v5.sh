#!/bin/bash
# DA with the localized probability-matched mirror readout (2026-10-30): the full suite (df1 + df9)
# on seed 2 of each model, as da_v4's plain_*_s2 / mirror_*_s2 (same launcher, grid, checkpoint).
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
S=model_outputs_ctkae/ablation_state
R () { cat $S/$1.done; }
job () {  # LABEL KEY [EXTRA...]
  local lab=$1 key=$2; shift 2
  [ -f model_outputs_ctkae/da_v5/$lab/run.log ] && grep -q "=== DONE" model_outputs_ctkae/da_v5/$lab/run.log && return
  LRS="0.3 1.0 3.0 10" DA_ROOT=da_v5 KAE_EXTRA="$*" bash ../audit_report/scripts/run_da_v2.sh $lab ctkae $(R $key) last.pth
}
job lpm_ref_s2 ref_s2 --kae-mirror --kae-avg lpm &
job lpm_back001_s2 back001_s2 --kae-mirror --kae-avg lpm &
wait
echo "=== DA v5 DONE $(date)"
