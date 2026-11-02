#!/bin/bash
# Seed 0 of both candidates: the full KS suite; seeds 1, 2: free-running forecast (FC) only.
K=/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder/model_outputs_ctkae_ks
O=/home/rg625/mnt/ocean_forecasting_impl/audit_report/results/ks_ctkae
E=/home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts/ks_ctkae/run_ks_eval.sh
run () { bash $E $(ls -d $K/ks_$1_s$2_*) $O/$1_s$2 $3; }
lane_ref () { run ref 0 full; }
lane_back () { run back001 0 full; }
lane_fc () {
  until [ -f $(ls -d $K/ks_back001_s2_*)/DONE ]; do sleep 30; done
  for m in ref back001; do for s in 1 2; do run $m $s fc; done; done
}
lane_ref & lane_back & lane_fc & wait
echo "=== KS EVAL DONE $(date)"
