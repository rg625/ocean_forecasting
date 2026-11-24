#!/bin/bash
# remaining da_v4 jobs, split over two chains (run_da_v4.sh's third chain was overloaded)
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
S=model_outputs_ctkae/ablation_state
R () { cat $S/$1.done; }
job () {
  local lab=$1 key=$2; shift 2
  [ -f model_outputs_ctkae/da_v4/$lab/run.log ] && grep -q "=== DONE" model_outputs_ctkae/da_v4/$lab/run.log && return
  LRS="0.3 1.0 3.0 10" DA_ROOT=da_v4 KAE_EXTRA="$*" bash ../audit_report/scripts/run_da_v2.sh $lab ctkae $(R $key) last.pth
}
chainA () { job mirror_back001_s2 back001_s2 --kae-mirror; job ens_mirror_back001 back001_s0 --kae-mirror --kae-ensemble $(R back001_s1) $(R back001_s2); }
chainB () { job ens_mirror_ref ref_s0 --kae-mirror --kae-ensemble $(R ref_s1) $(R ref_s2); }
chainA & chainB & wait
echo "=== DA v4b DONE $(date)"
