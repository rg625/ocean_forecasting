#!/bin/bash
# DA with the post-processed CT-KAE readouts (2026-10-28), all in da_v4: the full suite (df1 + df9)
# for both candidates, per seed, plain and with the mirror readout, and the 3-seed + mirror
# ensemble. Three chains run concurrently; same launcher, grid and checkpoint as da_v3.
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
S=model_outputs_ctkae/ablation_state
R () { cat $S/$1.done; }
job () {  # LABEL KEY [EXTRA...]
  local lab=$1 key=$2; shift 2
  [ -f model_outputs_ctkae/da_v4/$lab/run.log ] && grep -q "=== DONE" model_outputs_ctkae/da_v4/$lab/run.log && return
  LRS="0.3 1.0 3.0 10" DA_ROOT=da_v4 KAE_EXTRA="$*" bash ../audit_report/scripts/run_da_v2.sh $lab ctkae $(R $key) last.pth
}
chain1 () { job plain_ref_s1 ref_s1; job plain_ref_s2 ref_s2; job mirror_ref_s0 ref_s0 --kae-mirror; job mirror_ref_s1 ref_s1 --kae-mirror; }
chain2 () { job plain_back001_s1 back001_s1; job plain_back001_s2 back001_s2; job mirror_back001_s0 back001_s0 --kae-mirror; job mirror_back001_s1 back001_s1 --kae-mirror; }
chain3 () { job mirror_ref_s2 ref_s2 --kae-mirror; job mirror_back001_s2 back001_s2 --kae-mirror;
  job ens_mirror_ref ref_s0 --kae-mirror --kae-ensemble $(R ref_s1) $(R ref_s2);
  job ens_mirror_back001 back001_s0 --kae-mirror --kae-ensemble $(R back001_s1) $(R back001_s2); }
chain1 & chain2 & chain3 & wait
echo "=== DA v4 DONE $(date)"
