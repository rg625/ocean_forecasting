#!/bin/bash
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
for k in ref_s0 ref_s1 ref_s2 back001_s0 back001_s1 back001_s2; do
  [ -f /tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/cfcfc896-3b58-4c33-b471-e1efcfb360e4/scratchpad/pp3px/apply0_$k.npz ] && continue
  /home/rg625/anaconda3/envs/samudra/bin/python -u -W ignore ../audit_report/scripts/postproc2/pp2_apply.py $(cat model_outputs_ctkae/ablation_state/$k.done) /tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/cfcfc896-3b58-4c33-b471-e1efcfb360e4/scratchpad/pp3px/apply0_$k.npz enc:0:mirror > /tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/cfcfc896-3b58-4c33-b471-e1efcfb360e4/scratchpad/pp3px/apply0_$k.log 2>&1 || echo FAIL $k
done
echo APPLYDONE
