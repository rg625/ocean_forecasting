#!/bin/bash
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
for t in ref back001; do
  R=$(ls -d model_outputs_ctkae/PX_${t}_s0_*)
  /home/rg625/anaconda3/envs/samudra/bin/python -u -W ignore ../audit_report/scripts/postproc3/pp3_proxy.py $R /tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/cfcfc896-3b58-4c33-b471-e1efcfb360e4/scratchpad/pp3px/PX3_$t.json > /tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/cfcfc896-3b58-4c33-b471-e1efcfb360e4/scratchpad/pp3px/PX3_$t.log 2>&1
done
echo PROXYDONE >> /tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/cfcfc896-3b58-4c33-b471-e1efcfb360e4/scratchpad/pp3px/PX3_back001.log
