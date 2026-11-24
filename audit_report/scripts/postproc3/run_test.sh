#!/bin/bash
cd /tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/cfcfc896-3b58-4c33-b471-e1efcfb360e4/scratchpad/pp3px
for t in ref back001; do
  /home/rg625/anaconda3/envs/samudra/bin/python -u -W ignore /home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts/postproc3/pp3_test.py $t /tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/cfcfc896-3b58-4c33-b471-e1efcfb360e4/scratchpad/pp3px --cands "sharp(mirror,a0.5,c5)" "sharp(mirror,a1,c5)" "sharp(mix+mirror,a0.5,c5)" "sharp(mix+mirror,a1,c5)" "ampmean(mirror)" "qmap(mix+mirror)" > /tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/cfcfc896-3b58-4c33-b471-e1efcfb360e4/scratchpad/pp3px/test_$t.log 2>&1
done
echo TESTDONE >> /tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/cfcfc896-3b58-4c33-b471-e1efcfb360e4/scratchpad/pp3px/test_back001.log
