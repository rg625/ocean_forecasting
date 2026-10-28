#!/bin/bash
# post-processing lab over ref and back001 seeds (serialised GPU start, 4.5 GB free required;
# the lab caps itself at 3 GB)
P=/tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/093163e5-e75d-47aa-b04b-9bb0dda37f82/scratchpad/res/pp
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
PY=/home/rg625/anaconda3/envs/samudra/bin/python
ST=model_outputs_ctkae/ablation_state
wait_gpu () { until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge 4500 ]; do sleep 60; done; }
gpu_start () { exec 9>$ST/gpu.flock; flock 9; wait_gpu; "$@" & local pid=$!; sleep 90; flock -u 9; exec 9>&-; wait $pid; }
for k in ref_s0 back001_s0 ref_s1 back001_s1 ref_s2 back001_s2; do
  [ -f $P/$k.npz ] && continue
  echo "=== $(date) $k"
  gpu_start $PY -W ignore $P/pp_lab.py $(cat $ST/$k.done) $P/$k.npz --mem_gb 3.0 > $P/$k.log 2>&1
  tail -2 $P/$k.log
done
echo "=== $(date) ALL DONE"
