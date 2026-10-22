#!/bin/bash
# Loss-hyperparameter search, stage 1 (2026-10-22): one change at a time from the fair 2 -> 8,
# latent 256 reference, 10k steps, seed 0, with Ma 0.60 / 0.75 held out of training
# (data.val_ma); each run scored on validation only (val_score.py: 60-frame forecast MSE and
# df = 9 4D-Var on val_heldout.nc). Selection rule (fixed before the runs, summary script):
# S = fc_mse60 / ref + da_df9 / ref, lower is better; a change counts as helpful if it
# lowers S by >= 0.05 versus the reference run.
# Each training waits for >= 13 GB of free GPU memory so it cannot crash other jobs.
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
PY=/home/rg625/anaconda3/envs/samudra/bin/python
S=../audit_report/scripts
wait_gpu () { until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge 13000 ]; do sleep 60; done; }
BASE="--config=configs/ctkae/fair_h8.yaml model.latent_dim=256 data.val_ma=[0.6,0.75] train.steps=10000 train.val_every=2500 seed=0"
run () {
  tag=$1; shift
  if ls -d model_outputs_ctkae/LS_${tag}_s0_*/val_score.json >/dev/null 2>&1; then echo "skip $tag"; return; fi
  wait_gpu
  echo "=== $(date) train LS_$tag $*"
  $PY train_ctkae.py $BASE run_name=LS_$tag "$@" 2>&1 | grep -v Warning | tail -1
  r=$(ls -d model_outputs_ctkae/LS_${tag}_s0_* | tail -1)
  wait_gpu
  $PY -W ignore $S/val_score.py $r --json $r/val_score.json 2>&1 | grep -v INFO | tail -1
}
run R
run rec3 train.w_rec=3.0
run rec03 train.w_rec=0.3
run ramp2 train.w_lead_ramp=2.0
run lat1 train.w_lat=1.0
run lat0 train.w_lat=0.0
run amp0 train.w_amp=0.0
run amp05 train.w_amp=0.5
run uniform train.channel_weighting=uniform
run lr6e4 train.lr=6e-4
run lr15e5 train.lr=1.5e-4
echo "=== $(date) STAGE 1 DONE"
