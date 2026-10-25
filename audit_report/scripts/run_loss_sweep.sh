#!/bin/bash
# Loss-hyperparameter search, stage 1 (2026-10-22): one change at a time from the fair 2 -> 8,
# latent 256 reference, 10k steps, seed 0, trained on all data like the final model. At the
# user's request each run is scored on the TEST cases (test_score.py: interp / extrap /
# longer forecast MSE, and df = 9 4D-Var on gt_interp with lr tuned on gt_extrap), so the
# winner's screening numbers are optimistically biased; the final claim uses fresh seeds.
# Selection rule (fixed before the runs, loss_sweep_summary.py):
# S = mean(interp, extrap, longer MSE relative to the reference run) + da_df9 / ref, lower
# is better; a change counts as helpful if it lowers S by >= 0.05.
# Batch 16 is split into 2 chunks of 8 with exact gradient accumulation (train.accum,
# test_gradient_accumulation_is_exact): same training, peak 6.9 GB instead of 12.3 GB.
# Each job waits for >= 8 GB of free GPU memory so it cannot crash other jobs.
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$OCEAN_REPO/koopman_autoencoder"
S=../audit_report/scripts
wait_gpu () { until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge 8000 ]; do sleep 60; done; }
BASE="--config=configs/ctkae/fair_h8.yaml model.latent_dim=256 train.steps=10000 train.accum=2 seed=0"
run () {
  tag=$1; shift
  if ls -d model_outputs_ctkae/LS_${tag}_s0_*/test_score.json >/dev/null 2>&1; then echo "skip $tag"; return; fi
  wait_gpu
  echo "=== $(date) train LS_$tag $*"
  $PY train_ctkae.py $BASE run_name=LS_$tag "$@" 2>&1 | grep -v Warning | tail -1
  r=$(ls -d model_outputs_ctkae/LS_${tag}_s0_* | tail -1)
  wait_gpu
  $PY eval_ctkae.py $r --ckpt last.pth 2>&1 | grep -v Warn | tail -3
  $PY -W ignore $S/test_score.py $r --json $r/test_score.json 2>&1 | grep -v INFO | tail -1
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
