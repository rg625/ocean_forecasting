#!/bin/bash
# Loss sweep, final stage (2026-10-23). Stage 1 (results/loss_sweep_stage1.json) found two
# helpful changes by the pre-registered rule: w_amp 0.5 (S 1.905) and lr 6e-4 (S 1.923)
# vs reference 2.000. Here: their combination at the full 20k steps with three FRESH seeds
# (10, 11, 12; seed 0 was used for screening), each with the test evaluation and the full
# DA suite (df9 + df1, lr grid 0.3-10); then each change alone at 20k steps (seed 10) for
# attribution. Exact gradient accumulation (accum 2) so jobs fit beside other GPU users.
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$OCEAN_REPO/koopman_autoencoder"
S=../audit_report/scripts
wait_gpu () { until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge 8000 ]; do sleep 60; done; }
BASE="--config=configs/ctkae/fair_h8.yaml model.latent_dim=256 train.steps=20000 train.accum=2"
run () {
  name=$1; seed=$2; da=$3; shift 3
  if ls -d model_outputs_ctkae/${name}_s${seed}_*/test_score.json >/dev/null 2>&1; then echo "skip $name $seed"; return; fi
  wait_gpu
  echo "=== $(date) train $name seed $seed $*"
  $PY train_ctkae.py $BASE run_name=$name seed=$seed "$@" 2>&1 | grep -v Warning | tail -1
  r=$(ls -d model_outputs_ctkae/${name}_s${seed}_* | tail -1)
  wait_gpu
  $PY eval_ctkae.py $r --ckpt last.pth --dt_test --ma_sweep 2>&1 | grep -v Warn | tail -6
  $PY -W ignore $S/test_score.py $r --json $r/test_score.json 2>&1 | grep -v INFO | tail -1
  if [ "$da" = 1 ]; then
    LRS="0.3 1.0 3.0 10.0" DA_ROOT=da_v3 bash $S/run_da_v2.sh ctkae_${name}_s$seed ctkae $r last.pth
  fi
}
for seed in 10 11 12; do run LS2_combo $seed 1 train.w_amp=0.5 train.lr=6e-4; done
run LS2_amp05 10 0 train.w_amp=0.5
run LS2_lr6e4 10 0 train.lr=6e-4
echo "=== $(date) STAGE 2 DONE"
