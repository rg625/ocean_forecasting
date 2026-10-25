#!/bin/bash
# Data-assimilation comparison, legacy KAE vs CT-KAE, on IDENTICAL DA code (this branch).
# Mirrors run_tra_campaign.sh (canonical schedule) and run_sweeps_df9.sh (delta_f pinned
# at 9) for the KAE slot only (--methods KAE). Each model's 4D-Var learning rate is tuned
# the same way on the validation regime (gt_extrap) before the test regime (gt_interp).
#   usage: run_da_compare.sh LABEL KIND RUN_DIR [CKPT]
#     KIND = legacy | ctkae
set -u
LABEL=$1; KIND=$2; RUN=$3; CKPT=${4:-best.pth}
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$OCEAN_REPO/koopman_autoencoder"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE_OUT=model_outputs_ctkae/da/$LABEL
CAN=$BASE_OUT/canonical; DF9=$BASE_OUT/df9
mkdir -p $CAN $DF9
LOG=$BASE_OUT/run.log
M="--methods KAE --kae-kind $KIND --kae-run $RUN --kae-ckpt $CKPT --regime test"
run () { local label=$1; shift; echo "=== $label $(date -Is)" >> $LOG; $PY -m data_assimilation.tra.exp_tra $M "$@" >> $LOG 2>&1; echo "--- $label exit=$? $(date -Is)" >> $LOG; }
run tune          --out-dir $CAN --sections tune --tune-iters 400 --n-tune 6
cp $CAN/tuning.json $DF9/tuning.json
run headline      --out-dir $CAN --sections headline --iters 2000 --n-problems 12
for s in delta_f sparsity noise delta_l n_obs single_obs misspec_noise; do
  run can_$s --out-dir $CAN --sections $s --iters 500 --n-problems 8
done
run can_joint_sparsity --out-dir $CAN --sections joint_sparsity --iters 500 --n-problems 8 --joint-n-obs 1 4 8 16 32
for s in delta_l n_obs noise sparsity misspec_noise; do
  run df9_$s --out-dir $DF9 --pin-delta-f 9 --sections $s --iters 500 --n-problems 8
done
run df9_joint_sparsity --out-dir $DF9 --pin-delta-f 9 --sections joint_sparsity --iters 500 --n-problems 8 --joint-n-obs 1 4 8 16 32
echo "=== DONE $(date -Is)" >> $LOG
