#!/bin/bash
# Phase 3 (2026-10-20): in the 2->60 setting where the CT-KAE is competitive,
#  (1) one-step-then-long-horizon curriculum (keep the sharp one-step reconstruction),
#  (2) mu ablation: fixed_point=none, compared with the existing F_block seeds (with mu).
# Same base as the F_block runs: block generator, w_amp 0.1, strides {1,2,3}, held-out-Ma
# validation, selection on val_mse60, 12k steps.
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$OCEAN_REPO/koopman_autoencoder"
BASE="--config=configs/ctkae/base.yaml train.steps=12000 train.val_every=1000 train.select=val_mse60 train.strides=[1,2,3] model.generator=block train.w_amp=0.1"
run() { name=$1; seed=$2; shift 2; echo "=== $(date) $name seed=$seed $*"; $PY train_ctkae.py $BASE run_name=$name seed=$seed "$@" 2>&1 | grep -v Warning | tail -1; }
CUR="train.k_start=1 train.k_hold_frac=0.3 train.curriculum_frac=0.3"
run C_curr 0 $CUR
run N_nomu 0 model.fixed_point=none
run N_nomu 1 model.fixed_point=none
run C_curr 1 $CUR
echo "=== $(date) ALL DONE"
