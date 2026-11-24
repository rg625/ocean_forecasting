#!/bin/bash
# Full runs after the pilots (selection on val_mse60, held-out Ma 0.60/0.75), then the
# corrected legacy baseline E1. Interleaved so partial results are informative.
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$OCEAN_REPO/koopman_autoencoder"
BASE="--config=configs/ctkae/base.yaml train.steps=12000 train.val_every=1000 train.select=val_mse60 train.strides=[1,2,3]"
run() { name=$1; seed=$2; shift 2; echo "=== $(date) $name seed=$seed $*"; $PY train_ctkae.py $BASE run_name=$name seed=$seed "$@" 2>&1 | grep -v Warning | tail -2; }
run F_block 0 model.generator=block train.w_amp=0.1
run F_hurwitz 0 model.generator=hurwitz
run F_block 1 model.generator=block train.w_amp=0.1
run F_hurwitz 1 model.generator=hurwitz
run F_block 2 model.generator=block train.w_amp=0.1
echo "=== $(date) E1 legacy"
WANDB_MODE=disabled $PY train.py --config-name experiment/tra/continous_linear_128_heldout 2>&1 | grep -E "Early stopping|Training finished|Test Metrics|Traceback|Error" | tail -5
echo "=== $(date) ALL DONE"
