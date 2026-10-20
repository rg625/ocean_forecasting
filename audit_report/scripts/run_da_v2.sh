#!/bin/bash
# DA comparison v2 (2026-10-20). Fixes in v1: (a) the "canonical" sweeps silently ran at
# delta_f = 9 because --pin-delta-f defaults to 9 in the committed exp_tra.py, so df1 and
# df9 are now passed explicitly; (b) the CT-KAE's learning-rate search ended at the top of
# its grid, so the grid is widened to 0.03-1.0 for every model (still tuned on gt_extrap).
# Memory-hungry sweeps (n_obs, joint_sparsity) run last.
#   usage: run_da_v2.sh LABEL KIND RUN_DIR [CKPT]
set -u
LABEL=$1; KIND=$2; RUN=$3; CKPT=${4:-best.pth}
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
PY=/home/rg625/anaconda3/envs/samudra/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=model_outputs_ctkae/da_v2/$LABEL; mkdir -p $OUT/df9 $OUT/df1
LOG=$OUT/run.log
M="--methods KAE --kae-kind $KIND --kae-run $RUN --kae-ckpt $CKPT --regime test"
run () { local label=$1; shift; echo "=== $label $(date -Is)" >> $LOG; $PY -m data_assimilation.tra.exp_tra $M "$@" >> $LOG 2>&1; echo "--- $label exit=$? $(date -Is)" >> $LOG; }
run tune --out-dir $OUT/df9 --sections tune --tune-iters 400 --n-tune 6 --lrs 0.03 0.1 0.3 1.0
cp $OUT/df9/tuning.json $OUT/df1/tuning.json
LR=$($PY -c "import json;print(json.load(open('$OUT/df9/tuning.json'))['KAE']['lr'])")
echo "=== repr_check lr=$LR $(date -Is)" >> $LOG
$PY -m data_assimilation.tra.da_repr_check --kae-kind $KIND --kae-run $RUN --kae-ckpt $CKPT --lr $LR --json $OUT/repr_check.json >> $LOG 2>&1
run headline --out-dir $OUT/df1 --sections headline --iters 2000 --n-problems 12
for s in delta_f single_obs; do run $s --out-dir $OUT/df1 --sections $s --iters 500 --n-problems 8; done
for s in delta_l noise sparsity misspec_noise; do run df9_$s --out-dir $OUT/df9 --pin-delta-f 9 --sections $s --iters 500 --n-problems 8; done
for s in delta_l noise sparsity misspec_noise; do run df1_$s --out-dir $OUT/df1 --pin-delta-f 1 --sections $s --iters 500 --n-problems 8; done
for pin in 9 1; do
  run df${pin}_n_obs --out-dir $OUT/df$pin --pin-delta-f $pin --sections n_obs --iters 500 --n-problems 8
  run df${pin}_joint_sparsity --out-dir $OUT/df$pin --pin-delta-f $pin --sections joint_sparsity --iters 500 --n-problems 8 --joint-n-obs 1 4 8 16 32
done
echo "=== DONE $(date -Is)" >> $LOG
