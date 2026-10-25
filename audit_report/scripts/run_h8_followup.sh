#!/bin/bash
# Follow-up to run_fair_h8.sh (2026-10-21), strictly sequential on the GPU:
#  1. DA learning-rate grid widened upwards (0.3-10) for both H8 models (seed 0), because the
#     v2 search ended at its top value (1.0) while still improving. d256 gets the full DA
#     suite at the newly chosen rate; d128 is only re-tuned.
#  2. Seeds 1 and 2 of H8 latent 256: train, test forecast eval, DA tune + df9 sweeps.
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$OCEAN_REPO/koopman_autoencoder"
S=../audit_report/scripts
export LRS="0.3 1.0 3.0 10.0" DA_ROOT=da_v3
d256=$(ls -d model_outputs_ctkae/H8_d256_s0_* | tail -1)
d128=$(ls -d model_outputs_ctkae/H8_d128_s0_* | tail -1)
echo "=== $(date) DA v3 full: $d256"
bash $S/run_da_v2.sh ctkae_H8_d256_s0 ctkae $d256 last.pth
echo "=== $(date) DA v3 retune: $d128"
OUT=model_outputs_ctkae/da_v3/ctkae_H8_d128_s0; mkdir -p $OUT/df9
$PY -m data_assimilation.tra.exp_tra --methods KAE --kae-kind ctkae --kae-run $d128 --kae-ckpt last.pth --regime test \
  --out-dir $OUT/df9 --sections tune --tune-iters 400 --n-tune 6 --lrs $LRS > $OUT/run.log 2>&1
for seed in 1 2; do
  echo "=== $(date) train H8_d256 seed $seed"
  $PY train_ctkae.py --config=configs/ctkae/fair_h8.yaml run_name=H8_d256 seed=$seed model.latent_dim=256 2>&1 | grep -v Warning | tail -1
  run=$(ls -d model_outputs_ctkae/H8_d256_s${seed}_* | tail -1)
  echo "=== $(date) eval $run"
  $PY eval_ctkae.py $run --ckpt last.pth 2>&1 | grep -v Warn | tail -4
  echo "=== $(date) DA df9 $run"
  DF9_ONLY=1 bash $S/run_da_v2.sh ctkae_H8_d256_s$seed ctkae $run last.pth
done
echo "=== $(date) ALL DONE"
