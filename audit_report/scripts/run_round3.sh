#!/bin/bash
# Round 3 (2026-10-21), strictly sequential on the GPU:
#  1. curriculum CT-KAE: DA re-tuned on the widened grid (it was capped at lr 1.0 in v2)
#  2. original KAE with its final checkpoint (epoch_199, not selected on gt_interp), full DA
#  3. DA conditioning diagnostic (da_conditioning.py) for every DA model
#  4. multiscale-decoder CT-KAE (configs/ctkae/fair_h8_ms.yaml), seed 0: train, eval, DA,
#     conditioning; seeds 1-2 follow only if seed 0's interp lead-1 error is >= 10% below
#     H8_d256_s0's 0.261 (rule fixed before the run)
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$OCEAN_REPO/koopman_autoencoder"
S=../audit_report/scripts
LEG=$OCEAN_MAIN/koopman_autoencoder/model_outputs_tra/continous_linear_128/run-20260821_031121
CURR=$(ls -d model_outputs_ctkae/C_curr_s0_* | tail -1)
D256=$(ls -d model_outputs_ctkae/H8_d256_s0_* | tail -1)
D128=$(ls -d model_outputs_ctkae/H8_d128_s0_* | tail -1)
export DA_ROOT=da_v3
COND=model_outputs_ctkae/da_v3/conditioning; mkdir -p $COND
lr_of () { $PY -c "import json;print(json.load(open('model_outputs_ctkae/$1/df9/tuning.json'))['KAE']['lr'])"; }
cond () { echo "=== $(date) conditioning $1"; $PY -W ignore -m data_assimilation.tra.da_conditioning --kae-kind $2 --kae-run $3 ${4:+--kae-ckpt $4} --lr $5 --json $COND/$1.json 2>&1 | grep -v INFO | tail -3; }

echo "=== $(date) DA v3 curriculum"
LRS="0.3 1.0 3.0 10.0" bash $S/run_da_v2.sh ctkae_curr_s0 ctkae $CURR best.pth
echo "=== $(date) DA v3 legacy epoch_199"
LRS="0.03 0.1 0.3 1.0 3.0" bash $S/run_da_v2.sh legacy_ep199 legacy $LEG epoch_199.pth

cond legacy_best legacy $LEG "" $(lr_of da_v2/legacy)
cond legacy_ep199 legacy $LEG epoch_199.pth $(lr_of da_v3/legacy_ep199)
cond ctkae_curr_s0 ctkae $CURR best.pth $(lr_of da_v3/ctkae_curr_s0)
cond ctkae_H8_d256_s0 ctkae $D256 last.pth $(lr_of da_v3/ctkae_H8_d256_s0)
cond ctkae_H8_d128_s0 ctkae $D128 last.pth $(lr_of da_v3/ctkae_H8_d128_s0)

ms_seed () {
  echo "=== $(date) train H8ms_d256 seed $1"
  $PY train_ctkae.py --config=configs/ctkae/fair_h8_ms.yaml run_name=H8ms_d256 seed=$1 2>&1 | grep -v Warning | tail -1
  run=$(ls -d model_outputs_ctkae/H8ms_d256_s$1_* | tail -1)
  echo "=== $(date) eval $run"
  $PY eval_ctkae.py $run --ckpt last.pth $2 2>&1 | grep -v Warn | tail -4
  echo "=== $(date) DA $run"
  LRS="0.3 1.0 3.0 10.0" DF9_ONLY=$3 bash $S/run_da_v2.sh ctkae_H8ms_d256_s$1 ctkae $run last.pth
}
ms_seed 0 "--dt_test --ma_sweep" 0
MS0=$(ls -d model_outputs_ctkae/H8ms_d256_s0_* | tail -1)
cond ctkae_H8ms_d256_s0 ctkae $MS0 last.pth $(lr_of da_v3/ctkae_H8ms_d256_s0)
L1=$($PY -c "import json;print(json.load(open('$MS0/eval/last.json'))['interp']['mse_lead'][0])")
echo "=== multiscale s0 interp lead-1 = $L1 (seeds 1-2 run if < 0.235)"
if $PY -c "import sys;sys.exit(0 if $L1 < 0.235 else 1)"; then
  for seed in 1 2; do ms_seed $seed "" 1; done
fi
echo "=== $(date) ALL DONE"
