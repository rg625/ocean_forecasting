#!/bin/bash
# Two-timescale latent (2026-10-21): advection variant, then the decay-only control, seed 0,
# strictly sequential. Each: train -> test eval (+ dt / Ma checks) -> one-step error split
# -> adapter check (structured generator through solve_kae) -> DA tune + df9 sweeps.
# Target fixed before the runs: interp lead-1 <= 0.12, lead-10 <= 0.466 (H8_d256_s0).
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$OCEAN_REPO/koopman_autoencoder"
S=../audit_report/scripts
for v in ${VARIANTS:-detail decay}; do
  name=H8${v}_d256
  echo "=== $(date) train $name"
  $PY train_ctkae.py --config=configs/ctkae/fair_h8_$v.yaml run_name=$name seed=0 2>&1 | grep -v Warning | tail -1
  run=$(ls -d model_outputs_ctkae/${name}_s0_* | tail -1)
  echo "=== $(date) eval $run"
  $PY eval_ctkae.py $run --ckpt last.pth --dt_test --ma_sweep 2>&1 | grep -v Warn | tail -6
  $PY -W ignore $S/lead1_split.py $run --ckpt last.pth --json ../audit_report/results/lead1_split/${name}_s0.json 2>&1 | tail -2
  $PY -W ignore -m data_assimilation.tra.verify_ctkae_adapter $run last.pth 2>&1 | grep -E "diff|solve_kae|Error"
  echo "=== $(date) DA $run"
  LRS="0.3 1.0 3.0 10.0" DA_ROOT=da_v3 DF9_ONLY=1 bash $S/run_da_v2.sh ctkae_${name}_s0 ctkae $run last.pth
done
echo "=== $(date) ALL DONE"
