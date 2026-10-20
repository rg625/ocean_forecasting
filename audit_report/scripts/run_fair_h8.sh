#!/bin/bash
# Fair-horizon CT-KAE (configs/ctkae/fair_h8.yaml): latent 128 and 256, seed 0. Each run is
# followed by its test forecast evaluation and the DA comparison (KAE slot, df9 + df1),
# strictly sequentially so nothing shares the GPU with training.
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
PY=/home/rg625/anaconda3/envs/samudra/bin/python
for d in 128 256; do
  name=H8_d$d
  echo "=== $(date) train $name"
  $PY train_ctkae.py --config=configs/ctkae/fair_h8.yaml run_name=$name seed=0 model.latent_dim=$d 2>&1 | grep -v Warning | tail -1
  run=$(ls -d model_outputs_ctkae/${name}_s0_* | tail -1)
  echo "=== $(date) eval $run"
  $PY eval_ctkae.py $run --ckpt last.pth --dt_test --ma_sweep 2>&1 | grep -v Warn | tail -5
  echo "=== $(date) DA $run"
  bash ../audit_report/scripts/run_da_v2.sh ctkae_${name}_s0 ctkae $run last.pth
done
echo "=== $(date) ALL DONE"
