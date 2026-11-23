#!/bin/bash
# Representation pilots (6k steps, seed 0) on the F_block configuration: the test per-lead
# curves show a reconstruction floor (lead-1 MSE 0.66e-3 vs ACDM 0.03e-3) while long-lead
# error matches ACDM. Selection/comparison on held-out-Ma validation only.
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
PY=/home/rg625/anaconda3/envs/samudra/bin/python
BASE="--config=configs/ctkae/base.yaml train.steps=6000 train.val_every=1000 train.select=val_mse60 train.strides=[1,2,3] model.generator=block train.w_amp=0.1"
run() { name=$1; shift; echo "=== $(date) $name $*"; $PY train_ctkae.py $BASE run_name=$name "$@" 2>&1 | grep -v Warning | tail -2; rm -f model_outputs_ctkae/${name}_s0_*/last.pth; }
run R0_d256_ref
run R1_d512 model.latent_dim=512
run R2_d1024 model.latent_dim=1024
run R3_wrec5 train.w_rec=5.0
echo "=== $(date) ALL DONE"
