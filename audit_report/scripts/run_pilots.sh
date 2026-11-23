#!/bin/bash
# Pilot queue (6k steps each, one at a time on the shared GPU). Each pilot changes ONE
# thing relative to P0. Pilots keep only best.pth (selected on held-out Ma validation).
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
PY=/home/rg625/anaconda3/envs/samudra/bin/python
COMMON="--config=configs/ctkae/base.yaml train.steps=6000 train.val_every=1000"
run() { name=$1; shift; echo "=== $(date) $name $*"; $PY train_ctkae.py $COMMON run_name=$name "$@" 2>&1 | grep -v Warning | tail -3; rm -f model_outputs_ctkae/${name}_s0_*/last.pth; }
run P0_block
run P1_dense model.generator=dense
run P2_hurwitz model.generator=hurwitz
run P3_block_nocond model.film_encoder=false model.film_decoder=false model.cond_generator=false model.cond_fixed_point=false
run P4_block_amp train.w_amp=0.1
run P5_block_k10 train.k_start=10 train.k_max=10
run P6_block_strides train.strides=[1,2,3]
echo "=== $(date) ALL DONE"
