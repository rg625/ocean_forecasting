#!/bin/bash
# The two CT-KAE candidates on KS, 3 seeds each, two runs at a time (train_ctkae_ks.py).
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
PY=/home/rg625/anaconda3/envs/samudra/bin/python
mkdir -p model_outputs_ctkae_ks
lane () { for s in 0 1 2; do $PY -W ignore train_ctkae_ks.py --config=configs/ctkae_ks/$1.yaml seed=$s > model_outputs_ctkae_ks/train_$1_s$s.log 2>&1; done; }
lane ref & lane back001 & wait
echo "=== KS TRAIN DONE $(date)"
