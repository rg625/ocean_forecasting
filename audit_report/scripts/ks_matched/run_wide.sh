#!/bin/bash
# CT-KAE ref 128 with a larger 1-D encoder/decoder, the same building blocks (Tra-consistent):
#   w2: widths x2 = 11.3M parameters (paper KAE: 9.85M)
#   w3: widths x3 = 24.7M parameters (the transonic CT-KAE: 24.70M)
# One at a time (w3 trains at ~21 GB with activation checkpointing); each evaluated before
# the next starts, so evaluation never shares the GPU with a training run.
V=/home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts/ks_matched/run_variant.sh
$V ct_ref_w2_128_s0 --latent 128 --ct 'widths=[128,256,512]' --ct grad_checkpoint=true
$V ct_ref_w3_128_s0 --latent 128 --ct 'widths=[192,384,768]' --ct grad_checkpoint=true
echo "=== WIDE DONE $(date -Is)" >> /home/rg625/mnt/ocean_forecasting_impl/audit_report/results/ks_matched/logs/lanes.log
