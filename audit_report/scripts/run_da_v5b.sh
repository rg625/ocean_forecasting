#!/bin/bash
# re-run the two da_v5 sections that ran out of GPU memory (shared GPU), one at a time
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
PY=/home/rg625/anaconda3/envs/samudra/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=model_outputs_ctkae/ablation_state
R () { cat $S/$1.done; }
sec () {  # LABEL KEY SECTION [EXTRA...]
  local lab=$1 key=$2 s=$3; shift 3
  local OUT=model_outputs_ctkae/da_v5/$lab LOG=model_outputs_ctkae/da_v5/$lab/run.log pin=${s:2:1} name=${s:4}
  local args="--out-dir $OUT/df$pin --pin-delta-f $pin --sections $name --iters 500 --n-problems 8"
  [ $name = joint_sparsity ] && args="$args --joint-n-obs 1 4 8 16 32"
  echo "=== $s (v5b) $(date -Is)" >> $LOG
  $PY -m data_assimilation.tra.exp_tra --methods KAE --kae-kind ctkae --kae-run $(R $key) --kae-ckpt last.pth --regime test "$@" $args >> $LOG 2>&1
  echo "--- $s exit=$? $(date -Is)" >> $LOG
}
sec lpm_ref_s2 ref_s2 df9_n_obs --kae-mirror --kae-avg lpm
sec lpm_back001_s2 back001_s2 df1_joint_sparsity --kae-mirror --kae-avg lpm
echo "=== DA v5b DONE $(date)"
