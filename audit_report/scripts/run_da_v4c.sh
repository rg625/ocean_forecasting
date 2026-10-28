#!/bin/bash
# Finish da_v4 in two lanes (five concurrent suites ran out of GPU memory in the n_obs and
# joint_sparsity sweeps): re-run the failed sections and the sections of the two suites that were
# stopped. Each lane waits for one still-running suite to finish first.
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
PY=/home/rg625/anaconda3/envs/samudra/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=model_outputs_ctkae/ablation_state
R () { cat $S/$1.done; }
sec () {  # LABEL KEY SECTION [EXTRA...]
  local lab=$1 key=$2 s=$3; shift 3
  local OUT=model_outputs_ctkae/da_v4/$lab LOG=model_outputs_ctkae/da_v4/$lab/run.log pin=${s:2:1} name=${s:4}
  local args="--out-dir $OUT/df$pin --pin-delta-f $pin --sections $name --iters 500 --n-problems 8"
  [ $name = joint_sparsity ] && args="$args --joint-n-obs 1 4 8 16 32"
  echo "=== $s (v4c) $(date -Is)" >> $LOG
  $PY -m data_assimilation.tra.exp_tra --methods KAE --kae-kind ctkae --kae-run $(R $key) --kae-ckpt last.pth --regime test "$@" $args >> $LOG 2>&1
  echo "--- $s exit=$? $(date -Is)" >> $LOG
}
done_ () { echo "=== DONE $(date -Is)" >> model_outputs_ctkae/da_v4/$1/run.log; }
waitpid () { while kill -0 $1 2>/dev/null; do sleep 30; done; }
lane1 () {
  waitpid 360753
  for s in df1_misspec_noise df9_n_obs df9_joint_sparsity df1_n_obs df1_joint_sparsity; do sec plain_ref_s2 ref_s2 $s; done; done_ plain_ref_s2
  sec plain_ref_s1 ref_s1 df9_joint_sparsity
}
lane2 () {
  waitpid 563469
  for s in df9_n_obs df9_joint_sparsity df1_n_obs df1_joint_sparsity; do sec mirror_back001_s2 back001_s2 $s --kae-mirror; done; done_ mirror_back001_s2
  for s in df9_n_obs df9_joint_sparsity df1_joint_sparsity; do sec plain_back001_s1 back001_s1 $s; done
}
lane1 & lane2 & wait
echo "=== DA v4c DONE $(date)"
