#!/bin/bash
# Full CT-KAE ablation (2026-10-23). Worker: claims jobs from ablation_queue.txt in order
# (atomic mkdir lock) and, per job: trains the fair 2 -> 8, latent 256 model with the job's
# overrides (20k steps, all data, exact gradient accumulation, seed = job seed; paired with
# the reference seeds 0-2), unless a finished run of that name and seed exists; evaluates
# on the test cases; scores MSE (predicted / all frames), LSiM and df9 DA (test_score.py);
# runs the four df9 DA sweeps of the paper's Table 11 (window length, number of
# observations, sensor coverage, noise; lr tuned on gt_extrap). Several workers may run; GPU job
# starts are serialised (gpu_start) and each job is tried up to three times.
#   usage: run_ablation.sh WORKER_ID
W=${1:-0}
cd /home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder
PY=/home/rg625/anaconda3/envs/samudra/bin/python
S=../audit_report/scripts
Q=$S/ablation_queue.txt
LOCK=model_outputs_ctkae/ablation_state; mkdir -p $LOCK
BASE="--config=configs/ctkae/fair_h8.yaml model.latent_dim=256 train.steps=20000 train.accum=2"
wait_disk () { until [ $(df --output=avail -B1M / | tail -1) -ge 600 ]; do sleep 120; done; }
# serialise GPU job starts: check free memory and launch under one lock, then give the job
# 90 s to allocate before the next start may check (avoids two jobs grabbing the same memory)
gpu_start () { exec 9>$LOCK/gpu.flock; flock 9; wait_gpu; "$@" & local pid=$!; sleep 90; flock -u 9; exec 9>&-; wait $pid; }
wait_gpu () { until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge 8000 ]; do sleep 60; done; }
finished_run () { for d in $(ls -d model_outputs_ctkae/${1}_s${2}_* 2>/dev/null | sort -r); do [ -f $d/DONE ] && [ -f $d/last.pth ] && { echo $d; return; }; done; }
while true; do
  progress=0
  while IFS='|' read -r tag seed name ov; do
    [ -z "$tag" ] || [[ "$tag" == \#* ]] && continue
    key=${tag}_s${seed}
    [ -f $LOCK/$key.done ] || [ -f $LOCK/$key.failed ] && continue
    mkdir $LOCK/$key.lock 2>/dev/null || continue
    progress=1
    echo "=== [w$W] $(date) $key: $name seed $seed $ov"
    r=$(finished_run $name $seed)
    if [ -z "$r" ]; then
      wait_disk
      gpu_start $PY train_ctkae.py $BASE run_name=$name seed=$seed $ov > $LOCK/$key.trainlog 2>&1
      grep -v Warning $LOCK/$key.trainlog | tail -1
      r=$(finished_run $name $seed)
      if [ -z "$r" ]; then
        echo "!!! [w$W] $key training failed"
        if [ -f $LOCK/$key.fail2 ]; then touch $LOCK/$key.failed
        elif [ -f $LOCK/$key.fail1 ]; then touch $LOCK/$key.fail2; else touch $LOCK/$key.fail1; fi
        rmdir $LOCK/$key.lock; sleep 300; break
      fi
    fi
    [ -f $r/eval/last.json ] || gpu_start $PY eval_ctkae.py $r --ckpt last.pth > /dev/null 2>&1
    $PY -c "import json,sys;sys.exit(0 if 'interp_lsim' in json.load(open('$r/test_score.json')) else 1)" 2>/dev/null \
      || gpu_start $PY -W ignore $S/test_score.py $r --json $r/test_score.json > /dev/null 2>&1
    lab=ctkae_$(basename $r)
    grep -q "=== DONE" model_outputs_ctkae/da_abl/$lab/run.log 2>/dev/null || {
      DF9_ONLY=1 DF9_SECTIONS="delta_l n_obs sparsity noise" LRS="0.3 1.0 3.0 10.0" DA_ROOT=da_abl \
        gpu_start bash $S/run_da_v2.sh $lab ctkae $r last.pth; }
    # weight-sensitivity runs keep their scores but not their checkpoint (disk space)
    if [[ "$tag" == S_* ]] && [ -f $r/test_score.json ] && grep -q "=== DONE" model_outputs_ctkae/da_abl/$lab/run.log; then
      rm -f $r/last.pth; echo "last.pth deleted after scoring (disk space); scores in test_score.json and da_abl" > $r/CKPT_REMOVED
    fi
    echo "$r" > $LOCK/$key.done; rmdir $LOCK/$key.lock
    echo "--- [w$W] $(date) $key done"
    break  # restart from the top: always the first unfinished job in queue order
  done < $Q
  [ $progress = 0 ] && break
done
echo "=== [w$W] $(date) ALL DONE"
