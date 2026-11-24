#!/bin/bash
# Final stage, reordered (2026-10-23 12:05). The combination (w_amp 0.5 + lr 6e-4) seed 10
# at 20k steps is worse than the current best on every test metric (interp 4.13 vs
# 3.38-3.66, DA 0.118 vs 0.089-0.091), so its seeds 11-12 are not run. Instead each change
# alone at 20k steps (seed 10); then, by a rule fixed now, two more seeds (11, 12) plus the
# full DA suite only for the better single change if its score
#   S = mean_c(MSE_c / mean MSE_c of H8_d256 s0-2) + DA / mean DA of H8_d256 s0-2
# is <= 1.95 (the current best scores 2.00 by construction).
OLD=1635204
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$OCEAN_REPO/koopman_autoencoder"
S=../audit_report/scripts
until grep -q "=== DONE" model_outputs_ctkae/da_v3/ctkae_LS2_combo_s10/run.log 2>/dev/null; do sleep 10; done
kill $OLD 2>/dev/null; sleep 3
pkill -f "run_name=LS2_combo seed=11" 2>/dev/null; sleep 3
for d in $(ls -d model_outputs_ctkae/LS2_combo_s11_* 2>/dev/null); do echo "stopped at start: combination worse at seed 10 (run_loss_stage2b.sh)" > $d/ABORTED; done
wait_gpu () { until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge 8000 ]; do sleep 60; done; }
BASE="--config=configs/ctkae/fair_h8.yaml model.latent_dim=256 train.steps=20000 train.accum=2"
run () {
  name=$1; seed=$2; da=$3; shift 3
  if ls -d model_outputs_ctkae/${name}_s${seed}_*/test_score.json >/dev/null 2>&1; then echo "skip $name $seed"; return; fi
  wait_gpu
  echo "=== $(date) train $name seed $seed $*"
  $PY train_ctkae.py $BASE run_name=$name seed=$seed "$@" 2>&1 | grep -v Warning | tail -1
  r=$(ls -d model_outputs_ctkae/${name}_s${seed}_* | tail -1)
  wait_gpu
  $PY eval_ctkae.py $r --ckpt last.pth --dt_test --ma_sweep 2>&1 | grep -v Warn | tail -6
  $PY -W ignore $S/test_score.py $r --json $r/test_score.json 2>&1 | grep -v INFO | tail -1
  if [ "$da" = 1 ]; then LRS="0.3 1.0 3.0 10.0" DA_ROOT=da_v3 bash $S/run_da_v2.sh ctkae_${name}_s$seed ctkae $r last.pth; fi
}
run LS2_amp05 10 0 train.w_amp=0.5
run LS2_lr6e4 10 0 train.lr=6e-4
PICK=$($PY - <<'PYEOF'
import json, glob
ref = [json.load(open(f)) for f in glob.glob("model_outputs_ctkae/H8_d256_s[012]_*/test_score.json")]
m = {k: sum(r[k] for r in ref) / len(ref) for k in ("interp_mse", "extrap_mse", "longer_mse", "da_df9")}
best = None
for name in ("LS2_amp05", "LS2_lr6e4"):
    j = json.load(open(glob.glob(f"model_outputs_ctkae/{name}_s10_*/test_score.json")[0]))
    s = sum(j[f"{c}_mse"] / m[f"{c}_mse"] for c in ("interp", "extrap", "longer")) / 3 + j["da_df9"] / m["da_df9"]
    print(f"# {name} S={s:.3f}", file=__import__("sys").stderr)
    if s <= 1.95 and (best is None or s < best[1]):
        best = (name, s)
print(best[0] if best else "none")
PYEOF
)
echo "=== decision: $PICK"
case $PICK in
  LS2_amp05) A="train.w_amp=0.5";;
  LS2_lr6e4) A="train.lr=6e-4";;
  *) echo "=== $(date) no single change beats the current best; STAGE 2 DONE"; exit 0;;
esac
r10=$(ls -d model_outputs_ctkae/${PICK}_s10_* | tail -1)
LRS="0.3 1.0 3.0 10.0" DA_ROOT=da_v3 bash $S/run_da_v2.sh ctkae_${PICK}_s10 ctkae $r10 last.pth
for seed in 11 12; do run $PICK $seed 1 $A; done
echo "=== $(date) STAGE 2 DONE"
