#!/bin/bash
# One matched CT-KAE variant, trained with the paper's KS pipeline and evaluated as the other
# matched runs: run_variant.sh NAME TRAIN_ARGS...  (train_matched.py ct NAME TRAIN_ARGS), e.g.
#   run_variant.sh ct_ref_w2_128_s0 --latent 128 --ct 'widths=[128,256,512]' --ct grad_checkpoint=true
# Evaluation: forecast (fc_matched), init scale, DA tuning (warm start forced: best init='obs'
# point), delta_f = 9 DA (C2 C3 NL JS), FC/LB/PF long horizon.
PY=/home/rg625/anaconda3/envs/samudra/bin/python
T=/home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts/ks_matched/train_matched.py
D=/home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts/ks_ctkae/ks_driver.py
F=/home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts/ks_matched/eval_fc_matched.py
M=/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder/model_outputs_ks_matched
R=/home/rg625/mnt/ocean_forecasting_impl/audit_report/results/ks_matched
L=$R/logs/lanes.log
n=$1; shift
echo "=== $n $(date -Is)" >> $L
$PY -W ignore $T ct $M/$n "$@" > $R/logs/$n.log 2>&1
rc=$?; echo "--- $n exit=$rc $(date -Is)" >> $L; [ $rc = 0 ] || exit 1
RD=$(ls -d $M/$n/*/ | head -n 1); RD=${RD%/}; O=$R/$n; mkdir -p $O; G=$O/run.log
step () { echo "=== $1 $(date -Is)" >> $G; shift; "$@" >> $G 2>&1; echo "--- exit=$? $(date -Is)" >> $G; }
echo "=== eval $n $(date -Is)" >> $L
step fc_matched $PY -W ignore $F $O/fc_matched.npz $n=$RD
step init_scales $PY -W ignore $D init_scales $RD $O/init_scales.json
step tune $PY -W ignore $D tune $RD $O/init_scales.json $O/tuning.json
$PY - "$O" <<'PYEOF' >> $G 2>&1
import json, sys
o = sys.argv[1]; d = json.load(open(f"{o}/tuning.json")); k = d["results"]["KAE-expm"]
if k["best"]["init"] != "obs":
    b = min((g for g in k["grid"] if g["init"] == "obs"), key=lambda g: g["init_rel_l2_mean"])
    d["note"] = f"warm start forced: best init='obs' point (free choice was {k['best']})"
    k["best"] = dict(b)
print("tuning (warm):", k["best"])
json.dump(d, open(f"{o}/tuning_warm.json", "w"), indent=1)
PYEOF
C="--kae-run $RD --init-scales $O/init_scales.json --tuning $O/tuning_warm.json"
step geometry_df9 env KS_ONLY_KAE=1 $PY -W ignore $D run exp_geometry $C --methods KAE-expm --out-dir $O/geometry_df9 --sections C2 C3 NL JS
step long env KS_ONLY_KAE=1 $PY -W ignore $D run exp_long_horizon $C --test data/ks/da_test_long.nc --out-dir $O/long --sections FC LB PF
echo "--- eval $n done $(date -Is)" >> $L
