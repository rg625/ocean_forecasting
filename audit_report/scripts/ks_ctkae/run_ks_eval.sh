#!/bin/bash
# The paper's KS experiments for one CT-KAE run (ks_driver.py; main-repo code, paper defaults):
#   init scale (train) -> KAE learning-rate / init grid (val) -> long horizon (FC, HL, LB, PF on
#   da_test_long) -> geometry (S1, C1, C2, C3, NL, JS on da_test), KAE rows only (KS_ONLY_KAE /
#   --methods KAE-expm); U-Net and SDA rows are the paper's, on the same seeded problems.
#   usage: run_ks_eval.sh RUN_DIR OUT_DIR [full|fc]
set -u
R=$1; O=$2; MODE=${3:-full}
PY=/home/rg625/anaconda3/envs/samudra/bin/python
D=/home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts/ks_ctkae/ks_driver.py
mkdir -p $O; LOG=$O/run.log
step () { echo "=== $1 $(date -Is)" >> $LOG; shift; "$@" >> $LOG 2>&1; echo "--- exit=$? $(date -Is)" >> $LOG; }
step init_scales $PY -W ignore $D init_scales $R $O/init_scales.json
C="--kae-run $R --init-scales $O/init_scales.json"
if [ "$MODE" = fc ]; then
  # forecast only: the tuning file is read by Bench but not used by FC
  step fc env KS_ONLY_KAE=1 $PY -W ignore $D run exp_long_horizon $C --tuning da_results_v2/tuning.json --test data/ks/da_test_long.nc --out-dir $O/long --sections FC
  echo "=== DONE $(date -Is)" >> $LOG; exit 0
fi
step tune $PY -W ignore $D tune $R $O/init_scales.json $O/tuning.json
C="$C --tuning $O/tuning.json"
step long env KS_ONLY_KAE=1 $PY -W ignore $D run exp_long_horizon $C --test data/ks/da_test_long.nc --out-dir $O/long --sections FC HL LB PF
step geometry $PY -W ignore $D run exp_geometry $C --methods KAE-expm --out-dir $O/geometry --sections S1 C1 C2 C3 NL JS
echo "=== DONE $(date -Is)" >> $LOG
