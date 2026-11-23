#!/bin/bash
# C2, C3, NL, JS at the paper's delta_f = 1 frame (the campaign in da_results_geometry, which the
# paper cites). run_ks_eval.sh used exp_geometry's current default --pin-delta-f 9, which matches
# da_results_geometry_df9 instead; those results are kept in OUT_DIR/geometry.
#   usage: run_ks_df1.sh RUN_DIR OUT_DIR
set -u
R=$1; O=$2
PY=/home/rg625/anaconda3/envs/samudra/bin/python
D=/home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts/ks_ctkae/ks_driver.py
LOG=$O/run_df1.log
echo "=== geometry_df1 $(date -Is)" >> $LOG
$PY -W ignore $D run exp_geometry --kae-run $R --init-scales $O/init_scales.json --tuning $O/tuning.json \
  --methods KAE-expm --out-dir $O/geometry_df1 --sections C2 C3 NL JS --pin-delta-f 1 \
  --c2-delta-l 5 10 25 50 100 200 400 700 --c3-n 2 3 5 9 17 25 >> $LOG 2>&1
echo "--- exit=$? $(date -Is)" >> $LOG
echo "=== DONE $(date -Is)" >> $LOG
