#!/bin/bash
# LITERATURE_REVIEW_V3 items 1-2 (2026-10-22), seed 0 each, after the detail-floor run:
# harmonic-lattice generator (fair_h8_harm.yaml), then multi-start consistency
# (fair_h8_tc.yaml); same pipeline as run_detail.sh. Pre-registered success: 60-step
# interp MSE <= 3.19 with extrap <= 2.29.
S=/home/rg625/mnt/ocean_forecasting_impl/audit_report/scripts
P=/tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/563da406-8943-47fe-a8f7-28eddbd027a5/scratchpad
until grep -q "ALL DONE" $P/detailfloor.log; do sleep 60; done
VARIANTS="harm tc" bash $S/run_detail.sh
