#!/bin/bash
# CT-KAE in the exact ACDM training setting (configs/ctkae/acdm_matched.yaml), with and
# without the latent fixed point mu(Ma), 3 seeds each, interleaved. Final checkpoints used.
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
cd "$OCEAN_REPO/koopman_autoencoder"
run() { name=$1; seed=$2; shift 2; echo "=== $(date) $name seed=$seed $*"; $PY train_ctkae.py --config=configs/ctkae/acdm_matched.yaml run_name=$name seed=$seed "$@" 2>&1 | grep -v Warning | tail -1; }
for s in 0 1 2; do
  run A_mu $s model.fixed_point=conditional
  run A_nomu $s model.fixed_point=none
done
echo "=== $(date) ALL DONE"
