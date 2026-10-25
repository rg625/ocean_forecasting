#!/bin/bash
# Link a worktree to the data kept once per machine in the main checkout ($OCEAN_MAIN):
#   koopman_autoencoder/data/{acdm,ks,qg}  -> $OCEAN_MAIN/koopman_autoencoder/data/...
#   koopman_autoencoder/data/raw_tra       -> $OCEAN_MAIN/data/128_tra  (raw ACDM sims)
#   autoreg_pde_diffusion/.../LSiM.pth     -> the main checkout's copy
# Run once per new worktree:  bash setup_worktree.sh    (in the main checkout it only reports)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
echo "machine=$OCEAN_MACHINE repo=$OCEAN_REPO main=$OCEAN_MAIN PY=$PY"

link () {  # link TARGET_IN_MAIN PATH_IN_REPO
  local src="$OCEAN_MAIN/$1" dst="$OCEAN_REPO/$2"
  if [ ! -e "$src" ]; then echo "missing  $src"; return; fi
  if [ "$OCEAN_REPO" = "$OCEAN_MAIN" ] && [ "$1" = "$2" ]; then echo "ok       $2"; return; fi
  if [ -e "$dst" ] && [ ! -L "$dst" ]; then echo "exists   $dst (not a link, left alone)"; return; fi
  mkdir -p "$(dirname "$dst")" && ln -sfn "$src" "$dst" && echo "linked   $2"
}
for d in acdm ks qg; do link koopman_autoencoder/data/$d koopman_autoencoder/data/$d; done
link data/128_tra koopman_autoencoder/data/raw_tra
link autoreg_pde_diffusion/src/turbpred/lsim/models/LSiM.pth \
     autoreg_pde_diffusion/src/turbpred/lsim/models/LSiM.pth
