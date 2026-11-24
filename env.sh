# Per-machine settings, sourced by the run scripts:  source "$(git rev-parse --show-toplevel)/env.sh"
#   OCEAN_REPO     this checkout / worktree (where the code being run lives)
#   OCEAN_MAIN     the main checkout of the repo on this machine; data, the LSiM weights and
#                  legacy runs live there and every worktree links to them (setup_worktree.sh)
#   OCEAN_MACHINE  cambox | hpc | other
#   PY             python interpreter of the project environment
# Anything here can be overridden in env.local.sh next to this file (untracked, one per machine).

OCEAN_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
: "${OCEAN_MAIN:=$(cd "$OCEAN_REPO" && cd "$(git rev-parse --git-common-dir)/.." && pwd -P)}"

if [ -d /rds ] && [ -d "$HOME/rds/hpc-work" ]; then
  OCEAN_MACHINE=hpc
  : "${PY:=python}"  # conda env koopman-ocean, activated by the sbatch script
elif [ -x "$HOME/anaconda3/envs/samudra/bin/python" ]; then
  OCEAN_MACHINE=cambox
  : "${PY:=$HOME/anaconda3/envs/samudra/bin/python}"
else
  OCEAN_MACHINE=other
  : "${PY:=python}"
fi

[ -f "$OCEAN_REPO/env.local.sh" ] && source "$OCEAN_REPO/env.local.sh"
export OCEAN_REPO OCEAN_MAIN OCEAN_MACHINE PY
