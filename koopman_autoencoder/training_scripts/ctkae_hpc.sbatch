#!/bin/bash
# CT-KAE jobs on CSD3. Submit from anywhere inside the worktree you want to run:
#   sbatch koopman_autoencoder/training_scripts/ctkae_hpc.sbatch audit_report/scripts/run_full.sh
#   sbatch koopman_autoencoder/training_scripts/ctkae_hpc.sbatch train_ctkae.py --config=configs/ctkae/fair_h8.yaml seed=1
# A *.sh argument runs that script (paths relative to the repo root); anything else is passed
# to $PY from koopman_autoencoder/. Override resources on the command line, e.g. --time=4:00:00.
#SBATCH --job-name=ctkae
#SBATCH --output=ctkae_%j.out
#SBATCH --error=ctkae_%j.err
#SBATCH --account=GIROLAMI-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:45:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rg625@cam.ac.uk

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module unload miniconda/3
module load cuda/11.8
nvidia-smi

source "$HOME/.bashrc"
conda activate koopman-ocean

# sbatch runs a copy of this file, so locate the worktree from the submit directory
source "$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)/env.sh"
echo "machine=$OCEAN_MACHINE repo=$OCEAN_REPO main=$OCEAN_MAIN git=$(git -C "$OCEAN_REPO" rev-parse --short HEAD)"

if [[ "$1" == *.sh ]]; then
  cd "$OCEAN_REPO" && bash "$@"
else
  cd "$OCEAN_REPO/koopman_autoencoder" && "$PY" "$@"
fi
