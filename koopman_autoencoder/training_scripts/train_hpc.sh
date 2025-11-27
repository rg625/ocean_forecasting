#!/bin/bash
#SBATCH --job-name=koopman_training
#SBATCH --output=koopman_%A_%a.out
#SBATCH --error=koopman_%A_%a.err
#SBATCH --account=GIROLAMI-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:45:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rg625@cam.ac.uk
#SBATCH --array=0-15   # Total experiments: 2 TYPES x 2 MODLES x 4 DIMENSIONS = 16

# Load required modules
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module unload miniconda/3
module load cuda/11.8
module list
nvidia-smi

# Activate conda environment
source /home/rg625/.bashrc
conda activate koopman-ocean

# Move to project directory
cd /home/rg625/rds/hpc-work/ocean_forecasting/

# Define arrays of options
TYPES=("continous" "discrete")
MODELS=("mlp" "linear")
DIMENSIONS=(128 256 512 1024)  # Adjust if needed

# Map SLURM_ARRAY_TASK_ID to parameters
IDX=$SLURM_ARRAY_TASK_ID

NUM_DIM=${#DIMENSIONS[@]}
NUM_MODELS=${#MODELS[@]}
NUM_TYPES=${#TYPES[@]}

TYPE_IDX=$(( IDX / (NUM_MODELS * NUM_DIM) ))
MODEL_IDX=$(( (IDX / NUM_DIM) % NUM_MODELS ))
DIM_IDX=$(( IDX % NUM_DIM ))

TYPE=${TYPES[$TYPE_IDX]}
MODEL=${MODELS[$MODEL_IDX]}
DIM=${DIMENSIONS[$DIM_IDX]}

CONFIG_PATH=experiment/${TYPE}_${MODEL}_${DIM}.yaml

echo "IDX=$IDX TYPE=$TYPE MODEL=$MODEL DIM=$DIM CONFIG_PATH=$CONFIG_PATH"
echo "Running training for config: $CONFIG_PATH"

# Run training
python koopman_autoencoder/train.py --config "$CONFIG_PATH"