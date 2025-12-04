# # !/bin/bash

# Configuration Arrays
# arch in python script corresponds to "discrete" or "continous" (TYPES in your bash vars)
TYPES=("discrete" "continous")
# type in python script corresponds to "linear" or "mlp" (ARCHS in your bash vars)
ARCHS=("linear" "mlp")
# dimension
SIZES=(128 1024)
REGIME=("stable" "full")

# Output Directory
OUT_DIR="/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion/src/results/sampling/lowRey2/"

for size in "${SIZES[@]}"; do
  for type in "${TYPES[@]}"; do
    for arch in "${ARCHS[@]}"; do
      for regime in "${REGIME[@]}"; do

        # Construct config name: arch_type_dim
        # Note: 'type' var here maps to 'model_arch' (discrete/continuous)
        #       'arch' var here maps to 'model_type' (linear/mlp)
        CONFIG_NAME="${regime}/${type}_${arch}_${size}"

        echo "---------------------------------------------------"
        echo "Running Evaluation for: $CONFIG_NAME"
        echo "---------------------------------------------------"

        python evaluate.py \
          --config_name "$CONFIG_NAME" \
          --out_dir "$OUT_DIR"
      done
    done
  done
done
