# # !/bin/bash

# Configuration Arrays
TYPES=("continous" "discrete")
ARCHS=("linear" "mlp")
SIZES=(128 64)
REGIME=("stable" "full" "tra")

BASE_OUT_DIR="/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion/src/results/sampling"

# for size in "${SIZES[@]}"; do
#   for type in "${TYPES[@]}"; do
#     for arch in "${ARCHS[@]}"; do
#       for regime in "${REGIME[@]}"; do

#         # Choose sampling modes based on regime
#         if [[ "$regime" == "stable" || "$regime" == "full" ]]; then
#           MODES=("highRey" "lowRey")
#         elif [[ "$regime" == "tra" ]]; then
#           MODES=("extrap" "interp" "longer")
#         fi

#         for mode in "${MODES[@]}"; do
#           OUT_DIR="${BASE_OUT_DIR}/${mode}/"

#           CONFIG_NAME="${regime}/${type}_${arch}_${size}"

#           echo "---------------------------------------------------"
#           echo "Running Evaluation for:"
#           echo "  Regime : $regime"
#           echo "  Mode   : $mode"
#           echo "  Config : $CONFIG_NAME"
#           echo "---------------------------------------------------"

#           python evaluate.py \
#             --config_name "$CONFIG_NAME" \
#             --out_dir "$OUT_DIR" \
#             --ckpt "199"

#         done
#       done
#     done
#   done
# done


CONFIG_NAME="tra/continous_linear_128"
OUT_DIR="${BASE_OUT_DIR}/interp"


python evaluate.py \
  --dim 128 \
  --type continous \
  --arch linear \
  --regime tra \
  --ckpt 60 \
  --gpu 0 \
  --out_dir "$BASE_OUT_DIR" \
  --eval_cases interp extrap longer

python ../autoreg_pde_diffusion/src/plot_loss_all.py
# python ../autoreg_pde_diffusion/src/plot_data_custom.py
python ../autoreg_pde_diffusion/src/plot_data.py
