#!/bin/bash
# #############################################################################
# Script to run the static evaluation pipeline based on the provided config.
# #############################################################################

PYTHON_SCRIPT="evaluate.py"
EVAL_CONFIG_FILE=$1
OTHER_ARGS="${@:2}" # Captures optional flags like --generate_plots

if [ -z "$EVAL_CONFIG_FILE" ]; then
    echo "Error: No evaluation config file provided."
    echo "Usage: $0 path/to/your/evaluation_config.yaml"
    exit 1
fi
# ... (you can keep the other file existence checks if you wish) ...

echo "Starting model evaluation using config: $EVAL_CONFIG_FILE"
echo "---"

python "$PYTHON_SCRIPT" --eval_config "$EVAL_CONFIG_FILE" # --generate_plots

echo "---"
echo "Evaluation script finished."
