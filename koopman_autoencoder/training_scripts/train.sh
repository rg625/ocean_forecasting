#!/bin/bash

TYPES=("continous") # "discrete"
ARCHS=("linear") # "mlp"
REGIMES=("tra") # "stable" "full"
SIZES=(128) #64

for regime in "${REGIMES[@]}"; do
    for size in "${SIZES[@]}"; do
        for type in "${TYPES[@]}"; do
            for arch in "${ARCHS[@]}"; do
                # This check remains the same
                CONFIG="configs/experiment/${regime}/${type}_${arch}_${size}.yaml"

                if [[ -f "$CONFIG" ]]; then
                    echo "Launching training with config: $CONFIG"
                    python train.py --config-path configs --config-name experiment/${regime}/${type}_${arch}_${size}

                else
                    echo "Skipping missing config: $CONFIG"
                fi
            done
        done
    done
done
