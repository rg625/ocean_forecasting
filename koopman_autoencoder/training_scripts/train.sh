#!/bin/bash

# A script to train the Koopman Autoencoder model with the specified configuration

# Define the path to the configuration file
CONFIG_PATH=~/mnt/ocean_forecasting/koopman_autoencoder/configs/model/128_inc.yaml
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WANDB_API_KEY=763e928e18a8016e7072d06f3ae5f8e2b304f89c
# Run the training script with the configuration file
python ./train_ddp.py --config "$CONFIG_PATH"
