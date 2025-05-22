#!/bin/bash

# A script to train the Koopman Autoencoder model with the specified configuration

# Define the path to the configuration file
CONFIG_PATH=~/mnt/ocean_forecasting/koopman_autoencoder/configs/model/config_diff.yaml

# Run the training script with the configuration file
# python train.py --config "$CONFIG_PATH"
python ./train_diffusion.py --config "$CONFIG_PATH"
