#!/bin/bash

# A script to train the Koopman Autoencoder model with the specified configuration

# Define the path to the configuration file
CONFIG_PATH=~/mnt/ocean_forecasting/koopman_autoencoder/configs/data/config.yaml

# Run the training script with the configuration file
python koopman_autoencoder/data/generate_training_data.py --config "$CONFIG_PATH"
