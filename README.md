# Ocean Forecasting with Koopman Autoencoder

This repository provides a framework for ocean forecasting using machine learning techniques, specifically leveraging Koopman Autoencoders. The project aims to model and predict oceanic processes with high-dimensional spatiotemporal data.

## Features
- **Koopman Autoencoder Model**:
  - Implements a convolutional autoencoder to extract latent representations.
  - Uses Koopman operators for temporal predictions in latent space.
- **Custom Loss Function**:
  - Includes reconstruction loss, prediction loss, and latent consistency loss.
- **Data Handling**:
  - Tools for generating and preprocessing ocean simulation datasets.
  - Custom batch sampler for handling variable sequence lengths.
- **Visualization**:
  - Visualize reconstructions, predictions, and isotropic energy spectra.
- **Integration with W&B**:
  - Logs training metrics and visualizations to Weights & Biases.

## Repository Structure
```
ocean_forecasting/
├── koopman_autoencoder/
│   ├── configs/
│   │   ├── models/
│   │   │   └── config.yaml         # Model and training configuration
│   │   ├── data/
│   │   │   └── config.yaml         # Data generation configuration
│   ├── models/
│   │   ├── __init__.py
│   │   ├── autoencoder.py          # Model definition
│   │   ├── checkpoint.py           # Gradient checkpointing
│   │   ├── cnn.py                  # Base networks
│   │   ├── dataloader.py           # Custom dataset and data loader
│   │   ├── loss.py                 # Loss function
│   │   ├── lr_schedule.py          # Custom learning rate scheduler
│   │   ├── trainer.py              # Main training methods
│   │   ├── utils.py                # Visualization and logging utilities
│   ├── data/
│   │   ├── data/  # Script to generate training data
│   │   ├── generate_training_data.py  # Script to generate training data
│   ├── train.py                      # Main training script
│   ├── train.sh                      # Shell file for training
│   └── generate.sh                   # Shell file for data generation
└── README.md                         # Project documentation
```

## Installation

### Prerequisites
- Python 3.10+
- Additional libraries: `numpy`, `xarray`, `matplotlib`, `pyqg`, `wandb`, `yaml`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/rg625/ocean_forecasting.git
   cd ocean_forecasting
   ```

2. Set up a Python environment:
   ```bash
   conda env create -f environment.yml
   conda activate koopman-ocean
   ```

3. Configure the project:
   - Update `koopman_autoencoder/configs/model/config.yaml` for model and training parameters.
   - Update `koopman_autoencoder/config/data/config.yaml` for data generation parameters.

## Usage

### 1. Generate Training Data
Run the data generation script:
```bash
sh generate.sh
```

### 2. Train the Model
Train the Koopman Autoencoder:
```bash
sh train.sh
```
## Key Components

### Model
The Koopman Autoencoder consists of:
- **Encoder**: Compresses high-dimensional input data into a latent space.
- **Decoder**: Reconstructs the input data from the latent space.
- **Koopman Operator**: Predicts future latent states using linear dynamics.

### Custom Loss Function
The loss function balances:
- **Reconstruction Loss**: Measures how well the model reconstructs input data.
- **Prediction Loss**: Evaluates the accuracy of future predictions.
- **Latent Loss**: Ensures consistency in latent dynamics.

### Data
The repository includes tools for generating synthetic datasets using the `pyqg` library. Datasets are split into training, validation, and testing sets.

## Configuration
The main configurations are stored in YAML files:
- `configs/model/config.yaml`: Defines model architecture, training parameters, and loss weights.
- `configs/data/config.yaml`: Sets grid resolution, simulation time, and output directory.

## Results
- **Reconstruction**: The model reconstructs input fields with high fidelity.
- **Prediction**: The Koopman operator provides accurate short-term forecasts.
- **Energy Spectra**: The isotropic energy spectra are compared for true and predicted fields.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request. For better code consitency please run
```bash
pre-commit install
```

## Acknowledgments
This project uses the following libraries:
- [PyTorch](https://pytorch.org/)
- [xarray](http://xarray.pydata.org/)
- [PyQG](https://pyqg.readthedocs.io/en/latest/)
- [Weights & Biases](https://wandb.ai/)

## Contact
For questions or feedback, please contact [rg625](https://github.com/rg625).
