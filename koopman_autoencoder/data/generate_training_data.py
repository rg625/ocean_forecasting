import torch
import numpy as np
import xarray as xr
import pyqg
from pathlib import Path
from datetime import datetime
import yaml


def generate(config):
    """
    Generate a full dataset with 10,000 samples of a single time series.

    Args:
        config: dict
            Configuration dictionary containing all parameters.
    """
    # Parse config
    nx = config['grid']['nx']
    ny = config['grid']['ny']
    dt = config['simulation']['dt']
    tmax = config['simulation']['tmax']
    save_interval = config['simulation']['save_interval']
    model_type = config['simulation']['model_type']
    output_dir = Path(config['output']['output_dir'])

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the model based on type
    if model_type == 'eddy':
        m = pyqg.QGModel(nx=nx, ny=ny, dt=dt, tmax=tmax, tavestart=tmax / 2)
    elif model_type == 'jet':
        m = pyqg.QGModel(
            nx=nx, ny=ny, dt=dt, tmax=tmax, tavestart=tmax / 2, rek=7e-8, delta=0.1, beta=1e-11
        )
    elif model_type == 'barotropic':
        m = pyqg.BTModel(nx=nx, ny=ny, dt=dt, tmax=tmax, beta=1.5e-11, rd=1.0, U=0.0, rek=7e-8, filterfac=23.6)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Run the model and save data
    print("Starting simulation...")

    # Create empty lists to store data
    times = []
    if model_type in ['eddy', 'jet']:
        data = {var: [] for var in ['q1', 'q2', 'u1', 'v1', 'u2', 'v2']}
    else:
        data = {var: [] for var in ['q', 'u', 'v']}

    # Run the model with snapshots
    for i, _ in enumerate(m.run_with_snapshots(tsnapint=save_interval)):
        if len(times) >= 10000:  # Stop after generating 10,000 samples
            break

        if model_type in ['eddy', 'jet']:
            data['q1'].append(m.q[0].copy())
            data['q2'].append(m.q[1].copy())
            data['u1'].append(m.u[0].copy())
            data['v1'].append(m.v[0].copy())
            data['u2'].append(m.u[1].copy())
            data['v2'].append(m.v[1].copy())
        else:  # barotropic
            data['q'].append(m.q.copy())
            data['u'].append(m.u.copy())
            data['v'].append(m.v.copy())

        times.append(m.t)
        print(f"Saved snapshot {len(times)}/10000 at time {m.t:.2f}")

    # Convert lists to numpy arrays
    times = np.array(times)
    for var in data:
        data[var] = np.array(data[var])

    # Create xarray dataset
    data_vars = {}
    for var in data:
        data_vars[var] = (['time', 'y', 'x'], data[var])

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={'x': np.arange(nx), 'y': np.arange(ny), 'time': times},
    )

    # Add metadata
    dataset.attrs['description'] = f'{model_type} QG model simulation data for Koopman autoencoder training'
    dataset.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    dataset.attrs['model_parameters'] = str(config)

    # Split data into train, validation, and test sets
    n_samples = len(dataset.time)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    train_data = dataset.isel(time=slice(0, train_size))
    val_data = dataset.isel(time=slice(train_size, train_size + val_size))
    test_data = dataset.isel(time=slice(train_size + val_size, None))

    # Save the datasets
    train_data.to_netcdf(output_dir / f'qg_{model_type}_train_data.nc')
    val_data.to_netcdf(output_dir / f'qg_{model_type}_val_data.nc')
    test_data.to_netcdf(output_dir / f'qg_{model_type}_test_data.nc')

    print(f"Data saved to {output_dir}")
    return train_data, val_data, test_data


def main(config_path):
    """
    Main function to generate the dataset from a config file.

    Args:
        config_path: str
            Path to the YAML config file.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Generate dataset
    generate(config)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate QG full timeseries dataset")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    main(args.config)