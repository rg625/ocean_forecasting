import numpy as np
import xarray as xr
import pyqg
from datetime import datetime
import os
from pathlib import Path
import matplotlib.pyplot as plt

def generate_qg_data(nx=256, ny=256, dt=3600.0, tmax=311040000.0, save_interval=86400.0, 
                     beta=1.5e-11, rd=1.0, H1=1.0, H2=0.2, U1=0.0, U2=0.0,
                     output_dir='training_data', model_type='eddy'):
    """
    Generate training data using pyqg's quasi-geostrophic models.
    
    Args:
        nx, ny: Grid resolution
        dt: Time step in seconds
        tmax: Maximum simulation time in seconds
        save_interval: Time interval between saved snapshots in seconds
        beta: Planetary vorticity gradient
        rd: Deformation radius
        H1, H2: Layer heights
        U1, U2: Background zonal velocities
        output_dir: Directory to save the data
        model_type: Type of QG model ('eddy', 'jet', or 'barotropic')
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize the model based on type
    if model_type == 'eddy':
        m = pyqg.QGModel(nx=nx, ny=ny, dt=dt, tmax=tmax,
                        tavestart=tmax/2)  # Start averaging at half the simulation time
    elif model_type == 'jet':
        m = pyqg.QGModel(nx=nx, ny=ny, dt=dt, tmax=tmax,
                        tavestart=tmax/2,  # Start averaging at half the simulation time
                        rek=7e-8,  # Bottom drag coefficient
                        delta=0.1,  # Layer thickness ratio
                        beta=1e-11)  # Planetary vorticity gradient
    elif model_type == 'barotropic':
        m = pyqg.BTModel(nx=nx, ny=ny, dt=dt, tmax=tmax,
                        beta=beta, rd=rd, U=U1,
                        rek=7e-8,  # Bottom drag coefficient
                        filterfac=23.6)
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
    for _ in m.run_with_snapshots(tsnapint=save_interval):
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
        print(f"Time: {m.t:.2f}/{tmax:.2f}")
    
    # Convert lists to numpy arrays
    times = np.array(times)
    for var in data:
        data[var] = np.array(data[var])
    
    # Create xarray dataset
    data_vars = {}
    for var in data:
        data_vars[var] = (['time', 'y', 'x'], data[var])
    
    full_dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            'x': np.arange(nx),
            'y': np.arange(ny),
            'time': times
        }
    )
    
    # Add metadata
    full_dataset.attrs['description'] = f'{model_type} QG model simulation data for Koopman autoencoder training'
    full_dataset.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Convert model parameters to string representation
    model_params = {
        'nx': nx, 'ny': ny, 'dt': dt, 'tmax': tmax,
        'beta': beta, 'rd': rd, 'H1': H1, 'H2': H2,
        'U1': U1, 'U2': U2, 'model_type': model_type
    }
    full_dataset.attrs['model_parameters'] = str(model_params)
    
    # Split data into train, validation, and test sets
    n_samples = len(full_dataset.time)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_data = full_dataset.isel(time=slice(0, train_size))
    val_data = full_dataset.isel(time=slice(train_size, train_size + val_size))
    test_data = full_dataset.isel(time=slice(train_size + val_size, None))
    
    # Save the datasets
    train_data.to_netcdf(output_dir / f'qg_{model_type}_train_data.nc')
    val_data.to_netcdf(output_dir / f'qg_{model_type}_val_data.nc')
    test_data.to_netcdf(output_dir / f'qg_{model_type}_test_data.nc')
    
    print(f"Data saved to {output_dir}")
    return train_data, val_data, test_data

def plot_data_statistics(dataset, output_dir):
    """Plot statistics of the generated data"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Plot time series of mean values
    plt.figure(figsize=(12, 6))
    for var in dataset.data_vars:
        mean_series = dataset[var].mean(dim=['x', 'y'])
        plt.plot(dataset.time, mean_series, label=var)
    plt.xlabel('Time')
    plt.ylabel('Mean Value')
    plt.title('Time Series of Mean Values')
    plt.legend()
    plt.savefig(output_dir / 'mean_time_series.png')
    plt.close()
    
    # Plot histograms
    n_vars = len(dataset.data_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5*n_rows))
    for i, var in enumerate(dataset.data_vars):
        plt.subplot(n_rows, n_cols, i+1)
        plt.hist(dataset[var].values.flatten(), bins=50, density=True)
        plt.title(f'{var} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(output_dir / 'variable_distributions.png')
    plt.close()

def main():
    # Create base data directory
    base_data_dir = Path(__file__).parent / 'data'
    base_data_dir.mkdir(exist_ok=True)
    
    # Generate training data for both model types
    for model_type in ['eddy', 'jet']:
        print(f"\nGenerating {model_type} model data...")
        model_data_dir = base_data_dir / model_type
        train_data, val_data, test_data = generate_qg_data(
            nx=256, ny=256,  # High resolution grid
            dt=3600.0,     # Time step in seconds (1 hour)
            tmax=311040000.0,  # Maximum simulation time (10 years in seconds)
            save_interval=86400.0,  # Save every day
            output_dir=model_data_dir,
            model_type=model_type
        )
        
        # Plot statistics
        plot_data_statistics(train_data, model_data_dir / 'plots')
        
        # Print dataset information
        print(f"\n{model_type} Dataset Information:")
        print(train_data.info())
        
        # Print basic statistics
        print(f"\nBasic Statistics:")
        print(f"Number of time steps: {len(train_data.time)}")
        print(f"Time range: {train_data.time[0].values:.2f} to {train_data.time[-1].values:.2f}")
        print(f"Grid size: {train_data.x.size}x{train_data.y.size}")
        
        # Print variable statistics
        for var in train_data.data_vars:
            print(f"\n{var} statistics:")
            print(f"Mean: {train_data[var].mean().values:.4f}")
            print(f"Std: {train_data[var].std().values:.4f}")
            print(f"Min: {train_data[var].min().values:.4f}")
            print(f"Max: {train_data[var].max().values:.4f}")

if __name__ == '__main__':
    main() 