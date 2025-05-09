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
    nx = config["grid"]["nx"]
    ny = config["grid"]["ny"]
    dt = config["simulation"]["dt"]
    tmax = config["simulation"]["tmax"]
    save_interval = config["simulation"]["save_interval"]
    model_type = config["simulation"]["model_type"]
    output_dir = Path(config["output"]["output_dir"])

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the model based on type
    if model_type == "eddy":
        m = pyqg.QGModel(nx=nx, ny=ny, dt=dt, tmax=tmax, tavestart=tmax / 2)
    elif model_type == "jet":
        m = pyqg.QGModel(
            nx=nx,
            ny=ny,
            dt=dt,
            tmax=tmax,
            tavestart=tmax / 2,
            rek=7e-8,
            delta=0.1,
            beta=1e-11,
        )
    elif model_type == "barotropic":
        m = pyqg.BTModel(
            nx=nx,
            ny=ny,
            dt=dt,
            tmax=tmax,
            beta=1.5e-11,
            rd=1.0,
            U=0.0,
            rek=7e-8,
            filterfac=23.6,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Run the model and save data
    print("Starting simulation...")

    # Create empty lists to store data
    # Initialize times as a numpy array
    times = np.array([], dtype=float)

    if model_type in ["eddy", "jet"]:
        data: dict = {var: [] for var in ["q1", "q2", "u1", "v1", "u2", "v2"]}
    else:
        data = {var: [] for var in ["q", "u", "v"]}

    # Run the model with snapshots
    log_interval = 100  # Log every 100 snapshots

    for i, _ in enumerate(m.run_with_snapshots(tsnapint=save_interval)):
        if len(times) >= 10000:
            break

        # Save snapshot
        if model_type in ["eddy", "jet"]:
            q1, q2 = m.q[0].copy(), m.q[1].copy()
            u1, v1 = m.u[0].copy(), m.v[0].copy()
            u2, v2 = m.u[1].copy(), m.v[1].copy()

            data["q1"].append(q1)
            data["q2"].append(q2)
            data["u1"].append(u1)
            data["v1"].append(v1)
            data["u2"].append(u2)
            data["v2"].append(v2)

            if i % log_interval == 0:
                print(f"\nStats at snapshot {i} (time={m.t:.2f}):")
                for name, arr in zip(
                    ["q1", "q2", "u1", "v1", "u2", "v2"], [q1, q2, u1, v1, u2, v2]
                ):
                    print(
                        f"  {name}: mean={np.mean(arr):.4e}, std={np.std(arr):.4e}, "
                        f"min={np.min(arr):.4e}, max={np.max(arr):.4e}"
                    )
        else:
            q, u, v = m.q.copy(), m.u.copy(), m.v.copy()
            data["q"].append(q)
            data["u"].append(u)
            data["v"].append(v)

            if i % log_interval == 0:
                print(f"\nStats at snapshot {i} (time={m.t:.2f}):")
                for name, arr in zip(["q", "u", "v"], [q, u, v]):
                    print(
                        f"  {name}: mean={np.mean(arr):.4e}, std={np.std(arr):.4e}, "
                        f"min={np.min(arr):.4e}, max={np.max(arr):.4e}"
                    )

        times = np.append(times, m.t)
        print(f"Saved snapshot {len(times)}/10000 at time {m.t:.2f}")

    # Convert lists to numpy arrays
    times = np.array(times)
    for var in data:
        data[var] = np.array(data[var])

    # Create xarray dataset
    data_vars = {}
    for var in data:
        data_vars[var] = (["time", "y", "x"], data[var])

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={"x": np.arange(nx), "y": np.arange(ny), "time": times},
    )

    # Add metadata
    dataset.attrs["description"] = (
        f"{model_type} QG model simulation data for Koopman autoencoder training"
    )
    dataset.attrs["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset.attrs["model_parameters"] = str(config)

    # Split data into train, validation, and test sets
    n_samples = len(dataset.time)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    train_data = dataset.isel(time=slice(0, train_size))
    val_data = dataset.isel(time=slice(train_size, train_size + val_size))
    test_data = dataset.isel(time=slice(train_size + val_size, None))

    # Save the datasets
    train_data.to_netcdf(output_dir / f"qg_{model_type}_train_data.nc")
    val_data.to_netcdf(output_dir / f"qg_{model_type}_val_data.nc")
    test_data.to_netcdf(output_dir / f"qg_{model_type}_test_data.nc")

    print(f"Data saved to {output_dir}")
    return train_data, val_data, test_data


def clean_training_data(train_data_path, output_path, drop_samples=2000):
    """
    Remove initial unstable samples from training data.

    Args:
        train_data_path: str or Path
            Path to the training data NetCDF file.
        output_path: str or Path
            Path to save the cleaned training dataset.
        drop_samples: int
            Number of initial samples to drop.
    """
    # Load training data
    train_data = xr.open_dataset(train_data_path)
    print(f"Original training data time steps: {len(train_data.time)}")

    # Drop initial unstable samples
    cleaned_data = train_data.isel(time=slice(drop_samples, None))
    print(f"Cleaned training data time steps: {len(cleaned_data.time)}")

    # Save the cleaned data
    cleaned_data.to_netcdf(output_path)
    print(f"Cleaned training data saved to {output_path}")


def retain_q1_q2_only(data_path, output_path):
    """
    Retain only q1 and q2 variables in a NetCDF dataset.

    Args:
        data_path: str or Path
            Path to the original NetCDF dataset (train/val/test).
        output_path: str or Path
            Path to save the reduced dataset with only q1 and q2.
    """
    ds = xr.open_dataset(data_path)
    print(f"Original variables: {list(ds.data_vars)}")

    # Drop variables that are not q1 or q2
    ds_reduced = ds[["q1", "q2"]]
    print(f"Retaining variables: {list(ds_reduced.data_vars)}")

    # Save the reduced dataset
    ds_reduced.to_netcdf(output_path)
    print(f"Reduced dataset saved to {output_path}")


def main(config_path):
    """
    Main function to generate the dataset from a config file.

    Args:
        config_path: str
            Path to the YAML config file.
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Generate dataset
    generate(config)

    # Example usage
    train_data_path = f"{config['output']['output_dir']}/qg_{config['simulation']['model_type']}_train_data.nc"
    output_path = f"{config['output']['output_dir']}/qg_{config['simulation']['model_type']}_train_data_clean.nc"
    clean_training_data(train_data_path, output_path, drop_samples=2000)
    retain_q1_q2_only(train_data_path, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate QG full timeseries dataset")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    main(args.config)
