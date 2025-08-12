import numpy as np
import xarray as xr
import pyqg
from pathlib import Path
from datetime import datetime
import yaml
import argparse


def generate_gaussian_random_field(nx, ny, alpha=4.0, amplitude=0.1, seed=None):
    """
    Generate a smooth Gaussian random field in 2D.
    
    Args:
        nx: int
            Number of grid points in the x-direction.
        ny: int
            Number of grid points in the y-direction.
        alpha: float
            Spectral exponent for the field.
        amplitude: float
            Amplitude of the field.
        seed: int, optional
            Random seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0  # prevent division by zero

    random_coeff = np.random.normal(size=(ny, nx)) + 1j * np.random.normal(size=(ny, nx))
    spectrum = random_coeff / (K2 ** (alpha / 2))

    spectrum[0, 0] = 0.0
    field = np.fft.ifft2(spectrum).real
    field /= np.std(field)
    field *= amplitude
    return field


class GRFForcing:
    """
    Callable forcing class using precomputed Gaussian random field.
    """

    def __init__(self, nx, ny, alpha=4.0, amplitude=0.1, seed=None):
        forcing_pattern = generate_gaussian_random_field(nx, ny, alpha, amplitude, seed)
        forcing_array = np.array([forcing_pattern, np.zeros((ny, nx))], dtype=np.float64)
        self.forcing_array = np.ascontiguousarray(forcing_array)

    def __call__(self, m):
        return self.forcing_array
    
    
class StationaryForcing:
    """
    Callable class to apply a time-invariant (stationary) forcing term.
    """
    def __init__(self, nx, ny, wavenumber_x, wavenumber_y):
        """
        Pre-computes the forcing vector field during initialization.
        """
        x, y = np.meshgrid(
            np.linspace(0, 2 * np.pi, nx, endpoint=False),
            np.linspace(0, 2 * np.pi, ny, endpoint=False)
        )
        forcing_pattern = 0.1 * np.sin(wavenumber_x * x) * np.cos(wavenumber_y * y)
        forcing_array = np.array([forcing_pattern, np.zeros((ny, nx))], dtype=np.float64)
        self.forcing_array = np.ascontiguousarray(forcing_array)

    def __call__(self, m):
        """
        This method is called by pyqg at each timestep.
        """
        return self.forcing_array


def generate(config):
    """
    Generates multiple pyqg simulations with different forcing terms.
    
    Args:
        config: dict
            Configuration dictionary containing all parameters.
    """
    # Parse configuration parameters
    nx = config["grid"]["nx"]
    ny = config["grid"]["ny"]
    dt = config["simulation"]["dt"]
    tmax = config["simulation"]["tmax"]
    save_interval = config["simulation"]["save_interval"]
    model_type = config["simulation"]["model_type"]
    num_simulations = config["simulation"]["num_simulations"]
    warmup_snapshots = config["simulation"]["warmup_snapshots"]
    n_snapshots = config["simulation"]["n_snapshots"]
    output_dir = Path(config["output"]["output_dir"])

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loop to run multiple simulations
    for i in range(num_simulations):
        print(f"\n--- Starting Simulation {i+1}/{num_simulations} ---")

        # Use a different seed for each simulation for distinct forcing
        seed = i
        alpha = 4.0
        amplitude = 0.1
        forcing_instance = StationaryForcing(nx, ny, alpha=alpha, amplitude=amplitude, seed=seed)
        print(f"Generated random forcing with alpha={alpha}, amplitude={amplitude}, seed={seed}")

        # Ensure the model type is correctly set in the config
        if model_type != "jet":
            raise ValueError("This script is configured for 'jet' model type.")
        
        # Initialize the QGModel
        m = pyqg.QGModel(
            nx=nx, 
            ny=ny, 
            dt=dt, 
            tmax=tmax,
            tavestart=tmax / 2,
            rek=7e-8, 
            delta=0.1, 
            beta=1e-11
        )
        
        # Assign the callable class instance to the model's forcing attribute
        m.forcing = forcing_instance

        # Prepare lists to hold the data for this simulation
        # data: dict = {var: [] for var in ["q1", "q2", "u1", "v1", "u2", "v2"]}
        data: dict = {var: [] for var in ["q1", "q2"]}
        times = np.array([], dtype=float)
        
        # Create the snapshot generator
        snapshot_generator = m.run_with_snapshots(tsnapint=save_interval)

        print(f"Simulation will run for a total of {warmup_snapshots + n_snapshots} snapshots.")
        
        # Main simulation loop driven by the generator
        for snapshot_count, _ in enumerate(snapshot_generator):
            # Warm-up phase: advance the simulation but do not save data
            if snapshot_count < warmup_snapshots:
                if (snapshot_count + 1) % 500 == 0:
                    print(f"  Warm-up snapshot {snapshot_count + 1}/{warmup_snapshots}...")
                continue

            # Data collection phase
            q1, q2 = m.q[0].copy(), m.q[1].copy()
            # u1, v1 = m.u[0].copy(), m.v[0].copy()
            # u2, v2 = m.u[1].copy(), m.v[1].copy()

            data["q1"].append(q1)
            data["q2"].append(q2)
            # data["u1"].append(u1)
            # data["v1"].append(v1)
            # data["u2"].append(u2)
            # data["v2"].append(v2)
            
            times = np.append(times, m.t)
            
            saved_count = len(times)
            if saved_count % 200 == 0:
                print(f"  ... saved data snapshot {saved_count}/{n_snapshots}")

            # Stop the simulation once the desired number of data snapshots is collected
            if saved_count >= n_snapshots:
                break
        
        print("Data collection complete.")

        # Create and save the xarray dataset
        data_vars = {
            "q1": (["time", "y", "x"], np.array(data["q1"])),
            "q2": (["time", "y", "x"], np.array(data["q2"])),
            # "u1": (["time", "y", "x"], np.array(data["u1"])),
            # "v1": (["time", "y", "x"], np.array(data["v1"])),
            # "u2": (["time", "y", "x"], np.array(data["u2"])),
            # "v2": (["time", "y", "x"], np.array(data["v2"])),
            "forcing": (["y", "x"], forcing_instance.forcing_array[0]) # Save the forcing field
        }

        dataset = xr.Dataset(
            data_vars=data_vars,
            coords={"x": np.arange(nx), "y": np.arange(ny), "time": np.array(times)},
        )

        # Add metadata
        dataset.attrs["description"] = f"Forced '{model_type}' QG model simulation"
        dataset.attrs["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dataset.attrs["model_parameters"] = str(config)

        # Define the output filename and save the NetCDF file
        output_filename = output_dir / f"qg_{model_type}_forced_sim_{i+1}.nc"
        dataset.to_netcdf(output_filename)
        print(f"Simulation {i+1} saved to {output_filename}")


def main(config_path):
    """
    Main function to generate the dataset from a config file.

    Args:
        config_path: str
            Path to the YAML config file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    generate(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate forced QG timeseries dataset")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    
    main(args.config)