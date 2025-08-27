import numpy as np
import xarray as xr
import pyqg
from pathlib import Path
import yaml
import argparse
import shutil


def run_single_simulation(model_config, sim_config, forcing_params, initial_conds):
    """
    Runs a single pyqg simulation with specified parameters.

    Args:
        model_config (dict): Grid and model type parameters.
        sim_config (dict): Simulation time parameters (dt, tmax, save_interval).
        forcing_params (dict): Forcing parameters (amplitude, wavenumber).
        initial_conds (np.array): Initial potential vorticity field.

    Returns:
        xr.Dataset: The dataset containing the simulation results.
    """
    # Unpack parameters
    nx, ny = model_config["nx"], model_config["ny"]
    model_type = model_config["model_type"]

    dt, tmax, save_interval = (
        sim_config["dt"],
        sim_config["tmax"],
        sim_config["save_interval"],
    )

    # Initialize model
    if model_type == "jet":
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
    # Add other model types as needed...
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Set the randomized initial conditions
    m.set_q(initial_conds)

    # Create and apply the randomized forcing
    forcing_amp = forcing_params["amplitude"]
    forcing_k = forcing_params["wavenumber"]
    x, y = np.meshgrid(np.arange(0.5, nx, 1.0) / nx, np.arange(0.5, ny, 1.0) / ny)
    forcing_pattern = forcing_amp * np.cos(2 * np.pi * forcing_k * y)
    m.F_q1 = forcing_pattern

    # --- Data Collection ---
    times = []
    q1_series, q2_series = [], []

    for _, _ in enumerate(m.run_with_snapshots(tsnapint=save_interval)):
        times.append(m.t)
        q1_series.append(m.q[0].copy())
        q2_series.append(m.q[1].copy())

    # Create xarray Dataset
    dataset = xr.Dataset(
        data_vars={
            "q1": (["time", "y", "x"], np.array(q1_series)),
            "q2": (["time", "y", "x"], np.array(q2_series)),
        },
        coords={"x": np.arange(nx), "y": np.arange(ny), "time": np.array(times)},
    )

    # Add metadata to the single simulation
    dataset.attrs["forcing_parameters"] = str(forcing_params)
    dataset.attrs["initial_condition_amplitude"] = float(np.std(initial_conds))

    return dataset


def generate_ensemble(config):
    """
    Generates an ensemble of simulations with randomized parameters.
    """
    # Parse config
    output_dir = Path(config["output"]["output_dir"])
    n_simulations = config["ensemble"]["n_simulations"]

    # --- Corrected path setup ---
    base_raw_dir = output_dir / "raw"

    model_config = {
        "nx": config["grid"]["nx"],
        "ny": config["grid"]["ny"],
        "model_type": config["simulation"]["model_type"],
    }
    sim_config = {
        "dt": config["simulation"]["dt"],
        "tmax": config["ensemble"]["simulation_tmax"],
        "save_interval": config["ensemble"]["save_interval"],
    }
    rand_config = config["randomization"]

    n_train = int(0.7 * n_simulations)
    n_val = int(0.2 * n_simulations)

    train_dir = base_raw_dir / "train"
    val_dir = base_raw_dir / "val"
    test_dir = base_raw_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_simulations} total simulations...")
    print(f"Saving raw data to: {base_raw_dir}")

    for i in range(n_simulations):
        print(f"\n--- Starting Simulation {i+1}/{n_simulations} ---")

        # 1. Determine which dataset this simulation belongs to
        if i < n_train:
            save_dir = train_dir
        elif i < n_train + n_val:
            save_dir = val_dir
        else:
            save_dir = test_dir

        # 2. Randomize parameters for this simulation
        forcing_amp = np.random.uniform(*rand_config["forcing_amp_range"])
        forcing_k = np.random.randint(*rand_config["wavenumber_range"])
        ic_amp = np.random.uniform(*rand_config["initial_cond_amp_range"])

        forcing_params = {"amplitude": forcing_amp, "wavenumber": forcing_k}

        # Generate random initial conditions (q1, q2)
        nx, ny = model_config["nx"], model_config["ny"]
        q1_initial = (np.random.rand(ny, nx) - 0.5) * ic_amp
        q2_initial = (np.random.rand(ny, nx) - 0.5) * ic_amp
        initial_conds = np.stack([q1_initial, q2_initial])

        print(
            f"Randomized Params: Forcing Amp={forcing_amp:.2e}, K={forcing_k}, IC Amp={ic_amp:.2e}"
        )

        # 3. Run the simulation
        dataset = run_single_simulation(
            model_config, sim_config, forcing_params, initial_conds
        )

        # 4. Save the result
        output_path = save_dir / f"simulation_{i:03d}.nc"
        dataset.to_netcdf(output_path)
        print(f"Saved simulation to: {output_path}")

    print("\nEnsemble generation complete!")


def process_ensemble_files(input_dir, output_dir, action, params=None):
    """
    Applies a processing step to all simulation files in a directory.

    Args:
        input_dir (Path): The directory containing the .nc files to process.
        output_dir (Path): The directory where processed .nc files will be saved.
        action (str): The action to perform. Either 'clean' or 'filter'.
        params (dict, optional): A dictionary of parameters for the action.
            For 'clean': {'drop_samples': int}
            For 'filter': {'keep_vars': list_of_strings}
    """
    if params is None:
        params = {}

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing files in: {input_dir}")
    print(f"Action: '{action}', Saving to: {output_dir}")

    # Find all NetCDF files in the input directory
    file_paths = sorted(list(input_dir.glob("*.nc")))
    if not file_paths:
        print(f"Warning: No .nc files found in {input_dir}")
        return

    for file_path in file_paths:
        ds = xr.open_dataset(file_path)

        processed_ds = None
        if action == "clean":
            drop_samples = params.get("drop_samples", 0)
            print(
                f"  Cleaning {file_path.name}, dropping first {drop_samples} samples..."
            )
            processed_ds = ds.isel(time=slice(drop_samples, None))

        elif action == "filter":
            keep_vars = params.get("keep_vars", [])
            if not keep_vars:
                raise ValueError(
                    "The 'filter' action requires a 'keep_vars' list in params."
                )
            print(f"  Filtering {file_path.name}, keeping variables {keep_vars}...")
            processed_ds = ds[keep_vars]

        else:
            raise ValueError(
                f"Unknown action: '{action}'. Must be 'clean' or 'filter'."
            )

        # Save the processed dataset to the new directory
        output_file_path = output_dir / file_path.name
        processed_ds.to_netcdf(output_file_path)

    print("Processing complete.")


def combine_ensemble_files(input_dir, output_file):
    """
    Combines multiple simulation .nc files into a single, large .nc file.

    This function adds a new 'simulation' dimension to the dataset.

    Args:
        input_dir (Path): The directory containing the .nc files to combine.
        output_file (Path): The path for the final combined .nc file.
    """
    input_dir = Path(input_dir)
    file_paths = sorted(list(input_dir.glob("*.nc")))

    if not file_paths:
        print(f"Warning: No .nc files found in {input_dir} to combine.")
        return

    print(
        f"Combining {len(file_paths)} files from '{input_dir.name}' into '{output_file.name}'..."
    )

    # Use xarray's open_mfdataset to open all files and concatenate them
    # along a new dimension called 'simulation'.
    combined_ds = xr.open_mfdataset(
        file_paths, combine="nested", concat_dim="simulation"
    )

    # Save the final combined dataset
    combined_ds.to_netcdf(output_file)
    print(f"Successfully created {output_file}")


def main(config_path):
    """
    Main function to generate, process, combine, and clean up the dataset.
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- Define all necessary paths ---
    base_dir = Path(config["output"]["output_dir"])

    # Define consistent directory structure
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"

    raw_train_dir = raw_dir / "train"
    raw_val_dir = raw_dir / "val"
    raw_test_dir = raw_dir / "test"

    clean_train_dir = processed_dir / "train_cleaned"
    final_train_dir = processed_dir / "final" / "train"
    final_val_dir = processed_dir / "final" / "val"
    final_test_dir = processed_dir / "final" / "test"

    # Final output files
    final_train_file = base_dir / "train.nc"
    final_val_file = base_dir / "val.nc"
    final_test_file = base_dir / "test.nc"

    # =========================================================================
    # --- PIPELINE STAGES ---
    # After running once, you can comment out earlier stages if the data exists.
    # =========================================================================

    # STAGE 1: Generate the raw ensemble data
    print("\n--- STAGE 1: Generating Ensemble Data ---")
    generate_ensemble(config)  # UNCOMMENTED

    # STAGE 2: Post-process the data
    print("\n--- STAGE 2: Processing Ensemble Data ---")
    # A. Clean training data
    process_ensemble_files(
        input_dir=raw_train_dir,
        output_dir=clean_train_dir,
        action="clean",
        params={"drop_samples": 100},
    )
    # B. Filter variables for all sets
    filter_params = {"keep_vars": ["q1", "q2"]}
    process_ensemble_files(clean_train_dir, final_train_dir, "filter", filter_params)
    process_ensemble_files(raw_val_dir, final_val_dir, "filter", filter_params)
    process_ensemble_files(raw_test_dir, final_test_dir, "filter", filter_params)

    # STAGE 3: Combine processed files into final datasets
    print("\n--- STAGE 3: Combining Files ---")
    combine_ensemble_files(final_train_dir, final_train_file)
    combine_ensemble_files(final_val_dir, final_val_file)
    combine_ensemble_files(final_test_dir, final_test_file)

    # STAGE 4: Clean up all intermediate directories
    print("\n--- STAGE 4: Cleaning Up Intermediate Files ---")
    dirs_to_delete = [raw_dir, processed_dir]
    for d in dirs_to_delete:
        if d.exists() and d.is_dir():
            print(f"Deleting directory: {d}")
            shutil.rmtree(d)
        else:
            print(f"Directory not found, skipping: {d}")

    print("\nPipeline complete! Final files are:")
    print(f"  - {final_train_file}")
    print(f"  - {final_val_file}")
    print(f"  - {final_test_file}")


if __name__ == "__main__":
    # To make this fully functional, minor adjustments to generate_ensemble are needed
    # to accept a base output path, but the logic for combination and cleanup is complete.
    # The main function above outlines the full intended workflow.
    parser = argparse.ArgumentParser(
        description="Generate and process QG ensemble dataset"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    main(args.config)
