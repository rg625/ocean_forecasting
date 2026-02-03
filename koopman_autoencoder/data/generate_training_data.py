import torch_qg.model as torch_model
import torch_qg.parameterizations as torch_param
from tqdm import tqdm
import xarray as xr
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import glob


def run_test_sim(steps, hr_model=None, lr_model=None, sampling_freq=10, jet=False):
    """Run a qg simulation trajectory. We simulate in high res, and downsample to low res. We return
    downsampled fields, sampled at `sampling_freq` intervals in numerical timestep (because we generally
    emulate in different timesteps to numerical.

    Unfortunately torch_qg converts the tensors into xarrays, and we then convert this back into torch.
    If it becomes annoying can modify torch_qg, but will stick with this until then."""

    ## Prepare dictionary in case jet config is chosen
    ## If false, we just leave this dict empty
    flow_config = {}
    if jet:
        flow_config = {"rek": 7e-08, "delta": 0.1, "beta": 1e-11}
    if hr_model is None:
        hr_model = torch_model.PseudoSpectralModel(
            nx=256,
            dt=3600,
            dealias=True,
            parameterization=torch_param.Smagorinsky(),
            **flow_config,
        )
    if lr_model is None:
        lr_model = torch_model.PseudoSpectralModel(
            nx=64, dt=3600, dealias=True, **flow_config
        )
    ds = []

    ## Run spinup
    for aa in range(55990):
        hr_model._step_ab3()

    ## Run physical trajectory and save snapshots at whatever selected frequency
    for aa in tqdm(range(steps)):
        hr_model._step_ab3()
        if aa % sampling_freq == 0:
            ds.append(hr_model.forcing_dataset(lr_model))

    ds = xr.concat(ds, dim="time")
    return ds


# ----------------- Wrapper for a single simulation -----------------
def run_single_sim(sim_id, snapshots_per_sim):
    """
    Runs a single simulation and saves the output to a NetCDF file.
    """
    ds = run_test_sim(steps=snapshots_per_sim, sampling_freq=1, jet=False)
    save_file = f"/home/rg625/mnt/ocean_forecasting/koopman_autoencoder/data/qg/sims/sim_{sim_id:04d}.nc"
    ds.to_netcdf(save_file)
    return sim_id


def convert_qg_to_cfd(ds_qg, sim_id=0):
    """
    Convert a single QG dataset to CFD-like dataset with sim, t, x, y dims.
    ds_qg: xarray.Dataset with dims (time, lev, y, x)
    """
    # Extract q1 and q2
    q1 = ds_qg["q"][:, 0, :, :].values  # shape (time, y, x)
    q2 = ds_qg["q"][:, 1, :, :].values

    # Time dimension
    t = np.arange(q1.shape[0])

    # x and y coordinates
    x = np.linspace(-1, 1, q1.shape[2])  # y dimension resized
    y = np.linspace(-1, 1, q2.shape[1])  # x dimension

    # Create new dataset
    ds_cfd = xr.Dataset(
        {
            "q_1": (("sim", "t", "x", "y"), q1[np.newaxis, ...].astype(np.float32)),
            "q_2": (("sim", "t", "x", "y"), q2[np.newaxis, ...].astype(np.float32)),
        },
        coords={"sim": np.array([sim_id]), "t": t, "x": x, "y": y},
    )

    return ds_cfd


# ----------------- Main Parallel Execution -----------------
if __name__ == "__main__":
    n_sims = 1000  # Total simulations
    snapshots_per_sim = 500  # Snapshots per simulation
    parallel_sims = 4  # Number of simulations to run in parallel

    sim_ids = list(range(n_sims))

    # Run simulations in parallel threads
    with ThreadPoolExecutor(max_workers=parallel_sims) as executor:
        futures = [
            executor.submit(run_single_sim, sim_id, snapshots_per_sim)
            for sim_id in sim_ids
        ]

        for future in futures:
            sim_id_done = future.result()
            print(f"Simulation {sim_id_done} finished and saved.")

    files = sorted(
        glob.glob(
            "/home/rg625/mnt/ocean_forecasting/koopman_autoencoder/data/qg/sims/sim_*.nc"
        )
    )

    all_sims = []

    for i, f in enumerate(files):
        ds_qg = xr.open_dataset(f)
        ds_cfd = convert_qg_to_cfd(ds_qg, sim_id=i)
        all_sims.append(ds_cfd)

    # Concatenate along sim dimension
    final_ds = xr.concat(all_sims, dim="sim")

    # Save final dataset
    final_ds.to_netcdf(
        "/home/rg625/mnt/ocean_forecasting/koopman_autoencoder/data/qg/sims/qg_dataset.nc"
    )
