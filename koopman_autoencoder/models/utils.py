import numpy as np
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
import torch
from tensordict import TensorDict
from torch import Tensor
from models.autoencoder import KoopmanAutoencoder
from torch.optim.adam import Adam


def tensor_dict_to_json(tensor_dict: TensorDict):
    """
    Convert a TensorDict or Tensor to a JSON-serializable dictionary or list.

    Args:
        tensor_dict (TensorDict or torch.Tensor): Input TensorDict or tensor.

    Returns:
        dict or list or scalar: JSON-serializable dictionary, list, or scalar.
    """
    if isinstance(tensor_dict, torch.Tensor):
        # Handle tensors: return as a Python scalar if it's a single value, otherwise convert to a list
        return (
            tensor_dict.item()
            if tensor_dict.numel() == 1
            else tensor_dict.cpu().numpy().tolist()
        )
    elif isinstance(tensor_dict, TensorDict):
        # Handle TensorDict: recursively convert each item to JSON-serializable format
        return {key: tensor_dict_to_json(value) for key, value in tensor_dict.items()}
    else:
        raise TypeError(
            f"Unsupported type for tensor_dict_to_json: {type(tensor_dict)}"
        )


def plot_comparison(
    x: TensorDict,
    x_recon: TensorDict,
    output_dir: Path | str = "/home/koopman/",
    title: str = "reconstruction_comparison",
    mode="train",
):
    """
    Plots input vs reconstructed images for each variable across all timesteps in one image.
    Supports inputs of shape (B, T, H, W).
    """
    variables = x.keys()
    num_channels = len(variables)

    # Assume shape (B, T, H, W)
    T = next(iter(x.values())).shape[1]  # Number of timesteps

    # Create a figure with enough space to fit all the timesteps for all variables
    fig, axes = plt.subplots(3, T * num_channels, figsize=(6 * num_channels, 4 * 3))

    if num_channels == 1:
        axes = np.expand_dims(axes, axis=1)  # Ensure 2D layout

    # For each timestep, append the images in a grid
    for t in range(T):
        for j, var in enumerate(variables):
            x_var = x[var][0, t]  # (H, W)
            x_recon_var = x_recon[var][0, t]  # (H, W)
            error = x_var - x_recon_var

            # Plot the true image, reconstructed image, and error
            mat = axes[0, t + j * T].matshow(x_var.cpu().numpy(), cmap="RdBu_r")
            axes[0, t + j * T].set_title(f"True {var}")
            axes[0, t + j * T].axis("off")

            # Adjust the colorbar size with the shrink parameter
            cbar = plt.colorbar(mat, ax=axes[0, t + j * T], fraction=0.046, pad=0.04)
            cbar.set_label("Intensity")

            mat = axes[1, t + j * T].matshow(x_recon_var.cpu().numpy(), cmap="RdBu_r")
            axes[1, t + j * T].set_title(f"Recon {var}")
            axes[1, t + j * T].axis("off")

            cbar = plt.colorbar(mat, ax=axes[1, t + j * T], fraction=0.046, pad=0.04)
            cbar.set_label("Intensity")

            mat = axes[2, t + j * T].matshow(error.cpu().numpy(), cmap="hot")
            axes[2, t + j * T].set_title(f"Error {var}")
            axes[2, t + j * T].axis("off")

            cbar = plt.colorbar(mat, ax=axes[2, t + j * T], fraction=0.046, pad=0.04)
            cbar.set_label("Intensity")

    # Set a clear and large title depending on mode
    plt.suptitle(f"{title}", fontsize=16)

    # Apply tight_layout to ensure no clipping and good spacing
    plt.tight_layout()

    # Adjust the layout further to make space for the colorbars
    plt.subplots_adjust(
        right=0.85, hspace=0.5
    )  # Adjust right side to make room for colorbars

    # Save and log the figure if an output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{title}.png"
        plt.savefig(filename, dpi=150)
        wandb.log({f"figures/{mode}/{title}": wandb.Image(str(filename))})

    plt.close()


def compute_isotropic_energy_spectrum(field: Tensor):
    """
    Compute isotropic energy spectrum from a 2D velocity or vorticity field.

    Parameters:
        field (torch.Tensor): (B, H, W) tensor with a batch of spatial fields for a single variable.

    Returns:
        k_bins (np.ndarray): Radial wavenumber bins
        spectrum (np.ndarray): Isotropic energy spectrum
    """
    B, H, W = field.shape
    spectrum = np.zeros(H // 2)

    for b in range(B):
        # 2D FFT and power spectrum
        f_hat = np.fft.fft2(field[b].cpu().numpy())
        psd2d = np.abs(f_hat) ** 2

        # Generate wavenumber grid
        kx = np.fft.fftfreq(W).reshape(1, -1) * W
        ky = np.fft.fftfreq(H).reshape(-1, 1) * H
        k_mag = np.sqrt(kx**2 + ky**2)
        k_mag = np.fft.fftshift(k_mag)

        psd2d = np.fft.fftshift(psd2d)
        k_flat = k_mag.ravel()
        psd_flat = psd2d.ravel()

        # Bin the power spectrum isotropically
        k_bins = np.linspace(0, k_mag.max(), H // 2 + 1)  # +1 to match bin edges
        bin_indices = np.digitize(k_flat, k_bins) - 1  # Match bins with indices
        for i in range(len(k_bins) - 1):  # Iterate over the actual bins
            mask = bin_indices == i
            if mask.any():
                spectrum[i] += psd_flat[mask].mean()

    k_bins = 0.5 * (k_bins[:-1] + k_bins[1:])  # Use midpoints for plotting
    spectrum /= B  # Average over the batch
    return k_bins, spectrum


def plot_energy_spectrum(
    true_fields: Tensor,
    pred_fields: Tensor,
    output_dir: Path | str = "/home/koopman/",
    mode: str = "train",
):
    """
    Plots the isotropic energy spectrum comparison for each channel and logs to W&B.

    Parameters:
        true_fields (TensorDict): True fields TensorDict with tensors of shape (B, H, W) for each variable.
        pred_fields (TensorDict): Predicted fields TensorDict with tensors of shape (B, H, W) for each variable.
        output_dir (Path, optional): Directory to save the plot.
    """
    variables = true_fields.keys()  # Extract variable names from the TensorDict
    plt.figure(figsize=(12, 6))  # Adjusted size for better readability in W&B

    for var in variables:
        # Compute isotropic energy spectrum for each variable
        k_bins, true_spec = compute_isotropic_energy_spectrum(true_fields[var])
        _, pred_spec = compute_isotropic_energy_spectrum(pred_fields[var])

        # Plot the true and predicted spectra
        plt.loglog(k_bins, true_spec, label=f"True {var}", linestyle="--")
        plt.loglog(k_bins, pred_spec, label=f"Pred {var}")

    plt.xlabel("Wavenumber", fontsize=12)
    plt.ylabel("Energy Spectrum", fontsize=12)
    plt.title("Isotropic Energy Spectrum Comparison", fontsize=14)
    plt.legend(loc="best", fontsize=10, title="Variables")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / "energy_spectrum.png"
        plt.savefig(filename, dpi=150)  # High DPI for better quality in W&B

        # Log to W&B
        wandb.log({f"figures/{mode}/energy_spectrum": wandb.Image(str(filename))})

    plt.close()


def denormalize_and_visualize(
    input: TensorDict,
    target: TensorDict,
    x_recon: TensorDict,
    x_preds: TensorDict,
    output_dir: Path,
    mode: str,
):
    """
    Handles denormalization and visualization for the first batch.

    Args:
        input: Input TensorDict.
        target: Target TensorDict.
        x_recon: Reconstructed TensorDict.
        x_preds: Predicted TensorDict.
        output_dir: Output directory for saving plots.
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare and save reconstruction comparison plot
    true_to_recon = TensorDict(
        {
            key: value[:, -1].unsqueeze(1) for key, value in input.items()
        },  # Slice each tensor
        batch_size=input.batch_size,
    )
    est_to_recon = TensorDict(
        {
            key: value.unsqueeze(1) for key, value in x_recon.items()
        },  # Slice each tensor
        batch_size=input.batch_size,
    )
    plot_comparison(
        x=true_to_recon,
        x_recon=est_to_recon,
        output_dir=output_dir,
        title="Reconstruction",
        mode=mode,
    )

    # Prepare and save prediction comparison plot
    plot_comparison(
        x=target,
        x_recon=x_preds,
        output_dir=output_dir,
        title="Prediction",
        mode=mode,
    )

    # Prepare and save energy spectrum comparison plot
    true_fields = TensorDict(
        {
            key: value[:, -1] if value.dim() > 1 else None
            for key, value in target.items()
        },
        batch_size=target.batch_size,
    )
    pred_fields = TensorDict(
        {key: value[:, -1] for key, value in x_preds.items()},
        batch_size=x_preds.batch_size,
    )
    plot_energy_spectrum(
        true_fields=true_fields,
        pred_fields=pred_fields,
        output_dir=output_dir,
        mode=mode,
    )


def accumulate_losses(total_losses: dict, losses: dict):
    """
    Accumulates losses over batches.

    Args:
        total_losses: TensorDict to store accumulated losses.
        losses: Current batch losses as a TensorDict.

    Returns:
        Updated total_losses TensorDict.
    """
    for key, value in losses.items():
        if key not in total_losses:
            total_losses[key] = value
        else:
            total_losses[key] += value
    return total_losses


def average_losses(total_losses: dict, n_batches: int):
    """
    Averages the losses over the number of batches.

    Args:
        total_losses: TensorDict with accumulated losses.
        n_batches: Total number of batches.

    Returns:
        TensorDict with averaged losses.
    """
    for key in total_losses.keys():
        total_losses[key] /= n_batches
    return total_losses


def load_checkpoint(checkpoint_path: str, model: KoopmanAutoencoder, optimizer: Adam):
    """
    Loads training state from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    history = checkpoint.get("history", {})
    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"Resuming training from epoch {start_epoch}.")
    return model, optimizer, history, start_epoch
