import numpy as np
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
import torch
from tensordict import TensorDict


def plot_comparison(
    x, x_recon, output_dir=None, title="reconstruction_comparison_epoch", epoch=None
):
    """
    Plots input vs reconstructed images for each variable in the TensorDict with colorbars,
    and logs to W&B. Additionally, displays the error (absolute difference) between
    input and reconstructed images.

    Parameters:
        x (TensorDict): Input TensorDict with tensors of shape (B, H, W) for each variable.
        x_recon (TensorDict): Reconstructed TensorDict with tensors of shape (B, H, W) for each variable.
        output_dir (Path, optional): Directory to save the plot.
        title (str, optional): Title prefix for the saved plot and W&B log.
        epoch (int, optional): Current epoch for logging.
    """
    variables = x.keys()  # Get variable names from TensorDict
    num_channels = len(variables)  # Number of variables
    fig, axes = plt.subplots(
        3, num_channels, figsize=(4 * num_channels, 12)
    )  # Adjusted size

    # Iterate through variables to plot input, reconstructed, and error images
    for j, var in enumerate(variables):
        # Extract corresponding tensors for the first sample in the batch
        x_var = x[var][0]  # Shape: (H, W)
        x_recon_var = x_recon[var][0]  # Shape: (H, W)
        error = torch.abs(x_var - x_recon_var)  # Compute absolute error

        # Plot input image
        mat = axes[0, j].matshow(x_var.cpu().numpy(), cmap="RdBu_r", aspect="auto")
        axes[0, j].set_title(f"True {var}", fontsize=12)
        axes[0, j].axis("off")
        plt.colorbar(mat, ax=axes[0, j])

        # Plot reconstructed image
        mat = axes[1, j].matshow(
            x_recon_var.cpu().numpy(), cmap="RdBu_r", aspect="auto"
        )
        axes[1, j].set_title(f"Reconstructed {var}", fontsize=12)
        axes[1, j].axis("off")
        plt.colorbar(mat, ax=axes[1, j])

        # Plot error image
        mat = axes[2, j].matshow(error.cpu().numpy(), cmap="hot", aspect="auto")
        axes[2, j].set_title(f"Error {var}", fontsize=12)
        axes[2, j].axis("off")
        plt.colorbar(mat, ax=axes[2, j])

    plt.tight_layout()

    # Save the plot if an output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{title}_{epoch}.png"
        plt.savefig(filename, dpi=150)  # High DPI for better quality in W&B

        # Log to W&B
        # wandb.log({f"{title} {epoch}": wandb.Image(str(filename))})

    # Close the plot to free up memory
    plt.close()


def compute_isotropic_energy_spectrum(field):
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


def plot_energy_spectrum(true_fields, pred_fields, output_dir=None, epoch=None):
    """
    Plots the isotropic energy spectrum comparison for each channel and logs to W&B.

    Parameters:
        true_fields (TensorDict): True fields TensorDict with tensors of shape (B, H, W) for each variable.
        pred_fields (TensorDict): Predicted fields TensorDict with tensors of shape (B, H, W) for each variable.
        output_dir (Path, optional): Directory to save the plot.
        epoch (int, optional): Current epoch for logging.
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
        filename = output_dir / f"energy_spectrum_comparison_epoch_{epoch}.png"
        plt.savefig(filename, dpi=150)  # High DPI for better quality in W&B

        # Log to W&B
        wandb.log(
            {f"energy_spectrum_comparison_epoch {epoch}": wandb.Image(str(filename))}
        )

    plt.close()


def denormalize_and_visualize(
    epoch, input, target, x_recon, x_preds_denormalized, output_dir
):
    """
    Handles denormalization and visualization for the first batch.

    Args:
        epoch: Current epoch.
        input: Input TensorDict.
        target: Target TensorDict.
        x_recon: Reconstructed TensorDict.
        x_preds_denormalized: Predicted TensorDict (denormalized).
        output_dir: Output directory for saving plots.
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare and save reconstruction comparison plot
    plot_comparison(
        TensorDict(
            {key: value[:, -1] for key, value in input.items()},  # Slice each tensor
            batch_size=input.batch_size,
        ),
        x_recon,
        output_dir,
        title="reconstruction_comparison_epoch",
        epoch=epoch,
    )

    # Prepare and save prediction comparison plot
    plot_comparison(
        TensorDict(
            {
                key: value[:, -1] if value.dim() > 1 else None
                for key, value in target.items()
            },
            batch_size=target.batch_size,
        ),
        TensorDict(
            {key: value[:, -1] for key, value in x_preds_denormalized.items()},
            batch_size=x_preds_denormalized.batch_size,
        ),
        output_dir,
        title="prediction_comparison_epoch",
        epoch=epoch,
    )

    # Prepare and save energy spectrum comparison plot
    plot_energy_spectrum(
        TensorDict(
            {
                key: value[:, -1] if value.dim() > 1 else None
                for key, value in target.items()
            },
            batch_size=target.batch_size,
        ),
        TensorDict(
            {key: value[:, -1] for key, value in x_preds_denormalized.items()},
            batch_size=x_preds_denormalized.batch_size,
        ),
        output_dir,
        epoch=epoch,
    )

    # Log plots to W&B
    wandb.log(
        {
            f"Reconstruction Comparison (Epoch {epoch})": wandb.Image(
                str(output_dir / f"reconstruction_comparison_epoch_{epoch}.png")
            ),
            f"Prediction Comparison (Epoch {epoch})": wandb.Image(
                str(output_dir / f"prediction_comparison_epoch_{epoch}.png")
            ),
            f"Energy Spectrum (Epoch {epoch})": wandb.Image(
                str(output_dir / f"energy_spectrum_comparison_epoch_{epoch}.png")
            ),
        }
    )


def accumulate_losses(total_losses, losses):
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


def average_losses(total_losses, n_batches):
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
