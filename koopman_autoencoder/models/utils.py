import numpy as np
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
import torch
from tensordict import TensorDict


def tensor_dict_to_json(tensor_dict):
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
    x, x_recon, output_dir=None, title="reconstruction_comparison", mode="train"
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
    """
    variables = x.keys()  # Get variable names from TensorDict
    num_channels = len(variables)  # Number of variables

    # Reuse existing figure (create once, clear later)
    fig, axes = plt.subplots(
        3, num_channels, figsize=(4 * num_channels, 16)
    )  # Adjusted size

    # Iterate through variables to plot input, reconstructed, and error images
    for j, var in enumerate(variables):
        # Clear axes for reuse
        axes[0, j].clear()
        axes[1, j].clear()
        axes[2, j].clear()

        # Extract corresponding tensors for the first sample in the batch
        x_var = x[var][0]  # Shape: (H, W)
        x_recon_var = x_recon[var][0]  # Shape: (H, W)
        error = torch.abs(x_var - x_recon_var)  # Compute absolute error

        # Plot input image
        mat = axes[0, j].matshow(x_var.cpu().numpy(), cmap="RdBu_r", aspect="auto")
        axes[0, j].set_title(f"True {var}", fontsize=12)
        axes[0, j].axis("off")
        plt.colorbar(mat, ax=axes[0, j], orientation="vertical")

        # Plot reconstructed image
        mat = axes[1, j].matshow(
            x_recon_var.cpu().numpy(), cmap="RdBu_r", aspect="auto"
        )
        axes[1, j].set_title(f"Reconstructed {var}", fontsize=12)
        axes[1, j].axis("off")
        plt.colorbar(mat, ax=axes[1, j], orientation="vertical")

        # Plot error image
        mat = axes[2, j].matshow(error.cpu().numpy(), cmap="hot", aspect="auto")
        axes[2, j].set_title(f"Error {var}", fontsize=12)
        axes[2, j].axis("off")
        plt.colorbar(mat, ax=axes[2, j], orientation="vertical")

    plt.tight_layout()

    # Save the plot if an output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{title}.png"
        plt.savefig(filename, dpi=150)  # High DPI for better quality in W&B

        # Log to W&B
        wandb.log({f"figures/{mode}/{title}": wandb.Image(str(filename))})

    # Clear the figure to reuse it
    plt.clf()
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


def plot_energy_spectrum(true_fields, pred_fields, output_dir=None, mode="train"):
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
    input, target, x_recon, x_preds_denormalized, output_dir, mode
):
    """
    Handles denormalization and visualization for the first batch.

    Args:
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
        title="reconstruction_comparison",
        mode=mode,
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
        title="prediction_comparison",
        mode=mode,
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
        mode=mode,
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


def load_checkpoint(checkpoint_path, model, optimizer):
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
