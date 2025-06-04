import numpy as np
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
from tensordict import TensorDict
from torch import Tensor
import torch
import torch.distributed as dist


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def plot_comparison(
    x: TensorDict,
    x_recon: TensorDict,
    output_dir: Path | str = "/home/koopman/",
    title: str = "reconstruction_comparison",
    mode="train",
):
    """
    Plots input vs reconstructed images for each variable across all timesteps.
    Logs a separate figure per variable.
    Assumes input shape (B, T, H, W).
    """
    T = next(iter(x.values())).shape[1]  # Number of timesteps
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for var in x.keys():
        if var in ["seq_length", "Re"]:
            continue
        x_var = x[var][0]  # shape: (T, H, W)
        x_recon_var = x_recon[var][0]  # shape: (T, H, W)

        fig, axes = plt.subplots(
            T, 3, figsize=(12, 4 * T)
        )  # Rows = timesteps, cols = [True, Recon, Error]

        if T == 1:
            axes = np.expand_dims(axes, axis=0)  # Handle single timestep

        for t in range(T):
            true_img = x_var[t].cpu().numpy()
            recon_img = x_recon_var[t].cpu().numpy()
            error = true_img - recon_img

            ax = axes[t, 0]
            mat = ax.matshow(true_img, cmap="RdBu_r")
            ax.set_title(f"True {var}, t={t}")
            ax.axis("off")
            plt.colorbar(mat, ax=ax, fraction=0.046, pad=0.04).set_label("Intensity")

            ax = axes[t, 1]
            mat = ax.matshow(recon_img, cmap="RdBu_r")
            ax.set_title(f"Recon {var}, t={t}")
            ax.axis("off")
            plt.colorbar(mat, ax=ax, fraction=0.046, pad=0.04).set_label("Intensity")

            ax = axes[t, 2]
            mat = ax.matshow(error, cmap="hot")
            ax.set_title(f"Error {var}, t={t}")
            ax.axis("off")
            plt.colorbar(mat, ax=ax, fraction=0.046, pad=0.04).set_label("Intensity")

        plt.suptitle(f"{title} - {var}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(right=0.85, hspace=0.5)

        filename = output_dir / f"{title}_{var}.png"
        plt.savefig(filename, dpi=150)
        if is_main_process():
            wandb.log({f"figures/{mode}/{title}_{var}": wandb.Image(str(filename))})
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
        if var in ["seq_length", "Re"]:
            continue
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
        if is_main_process():
            wandb.log({f"figures/{mode}/energy_spectrum": wandb.Image(str(filename))})

    plt.close()


def compute_re(
    true_fields: TensorDict,
    pred_fields: TensorDict,
    output_dir: Path | str = "/home/koopman/",
    mode: str = "train",
    nu: float = 0.0003061224489795918,
    L: float = 0.6,
):
    """
    Compute Reynolds number and log in W&B
    """
    Re_logs = {}
    true_Re = true_fields["Re"].detach().cpu().numpy()
    v_x = pred_fields["v_x"]
    v_y = pred_fields["v_y"]
    v = torch.sqrt(v_x**2 + v_y**2).mean(dim=(1, 2, 3))
    pred_Re = v.detach().cpu().numpy() * L / nu

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        Re_logs[f"figures/{mode}/true_Re"] = true_Re[
            0
        ]  # indexed to ony show one sample per batch
        Re_logs[f"figures/{mode}/pred_Re"] = pred_Re[
            0
        ]  # indexed to ony show one sample per batch
        Re_logs[f"figures/{mode}/diff_Re"] = np.mean(true_Re - pred_Re)
        # Log to W&B
        if is_main_process():
            wandb.log(Re_logs)


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

    compute_re(
        true_fields=target,
        pred_fields=x_preds,
        output_dir=output_dir,
        mode=mode,
    )
