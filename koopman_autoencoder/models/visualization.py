import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_results(model, dataset, num_samples=4, device="cpu", output_dir=None):
    """
    Visualizes model outputs, including reconstruction, prediction, error maps, and energy spectrum.

    Args:
        model: Trained PyTorch model.
        dataset: Dataset object with `denormalize` method.
        num_samples: Number of samples to visualize.
        device: Device to run the model on ('cpu' or 'cuda').
        output_dir: Directory to save the visualizations.
    """
    model.eval()
    with torch.no_grad():
        # Load a batch of data
        x, x_next_seq = next(
            iter(DataLoader(dataset, batch_size=num_samples, shuffle=True))
        )
        x, x_next_seq = x.to(device), x_next_seq.to(device)

        # Forward pass through the model
        x_recon, x_preds, _, _ = model(x, seq_length=x_next_seq.size(1))

        # Denormalize data for visualization
        x = dataset.denormalize(x)
        x_next_seq = dataset.denormalize(x_next_seq)
        x_recon = dataset.denormalize(x_recon)
        x_preds = torch.stack([dataset.denormalize(pred) for pred in x_preds], dim=1)

        # Ensure output directory exists
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

        # Reconstruction visualization
        plot_reconstruction_comparison(x, x_recon, num_samples, output_dir)

        # Prediction visualization
        plot_prediction_comparison(
            x_next_seq, x_preds[:, :, 0], num_samples, output_dir
        )

        # Error visualization
        plot_error_maps(
            x, x_recon, x_next_seq[:, 0], x_preds[:, 0], num_samples, output_dir
        )

        # Energy spectrum visualization
        plot_energy_spectrum(x_next_seq[:, 0], x_preds[:, 0], output_dir)


def plot_reconstruction_comparison(x, x_recon, num_samples, output_dir):
    """
    Plots input vs reconstructed images.
    """
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    for i in range(num_samples):
        for j, (title, data) in enumerate(
            [("Input", x[i, 0]), ("Reconstructed", x_recon[i, 0])]
        ):
            axes[i, j].imshow(data.cpu().numpy(), cmap="RdBu_r")
            axes[i, j].set_title(title)
            axes[i, j].axis("off")
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "reconstruction_comparison.png")
    plt.close()


def plot_prediction_comparison(x_next_seq, x_pred, num_samples, output_dir):
    """
    Plots true next state vs predicted next state.
    """
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    for i in range(num_samples):
        for j, (title, data) in enumerate(
            [("True Next", x_next_seq[i, 0, 0]), ("Predicted Next", x_pred[i, 0])]
        ):
            axes[i, j].imshow(data.cpu().numpy(), cmap="RdBu_r")
            axes[i, j].set_title(title)
            axes[i, j].axis("off")
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "prediction_comparison.png")
    plt.close()


def plot_error_maps(x, x_recon, x_next_seq, x_pred, num_samples, output_dir):
    """
    Plots error maps for reconstruction and prediction.
    """
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    for i in range(num_samples):
        recon_error = (x_recon[i, 0] - x[i, 0]).cpu().numpy()
        pred_error = (x_pred[i, 0] - x_next_seq[i, 0]).cpu().numpy()

        axes[i, 0].imshow(recon_error, cmap="RdBu_r")
        axes[i, 0].set_title("Reconstruction Error")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_error, cmap="RdBu_r")
        axes[i, 1].set_title("Prediction Error")
        axes[i, 1].axis("off")
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "error_maps.png")
    plt.close()


def plot_energy_spectrum(true_fields, pred_fields, output_dir):
    """
    Plots the isotropic energy spectrum comparison.
    """
    k_bins, true_spec = compute_isotropic_energy_spectrum(true_fields)
    _, pred_spec = compute_isotropic_energy_spectrum(pred_fields)

    plt.figure(figsize=(8, 6))
    for c in range(true_spec.shape[0]):
        plt.loglog(k_bins, true_spec[c], label=f"True Layer {c + 1}", linestyle="--")
        plt.loglog(k_bins, pred_spec[c], label=f"Pred Layer {c + 1}")
    plt.xlabel("Wavenumber")
    plt.ylabel("Energy Spectrum")
    plt.title("Isotropic Energy Spectrum Comparison")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "energy_spectrum_comparison.png")
    plt.close()


def compute_isotropic_energy_spectrum(field):
    """
    Compute isotropic energy spectrum from a 2D velocity or vorticity field.

    Parameters:
        field (torch.Tensor): (B, C, H, W) tensor with batch of spatial fields

    Returns:
        k_bins (np.ndarray): Radial wavenumber bins
        E_k (np.ndarray): Isotropic energy spectrum
    """
    B, C, H, W = field.shape
    spectrum = np.zeros((C, H // 2))

    for b in range(B):
        for c in range(C):
            # 2D FFT and power spectrum
            f_hat = np.fft.fft2(field[b, c].cpu().numpy())
            psd2d = np.abs(f_hat) ** 2

            # Generate wavenumber grid
            kx = np.fft.fftfreq(W).reshape(1, -1)
            ky = np.fft.fftfreq(H).reshape(-1, 1)
            k_mag = np.sqrt(kx**2 + ky**2)
            k_mag = np.fft.fftshift(k_mag)

            psd2d = np.fft.fftshift(psd2d)
            k_flat = k_mag.ravel()
            psd_flat = psd2d.ravel()

            # Bin the power spectrum isotropically
            k_bins = np.linspace(0, 0.5, H // 2)
            bin_indices = np.digitize(k_flat, k_bins)
            for i in range(1, len(k_bins)):
                spectrum[c, i - 1] += psd_flat[bin_indices == i].mean()

    k_bins = 0.5 * (k_bins[:-1] + k_bins[1:])
    return k_bins, spectrum
