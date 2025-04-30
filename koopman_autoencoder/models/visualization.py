import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_results(model, dataset, num_samples=4, device='cpu', output_dir=None):
    model.eval()
    with torch.no_grad():
        x, x_next_seq = next(iter(DataLoader(dataset, batch_size=num_samples, shuffle=True)))
        x, x_next_seq = x.to(device), x_next_seq.to(device)

        # Updated unpacking to handle additional return value
        x_recon, x_preds, _, _ = model(x, rollout_steps=x_next_seq.size(1))

        x = dataset.denormalize(x)
        x_next_seq = dataset.denormalize(x_next_seq)
        x_recon = dataset.denormalize(x_recon)
        x_preds = [dataset.denormalize(pred) for pred in x_preds]

        # Only visualize reconstruction and first prediction
        x_pred = x_preds[0]

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

        # Reconstruction plot
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
        for i in range(num_samples):
            for j, (title, data) in enumerate([
                ('Input', x[i, 0]),
                ('Reconstructed', x_recon[i, 0])
            ]):
                axes[i, j].imshow(data.cpu().numpy(), cmap='RdBu_r')
                axes[i, j].set_title(title)
                axes[i, j].axis('off')
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / 'reconstruction_comparison.png')
        plt.close()

        # Prediction plot
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
        for i in range(num_samples):
            for j, (title, data) in enumerate([
                ('True Next', x_next_seq[i, 0, 0]),
                ('Predicted Next', x_pred[i, 0])
            ]):
                axes[i, j].imshow(data.cpu().numpy(), cmap='RdBu_r')
                axes[i, j].set_title(title)
                axes[i, j].axis('off')
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / 'prediction_comparison.png')
        plt.close()

        # Error maps
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
        for i in range(num_samples):
            recon_error = (x_recon[i, 0] - x[i, 0]).cpu().numpy()
            pred_error = (x_pred[i, 0] - x_next_seq[i, 0, 0]).cpu().numpy()

            axes[i, 0].imshow(recon_error, cmap='RdBu_r')
            axes[i, 0].set_title('Reconstruction Error')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(pred_error, cmap='RdBu_r')
            axes[i, 1].set_title('Prediction Error')
            axes[i, 1].axis('off')
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / 'error_maps.png')
        plt.close()

        k_bins, true_spec = compute_isotropic_energy_spectrum(x_next_seq)
        _, pred_spec = compute_isotropic_energy_spectrum(x_pred)

        plt.figure(figsize=(8, 6))
        for c in range(true_spec.shape[0]):
            plt.loglog(k_bins, true_spec[c], label=f'True Layer {c+1}', linestyle='--')
            plt.loglog(k_bins, pred_spec[c], label=f'Pred Layer {c+1}')
        plt.xlabel('Wavenumber')
        plt.ylabel('Energy Spectrum')
        plt.title('Isotropic Energy Spectrum Comparison')
        plt.legend()
        plt.grid(True, which='both', ls=':')
        plt.tight_layout()
        plt.savefig(output_dir / 'energy_spectrum_comparison.png')
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
            k_mag = np.sqrt(kx ** 2 + ky ** 2)
            k_mag = np.fft.fftshift(k_mag)

            psd2d = np.fft.fftshift(psd2d)
            k_flat = k_mag.ravel()
            psd_flat = psd2d.ravel()

            # Bin the power spectrum isotropically
            k_bins = np.linspace(0, 0.5, H // 2)
            bin_indices = np.digitize(k_flat, k_bins)
            for i in range(1, len(k_bins)):
                spectrum[c, i-1] += psd_flat[bin_indices == i].mean()

    k_bins = 0.5 * (k_bins[:-1] + k_bins[1:])
    return k_bins, spectrum