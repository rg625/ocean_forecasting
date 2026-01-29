# models/utils.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import wandb
from pathlib import Path
from tensordict import TensorDict
from torch import Tensor
import torch
import torch.distributed as dist
import seaborn as sns
from typing import List, Optional, Dict, Union
import pandas as pd

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from .metrics_utils import compute_vorticity

plt.style.use("seaborn-v0_8-whitegrid")

cmap = sns.color_palette("icefire", as_cmap=True)


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
        if var in ["seq_length", "cond_target", "cond_input", "obstacle_mask"]:
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
            mat = ax.matshow(true_img, cmap=cmap)
            ax.set_title(f"True {var}, t={t}")
            ax.axis("off")
            plt.colorbar(mat, ax=ax, fraction=0.046, pad=0.04).set_label("Intensity")

            ax = axes[t, 1]
            mat = ax.matshow(recon_img, cmap=cmap)
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
        if var in ["seq_length", "cond_target", "cond_input", "obstacle_mask"]:
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
    true_cond = true_fields["cond_target"].detach().cpu().numpy()
    v_x = pred_fields["v_x"]
    v_y = pred_fields["v_y"]
    v = torch.sqrt(v_x**2 + v_y**2).mean(dim=(1, 2, 3))
    pred_cond = v.detach().cpu().numpy() * L / nu

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        Re_logs[f"figures/{mode}/true_Re"] = true_cond[
            0
        ]  # indexed to ony show one sample per batch
        Re_logs[f"figures/{mode}/pred_Re"] = pred_cond[
            0
        ]  # indexed to ony show one sample per batch
        Re_logs[f"figures/{mode}/diff_Re"] = np.mean(true_cond[:, 0] - pred_cond)
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
        batch_size=input.batch_size[0],
    )
    est_to_recon = TensorDict(
        {
            key: value.unsqueeze(1) for key, value in x_recon.items()
        },  # Slice each tensor
        batch_size=input.batch_size[0],
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
        batch_size=target.batch_size[0],
    )
    pred_fields = TensorDict(
        {key: value[:, -1] for key, value in x_preds.items()},
        batch_size=x_preds.batch_size[0],
    )
    plot_energy_spectrum(
        true_fields=true_fields,
        pred_fields=pred_fields,
        output_dir=output_dir,
        mode=mode,
    )

    # compute_re(
    #     true_fields=target,
    #     pred_fields=x_preds,
    #     output_dir=output_dir,
    #     mode=mode,
    # )


def plot_joint_rollout(
    gt_dict: TensorDict,
    pred_dict: TensorDict,
    diff_pred_dict: TensorDict,
    variable_name: str,
    frames: Optional[List[int]] = None,
    frame_stride: int = 5,
    re: Union[int, float, torch.Tensor] = 1000,
    pad: Optional[List] = [0.012, 0.018, 0.015, 0.025],
    highlight: bool = False,  # Set to False by default; toggle box here
    model_name: str = "Exp",
    show_intermodel_diff: bool = False,
    cmap=cmap,
):
    """
    Versatile and robust rollout plotter.
    - Handles temporal mismatches between ground truth and predictions.
    - Uses axes_grid1 for consistent colorbar sizing.
    - Prevents 'tight_layout' warnings with manual spacing control.
    """

    # 1. Standardize and remove batch dimension
    def sanitize(td):
        if getattr(td, "batch_dims", 0) == 1 and td.batch_size[0] == 1:
            return td.squeeze(0)
        return td

    gt_dict = sanitize(gt_dict)
    pred_dict = sanitize(pred_dict)
    diff_pred_dict = sanitize(diff_pred_dict)

    gt = gt_dict[variable_name]
    pred = pred_dict[variable_name]
    diff_pred = diff_pred_dict[variable_name]

    # Extract Reynolds number value
    re_val = re.item() if isinstance(re, torch.Tensor) else re

    # 2. Build indices and align lengths
    # We find the smallest sequence length to avoid indexing errors
    min_frames = min(gt.shape[0], pred.shape[0], diff_pred.shape[0])

    if frames is not None:
        indices = [idx for idx in frames if idx < min_frames]
    else:
        indices = [0] + list(range(1, min_frames, frame_stride))

    num_plots = min(len(indices), 15)
    indices = indices[:num_plots]

    # 3. Setup Grid
    num_rows = 6 if show_intermodel_diff else 5
    # Increased width (2.5) and height (2.2) per plot for better spacing
    fig = plt.figure(figsize=(2.5 * num_plots, 2.2 * num_rows))
    spec = gridspec.GridSpec(num_rows, num_plots)

    # Store axes for the highlight rectangle
    columns_axes: List = [[] for _ in range(num_plots)]

    for i, idx in enumerate(indices):
        t_label = "t=0" if idx == 0 else f"t={idx}"

        # Logic to safely extract frames
        f_gt = gt[idx].cpu()
        f_pred = pred[idx].cpu()
        f_diff = diff_pred[idx].cpu()

        # Define data rows: (data, label)
        row_data = [
            (f_gt, f"Ground Truth\nRe = {re_val:.1f}"),
            (f_pred, "KAE Prediction"),
            (f_diff, f"{model_name} Prediction"),
            (f_gt - f_pred, "KAE Error"),
            (f_gt - f_diff, f"{model_name} Error"),
        ]
        if show_intermodel_diff:
            row_data.append((f_pred - f_diff, f"KAE vs {model_name}\n(Inter-Model)"))

        for row_idx, (frame_data, label_text) in enumerate(row_data):
            ax = fig.add_subplot(spec[row_idx, i])
            im = ax.imshow(frame_data, cmap=cmap)
            ax.axis("off")

            # Force colorbar to match image height exactly
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=8)

            # Store for highlighting
            columns_axes[i].extend([ax, cax])

            if row_idx == 0:
                ax.set_title(t_label, fontsize=12, fontweight="bold")

            # Row labels on the far left column
            if i == 0:
                ax.text(
                    -0.35,
                    0.5,
                    label_text,
                    fontsize=11,
                    fontweight="bold",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                )

    # 4. Refined Layout Management
    # Manual adjustment avoids the 'Axes too small' UserWarning from tight_layout
    plt.subplots_adjust(
        left=0.15, right=0.95, top=0.92, bottom=0.05, wspace=0.45, hspace=0.35
    )
    fig.suptitle(
        f"Analytical vs Numerical Koopman Rollout: {variable_name}",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    # 5. Conditional Highlight Logic
    if highlight and pad is not None:
        fig.canvas.draw()
        pad_left, pad_right, pad_bottom, pad_top = pad

        # Calculate the bounding box for the first column
        bboxes = [ax.get_position() for ax in columns_axes[0]]
        x0 = min(b.x0 for b in bboxes) - pad_left
        x1 = max(b.x1 for b in bboxes) + pad_right
        y0 = min(b.y0 for b in bboxes) - pad_bottom
        y1 = max(b.y1 for b in bboxes) + pad_top

        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=3,
            edgecolor="red",
            facecolor="none",
            transform=fig.transFigure,
            clip_on=False,
            zorder=10,
        )
        fig.add_artist(rect)

    plt.show()


def plot_rollout(gt_dict, pred_dict, variable_name, frame_stride=5):
    """
    Plots the rollout comparison between ground truth and predicted sequence for a given variable.

    Args:
        gt_dict (TensorDict): TensorDict of ground truth tensors [T, H, W] per variable.
        pred_dict (TensorDict): TensorDict of predicted tensors [T, H, W] per variable.
        variable_name (str): The key for the variable to visualize.
        frame_stride (int): Step between frames to plot.
    """
    # Remove batch dimension if present
    if gt_dict.batch_dims == 1 and gt_dict.batch_size[0] == 1:
        gt_dict = gt_dict.squeeze(0)
    if pred_dict.batch_dims == 1 and pred_dict.batch_size[0] == 1:
        pred_dict = pred_dict.squeeze(0)

    gt = gt_dict[variable_name]  # shape: [T, H, W]
    pred = pred_dict[variable_name]  # shape: [T, H, W]

    print(f"Stats for variable: {variable_name}")
    print(
        f"GT   | Min: {gt.min():.4f}, Max: {gt.max():.4f}, Mean: {gt.mean():.4f}, Std: {gt.std():.4f}"
    )
    print(
        f"Pred | Min: {pred.min():.4f}, Max: {pred.max():.4f}, Mean: {pred.mean():.4f}, Std: {pred.std():.4f}"
    )

    num_frames = min(gt.shape[0], pred.shape[0])
    indices = list(range(0, num_frames, frame_stride))
    num_plots = min(len(indices), 15)

    fig = plt.figure(figsize=(1.8 * num_plots, 7.5))
    spec = gridspec.GridSpec(3, num_plots + 1, width_ratios=[1] * num_plots + [0.05])

    for i, idx in enumerate(indices[:num_plots]):
        # ----- Ground Truth -----
        ax_gt = fig.add_subplot(spec[0, i])
        im_gt = ax_gt.imshow(gt[idx].cpu(), cmap=cmap)
        ax_gt.axis("off")
        plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
        ax_gt.set_title(f"t={idx}", fontsize=10)

        # Add row label only above the first frame
        if i == 0:
            ax_gt.text(
                -0.2,
                1.5,
                "Ground Truth",
                fontsize=12,
                fontweight="bold",
                transform=ax_gt.transAxes,
                ha="left",
                va="bottom",
            )

        # ----- Prediction -----
        ax_pred = fig.add_subplot(spec[1, i])
        im_pred = ax_pred.imshow(pred[idx].cpu(), cmap=cmap)
        ax_pred.axis("off")
        plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

        if i == 0:
            ax_pred.text(
                -0.2,
                1.1,
                "Prediction",
                fontsize=12,
                fontweight="bold",
                transform=ax_pred.transAxes,
                ha="left",
                va="bottom",
            )

        # ----- Error -----
        err = gt[idx] - pred[idx]
        ax_err = fig.add_subplot(spec[2, i])
        im_err = ax_err.imshow(err.cpu(), cmap=cmap)
        ax_err.axis("off")
        plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)

        if i == 0:
            ax_err.text(
                -0.2,
                1.1,
                "Error",
                fontsize=12,
                fontweight="bold",
                transform=ax_err.transAxes,
                ha="left",
                va="bottom",
            )

    fig.suptitle(f"Koopman AE Rollout for Variable: {variable_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.4, 0.98, 0.95])
    plt.show()


def plot_model_rollouts(
    models,
    gt_dict,
    variable_name: str,
    ic: torch.Tensor,
    rollout_steps: int,
    ds: torch.utils.data.Dataset,
    frame_stride: int = 5,
    max_frames: int = 15,
    re: int = 1000,
    cmap="viridis",
    err_cmap="coolwarm",
    device="cuda",
):
    """
    Runs multiple models (with their own rollout functions) and plots GT vs predictions + errors.
    Horizontal layout: frames evolve left → right, model names on the left, timestamp above each frame.

    Args:
        models (dict): mapping name -> {"model": nn.Module, "rollout_fn": callable}.
                       rollout_fn(model, ic, steps) -> TensorDict with variables.
        gt_dict (TensorDict): ground truth dict with [T,H,W] tensors per variable.
        variable_name (str): variable to visualize.
        ic (torch.Tensor): initial condition for rollout.
        rollout_steps (int): number of rollout steps.
        frame_stride (int): stride between frames to display.
        max_frames (int): maximum number of frames to show.
        re (int, optional): Reynolds number.
        cmap (str): colormap for fields.
        err_cmap (str): colormap for errors.
        device (str): "cuda" or "cpu".
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    ic = ic.to(device)
    gt = (
        gt_dict.get(variable_name)
        if hasattr(gt_dict, "get")
        else gt_dict[variable_name]
    )
    if gt.ndim == 4 and gt.shape[0] == 1:
        gt = gt.squeeze(0)
    gt = gt[:rollout_steps].cpu()

    # Run models and store predictions
    preds = {}
    for name, entry in models.items():
        model = entry["model"].to(device).eval()
        rollout_fn = entry["rollout_fn"]

        with torch.no_grad():
            td_out = rollout_fn(model, ic)
            td_out = ds.denormalize(td_out) if "acdm" not in name.lower() else td_out
            if variable_name == "vort":
                td_out["vort"] = compute_vorticity(td_out["v_x"], td_out["v_y"])
            out = (
                td_out.get(variable_name)
                if hasattr(td_out, "get")
                else td_out[variable_name]
            )
            if out.ndim == 4 and out.shape[0] == 1:
                out = out.squeeze(0)
            preds[name] = out.cpu()

    # Determine frames to plot
    num_frames = min(gt.shape[0], min(p.shape[0] for p in preds.values()))
    indices = list(range(0, num_frames, frame_stride))[:max_frames]
    num_cols = len(indices)
    num_preds = len(preds)
    num_rows = 1 + 2 * num_preds  # GT + all preds + all errors
    fig_height = 2.0 * num_rows
    fig_width = 2.6 * num_cols
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    spec = gridspec.GridSpec(num_rows, num_cols, hspace=0.6, wspace=0.6)

    # Top row: Ground truth
    for j, idx in enumerate(indices):
        ax = fig.add_subplot(spec[0, j])
        im = ax.imshow(gt[idx], cmap=cmap)
        ax.axis("off")
        if j == 0:
            ax.text(
                -0.3,
                0.5,
                "Ground Truth",
                fontsize=12,
                fontweight="bold",
                transform=ax.transAxes,
                ha="right",
                va="center",
            )
        ax.set_title(f"t={idx}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Predictions (grouped together)
    for i, (name, pred) in enumerate(preds.items()):
        for j, idx in enumerate(indices):
            axp = fig.add_subplot(spec[1 + i, j])  # predictions start at row 1
            imp = axp.imshow(pred[idx], cmap=cmap)
            axp.axis("off")
            if j == 0:
                axp.text(
                    -0.3,
                    0.5,
                    f"{name} Prediction",
                    fontsize=12,
                    fontweight="bold",
                    transform=axp.transAxes,
                    ha="right",
                    va="center",
                )
            plt.colorbar(imp, ax=axp, fraction=0.046, pad=0.04)

    # Errors (grouped after predictions)
    for i, (name, pred) in enumerate(preds.items()):
        err = gt[: pred.shape[0]] - pred
        for j, idx in enumerate(indices):
            axe = fig.add_subplot(
                spec[1 + num_preds + i, j]
            )  # errors start after predictions
            ime = axe.imshow(err[idx], cmap=err_cmap)
            axe.axis("off")
            if j == 0:
                axe.text(
                    -0.3,
                    0.5,
                    f"{name} Error",
                    fontsize=12,
                    fontweight="bold",
                    transform=axe.transAxes,
                    ha="right",
                    va="center",
                )
            plt.colorbar(ime, ax=axe, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Rollout Comparison for '{variable_name.upper()}', Re={re}",
        fontsize=16,
        y=0.98,
    )
    # plt.tight_layout(rect=[0.05, 0, 0.95, 0.92])  # leave space for suptitle
    plt.show()


def plot_stability_metrics(metrics: Dict, ttd_threshold: float = 0.1):
    time_steps = len(metrics["slope_error_pred"])
    time = np.arange(time_steps)

    plt.figure(figsize=(12, 5))

    # Slope Error
    plt.subplot(1, 2, 1)
    plt.plot(time, metrics["slope_error_pred"], label="Pred Error")
    if metrics["slope_error_diff"] is not None:
        plt.plot(time, metrics["slope_error_diff"], label="Diff Error")
    if ttd_threshold is not None:
        plt.axhline(ttd_threshold, color="red", linestyle="--", label="TTD Threshold")
    plt.axvline(
        metrics["ttd_pred"],
        color="orange",
        linestyle=":",
        label=f'TTD Pred = {metrics["ttd_pred"]}',
    )
    if metrics["ttd_diff"] is not None:
        plt.axvline(
            metrics["ttd_diff"],
            color="green",
            linestyle=":",
            label=f'TTD Diff = {metrics["ttd_diff"]}',
        )
    plt.xlabel("Timestep")
    plt.ylabel("L2 Error")
    plt.title("Slope Error Over Time")
    plt.legend()
    plt.grid(True)

    # Energy Drift
    plt.subplot(1, 2, 2)
    plt.plot(time, metrics["energy_drift_pred"], label="Pred Energy Drift")
    if metrics["energy_drift_diff"] is not None:
        plt.plot(time, metrics["energy_drift_diff"], label="Diff Energy Drift")
    plt.axvline(
        metrics["ttd_pred"],
        color="orange",
        linestyle=":",
        label=f'TTD Pred = {metrics["ttd_pred"]}',
    )
    if metrics["ttd_diff"] is not None:
        plt.axvline(
            metrics["ttd_diff"],
            color="green",
            linestyle=":",
            label=f'TTD Diff = {metrics["ttd_diff"]}',
        )
    plt.xlabel("Timestep")
    plt.ylabel("Energy Drift")
    plt.title("Energy Drift Over Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def save_figure(fig, path: Path, dpi: int = 200, tight: bool = True):
    """
    Save a matplotlib figure safely:
    - creates parent directories if missing
    - applies tight layout
    - prints saved path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def run_full_analysis(
    gt_path: str,
    kae_path: str,
    acdm_path: str,
    output_folder: str,
    obstacle_mask: Optional[np.ndarray] = None,
    sequence_idx: int = 0,
    time_idx: int = -1,
    channel_idx: int = 0,
    model_label: str = "model",
    dataset_label: str = "dataset",
    regime: str = "tra",
    save_metadata: bool = True,
) -> Dict:
    """
    Unified workflow: load GT/KAE/ACDM, apply optional obstacle mask, compute MSE,
    and generate plots including MSE vs Mach, Temporal MSE, and Distributions.
    """

    sns.set_context("talk")
    sns.set_style("whitegrid")

    fields = ["v_x", "v_y", "p", "rho"] if regime == "tra" else ["v_x", "v_y", "p"]
    cond = "Mach" if regime == "tra" else "Re"

    # --- Setup directories ---
    BASE_RESULTS = Path(output_folder)
    RESULTS_DIR = BASE_RESULTS / dataset_label / model_label
    DIR_MACH = RESULTS_DIR / f"mse_vs_{cond}"
    DIR_FIELD = RESULTS_DIR / "mse_vs_field"
    DIR_TEMPORAL = RESULTS_DIR / "temporal_mse"
    DIR_DIST = RESULTS_DIR / "mse_distributions"

    for d in [DIR_MACH, DIR_FIELD, DIR_TEMPORAL, DIR_DIST]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    gt_sample = torch.load(gt_path)
    gt_data = gt_sample["data"].cpu().numpy()  # (N_seq, T, C, H, W)
    kae_data = np.load(kae_path)["arr_0"]
    acdm_data = np.load(acdm_path)["arr_0"]

    print("Loaded shapes:", gt_data.shape, kae_data.shape, acdm_data.shape)
    N_seq, T, C, H, W = gt_data.shape

    # --- Prepare mask ---
    if obstacle_mask is None:
        obstacle_mask = np.ones((H, W))

    def apply_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        data_masked = np.copy(data)
        mask_bc = mask
        for _ in range(data_masked.ndim - 2):
            mask_bc = mask_bc[None, ...]
        data_masked[..., :-1, :, :] *= mask_bc
        return data_masked

    # --- Mask all arrays ---
    gt_masked_all = apply_mask(gt_data[None, None, ...], obstacle_mask)
    kae_masked_all = apply_mask(kae_data, obstacle_mask)
    acdm_masked_all = apply_mask(acdm_data, obstacle_mask)

    # --- Compute per-sample, per-field, per-time MSE (Shape: N_seq, T, C-1) ---
    kae_mse = np.mean(
        (kae_masked_all[0, 0, ..., :-1, :, :] - gt_masked_all[0, 0, ..., :-1, :, :])
        ** 2,
        axis=(-2, -1),
    )
    acdm_mse = np.mean(
        (acdm_masked_all[0, 0, ..., :-1, :, :] - gt_masked_all[0, 0, ..., :-1, :, :])
        ** 2,
        axis=(-2, -1),
    )

    # --- Extract Mach numbers ---
    Mach_numbers = np.round(kae_data[0, 0, :, 0, -1, 0, 0].astype(float), 2)

    # --- 1. Visual Comparison + Error Field ---
    gt_vx = gt_data[sequence_idx, time_idx, channel_idx, :, :]
    kae_vx = kae_masked_all[0, 0, sequence_idx, time_idx, channel_idx, :, :]
    acdm_vx = acdm_masked_all[0, 0, sequence_idx, time_idx, channel_idx, :, :]
    error_kae = kae_vx - gt_vx
    error_acdm = acdm_vx - gt_vx
    vmax = max(np.abs(error_kae).max(), np.abs(error_acdm).max())
    vmin = -vmax

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[0, 0].imshow(gt_vx.T, origin="lower")
    axes[0, 0].set_title("GT Field")
    axes[0, 1].imshow(kae_vx.T, origin="lower")
    axes[0, 1].set_title("KAE Prediction")
    axes[0, 2].imshow(acdm_vx.T, origin="lower")
    axes[0, 2].set_title("ACDM Prediction")

    _ = axes[1, 0].imshow(
        error_kae.T, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax
    )
    axes[1, 0].set_title("KAE Error")
    im2 = axes[1, 1].imshow(
        error_acdm.T, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax
    )
    axes[1, 1].set_title("ACDM Error")
    axes[1, 2].axis("off")
    fig.colorbar(im2, ax=axes[1, :], shrink=0.6, label="Error")
    save_figure(fig, DIR_FIELD / "comparison_field_error.png")

    # --- 2. MSE vs Mach (Bar + Facet Line) ---
    df_list = []
    for model_name, mse_arr in zip(["KAE", "ACDM"], [kae_mse, acdm_mse]):
        mse_per_sample_field = mse_arr.mean(axis=1)  # (N_seq, C)
        for i in range(N_seq):
            for j, field in enumerate(fields):
                df_list.append(
                    {
                        "Model": model_name,
                        cond: Mach_numbers[i],
                        "Field": field,
                        "MSE": mse_per_sample_field[i, j],
                    }
                )
    df_mach = pd.DataFrame(df_list)
    df_mach.to_csv(DIR_MACH / f"mse_vs_{cond}.csv", index=False)

    fig = plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_mach, x=cond, y="MSE", hue="Model", errorbar="sd", palette="Set2"
    )
    plt.yscale("log")
    plt.title(f"Field-averaged MSE vs {cond}")
    save_figure(fig, DIR_MACH / f"bar_mse_vs_{cond}.png")

    g = sns.FacetGrid(
        df_mach, col="Field", hue="Model", sharey=False, height=4, aspect=1.2
    )
    g.map(sns.lineplot, cond, "MSE", marker="o")
    g.add_legend()
    for ax in g.axes.flat:
        ax.set_yscale("log")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(f"MSE vs {cond} Number per Field")
    g.savefig(DIR_MACH / f"fieldwise_mse_vs_{cond}.png", dpi=200)
    plt.close()

    # --- 3. Temporal MSE ---
    kae_time_field = kae_mse.mean(axis=0)  # (T, C)
    acdm_time_field = acdm_mse.mean(axis=0)
    fig = plt.figure(figsize=(10, 6))
    for j, field in enumerate(fields):
        plt.plot(np.arange(T)[2:], kae_time_field[2:, j], label=f"KAE {field}")
        plt.plot(np.arange(T)[2:], acdm_time_field[2:, j], "--", label=f"ACDM {field}")
    plt.yscale("log")
    plt.xlabel("Timestep")
    plt.ylabel("Mean MSE")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Temporal Evolution of MSE")
    save_figure(fig, DIR_TEMPORAL / "temporal_mse.png")

    # --- 4. Distribution of per-sample MSE (Violin) ---
    # Flattening into (N_seq * T * C) long-form
    df_dist_list = []
    for model_name, mse_arr in zip(["KAE", "ACDM"], [kae_mse, acdm_mse]):
        for i in range(N_seq):
            for t in range(T):
                for j, field in enumerate(fields):
                    df_dist_list.append(
                        {"Model": model_name, "Field": field, "MSE": mse_arr[i, t, j]}
                    )

    df_dist = pd.DataFrame(df_dist_list)
    df_dist.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_dist.dropna(subset=["MSE"], inplace=True)
    df_dist["log_MSE"] = np.log10(df_dist["MSE"].clip(lower=1e-4))
    df_dist.to_csv(DIR_DIST / "mse_distribution.csv", index=False)

    fig = plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df_dist,
        x="Field",
        y="log_MSE",
        hue="Model",
        inner="quartile",
        cut=2,
        bw_adjust=0.5,
        palette="Set2",
    )
    plt.ylabel("log10(MSE)")
    plt.title("MSE Distribution across all Timesteps/Simulations")
    save_figure(fig, DIR_DIST / "violin_mse_distribution.png")

    # --- Metadata ---
    if save_metadata:
        metadata = {
            "MODEL": model_label,
            "REGIME": regime,
            "DATASET": dataset_label,
            "N_seq": N_seq,
            "T": T,
            "Fields": ", ".join(fields),
        }
        pd.Series(metadata).to_csv(RESULTS_DIR / "metadata.csv")
        print(f"[saved] metadata to {RESULTS_DIR}")

    return {
        "gt_masked_all": gt_masked_all,
        "kae_masked_all": kae_masked_all,
        "acdm_masked_all": acdm_masked_all,
        "kae_mse": kae_mse,
        "acdm_mse": acdm_mse,
        f"{cond}_numbers": Mach_numbers,
    }


def plot_koopman_modes(
    model,
    eigenvalues_film: np.ndarray,
    num_modes: int = 16,
    device: str = "cpu",
    channel_titles: list[str] = ["v_x", "v_y", "p"],
):
    """
    Decode and visualize Koopman eigenvectors (modes) from latent space.

    Args:
        model: Koopman model with a decoder (model.decoder).
        eigenvalues_film: latent Koopman modes / eigenvectors, shape (num_modes, latent_dim) or similar.
        num_modes: number of leading modes to visualize.
        device: torch device to use for decoding.
        channel_titles: optional list of channel names (for 3D data).
    """
    # Ensure channel_titles for 3D plotting
    assert channel_titles is not None, "Cannot name variables"

    # Convert eigenvalues to latent tensor
    latent_modes = np.real(eigenvalues_film).squeeze()
    latent_modes_tensor = torch.tensor(latent_modes, dtype=torch.float32, device=device)

    # Decode latent modes to input space
    with torch.no_grad():
        decoded_modes = model.decoder(latent_modes_tensor).cpu().numpy()
    decoded_modes = np.atleast_3d(decoded_modes)  # ensure at least 3D

    # Plot only as many modes as available
    num_available = decoded_modes.shape[0]
    num_to_plot = min(num_modes, num_available)

    for i in range(num_to_plot):
        mode = decoded_modes[i]
        λ = (
            eigenvalues_film.squeeze(0)[i]
            if eigenvalues_film.ndim > 1
            else eigenvalues_film[i]
        )

        if mode.ndim == 1:
            # 1-D signal
            plt.figure(figsize=(8, 2))
            plt.plot(mode)
            plt.title(f"Koopman Mode {i}  (λ = {λ:.3g})")
            plt.xlabel("Input Dimension")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        elif mode.ndim == 2:
            # 2-D single-channel image
            plt.figure(figsize=(4, 4))
            vmax = np.max(np.abs(mode))
            plt.imshow(mode, cmap="bwr", vmin=-vmax, vmax=vmax)
            plt.title(f"Koopman Mode {i}  (λ = {λ:.3g})")
            plt.axis("off")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

        elif mode.ndim == 3 and mode.shape[0] in [3, 4]:
            # 3-channel 2-D input (e.g., u-v-w flow)
            fig, axs = plt.subplots(1, mode.shape[0], figsize=(13, 4))
            vmax = np.max(np.abs(mode))

            for j in range(mode.shape[0]):
                im = axs[j].imshow(mode[j], cmap="bwr", vmin=-vmax, vmax=vmax)
                axs[j].set_title(
                    channel_titles[j] if j < len(channel_titles) else f"Channel {j}"
                )
                axs[j].axis("off")
                cb = fig.colorbar(im, ax=axs[j], fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=8)

            fig.suptitle(f"Koopman Mode {i}  (λ = {λ:.3g})", fontsize=14)
            plt.tight_layout()
            plt.show()
