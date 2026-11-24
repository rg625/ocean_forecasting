# ruff: noqa: F841
# mypy: disable-error-code="assignment, var-annotated"
import argparse
import logging
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from tqdm import tqdm
from dataclasses import dataclass
from tensordict import TensorDict
from scipy.stats import pearsonr
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
from torchvision.transforms import GaussianBlur
from einops import rearrange

# --- LOCAL IMPORTS ---
from models.autoencoder import KoopmanAutoencoder
from models.transformer import ViT
from models.dataloader import create_dataloaders
from models.utils import load_checkpoint, load_datasets, load_config
from models.metrics_utils import run_exp_kae_rollout
from models.metrics_utils import run_kae_rollout
from models.qg import QGPhysics

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

QG_MEAN = {
    "q_1": 0.0,
    "q_2": 0.0,
}

QG_STD = {
    "q_1": 1.0,
    "q_2": 1.0,
}


@dataclass
class QGEvalConfig:
    config_path: str
    ckpt_paths: list
    model_labels: list
    rollout_steps: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./thermalizer_results"
    test_file: str = None
    blur_sigma: float = 1.0


class ThermalizerEvaluator:
    def __init__(self, cfg: QGEvalConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        Path(self.cfg.save_dir).mkdir(parents=True, exist_ok=True)
        self.blur_transform = self._init_blur(self.cfg.blur_sigma)

        self._initialize_models()
        self.physics = QGPhysics(device=self.device)
        # Store physical wavenumber scale
        self.k_min = 2 * np.pi / self.physics.L

    def _init_blur(self, sigma):
        """Initializes the GaussianBlur module dynamically based on sigma."""
        if sigma is None or sigma <= 0.0:
            return None
        # Rule of thumb: kernel size should be ~ 4*sigma + 1 (must be odd)
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        logger.info(
            f"Initialized Gaussian Blur with sigma={sigma}, kernel_size={kernel_size}"
        )
        return GaussianBlur(kernel_size=kernel_size, sigma=sigma).to(self.device)

    def apply_blur_to_td(self, td: TensorDict) -> TensorDict:
        """Applies Gaussian Blur to all spatial tensors in the TensorDict."""
        if self.blur_transform is None:
            return td

        blurred_td = td.clone()
        for k in list(blurred_td.keys()):
            tensor = blurred_td[k]
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                continue

            # Handle typical shapes: [B, T, H, W] or [B, T, C, H, W]
            if tensor.ndim == 4:
                b, t, h, w = tensor.shape
                flat = rearrange(tensor, "b t h w -> (b t) 1 h w")
                blurred = self.blur_transform(flat)
                blurred_td[k] = rearrange(blurred, "(b t) 1 h w -> b t h w", b=b, t=t)
            elif tensor.ndim == 5:
                b, t, c, h, w = tensor.shape
                flat = rearrange(tensor, "b t c h w -> (b t) c h w")
                blurred = self.blur_transform(flat)
                blurred_td[k] = rearrange(blurred, "(b t) c h w -> b t c h w", b=b, t=t)

        return blurred_td

    def _initialize_models(self):
        logger.info(f"Loading Config: {self.cfg.config_path}")
        self.exp_cfg = load_config(self.cfg.config_path)

        if self.cfg.test_file:
            self.exp_cfg.data.test_file = self.cfg.test_file

        logger.info("Loading Datasets...")
        self.train_dataset, self.val_dataset, self.test_dataset = load_datasets(
            self.exp_cfg
        )
        self.train_loader, _, _ = create_dataloaders(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.exp_cfg.training,
            num_workers=0,
        )

        logger.info(f"Building {len(self.cfg.ckpt_paths)} Models...")
        self.models = []
        for step, ckpt in enumerate(self.cfg.ckpt_paths):
            logger.info(f"Loading Checkpoint: {ckpt}")
            if step == 0:
                model = KoopmanAutoencoder(
                    data_variables=self.exp_cfg.data.variables,
                    input_frames=self.exp_cfg.data.input_sequence_length,
                    height=self.exp_cfg.model.height,
                    width=self.exp_cfg.model.width,
                    latent_dim=self.exp_cfg.model.latent_dim,
                    cond_embedding_dim=self.exp_cfg.model.cond_embedding_dim,
                    cond_type=self.exp_cfg.model.cond_type,
                    operator_mode=self.exp_cfg.model.operator_mode,
                    hidden_dims=self.exp_cfg.model.hidden_dims,
                    block_size=self.exp_cfg.model.block_size,
                    kernel_size=self.exp_cfg.model.kernel_size,
                    transformer_config=self.exp_cfg.model.transformer,
                    use_checkpoint=self.exp_cfg.training.use_checkpoint,
                    predict_cond=self.exp_cfg.model.predict_cond,
                    cond_grad_enabled=self.exp_cfg.model.cond_grad_enabled,
                    disturb_std=None,
                    is_continuous=self.exp_cfg.model.is_continuous,
                    rank=self.exp_cfg.model.rank,
                    cond_expansion_type=self.exp_cfg.data.selection_param,
                    use_attention=self.exp_cfg.model.use_attention,
                    spectral=self.exp_cfg.model.spectral,
                    **self.exp_cfg.model.conv_kwargs,
                ).to(self.device)

                optimizer = optim.Adam(model.parameters(), lr=1e-4)
                model, _, _, _ = load_checkpoint(
                    ckpt, model=model, optimizer=optimizer, strict=True
                )
                model.eval()
                self.models.append(model)
            elif step == 1:
                model = ViT(
                    data_variables=self.exp_cfg.data.variables,
                    input_frames=self.exp_cfg.data.input_sequence_length,
                    height=self.exp_cfg.model.height,
                    width=self.exp_cfg.model.width,
                    latent_dim=self.exp_cfg.model.latent_dim,
                    cond_embedding_dim=self.exp_cfg.model.cond_embedding_dim,
                    cond_type=self.exp_cfg.model.cond_type,
                    operator_mode=self.exp_cfg.model.operator_mode,
                    hidden_dims=self.exp_cfg.model.hidden_dims,
                    block_size=self.exp_cfg.model.block_size,
                    kernel_size=self.exp_cfg.model.kernel_size,
                    transformer_config=self.exp_cfg.model.transformer,
                    use_checkpoint=self.exp_cfg.training.use_checkpoint,
                    predict_cond=self.exp_cfg.model.predict_cond,
                    cond_grad_enabled=self.exp_cfg.model.cond_grad_enabled,
                    disturb_std=None,
                    is_continuous=self.exp_cfg.model.is_continuous,
                    rank=self.exp_cfg.model.rank,
                    cond_expansion_type=self.exp_cfg.data.selection_param,
                    use_attention=self.exp_cfg.model.use_attention,
                    spectral=self.exp_cfg.model.spectral,
                    **self.exp_cfg.model.conv_kwargs,
                ).to(self.device)

                optimizer = optim.Adam(model.parameters(), lr=1e-4)
                model, _, _, _ = load_checkpoint(
                    ckpt, model=model, optimizer=optimizer, strict=True
                )
                model.eval()
                self.models.append(model)

    def _extract_qg_state(self, td: TensorDict) -> torch.Tensor:
        if "q_1" in td.keys() and "q_2" in td.keys():
            return torch.stack([td["q_1"], td["q_2"]], dim=2)
        elif "q" in td.keys():
            q = td["q"]
            return q.unsqueeze(2) if q.ndim == 4 else q
        return td.get("state")

    def denormalize_unit_variance(self, q_norm):
        q_phys = torch.zeros_like(q_norm)
        q_phys[:, :, 0] = q_norm[:, :, 0] * QG_STD["q_1"]
        q_phys[:, :, 1] = q_norm[:, :, 1] * QG_STD["q_2"]
        return q_phys

    def normalize_unit_variance(self, q_norm):
        q_phys = torch.zeros_like(q_norm)
        q_phys[:, :, 0] = q_norm[:, :, 0] / QG_STD["q_1"]
        q_phys[:, :, 1] = q_norm[:, :, 1] / QG_STD["q_2"]
        return q_phys

    def _compute_acc(self, preds, truth):
        """Anomaly Correlation Coefficient (ACC)"""
        B, T, C, H, W = preds.shape

        # climatology (mean over time)
        clim = truth.mean(axis=1, keepdims=True)

        preds_anom = preds - clim
        truth_anom = truth - clim

        num = (preds_anom * truth_anom).reshape(B, T, -1).sum(axis=-1)
        den = np.linalg.norm(preds_anom.reshape(B, T, -1), axis=-1) * np.linalg.norm(
            truth_anom.reshape(B, T, -1), axis=-1
        )

        return num / (den + 1e-8)

    def _compute_drift(self, series):
        """Drift relative to initial value"""
        return (series - series[:, [0]]) / (np.abs(series[:, [0]]) + 1e-8)

    def _compute_temporal_autocorrelation(self, trajectory):
        B, T, C, H, W = trajectory.shape
        flat_traj = trajectory.reshape(B, T, -1)
        x0 = flat_traj[:, 0:1, :]
        dot_prod = (x0 * flat_traj).sum(dim=-1)
        norm_0 = torch.norm(x0, dim=-1)
        norm_t = torch.norm(flat_traj, dim=-1)
        acf = dot_prod / (norm_0 * norm_t)
        return acf.cpu().numpy()

    def _compute_total_ke(self, q_norm):
        B, T, C, H, W = q_norm.shape
        q_flat = q_norm.reshape(-1, C, H, W).to(self.device)

        psi = self.physics.invert_pv_to_streamfunction_unit(q_flat)
        ke_total = 0.5 * (psi**2).sum(dim=(1, 2, 3)) / (H * W)

        return ke_total.reshape(B, T).cpu().numpy()

    def _compute_total_enstrophy(self, q_norm):
        """
        Compute layer-weighted total enstrophy:
        Z = 0.5 * <q^2>
        """
        B, T, C, H, W = q_norm.shape

        # Flatten batch and time
        q_flat = q_norm.reshape(-1, C, H, W).to(self.device)

        # Layer weights (same as KE)
        w1 = self.physics.H1 / self.physics.H_total
        w2 = self.physics.H2 / self.physics.H_total

        # Enstrophy per layer
        z1 = 0.5 * (q_flat[:, 0] ** 2).mean(dim=(1, 2))
        z2 = 0.5 * (q_flat[:, 1] ** 2).mean(dim=(1, 2))

        # Weighted total
        z_total = w1 * z1 + w2 * z2

        return z_total.reshape(B, T).cpu().numpy()

    # ----------------------------
    # Compute 2D FFT spectrum (unit-variance)
    # ----------------------------
    def _compute_full_spectrum(self, q_norm):
        """
        Computes the Layer-Thickness Weighted Kinetic Energy Spectrum.

        Matches paper description:
        1. Input: Normalized PV (q)
        2. Inversion: Normalized (Unit) Inversion
        3. Metric: Kinetic Energy (k^2 * |psi|^2)
        4. Aggregation: Weighted average of Upper (H1) and Lower (H2) layers.
        """
        B, T, C, H, W = q_norm.shape

        # --- 1. SETUP ---
        q_flat = q_norm.reshape(-1, C, H, W).to(self.device)
        w1 = self.physics.H1 / self.physics.H_total
        w2 = self.physics.H2 / self.physics.H_total

        # --- 2. INVERSION (Normalized) ---
        psi = self.physics.invert_pv_to_streamfunction_unit(q_flat)

        # --- 3. FFT ---
        psi_hat = torch.fft.fftn(psi, dim=(-2, -1))

        # --- 4. COMPUTE KINETIC ENERGY DENSITY ---
        kx = torch.fft.fftfreq(W, device=self.device) * W * 2 * np.pi
        ky = torch.fft.fftfreq(H, device=self.device) * H * 2 * np.pi
        kx, ky = torch.meshgrid(kx, ky, indexing="ij")
        k_sq = kx**2 + ky**2

        psi_power = (psi_hat.abs() ** 2) / (H * W) ** 2
        ke_2d = k_sq.unsqueeze(0).unsqueeze(0) * psi_power

        # --- 5. LAYER THICKNESS WEIGHTING ---
        ke_weighted = (w1 * ke_2d[:, 0, :, :]) + (w2 * ke_2d[:, 1, :, :])

        # --- 6. AZIMUTHAL INTEGRATION (BINNING) ---
        k_r = torch.sqrt(kx**2 + ky**2) / (2 * np.pi)
        k_r = k_r.round().long()

        max_k = min(H, W) // 2
        max_bin = int(k_r.max().item())

        out_spec = torch.zeros((B * T, max_bin + 1), device=self.device)
        k_r_flat = k_r.flatten()
        ke_flat = ke_weighted.reshape(-1, H * W)

        out_spec.index_add_(1, k_r_flat, ke_flat)

        # --- 7. RETURN ---
        spec_1d = out_spec[:, :max_k].reshape(B, T, max_k).cpu().numpy()
        return spec_1d

    def _compute_enstrophy_spectrum(self, spec_ke):
        """Convert KE spectrum to enstrophy spectrum: Z(k) = k^2 * KE(k)"""
        k_indices = np.arange(spec_ke.shape[-1])
        k_phys = k_indices * (2 * np.pi / self.physics.L)

        k_sq = k_phys**2
        k_sq[0] = 0.0  # avoid nan at k=0

        return spec_ke * k_sq[None, None, :]

    def _compute_error_growth(self, mse):
        """Fit exponential growth rate: MSE ~ exp(lambda t)"""
        log_mse = np.log(mse + 1e-12)
        B, T = log_mse.shape
        growth_rates = np.zeros(B)

        for b in range(B):
            t = np.arange(T)
            slope, _ = np.polyfit(t[:50], log_mse[b, :50], 1)  # early-time fit
            growth_rates[b] = slope

        return growth_rates

    def _compute_spectral_error(self, spec_pred, spec_true):
        return np.abs(spec_pred - spec_true) / (spec_true + 1e-8)

    def run_eval(self, num_samples=16):
        logger.info(
            f"Running rollouts for {num_samples} samples across {len(self.models)} models..."
        )
        preds_buffers = [[] for _ in self.models]
        truth_buffer = []

        dt = self.exp_cfg.data.subsample * 0.05
        indices = np.arange(self.cfg.rollout_steps)

        with torch.inference_mode():
            total_samples = min(num_samples, len(self.val_dataset))
            for idx in tqdm(range(total_samples)):
                input_seq, ground_truth, _ = self.val_dataset[
                    idx, self.cfg.rollout_steps
                ]
                if input_seq.batch_size == torch.Size([]):
                    input_seq = input_seq.unsqueeze(0)

                truth_raw = ground_truth.unsqueeze(0)
                t_state = self._extract_qg_state(truth_raw)

                T_common_min = t_state.shape[1]
                p_states = []
                pred_raws = []
                truth_raw = self.apply_blur_to_td(truth_raw)
                t_state = self._extract_qg_state(truth_raw)

                # Evaluate all models on this sample
                for m_idx, model in enumerate(self.models):
                    if m_idx == 0:
                        rollout_td = run_exp_kae_rollout(
                            model,
                            input_seq,
                            self.cfg.rollout_steps,
                            return_xpreds=False,
                            dt=dt,
                            indices=indices,
                        )
                        pred_raw = rollout_td.x_preds
                        # --- APPLY BLUR TO PREDICTIONS HERE ---
                        # pred_raw = self.apply_blur_to_td(pred_raw)
                        p_state = self._extract_qg_state(pred_raw)

                        p_states.append(p_state)
                        pred_raws.append(pred_raw)
                        T_common_min = min(T_common_min, p_state.shape[1])
                    elif m_idx == 1:
                        rollout_td = run_kae_rollout(
                            model,
                            input_seq,
                            self.cfg.rollout_steps,
                            return_xpreds=False,
                        )
                        # --- APPLY BLUR TO PREDICTIONS HERE ---
                        pred_raw = rollout_td.x_preds
                        pred_raw = self.apply_blur_to_td(pred_raw)
                        p_state = self._extract_qg_state(pred_raw)

                        p_states.append(p_state)
                        pred_raws.append(pred_raw)
                        T_common_min = min(T_common_min, p_state.shape[1])

                truth_buffer.append(t_state[:, :T_common_min].cpu())
                for m_idx in range(len(self.models)):
                    preds_buffers[m_idx].append(p_states[m_idx][:, :T_common_min].cpu())

                # We return the raw dicts of the last evaluated sample for ensemble visualizations
                last_pred_raws = pred_raws
                last_truth_raw = truth_raw

        metrics_list = []
        logger.info("Computing Time-Series Metrics...")

        # Ground Truth Metrics (Computed Once)
        full_truth = torch.cat(truth_buffer, dim=0)
        full_truth_norm = self.normalize_unit_variance(full_truth)

        gt_metrics = {
            "ke_true": self._compute_total_ke(full_truth),
            "enstrophy_true": self._compute_total_enstrophy(full_truth),
            "spec_true": self._compute_full_spectrum(full_truth),
            "acf_true": self._compute_temporal_autocorrelation(full_truth),
            "vis_true": full_truth[0, :, 0].numpy(),
            "spec_k": np.arange(self.physics.NX // 2) * self.k_min,
        }
        gt_metrics["zspec_true"] = self._compute_enstrophy_spectrum(
            gt_metrics["spec_true"]
        )

        # Per-Model Prediction Metrics
        for m_idx in range(len(self.models)):
            full_preds = torch.cat(preds_buffers[m_idx], dim=0)
            full_preds_norm = self.normalize_unit_variance(full_preds)

            metrics = gt_metrics.copy()
            metrics["ke_pred"] = self._compute_total_ke(full_preds)
            metrics["enstrophy_pred"] = self._compute_total_enstrophy(full_preds)
            metrics["spec_pred"] = self._compute_full_spectrum(full_preds)

            B, T, C, H, W = full_preds_norm.shape
            metrics["mse"] = (
                ((full_preds - full_truth) ** 2).mean(dim=(2, 3, 4)).numpy()
            )
            metrics["rmse"] = np.sqrt(metrics["mse"])

            metrics["acf_pred"] = self._compute_temporal_autocorrelation(full_preds)
            metrics["acc"] = self._compute_acc(full_preds.numpy(), full_truth.numpy())

            corrs = np.zeros((B, T))
            fp_flat = full_preds.reshape(B, T, -1).numpy()
            ft_flat = full_truth.reshape(B, T, -1).numpy()
            for b in range(B):
                for t in range(T):
                    corrs[b, t] = pearsonr(fp_flat[b, t], ft_flat[b, t])[0]
            metrics["pattern_corr"] = corrs

            metrics["zspec_pred"] = self._compute_enstrophy_spectrum(
                metrics["spec_pred"]
            )
            metrics["vis_pred"] = full_preds[0, :, 0].numpy()
            metrics["ke_drift"] = self._compute_drift(metrics["ke_pred"])
            metrics["z_drift"] = self._compute_drift(metrics["enstrophy_pred"])
            metrics["error_growth"] = self._compute_error_growth(metrics["mse"])
            metrics["spec_error"] = self._compute_spectral_error(
                metrics["spec_pred"], metrics["spec_true"]
            )

            metrics_list.append(metrics)

        return metrics_list, last_pred_raws, last_truth_raw

    # =========================================================================
    # PLOTTING HELPER FUNCTIONS
    # =========================================================================

    def _setup_plot(self):
        sns.set_context("paper", font_scale=1.5)
        sns.set_style("ticks")
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        return fig, ax

    def _plot_mean_std(
        self,
        ax,
        data_mains,
        data_ref=None,
        labels_main=None,
        label_ref="Reference",
        colors_main=None,
        color_ref="gray",
        title="",
        ylabel="",
    ):

        # --- REFERENCE ---
        if data_ref is not None:
            if data_ref.ndim > 2:
                data_ref = data_ref.reshape(
                    data_ref.shape[0], data_ref.shape[1], -1
                ).mean(axis=2)

            mean_ref = data_ref.mean(axis=0)
            std_ref = data_ref.std(axis=0)
            t = np.arange(data_ref.shape[1])

            ax.plot(t, mean_ref, color=color_ref, lw=2.5, label=label_ref)
            ax.fill_between(
                t, mean_ref - std_ref, mean_ref + std_ref, color=color_ref, alpha=0.2
            )

        # --- MAIN (MULTIPLE MODELS) ---
        for i, data_main in enumerate(data_mains):
            if data_main.ndim > 2:
                data_main = data_main.reshape(
                    data_main.shape[0], data_main.shape[1], -1
                ).mean(axis=2)

            t = np.arange(data_main.shape[1])
            mean_main = data_main.mean(axis=0)
            std_main = data_main.std(axis=0)

            color = colors_main[i]
            label = labels_main[i]

            ax.plot(t, mean_main, color=color, lw=2.5, label=label)
            ax.fill_between(
                t, mean_main - std_main, mean_main + std_main, color=color, alpha=0.2
            )

            # --- OPTIONAL: few trajectories ---
            for j in range(min(3, data_main.shape[0])):
                ax.plot(t, data_main[j], color=color, alpha=0.1, lw=0.5)

        ax.set_title(title)
        ax.set_xlabel("Time Steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_time_series_metric(
        self,
        metrics_list,
        key_main,
        title,
        ylabel,
        filename,
        labels_main,
        key_ref=None,
        color_ref="gray",
        label_ref="Simulation",
        y_scale="linear",
        ylim=None,
    ):
        """Unified helper to plot simple time-series metrics to keep the code DRY."""
        fig, ax = self._setup_plot()

        data_ref = metrics_list[0][key_ref] if key_ref else None
        data_mains = [m[key_main] for m in metrics_list]
        colors_main = ["crimson", "dodgerblue", "forestgreen", "darkorange", "purple"][
            : len(metrics_list)
        ]

        self._plot_mean_std(
            ax,
            data_mains,
            data_ref,
            labels_main,
            label_ref,
            colors_main,
            color_ref,
            title,
            ylabel,
        )

        if y_scale == "log":
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)

        plt.savefig(f"{self.cfg.save_dir}/{filename}", dpi=200)
        plt.close()

    # =========================================================================
    # PLOTTING METHODS
    # =========================================================================

    def plot_acc(self, metrics_list, labels):
        self._plot_time_series_metric(
            metrics_list,
            "acc",
            "Anomaly Correlation Coefficient",
            "ACC",
            "metric_acc.png",
            labels,
            ylim=(0, 1.05),
        )

    def plot_mse(self, metrics_list, labels):
        self._plot_time_series_metric(
            metrics_list,
            "mse",
            "Mean Squared Error",
            "MSE",
            "metric_mse.png",
            labels,
            y_scale="log",
        )

    def plot_rmse(self, metrics_list, labels):
        self._plot_time_series_metric(
            metrics_list,
            "rmse",
            "Root Mean Squared Error",
            "RMSE",
            "metric_rmse.png",
            labels,
            y_scale="log",
        )

    def plot_acf(self, metrics_list, labels):
        self._plot_time_series_metric(
            metrics_list,
            "acf_pred",
            "Temporal Autocorrelation",
            "Autocorrelation",
            "metric_autocorrelation.png",
            labels,
            key_ref="acf_true",
            ylim=(-0.2, 1.05),
        )

    def plot_ke_time(self, metrics_list, labels):
        self._plot_time_series_metric(
            metrics_list,
            "ke_pred",
            "Total Kinetic Energy",
            "Total KE [$m^2/s^2$]",
            "metric_ke_time.png",
            labels,
            key_ref="ke_true",
            label_ref="Simulation KE",
        )

    def plot_pattern_corr(self, metrics_list, labels):
        self._plot_time_series_metric(
            metrics_list,
            "pattern_corr",
            "Spatial Pattern Correlation",
            "Pearson R",
            "metric_pattern_correlation.png",
            labels,
            ylim=(0, 1.05),
        )

    def plot_ke_drift(self, metrics_list, labels):
        self._plot_time_series_metric(
            metrics_list,
            "ke_drift",
            "Energy Drift (Stability)",
            "Relative Drift",
            "metric_ke_drift.png",
            labels,
        )

    def plot_enstrophy_time(self, metrics_list, labels):
        self._plot_time_series_metric(
            metrics_list,
            "enstrophy_pred",
            "Total Enstrophy",
            "Enstrophy [$s^{-2}$]",
            "metric_enstrophy_time.png",
            labels,
            key_ref="enstrophy_true",
            label_ref="Simulation Z",
        )

    def plot_spectral_error(self, metrics_list, labels, timestep=-1):
        fig, ax = self._setup_plot()
        k = metrics_list[0]["spec_k"]
        colors = ["crimson", "dodgerblue", "forestgreen", "darkorange", "purple"]

        for m_idx, metrics in enumerate(metrics_list):
            err = metrics["spec_error"][:, timestep]
            err_mean = err.mean(axis=0)
            c = colors[m_idx % len(colors)]
            ax.loglog(k[1:], err_mean[1:], lw=2, color=c, label=labels[m_idx])

        ax.set_xlabel("k")
        ax.set_ylabel("Relative Error")
        ax.set_title("Spectral Error")
        ax.legend()

        plt.savefig(f"{self.cfg.save_dir}/metric_spectral_error.png", dpi=200)
        plt.close()

    def plot_spectrum(self, metrics_list, labels, timestep=-1):
        """Plot KE spectra for a specific timestep across all trajectories."""
        fig, ax = self._setup_plot()

        # X-Axis: Physical Wavenumber
        k_indices = np.arange(1, metrics_list[0]["spec_pred"].shape[-1])
        k_phys = k_indices * (2 * np.pi / self.physics.L)

        dark_colors = ["crimson", "dodgerblue", "forestgreen", "darkorange", "purple"]

        # Plot Simulation spectra
        st_full = metrics_list[0]["spec_true"][:, timestep, 1:]
        st_full = np.maximum(st_full, 1e-16)

        # Faint cloud
        for spec in st_full:
            ax.loglog(k_phys, spec, color="navy", alpha=0.05, lw=0.5)
        # Bold mean
        ax.loglog(
            k_phys,
            st_full.mean(axis=0),
            color="navy",
            alpha=0.9,
            lw=2.5,
            label="Simulation",
            zorder=3,
        )

        # Plot Emulator spectra
        for m_idx, metrics in enumerate(metrics_list):
            sp_full = np.maximum(metrics["spec_pred"][:, timestep, 1:], 1e-16)
            c = dark_colors[m_idx % len(dark_colors)]

            # Faint cloud
            for spec in sp_full:
                ax.loglog(k_phys, spec, color=c, alpha=0.05, lw=0.5)
            # Bold mean
            ax.loglog(
                k_phys,
                sp_full.mean(axis=0),
                color=c,
                alpha=0.9,
                lw=2.5,
                label=labels[m_idx],
                zorder=4 + m_idx,
            )

        ax.set_xlabel(r"Wavenumber $k$ [$m^{-1}$]")
        ax.set_ylabel(r"Kinetic Energy [$m^2/s^2$]")
        ax.set_ylim(1e-3, 1e2)
        ax.set_xlim(left=5e-6, right=2e-4)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="lower left")

        plt.savefig(f"{self.cfg.save_dir}/metric_spectrum_t{timestep}.png", dpi=200)
        plt.close()

    def plot_enstrophy_spectrum(self, metrics_list, labels, timestep=-1):
        fig, ax = self._setup_plot()

        k_indices = np.arange(1, metrics_list[0]["zspec_pred"].shape[-1])
        k_phys = k_indices * (2 * np.pi / self.physics.L)

        dark_colors = ["crimson", "dodgerblue", "forestgreen", "darkorange", "purple"]

        # Simulation
        zt = np.maximum(metrics_list[0]["zspec_true"][:, timestep, 1:], 1e-16)
        # Faint cloud
        for spec in zt:
            ax.loglog(k_phys, spec, color="black", alpha=0.05, lw=0.5)
        # Bold mean
        ax.loglog(
            k_phys,
            zt.mean(axis=0),
            color="black",
            alpha=0.9,
            lw=2.5,
            label="Simulation",
            zorder=3,
        )

        # Emulators
        for m_idx, metrics in enumerate(metrics_list):
            zp = np.maximum(metrics["zspec_pred"][:, timestep, 1:], 1e-16)
            c = dark_colors[m_idx % len(dark_colors)]

            # Faint cloud
            for spec in zp:
                ax.loglog(k_phys, spec, color=c, alpha=0.05, lw=0.5)
            # Bold mean
            ax.loglog(
                k_phys,
                zp.mean(axis=0),
                color=c,
                alpha=0.9,
                lw=2.5,
                label=labels[m_idx],
                zorder=4 + m_idx,
            )

        ax.set_xlabel(r"Wavenumber $k$ [$m^{-1}$]")
        ax.set_ylabel(r"Enstrophy [$s^{-2}$]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        plt.savefig(f"{self.cfg.save_dir}/metric_enstrophy_spectrum.png", dpi=200)
        plt.close()

    def plot_field_comparison(
        self, metrics_list, labels, step_indices=[0, 50, 100, 199]
    ):
        try:
            cmap = sns.cm.icefire
        except AttributeError:
            cmap = "RdBu_r"

        truth_seq = metrics_list[0]["vis_true"]
        valid_steps = [s for s in step_indices if s < len(truth_seq)]

        num_models = len(metrics_list)
        cols = 1 + 2 * num_models  # 1 for GT, 2 per model (Pred + Diff)

        fig, axes = plt.subplots(
            len(valid_steps),
            cols,
            figsize=(4 * cols, 3.5 * len(valid_steps)),
            constrained_layout=True,
        )
        if len(valid_steps) == 1:
            axes = axes[None, :]

        for row_idx, t in enumerate(valid_steps):
            t_img = truth_seq[t]

            # Calculate consistent VMAX across all models for better visual comparison
            max_val = np.max(np.abs(t_img))
            for m_idx in range(num_models):
                max_val = max(
                    max_val, np.max(np.abs(metrics_list[m_idx]["vis_pred"][t]))
                )

            # Col 0: Simulation Ground Truth
            ax = axes[row_idx, 0]
            im1 = ax.imshow(
                t_img, cmap=cmap, interpolation="none", vmin=-max_val, vmax=max_val
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(f"t={t}", fontsize=12)
            if row_idx == 0:
                ax.set_title("Simulation")
            fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

            # Rest of the Columns: Models and their Differences
            for m_idx in range(num_models):
                p_img = metrics_list[m_idx]["vis_pred"][t]
                diff = t_img - p_img

                # Pred Col
                ax = axes[row_idx, 1 + 2 * m_idx]
                im2 = ax.imshow(
                    p_img, cmap=cmap, interpolation="none", vmin=-max_val, vmax=max_val
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(labels[m_idx])
                fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

                # Diff Col
                ax = axes[row_idx, 2 + 2 * m_idx]
                diff_max = np.max(np.abs(diff))
                im3 = ax.imshow(
                    diff, cmap=cmap, interpolation="none", vmin=-diff_max, vmax=diff_max
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(f"{labels[m_idx]} Diff")
                fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

        save_path = f"{self.cfg.save_dir}/thermalizer_fields.png"
        plt.savefig(save_path, dpi=200)
        logger.info(f"Saved field comparison to {save_path}")
        plt.close()

    def plot_rollout_with_spectra_ensemble(
        self,
        gt_dict,
        preds_list,
        metrics_list,
        labels,
        variable_name,
        frame_stride=20,
        max_frames=6,
    ):
        """
        Ensemble visualization:
        Row 1: Ground-truth field (trajectory 0)
        Row 2...N+1: KAE predictions for each model
        Last Row: Kinetic energy spectra (ensemble: GT + all KAEs)
        """
        # =====================
        # Data extraction
        # =====================
        gt = gt_dict[variable_name]
        B, T, H, W = gt.shape
        assert B >= 1, "Need at least one trajectory"

        frame_indices = list(range(0, T, frame_stride))[:max_frames]
        num_frames = len(frame_indices)

        num_models = len(preds_list)
        num_rows = num_models + 2  # GT + (Models) + Spectra

        # =====================
        # Figure layout
        # =====================
        fig = plt.figure(figsize=(2.6 * num_frames, 2.6 * num_rows + 0.5))
        height_ratios = [1] * (num_rows - 1) + [1.3]
        gs = gridspec.GridSpec(
            nrows=num_rows,
            ncols=num_frames,
            height_ratios=height_ratios,
            hspace=0.15,
        )

        cmap = "RdBu_r"
        eps = 1e-16
        k_factor = 2 * np.pi / self.physics.L

        # =====================
        # Helper functions
        # =====================
        def plot_field(ax, field, v_max, title=None, ylabel=None):
            ax.imshow(field.cpu(), cmap=cmap, vmin=-v_max, vmax=v_max)
            if title is not None:
                ax.set_title(title, fontsize=11, fontweight="bold", pad=20)
            if ylabel is not None:
                ax.text(
                    -0.08,
                    0.5,
                    ylabel,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=11,
                    fontweight="bold",
                )
            ax.axis("off")

        def plot_spectra(
            ax, k_phys, spec_preds, spec_true, model_labels, show_legend=False
        ):
            pred_colors = [
                "#FF4500",
                "#00BFFF",
                "#32CD32",
                "#9370DB",
                "#FFD700",
            ]  # distinct palette
            true_color = "#001F7F"  # deep navy blue

            # Plot GT first (so it's in the background)
            for spec in spec_true:
                ax.loglog(k_phys, spec, color=true_color, alpha=0.05, lw=0.5)
            # GT Mean
            ax.loglog(
                k_phys,
                spec_true.mean(axis=0),
                color=true_color,
                alpha=0.9,
                lw=2.5,
                zorder=3,
            )

            # Plot models
            for m_idx, spec in enumerate(spec_preds):
                c = pred_colors[m_idx % len(pred_colors)]
                # Faint cloud
                for s in spec:
                    ax.loglog(k_phys, s, color=c, alpha=0.05, lw=0.5)
                # Bold mean
                ax.loglog(
                    k_phys,
                    spec.mean(axis=0),
                    color=c,
                    alpha=0.9,
                    lw=2.5,
                    zorder=4 + m_idx,
                )

            ax.set_ylim(0.7e-3, 1.5e2)
            ax.set_xlim(left=5e-6, right=2e-4)

            # --- FORCE TICKS ON ---
            ax.minorticks_on()
            ax.tick_params(
                axis="both",
                which="major",
                direction="in",
                length=6,
                width=1.5,
                colors="black",
                top=False,
                right=False,
            )
            ax.tick_params(
                axis="y",
                which="minor",
                direction="in",
                length=3.5,
                width=1.0,
                colors="black",
                left=True,
                right=False,
            )

            minor_locator = LogLocator(
                base=10.0, subs=np.arange(2, 10) * 1.0, numticks=10
            )
            ax.yaxis.set_minor_locator(minor_locator)
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_box_aspect(1)

            if show_legend:
                ax.set_ylabel("KE Spectrum")
                legend_handles = []
                for m_idx in range(len(spec_preds)):
                    c = pred_colors[m_idx % len(pred_colors)]
                    legend_handles.append(
                        Line2D([0], [0], color=c, lw=3, label=model_labels[m_idx])
                    )
                legend_handles.append(
                    Line2D([0], [0], color=true_color, lw=3, label="Simulation")
                )

                ax.legend(
                    handles=legend_handles, frameon=False, fontsize=10, handlelength=3
                )
            else:
                ax.set_yticklabels([])

        # =====================
        # Main plotting loop
        # =====================
        for col, t_idx in enumerate(frame_indices):
            # Shared color scale for GT / ALL KAEs
            v_max = gt[0, t_idx].abs().max().item()
            for p_dict in preds_list:
                v_max = max(v_max, p_dict[variable_name][0, t_idx].abs().max().item())

            # ---- Row 0: Ground truth ----
            ax_gt = fig.add_subplot(gs[0, col])
            t_days = t_idx * 416.6
            step_values = [0, 2000, 4000, 6000, 8000, 9999]
            plot_field(
                ax_gt,
                gt[0, t_idx],
                v_max,
                title=f"t = {int(t_days)} days\nstep = {step_values[col]}",
                ylabel="Ground Truth" if col == 0 else None,
            )

            # ---- Row 1..N: KAEs ----
            for m_idx, p_dict in enumerate(preds_list):
                pred = p_dict[variable_name].clone()
                if t_idx == 0:
                    pred[0, 0] = torch.tensor(
                        # gaussian_filter(gt[0, 0].cpu().numpy(), sigma=1., mode="wrap"),
                        gt[0, 0].cpu().numpy(),
                        device=pred.device,
                        dtype=pred.dtype,
                    )
                ax_pred = fig.add_subplot(gs[m_idx + 1, col])
                plot_field(
                    ax_pred,
                    pred[0, t_idx],
                    v_max,
                    ylabel=labels[m_idx] if col == 0 else None,
                )

            # ---- Last Row: Spectra (ensemble) ----
            ax_spec = fig.add_subplot(gs[num_rows - 1, col])
            k_indices = np.arange(1, metrics_list[0]["spec_pred"].shape[-1])
            k_phys = k_indices * k_factor

            spec_preds = [
                np.maximum(m["spec_pred"][:, t_idx, 1:], eps) for m in metrics_list
            ]
            spec_true = np.maximum(metrics_list[0]["spec_true"][:, t_idx, 1:], eps)

            plot_spectra(
                ax_spec, k_phys, spec_preds, spec_true, labels, show_legend=(col == 0)
            )

        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
        save_path = f"{self.cfg.save_dir}/spectra.png"
        plt.savefig(save_path, dpi=200)
        plt.close()

        logger.info(f"Saved field comparison to {save_path}")

    def plot_temporal_resolution_invariance(self, sample_idx=0, base_steps=100):
        """
        Demonstrates the continuous-time properties of the Koopman Operator.
        Generates two plots:
        1. Qualitative: A grid showing the evolution of the field across different dt.
        2. Quantitative: Time-series of Kinetic Energy across different dt.
        """
        logger.info("Generating Temporal Resolution Invariance Plots...")

        # 1. Identify the continuous Koopman model
        kae_model = None
        for i, m in enumerate(self.models):
            if isinstance(m, KoopmanAutoencoder) and getattr(
                m.koopman_operator, "is_continuous", False
            ):
                kae_model = m
                break

        if kae_model is None:
            logger.warning(
                "No continuous Koopman Autoencoder found. Skipping temporal invariance plot."
            )
            return

        # 2. Get initial conditions
        input_seq, _, _ = self.val_dataset[sample_idx, 10]
        if input_seq.batch_size == torch.Size([]):
            input_seq = input_seq.unsqueeze(0)

        # 3. Setup configurations
        base_dt = self.exp_cfg.data.subsample * 0.05
        target_physical_time = base_steps * base_dt

        resolutions = [
            {
                "label": "Coarse (2Δt)",
                "factor": 2.0,
                "color": "darkorange",
                "ls": "--",
                "lw": 3,
            },
            {
                "label": "Base (1Δt)",
                "factor": 1.0,
                "color": "crimson",
                "ls": "-",
                "lw": 2.5,
            },
            {
                "label": "Fine (0.5Δt)",
                "factor": 0.5,
                "color": "dodgerblue",
                "ls": ":",
                "lw": 2.5,
            },
            {
                "label": "Very Fine (0.25Δt)",
                "factor": 0.25,
                "color": "forestgreen",
                "ls": "-.",
                "lw": 1.5,
            },
        ]

        results = {}

        with torch.inference_mode():
            for res in resolutions:
                factor = res["factor"]
                current_dt = base_dt * factor
                # Calculate number of steps needed to reach the exact same target physical time
                steps = int(target_physical_time / current_dt)
                steps = max(1, steps)  # Ensure at least 1 step
                indices = np.arange(steps)

                rollout_td = run_exp_kae_rollout(
                    kae_model,
                    input_seq,
                    steps,
                    return_xpreds=False,
                    dt=current_dt,
                    indices=indices,
                )

                pred_raw = self.apply_blur_to_td(rollout_td.x_preds)
                state = self._extract_qg_state(pred_raw)

                ke = self._compute_total_ke(state)[0]  # Take first batch item [T]

                results[res["label"]] = {
                    "time": (np.arange(steps) + 1)
                    * current_dt,  # Physical time for each predicted frame
                    "ke": ke,
                    "full_trajectory": state[0]
                    .cpu()
                    .numpy(),  # Shape: [steps, C, H, W]
                    "color": res["color"],
                    "ls": res["ls"],
                    "lw": res["lw"],
                }

        # --- 4. QUALITATIVE PLOT (Evolution Grid) ---
        try:
            cmap = sns.cm.icefire
        except AttributeError:
            cmap = "RdBu_r"

        num_milestones = 5
        # Pick 5 checkpoints in time, spacing them out evenly up to the target physical time
        milestone_fractions = np.linspace(0.2, 1.0, num_milestones)
        milestone_times = milestone_fractions * target_physical_time

        # Ensure consistent color scaling across all plots and timesteps
        vmax = max([np.abs(d["full_trajectory"]).max() for d in results.values()])

        fig_qual, axes = plt.subplots(
            len(resolutions),
            num_milestones,
            figsize=(3.5 * num_milestones, 3.5 * len(resolutions)),
            constrained_layout=True,
        )

        for row, res in enumerate(resolutions):
            label = res["label"]
            data = results[label]
            traj = data["full_trajectory"]
            times = data["time"]

            for col, t_target in enumerate(milestone_times):
                ax = axes[row, col]

                # Find the frame closest to this physical milestone
                idx = (np.abs(times - t_target)).argmin()

                # Plot channel 0 (q_1)
                field_to_plot = traj[idx, 0] if traj.ndim == 4 else traj[idx]
                ax.imshow(field_to_plot, cmap=cmap, vmin=-vmax, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])

                # Set column titles (Time)
                if row == 0:
                    ax.set_title(
                        f"Physical Time: t = {50*t_target:.2f}h",
                        fontsize=16,
                        fontweight="bold",
                        pad=15,
                    )
                # Set row labels (Resolution)
                if col == 0:
                    ax.set_ylabel(label, fontsize=16, fontweight="bold", labelpad=15)

        fig_qual.suptitle(
            "Qualitative Temporal Invariance: Spatiotemporal Evolution by Resolution",
            fontsize=22,
            fontweight="bold",
            y=1.05,
        )
        qual_path = (
            Path(self.cfg.save_dir) / "metric_temporal_invariance_qualitative.png"
        )
        fig_qual.savefig(qual_path, dpi=200, bbox_inches="tight")
        plt.close(fig_qual)

        # --- 5. QUANTITATIVE PLOT (KE Time-Series) ---
        fig_quant, ax_quant = plt.subplots(figsize=(10, 6), constrained_layout=True)

        for label, data in results.items():
            ax_quant.plot(
                data["time"],
                data["ke"],
                label=label,
                color=data["color"],
                linestyle=data["ls"],
                linewidth=data["lw"],
                alpha=0.8,
            )

        ax_quant.set_title(
            "Kinetic Energy Tracking Across Temporal Resolutions",
            fontsize=16,
            fontweight="bold",
        )
        ax_quant.set_xlabel("Physical Time (t)", fontsize=14)
        ax_quant.set_ylabel("Total KE", fontsize=14)
        ax_quant.grid(True, alpha=0.3)
        ax_quant.legend(fontsize=12)

        quant_path = (
            Path(self.cfg.save_dir) / "metric_temporal_invariance_quantitative.png"
        )
        fig_quant.savefig(quant_path, dpi=200)
        plt.close(fig_quant)

        logger.info(f"Saved qualitative invariance grid to {qual_path}")
        logger.info(f"Saved quantitative invariance graph to {quant_path}")

    def print_summary(self, metrics_list, labels):
        print("\n===== SUMMARY =====")
        for i, metrics in enumerate(metrics_list):
            print(f"\n--- {labels[i]} ---")
            print(
                f"Final RMSE: {metrics['rmse'][:, -1].mean():.4e} ± {metrics['rmse'][:, -1].std():.4e}"
            )
            print(f"Final ACC: {metrics['acc'][:, -1].mean():.3f}")
            print(f"KE Drift: {metrics['ke_drift'][:, -1].mean():.3f}")
            print(f"Enstrophy Drift: {metrics['z_drift'][:, -1].mean():.3f}")
            print(f"Error Growth Rate: {metrics['error_growth'].mean():.4f}")

    def generate_all_reports(self, metrics_list, preds_list, gts, args):
        """Unified orchestrator to generate all plots and summaries cleanly."""
        logger.info("Generating plots...")
        labels = self.cfg.model_labels

        self.plot_mse(metrics_list, labels)
        self.plot_acf(metrics_list, labels)
        self.plot_ke_time(metrics_list, labels)
        self.plot_pattern_corr(metrics_list, labels)
        self.plot_spectrum(metrics_list, labels, timestep=-1)
        self.plot_field_comparison(
            metrics_list, labels, step_indices=[10, 50, 100, args.steps - 1]
        )
        self.plot_rollout_with_spectra_ensemble(
            gts,
            preds_list,
            metrics_list,
            labels,
            variable_name="q_1",
            frame_stride=1,
            max_frames=20,
        )
        self.plot_enstrophy_time(metrics_list, labels)
        self.plot_enstrophy_spectrum(metrics_list, labels, timestep=-1)
        self.plot_temporal_resolution_invariance(sample_idx=0, base_steps=500)
        self.print_summary(metrics_list, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument(
        "--ckpts",
        nargs="+",
        type=str,
        required=True,
        help="Paths to checkpoint.pth (accepts multiple)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        default=None,
        help="Labels for each checkpoint in plots",
    )
    parser.add_argument("--steps", type=int, default=200, help="Rollout steps")
    parser.add_argument(
        "--samples", type=int, default=32, help="Number of trajectories"
    )
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument(
        "--blur_sigma",
        type=float,
        default=0.0,
        help="Sigma for Gaussian blur (0.0 means no blur)",
    )

    args = parser.parse_args()

    # Automatically generate default labels if not provided
    if args.labels is None:
        args.labels = [f"KAE {i+1}" for i in range(len(args.ckpts))]
    elif len(args.labels) != len(args.ckpts):
        raise ValueError(
            f"Number of labels ({len(args.labels)}) must match number of checkpoints ({len(args.ckpts)})"
        )

    cfg = QGEvalConfig(
        config_path=args.config,
        ckpt_paths=args.ckpts,
        model_labels=args.labels,
        rollout_steps=args.steps,
        test_file=args.test_file,
        blur_sigma=args.blur_sigma,
    )

    evaluator = ThermalizerEvaluator(cfg)
    metrics_list, preds_list, gts = evaluator.run_eval(num_samples=args.samples)

    # Execution wrapped in a single, clean call
    evaluator.generate_all_reports(metrics_list, preds_list, gts, args)
