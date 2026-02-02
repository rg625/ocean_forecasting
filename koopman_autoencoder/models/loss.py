import torch
from torch import nn, Tensor
import torch.nn.functional as F
from tensordict import TensorDict
from einops import reduce, rearrange, repeat
from torchvision.transforms import GaussianBlur
from torchmetrics.functional import structural_similarity_index_measure as ssim
import logging
from typing import Optional, Dict, Literal, Callable, Tuple
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration Data Classes
# ==============================================================================


@dataclass
class LossConfig:
    """
    Central configuration for KoopmanLoss.
    """

    # General
    loss_type: Literal["l1", "l2"] = "l2"

    # Weights
    alpha: float = 20.0  # Prediction/Rollout weight (High for prediction-first)
    beta: float = 5.0  # Latent consistency weight (High for linearity)
    physics_weight: float = 1.0  # Global physics weight
    re_weight: Optional[float] = None
    stability_weight: Optional[float] = None

    # Physics Sub-weights
    gamma_time: float = 1.0  # Temporal gradient (Velocity)
    gamma_space: float = 1.0  # Spatial gradient (Sobolev)
    gamma_spectral: float = 1.0  # FFT consistency

    # Advanced Options
    ssim_weight: Optional[float] = None
    sigma_blur: Optional[float] = None
    weighting_type: Literal["cosine", "uniform"] = "cosine"


@dataclass
class LossResult:
    """Structured return type for the loss function."""

    total_loss: Tensor
    metrics: Dict[str, float]


# ==============================================================================
# Functional Utilities (Stateless)
# ==============================================================================


class LossUtils:
    """Stateless utility functions for physics-informed losses."""

    @staticmethod
    def calc_distance(
        pred: Tensor, target: Tensor, mode: Literal["l1", "l2"] = "l2"
    ) -> Tensor:
        if mode == "l1":
            return F.l1_loss(pred, target, reduction="none")
        return F.mse_loss(pred, target, reduction="none")

    @staticmethod
    def sobolev_gradients(tensor: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Computes spatial gradients (dx, dy). Expects [B, T, C, H, W] or [B, C, H, W]."""
        if tensor.ndim == 5:  # B, T, C, H, W
            dx = tensor[..., :, 1:] - tensor[..., :, :-1]
            dy = tensor[..., 1:, :] - tensor[..., :-1, :]
            return dx, dy
        elif tensor.ndim == 4:  # B, C, H, W
            dx = tensor[..., :, 1:] - tensor[..., :, :-1]
            dy = tensor[..., 1:, :] - tensor[..., :-1, :]
            return dx, dy
        return None, None

    @staticmethod
    def temporal_derivative(tensor: Tensor) -> Tensor:
        """Computes d/dt via finite difference."""
        if tensor.shape[1] < 2:
            return torch.zeros_like(tensor)
        return tensor[:, 1:] - tensor[:, :-1]

    @staticmethod
    def fft_consistency_pixels(
        pred: Tensor, target: Tensor, mode: Literal["l1", "l2"]
    ) -> Tensor:
        """Computes spectral consistency loss for spatial data (2D FFT)."""
        # Collapse batch and time for 2D Spatial FFT
        p_flat = rearrange(pred, "b t c h w -> (b t) c h w")
        t_flat = rearrange(target, "b t c h w -> (b t) c h w")

        fft_pred = torch.fft.rfft2(p_flat, norm="ortho")
        fft_target = torch.fft.rfft2(t_flat, norm="ortho")

        amp_loss = LossUtils.calc_distance(
            fft_pred.abs(), fft_target.abs(), mode
        ).mean()
        # Complex distance for phase
        phase_loss = LossUtils.calc_distance(
            torch.view_as_real(fft_pred), torch.view_as_real(fft_target), mode
        ).mean()

        return amp_loss + phase_loss

    @staticmethod
    def fft_consistency_latent(pred: Tensor, target: Tensor) -> Tensor:
        """
        Computes spectral consistency including PHASE.
        """
        # fft: [Batch, Freq, D]
        fft_pred = torch.fft.rfft(pred, dim=1, norm="ortho")
        fft_true = torch.fft.rfft(target, dim=1, norm="ortho")

        # 1. Magnitude Loss (Frequency Strength)
        # Use L1 for cleaner spectra
        mag_loss = F.l1_loss(fft_pred.abs(), fft_true.abs())

        # 2. Phase Loss (Real/Imaginary Alignment)
        # view_as_real treats complex numbers as vectors [Real, Imag]
        # This forces the peaks of the waves to line up.
        complex_pred = torch.view_as_real(fft_pred)
        complex_true = torch.view_as_real(fft_true)
        phase_loss = F.mse_loss(complex_pred, complex_true)

        # Weigh phase less heavily so the model finds the frequency first
        return mag_loss + 0.1 * phase_loss


# ==============================================================================
# Modular Loss Components
# ==============================================================================


class BaseModule(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.cfg = config
        self.blur_transform = self._init_blur(config.sigma_blur)

    def _init_blur(self, sigma: Optional[float]) -> Optional[GaussianBlur]:
        if sigma is None or sigma <= 0:
            return None
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        return GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def preprocess(self, tensor: Tensor) -> Tensor:
        """Applies Gaussian blur if configured (for multi-scale stability)."""
        if self.blur_transform is None:
            return tensor

        if tensor.ndim == 5:
            b, t, c, h, w = tensor.shape
            flat = rearrange(tensor, "b t c h w -> (b t) c h w")
            blurred = self.blur_transform(flat)
            return rearrange(blurred, "(b t) c h w -> b t c h w", b=b, t=t)
        elif tensor.ndim == 4:
            return self.blur_transform(tensor)
        return tensor


class ReconstructionLoss(BaseModule):
    """Handles standard L1/L2 reconstruction and optional SSIM."""

    def __init__(self, config: LossConfig, to_unit_range: Optional[Callable] = None):
        super().__init__(config)
        self.to_unit_range = to_unit_range

        if (
            self.cfg.ssim_weight is not None
            and self.cfg.ssim_weight > 0
            and self.to_unit_range is None
        ):
            raise ValueError("to_unit_range callable required for SSIM loss.")

    def forward(
        self, pred: TensorDict, true: TensorDict
    ) -> Tuple[Tensor, Dict[str, float]]:
        common_keys = pred.keys() & true.keys()
        if common_keys:
            device = pred[next(iter(common_keys))].device
        else:
            # Fallback
            device = torch.device("cpu")

        total_loss = torch.tensor(0.0, device=device)
        metrics = {}

        for key in common_keys:
            p, t = pred[key], self.preprocess(true[key])

            # --- 1. CALCULATE WEIGHT MAP ---
            # Calculate gradient magnitude of the TARGET
            if t.ndim >= 4:
                # Handle both [B, C, H, W] and [B, T, C, H, W] using ellipsis
                dx = t[..., :, 1:] - t[..., :, :-1]
                dy = t[..., 1:, :] - t[..., :-1, :]

                # Pad to match original shape
                dx = F.pad(dx, (0, 1, 0, 0))
                dy = F.pad(dy, (0, 0, 0, 1))

                grad_mag = torch.sqrt(dx**2 + dy**2 + 1e-6)

                # Create a weight map: 1.0 for background, up to 10.0 for shocks
                # tanh forces it to cap at 10.0 (1 + 9*1)
                pixel_weights = 1.0 + 2.0 * torch.tanh(grad_mag)
            else:
                pixel_weights = 1.0

            # --- 2. WEIGHTED LOSS ---
            if self.cfg.loss_type == "l2":
                dist = (p - t) ** 2
            else:
                dist = (p - t).abs()

            # Apply the weights
            weighted_dist = dist * pixel_weights

            # Take the mean (This is your final pixel loss)
            base_loss = weighted_dist.mean()

            # --- 3. SSIM LOSS (Optional) ---
            if self.cfg.ssim_weight is not None and self.cfg.ssim_weight > 0:
                assert (
                    self.to_unit_range is not None
                ), f"Expected to_unit_range to exist bot got {type(self.to_unit_range)} instead."
                p_norm = self.to_unit_range(p)
                t_norm = self.to_unit_range(t)

                # SSIM requires [B, C, H, W], so flatten time dim if it exists
                if p_norm.ndim == 5:
                    p_norm = rearrange(p_norm, "b t c h w -> (b t) c h w")
                    t_norm = rearrange(t_norm, "b t c h w -> (b t) c h w")

                # ssim_val is similarity (1.0 is good). Loss is (1 - ssim).
                ssim_val = ssim(p_norm, t_norm, data_range=1.0)
                base_loss += self.cfg.ssim_weight * (1.0 - ssim_val)

            total_loss = total_loss + base_loss
            metrics[f"recon_{key}"] = base_loss.detach().item()

        return total_loss, metrics


class PredictionLoss(BaseModule):
    """Handles time-weighted rollout prediction loss."""

    def forward(
        self, pred: TensorDict, true: TensorDict
    ) -> Tuple[Tensor, Dict[str, float]]:
        if "seq_length" not in true:
            device = (
                pred.device if getattr(pred, "device", None) else torch.device("cpu")
            )
            return torch.tensor(0.0, device=device), {}

        device = true["seq_length"].device
        seq_len = int(true["seq_length"][0, 0].item())
        weights = self._get_weights(seq_len, device=device)

        total_loss = torch.tensor(0.0, device=device)
        metrics = {}

        for key in pred.keys() & true.keys():
            diff = (pred[key] - self.preprocess(true[key])) ** 2
            step_loss = reduce(diff, "b t ... -> b t", "mean")

            weighted_loss = step_loss * weights.view(1, -1)
            loss_val = reduce(weighted_loss, "b t ->", "mean")

            total_loss = total_loss + loss_val
            metrics[f"pred_{key}"] = loss_val.detach().item()

        return total_loss * self.cfg.alpha, metrics

    def _get_weights(self, timesteps: int, device: torch.device) -> Tensor:
        if timesteps <= 0:
            return torch.tensor([], device=device)

        if self.cfg.weighting_type == "uniform":
            return torch.ones(timesteps, device=device) / timesteps

        # Cosine weighting
        idx = torch.arange(timesteps, device=device, dtype=torch.float32)
        if timesteps > 1:
            weights = 0.5 * (1 + torch.cos(torch.pi * idx / (timesteps - 1)))
        else:
            weights = torch.ones(1, device=device)

        return weights / (weights.sum() + 1e-8)


class PhysicsConsistencyLoss(BaseModule):
    """Handles Sobolev (spatial), Temporal (velocity), and Spectral (FFT) consistency on PIXELS."""

    def forward(
        self, pred: TensorDict, true: TensorDict
    ) -> Tuple[Tensor, Dict[str, float]]:
        if pred.keys():
            device = pred[next(iter(pred.keys()))].device
        else:
            device = torch.device("cpu")

        if self.cfg.physics_weight <= 0:
            return torch.tensor(0.0, device=device), {}

        total_phys = torch.tensor(0.0, device=device)
        metrics = {}

        for key in pred.keys() & true.keys():
            p, t = pred[key], true[key]

            if self.cfg.gamma_time > 0:
                v_pred = LossUtils.temporal_derivative(p)
                v_true = LossUtils.temporal_derivative(t)
                loss_t = LossUtils.calc_distance(
                    v_pred, v_true, self.cfg.loss_type
                ).mean()
                total_phys += self.cfg.gamma_time * loss_t
                metrics[f"phys_time_{key}"] = loss_t.detach().item()

            if self.cfg.gamma_space > 0 and p.ndim >= 4:
                dx_p, dy_p = LossUtils.sobolev_gradients(p)
                dx_t, dy_t = LossUtils.sobolev_gradients(t)
                if dx_p is not None:
                    loss_s = (
                        LossUtils.calc_distance(dx_p, dx_t, self.cfg.loss_type).mean()
                        + LossUtils.calc_distance(dy_p, dy_t, self.cfg.loss_type).mean()
                    )
                    total_phys += self.cfg.gamma_space * loss_s
                    metrics[f"phys_space_{key}"] = loss_s.detach().item()

            if self.cfg.gamma_spectral > 0 and p.ndim == 5:
                # Note: Pixel spectral loss is usually less critical than latent spectral
                loss_fft = LossUtils.fft_consistency_pixels(p, t, self.cfg.loss_type)
                total_phys += self.cfg.gamma_spectral * loss_fft
                metrics[f"phys_fft_{key}"] = loss_fft.detach().item()

        return total_phys * self.cfg.physics_weight, metrics


class LatentDynamicsLoss(BaseModule):
    """
    Handles constraints on the latent space Z.
    Includes the critical 'Spectral Loss' for Phase Drift.
    """

    def forward(
        self,
        latent_pred: Tensor,
        true_latents: Optional[Tensor],
        koopman_op: nn.Module,
        future_cond: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:

        if self.cfg.beta <= 0 or true_latents is None:
            return torch.tensor(0.0, device=latent_pred.device), {}

        loss_accum = torch.tensor(0.0, device=latent_pred.device)
        metrics = {}

        # 1. Basic Trajectory Matching
        traj_loss = F.mse_loss(latent_pred, true_latents)
        loss_accum += traj_loss
        metrics["latent_traj"] = traj_loss.detach().item()

        # 2. Linearity Consistency (Forward-Backward Check)
        if hasattr(koopman_op, "dt_train"):
            lin_loss = self._linearity_check(koopman_op, true_latents, future_cond)
            loss_accum += lin_loss
            metrics["latent_linearity"] = lin_loss.detach().item()

        # 3. Norm/Energy Matching (Prevent dissipative collapse)
        norm_p = torch.norm(latent_pred, p=2, dim=-1)
        norm_t = torch.norm(true_latents, p=2, dim=-1)
        energy_loss = F.mse_loss(norm_p, norm_t)
        loss_accum += 1.0 * energy_loss  # Weight from previous diagnosis
        metrics["latent_energy"] = energy_loss.detach().item()

        # 4. Smoothness (2nd derivative minimization) - Manifold Mismatch Fix
        if true_latents.shape[1] >= 3:
            z_acc = (
                true_latents[:, 2:] - 2 * true_latents[:, 1:-1] + true_latents[:, :-2]
            )
            smooth_loss = torch.mean(z_acc**2)
            loss_accum += 1.0 * smooth_loss
            metrics["latent_smooth"] = smooth_loss.detach().item()

        # 5. Latent Spectral Consistency - Phase Drift Fix
        # Anchors the frequency response of the model
        if self.cfg.gamma_spectral > 0:
            spec_loss = LossUtils.fft_consistency_latent(latent_pred, true_latents)
            loss_accum += 2.0 * spec_loss * self.cfg.gamma_spectral
            metrics["latent_spectral"] = spec_loss.detach().item()

        return loss_accum * self.cfg.beta, metrics

    def _linearity_check(
        self, op: nn.Module, z_true: Tensor, cond: Optional[Tensor]
    ) -> Tensor:
        """Checks Forward and Backward consistency of the operator."""
        z0 = z_true[:, :-1].reshape(-1, z_true.shape[-1])
        z1 = z_true[:, 1:].reshape(-1, z_true.shape[-1])

        cond_flat = None
        if cond is not None:
            if cond.shape[0] == z_true.shape[0]:  # [B, T, ...]
                c_slice = cond[:, :-1]
                cond_flat = c_slice.reshape(-1, *c_slice.shape[2:])
            elif cond.shape[0] == z_true.shape[0] and cond.ndim == 1:
                # Static cond needs broadcasting to time
                cond_expanded = repeat(cond, "b -> (b t)", t=z_true.shape[1] - 1)
                cond_flat = cond_expanded

        z1_pred = op(z0, cond=cond_flat, dt=op.dt_train)
        fwd_loss = F.mse_loss(z1_pred, z1)

        z0_pred = op(z1, cond=cond_flat, dt=-op.dt_train)
        bwd_loss = F.mse_loss(z0_pred, z0)

        return fwd_loss + bwd_loss


# ==============================================================================
# Main Composer Class
# ==============================================================================


class KoopmanLoss(nn.Module):
    """
    A modular, production-ready loss function for Koopman Operator models.
    """

    def __init__(
        self,
        config: Optional[LossConfig] = None,
        to_unit_range: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()

        if config is None:
            # Filter kwargs to only those valid for LossConfig
            # valid_keys = LossConfig.__annotations__.keys()
            # config_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            self.cfg = LossConfig(**kwargs)
        else:
            self.cfg = config

        self.to_unit_range = to_unit_range

        # Sub-modules
        self.recon_loss = ReconstructionLoss(self.cfg, to_unit_range)
        self.pred_loss = PredictionLoss(self.cfg)
        self.physics_loss = PhysicsConsistencyLoss(self.cfg)
        self.latent_loss = LatentDynamicsLoss(self.cfg)

    def forward(
        self,
        koopman_operator: nn.Module,
        x_recon: TensorDict,  # Autoencoder output
        x_preds: TensorDict,  # Rollout prediction
        latent_pred: Tensor,  # Predicted latent trajectory
        x_true: TensorDict,  # Ground truth input
        x_future: TensorDict,  # Ground truth future (target)
        true_latents: Optional[Tensor] = None,  # Ground truth latent (if available)
        reynolds: Optional[Tensor] = None,  # Optional physics param
        disturbed_latents: Optional[Tensor] = None,  # For stability loss
        dz_dt: Optional[Tensor] = None,
        dz_dt_disturbed: Optional[Tensor] = None,
    ) -> LossResult:

        # Dict to return compatible with existing Trainer
        # All scalar values in this dict will be logged
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=latent_pred.device)

        # 1. Reconstruction Loss
        l_recon, m_recon = self.recon_loss(x_recon, x_true)
        total_loss = total_loss + l_recon
        loss_dict["loss_recon"] = l_recon.detach().item()
        loss_dict.update(m_recon)  # Add detailed metrics

        # 2. Prediction (Rollout) Loss
        l_pred, m_pred = self.pred_loss(x_preds, x_future)
        total_loss = total_loss + l_pred
        loss_dict["loss_pred"] = l_pred.detach().item()
        loss_dict.update(m_pred)

        # 3. Physics Consistency (Gradients/FFT on Pixels)
        l_phys, m_phys = self.physics_loss(x_preds, x_future)
        total_loss = total_loss + l_phys
        loss_dict["loss_phys"] = l_phys.detach().item()
        loss_dict.update(m_phys)

        # 4. Latent Space Dynamics (Critical for Phase/Stability)
        cond_target = x_future.get("cond_target", None)
        l_latent, m_latent = self.latent_loss(
            latent_pred, true_latents, koopman_operator, cond_target
        )
        total_loss = total_loss + l_latent
        loss_dict["loss_latent"] = l_latent.detach().item()
        loss_dict.update(m_latent)

        # 5. Auxiliary: Reynolds Number Loss
        if (
            self.cfg.re_weight is not None
            and self.cfg.re_weight > 0
            and reynolds is not None
        ):
            target_re = x_future.get("cond_target")
            if target_re is not None:
                if target_re.ndim == 3 and reynolds.ndim == 2:
                    target_re = target_re.mean(dim=1)

                l_re = F.mse_loss(reynolds, target_re) * self.cfg.re_weight
                total_loss = total_loss + l_re
                loss_dict["aux_reynolds"] = l_re.detach().item()

        # 6. Stability Regularization
        if (
            self.cfg.stability_weight is not None
            and self.cfg.stability_weight > 0
            and disturbed_latents is not None
            and dz_dt_disturbed is not None
            and dz_dt is not None
        ):
            state_diff = F.mse_loss(latent_pred, disturbed_latents)
            deriv_diff = F.mse_loss(dz_dt, dz_dt_disturbed)

            l_stab = (state_diff + deriv_diff) * self.cfg.stability_weight
            total_loss = total_loss + l_stab.detach()  # Detach to act as regularizer
            loss_dict["stability"] = l_stab.detach().item()

        # Final key for trainer compatibility
        loss_dict["total_loss"] = total_loss.detach().item()

        return LossResult(total_loss=total_loss, metrics=loss_dict)
