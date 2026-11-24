# mypy: disable-error-code="index, union-attr"
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from tensordict import TensorDict
from einops import reduce, rearrange
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
    alpha: float = 20.0  # Prediction/Rollout weight
    beta: float = 5.0  # Latent consistency weight
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
        if tensor.ndim == 5:
            dx = tensor[..., :, 1:] - tensor[..., :, :-1]
            dy = tensor[..., 1:, :] - tensor[..., :-1, :]
            return dx, dy
        elif tensor.ndim == 4:
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
    def __init__(self, config: LossConfig, to_unit_range: Optional[Callable] = None):
        super().__init__(config)
        self.to_unit_range = to_unit_range

    def forward(
        self, pred: TensorDict, true: TensorDict
    ) -> Tuple[Tensor, Dict[str, float]]:
        common_keys = pred.keys() & true.keys()
        if not common_keys:
            return torch.tensor(0.0), {}

        device = pred[next(iter(common_keys))].device
        total_loss = torch.tensor(0.0, device=device)
        metrics = {}

        for key in common_keys:
            p, t = pred[key], self.preprocess(true[key])

            if t.ndim >= 4:
                dx = t[..., :, 1:] - t[..., :, :-1]
                dy = t[..., 1:, :] - t[..., :-1, :]
                dx = F.pad(dx, (0, 1, 0, 0))
                dy = F.pad(dy, (0, 0, 0, 1))
                grad_mag = torch.sqrt(dx**2 + dy**2 + 1e-6)
                pixel_weights = 1.0 + 2.0 * torch.tanh(grad_mag)
            else:
                pixel_weights = 1.0

            dist = (p - t) ** 2 if self.cfg.loss_type == "l2" else (p - t).abs()
            base_loss = (dist * pixel_weights).mean()

            if (
                self.cfg.ssim_weight is not None
                and self.cfg.ssim_weight > 0
                and self.to_unit_range
            ):
                p_norm, t_norm = self.to_unit_range(p), self.to_unit_range(t)
                if p_norm.ndim == 5:
                    p_norm = rearrange(p_norm, "b t c h w -> (b t) c h w")
                    t_norm = rearrange(t_norm, "b t c h w -> (b t) c h w")
                base_loss += self.cfg.ssim_weight * (
                    1.0 - ssim(p_norm, t_norm, data_range=1.0)
                )

            total_loss = total_loss + base_loss
            metrics[f"recon_{key}"] = base_loss.detach().item()

        return total_loss, metrics


class PredictionLoss(BaseModule):
    def forward(
        self, pred: TensorDict, true: TensorDict
    ) -> Tuple[Tensor, Dict[str, float]]:
        if "seq_length" not in true:
            device = (
                pred[list(pred.keys())[0]].device
                if pred.keys()
                else torch.device("cpu")
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
            loss_val = reduce(step_loss * weights.view(1, -1), "b t ->", "mean")

            total_loss = total_loss + loss_val
            metrics[f"pred_{key}"] = loss_val.detach().item()

        return total_loss * self.cfg.alpha, metrics

    def _get_weights(self, timesteps: int, device: torch.device) -> Tensor:
        if timesteps <= 0:
            return torch.tensor([], device=device)
        if self.cfg.weighting_type == "uniform":
            return torch.ones(timesteps, device=device) / timesteps
        idx = torch.arange(timesteps, device=device, dtype=torch.float32)
        weights = (
            0.5 * (1 + torch.cos(torch.pi * idx / (timesteps - 1)))
            if timesteps > 1
            else torch.ones(1, device=device)
        )
        return weights / (weights.sum() + 1e-8)


class PhysicsConsistencyLoss(BaseModule):
    def forward(
        self, pred: TensorDict, true: TensorDict
    ) -> Tuple[Tensor, Dict[str, float]]:
        if not pred.keys():
            return torch.tensor(0.0), {}

        device = pred[next(iter(pred.keys()))].device
        if self.cfg.physics_weight <= 0:
            return torch.tensor(0.0, device=device), {}

        total_phys = torch.tensor(0.0, device=device)
        metrics = {}

        for key in pred.keys() & true.keys():
            p, t = pred[key], true[key]

            # 1. Velocity Consistency
            if self.cfg.gamma_time > 0:
                loss_t = LossUtils.calc_distance(
                    LossUtils.temporal_derivative(p),
                    LossUtils.temporal_derivative(t),
                    self.cfg.loss_type,
                ).mean()
                total_phys += self.cfg.gamma_time * loss_t
                metrics[f"phys_time_{key}"] = loss_t.detach().item()

            # 2. Laplacian/Enstrophy Consistency
            if self.cfg.gamma_space > 0 and p.ndim >= 4:
                dx_p, dy_p = LossUtils.sobolev_gradients(p)
                dx_t, dy_t = LossUtils.sobolev_gradients(t)
                if dx_p is not None:
                    grad_loss = (
                        LossUtils.calc_distance(dx_p, dx_t, self.cfg.loss_type).mean()
                        + LossUtils.calc_distance(dy_p, dy_t, self.cfg.loss_type).mean()
                    )
                    dxx_p, _ = LossUtils.sobolev_gradients(dx_p)
                    dxx_t, _ = LossUtils.sobolev_gradients(dx_t)
                    _, dyy_p = LossUtils.sobolev_gradients(dy_p)
                    _, dyy_t = LossUtils.sobolev_gradients(dy_t)

                    if dxx_p is not None and dyy_p is not None:
                        lap_p = dxx_p[..., 1:-1, :] + dyy_p[..., :, 1:-1]
                        lap_t = dxx_t[..., 1:-1, :] + dyy_t[..., :, 1:-1]
                        grad_loss += LossUtils.calc_distance(
                            lap_p, lap_t, self.cfg.loss_type
                        ).mean()

                    total_phys += self.cfg.gamma_space * grad_loss
                    metrics[f"phys_space_{key}"] = grad_loss.detach().item()

            # 3. Spectral Consistency
            if self.cfg.gamma_spectral > 0 and p.ndim == 5:
                p_flat = p.reshape(-1, *p.shape[2:]).to(torch.float32)
                t_flat = t.reshape(-1, *t.shape[2:]).to(torch.float32)
                mag_p = torch.log(torch.fft.rfft2(p_flat, norm="ortho").abs() + 1e-8)
                mag_t = torch.log(torch.fft.rfft2(t_flat, norm="ortho").abs() + 1e-8)
                loss_fft = F.mse_loss(mag_p, mag_t)
                total_phys += self.cfg.gamma_spectral * loss_fft
                metrics[f"phys_fft_{key}"] = loss_fft.detach().item()

        return total_phys * self.cfg.physics_weight, metrics


class LatentDynamicsLoss(BaseModule):
    def forward(
        self,
        latent_pred: Tensor,
        true_latents: Optional[Tensor],
        koopman_op: nn.Module,
        future_cond: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        # Fast exit: if beta <= 0, or missing true latents
        if self.cfg.beta <= 0 or true_latents is None:
            return torch.tensor(0.0, device=latent_pred.device), {}

        loss_accum = torch.tensor(0.0, device=latent_pred.device)
        metrics = {}

        traj_loss = F.mse_loss(latent_pred, true_latents)
        loss_accum += traj_loss
        metrics["latent_traj"] = traj_loss.detach().item()

        if hasattr(koopman_op, "dt_train"):
            z0 = true_latents[:, :-1].reshape(-1, true_latents.shape[-1])
            z1 = true_latents[:, 1:].reshape(-1, true_latents.shape[-1])
            cond_flat = (
                future_cond[:, :-1].reshape(-1, *future_cond.shape[2:])
                if (
                    future_cond is not None
                    and future_cond.shape[0] == true_latents.shape[0]
                )
                else None
            )

            # Forward / Backward
            fwd_loss = F.mse_loss(
                koopman_op(z0, cond=cond_flat, dt=koopman_op.dt_train), z1
            )
            bwd_loss = F.mse_loss(
                koopman_op(z1, cond=cond_flat, dt=-koopman_op.dt_train), z0
            )
            lin_loss = fwd_loss + bwd_loss
            loss_accum += lin_loss
            metrics["latent_linearity"] = lin_loss.detach().item()

        # Covariance - NOTE: Invalid if X is normalized and Z does not naturally follow N(0,I).
        z_flat = true_latents.reshape(-1, true_latents.shape[-1])
        z_centered = z_flat - z_flat.mean(dim=0)
        cov_z = (z_centered.T @ z_centered) / (max(z_centered.shape[0] - 1, 1))

        var_loss = F.mse_loss(
            torch.diagonal(cov_z), torch.ones_like(torch.diagonal(cov_z))
        )
        off_diag_mask = ~torch.eye(cov_z.shape[0], device=cov_z.device).bool()
        off_diag_loss = (
            (cov_z[off_diag_mask] ** 2).mean()
            if off_diag_mask.any()
            else torch.tensor(0.0)
        )

        loss_accum += 1e-4 * var_loss + 1e-4 * off_diag_loss
        metrics["latent_cov"] = off_diag_loss.detach().item()
        metrics["latent_var"] = var_loss.detach().item()

        if true_latents.shape[1] > 1:
            smoothness_loss = torch.mean(
                (true_latents[:, 1:] - true_latents[:, :-1]) ** 2
            )
            loss_accum += 0.5 * smoothness_loss
            metrics["latent_smooth"] = smoothness_loss.detach().item()

        if self.cfg.gamma_spectral > 0:
            fft_p = torch.fft.rfft(latent_pred.float(), dim=1, norm="ortho")
            fft_t = torch.fft.rfft(true_latents.float(), dim=1, norm="ortho")
            psd_loss = F.l1_loss(fft_p.abs(), fft_t.abs())
            loss_accum += self.cfg.gamma_spectral * psd_loss
            metrics["latent_psd"] = psd_loss.detach().item()

        return loss_accum * self.cfg.beta, metrics


# ==============================================================================
# Main Composer Class
# ==============================================================================


class KoopmanLoss(nn.Module):
    def __init__(
        self,
        config: Optional[LossConfig] = None,
        to_unit_range: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        self.cfg = config if config else LossConfig(**kwargs)
        self.to_unit_range = to_unit_range

        self.recon_loss = ReconstructionLoss(self.cfg, to_unit_range)
        self.pred_loss = PredictionLoss(self.cfg)
        self.physics_loss = PhysicsConsistencyLoss(self.cfg)
        self.latent_loss = LatentDynamicsLoss(self.cfg)

    def forward(
        self,
        koopman_operator: Optional[nn.Module],  # Optional now! (None if ViT)
        x_recon: TensorDict,
        x_preds: TensorDict,
        latent_pred: Tensor,
        x_true: TensorDict,
        x_future: TensorDict,
        true_latents: Optional[Tensor] = None,
        reynolds: Optional[Tensor] = None,
        disturbed_latents: Optional[Tensor] = None,
        dz_dt: Optional[Tensor] = None,
        dz_dt_disturbed: Optional[Tensor] = None,
    ) -> LossResult:

        # ---------------------------------------------------------
        # 1. ARCHITECTURE DETECTION (Koopman vs ViT)
        # ---------------------------------------------------------
        # If no koopman_operator is passed OR it's a generic ViT class,
        # we bypass all the latent constraints and autoencoding losses.
        is_vit_baseline = (
            koopman_operator is None
            or getattr(koopman_operator, "__class__", None).__name__
            == "PureSequenceViT"
        )

        loss_dict = {}
        # Ensure we have a valid device, falling back to x_preds if latent is a dummy
        device = (
            latent_pred.device
            if not is_vit_baseline
            else x_preds[next(iter(x_preds.keys()))].device
        )
        total_loss = torch.tensor(0.0, device=device)

        # ---------------------------------------------------------
        # 2. CORE LOSSES (Applied to both models)
        # ---------------------------------------------------------

        # Prediction (Rollout) Loss
        l_pred, m_pred = self.pred_loss(x_preds, x_future)
        total_loss = total_loss + l_pred
        loss_dict["loss_pred"] = l_pred.detach().item()
        loss_dict.update(m_pred)

        # Physics Consistency (Gradients/FFT on Pixels)
        # Note: The ViT benefits immensely from this as it forces physical realism!
        l_phys, m_phys = self.physics_loss(x_preds, x_future)
        total_loss = total_loss + l_phys
        loss_dict["loss_phys"] = l_phys.detach().item()
        loss_dict.update(m_phys)

        # ---------------------------------------------------------
        # 3. KOOPMAN-SPECIFIC LOSSES (Bypassed for ViT)
        # ---------------------------------------------------------
        if not is_vit_baseline:
            # Reconstruction Loss
            l_recon, m_recon = self.recon_loss(x_recon, x_true)
            total_loss = total_loss + l_recon
            loss_dict["loss_recon"] = l_recon.detach().item()
            loss_dict.update(m_recon)

            # Latent Space Dynamics
            cond_target = x_future.get("cond_target", None)
            l_latent, m_latent = self.latent_loss(
                latent_pred, true_latents, koopman_operator, cond_target
            )
            total_loss = total_loss + l_latent
            loss_dict["loss_latent"] = l_latent.detach().item()
            loss_dict.update(m_latent)

            # Auxiliary: Reynolds Number Loss
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

            # Stability Regularization
            if (
                self.cfg.stability_weight is not None
                and self.cfg.stability_weight > 0
                and disturbed_latents is not None
                and dz_dt_disturbed is not None
                and dz_dt is not None
            ):
                l_stab = (
                    F.mse_loss(latent_pred, disturbed_latents)
                    + F.mse_loss(dz_dt, dz_dt_disturbed)
                ) * self.cfg.stability_weight
                total_loss = total_loss + l_stab.detach()
                loss_dict["stability"] = l_stab.detach().item()
        else:
            # Fill dict with zeros so trainer logs don't break
            loss_dict.update(
                {
                    "loss_recon": 0.0,
                    "loss_latent": 0.0,
                    "aux_reynolds": 0.0,
                    "stability": 0.0,
                }
            )

        loss_dict["total_loss"] = total_loss.detach().item()
        return LossResult(total_loss=total_loss, metrics=loss_dict)
