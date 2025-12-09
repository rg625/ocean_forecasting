# models/loss.py

import torch
from torch import nn, Tensor
from tensordict import TensorDict
from einops import reduce, rearrange, repeat
from torchvision.transforms import GaussianBlur
import logging
from typing import Union, Optional, Dict, cast, Literal, Callable
from torchmetrics.functional import structural_similarity_index_measure as ssim

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class KoopmanLoss(nn.Module):
    """
    A robust, production-ready loss function for Koopman-based deep learning models.

    This module computes a weighted sum of several loss components:
    1. Reconstruction Loss: Mean Squared Error on the autoencoded initial state.
    2. Prediction/Rollout Loss: Time-weighted MSE on future state predictions.
    3. Latent Regularization Loss: KL divergence to encourage a standard normal distribution
       in the latent space.
    4. Reynolds Number Prediction Loss: An optional auxiliary loss on a physical parameter.

    Args:
        alpha (float): Weight for the prediction (rollout) loss term.
        beta (float): Weight for the latent divergence loss term.
        re_weight (float): Weight for the optional Reynolds number prediction loss.
        stability_weight (float): Weight for regularizing the latent space to be invariant to small perturbations.
        weighting_type (str): Rollout weighting schedule ('cosine' or 'uniform').
        sigma_blur (Optional[float]): If provided, applies a Gaussian blur with this sigma
                                      to the ground truth tensors before loss calculation.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        loss_type: Literal["l1", "l2"] = "l2",
        ssim_weight: Optional[float] = None,
        to_unit_range: Optional[Callable] = None,
        re_weight: Optional[float] = None,
        stability_weight: Optional[float] = None,
        weighting_type: str = "cosine",
        sigma_blur: Optional[float] = None,
        grad_weight: Optional[float] = 1.0,
    ):
        super().__init__()

        if not (alpha >= 0 and beta >= 0):
            raise ValueError("Core loss weights (alpha, beta) must be non-negative.")
        if ssim_weight is not None and ssim_weight > 0 and to_unit_range is None:
            raise ValueError(
                "A `to_unit_range` method must be provided if `ssim_weight` is used."
            )
        if weighting_type not in ["cosine", "uniform"]:
            raise ValueError(f"Unknown weighting type: {weighting_type}")
        if loss_type not in ["l1", "l2"]:
            raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'l1' or 'l2'.")

        self.alpha = alpha
        self.beta = beta
        self.loss_type = loss_type
        self.ssim_weight = ssim_weight if ssim_weight is not None else 0.0
        self.to_unit_range = to_unit_range
        self.re_weight = re_weight
        self.stability_weight = stability_weight
        self.weighting_type = weighting_type
        self.gaussian_blur = self._init_blur_transform(sigma_blur)
        self.grad_weight = grad_weight

    @staticmethod
    def _init_blur_transform(sigma: Optional[float]) -> Optional[GaussianBlur]:
        """Initializes the GaussianBlur transform with a dynamically sized kernel."""
        if sigma is None:
            return None
        if sigma <= 0:
            raise ValueError("sigma_blur must be a positive float.")
        # Rule of thumb: kernel size should be ~4 sigmas in each direction from center.
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        logger.info(
            f"Initializing GaussianBlur with sigma={sigma} and kernel_size={kernel_size}"
        )
        return GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def _blur(self, tensor: Tensor) -> Tensor:
        """Applies the configured Gaussian blur to a tensor."""
        if self.gaussian_blur is None:
            return tensor

        # We only apply blur to spatial data, which we expect to be 4D or 5D.
        if tensor.ndim == 5:  # [B, T, C, H, W]
            B, T, C, H, W = tensor.shape
            tensor_reshaped = rearrange(tensor, "b t c h w -> (b t) c h w")
            blurred_reshaped = self.gaussian_blur(tensor_reshaped)
            return rearrange(blurred_reshaped, "(b t) c h w -> b t c h w", b=B, t=T)
        elif tensor.ndim == 4:  # [B, C, H, W]
            return self.gaussian_blur(tensor)
        else:
            # Do not blur non-spatial data (e.g., latent vectors)
            return tensor

    def _compute_recon_loss(
        self, pred: TensorDict, true: TensorDict
    ) -> Dict[str, Tensor]:
        """Computes reconstruction loss safely over common keys."""
        mse_dict: Dict[str, Tensor] = {}
        common_keys = pred.keys() & true.keys()
        if not common_keys:
            logger.warning(
                "No common keys found between prediction and ground truth for reconstruction loss."
            )
            return {}

        # Only compute SSIM tensors if both utilities are available
        use_ssim = (
            self.ssim_weight is not None
            and self.ssim_weight != 0
            and getattr(self, "to_unit_range", None) is not None
        )

        if use_ssim:
            assert (
                self.to_unit_range is not None
            ), "Expeted to unit range method to be callable"
            pred_ssim = self.to_unit_range(pred)
            true_ssim = self.to_unit_range(true)

        for key in common_keys:
            # Compute base L1 or L2 term
            if self.loss_type == "l1":
                diff = torch.abs(pred[key] - self._blur(true[key]))
            else:  # 'l2'
                diff = (pred[key] - self._blur(true[key])) ** 2

            mse_per_sample = reduce(diff, "b ... -> b", "mean")
            base_loss = reduce(mse_per_sample, "b ->", "mean")

            # Add SSIM term if enabled
            if use_ssim:
                ssim_loss = self._ssim_loss(pred_ssim[key], self._blur(true_ssim[key]))
                base_loss = base_loss + self.ssim_weight * ssim_loss

            mse_dict[key] = base_loss

        return mse_dict

    def _compute_rollout_loss(
        self, pred: TensorDict, true: TensorDict
    ) -> Dict[str, Tensor]:
        """Computes time-weighted rollout loss safely over common keys."""
        if "seq_length" not in true:
            logger.warning(
                "'seq_length' not in ground truth TensorDict. Cannot compute rollout loss."
            )
            return {}

        # This assertion enforces the contract that this loss function expects
        # batches where all samples have the same sequence length.
        seq_len_tensor = true["seq_length"]
        assert torch.all(
            seq_len_tensor == seq_len_tensor[0]
        ), "All samples in a batch must have the same sequence length for this loss implementation."

        seq_len = seq_len_tensor[0, 0].item()
        weights = self._get_rollout_weights(seq_len, device=seq_len_tensor.device)

        loss_dict = {}
        common_keys = pred.keys() & true.keys()
        if not common_keys:
            logger.warning(
                "No common keys found between prediction and ground truth for rollout loss."
            )
            return {}

        for key in common_keys:
            diff = pred[key] - self._blur(true[key])
            per_step_loss = reduce(diff**2, "b n ... -> b n", "mean")
            weighted_loss = per_step_loss * weights.view(1, -1)
            loss_dict[key] = reduce(weighted_loss, "b n ->", "mean")
        return loss_dict

    def _compute_gradient_loss(self, pred: TensorDict, true: TensorDict) -> Tensor:
        """
        Computes 1st order derivative loss (Sobel-like) to fix phase errors/drift.
        """
        # FIX: Init as float 0.0. Python auto-promotes 0.0 + Tensor(cuda) -> Tensor(cuda).
        # Initializing with torch.tensor(0.0) risks picking the wrong device (CPU vs CUDA).
        total_grad_loss = 0.0

        common_keys = set(pred.keys()) & set(true.keys())

        # Track if we processed anything so we can return a valid Tensor at the end
        processed_any = False
        fallback_device = torch.device("cpu")  # Fallback

        for key in common_keys:
            pred_img = pred[key]
            true_img = true[key]

            # Capture device for fallback return if needed
            fallback_device = pred_img.device

            # Handle Temporal dim if present
            if pred_img.ndim == 5:
                pred_img = rearrange(pred_img, "b t c h w -> (b t) c h w")
                true_img = rearrange(true_img, "b t c h w -> (b t) c h w")

            # Compute Spatial Gradients (dy, dx)
            # Diff in height (dim -2)
            pred_dy = torch.abs(pred_img[..., 1:, :] - pred_img[..., :-1, :])
            true_dy = torch.abs(true_img[..., 1:, :] - true_img[..., :-1, :])

            # Diff in width (dim -1)
            pred_dx = torch.abs(pred_img[..., :, 1:] - pred_img[..., :, :-1])
            true_dx = torch.abs(true_img[..., :, 1:] - true_img[..., :, :-1])

            loss_dy = torch.mean(torch.abs(pred_dy - true_dy))
            loss_dx = torch.mean(torch.abs(pred_dx - true_dx))

            total_grad_loss = total_grad_loss + (loss_dy + loss_dx)
            processed_any = True

        # Ensure we return a Tensor (for .detach() calls downstream)
        if not processed_any:
            return torch.tensor(0.0, device=fallback_device)

        return total_grad_loss

    def _get_rollout_weights(self, timesteps: int, device: torch.device) -> Tensor:
        """Computes weights for each step in the rollout loss."""
        if timesteps <= 0:
            return torch.tensor([], device=device)
        idx = torch.arange(timesteps, device=device, dtype=torch.float32)

        if self.weighting_type == "cosine":
            if timesteps > 1:
                weights = 0.5 * (1 + torch.cos(torch.pi * idx / (timesteps - 1)))
            else:
                weights = torch.ones(1, device=device)
            return weights / weights.sum().clamp(
                min=1e-8
            )  # Normalize and prevent div by zero
        elif self.weighting_type == "uniform":
            return torch.ones(timesteps, device=device) / timesteps
        else:
            # This case is already checked in __init__, but as a safeguard:
            raise NotImplementedError(
                f"Weighting type '{self.weighting_type}' not implemented."
            )

    def _true_latents_loss(self, true_latents, latent_pred) -> Dict[str, Tensor]:
        """Computes time-weighted rollout loss safely over common latent space tensors."""
        diff = true_latents - latent_pred
        loss = reduce(diff**2, "b n ... -> b n", "mean")
        return cast(Dict, reduce(loss, "b n ->", "mean"))

    @staticmethod
    def stability_loss(latent_pred: Tensor, disturbed_latents: Tensor) -> Tensor:
        """Computes stability regularization for latent vectors."""
        assert (
            latent_pred.shape == disturbed_latents.shape
        ), f"expected latent vectors and their disturbed version to have the same shape but got {latent_pred.shape} and {disturbed_latents.shape} instead"
        diff = latent_pred - disturbed_latents
        per_step_loss = reduce(diff**2, "b ... -> b", "mean")
        return reduce(per_step_loss, "b ->", "mean")

    def _ssim_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Computes SSIM loss = 1 - SSIM, averaged over batch.
        Accepts [B, H, W], [B, C, H, W], or [B, T, C, H, W].
        """
        if pred.ndim == 3:  # [B, H, W]
            pred = pred.unsqueeze(1)  # -> [B, 1, H, W]
            target = target.unsqueeze(1)
        elif pred.ndim == 5:  # [B, T, C, H, W] → flatten time into batch for SSIM
            B, T, C, H, W = pred.shape
            pred = pred.reshape(B * T, C, H, W)
            target = target.reshape(B * T, C, H, W)
        return 1.0 - ssim(pred, target, data_range=1.0)

    def _compute_re_loss(self, pred_re: Tensor, true_re: Tensor) -> Tensor:
        """Computes MSE loss for the Reynolds number prediction."""

        if true_re.ndim == 2:
            true_re = true_re[:, 0]
        true_expanded = repeat(true_re, "b -> b t 1", t=pred_re.shape[1])

        if true_expanded.shape != pred_re.shape:
            logger.error(
                f"Shape mismatch in Reynolds loss: true {true_expanded.shape}, pred {pred_re.shape}"
            )
            return torch.tensor(0.0, device=pred_re.device)

        loss_per_sample = reduce((true_expanded - pred_re) ** 2, "b t 1 -> b", "mean")
        return reduce(loss_per_sample, "b ->", "mean")

    def forward(
        self,
        x_recon: TensorDict,
        x_preds: TensorDict,
        latent_pred: Tensor,
        x_true: TensorDict,
        x_future: TensorDict,
        true_latents: Optional[Tensor],
        reynolds: Optional[Tensor],
        disturbed_latents: Optional[Tensor] = None,
        dz_dt: Optional[Tensor] = None,
        dz_dt_disturbed: Optional[Tensor] = None,
    ) -> Dict[str, Union[Tensor, float, Dict[str, float]]]:
        """
        Computes the full, weighted Koopman loss.

        Returns:
            A dictionary containing the total loss for backpropagation and detached
            scalar values for all individual loss components for logging.
        """
        # --- Compute Individual Loss Components ---
        recon_loss_dict = self._compute_recon_loss(
            x_recon, x_true.select(*x_recon.keys())
        )
        pred_loss_dict = self._compute_rollout_loss(x_preds, x_future)
        # latent_loss = self._kl_divergence(latent_pred)
        # --- Sum Weighted Losses for Backpropagation ---
        total_recon_loss = (
            sum(recon_loss_dict.values())
            if recon_loss_dict
            else torch.tensor(0.0, device=latent_pred.device)
        )
        total_pred_loss = (
            sum(pred_loss_dict.values())
            if pred_loss_dict
            else torch.tensor(0.0, device=latent_pred.device)
        )

        latent_loss = torch.tensor(0.0, device=latent_pred.device)
        if self.beta is not None and true_latents is not None:
            latent_loss = self._true_latents_loss(
                true_latents=true_latents, latent_pred=latent_pred
            )

        grad_loss = torch.tensor(0.0, device=latent_pred.device)
        if self.grad_weight is not None and self.grad_weight > 0:
            grad_loss += self._compute_gradient_loss(
                x_recon, x_true.select(*x_recon.keys())
            )
            grad_loss += self._compute_gradient_loss(x_preds, x_future)

        total_loss = (
            total_recon_loss
            + self.alpha * total_pred_loss
            + self.beta * latent_loss
            + self.grad_weight * grad_loss
        )
        # --- Conditionally Add Optional Losses ---
        re_loss = torch.tensor(0.0, device=latent_pred.device)
        # Only compute and add the loss if the weight is specified and inputs are available
        if self.re_weight is not None and reynolds is not None:
            assert (
                self.re_weight > 0
            ), f"re_weight must be positive, but got {self.re_weight}"
            re_loss = self._compute_re_loss(reynolds, x_future["cond_target"])
            total_loss = total_loss + self.re_weight * re_loss

        stability_loss = torch.tensor(0.0, device=latent_pred.device)
        # Only compute and add the loss if the weight is specified and inputs are available
        if (
            self.stability_weight is not None
            and disturbed_latents is not None
            and dz_dt_disturbed is not None
        ):
            assert (
                self.stability_weight > 0
            ), f"stability_weight must be positive, but got {self.stability_weight}"
            stability_loss = self.stability_loss(
                latent_pred=latent_pred, disturbed_latents=disturbed_latents
            ) + self._true_latents_loss(true_latents=dz_dt, latent_pred=dz_dt_disturbed)
            # Note: Using .detach() here means the stability loss acts as a regularizer
            # but gradients do not flow back through this specific calculation.
            total_loss = total_loss + self.stability_weight * stability_loss.detach()

        # --- Prepare Detached Dictionary for Logging ---
        return {
            "total_loss": total_loss,  # Keep this attached to the graph for backprop
            "loss_recon": total_recon_loss.detach(),
            "loss_pred": total_pred_loss.detach(),
            "loss_latent": latent_loss.detach(),
            "loss_re": re_loss.detach(),
            "loss_stability": stability_loss.detach(),
            "details_recon": {k: v.detach() for k, v in recon_loss_dict.items()},
            "details_pred": {k: v.detach() for k, v in pred_loss_dict.items()},
        }
