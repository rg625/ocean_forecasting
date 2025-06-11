import torch
from torch import nn, Tensor
from tensordict import TensorDict
from einops import reduce, rearrange, repeat
from torchvision.transforms import GaussianBlur
import logging
from typing import Union, Optional, Dict

# Configure logging
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
        beta (float): Weight for the latent KL divergence loss term.
        re_weight (float): Weight for the optional Reynolds number prediction loss.
        weighting_type (str): Rollout weighting schedule ('cosine' or 'uniform').
        sigma_blur (Optional[float]): If provided, applies a Gaussian blur with this sigma
                                      to the ground truth tensors before loss calculation.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        re_weight: float = 0.1,
        weighting_type: str = "cosine",
        sigma_blur: Optional[float] = None,
    ):
        super().__init__()

        if not (alpha >= 0 and beta >= 0 and re_weight >= 0):
            raise ValueError(
                "Loss weights (alpha, beta, re_weight) must be non-negative."
            )
        if weighting_type not in ["cosine", "uniform"]:
            raise ValueError(f"Unknown weighting type: {weighting_type}")

        self.alpha = alpha
        self.beta = beta
        self.re_weight = re_weight
        self.weighting_type = weighting_type

        self.gaussian_blur = self._init_blur_transform(sigma_blur)

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
        loss_dict = {}
        common_keys = pred.keys() & true.keys()
        if not common_keys:
            logger.warning(
                "No common keys found between prediction and ground truth for reconstruction loss."
            )
            return {}

        for key in common_keys:
            diff = pred[key] - self._blur(true[key])
            loss_per_sample = reduce(diff**2, "b ... -> b", "mean")
            loss_dict[key] = reduce(loss_per_sample, "b ->", "mean")
        return loss_dict

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

    @staticmethod
    def _kl_divergence(latent_pred: Tensor) -> Tensor:
        """Computes KL divergence to a standard normal distribution N(0,1)."""
        mu = reduce(latent_pred, "b t d -> b t", "mean")
        # Add epsilon for numerical stability before taking the log
        var = (
            reduce((latent_pred - mu.unsqueeze(-1)) ** 2, "b t d -> b t", "mean") + 1e-6
        )
        # Correctly average over time for each batch sample -> (B,)
        kl_per_sample = -0.5 * reduce(
            1.0 + torch.log(var) - mu**2 - var, "b t -> b", "mean"
        )
        return reduce(kl_per_sample, "b ->", "mean")

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
        reynolds: Optional[Tensor],
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
        latent_loss = self._kl_divergence(latent_pred)

        re_loss = torch.tensor(0.0, device=latent_pred.device)
        if self.re_weight > 0 and reynolds is not None and "Re" in x_future:
            re_loss = self._compute_re_loss(reynolds, x_future["Re"])

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

        total_loss = (
            total_recon_loss
            + self.alpha * total_pred_loss
            + self.beta * latent_loss
            + self.re_weight * re_loss
        )

        # --- Prepare Detached Dictionary for Logging ---
        return {
            "total_loss": total_loss,  # Keep this attached to the graph for backprop
            "loss_recon": total_recon_loss.detach(),
            "loss_pred": total_pred_loss.detach(),
            "loss_latent": latent_loss.detach(),
            "loss_re": re_loss.detach(),
            "details_recon": {k: v.detach() for k, v in recon_loss_dict.items()},
            "details_pred": {k: v.detach() for k, v in pred_loss_dict.items()},
        }
