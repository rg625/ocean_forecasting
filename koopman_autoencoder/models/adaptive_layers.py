import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class AdaLNConv(nn.Module):
    """
    Adaptive Layer Normalization.
    Projects a conditioning vector to a scale and shift for normalization.
    Uses GroupNorm with 1 group, which is equivalent to LayerNorm across spatial dims.
    """

    def __init__(self, C_out: int, cond_dim: int):
        super().__init__()
        # Use GroupNorm for spatial layer normalization, affine is false as we compute it ourselves
        self.norm = nn.GroupNorm(1, C_out, affine=False)
        # Projection layer to get scale (gamma) and shift (beta)
        self.projection = nn.Linear(cond_dim, 2 * C_out)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # Project conditioning vector to get gamma and beta
        # cond shape: [B, cond_dim] -> gamma/beta shape: [B, C_out]
        gamma, beta = self.projection(cond).chunk(2, dim=1)

        # Normalize the input tensor
        x_normalized = self.norm(x)

        # Apply the adaptive scale and shift
        # Reshape gamma/beta to [B, C, 1, 1] for broadcasting over spatial dims (H, W)
        return gamma.view(*gamma.shape, 1, 1) * x_normalized + beta.view(
            *beta.shape, 1, 1
        )


class AdaIN(nn.Module):
    """Adaptive Instance Normalization with clamped modulation."""

    def __init__(self, style_dim: int, channels: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_proj = nn.Linear(style_dim, 2 * channels)

    def forward(self, x, w):
        x = self.norm(x)
        style = self.style_proj(w).view(w.shape[0], -1, 1, 1)
        gamma, beta = style.chunk(2, dim=1)
        gamma = torch.tanh(gamma)  # stabilize modulation
        beta = torch.tanh(beta)
        return (1 + gamma) * x + beta


class AdaLNMLP(nn.Module):
    """
    Adaptive multiplicative modulation conditioned on Reynolds number embedding.

    Produces only a scale (gamma) — no shift — to preserve interpretability.
    Supports both scalar and pre-encoded (e.g., Fourier-expanded) `re` inputs.
    """

    def __init__(self, latent_dim: int, cond_embedding_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_embedding_dim = cond_embedding_dim

        # Process the Reynolds embedding (scalar or Fourier-expanded)
        self.cond_embedding = nn.Sequential(
            nn.Linear(cond_embedding_dim, cond_embedding_dim),
            nn.SiLU(),
            nn.Linear(cond_embedding_dim, cond_embedding_dim),
            nn.SiLU(),
            nn.Linear(cond_embedding_dim, latent_dim),
            nn.Tanh(),
        )

        nn.init.zeros_(self.cond_embedding[-2].weight)
        nn.init.zeros_(self.cond_embedding[-2].bias)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vector (B, latent_dim)
            cond: Condition scalar (B, 1) or pre-expanded embedding (B, D_cond)
        Returns:
            Scaled latent (B, latent_dim)
        """
        # Handle shape: allow both scalar and expanded cond
        if cond.ndim == 1:
            cond = cond.unsqueeze(-1)  # (B, 1)
        elif cond.ndim > 2:
            cond = cond.view(cond.shape[0], -1)  # flatten

        # If scalar, lift to cond_embedding_dim first
        if cond.shape[-1] != self.cond_embedding_dim:
            # Expand scalar cond into embedding_dim linearly
            cond = F.linear(
                cond,
                torch.eye(cond.shape[1], self.cond_embedding_dim, device=cond.device),
            )

        gamma = self.cond_embedding(cond) * 0.1  # (B, latent_dim)

        # Pure multiplicative modulation
        return z * (1 + gamma)

    def get_gamma(self, cond):
        if cond.ndim == 1:
            cond = cond.unsqueeze(-1)  # (B, 1)
        elif cond.ndim > 2:
            cond = cond.view(cond.shape[0], -1)  # flatten

        # If scalar, lift to cond_embedding_dim first
        if cond.shape[-1] != self.cond_embedding_dim:
            # Expand scalar cond into embedding_dim linearly
            cond = F.linear(
                cond,
                torch.eye(cond.shape[1], self.cond_embedding_dim, device=cond.device),
            )

        return self.cond_embedding(cond) * 0.1  # (B, latent_dim)
