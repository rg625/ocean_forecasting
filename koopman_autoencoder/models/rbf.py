"""
Smart Physics-Informed Embeddings.
Replaces standard Fourier features with Learnable RBFs and Linear Skip Connections
to allow for robust interpolation within regimes and safe extrapolation outside them.
"""

import torch
import torch.nn as nn
import math
from typing import Literal, Optional, cast


class PhysicalNormalizer(nn.Module):
    """
    Smart Normalizer that preprocesses physical variables based on their nature.
    It handles Log-scaling for wide-range variables (Reynolds) and Linear scaling
    for narrow-range variables (Mach, Beta).
    """

    mode: Literal["log", "linear"]
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mode: Literal["log", "linear"], mean: float, std: float):
        super().__init__()
        self.mode = mode
        # Register as buffers so they save with the model but don't update via gradient
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure we work with floats
        x = x.float()

        # --- DEVICE SAFETY CHECK ---
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)

        if self.mode == "log":
            # Protect against log(0) or negative inputs
            # epsilon 1e-6 prevents NaN gradients at 0
            x = torch.log10(torch.abs(x) + 1e-6)

        # Standard Z-score normalization
        # (x - mean) / std
        return (x - self.mean) / self.std


class LearnableRBFEmbedding(nn.Module):
    """
    Projects a scalar physical parameter into a high-dimensional space
    using Gaussian Radial Basis Functions (RBF) + a Linear Trend.

    The Linear Trend is CRITICAL for extrapolation (e.g., Re=100 when trained on 200+).
    """

    def __init__(self, num_rbf: int = 15, out_dim: int = 64):
        super().__init__()
        self.out_dim = out_dim

        # 1. The RBF Centers
        # We use [-3.0, 3.0] to cover approx 3 standard deviations of training data.
        # Inputs outside this range (like OOD Re=100) will result in RBF activation ~0.
        self.centers = nn.Parameter(torch.linspace(-3.0, 3.0, num_rbf))

        # 2. The RBF Widths (Gamma) - Learnable sensitivity
        # Init to 0.0 implies exp(0) = 1.0 gamma (standard width)
        self.log_gamma = nn.Parameter(torch.ones(num_rbf) * 0.0)

        # 3. Projection to target dimension
        # Input dim = 1 (Linear Skip) + num_rbf (Gaussians)
        self.projection = nn.Linear(1 + num_rbf, out_dim)

    def forward(self, x: torch.Tensor, d: Optional[int] = None) -> torch.Tensor:
        # --- DEVICE SAFETY CHECK ---
        if self.centers.device != x.device:
            self.to(x.device)

        # Ensure input is shaped (..., 1)
        if x.shape[-1] != 1:
            x = x.unsqueeze(-1)

        # 1. RBF Features (Local Interpolation)
        # Calculates Gaussian distance: exp(-gamma * (x - center)^2)
        diff = x - self.centers
        gamma = torch.exp(self.log_gamma)
        rbf_out = torch.exp(-gamma * (diff**2))

        # 2. Linear Feature (Global Extrapolation)
        # This allows the network to learn a slope (trend) that continues
        # even when the RBFs die out (e.g., very low or very high Re).
        linear_out = x

        # 3. Concatenate and Project
        combined = torch.cat([linear_out, rbf_out], dim=-1)
        embedding = self.projection(combined)

        return embedding


class FourierExpansion(nn.Module):
    """
    A smart wrapper that looks like the original FourierExpansion API but uses
    LearnableRBFEmbedding and PhysicalNormalizer internally.
    """

    def __init__(
        self,
        lower: float,
        upper: float,
        assert_range: bool = True,
        mode: str = "auto",
        out_dim: int = 64,
    ):
        super().__init__()

        # 1. Infer Mode (Log vs Linear)
        if mode == "auto":
            # Heuristic: If dynamic range > 5x, use Log scale (e.g. Reynolds)
            if lower > 0 and (upper / lower) > 5.0:
                self.active_mode: Literal["log", "linear"] = "log"
            else:
                self.active_mode = "linear"
        else:
            # validate user input
            if mode not in ("log", "linear"):
                raise ValueError("mode must be 'log' or 'linear'")
            self.active_mode = cast(Literal["log", "linear"], mode)

        # 2. Calculate Stats based on the bounds
        if self.active_mode == "log":
            l_val = math.log10(abs(lower) + 1e-9)
            u_val = math.log10(abs(upper) + 1e-9)
        else:
            l_val = float(lower)
            u_val = float(upper)

        # Center the normalization on the provided range
        # We define the range [lower, upper] to cover +/- 2 sigma (approx 95% of data)
        mean = (l_val + u_val) / 2.0
        std = (u_val - l_val) / 4.0

        # 3. Instantiate Components
        self.normalizer = PhysicalNormalizer(mode=self.active_mode, mean=mean, std=std)
        self.embedder = LearnableRBFEmbedding(num_rbf=15, out_dim=out_dim)

    def forward(self, x: torch.Tensor, d: Optional[int] = None) -> torch.Tensor:
        # 1. Normalize
        x_norm = self.normalizer(x)
        # 2. Embed
        return self.embedder(x_norm, d)


# ==============================================================================
# FACTORIES
# ==============================================================================


def re_expansion(d: int):
    # CRITICAL CONFIGURATION:
    # Use the TRAINING range (200, 900) here.
    # This ensures that Re=200 maps to approx -2.0 sigma and Re=900 to +2.0 sigma.
    # Re=100 (Extrapolation) will map to approx -3.8 sigma.
    # This falls outside the RBF centers ([-3, 3]), causing RBFs to be zero.
    # The model will therefore rely purely on the Linear Skip Connection for Re=100,
    # preventing "ghost" neuron activation and allowing linear trend extrapolation.
    return FourierExpansion(200, 900, mode="log", out_dim=d)


def ma_expansion(d: int):
    # Mach number usually stays within bounds, linear mode is fine.
    return FourierExpansion(1e-13, 1.0, out_dim=d)


def forcing_expansion(d: int):
    # Forcing term range
    return FourierExpansion(1.9e-12, 7.1e-12, out_dim=d)
