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

    def __init__(self, mode: Literal["log", "linear"], mean: float, std: float):
        super().__init__()
        self.mode = mode
        # Register as buffers so they save with the model but don't update via gradient
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
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
            x = torch.log10(torch.abs(x) + 1e-6)

        # Standard Z-score normalization
        return (x - self.mean) / self.std


class LearnableRBFEmbedding(nn.Module):
    """
    Projects a scalar physical parameter into a high-dimensional space
    using Gaussian Radial Basis Functions (RBF) + a Linear Trend.
    """

    def __init__(self, num_rbf: int = 15, out_dim: int = 64):
        super().__init__()
        self.out_dim = out_dim

        # 1. The RBF Centers (Where the "sensors" are placed)
        self.centers = nn.Parameter(torch.linspace(-3.0, 3.0, num_rbf))

        # 2. The RBF Widths (Gamma) - Learnable sensitivity
        self.log_gamma = nn.Parameter(torch.ones(num_rbf) * 0.0)

        # 3. Projection to target dimension
        # Input dim = 1 (Linear) + num_rbf (Gaussians)
        self.projection = nn.Linear(1 + num_rbf, out_dim)

    def forward(self, x: torch.Tensor, d: Optional[int] = None) -> torch.Tensor:
        # --- DEVICE SAFETY CHECK ---
        if self.centers.device != x.device:
            self.to(x.device)

        # Ensure input is shaped (..., 1)
        if x.shape[-1] != 1:
            x = x.unsqueeze(-1)

        # 1. RBF Features
        diff = x - self.centers
        gamma = torch.exp(self.log_gamma)
        rbf_out = torch.exp(-gamma * (diff**2))

        # 2. Linear Feature
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
            if lower > 0 and (upper / lower) > 5.0:
                self.active_mode: Literal["log", "linear"] = "log"
            else:
                self.active_mode = "linear"
        else:
            # validate user input
            if mode not in ("log", "linear"):
                raise ValueError("mode must be 'log' or 'linear'")
            self.active_mode = cast(Literal["log", "linear"], mode)

        # 2. Calculate Stats
        if self.active_mode == "log":
            l_val = math.log10(abs(lower) + 1e-9)
            u_val = math.log10(abs(upper) + 1e-9)
        else:
            l_val = float(lower)
            u_val = float(upper)

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
# FACTORIES (Updated to be functions instead of global instances)
# ==============================================================================


def re_expansion(d: int):
    return FourierExpansion(100, 1000, out_dim=d)


def ma_expansion(d: int):
    return FourierExpansion(1e-13, 1.0, out_dim=d)


def forcing_expansion(d: int):
    return FourierExpansion(1.9e-12, 7.1e-12, out_dim=d)
