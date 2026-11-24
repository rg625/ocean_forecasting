"""A 1-D U-Net autoregressive forecaster for the Kuramoto--Sivashinsky equation.

This is the deterministic baseline whose 4D-Var analogue is compared against the
continuous Koopman autoencoder.  The repository's existing U-Nets (``thermalizer``,
``autoreg_pde_diffusion``) are 2-D and cannot be applied to the 64-point 1-D KS grid,
so a 1-D counterpart is defined here following the same design (residual blocks,
GroupNorm, SiLU, skip connections), with **circular** padding to respect the periodic
KS domain.

The model is a one-step map on the *normalised* field at the data cadence
``dt = 0.1``::

    x_{t+dt} = F_theta(x_t) = x_t + N_theta(x_t)

The residual parameterisation matches standard neural-PDE-surrogate practice and is
what makes long autoregressive rollouts stable.

Capacity and training protocol mirror the Koopman autoencoder's KS run
(``hidden_dims=[64,128,256]``, Adam, lr 1e-3 with warmup/linear decay, L2 loss,
rollout-10 curriculum) so that neither model is advantaged by its training budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    """GroupNorm with a channel-count-safe number of groups."""
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


class ResBlock1d(nn.Module):
    """Two circular-padded 1-D convolutions with a residual connection."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(
            c_in, c_out, kernel_size, padding=pad, padding_mode="circular"
        )
        self.conv2 = nn.Conv1d(
            c_out, c_out, kernel_size, padding=pad, padding_mode="circular"
        )
        self.norm1 = _gn(c_in)
        self.norm2 = _gn(c_out)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


@dataclass
class UNet1dConfig:
    in_channels: int = 1
    out_channels: int = 1
    hidden_dims: tuple = (64, 128, 256)
    kernel_size: int = 3
    blocks_per_level: int = 2
    residual: bool = True  # predict the increment rather than the state


class UNet1d(nn.Module):
    """1-D U-Net, ``[B, C, X] -> [B, C, X]``, periodic in ``X``."""

    def __init__(self, cfg: UNet1dConfig | None = None, **kw):
        super().__init__()
        cfg = cfg or UNet1dConfig(**kw)
        self.cfg = cfg
        dims: List[int] = list(cfg.hidden_dims)
        k = cfg.kernel_size

        self.stem = nn.Conv1d(
            cfg.in_channels, dims[0], k, padding=k // 2, padding_mode="circular"
        )

        # ---- encoder --------------------------------------------------------
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.down_blocks.append(
                nn.Sequential(
                    *[
                        ResBlock1d(dims[i] if b == 0 else dims[i], dims[i], k)
                        for b in range(cfg.blocks_per_level)
                    ]
                )
            )
            self.downsample.append(
                nn.Conv1d(
                    dims[i],
                    dims[i + 1],
                    4,
                    stride=2,
                    padding=1,
                    padding_mode="circular",
                )
            )

        # ---- bottleneck -----------------------------------------------------
        self.mid = nn.Sequential(
            *[ResBlock1d(dims[-1], dims[-1], k) for _ in range(cfg.blocks_per_level)]
        )

        # ---- decoder --------------------------------------------------------
        self.upsample = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(dims) - 1)):
            self.upsample.append(
                nn.ConvTranspose1d(dims[i + 1], dims[i], 4, stride=2, padding=1)
            )
            self.up_blocks.append(
                nn.Sequential(
                    ResBlock1d(dims[i] * 2, dims[i], k),
                    *[
                        ResBlock1d(dims[i], dims[i], k)
                        for _ in range(cfg.blocks_per_level - 1)
                    ],
                )
            )

        self.out_norm = _gn(dims[0])
        self.out_conv = nn.Conv1d(
            dims[0], cfg.out_channels, k, padding=k // 2, padding_mode="circular"
        )
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)  # start as the identity map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One time step. ``x``: [B, C, X] normalised field."""
        h = self.stem(x)
        skips = []
        for blocks, down in zip(self.down_blocks, self.downsample):
            h = blocks(h)
            skips.append(h)
            h = down(h)
        h = self.mid(h)
        for up, blocks in zip(self.upsample, self.up_blocks):
            h = up(h)
            h = blocks(torch.cat([h, skips.pop()], dim=1))
        out = self.out_conv(F.silu(self.out_norm(h)))
        return x + out if self.cfg.residual else out

    # -- convenience ---------------------------------------------------------
    def rollout(self, x: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Apply the map ``n_steps`` times (``n_steps=0`` returns ``x``)."""
        for _ in range(n_steps):
            x = self(x)
        return x
