import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import random

seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class GaussianFourierFeatureTransform(nn.Module):
    """
    Maps coordinates (x, y) to a high-dimensional vector using random Fourier features.
    This effectively allows the network to learn high-frequency functions (sharp edges, turbulence)
    much faster than standard coordinate inputs.

    Output dim will be 2 * mapping_size.
    Scale determines the frequency spread (higher = sharper but potentially noisier).
    """

    def __init__(self, in_channels=2, mapping_size=64, scale=10.0):
        super().__init__()
        self._in_channels = in_channels
        self._mapping_size = mapping_size

        # B matrix is fixed, not learned (random projection)
        # Shape: [in_channels, mapping_size]
        self.register_buffer("B", torch.randn(in_channels, mapping_size) * scale)

    def forward(self, x):
        # x: [B, 2, H, W] -> [B, H, W, 2]
        x = x.permute(0, 2, 3, 1)

        # Project: (2 * pi * x) @ B
        # [B, H, W, 2] @ [2, M] -> [B, H, W, M]
        x_proj = (2 * np.pi * x) @ self.B

        # Compute sin and cos
        # Final shape: [B, H, W, 2*M]
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        # Back to [B, 2*M, H, W]
        return out.permute(0, 3, 1, 2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)].to(x.device)
