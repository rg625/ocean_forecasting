import torch
from torch import nn
from torch.nn import utils as nn_utils
import torch.nn.functional as F
from einops import rearrange
from typing import List, Optional
from torch import Tensor
from dataclasses import dataclass

# Attempt to import from local modules
try:
    from .fourier import PositionalEncoding
    from .adaptive_layers import AdaLNMLP
    from .rbf import re_expansion, ma_expansion, forcing_expansion
except ImportError:
    pass

# ==========================================
#   ROBUST LAYERS (Spectral Normalized)
# ==========================================


def sn_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """Spectral Normalized Conv2d. Stabilizes gradients and prevents explosion."""
    return nn_utils.spectral_norm(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    )


def sn_linear(in_features, out_features):
    """Spectral Normalized Linear."""
    return nn_utils.spectral_norm(nn.Linear(in_features, out_features))


class PreActResBlock(nn.Module):
    """
    Pre-activation Residual Block (He et al., 2016).
    Structure: GN -> SiLU -> Weight -> GN -> SiLU -> Weight
    Best for training deep networks and preserving signal propagation.
    """

    def __init__(self, in_ch, out_ch, stride=1, use_spectral_norm=True):
        super().__init__()

        # Select conv constructor
        conv_fn = sn_conv2d if use_spectral_norm else nn.Conv2d

        # GroupNorm is preferred over BatchNorm for physics/small batches
        self.bn1 = nn.GroupNorm(8, in_ch)
        self.conv1 = conv_fn(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

        self.bn2 = nn.GroupNorm(8, out_ch)
        self.conv2 = conv_fn(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        # Shortcut handling
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            # Projection shortcut
            self.shortcut = conv_fn(
                in_ch, out_ch, kernel_size=1, stride=stride, padding=0
            )

    def forward(self, x):
        # Pre-activation path
        x_norm = F.silu(self.bn1(x))
        # Handle shortcut on pre-activated input or raw input?
        # Standard PreAct applies shortcut to raw x, but branches off x_norm
        residual = self.shortcut(x)

        out = self.conv1(x_norm)
        out = self.conv2(F.silu(self.bn2(out)))

        return out + residual


# ==========================================
#   ENCODER (Spectral ResNet)
# ==========================================


class ConvEncoder(nn.Module):
    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        latent_dim: int,
        hiddens: List[int] = [64, 128, 256],
        cond_embedding_dim: Optional[int] = None,
        cond_type: Optional[str] = None,
        cond_expansion_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.cond_type = cond_type
        self.cond_embedding_dim = cond_embedding_dim

        # Setup Conditioning Expansions
        self.cond_expansion_type = cond_expansion_type
        dim_to_use = cond_embedding_dim if cond_embedding_dim is not None else 64
        self.expansion_map = nn.ModuleDict(
            {
                "Re": re_expansion(dim_to_use),
                "Ma": ma_expansion(dim_to_use),
                "forcing": forcing_expansion(dim_to_use),
            }
        )

        # Input channels: Data(C) + Coords(2)
        input_channels = C + 2

        # Coordinate Grid Buffer
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
        )
        self.register_buffer("grid", torch.stack([xx, yy], dim=0))

        # Initial Feature Extraction
        self.init_conv = sn_conv2d(input_channels, hiddens[0], kernel_size=3, padding=1)

        # Downsampling ResBlocks
        layers = []
        in_c = hiddens[0]

        for out_c in hiddens:
            layers.append(PreActResBlock(in_c, out_c, stride=2))  # Downsample
            layers.append(PreActResBlock(out_c, out_c, stride=1))  # Depth
            in_c = out_c

        self.backbone = nn.Sequential(*layers)

        # Calculate output spatial dimension
        scale_factor = 2 ** len(hiddens)
        self.H_out = H // scale_factor
        self.W_out = W // scale_factor

        flat_features = hiddens[-1] * self.H_out * self.W_out

        # Conditioning handling for the linear projection
        if self.cond_type == "late_fusion":
            flat_features += dim_to_use

        # Final projection to latent
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.SiLU(),
            sn_linear(flat_features, latent_dim),
        )
        self.latent_dim = latent_dim

    def _encode_cond(self, cond: Optional[Tensor]) -> Optional[Tensor]:
        if cond is None:
            return None
        if cond.ndim == 1:
            cond = cond.unsqueeze(-1)
        if self.cond_expansion_type in self.expansion_map:
            return self.expansion_map[self.cond_expansion_type](cond)
        return cond

    def forward(self, x: Tensor, cond: Optional[Tensor] = None):
        # 1. Coordinate Injection
        grid = self.grid.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, grid], dim=1)

        # 2. Condition Embedding
        cond_emb = None
        if self.cond_type is not None and cond is not None:
            cond_emb = self._encode_cond(cond)

        # 3. Extract Features
        x = self.init_conv(x)
        x = self.backbone(x)

        # 4. Flatten
        flat = x.flatten(1)

        # 5. Late Fusion (if configured)
        if self.cond_type == "late_fusion" and cond_emb is not None:
            flat = torch.cat([flat, cond_emb], dim=1)

        # 6. Project
        z = self.to_latent(flat)
        return z


# ==========================================
#   DECODER (PixelShuffle ResNet)
# ==========================================


class ConvDecoder(nn.Module):
    """
    Robust Decoder combining ResNet blocks with PixelShuffle upsampling.
    Replaces AdaIN with AdaLN modulation on the latent Z for stability.
    """

    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        latent_dim: int,
        hiddens: List[int] = [64, 128, 256],
        cond_embedding_dim: Optional[int] = None,
        cond_type: Optional[str] = None,
        cond_expansion_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__()

        # Reverse hiddens for decoder: [256, 128, 64]
        hiddens = hiddens[::-1]

        scale_factor = 2 ** len(hiddens)
        self.H_start = H // scale_factor
        self.W_start = W // scale_factor
        self.cond_type = cond_type

        # Conditioning Modulator
        self.conditioner = None
        if cond_type is not None:
            assert cond_embedding_dim is not None
            self.conditioner = AdaLNMLP(latent_dim, cond_embedding_dim)

        self.cond_expansion_type = cond_expansion_type
        dim_to_use = cond_embedding_dim if cond_embedding_dim is not None else 64
        self.expansion_map = nn.ModuleDict(
            {
                "Re": re_expansion(dim_to_use),
                "Ma": ma_expansion(dim_to_use),
                "forcing": forcing_expansion(dim_to_use),
            }
        )

        # 1. Project Latent -> Spatial Volume
        self.flat_features = hiddens[0] * self.H_start * self.W_start
        self.from_latent = sn_linear(latent_dim, self.flat_features)

        # 2. Upsampling ResBlocks
        layers = []
        in_c = hiddens[0]

        for i, out_c in enumerate(hiddens):
            # Step A: Refine at current resolution
            layers.append(PreActResBlock(in_c, in_c, stride=1))

            # Step B: Upsample using PixelShuffle
            layers.append(sn_conv2d(in_c, out_c * 4, kernel_size=3, padding=1))
            layers.append(nn.PixelShuffle(2))
            layers.append(nn.SiLU())

            in_c = out_c

        self.backbone = nn.Sequential(*layers)

        # 3. Final Prediction
        self.final_conv = sn_conv2d(hiddens[-1], C, kernel_size=3, padding=1)

    def _encode_cond(self, cond: Optional[Tensor]) -> Optional[Tensor]:
        if cond is None:
            return None
        if cond.ndim == 1:
            cond = cond.unsqueeze(-1)
        if self.cond_expansion_type in self.expansion_map:
            return self.expansion_map[self.cond_expansion_type](cond)
        return cond

    def forward(self, z: Tensor, cond: Optional[Tensor] = None):
        # 1. Apply Conditioning to Latent Z
        if self.conditioner is not None and cond is not None:
            cond_emb = self._encode_cond(cond)
            if cond_emb is not None:
                z = self.conditioner(z, cond_emb)

        # 2. Expand Latent
        x = self.from_latent(z)
        x = F.silu(x)
        x = x.view(
            -1, x.shape[1] // (self.H_start * self.W_start), self.H_start, self.W_start
        )

        # 3. Upsample
        x = self.backbone(x)

        # 4. To Image
        return self.final_conv(x)


# ==========================================
#   HISTORY ENCODER (Transformer)
# ==========================================


@dataclass
class TransformerConfig:
    num_layers: int = 4
    nhead: int = 8
    ff_mult: int = 4
    max_len: int = 1000
    dropout: float = 0.1


class HistoryEncoder(nn.Module):
    """
    Temporal encoder that aggregates a sequence of frame embeddings.
    Wraps the ConvEncoder backbone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        use_positional_encoding: bool = True,
        transformer_config: TransformerConfig = TransformerConfig(),
    ):
        super().__init__()
        self.backbone = backbone

        if hasattr(backbone, "latent_dim"):
            self.latent_dim = backbone.latent_dim
        else:
            self.latent_dim = 128  # Fallback

        self.norm = nn.LayerNorm(self.latent_dim)
        self.pos_enc = (
            PositionalEncoding(self.latent_dim, max_len=transformer_config.max_len)
            if use_positional_encoding
            else nn.Identity()
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=transformer_config.nhead,
                dim_feedforward=self.latent_dim * transformer_config.ff_mult,
                dropout=transformer_config.dropout,
                batch_first=True,
            ),
            num_layers=transformer_config.num_layers,
        )

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        B, T, C, H, W = x.shape
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")

        features = self.backbone(x_flat, cond=None)

        features = rearrange(features, "(b t) d -> b t d", t=T)
        features = self.norm(features)
        features = self.pos_enc(features)
        out = self.transformer(features)

        return out.mean(dim=1)
