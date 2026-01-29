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
    from .adaptive_layers import AdaGN
    from .rbf import re_expansion, ma_expansion, forcing_expansion
except ImportError:
    # Minimal fallback if local imports fail
    pass

# ==========================================
#   HELPERS & NORMALIZATION
# ==========================================


def sn_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """Spectral Normalized Conv2d. Stabilizes Koopman operator gradients."""
    return nn_utils.spectral_norm(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    )


def sn_linear(in_features, out_features):
    """Spectral Normalized Linear."""
    return nn_utils.spectral_norm(nn.Linear(in_features, out_features))


# def sn_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
#     """Spectral Normalized Conv2d. Stabilizes Koopman operator gradients."""
#     return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

# def sn_linear(in_features, out_features):
#     """Spectral Normalized Linear."""
#     return nn.Linear(in_features, out_features)


# ==========================================
#   ATTENTION MECHANISMS
# ==========================================


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Helps the encoder focus on 'active' fluid regions (vortices) vs background.
    """

    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.channels = channels

        # Channel Attention
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels),
        )

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # Channel Attention
        avg_pool = F.avg_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
        )
        max_pool = F.max_pool2d(
            x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
        )
        channel_att = (
            torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
            .unsqueeze(2)
            .unsqueeze(3)
        )
        x = x * channel_att

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_att = torch.sigmoid(self.conv_spatial(spatial_out))
        return x * spatial_att


# ==========================================
#   RESIDUAL BLOCKS
# ==========================================


class PreActResBlock(nn.Module):
    """Standard Pre-activation ResBlock for Encoder."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.bn1 = nn.GroupNorm(8, in_ch)
        self.conv1 = sn_conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.GroupNorm(8, out_ch)
        self.conv2 = sn_conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = sn_conv2d(
                in_ch, out_ch, kernel_size=1, stride=stride, padding=0
            )

    def forward(self, x):
        x_norm = F.silu(self.bn1(x))
        residual = self.shortcut(x)  # Note: He et al. apply shortcut to x, not x_norm
        out = self.conv1(x_norm)
        out = self.conv2(F.silu(self.bn2(out)))
        return out + residual


class ConditionedResBlock(nn.Module):
    """Decoder ResBlock with Physics Injection (AdaGN)."""

    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        # AdaGN replaces static GroupNorm
        self.gn1 = AdaGN(in_ch, cond_dim)
        self.conv1 = sn_conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.gn2 = AdaGN(out_ch, cond_dim)
        self.conv2 = sn_conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = sn_conv2d(in_ch, out_ch, kernel_size=1, padding=0)

    def forward(self, x, cond):
        # 1. Norm + Condition
        x_norm = F.silu(self.gn1(x, cond))

        # 2. Convolution
        out = self.conv1(x_norm)

        # 3. Norm + Condition
        out = F.silu(self.gn2(out, cond))
        out = self.conv2(out)

        return out + self.shortcut(x)


# ==========================================
#   ENCODER
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
        self.cond_type = cond_type

        # Physics Parameter Expansion (Re -> High Dim)
        dim_to_use = cond_embedding_dim if cond_embedding_dim is not None else 64
        self.expansion_map = nn.ModuleDict(
            {
                "Re": re_expansion(dim_to_use),
                "Ma": ma_expansion(dim_to_use),
                "forcing": forcing_expansion(dim_to_use),
            }
        )
        self.cond_expansion_type = cond_expansion_type

        # 1. Coordinate Injection Setup
        input_channels = C + 2
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
        )
        self.register_buffer("grid", torch.stack([xx, yy], dim=0))

        # 2. Backbone
        self.init_conv = sn_conv2d(input_channels, hiddens[0], kernel_size=3, padding=1)

        layers = []
        in_c = hiddens[0]
        for out_c in hiddens:
            layers.append(PreActResBlock(in_c, out_c, stride=2))  # Downsample
            layers.append(PreActResBlock(out_c, out_c, stride=1))  # Refine
            in_c = out_c
        self.backbone = nn.Sequential(*layers)

        # 3. Bottleneck Attention
        self.attention = CBAM(in_c)

        # 4. Projection
        scale_factor = 2 ** len(hiddens)
        self.H_out = H // scale_factor
        self.W_out = W // scale_factor
        flat_features = in_c * self.H_out * self.W_out

        if self.cond_type == "late_fusion":
            flat_features += dim_to_use

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
        # Coordinate Injection
        grid = self.grid.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, grid], dim=1)

        cond_emb = None
        if self.cond_type is not None and cond is not None:
            cond_emb = self._encode_cond(cond)

        x = self.init_conv(x)
        x = self.backbone(x)

        # Apply Attention at low-res
        x = self.attention(x)

        flat = x.flatten(1)
        if self.cond_type == "late_fusion" and cond_emb is not None:
            flat = torch.cat([flat, cond_emb], dim=1)

        return self.to_latent(flat)


# ==========================================
#   DECODER (Resize-Conv + AdaGN)
# ==========================================


class ConvDecoder(nn.Module):
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
        # Decoder goes from small to large: [256, 128, 64]
        hiddens = hiddens[::-1]

        scale_factor = 2 ** len(hiddens)
        self.H_start = H // scale_factor
        self.W_start = W // scale_factor

        # Physics Expansion Setup
        self.cond_expansion_type = cond_expansion_type
        dim_to_use = cond_embedding_dim if cond_embedding_dim is not None else 64
        self.expansion_map = nn.ModuleDict(
            {
                "Re": re_expansion(dim_to_use),
                "Ma": ma_expansion(dim_to_use),
                "forcing": forcing_expansion(dim_to_use),
            }
        )

        # 1. Expand Latent
        self.flat_features = hiddens[0] * self.H_start * self.W_start
        self.from_latent = sn_linear(latent_dim, self.flat_features)

        # 2. Build Layers (Manual list to handle conditional forwarding)
        self.layers = nn.ModuleList()
        in_c = hiddens[0]

        # If no condition is provided, we need a dummy dim for AdaGN
        self.cond_dim = dim_to_use

        for out_c in hiddens:
            # Resize-Conv Block
            # 1. Refine (Conditioned)
            self.layers.append(ConditionedResBlock(in_c, in_c, self.cond_dim))

            # 2. Upsample (Nearest + Conv) -> Eliminates Checkerboard
            upsample_block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                sn_conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.layers.append(upsample_block)
            in_c = out_c

        self.final_conv = sn_conv2d(in_c, C, kernel_size=3, padding=1)

    def _encode_cond(self, cond: Optional[Tensor]) -> Tensor:
        # Defaults to zero vector if cond is None, to satisfy AdaGN
        if cond is None:
            # Create a zero condition on the fly?
            # Ideally this shouldn't happen if cond_type is set.
            # But for safety, we return a zero tensor if strictly needed,
            # though usually the caller ensures cond exists.
            return torch.zeros(1, self.cond_dim).to(next(self.parameters()).device)

        if cond.ndim == 1:
            cond = cond.unsqueeze(-1)

        if self.cond_expansion_type in self.expansion_map:
            return self.expansion_map[self.cond_expansion_type](cond)

        # If standard scalar, project it if dimensions don't match or use as is
        # Note: AdaGN expects [B, cond_dim].
        return cond

    def forward(self, z: Tensor, cond: Optional[Tensor] = None):
        # Prepare Condition
        cond_emb = self._encode_cond(cond)
        # Ensure batch size matches if we generated a dummy
        if cond_emb.shape[0] != z.shape[0]:
            cond_emb = cond_emb.expand(z.shape[0], -1)

        # 1. Expand Z
        x = self.from_latent(z)
        x = F.silu(x)
        x = x.view(
            -1,
            self.flat_features // (self.H_start * self.W_start),
            self.H_start,
            self.W_start,
        )

        # 2. Upsample Loop
        for layer in self.layers:
            if isinstance(layer, ConditionedResBlock):
                x = layer(x, cond_emb)
            else:
                x = layer(x)

        return self.final_conv(x)


# ==========================================
#   HISTORY ENCODER
# ==========================================


@dataclass
class TransformerConfig:
    num_layers: int = 4
    nhead: int = 8
    ff_mult: int = 4
    max_len: int = 1000
    dropout: float = 0.1


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        use_positional_encoding: bool = True,
        transformer_config: TransformerConfig = TransformerConfig(),
    ):
        super().__init__()
        self.backbone = backbone
        self.latent_dim = getattr(backbone, "latent_dim", 128)

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

        # Encoder likely doesn't need temporal cond, or it's handled via broadcasting
        # We pass cond if the backbone supports it
        features = self.backbone(x_flat, cond=None)

        features = rearrange(features, "(b t) d -> b t d", t=T)
        features = self.norm(features)
        features = self.pos_enc(features)
        out = self.transformer(features)

        # Average pooling over time for the history context
        return out.mean(dim=1)
