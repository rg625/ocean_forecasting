from itertools import pairwise
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from torch import Tensor
from typing import List, Union, Tuple, Any, Optional
import torch.nn.functional as F
from dataclasses import dataclass

from .fourier import (
    GaussianFourierFeatureTransform,
    PositionalEncoding,
)
from .rbf import (
    re_expansion,
    ma_expansion,
    forcing_expansion,
)
from .adaptive_layers import (
    AdaIN,
    AdaLNConv,
    AdaLNMLP,
)

# ==========================================
#   BLOCKS & LAYERS
# ==========================================


class ConvBlock(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        block_size: int = 1,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        decoder_block: bool = False,
        use_checkpoint: bool = False,
        cond_type: Optional[str] = None,
        cond_embedding_dim: Optional[int] = None,
        **conv_kwargs: Any,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.cond_type = cond_type
        self.stack = nn.ModuleList()

        layers = []
        if not decoder_block:
            layers.append(nn.Conv2d(C_in, C_out, kernel_size, **conv_kwargs))
            if self.cond_type == "adaln":
                assert (
                    cond_embedding_dim is not None
                ), f"Expected conditional embedding to be int but got {type(cond_embedding_dim)} instead"
                layers.append(AdaLNConv(C_out=C_out, cond_dim=cond_embedding_dim))
            layers.append(nn.SiLU())

            for _ in range(block_size - 1):
                layers.append(nn.Conv2d(C_out, C_out, kernel_size, **conv_kwargs))
                if self.cond_type == "adaln":
                    assert (
                        cond_embedding_dim is not None
                    ), f"Expected conditional embedding to be int but got {type(cond_embedding_dim)} instead"
                    layers.append(AdaLNConv(C_out=C_out, cond_dim=cond_embedding_dim))
                layers.append(nn.SiLU())
        else:
            for i in range(block_size - 1):
                C_intermediate = C_in if i == 0 else C_in
                layers.append(
                    nn.Conv2d(C_intermediate, C_in, kernel_size, **conv_kwargs)
                )
                if self.cond_type == "adaln":
                    assert (
                        cond_embedding_dim is not None
                    ), f"Expected conditional embedding to be int but got {type(cond_embedding_dim)} instead"
                    layers.append(AdaLNConv(C_out=C_in, cond_dim=cond_embedding_dim))
                layers.append(nn.SiLU())
            layers.append(nn.Conv2d(C_in, C_out, kernel_size, **conv_kwargs))

        self.stack = nn.ModuleList(layers)

    def forward(self, x: Tensor, cond_emb: Optional[Tensor] = None) -> Tensor:
        if self.use_checkpoint:
            return checkpoint(
                lambda t: self._forward(t, cond_emb), x, use_reentrant=True
            )
        else:
            return self._forward(x, cond_emb)

    def _forward(self, x: Tensor, cond_emb: Optional[Tensor] = None) -> Tensor:
        for module in self.stack:
            if isinstance(module, AdaLNConv):
                if cond_emb is None:
                    raise ValueError("AdaLNConv requires cond_emb.")
                x = module(x, cond_emb)
            else:
                x = module(x)
        return x


class MappingNetwork(nn.Module):
    """Latent-to-style mapping network (similar to StyleGAN)."""

    def __init__(self, latent_dim, style_dim, n_layers=4):
        super().__init__()
        layers = [nn.Linear(latent_dim, style_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(style_dim, style_dim), nn.SiLU()])
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)


class PixelShuffleStyledBlock(nn.Module):
    """
    Upsamples using PixelShuffle for maximum sharpness.
    Flow: Conv (expand channels) -> AdaIN -> Act -> PixelShuffle
    """

    def __init__(self, C_in, C_out, style_dim, kernel_size=3, upsample=False):
        super().__init__()
        self.upsample = upsample

        if upsample:
            # PixelShuffle with upscale_factor=2 reduces channels by factor of 4.
            # So we need to output 4 * C_out channels from the conv.
            self.out_channels_conv = C_out * 4
            self.pixel_shuffle = nn.PixelShuffle(2)
        else:
            self.out_channels_conv = C_out
            self.pixel_shuffle = nn.Identity()

        self.conv = nn.Conv2d(
            C_in,
            self.out_channels_conv,
            kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
        )

        # Modulate the HIGH dimensional representation before shuffling
        self.mod = AdaIN(style_dim, self.out_channels_conv)
        self.act = nn.SiLU()

    def forward(self, x, w):
        x = self.conv(x)
        x = self.mod(x, w)
        x = self.act(x)
        if self.upsample:
            x = self.pixel_shuffle(x)
        return x


# ==========================================
#   ENCODERS & DECODERS
# ==========================================


class BaseEncoderDecoder(nn.Module):
    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        latent_dim: int,
        hiddens: List[int],
        block_size: int = 1,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        is_encoder: bool = True,
        use_checkpoint: bool = False,
        cond_embedding_dim: Optional[int] = None,
        cond_type: Optional[str] = None,
        cond_expansion_type: Optional[str] = None,
        **conv_kwargs,
    ):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.D = latent_dim
        self.hiddens = hiddens
        self.is_encoder = is_encoder
        self.use_checkpoint = use_checkpoint
        self.cond_embedding_dim = cond_embedding_dim
        self.cond_type = cond_type
        self.cond_expansion_type = cond_expansion_type

        # Compute the output dimensions after pooling
        self.n_pools = len(hiddens)
        if H % (2**self.n_pools) != 0 or W % (2**self.n_pools) != 0:
            raise ValueError(
                f"Input dimensions (H={H}, W={W}) must be divisible by 2^{self.n_pools}."
            )
        self.H_out = H // (2 ** (self.n_pools))
        self.W_out = W // (2 ** (self.n_pools))

        dim_to_use = cond_embedding_dim if cond_embedding_dim is not None else 64

        self.expansion_map = nn.ModuleDict(
            {
                "Re": re_expansion(dim_to_use),
                "Ma": ma_expansion(dim_to_use),
                "forcing": forcing_expansion(dim_to_use),
            }
        )

        if is_encoder:
            encoder_in_features = hiddens[-1] * self.H_out * self.W_out
            if self.cond_type == "late_fusion":
                assert cond_embedding_dim is not None
                encoder_in_features += cond_embedding_dim
            self.linear = nn.Linear(encoder_in_features, latent_dim)
        else:
            self.linear = nn.Linear(latent_dim, hiddens[-1] * self.H_out * self.W_out)

        self.layers = self._build_layers(block_size, kernel_size, conv_kwargs)

    def _encode_cond(self, cond: Optional[Tensor]) -> Optional[Tensor]:
        if cond is None:
            return None
        if cond.ndim == 1:
            cond = cond.unsqueeze(-1)

        if self.cond_expansion_type is None or self.cond_expansion_type == "none":
            if self.cond_type is not None:
                if cond.shape[1] == 1:
                    assert self.cond_embedding_dim is not None
                    return F.linear(
                        cond, torch.eye(self.cond_embedding_dim, 1, device=cond.device)
                    )
                else:
                    raise ValueError(
                        "cond_expansion_type must be set for non-scalar cond."
                    )
            return None

        if self.cond_expansion_type not in self.expansion_map:
            raise KeyError(
                f"Unknown cond_expansion_type: '{self.cond_expansion_type}'."
            )

        expansion_func = self.expansion_map[self.cond_expansion_type]
        if cond.ndim > 2:
            cond = cond.view(cond.shape[0], -1)
        cond_encoded = expansion_func(cond)
        return cond_encoded

    def _build_layers(self, block_size, kernel_size, conv_kwargs):
        layers = nn.ModuleList()
        conv_block_args = {
            "block_size": block_size,
            "kernel_size": kernel_size,
            "use_checkpoint": self.use_checkpoint,
            "cond_type": self.cond_type,
            "cond_embedding_dim": self.cond_embedding_dim,
            **conv_kwargs,
        }

        if self.is_encoder:
            layers.append(
                ConvBlock(
                    self.C, self.hiddens[0], decoder_block=False, **conv_block_args
                )
            )
            layers.append(nn.MaxPool2d(kernel_size=2))
            for C_n, C_np1 in pairwise(self.hiddens):
                layers.append(
                    ConvBlock(C_n, C_np1, decoder_block=False, **conv_block_args)
                )
                layers.append(nn.MaxPool2d(kernel_size=2))
        else:
            # Note: This is the legacy decoder path.
            # The new ConvDecoder class below should be used instead for better results.
            for C_np1, C_n in pairwise(self.hiddens[::-1]):
                layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))
                layers.append(
                    ConvBlock(C_np1, C_n, decoder_block=True, **conv_block_args)
                )
            layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            layers.append(
                ConvBlock(
                    self.hiddens[0], self.C, decoder_block=True, **conv_block_args
                )
            )
        return layers

    def forward(self, x: Tensor, cond: Optional[Tensor] = None):
        cond_emb = None
        if self.cond_type is not None:
            if cond is None:
                raise ValueError(f"Condition tensor required for '{self.cond_type}'")
            cond_emb = self._encode_cond(cond)

        if self.is_encoder:
            for layer in self.layers:
                if isinstance(layer, ConvBlock) and self.cond_type == "adaln":
                    x = layer(x, cond_emb)
                else:
                    x = layer(x)
            out = rearrange(x, "b c h w -> b (c h w)")
            if self.cond_type == "late_fusion":
                out = torch.cat([out, cond_emb], dim=1)
            return self.linear(out)
        else:
            out = self.linear(x)
            out = rearrange(
                out,
                "b (c h w) -> b c h w",
                c=self.hiddens[-1],
                h=self.H_out,
                w=self.W_out,
            )
            for layer in self.layers:
                if isinstance(layer, ConvBlock) and self.cond_type == "adaln":
                    out = layer(out, cond_emb)
                else:
                    out = layer(out)
            return out


class ConvEncoder(BaseEncoderDecoder):
    def __init__(
        self, C: int, H: int, W: int, latent_dim: int, hiddens: List[int], **kwargs
    ):
        super().__init__(C, H, W, latent_dim, hiddens, is_encoder=True, **kwargs)


@dataclass
class TransformerConfig:
    """Configuration dataclass for the TransformerEncoder in HistoryEncoder."""

    num_layers: int = 4  # Number of transformer encoder layers
    nhead: int = 8  # Number of attention heads
    ff_mult: int = 4  # Multiplier for the feed-forward layer dimension
    max_len: int = 1000  # Maximum sequence length for positional encoding
    dropout: float = 0.1  # Dropout rate


class HistoryEncoder(nn.Module):
    """
    Encodes a sequence of images into a single latent vector using a shared backbone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        use_positional_encoding: bool = True,
        transformer_config: TransformerConfig = TransformerConfig(),
    ):
        super().__init__()
        # Use the provided backbone (ConvEncoder) via composition instead of inheritance
        # This ensures weights are shared with the present_encoder.
        self.backbone = backbone

        # Infer latent dim from the backbone (BaseEncoderDecoder stores it as D)
        if hasattr(backbone, "D"):
            self.latent_dim = backbone.D
        elif hasattr(backbone, "latent_dim"):
            self.latent_dim = backbone.latent_dim
        else:
            # Fallback check on linear layer
            self.latent_dim = backbone.linear.out_features

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
        """
        Args:
            x (Tensor): Input tensor of image frames. Shape: (B, T, C, H, W).
            cond (Optional[Tensor]): Optional conditioning. Shape: (B, T).
        """
        B, T, C, H, W = x.shape
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")

        # Handle conditioning logic based on the backbone's configuration
        cond_expanded = None
        # Access cond_type from backbone (BaseEncoderDecoder)
        backbone_cond_type = getattr(self.backbone, "cond_type", None)

        if backbone_cond_type is not None:
            if cond is None:
                raise ValueError(
                    f"Condition tensor must be provided for conditioning type '{backbone_cond_type}'"
                )
            if cond.ndim != 2 or cond.shape != (B, T):
                raise ValueError(
                    f"Expected Condition tensor of shape (B, T) = ({B}, {T}), but got {cond.shape}"
                )
            # Flatten cond from (B, T) to (B*T,)
            cond_expanded = cond.reshape(-1)

        # Pass through the shared backbone
        features = self.backbone(x_flat, cond=cond_expanded)

        # Un-flatten features
        features = rearrange(features, "(b t) d -> b t d", t=T)
        features = self.norm(features)
        features = self.pos_enc(features)
        out = self.transformer(features)

        return out.mean(dim=1)


class ConvDecoder(nn.Module):
    """
    State-of-the-art Decoder combining:
    1. Coordinate Injection (Spatial Awareness)
    2. Fourier Features (High-frequency details / Spectral Bias mitigation)
    3. PixelShuffle (Sharp upsampling)
    4. AdaIN (Style modulation)
    """

    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        latent_dim: int,
        hiddens: List[int],
        block_size: int = 1,  # Kept for API compatibility
        kernel_size: int = 3,
        use_checkpoint: bool = False,
        cond_embedding_dim: Optional[int] = None,
        cond_type: Optional[str] = None,
        style_dim: Optional[int] = None,
        fourier_scale: float = 2.0,
        fourier_mapping_size: int = 64,
        **conv_kwargs,
    ):
        super().__init__()

        # --- Dimensions ---
        n_downsamples = len(hiddens)
        self.H_start = H // (2**n_downsamples)
        self.W_start = W // (2**n_downsamples)
        style_dim = style_dim or latent_dim

        # --- Modules ---
        self.mapping = MappingNetwork(latent_dim, style_dim)

        # 1. Fourier Feature Transform
        # Output channels will be mapping_size * 2 (sin + cos)
        self.fourier = GaussianFourierFeatureTransform(
            in_channels=2, mapping_size=fourier_mapping_size, scale=fourier_scale
        )
        fourier_dim = fourier_mapping_size * 2

        # 2. Latent Projection
        # We project z to match the spatial resolution
        # This replaces the initial "constant" learned in StyleGAN
        self.z_to_spatial = nn.Linear(
            latent_dim, hiddens[-1] * self.H_start * self.W_start
        )

        # 3. Base Coordinate Grid (Fixed)
        # Create a grid from -1 to 1.
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, self.H_start),
            torch.linspace(-1, 1, self.W_start),
            indexing="ij",
        )
        # Shape: [2, H, W]
        self.register_buffer("grid", torch.stack([xx, yy], dim=0))

        # 4. Convolutional Blocks
        layers = []

        # Input channels = (Z spatial features) + (Fourier Grid features)
        C_in = hiddens[-1] + fourier_dim

        # First block: No upsample, just processing the injected features
        layers.append(
            PixelShuffleStyledBlock(C_in, hiddens[-1], style_dim, upsample=False)
        )

        # Upsampling blocks
        # Iterate in reverse: [256, 128, 64]
        current_C = hiddens[-1]
        for C_out in reversed(hiddens):
            layers.append(
                PixelShuffleStyledBlock(current_C, C_out, style_dim, upsample=True)
            )
            current_C = C_out

        self.conv_layers = nn.ModuleList(layers)

        # Final projection to RGB/Physics variables
        self.to_rgb = nn.Conv2d(hiddens[0], C, kernel_size=1)

        # Optional conditioning (e.g. Reynolds number modulation on the latent z)
        self.conditioner = None
        self.cond_type = cond_type
        if cond_type is not None:
            if cond_embedding_dim is None:
                raise ValueError("'cond_embedding_dim' must be provided.")
            self.conditioner = AdaLNMLP(latent_dim, cond_embedding_dim)

    def forward(self, z, cond: Optional[torch.Tensor] = None):
        # 1. Conditioning on Z
        if self.conditioner is not None and cond is not None:
            z = self.conditioner(z, cond)

        # 2. Get Style Vector
        w = self.mapping(z)

        # 3. Project Latent Z to Spatial
        # [B, Latent] -> [B, C*H*W] -> [B, C, H, W]
        x_z = self.z_to_spatial(z)
        x_z = x_z.view(
            -1,
            self.conv_layers[0].conv.in_channels - self.fourier.B.shape[1] * 2,
            self.H_start,
            self.W_start,
        )

        # 4. Create Fourier Grid
        # Expand grid to batch size: [B, 2, H, W]
        grid_batch = self.grid.unsqueeze(0).expand(z.shape[0], -1, -1, -1)
        # Apply Fourier Transform: [B, 128, H, W]
        x_coords = self.fourier(grid_batch)

        # 5. Concatenate Latent + Fourier Coords
        # The conv layer sees both the physics state and the precise location embedding
        x = torch.cat([x_z, x_coords], dim=1)

        # 6. Decode
        for layer in self.conv_layers:
            x = layer(x, w)

        return self.to_rgb(x)
