from itertools import pairwise
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from torch import Tensor
from typing import List, Union, Tuple, Any, Optional, Dict
import torch.nn.functional as F

# --- NEW IMPORTS ---
from .fourier import re_expansion, ma_expansion, forcing_expansion

# --- END NEW IMPORTS ---


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


class StyledConv(nn.Module):
    """Styled convolution block with optional upsampling."""

    def __init__(
        self, C_in, C_out, style_dim, kernel_size=3, upsample=False, use_deconv=True
    ):
        super().__init__()
        self.upsample = None
        if upsample:
            if use_deconv:
                # Learnable upsampling (sharper)
                self.upsample = nn.ConvTranspose2d(
                    C_in, C_in, kernel_size=4, stride=2, padding=1
                )
            else:
                # Smooth upsampling
                self.upsample = nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=False
                )
        self.conv = nn.Conv2d(
            C_in, C_out, kernel_size, padding=kernel_size // 2, padding_mode="circular"
        )
        self.mod = AdaIN(style_dim, C_out)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.mod(x, w)
        x = self.act(x)
        return x


class MappingNetwork(nn.Module):
    """Latent-to-style mapping network."""

    def __init__(self, latent_dim, style_dim, n_layers=4):
        super().__init__()
        layers = [nn.Linear(latent_dim, style_dim), nn.LeakyReLU(0.2)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(style_dim, style_dim), nn.LeakyReLU(0.2)])
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)


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
        """
        A modular convolutional block consisting of convolutional layers followed by ReLU activations.

        Parameters:
            C_in: int
                Number of input channels.
            C_out: int
                Number of output channels.
            block_size: int
                Number of convolutional layers in the block.
            kernel_size: int
                Size of the convolution kernel.
            decoder_block: bool
                If True, the block is used in a decoder, and layer configurations are adjusted accordingly.
            use_checkpoint: bool
                If True, enables gradient checkpointing for the block.
            cond_type: str
                Choose between AdaLN, Late Fusion and None.
            cond_embedding_dim: int
                Embedding dimension for Reynolds number in case of conditioning.
            conv_kwargs: dict
                Additional arguments for nn.Conv2d.
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint  # Store the checkpointing flag
        self.cond_type = cond_type
        self.stack = nn.ModuleList()

        layers = []
        if not decoder_block:
            # First layer
            layers.append(nn.Conv2d(C_in, C_out, kernel_size, **conv_kwargs))
            if self.cond_type == "adaln":
                assert (cond_embedding_dim is not None) and (
                    isinstance(cond_embedding_dim, int)
                ), f"cond_embedding_dim must be provided for adaln as int but got type {type(cond_embedding_dim)}"
                layers.append(AdaLNConv(C_out, cond_embedding_dim))
            layers.append(nn.ReLU())

            # Subsequent layers
            for _ in range(block_size - 1):
                layers.append(nn.Conv2d(C_out, C_out, kernel_size, **conv_kwargs))
                if self.cond_type == "adaln":
                    assert (cond_embedding_dim is not None) and (
                        isinstance(cond_embedding_dim, int)
                    ), f"cond_embedding_dim must be provided for adaln as int but got type {type(cond_embedding_dim)}"
                    layers.append(AdaLNConv(C_out, cond_embedding_dim))
                layers.append(nn.ReLU())
        else:
            # Initial layers
            for i in range(block_size - 1):
                C_intermediate = C_in if i == 0 else C_in
                layers.append(
                    nn.Conv2d(C_intermediate, C_in, kernel_size, **conv_kwargs)
                )
                if self.cond_type == "adaln":
                    assert (
                        cond_embedding_dim is not None
                    ), "cond_embedding_dim must be provided for adaln"
                    layers.append(AdaLNConv(C_in, cond_embedding_dim))
                layers.append(nn.ReLU())

            # Output layer (no activation after this one)
            layers.append(nn.Conv2d(C_in, C_out, kernel_size, **conv_kwargs))

        self.stack = nn.ModuleList(layers)

    def forward(self, x: Tensor, cond_emb: Optional[Tensor] = None) -> Tensor:
        if self.use_checkpoint:
            # Checkpoint doesn't easily support extra args, so we wrap the call
            return checkpoint(
                lambda t: self._forward(t, cond_emb), x, use_reentrant=True
            )
        else:
            return self._forward(x, cond_emb)

    def _forward(self, x: Tensor, cond_emb: Optional[Tensor] = None) -> Tensor:
        for module in self.stack:
            if isinstance(module, AdaLNConv):
                # AdaLNConv requires the conditioning embedding
                if cond_emb is None:
                    raise ValueError(
                        "AdaLNConv layer requires cond_emb, but it was not provided."
                    )
                x = module(x, cond_emb)
            else:
                x = module(x)
        return x


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
        cond_type: Optional[str] = None,  # Options: None, "late_fusion", "adaln"
        cond_expansion_type: Optional[str] = None,
        **conv_kwargs,
    ):
        """
        Base class for both encoder and decoder blocks.

        Parameters:
            C: int
                Number of input/output channels.
            H: int
                Input height dimension.
            W: int
                Input width dimension.
            latent_dim: int
                Latent dimensionality.
            hiddens: list of int
                List of hidden dimensions for each block.
            block_size: int
                Number of convolutional layers in each block.
            kernel_size: int
                Size of the convolutional kernel.
            is_encoder: bool
                Specifies whether the block is an encoder (True) or decoder (False).
            use_checkpoint: bool
                If True, enables gradient checkpointing for convolutional blocks.
            cond_embedding_dim: int
                Optional Reynolds number conditioning dimension.
            cond_type: str
                Optional Reynolds number conditioning mode.
            cond_expansion_type: str
                Specifies the type of fourier expansion ('re', 'ma', 'forcing').
            conv_kwargs: dict
                Additional arguments for nn.Conv2d.
        """
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
        self.cond_expansion_type = cond_expansion_type  # <-- NEW

        # Compute the output dimensions after pooling
        if cond_type not in [None, "late_fusion", "adaln"]:
            raise ValueError(f"Unknown cond_type: {cond_type}")
        if cond_type is not None and cond_embedding_dim is None:
            raise ValueError(f"{cond_type} requires cond_embedding_dim to be set.")

        self.n_pools = len(hiddens)
        if H % (2**self.n_pools) != 0 or W % (2**self.n_pools) != 0:
            raise ValueError(
                f"Input dimensions (H={H}, W={W}) must be divisible by 2^{self.n_pools} due to pooling."
            )
        self.H_out = H // (2 ** (self.n_pools))
        self.W_out = W // (2 ** (self.n_pools))

        # --- REMOVED old re_embedding ---
        # self.re_embedding = nn.Sequential(...)

        # --- NEW: Add expansion map (same as in modules.py) ---
        self.expansion_map: Dict[str, nn.Module] = {
            "Re": re_expansion,
            "Ma": ma_expansion,
            "forcing": forcing_expansion,
        }
        # --- END NEW ---

        # Define the linear layer
        if is_encoder:
            encoder_in_features = hiddens[-1] * self.H_out * self.W_out
            if self.cond_type == "late_fusion":
                assert cond_embedding_dim is not None
                encoder_in_features += cond_embedding_dim
            self.linear = nn.Linear(encoder_in_features, latent_dim)
        else:
            self.linear = nn.Linear(latent_dim, hiddens[-1] * self.H_out * self.W_out)

        # Build convolutional layers
        self.layers = self._build_layers(block_size, kernel_size, conv_kwargs)

    def _encode_cond(self, cond: Optional[Tensor]) -> Optional[Tensor]:
        """
        Fourier-expand the condition tensor using the configured expansion type.
        """
        if cond is None:
            return None

        # This module (ConvEncoder) can be called by HistoryEncoder, which
        # flattens the time dimension, resulting in cond.ndim == 1.
        if cond.ndim == 1:
            cond = cond.unsqueeze(-1)  # (B*T,) -> (B*T, 1)

        if self.cond_expansion_type is None or self.cond_expansion_type == "none":
            if self.cond_type is not None:
                print(f"cond: {cond}")
                # Fallback to simple linear projection if no expansion is specified
                # This supports the old behavior of embedding a scalar
                if cond.shape[1] == 1:
                    assert self.cond_embedding_dim is not None
                    # Simple linear projection as a fallback
                    # --- FIX: Swapped arguments to create (C_out, C_in) -> (128, 1) ---
                    return F.linear(
                        cond, torch.eye(self.cond_embedding_dim, 1, device=cond.device)
                    )
                else:
                    # We have a tensor, but don't know how to expand it.
                    raise ValueError(
                        "cond_expansion_type must be set (e.g., 're', 'ma', 'forcing') "
                        f"for cond_type '{self.cond_type}' when cond is not a scalar."
                    )
            return None  # No conditioning

        if self.cond_expansion_type not in self.expansion_map:
            raise KeyError(
                f"Unknown cond_expansion_type: '{self.cond_expansion_type}'. "
                f"Available types are: {list(self.expansion_map.keys())}"
            )

        # Look up the correct expansion function
        expansion_func = self.expansion_map[self.cond_expansion_type]

        if cond.ndim > 2:
            cond = cond.view(cond.shape[0], -1)  # Flatten

        # Use the dynamically selected function
        assert self.cond_embedding_dim is not None
        cond_encoded = expansion_func(cond, d=self.cond_embedding_dim)

        return cond_encoded.squeeze(-2)  # (B, D) or (B*T, D)

    def _build_layers(
        self,
        block_size: int,
        kernel_size: Union[int, tuple[int, int]],
        conv_kwargs: dict,
    ):
        """
        Build the layers for the encoder or decoder.

        Parameters:
            block_size: int
                Number of convolutional layers in each block.
            kernel_size: int
                Size of the convolutional kernel.
            conv_kwargs: dict
                Additional arguments for nn.Conv2d.

        Returns:
            nn.Sequential: Sequential container of layers.
        """

        # Must return nn.ModuleList to allow passing extra args in forward
        layers = nn.ModuleList()
        # Common args for all ConvBlocks
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
        # 1. Compute condition embedding if needed
        cond_emb = None
        if self.cond_type is not None:
            if cond is None:
                raise ValueError(
                    f"Condition tensor `cond` must be provided for conditioning type '{self.cond_type}'"
                )
            # Use the new generic method
            cond_emb = self._encode_cond(cond)

        # 2. Apply layers based on role (encoder/decoder)
        if self.is_encoder:
            # Pass through convolutional layers
            for layer in self.layers:
                if isinstance(layer, ConvBlock) and self.cond_type == "adaln":
                    x = layer(x, cond_emb)
                else:
                    x = layer(x)

            # Flatten for linear layer
            out = rearrange(x, "b c h w -> b (c h w)")

            # Apply late fusion if configured
            if self.cond_type == "late_fusion":
                assert cond_emb is not None
                out = torch.cat([out, cond_emb], dim=1)

            return self.linear(out)
        else:  # Decoder
            # Apply linear layer and unflatten
            out = self.linear(x)
            out = rearrange(
                out,
                "b (c h w) -> b c h w",
                c=self.hiddens[-1],
                h=self.H_out,
                w=self.W_out,
            )
            # Pass through convolutional layers
            for layer in self.layers:
                if isinstance(layer, ConvBlock) and self.cond_type == "adaln":
                    out = layer(out, cond_emb)
                else:
                    out = layer(out)
            return out


class ConvEncoder(BaseEncoderDecoder):
    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        latent_dim: int,
        hiddens: List[int],
        block_size: int = 1,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        use_checkpoint: bool = False,
        cond_embedding_dim: Optional[int] = None,
        cond_type: Optional[str] = None,
        cond_expansion_type: Optional[str] = None,
        **conv_kwargs,
    ):
        super().__init__(
            C,
            H,
            W,
            latent_dim,
            hiddens,
            block_size=block_size,
            kernel_size=kernel_size,
            is_encoder=True,
            use_checkpoint=use_checkpoint,
            cond_embedding_dim=cond_embedding_dim,
            cond_type=cond_type,
            cond_expansion_type=cond_expansion_type,
            **conv_kwargs,
        )


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

        # padding_mode="circular" is crucial for periodic boundaries in CFD,
        # use "zeros" or "replicate" if you have hard walls.
        self.conv = nn.Conv2d(
            C_in,
            self.out_channels_conv,
            kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
        )

        # We modulate the HIGH dimensional representation before shuffling
        self.mod = AdaIN(style_dim, self.out_channels_conv)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        x = self.conv(x)
        x = self.mod(x, w)
        x = self.act(x)
        if self.upsample:
            x = self.pixel_shuffle(x)
        return x


class ConvDecoder(nn.Module):
    """
    Improved Decoder with Spatial Awareness (Coordinate Injection)
    and PixelShuffle for sharp detail reconstruction.
    """

    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        latent_dim: int,
        hiddens: List[int],
        block_size: int = 1,  # Kept for API compatibility, usually 1 for StyleGAN types
        kernel_size: int = 3,
        use_checkpoint: bool = False,
        cond_embedding_dim: Optional[int] = None,
        cond_type: Optional[str] = None,
        style_dim: Optional[int] = None,
        **conv_kwargs,
    ):
        super().__init__()

        n_downsamples = len(hiddens)
        # Calculate starting spatial resolution
        self.H_start = H // (2**n_downsamples)
        self.W_start = W // (2**n_downsamples)

        style_dim = style_dim or latent_dim
        self.mapping = MappingNetwork(latent_dim, style_dim)

        # --- IMPROVEMENT 1: Spatial Injection Setup ---
        # Instead of learning a constant, we project z to a feature map.
        self.z_to_spatial = nn.Linear(
            latent_dim, hiddens[-1] * self.H_start * self.W_start
        )

        # We also create a fixed coordinate grid buffer (not a learnable parameter)
        # Shape: [2, H_start, W_start]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, self.H_start),
            torch.linspace(-1, 1, self.W_start),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=0)
        self.register_buffer("grid", grid)

        # --- IMPROVEMENT 2: PixelShuffle Blocks ---
        layers = []

        # The first block receives:
        #   (Projected Z features) + (2 Coordinate Channels)
        C_in = hiddens[-1] + 2

        # First block: No upsample, just processing the injected features
        layers.append(
            PixelShuffleStyledBlock(C_in, hiddens[-1], style_dim, upsample=False)
        )

        current_C = hiddens[-1]

        # Upsampling blocks
        for i, C_out in enumerate(reversed(hiddens)):
            # We use upsampling for all blocks except the very last refinement (optional)
            # or strictly follow hiddens structure.
            # Assuming hiddens are [64, 128, 256] -> we go 256->128->64

            # Note: PixelShuffleBlock handles the channel expansion internally
            layers.append(
                PixelShuffleStyledBlock(current_C, C_out, style_dim, upsample=True)
            )
            current_C = C_out

        self.conv_layers = nn.ModuleList(layers)

        # Final projection to RGB/Physics variables
        self.to_rgb = nn.Conv2d(hiddens[0], C, kernel_size=1, padding=0)

        # Optional conditioning (Same as before)
        self.conditioner = None
        self.cond_type = cond_type
        if cond_type is not None:
            if cond_embedding_dim is None:
                raise ValueError("'cond_embedding_dim' must be provided.")
            self.conditioner = AdaLNMLP(latent_dim, cond_embedding_dim)

    def forward(self, z, cond: Optional[torch.Tensor] = None):
        # 1. Conditioning
        if self.conditioner is not None and cond is not None:
            z = self.conditioner(z, cond)

        # 2. Get Style Vector
        w = self.mapping(z)

        # 3. SPATIAL INJECTION (The "Placement" Fix)
        # Project z to spatial dimensions: [B, C*H*W] -> [B, C, H, W]
        x = self.z_to_spatial(z)
        x = x.view(
            -1, self.conv_layers[0].conv.in_channels - 2, self.H_start, self.W_start
        )

        # Create coordinate batch: [B, 2, H, W]
        grid_batch = self.grid.unsqueeze(0).expand(x.shape[0], -1, -1, -1)

        # Concatenate: [B, C+2, H, W]
        # This tells the conv layer EXACTLY where each pixel is.
        x = torch.cat([x, grid_batch], dim=1)

        # 4. Upsample with PixelShuffle (The "Crispness" Fix)
        for layer in self.conv_layers:
            x = layer(x, w)

        return self.to_rgb(x)


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
