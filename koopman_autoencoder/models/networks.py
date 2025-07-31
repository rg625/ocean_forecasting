# models/networks.py

from itertools import pairwise
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from torch import Tensor
from typing import List, Union, Tuple, Any, Optional, Literal
from dataclasses import dataclass


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


class ConvBlock(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        block_size: int = 1,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        decoder_block: bool = False,
        use_checkpoint: bool = False,
        re_cond_type: Optional[str] = None,
        re_embedding_dim: Optional[int] = None,
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
            re_cond_type: str
                Choose between AdaLN, Late Fusion and None.
            re_embedding_dim: int
                Embedding dimension for Reynolds number in case of conditioning.
            conv_kwargs: dict
                Additional arguments for nn.Conv2d.
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint  # Store the checkpointing flag
        self.re_cond_type = re_cond_type
        self.stack = nn.ModuleList()

        layers = []
        if not decoder_block:
            # First layer
            layers.append(nn.Conv2d(C_in, C_out, kernel_size, **conv_kwargs))
            if self.re_cond_type == "adaln":
                assert (re_embedding_dim is not None) and (
                    isinstance(re_embedding_dim, int)
                ), f"re_embedding_dim must be provided for adaln as int but got type {type(re_embedding_dim)}"
                layers.append(AdaLNConv(C_out, re_embedding_dim))
            layers.append(nn.ReLU())

            # Subsequent layers
            for _ in range(block_size - 1):
                layers.append(nn.Conv2d(C_out, C_out, kernel_size, **conv_kwargs))
                if self.re_cond_type == "adaln":
                    assert (re_embedding_dim is not None) and (
                        isinstance(re_embedding_dim, int)
                    ), f"re_embedding_dim must be provided for adaln as int but got type {type(re_embedding_dim)}"
                    layers.append(AdaLNConv(C_out, re_embedding_dim))
                layers.append(nn.ReLU())
        else:
            # Initial layers
            for i in range(block_size - 1):
                C_intermediate = C_in if i == 0 else C_in
                layers.append(
                    nn.Conv2d(C_intermediate, C_in, kernel_size, **conv_kwargs)
                )
                if self.re_cond_type == "adaln":
                    assert (
                        re_embedding_dim is not None
                    ), "re_embedding_dim must be provided for adaln"
                    layers.append(AdaLNConv(C_in, re_embedding_dim))
                layers.append(nn.ReLU())

            # Output layer (no activation after this one)
            layers.append(nn.Conv2d(C_in, C_out, kernel_size, **conv_kwargs))

        self.stack = nn.ModuleList(layers)

    def forward(self, x: Tensor, re_emb: Optional[Tensor] = None) -> Tensor:
        if self.use_checkpoint:
            # Checkpoint doesn't easily support extra args, so we wrap the call
            return checkpoint(lambda t: self._forward(t, re_emb), x, use_reentrant=True)
        else:
            return self._forward(x, re_emb)

    def _forward(self, x: Tensor, re_emb: Optional[Tensor] = None) -> Tensor:
        for module in self.stack:
            if isinstance(module, AdaLNConv):
                # AdaLNConv requires the conditioning embedding
                if re_emb is None:
                    raise ValueError(
                        "AdaLNConv layer requires re_emb, but it was not provided."
                    )
                x = module(x, re_emb)
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
        re_embedding_dim: Optional[int] = None,
        re_cond_type: Optional[str] = None,  # Options: None, "late_fusion", "adaln"
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
            re_embedding_dim: int
                Optional Reynolds number conditioning dimension.
            re_cond_type: str
                Optional Reynolds number conditioning mode.
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
        self.re_embedding_dim = re_embedding_dim
        self.re_cond_type = re_cond_type

        # Compute the output dimensions after pooling
        if re_cond_type not in [None, "late_fusion", "adaln"]:
            raise ValueError(f"Unknown re_cond_type: {re_cond_type}")
        if re_cond_type is not None and re_embedding_dim is None:
            raise ValueError(f"{re_cond_type} requires re_embedding_dim to be set.")

        self.n_pools = len(hiddens)
        if H % (2**self.n_pools) != 0 or W % (2**self.n_pools) != 0:
            raise ValueError(
                f"Input dimensions (H={H}, W={W}) must be divisible by 2^{self.n_pools} due to pooling."
            )
        self.H_out = H // (2 ** (self.n_pools))
        self.W_out = W // (2 ** (self.n_pools))

        # Define the reynolds number embedding layer
        if self.re_cond_type is not None:
            self.re_embedding = nn.Sequential(
                nn.Linear(1, re_embedding_dim),
                nn.SiLU(),
                nn.Linear(re_embedding_dim, re_embedding_dim),
            )

        # Define the linear layer
        if is_encoder:
            encoder_in_features = hiddens[-1] * self.H_out * self.W_out
            if self.re_cond_type == "late_fusion":
                encoder_in_features += re_embedding_dim
            self.linear = nn.Linear(encoder_in_features, latent_dim)
        else:
            self.linear = nn.Linear(latent_dim, hiddens[-1] * self.H_out * self.W_out)

        # Build convolutional layers
        self.layers = self._build_layers(block_size, kernel_size, conv_kwargs)

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
            "re_cond_type": self.re_cond_type,
            "re_embedding_dim": self.re_embedding_dim,
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
        return nn.Sequential(*layers)

    def forward(self, x: Tensor, re: Optional[Tensor] = None):
        # 1. Compute Reynolds number embedding if needed
        re_emb = None
        if self.re_cond_type is not None:
            if re is None:
                raise ValueError(
                    f"Reynolds number `re` must be provided for conditioning type '{self.re_cond_type}'"
                )
            # Re has shape [B], needs to be [B, 1] for linear layer
            assert (
                re.ndim == 1
            ), f"Expected Re number to be scalar but got tensor of shape {re.shape} instead"
            re_emb = self.re_embedding(re.view(-1, 1))

        # 2. Apply layers based on role (encoder/decoder)
        if self.is_encoder:
            # Pass through convolutional layers
            for layer in self.layers:
                if isinstance(layer, ConvBlock) and self.re_cond_type == "adaln":
                    x = layer(x, re_emb)
                else:
                    x = layer(x)

            # Flatten for linear layer
            out = rearrange(x, "b c h w -> b (c h w)")

            # Apply late fusion if configured
            if self.re_cond_type == "late_fusion":
                out = torch.cat([out, re_emb], dim=1)

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
                if isinstance(layer, ConvBlock) and self.re_cond_type == "adaln":
                    out = layer(out, re_emb)
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
        re_embedding_dim: Optional[int] = None,
        re_cond_type: Optional[str] = None,
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
            re_embedding_dim=re_embedding_dim,
            re_cond_type=re_cond_type,
            **conv_kwargs,
        )


class ConvDecoder(BaseEncoderDecoder):
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
        re_embedding_dim: Optional[int] = None,
        re_cond_type: Optional[str] = None,
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
            is_encoder=False,
            use_checkpoint=use_checkpoint,
            re_embedding_dim=re_embedding_dim,
            re_cond_type=re_cond_type,
            **conv_kwargs,
        )

    # # The base class forward now handles the logic, but we need to ensure
    # # the signature here accepts `re` to pass it to super().
    # def forward(self, x: Tensor, re: Optional[Tensor] = None) -> Tensor:
    #     return super().forward(x, re=re)


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


@dataclass
class TransformerConfig:
    num_layers: int = 4
    nhead: int = 8
    ff_mult: int = 4
    max_len: int = 1000
    dropout: float = 0.1


class HistoryEncoder(ConvEncoder):
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
        re_embedding_dim: Optional[int] = None,
        re_cond_type: Optional[str] = None,
        use_positional_encoding: bool = True,
        transformer_config: TransformerConfig = TransformerConfig(),
        **conv_kwargs,
    ):
        # Pass the new config to the parent ConvEncoder -> BaseEncoderDecoder
        super().__init__(
            C=C,
            H=H,
            W=W,
            latent_dim=latent_dim,
            hiddens=hiddens,
            block_size=block_size,
            kernel_size=kernel_size,
            use_checkpoint=use_checkpoint,
            re_embedding_dim=re_embedding_dim,
            re_cond_type=re_cond_type,
            **conv_kwargs,
        )

        self.pos_enc = (
            PositionalEncoding(latent_dim, max_len=transformer_config.max_len)
            if use_positional_encoding
            else nn.Identity()
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=transformer_config.nhead,
                dim_feedforward=latent_dim * transformer_config.ff_mult,
                dropout=transformer_config.dropout,
                batch_first=True,
            ),
            num_layers=transformer_config.num_layers,
        )

    # The rest of the HistoryEncoder class (forward method) remains unchanged
    # as the super().forward() call will now handle the conditioning correctly.
    def forward(self, x: Tensor, re: Optional[Tensor] = None) -> Tensor:
        """
        Encodes a sequence of frames, conditioning each frame on its corresponding Re value.
        x: Tensor of shape (B, T, C, H, W)
        re: Optional tensor of shape (B, T)
        returns: Tensor of shape (B, latent_dim)
        """
        B, T, C, H, W = x.shape
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")

        re_expanded = None
        # Check for conditioning and prepare re tensor
        if self.re_cond_type is not None:
            if re is None:
                raise ValueError(
                    f"re tensor must be provided for conditioning type '{self.re_cond_type}'"
                )
            if re.ndim != 2 or re.shape != (B, T):
                raise ValueError(
                    f"Expected Re tensor of shape (B, T) = ({B}, {T}), but got {re.shape}"
                )
            # Flatten Re from (B, T) to (B*T,) to match the flattened image data
            re_expanded = re.reshape(-1)

        # Pass flattened image and expanded Re to the parent's forward method
        features = super().forward(x_flat, re=re_expanded)

        # Reshape for transformer and process the sequence
        features = rearrange(features, "(b t) d -> b t d", t=T)
        features = self.pos_enc(features)
        out = self.transformer(features)

        return out.mean(dim=1)


class AdaLNMLP(nn.Module):
    """
    Adaptive Layer Norm for conditioning a latent vector based on a physical parameter.

    This module takes a latent vector 'z' and a corresponding Reynolds number 're'.
    It first creates a high-dimensional embedding of 're', then uses a linear
    projection to predict a feature-wise scale (gamma) and shift (beta). These are
    applied to modulate the latent vector 'z'.
    """

    def __init__(self, latent_dim: int, re_embedding_dim: int):
        """
        Initializes the AdaLNMLP module.

        Args:
            latent_dim (int): The dimension of the latent vector to be modulated.
            re_embedding_dim (int): The dimension of the intermediate Reynolds number embedding.
        """
        super().__init__()
        self.latent_dim = latent_dim

        # 1. A small MLP to embed the scalar Reynolds number into a vector space.
        self.re_embedding = nn.Sequential(
            nn.Linear(1, re_embedding_dim),
            nn.ReLU(),
            nn.Linear(re_embedding_dim, re_embedding_dim),
        )

        # 2. A linear layer to project the embedding to the scale and shift parameters.
        #    We need 2 * latent_dim outputs: one for gamma (scale) and one for beta (shift).
        self.projection = nn.Linear(re_embedding_dim, latent_dim * 2)

    def forward(self, z: Tensor, re: Tensor) -> Tensor:
        """
        Applies the adaptive modulation.

        Args:
            z (Tensor): The input latent vector. Shape: (B, latent_dim).
            re (Tensor): The corresponding Reynolds numbers. Shape: (B, 1).

        Returns:
            Tensor: The modulated latent vector. Shape: (B, latent_dim).
        """
        # Ensure re has the correct shape (B, 1)
        if re.ndim == 1:
            re = re.unsqueeze(1)

        # Create the embedding from the Reynolds number
        re_emb = self.re_embedding(re)

        # Predict scale (gamma) and shift (beta) from the embedding
        gamma, beta = self.projection(re_emb).chunk(2, dim=1)

        # Apply the modulation: z_new = gamma * z + beta
        # This is a feature-wise affine transformation.
        return gamma * z + beta


class KoopmanOperator(nn.Module):
    """
    Koopman operator for learning linear dynamics in latent space.
    This class supports multiple operational modes for the operator K:
    - 'linear': A standard single linear layer K.
    - 'eigen': Learns the eigenvalues and eigenvectors of K directly.
    - 'mlp': A non-linear multi-layer perceptron.
    """

    def __init__(
        self,
        latent_dim: int,
        re_embedding_dim: Optional[int] = None,
        mode: Literal["linear", "eigen", "mlp"] = "linear",
        assume_orthogonal_eigenvectors: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if mode not in ["linear", "eigen", "mlp"]:
            raise ValueError("mode must be one of 'linear', 'eigen', or 'mlp'.")

        self.latent_dim = latent_dim
        self.mode = mode
        self.use_checkpoint = use_checkpoint
        self.assume_orthogonal = assume_orthogonal_eigenvectors
        self.re_embedding_dim = re_embedding_dim

        # Instantiate the AdaLNMLP conditioner if an embedding dimension is provided.
        self.adaln_conditioner = (
            AdaLNMLP(latent_dim, re_embedding_dim) if re_embedding_dim else None
        )

        if self.mode == "linear":
            self.koopman_linear = nn.Linear(latent_dim, latent_dim, bias=False)

        elif self.mode == "eigen":
            # Initialize eigenvalues to be stable (magnitudes <= 1.0)
            # using tanh to constrain the range, promoting stability.
            self.unconstrained_eigenvalues = nn.Parameter(torch.randn(self.latent_dim))
            eigenvectors_init = torch.randn(self.latent_dim, self.latent_dim)
            qr_decomp = torch.linalg.qr(eigenvectors_init)
            self.eigenvectors = nn.Parameter(qr_decomp.Q)

        elif self.mode == "mlp":
            self.koopman_mlp = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 8),
                nn.ReLU(),
                nn.Linear(latent_dim // 8, latent_dim),
            )

    @property
    def eigenvalues(self):
        """Constrains eigenvalues to have magnitude <= 1.0 for stability."""
        if self.mode == "eigen":
            return torch.tanh(self.unconstrained_eigenvalues)
        return None

    def _forward_impl(self, z: Tensor) -> Tensor:
        """Internal forward implementation based on the selected mode."""
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected input latent tensor z of shape (B, {self.latent_dim}), "
                f"but got {z.shape}."
            )

        # The operation is z_{t+1} = K(z_t)
        if self.mode == "linear":
            return self.koopman_linear(z)
        elif self.mode == "eigen":
            return self._forward_eigen(z)
        elif self.mode == "mlp":
            return self.koopman_mlp(z)
        raise RuntimeError(f"Invalid mode '{self.mode}' encountered in forward pass.")

    def _forward_eigen(self, z: Tensor) -> Tensor:
        """
        Performs the linear transformation: K@z = P @ diag(λ) @ P_inv @ z
        """
        P = self.eigenvectors
        P_inv = P.T if self.assume_orthogonal else torch.linalg.pinv(P)

        z_eig_space = P_inv @ z.T
        Lambda = torch.diag_embed(self.eigenvalues.to(z.dtype))
        scaled_z = Lambda @ z_eig_space
        recomposed_z = P @ scaled_z
        return recomposed_z.T

    def forward(self, z: Tensor, re: Optional[Tensor] = None) -> Tensor:
        """
        Applies the full one-step evolution: condition, then operate.

        Args:
            z (Tensor): The current latent state.
            re (Tensor, optional): The Reynolds number for conditioning.
        """
        # 1. First, apply the AdaLNMLP conditioning if available.
        z_conditioned = (
            self.adaln_conditioner(z, re)
            if self.adaln_conditioner is not None and re is not None
            else z
        )

        # 2. Then, apply the core dynamics operator to the conditioned state.
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, z_conditioned, use_reentrant=True)
        else:
            return self._forward_impl(z_conditioned)


class Re(nn.Module):
    """
    Neural network module to predict a scalar Reynolds number from the latent space.
    """

    def __init__(self, latent_dim: int, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.latent_dim = latent_dim
        self.re_predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 8),
            nn.SiLU(),
            nn.Linear(latent_dim // 8, 1),
            nn.Softplus(),
        )

    def _forward_impl(self, z: Tensor) -> Tensor:
        original_shape = z.shape
        if z.ndim > 2:
            z = z.view(-1, self.latent_dim)
        reynolds = self.re_predictor(z)
        if len(original_shape) > 2:
            reynolds = reynolds.view(*original_shape[:-1], 1)
        return reynolds

    def forward(self, z: Tensor) -> Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, z, use_reentrant=False)
        else:
            return self._forward_impl(z)
