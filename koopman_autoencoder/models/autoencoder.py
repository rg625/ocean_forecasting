import torch
from torch import nn
from models.cnn import (
    ConvEncoder,
    ConvDecoder,
    BaseEncoderDecoder,
    HistoryEncoder,
    TransformerConfig,
)
from tensordict import TensorDict
from torch import Tensor
from models.checkpoint import checkpoint
from typing import Union, Tuple, Optional
from dataclasses import dataclass


@dataclass
class KoopmanOutput:
    x_recon: Tensor
    x_preds: TensorDict
    z_preds: Tensor
    reynolds: Optional[Tensor]


class KoopmanOperator(nn.Module):
    def __init__(self, latent_dim: int = 1024, use_checkpoint: bool = False):
        """
        Koopman operator for linear dynamics in latent space.

        Parameters:
            latent_dim: int
                Dimensionality of the latent space.
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.latent_dim = latent_dim
        self.koopman_operator = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, z):
        # Use gradient checkpointing if the flag is enabled
        if self.use_checkpoint:
            return checkpoint(
                self._forward, (z,), self.parameters(), self.use_checkpoint
            )
        else:
            return self._forward(z)

    def _forward(self, z):
        """
        Apply Koopman operator to predict the next state.
        Parameters:
            z: torch.Tensor
        Returns:
            torch.Tensor:
                Residual change in latent space.
        """
        return z + self.koopman_operator(
            z
        )  # Residual latent connection z_{t+1} = (A + Id) z_t


class Re(nn.Module):
    def __init__(self, latent_dim: int = 1024, use_checkpoint: bool = False):
        """
        Koopman operator for linear dynamics in latent space.

        Parameters:
            latent_dim: int
                Dimensionality of the latent space.
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.latent_dim = latent_dim
        self.re = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, z):
        # Use gradient checkpointing if the flag is enabled
        if self.use_checkpoint:
            return checkpoint(
                self._forward, (z,), self.parameters(), self.use_checkpoint
            )
        else:
            return self._forward(z)

    def _forward(self, z):
        """
        Apply Koopman operator to predict the next state.
        Parameters:
            z: torch.Tensor
        Returns:
            torch.Tensor:
                Residual change in latent space.
        """
        return self.re(z)


class KoopmanAutoencoder(nn.Module):
    def __init__(
        self,
        input_frames: int = 2,
        input_channels: int = 6,
        height: int = 64,
        width: int = 64,
        latent_dim: int = 32,
        hidden_dims: list[int] = [64, 128, 64],
        block_size: int = 2,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        use_checkpoint: bool = False,
        transformer_config: Optional[TransformerConfig] = None,
        predict_re: bool = False,
        **conv_kwargs,
    ):
        """
        Koopman Autoencoder for learning dynamical systems in latent space.

        Parameters:
            input_channels: int
                Number of input channels in the data.
            height: int
                Height of the input data.
            width: int
                Width of the input data.
            latent_dim: int
                Dimensionality of the latent space.
            hidden_dims: list of int
                List of hidden dimensions for encoder/decoder layers.
            block_size: int
                Number of convolutional layers in a block.
            kernel_size: int
                Size of the convolution kernel.
            use_checkpoint: bool
                Flag for gradient checkpointing.
            transformer_config: dict,
                Additional arguments for transformer layers.
            predict_re: bool
                Flag for predicting Reynolds Number.
            conv_kwargs: dict
                Additional arguments for convolutional layers.
        """
        super().__init__()
        assert transformer_config is not None, "transformer_config must be provided"
        self.predict_re = predict_re
        # Initialize Encoder
        self.history_encoder = HistoryEncoder(
            C=input_channels,
            H=height,
            W=width,
            latent_dim=latent_dim,
            hiddens=hidden_dims,
            block_size=block_size,
            kernel_size=kernel_size,
            use_checkpoint=use_checkpoint,
            transformer_config=transformer_config,
            **conv_kwargs,
        )

        # Initialize Encoder
        self.encoder: BaseEncoderDecoder = ConvEncoder(
            C=input_channels,
            H=height,
            W=width,
            latent_dim=latent_dim,
            hiddens=hidden_dims,
            block_size=block_size,
            kernel_size=kernel_size,
            use_checkpoint=use_checkpoint,
            **conv_kwargs,
        )

        # Initialize Decoder
        self.decoder: BaseEncoderDecoder = ConvDecoder(
            C=input_channels,
            H=height,
            W=width,
            latent_dim=latent_dim,
            hiddens=hidden_dims,
            block_size=block_size,
            kernel_size=kernel_size,
            use_checkpoint=use_checkpoint,
            **conv_kwargs,
        )

        # Initialize Koopman Operator
        self.koopman_operator = KoopmanOperator(
            latent_dim=latent_dim, use_checkpoint=use_checkpoint
        )
        self.re = (
            Re(latent_dim=latent_dim, use_checkpoint=use_checkpoint)
            if predict_re
            else None
        )

    def encode(self, x: TensorDict):
        """
        Encode the input data into the latent space.

        Parameters:
            x: TensorDict
                Input TensorDict with tensors of shape (B, T, H, W) per variable.

        Returns:
            Tensor: Latent representation.
        """
        # Stack variables along the channel dimension for history
        history_list = [
            x[var][:, :-1].unsqueeze(2) for var in self.vars
        ]  # (B, T-1, 1, H, W)
        stacked_history = torch.cat(history_list, dim=2)  # (B, T-1, C, H, W)

        # Stack variables along the channel dimension for present frame
        present_list = [x[var][:, -1].unsqueeze(1) for var in self.vars]  # (B, 1, H, W)
        stacked_present = torch.cat(present_list, dim=1)  # (B, C, H, W)

        # Pass through encoders
        latent_history = self.history_encoder(
            stacked_history
        )  # expects (B, T, C, H, W)
        latent_present = self.encoder(stacked_present)

        return latent_history + latent_present

    def decode(self, x: Tensor, obstacle_mask: Optional[torch.Tensor] = None):
        """
        Decode the latent representation back to the input space.

        Parameters:
            x: Tensor
                Latent representation of shape (batch_size, latent_dim).
            obstacle_mask: Optional[Tensor]
                Mask tensor of shape (1, H, W) or (B, H, W) to zero out obstacle regions.

        Returns:
            TensorDict: Decoded output per variable, masked if obstacle_mask is provided.
        """
        reconstructed = self.decoder(x)  # (B, C, H, W)

        if obstacle_mask is not None:
            # Ensure shape is (B, 1, H, W) for broadcasting
            if obstacle_mask.ndim == 3:  # (B, H, W)
                obstacle_mask = obstacle_mask.unsqueeze(1)
            elif obstacle_mask.ndim == 2:  # (H, W)
                obstacle_mask = obstacle_mask.unsqueeze(0).unsqueeze(0)
            elif obstacle_mask.ndim == 4:  # already (B, 1, H, W)
                pass
            else:
                raise ValueError(
                    f"Unexpected obstacle_mask shape: {obstacle_mask.shape}"
                )

            reconstructed = reconstructed * obstacle_mask

        # Split into per-variable tensors
        var_count = len(self.vars)
        return TensorDict(
            {self.vars[i]: reconstructed[:, i] for i in range(var_count)},
            batch_size=x.size(0),
        )

    def predict_latent(self, z: Tensor):
        """
        Predict the next state in latent space.

        Parameters:
            z: TensorDict
                TensorDict with key 'latent'.

        Returns:
            TensorDict: Updated TensorDict with key 'latent_pred'.
        """
        return self.koopman_operator(z)

    def forward(self, x: TensorDict, seq_length: Tensor) -> KoopmanOutput:
        """
        Forward pass through the autoencoder with Koopman prediction.

        Parameters:
            x: TensorDict
                Input tensor of shape (batch_size, channels, height, width).
            seq_length: int
                Sequence length for predictions.

        Returns:
            tuple: (reconstructed input, predictions, latent predictions, reynolds estimate)
        """
        # Encode the input
        self.vars = [k for k in x.keys() if k != "obstacle_mask"]
        obstacle_mask = x.get("obstacle_mask", None)
        z = self.encode(x)
        z_pred = z
        z_preds = [z]
        seq_length = seq_length[0] if isinstance(seq_length, Tensor) else seq_length

        # Roll out predictions for the given sequence length
        for _ in range(seq_length):
            z_pred = self.predict_latent(z_pred)
            z_preds.append(z_pred)

        # Decode predictions
        x_recon = self.decode(z_preds[0], obstacle_mask=obstacle_mask)

        x_preds = TensorDict(
            {
                key: torch.stack(
                    [
                        self.decode(z_step, obstacle_mask=obstacle_mask)[key]
                        for z_step in z_preds[1:]
                    ],
                    dim=1,
                )
                for key in self.vars
            },
            batch_size=[x.shape[0]],
        )

        # Compute latent prediction differences
        z_preds = torch.stack(z_preds, dim=1)

        reynolds = self.re(z_preds.detach()) if callable(self.re) else None

        return KoopmanOutput(
            x_recon=x_recon,
            x_preds=x_preds,
            z_preds=z_preds,
            reynolds=reynolds,
        )
