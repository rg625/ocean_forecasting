import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from tensordict import TensorDict
from torch import Tensor
from typing import Union, Tuple, Optional, List, Dict, Literal
from collections.abc import Mapping
from dataclasses import dataclass
import logging

# Assume these are correctly defined elsewhere
from models.cnn import (
    ConvEncoder,
    ConvDecoder,
    HistoryEncoder,
    TransformerConfig,
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class KoopmanOutput:
    """
    Dataclass to hold the outputs of the KoopmanAutoencoder's forward pass.
    """

    x_recon: TensorDict
    x_preds: TensorDict
    z_preds: Tensor
    reynolds: Optional[Tensor]


class KoopmanOperator(nn.Module):
    """
    Koopman operator for learning linear dynamics in latent space.
    Implements a residual connection: z_{t+1} = K(z_t).

    This class supports multiple operational modes for the operator K:
    - 'linear': A standard single linear layer K.
    - 'eigen': Learns the eigenvalues and eigenvectors of K directly and
               reconstructs the operation z -> K @ z.
    - 'mlp': A non-linear multi-layer perceptron.
    """

    def __init__(
        self,
        latent_dim: int,
        mode: Literal["linear", "eigen", "mlp"] = "linear",
        assume_orthogonal_eigenvectors: bool = True,
        use_checkpoint: bool = False,
    ):
        """
        Initializes the KoopmanOperator.

        Args:
            latent_dim (int): The dimension of the latent space.
            mode (str): The operational mode. One of 'linear', 'eigen', 'mlp'.
            assume_orthogonal_eigenvectors (bool): In 'eigen' mode, if True,
                assumes eigenvectors are orthogonal (P_inv = P.T), which is
                more stable and faster. Defaults to False.
            use_checkpoint (bool): Whether to use checkpointing during training.
        """
        super().__init__()
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if mode not in ["linear", "eigen", "mlp"]:
            raise ValueError("mode must be one of 'linear', 'eigen', or 'mlp'.")

        self.latent_dim = latent_dim
        self.mode = mode
        self.use_checkpoint = use_checkpoint
        self.assume_orthogonal = assume_orthogonal_eigenvectors

        if self.mode == "linear":
            self.koopman_linear = nn.Linear(latent_dim, latent_dim, bias=False)

        elif self.mode == "eigen":
            self.eigenvalues = nn.Parameter(torch.randn(self.latent_dim))
            self.eigenvectors = nn.Parameter(
                torch.randn(self.latent_dim, self.latent_dim)
            )
            # Optional: Initialize eigenvectors to be nearly orthogonal
            # qr_decomp = torch.linalg.qr(self.eigenvectors.data)
            # self.eigenvectors.data = qr_decomp.Q

        elif self.mode == "mlp":
            self.koopman_mlp = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 8),
                nn.ReLU(),
                nn.Linear(latent_dim // 8, latent_dim),
            )

    def _forward_impl(self, z: Tensor) -> Tensor:
        """Internal forward implementation based on the selected mode."""
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected input latent tensor z of shape (B, {self.latent_dim}), "
                f"but got {z.shape}."
            )

        if self.mode == "linear":
            return self.koopman_linear(z)
        elif self.mode == "eigen":
            return self._forward_eigen(z)
        elif self.mode == "mlp":
            return self.koopman_mlp(z)

    def _forward_eigen(self, z: Tensor) -> Tensor:
        """
        Performs the linear transformation by recomposing from learned
        eigenvalues and eigenvectors: K@z = P @ diag(λ) @ P_inv @ z
        """
        P = self.eigenvectors

        # Determine the inverse of the eigenvector matrix P
        if self.assume_orthogonal:
            P_inv = P.T
        else:
            try:
                P_inv = torch.linalg.inv(P)
            except torch.linalg.LinAlgError:
                return torch.zeros_like(z)

        z_eig_space = P_inv @ z.T
        Lambda = torch.diag_embed(self.eigenvalues)
        scaled_z = Lambda @ z_eig_space
        recomposed_z = P @ scaled_z
        return recomposed_z.T

    def forward(self, z: Tensor) -> Tensor:
        """
        Applies the Koopman operator to the latent state z.
        The full dynamics are z_{t+1} = z_t + K(z_t).
        """
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, z, use_reentrant=True)
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
        # return z + self.koopman_operator(
        return self.koopman_operator(
            z
        )  # Residual latent connection z_{t+1} = (A + Id) z_t


class Re(nn.Module):
    """
    Neural network module to predict a scalar Reynolds number from the latent space.
    """

    def __init__(self, latent_dim: int, use_checkpoint: bool = False):
        super().__init__()
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
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
            z = z.view(-1, z.shape[-1])

        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected input latent tensor z last dim to be {self.latent_dim}, "
                f"but got {z.shape[-1]} in shape {original_shape}."
            )
        reynolds = self.re_predictor(z)

        if len(original_shape) > 2:
            reynolds = reynolds.view(*original_shape[:-1], 1)
        return reynolds

    def forward(self, z: Tensor) -> Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, z, use_reentrant=False)
        else:
            return self._forward_impl(z)


class KoopmanAutoencoder(nn.Module):
    """
    Koopman Autoencoder for learning and predicting dynamical systems.
    This production-ready version features efficient batched decoding,
    robust input handling, and a flexible API.
    """

    def __init__(
        self,
        data_variables: Dict[str, int],
        input_frames: int = 2,
        height: int = 64,
        width: int = 64,
        latent_dim: int = 32,
        operator_mode: Literal["linear", "eigen", "mlp"] = "linear",
        hidden_dims: List[int] = [64, 128, 64],
        block_size: int = 2,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        use_checkpoint: bool = False,
        transformer_config: Optional[TransformerConfig] = None,
        predict_re: bool = False,
        re_grad_enabled: bool = False,
        **conv_kwargs,
    ):
        """
        Initializes the Koopman Autoencoder.

        Args:
            data_variables (Dict[str, int]): **Crucial**: A dictionary mapping variable names to their
                                             respective number of channels. E.g., {'pressure': 1, 'velocity': 2}.
            input_frames (int): Total number of input frames (e.g., 2 for history + present). Must be >= 1.
            height (int): Height of the input data spatial dimension.
            width (int): Width of the input data spatial dimension.
            latent_dim (int): Dimensionality of the latent space.
            operator_mode (ste): Mode for operator matrix calculation.
            hidden_dims (List[int]): List of hidden dimensions for encoder/decoder layers.
            block_size (int): Number of convolutional layers in a block.
            kernel_size (Union[int, Tuple[int, int]]): Size of the convolution kernel.
            use_checkpoint (bool): Flag for gradient checkpointing.
            transformer_config (TransformerConfig): Configuration for transformer layers in HistoryEncoder.
            predict_re (bool): Flag for creating the Reynolds Number prediction head.
            re_grad_enabled (bool): If True, allows gradients from the Reynolds loss to flow back
                                    to the main model, acting as a physics-informed regularizer.
            conv_kwargs (dict): Additional arguments for convolutional layers.
        """
        super().__init__()

        # --- Configuration Validation ---
        if not (isinstance(data_variables, Mapping) and data_variables):
            raise ValueError(
                "data_variables must be a non-empty dictionary mapping names to channel counts."
            )
        if input_frames < 1:
            raise ValueError("input_frames must be at least 1.")
        if transformer_config is None and input_frames > 1:
            raise ValueError("transformer_config must be provided if input_frames > 1.")

        self.data_variables = data_variables
        self.input_frames = input_frames
        self.height = height
        self.width = width
        self.latent_dim = latent_dim
        self.use_checkpoint = use_checkpoint
        self.predict_re = predict_re
        self.re_grad_enabled = re_grad_enabled

        # The total number of channels is derived from the data_variables dictionary
        self.total_input_channels = sum(self.data_variables.values())

        # --- Module Initialization ---
        encoder_args = {
            "H": height,
            "W": width,
            "latent_dim": latent_dim,
            "hiddens": hidden_dims,
            "block_size": block_size,
            "kernel_size": kernel_size,
            "use_checkpoint": use_checkpoint,
            **conv_kwargs,
        }

        # History Encoder is only needed if we have history to process
        self.history_encoder = None
        if self.input_frames > 1:
            assert (
                transformer_config is not None
            ), "Transformer config cannot be None when using history."
            self.history_encoder = HistoryEncoder(
                C=self.total_input_channels,
                transformer_config=transformer_config,
                **encoder_args,
            )

        # Encoder for the present frame
        self.encoder = ConvEncoder(C=self.total_input_channels, **encoder_args)

        # Decoder
        self.decoder = ConvDecoder(C=self.total_input_channels, **encoder_args)

        # Koopman Operator and Reynolds Predictor
        self.koopman_operator = KoopmanOperator(
            latent_dim=latent_dim, mode=operator_mode, use_checkpoint=use_checkpoint
        )
        self.re_predictor = (
            Re(latent_dim=latent_dim, use_checkpoint=use_checkpoint)
            if predict_re
            else None
        )

    def encode(self, x: TensorDict) -> Tensor:
        """Encodes input data (history + present) into the latent space."""
        if not all(var in x.keys() for var in self.data_variables):
            raise ValueError("Input TensorDict 'x' is missing required data variables.")

        present_list = []
        for var, num_channels in self.data_variables.items():
            tensor_slice = x[var][:, -1]  # Shape: (B, H, W) or (B, C, H, W)

            # Add channel dimension if it's a single-channel variable and missing
            if num_channels == 1 and tensor_slice.ndim == 3:
                tensor_slice = tensor_slice.unsqueeze(1)  # -> (B, 1, H, W)
            present_list.append(tensor_slice)

        stacked_present = torch.cat(present_list, dim=1)

        # Validate spatial dimensions
        if stacked_present.shape[-2:] != (self.height, self.width):
            raise ValueError(
                f"Spatial dimension mismatch. Model configured for ({self.height}, {self.width}), "
                f"but received data with shape {stacked_present.shape[-2:]}."
            )

        latent_present = self.encoder(stacked_present)

        if self.input_frames > 1 and self.history_encoder is not None:
            history_list = []
            for var, num_channels in self.data_variables.items():
                tensor_slice = x[var][:, :-1]  # Shape: (B, T, H, W) or (B, T, C, H, W)

                # Add channel dimension for history as well
                if num_channels == 1 and tensor_slice.ndim == 4:
                    tensor_slice = tensor_slice.unsqueeze(2)  # -> (B, T, 1, H, W)
                history_list.append(tensor_slice)

            stacked_history = torch.cat(history_list, dim=2)
            latent_history = self.history_encoder(stacked_history)
            return latent_history + latent_present

        return latent_present

    def decode(self, z: Tensor, obstacle_mask: Optional[Tensor] = None) -> TensorDict:
        """Decodes a batch of latent states back to the physical domain."""
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent tensor z of shape (B, {self.latent_dim}), got {z.shape}."
            )

        reconstructed_channels = self.decoder(z)  # Shape: (B, C_total, H, W)

        if obstacle_mask is not None:
            # Broadcast mask to apply to all channels
            mask = obstacle_mask[0, 0] if obstacle_mask.ndim == 4 else obstacle_mask
            reconstructed_channels = reconstructed_channels * mask[None, None, :, :]

        # Dynamically split channels based on the data_variables dictionary
        decoded_data = {}
        current_channel = 0
        for var, num_channels in self.data_variables.items():
            end_channel = current_channel + num_channels
            var_tensor = reconstructed_channels[:, current_channel:end_channel]
            # Squeeze channel dim if it's singular, otherwise keep it
            decoded_data[var] = (
                var_tensor.squeeze(1) if num_channels == 1 else var_tensor
            )
            current_channel = end_channel

        return TensorDict(decoded_data, batch_size=[z.size(0)])

    def forward(self, x: TensorDict, seq_length: Union[int, Tensor]) -> KoopmanOutput:
        """Forward pass: Encode, roll out predictions, and decode."""
        # --- Input Handling ---
        obstacle_mask = x.get("obstacle_mask", None)
        x_data = x.exclude("obstacle_mask") if "obstacle_mask" in x else x
        # --- Sequence Length Parsing ---
        seq_length_int = (
            seq_length[0, 0].item()
            if isinstance(seq_length, Tensor)
            else int(seq_length)
        )
        if seq_length_int < 0:
            raise ValueError(f"seq_length cannot be negative, got {seq_length_int}.")

        # --- Encode initial state ---
        z0 = self.encode(x_data)

        # --- Handle Zero-Length Prediction Edge Case ---
        if seq_length_int == 0:
            x_recon = self.decode(z0, obstacle_mask)
            z_preds = z0.unsqueeze(1)  # Shape: (B, 1, D)
            reynolds = (
                self.re_predictor(z_preds.detach()) if self.re_predictor else None
            )
            # Create an empty TensorDict for predictions
            empty_preds = TensorDict(
                {
                    key: torch.empty(
                        z0.size(0), 0, *val.shape[-2:], device=z0.device, dtype=z0.dtype
                    )
                    for key, val in x_recon.items()
                },
                batch_size=[z0.size(0)],
            )
            return KoopmanOutput(
                x_recon=x_recon, x_preds=empty_preds, z_preds=z_preds, reynolds=reynolds
            )

        # --- Autoregressive Rollout in Latent Space ---
        z_current = z0
        z_preds_list = []
        for _ in range(seq_length_int):
            z_current = self.koopman_operator(z_current)
            z_preds_list.append(z_current)

        z_preds_stacked = torch.stack(z_preds_list, dim=1)  # Shape: (B, T_pred+1, D)

        # --- Decode Reconstruction and Predictions ---
        # Decode reconstruction of the initial state
        x_recon = self.decode(z0, obstacle_mask)

        # **PERFORMANCE-CRITICAL**: Decode all future steps in a single batched pass
        future_z_batch = z_preds_stacked.view(
            -1, self.latent_dim
        )  # Shape: (B * T_pred, D)
        if future_z_batch.shape[0] > 0:
            decoded_batch = self.decode(future_z_batch, obstacle_mask)
            # Reshape back to (B, T_pred, ...) for each variable
            x_preds = decoded_batch.apply(
                lambda t: t.view(z0.size(0), seq_length_int, *t.shape[1:]),
                batch_size=[z0.size(0), seq_length_int],
            )
        else:  # Handle case where seq_length might have been 1, so future steps are empty
            x_preds = x_recon.apply(
                lambda t: t.unsqueeze(1)[:, :0], batch_size=[z0.size(0), 0]
            )

        # --- Predict Reynolds Number ---
        reynolds = None
        if self.predict_re and self.re_predictor is not None:
            # Conditionally detach based on the re_grad_enabled flag
            if self.re_grad_enabled:
                # Gradients will flow back to the main model
                z_for_re = z_preds_stacked
            else:
                # Gradients are stopped, only the Re head is trained
                z_for_re = z_preds_stacked.detach()

            reynolds = self.re_predictor(z_for_re)

        return KoopmanOutput(
            x_recon=x_recon, x_preds=x_preds, z_preds=z_preds_stacked, reynolds=reynolds
        )
