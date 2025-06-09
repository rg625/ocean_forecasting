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
from typing import Union, Tuple, Optional, List
from dataclasses import dataclass
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class KoopmanOutput:
    """
    Dataclass to hold the outputs of the KoopmanAutoencoder's forward pass.

    Attributes:
        x_recon (TensorDict): Reconstructed input data. Each key holds a tensor
                              of shape (B, H, W).
        x_preds (TensorDict): Predicted future data sequences. Each key holds a tensor
                              of shape (B, T_pred, H, W).
        z_preds (Tensor): Latent space predictions over time. Shape (B, T_pred+1, latent_dim).
                          z_preds[0] is the encoded input, z_preds[1:] are the rollouts.
        reynolds (Optional[Tensor]): Predicted Reynolds number (B, 1) if predict_re is True,
                                     otherwise None.
    """

    x_recon: TensorDict
    x_preds: TensorDict
    z_preds: Tensor
    reynolds: Optional[Tensor]


class KoopmanOperator(nn.Module):
    """
    Koopman operator for learning linear dynamics in latent space.
    Implements a residual connection: z_{t+1} = z_t + A * z_t.
    """

    def __init__(self, latent_dim: int = 1024, use_checkpoint: bool = False):
        """
        Initializes the KoopmanOperator.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            use_checkpoint (bool): Flag to enable gradient checkpointing for memory efficiency.
        """
        super().__init__()
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if not isinstance(use_checkpoint, bool):
            raise TypeError("use_checkpoint must be a boolean.")

        self.use_checkpoint = use_checkpoint
        self.latent_dim = latent_dim
        # Koopman operator learns the matrix A in z_{t+1} = (I + A)z_t
        self.koopman_linear = nn.Linear(latent_dim, latent_dim, bias=False)

    def _forward_impl(self, z: Tensor) -> Tensor:
        """
        Internal implementation of the forward pass without checkpointing logic.

        Args:
            z (Tensor): Latent state tensor of shape (B, latent_dim).

        Returns:
            Tensor: Predicted next latent state of shape (B, latent_dim).
        """
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected input latent tensor z of shape (B, {self.latent_dim}), "
                f"but got {z.shape}."
            )
        return z + self.koopman_linear(z)

    def forward(self, z: Tensor) -> Tensor:
        """
        Apply Koopman operator to predict the next state in latent space.

        Args:
            z (Tensor): Latent state tensor of shape (B, latent_dim).

        Returns:
            Tensor: Predicted next latent state of shape (B, latent_dim).
        """
        if self.use_checkpoint:
            # Assumes checkpoint function correctly handles (func, args, kwargs, use_checkpoint_flag)
            # and passes gradients through properly.
            return checkpoint(
                self._forward_impl, (z,), self.parameters(), self.use_checkpoint
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
        # return z + self.koopman_operator(
        return self.koopman_operator(
            z
        )  # Residual latent connection z_{t+1} = (A + Id) z_t


class Re(nn.Module):
    """
    Neural network module to predict a scalar Reynolds number from the latent space.
    """

    def __init__(self, latent_dim: int = 1024, use_checkpoint: bool = False):
        """
        Initializes the Re predictor.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            use_checkpoint (bool): Flag to enable gradient checkpointing.
        """
        super().__init__()
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if not isinstance(use_checkpoint, bool):
            raise TypeError("use_checkpoint must be a boolean.")

        self.use_checkpoint = use_checkpoint
        self.latent_dim = latent_dim
        self.re_predictor = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Softplus(),
        )

    def _forward_impl(self, z: Tensor) -> Tensor:
        """
        Internal implementation of the forward pass without checkpointing logic.

        Args:
            z (Tensor): Latent state tensor of shape (B, latent_dim) or (B, T, latent_dim).

        Returns:
            Tensor: Predicted Reynolds number(s) of shape (B, 1) or (B, T, 1).
        """
        # Allow z to be (B, latent_dim) or (B, T, latent_dim)
        original_shape = z.shape
        if z.ndim > 2:
            z = z.view(
                -1, z.shape[-1]
            )  # Flatten batch and time dimensions for linear layers

        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected input latent tensor z last dim to be {self.latent_dim}, "
                f"but got {z.shape[-1]} in shape {original_shape}."
            )

        reynolds = self.re_predictor(z)

        if len(original_shape) > 2:
            reynolds = reynolds.view(
                *original_shape[:-1], 1
            )  # Reshape back to (B, T, 1)

        return reynolds

    def forward(self, z: Tensor) -> Tensor:
        """
        Predicts the Reynolds number from the latent space.

        Args:
            z (Tensor): Latent state tensor of shape (B, latent_dim) or (B, T, latent_dim).

        Returns:
            Tensor: Predicted Reynolds number(s) of shape (B, 1) or (B, T, 1).
        """
        if self.use_checkpoint:
            return checkpoint(
                self._forward_impl, (z,), self.parameters(), self.use_checkpoint
            )
        else:
            return self._forward_impl(z)


class KoopmanAutoencoder(nn.Module):
    """
    Koopman Autoencoder for learning dynamical systems in latent space.
    Combines an encoder, decoder, Koopman operator for latent dynamics,
    and optionally a Reynolds number predictor.
    """

    def __init__(
        self,
        input_frames: int = 2,  # Represents T_input (e.g., history + present)
        input_channels: int = 6,  # C
        height: int = 64,  # H
        width: int = 64,  # W
        latent_dim: int = 32,
        hidden_dims: List[int] = [64, 128, 64],
        block_size: int = 2,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        use_checkpoint: bool = False,
        transformer_config: Optional[TransformerConfig] = None,
        predict_re: bool = False,
        data_variables: Optional[List[str]] = None,
        **conv_kwargs,
    ):
        """
        Initializes the Koopman Autoencoder.

        Args:
            input_frames (int): Total number of input frames (e.g., 2 for history + present).
            input_channels (int): Number of physical channels in the data (e.g., u, v, p, omega_x, etc.).
            height (int): Height of the input data spatial dimension.
            width (int): Width of the input data spatial dimension.
            latent_dim (int): Dimensionality of the latent space.
            hidden_dims (List[int]): List of hidden dimensions for encoder/decoder layers.
            block_size (int): Number of convolutional layers in a block.
            kernel_size (Union[int, Tuple[int, int]]): Size of the convolution kernel.
            use_checkpoint (bool): Flag for gradient checkpointing.
            transformer_config (TransformerConfig): Configuration for transformer layers in HistoryEncoder.
            predict_re (bool): Flag for predicting Reynolds Number.
            data_variables (List[str]): **Crucial**: A fixed list of string names for the data variables
                                         (e.g., ['u', 'v', 'p']). This should come from the dataset.
            conv_kwargs (dict): Additional arguments for convolutional layers.
        """
        super().__init__()

        if transformer_config is None:
            raise ValueError("transformer_config must be provided and not None.")
        if not isinstance(transformer_config, TransformerConfig):
            raise TypeError(
                "transformer_config must be an instance of TransformerConfig."
            )

        if (
            data_variables is None
            or not isinstance(data_variables, list)
            or not data_variables
        ):
            raise ValueError(
                "data_variables must be a non-empty list of strings, provided from the dataset."
            )

        self.input_frames = input_frames
        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.latent_dim = latent_dim
        self.use_checkpoint = use_checkpoint
        self.predict_re = predict_re
        self.data_variables = data_variables  # Store the fixed list of variables

        # Validate input_channels matches expected channels from data_variables
        if len(self.data_variables) != input_channels:
            raise ValueError(
                f"Number of data_variables ({len(self.data_variables)}) "
                f"does not match specified input_channels ({input_channels})."
            )

        # Initialize History Encoder for historical frames (input_frames - 1)
        self.history_encoder = HistoryEncoder(
            C=input_channels,  # This C is per variable if history_encoder concatenates
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

        # Initialize Encoder for the present frame
        self.encoder: BaseEncoderDecoder = ConvEncoder(
            C=input_channels,  # This C is the total concatenated channels
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
            C=input_channels,  # Output channels for decoder
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
        # Initialize Reynolds number predictor if required
        self.re = (
            Re(latent_dim=latent_dim, use_checkpoint=use_checkpoint)
            if predict_re
            else None
        )

    def encode(self, x: TensorDict) -> Tensor:
        """
        Encode the input data (history + present) into the latent space.

        Args:
            x (TensorDict): Input TensorDict. Expected structure:
                            Each key (variable name) should contain a tensor
                            of shape (B, T_input, H, W), where T_input is `self.input_frames`.

        Returns:
            Tensor: Latent representation of shape (B, latent_dim).
        """
        # Validate input x structure
        if not isinstance(x, TensorDict):
            raise TypeError("Input 'x' must be a TensorDict.")

        if not all(var in x.keys() for var in self.data_variables):
            missing_vars = [var for var in self.data_variables if var not in x.keys()]
            raise ValueError(
                f"Input TensorDict 'x' is missing required data variables: {missing_vars}"
            )

        # Extract history and present frames based on input_frames
        # History frames are x[:, :-1], Present frame is x[:, -1]

        # history_list will contain tensors of shape (B, input_frames-1, H, W)
        history_list = [x[var][:, :-1] for var in self.data_variables]

        # Validate history frame shapes and input_frames
        if not history_list:
            raise ValueError(
                "No history frames found for encoding. Check input_frames and data variables."
            )
        if history_list[0].ndim != 4 or history_list[0].shape[1] != (
            self.input_frames - 1
        ):
            raise ValueError(
                f"History frames for encoding should be (B, {self.input_frames - 1}, H, W). "
                f"Got {history_list[0].shape} after slicing."
            )

        # Stack variables along the channel dimension for history
        # (B, T_hist, H, W) -> (B, T_hist, C_var, H, W) for each var, then cat along C_var
        # Current HistoryEncoder expects (B, T, C_total, H, W) where T is time and C_total is concatenated channels
        stacked_history = torch.cat(
            [var_tensor.unsqueeze(2) for var_tensor in history_list], dim=2
        )  # (B, T_hist, C_total, H, W)

        # present_list will contain tensors of shape (B, H, W)
        present_list = [x[var][:, -1] for var in self.data_variables]

        # Validate present frame shapes
        if not present_list:
            raise ValueError(
                "No present frame found for encoding. Check input_frames and data variables."
            )
        if present_list[0].ndim != 3:  # (B, H, W)
            raise ValueError(
                f"Present frame for encoding should be (B, H, W). Got {present_list[0].shape}."
            )

        # Stack variables along the channel dimension for present frame
        stacked_present = torch.cat(
            [var_tensor.unsqueeze(1) for var_tensor in present_list], dim=1
        )  # (B, C_total, H, W)

        # Pass through encoders
        latent_history = self.history_encoder(stacked_history)  # (B, latent_dim)
        latent_present = self.encoder(stacked_present)  # (B, latent_dim)

        # Ensure consistent latent dimensions before adding
        if latent_history.shape != latent_present.shape:
            raise RuntimeError(
                f"Mismatch in latent dimensions from history_encoder ({latent_history.shape}) "
                f"and encoder ({latent_present.shape})."
            )

        return latent_history + latent_present

    def decode(
        self, z: Tensor, obstacle_mask: Optional[torch.Tensor] = None
    ) -> TensorDict:
        """
        Decode the latent representation back to the input space.

        Args:
            z (Tensor): Latent representation tensor of shape (B, latent_dim).
            obstacle_mask (Optional[Tensor]): Mask tensor of shape (H, W), (1, H, W), or (B, H, W)
                                             to zero out obstacle regions.

        Returns:
            TensorDict: Decoded output per variable, masked if obstacle_mask is provided.
                        Each key holds a tensor of shape (B, H, W).
        """
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected input latent tensor z for decoding to be (B, {self.latent_dim}), "
                f"but got {z.shape}."
            )

        reconstructed_channels = self.decoder(z)  # Output shape (B, C_total, H, W)

        if reconstructed_channels.shape[1] != self.input_channels:
            raise RuntimeError(
                f"Decoder output channels ({reconstructed_channels.shape[1]}) "
                f"do not match expected input_channels ({self.input_channels})."
            )

        if obstacle_mask is not None:
            if not isinstance(obstacle_mask, torch.Tensor):
                raise TypeError("obstacle_mask must be a torch.Tensor.")

            # Ensure obstacle_mask is (B, 1, H, W) for broadcasting
            if obstacle_mask.ndim == 2:  # (H, W) -> (1, 1, H, W)
                obstacle_mask_broadcast = obstacle_mask.unsqueeze(0).unsqueeze(0)
            elif obstacle_mask.ndim == 3:  # (B, H, W) -> (B, 1, H, W)
                obstacle_mask_broadcast = obstacle_mask.unsqueeze(1)
            elif (
                obstacle_mask.ndim == 4 and obstacle_mask.shape[1] == 1
            ):  # (B, 1, H, W)
                obstacle_mask_broadcast = obstacle_mask
            else:
                raise ValueError(
                    f"Unexpected obstacle_mask shape: {obstacle_mask.shape}. "
                    "Expected (H, W), (B, H, W), or (B, 1, H, W)."
                )

            # Ensure mask matches spatial dimensions
            if obstacle_mask_broadcast.shape[-2:] != reconstructed_channels.shape[-2:]:
                raise ValueError(
                    f"Obstacle mask spatial dimensions {obstacle_mask_broadcast.shape[-2:]} "
                    f"do not match reconstructed data dimensions {reconstructed_channels.shape[-2:]}."
                )

            reconstructed_channels = reconstructed_channels * obstacle_mask_broadcast

        # Split concatenated channels back into per-variable tensors
        # Each var gets `self.input_channels / len(self.data_variables)` channels if multi-channel per var
        # Assuming each variable is a single channel for now.
        if self.input_channels % len(self.data_variables) != 0:
            raise ValueError(
                f"Total input_channels ({self.input_channels}) is not evenly divisible "
                f"by the number of data_variables ({len(self.data_variables)}). "
                "This indicates an issue in channel distribution."
            )

        channels_per_var = self.input_channels // len(self.data_variables)

        decoded_data = {}
        for i, var_name in enumerate(self.data_variables):
            start_channel = i * channels_per_var
            end_channel = (i + 1) * channels_per_var
            # If channels_per_var is 1, then we extract and remove the channel dim to get (B, H, W)
            if channels_per_var == 1:
                decoded_data[var_name] = reconstructed_channels[:, start_channel, :, :]
            else:  # If a variable itself has multiple channels
                decoded_data[var_name] = reconstructed_channels[
                    :, start_channel:end_channel, :, :
                ]

        return TensorDict(decoded_data, batch_size=z.size(0))

    def predict_latent(self, z: Tensor) -> Tensor:
        """
        Predict the next state in latent space using the Koopman operator.

        Args:
            z (Tensor): Latent representation of shape (B, latent_dim).

        Returns:
            Tensor: Predicted next latent state of shape (B, latent_dim).
        """
        return self.koopman_operator(z)

    def forward(self, x: TensorDict, seq_length: Union[int, Tensor]) -> KoopmanOutput:
        """
        Forward pass through the autoencoder with Koopman prediction rollout.

        Args:
            x (TensorDict): Input tensor. Expected structure:
                            Each key (variable name) should contain a tensor
                            of shape (B, T_input, H, W), where T_input is `self.input_frames`.
                            May also contain 'obstacle_mask' if applicable.
            seq_length (Union[int, Tensor]): The number of future steps to predict.
                                             If a Tensor, it must be a scalar (e.g., from DataLoader).

        Returns:
            KoopmanOutput: A dataclass containing reconstructed input,
                           future predictions, latent predictions, and optionally Reynolds estimate.
        """
        # Validate input x
        if not isinstance(x, TensorDict):
            raise TypeError("Input 'x' must be a TensorDict.")
        if "obstacle_mask" in x.keys():
            obstacle_mask = x.get("obstacle_mask", None)
            # Remove obstacle_mask from x for processing main data variables
            # Create a shallow copy to avoid modifying the original TensorDict in place
            x_data = x.exclude("obstacle_mask")
        else:
            obstacle_mask = None
            x_data = x

        # Ensure all expected data variables are present and have correct first dimension for sequence
        for var in self.data_variables:
            if var not in x_data.keys():
                raise ValueError(
                    f"Required data variable '{var}' not found in input TensorDict 'x'."
                )
            if x_data[var].ndim != 4 or x_data[var].shape[1] != self.input_frames:
                raise ValueError(
                    f"Variable '{var}' in input TensorDict 'x' has unexpected shape {x_data[var].shape}. "
                    f"Expected (B, {self.input_frames}, H, W)."
                )

        # Handle seq_length: ensure it's a scalar integer
        if isinstance(seq_length, torch.Tensor):
            if seq_length.numel() == 0:
                # Handle case where seq_length tensor might be empty (e.g. if batch is empty)
                logger.warning(
                    "seq_length tensor is empty. Setting target_length to 0."
                )
                seq_length_int = 0
            else:
                # Assuming all elements in the batch have the same sequence length
                # due to the BatchSampler's design.
                seq_length_int = seq_length[
                    0
                ].item()  # Take the first element and convert to Python int
        elif isinstance(seq_length, int):
            seq_length_int = seq_length
        else:
            raise TypeError(
                f"seq_length must be an int or a Tensor, but got {type(seq_length)}."
            )

        if seq_length_int < 0:
            raise ValueError(
                f"seq_length cannot be negative, but got {seq_length_int}."
            )

        if seq_length_int <= 0:
            logger.warning(
                f"Requested seq_length is {seq_length_int}. No future predictions will be made."
            )
            # Handle this case by returning empty predictions, but with valid initial z_preds
            z = self.encode(x_data)
            x_recon = self.decode(z, obstacle_mask=obstacle_mask)
            z_preds = z.unsqueeze(1)  # (B, 1, latent_dim)
            empty_x_preds = TensorDict(
                {
                    key: torch.empty(
                        x.batch_size[0],
                        0,
                        self.height,
                        self.width,
                        device=x.device,
                        dtype=x.dtype,
                    )
                    for key in self.data_variables
                },
                batch_size=[x.batch_size[0]],
            )
            reynolds = (
                self.re(z_preds.detach()) if self.predict_re and self.re else None
            )  # Re could still be predicted from initial z
            return KoopmanOutput(
                x_recon=x_recon,
                x_preds=empty_x_preds,
                z_preds=z_preds,
                reynolds=reynolds,
            )

        # Encode the input (the last frame for present, and previous for history)
        z = self.encode(x_data)  # z is (B, latent_dim)

        z_pred_current = z  # Start with the encoded initial state
        z_preds_list = [z]  # Store initial z as the first element

        # Roll out predictions for the given sequence length
        for _ in range(seq_length_int):
            z_pred_current = self.predict_latent(z_pred_current)
            z_preds_list.append(z_pred_current)

        # Decode initial input (for reconstruction loss)
        x_recon = self.decode(z_preds_list[0], obstacle_mask=obstacle_mask)

        # Decode future predictions
        # z_preds_list[1:] contains only the predicted future latent states
        x_preds_dict = {}
        for key in self.data_variables:
            # Stack decoded steps for each variable along a new 'time' dimension (dim=1)
            x_preds_dict[key] = torch.stack(
                [
                    self.decode(z_step, obstacle_mask=obstacle_mask)[key]
                    for z_step in z_preds_list[1:]
                ],
                dim=1,  # Stack along the sequence length dimension
            )
        x_preds = TensorDict(
            x_preds_dict, batch_size=x.batch_size
        )  # x.batch_size is a tuple (B,)

        # Stack all latent predictions (initial + rollouts)
        z_preds_stacked = torch.stack(
            z_preds_list, dim=1
        )  # (B, seq_length_int + 1, latent_dim)

        # Compute Reynolds prediction if enabled
        # Crucially: decide if 'reynolds' prediction should flow gradient through Koopman operator
        # Currently, it uses z_preds_stacked.detach(), meaning no gradient flows back to K.O. or encoder.
        # If this is desired, remove .detach()
        reynolds = None
        if self.predict_re and self.re:
            reynolds = self.re(z_preds_stacked.detach())  # Shape (B, T_pred+1, 1)

        return KoopmanOutput(
            x_recon=x_recon,
            x_preds=x_preds,
            z_preds=z_preds_stacked,
            reynolds=reynolds,
        )
