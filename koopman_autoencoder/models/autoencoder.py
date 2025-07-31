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
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
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
    This class supports multiple operational modes for the operator K:
    - 'linear': A standard single linear layer K.
    - 'eigen': Learns the eigenvalues and eigenvectors of K directly.
    - 'mlp': A non-linear multi-layer perceptron.
    """

    def __init__(
        self,
        latent_dim: int,
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

    def forward(self, z: Tensor) -> Tensor:
        """
        Applies the Koopman operator to the latent state z to get the next state.
        Note: Checkpointing inside a loop is ineffective for back-propagating
        through the sequence, so it's applied here only to the single step.
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


class KoopmanAutoencoder(nn.Module):
    """
    Koopman Autoencoder for learning and predicting dynamical systems.
    """

    def __init__(
        self,
        data_variables: Dict[str, int],
        input_frames: int = 2,
        height: int = 64,
        width: int = 64,
        latent_dim: int = 32,
        re_embedding_dim: Optional[int] = 64,
        re_cond_type: Literal[None, "late_fusion", "adaln"] = None,
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
        super().__init__()
        # --- Configuration Validation ---
        if not (isinstance(data_variables, Mapping) and data_variables):
            raise ValueError("data_variables must be a non-empty dictionary.")
        if input_frames < 1:
            raise ValueError("input_frames must be at least 1.")
        if input_frames > 1 and transformer_config is None:
            raise ValueError("transformer_config must be provided if input_frames > 1.")

        self.data_variables = data_variables
        self.input_frames = input_frames
        self.height = height
        self.width = width
        self.latent_dim = latent_dim
        self.use_checkpoint = use_checkpoint
        self.predict_re = predict_re
        self.re_grad_enabled = re_grad_enabled
        self.total_input_channels = sum(self.data_variables.values())

        # --- Module Initialization ---
        common_args = {
            "H": height,
            "W": width,
            "latent_dim": latent_dim,
            "re_embedding_dim": re_embedding_dim,
            "re_cond_type": re_cond_type,
            "hiddens": hidden_dims,
            "block_size": block_size,
            "kernel_size": kernel_size,
            "use_checkpoint": use_checkpoint,
            **conv_kwargs,
        }

        self.history_encoder = None
        if self.input_frames > 1:
            assert (
                transformer_config is not None
            ), "transformer_config must be provided if input_frames > 1"
            self.history_encoder = HistoryEncoder(
                C=self.total_input_channels,
                transformer_config=transformer_config,
                **common_args,
            )

        self.encoder = ConvEncoder(C=self.total_input_channels, **common_args)
        self.decoder = ConvDecoder(C=self.total_input_channels, **common_args)

        self.koopman_operator = KoopmanOperator(
            latent_dim=latent_dim, mode=operator_mode, use_checkpoint=use_checkpoint
        )
        self.re_predictor = (
            Re(latent_dim=latent_dim, use_checkpoint=use_checkpoint)
            if predict_re
            else None
        )

    def encode(self, x: TensorDict, re_input: Optional[Tensor] = None) -> Tensor:
        """Encodes input data (history + present) into the latent space."""
        # --- Prepare Present Frame ---
        present_list = []
        for var, num_channels in self.data_variables.items():
            tensor_slice = x[var][:, -1]
            if num_channels == 1 and tensor_slice.ndim == 3:
                tensor_slice = tensor_slice.unsqueeze(1)
            present_list.append(tensor_slice)
        stacked_present = torch.cat(present_list, dim=1)

        re_present = (
            re_input[:, -1] if re_input is not None and re_input.ndim == 2 else re_input
        )
        latent_present = self.encoder(stacked_present, re=re_present)

        if self.input_frames > 1 and self.history_encoder is not None:
            history_list = []
            for var, num_channels in self.data_variables.items():
                tensor_slice = x[var][:, :-1]
                if num_channels == 1 and tensor_slice.ndim == 4:
                    tensor_slice = tensor_slice.unsqueeze(2)
                history_list.append(tensor_slice)
            stacked_history = torch.cat(history_list, dim=2)

            re_history = re_input[:, :-1] if re_input is not None else None
            latent_history = self.history_encoder(stacked_history, re=re_history)
            return latent_history + latent_present

        return latent_present

    def decode(
        self,
        z: Tensor,
        re_decode: Optional[Tensor] = None,
        obstacle_mask: Optional[Tensor] = None,
    ) -> TensorDict:
        """Decodes a batch of latent states back to the physical domain, conditioned on Re."""
        reconstructed_channels = self.decoder(z, re=re_decode)

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
        obstacle_mask = x.get("obstacle_mask")
        re_input = x.get("Re_input")
        keys_to_exclude = []
        if "obstacle_mask" in x:
            keys_to_exclude.append("obstacle_mask")
        if "Re_input" in x:
            keys_to_exclude.append("Re_input")

        # Exclude the keys that were found
        x_data = x.exclude(*keys_to_exclude)

        if isinstance(seq_length, Tensor):
            # Flatten the tensor and take the first element, assuming it's the
            # same for the whole batch. This handles 0D, 1D, and 2D tensors.
            seq_length_int = int(seq_length.view(-1)[0].item())
        else:
            # Handle cases where seq_length is already a number (e.g., int, float)
            seq_length_int = int(seq_length)

        # Encode initial state
        z0 = self.encode(x_data, re_input=re_input)
        re_decode_cond = (
            re_input[:, -1] if re_input is not None and re_input.ndim == 2 else re_input
        )

        # Autoregressive Rollout
        z_preds_list = []
        if seq_length_int > 0:
            z_current = z0
            for _ in range(seq_length_int):
                z_current = self.koopman_operator(z_current)
                z_preds_list.append(z_current)

        z_preds_stacked = (
            torch.stack(z_preds_list, dim=1)
            if z_preds_list
            else torch.empty(z0.size(0), 0, self.latent_dim, device=z0.device)
        )

        # Decode reconstruction and predictions
        x_recon = self.decode(z0, re_decode=re_decode_cond, obstacle_mask=obstacle_mask)

        if seq_length_int > 0:
            future_z_batch = z_preds_stacked.view(-1, self.latent_dim)
            # For all future predictions, we use the Reynolds number of the last known frame
            re_future_batch = (
                re_decode_cond.repeat_interleave(seq_length_int, dim=0)
                if re_decode_cond is not None
                else None
            )
            decoded_batch = self.decode(
                future_z_batch, re_decode=re_future_batch, obstacle_mask=obstacle_mask
            )
            x_preds = decoded_batch.apply(
                lambda t: t.view(z0.size(0), seq_length_int, *t.shape[1:]),
                batch_size=[z0.size(0), seq_length_int],
            )
        else:
            x_preds = TensorDict({}, batch_size=[z0.size(0), 0])

        # Predict Reynolds Number from all predicted latent states
        reynolds = None
        if self.predict_re and self.re_predictor is not None:
            z_all = torch.cat(
                [z0.unsqueeze(1), z_preds_stacked], dim=1
            )  # Include initial state
            z_for_re = z_all if self.re_grad_enabled else z_all.detach()
            reynolds = self.re_predictor(z_for_re)

        return KoopmanOutput(
            x_recon=x_recon, x_preds=x_preds, z_preds=z_preds_stacked, reynolds=reynolds
        )
