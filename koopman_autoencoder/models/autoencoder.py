import torch
from torch import nn
from tensordict import TensorDict
from torch import Tensor
from typing import Union, Tuple, Optional, List, Dict, Literal
from collections.abc import Mapping
from dataclasses import dataclass
import logging

# Assume these are correctly defined elsewhere
from models.utils import cuda_timer, elapsed_time
from models.networks import (
    ConvEncoder,
    ConvDecoder,
    HistoryEncoder,
    TransformerConfig,
    KoopmanOperator,
    Re,
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
    disturbed_latents: Optional[Tensor]


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
        disturb_std: float = 1e-2,
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
        self.latent_dim = latent_dim
        self.total_input_channels = sum(self.data_variables.values())
        self.use_checkpoint = use_checkpoint
        self.predict_re = predict_re
        self.re_grad_enabled = re_grad_enabled
        self.disturb_std = disturb_std
        self.timings: Dict = {}

        # --- Module Initialization ---
        common_args = {
            "H": height,
            "W": width,
            "latent_dim": latent_dim,
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
            ), f"Expected valid transformer config but got {type(transformer_config)} instead"
            self.history_encoder = HistoryEncoder(
                C=self.total_input_channels,
                transformer_config=transformer_config,
                re_embedding_dim=re_embedding_dim,
                re_cond_type=re_cond_type,
                **common_args,
            )

        self.encoder = ConvEncoder(C=self.total_input_channels, **common_args)
        self.decoder = ConvDecoder(C=self.total_input_channels, **common_args)

        self.koopman_operator = KoopmanOperator(
            latent_dim=latent_dim,
            re_embedding_dim=re_embedding_dim,
            mode=operator_mode,
            use_checkpoint=use_checkpoint,
        )
        self.re_predictor = (
            Re(latent_dim=latent_dim, use_checkpoint=use_checkpoint)
            if predict_re
            else None
        )

    def re_norm(self, re_value: Optional[Tensor]) -> Optional[Tensor]:
        """Applies Z-score normalization to a Reynolds number tensor."""
        if re_value is None:
            return None
        RE_MEAN, RE_STD = 550.0, 261.1513
        return (re_value - RE_MEAN) / RE_STD

    def encode(self, x: TensorDict, re_input: Optional[Tensor] = None) -> Tensor:
        """Encodes input data (history + present) into the latent space."""
        present_list = [x[var][:, -1] for var in self.data_variables]
        for i, (var, num_channels) in enumerate(self.data_variables.items()):
            if num_channels == 1 and present_list[i].ndim == 3:
                present_list[i] = present_list[i].unsqueeze(1)
        stacked_present = torch.cat(present_list, dim=1)
        latent_present = self.encoder(stacked_present)

        if self.input_frames > 1 and self.history_encoder is not None:
            history_list = [x[var][:, :-1] for var in self.data_variables]
            for i, (var, num_channels) in enumerate(self.data_variables.items()):
                if num_channels == 1 and history_list[i].ndim == 4:
                    history_list[i] = history_list[i].unsqueeze(2)
            stacked_history = torch.cat(history_list, dim=2)
            re_history = re_input[:, :-1] if re_input is not None else None
            latent_history = self.history_encoder(stacked_history, re=re_history)
            return latent_history + latent_present
        return latent_present

    def decode(self, z: Tensor, obstacle_mask: Optional[Tensor] = None) -> TensorDict:
        """Decodes a batch of latent states back to the physical domain."""
        reconstructed_channels = self.decoder(z)
        if obstacle_mask is not None:
            mask = obstacle_mask[0, 0] if obstacle_mask.ndim == 4 else obstacle_mask
            reconstructed_channels = reconstructed_channels * mask[None, None, :, :]

        decoded_data = {}
        current_channel = 0
        for var, num_channels in self.data_variables.items():
            end_channel = current_channel + num_channels
            var_tensor = reconstructed_channels[:, current_channel:end_channel]
            decoded_data[var] = (
                var_tensor.squeeze(1) if num_channels == 1 else var_tensor
            )
            current_channel = end_channel
        return TensorDict(decoded_data, batch_size=[z.size(0)])

    def _prepare_inputs(self, x: TensorDict, seq_length: Union[int, Tensor]):
        """Handles input preparation and normalization."""
        obstacle_mask = x.get("obstacle_mask")
        re_input = self.re_norm(x.get("Re_input"))
        keys_to_exclude = [k for k in ["obstacle_mask", "Re_input"] if k in x]
        x_data = x.exclude(*keys_to_exclude)
        seq_len_int = (
            int(seq_length.view(-1)[0].item())
            if isinstance(seq_length, Tensor)
            else int(seq_length)
        )
        return obstacle_mask, re_input, x_data, seq_len_int

    def _encode_initial_states(self, x_data: TensorDict, re_input: Optional[Tensor]):
        """Encodes initial states and creates a perturbed version if needed."""
        z0 = self.encode(x_data, re_input=re_input)
        z0_disturbed = None
        if self.training and (self.disturb_std is not None):
            noise = torch.randn_like(z0) * self.disturb_std
            z0_disturbed = z0 + noise
        return z0, z0_disturbed

    def _autoregressive_rollout(
        self, z_init: Tensor, seq_length: int, re_for_prediction: Optional[Tensor]
    ):
        """Performs autoregressive rollout in the latent space."""
        if seq_length <= 0:
            return torch.empty(z_init.size(0), 0, self.latent_dim, device=z_init.device)

        z_preds_list = []
        z_current = z_init
        for _ in range(seq_length):
            z_current = self.koopman_operator(z_current, re=re_for_prediction)
            z_preds_list.append(z_current)
        return torch.stack(z_preds_list, dim=1)

    def _decode_outputs(
        self,
        z0: Tensor,
        z_preds_stacked: Tensor,
        seq_length: int,
        obstacle_mask: Optional[Tensor],
    ):
        """Decodes the reconstruction and prediction sequences."""
        x_recon = self.decode(z0, obstacle_mask=obstacle_mask)
        if seq_length > 0:
            future_z_batch = z_preds_stacked.view(-1, self.latent_dim)
            decoded_batch = self.decode(future_z_batch, obstacle_mask=obstacle_mask)
            x_preds = decoded_batch.apply(
                lambda t: t.view(z0.size(0), seq_length, *t.shape[1:]),
                batch_size=[z0.size(0), seq_length],
            )
        else:
            x_preds = TensorDict({}, batch_size=[z0.size(0), 0])
        return x_recon, x_preds

    def forward(self, x: TensorDict, seq_length: Union[int, Tensor]) -> KoopmanOutput:
        """Forward pass: Encode, roll out predictions, and decode."""
        total_start, total_end = cuda_timer()
        total_start.record()
        # 1. Prepare inputs
        obstacle_mask, re_input, x_data, seq_length_int = self._prepare_inputs(
            x, seq_length
        )

        # 2. Encode initial states (original and optionally disturbed)
        start, end = cuda_timer()
        start.record()
        z0, z0_disturbed = self._encode_initial_states(x_data, re_input)
        end.record()
        torch.cuda.synchronize()
        self.timings["encode"] = elapsed_time(start, end)

        # 3. Get conditioning Reynolds number for the prediction phase
        re_for_prediction = (
            re_input[:, -1]
            if (re_input is not None and re_input.ndim == 2)
            else re_input
        )

        # 4. Perform autoregressive rollout for the main trajectory
        start, end = cuda_timer()
        start.record()
        z_preds_stacked = self._autoregressive_rollout(
            z0, seq_length_int, re_for_prediction
        )
        end.record()
        torch.cuda.synchronize()
        self.timings["rollout"] = elapsed_time(start, end)

        # 5. Perform rollout for the disturbed trajectory if needed for stability loss
        disturbed_latents = None
        if z0_disturbed is not None:
            disturbed_latents = self._autoregressive_rollout(
                z0_disturbed, seq_length_int, re_for_prediction
            )

        # 6. Decode outputs for the main trajectory
        start, end = cuda_timer()
        start.record()
        x_recon, x_preds = self._decode_outputs(
            z0, z_preds_stacked, seq_length_int, obstacle_mask
        )
        end.record()
        torch.cuda.synchronize()
        self.timings["decode"] = elapsed_time(start, end)

        # 7. Predict Reynolds number from the main trajectory (optional)
        reynolds = None
        if self.predict_re and self.re_predictor is not None:
            z_all = torch.cat([z0.unsqueeze(1), z_preds_stacked], dim=1)
            z_for_re = z_all if self.re_grad_enabled else z_all.detach()
            reynolds = self.re_predictor(z_for_re)

        total_end.record()
        torch.cuda.synchronize()
        self.timings["total"] = elapsed_time(total_start, total_end)

        # 8. Return the final output structure
        return KoopmanOutput(
            x_recon=x_recon,
            x_preds=x_preds,
            z_preds=z_preds_stacked,
            reynolds=reynolds,
            disturbed_latents=disturbed_latents,
        )
