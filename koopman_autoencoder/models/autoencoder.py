import torch
from torch import nn
from tensordict import TensorDict
from torch import Tensor
from typing import Union, Tuple, Optional, List, Dict, Literal
from collections.abc import Mapping
from dataclasses import dataclass
import logging

# Assume these are correctly defined elsewhere
from .utils import cuda_timer, elapsed_time
from .networks import (
    ConvEncoder,
    ConvDecoder,
    HistoryEncoder,
    TransformerConfig,
)
from .modules import (
    KoopmanOperator,
    Re,
)

# Import locally to avoid circular dependencies if factories are in .rbf
from .rbf import re_expansion, ma_expansion, forcing_expansion

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
    dz_dt: Optional[Tensor] = None
    dz_dt_disturbed: Optional[Tensor] = None


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
        cond_embedding_dim: Optional[int] = 64,
        cond_type: Literal[None, "late_fusion", "adaln"] = None,
        operator_mode: Literal["linear", "eigen", "mlp"] = "linear",
        hidden_dims: List[int] = [64, 128, 64],
        block_size: int = 2,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        use_checkpoint: bool = False,
        transformer_config: Optional[TransformerConfig] = None,
        predict_cond: bool = False,
        cond_grad_enabled: bool = False,
        disturb_std: float = 1e-2,
        is_continuous: bool = False,
        rank: int = 4,
        cond_expansion_type: Optional[str] = None,
        use_attention: bool = True,
        spectral: bool = False,
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
        self.cond_type = cond_type
        self.total_input_channels = sum(self.data_variables.values())
        self.use_checkpoint = use_checkpoint
        self.predict_cond = predict_cond
        self.cond_grad_enabled = cond_grad_enabled
        self.disturb_std = disturb_std
        self.use_attention = use_attention
        self.timings: Dict = {}

        # 1. Establish the Single Source of Truth for Physics Embeddings
        # Ensure we have a concrete integer for dimensions (default to 64 if None)
        if cond_embedding_dim is not None:
            self.shared_embed_dim = cond_embedding_dim

            # Instantiate the Learnable RBFs ONCE.
            # These objects hold the learnable centers/gammas that must stay synchronized.
            self.shared_expansion_map = nn.ModuleDict(
                {
                    "Re": re_expansion(self.shared_embed_dim),
                    "Ma": ma_expansion(self.shared_embed_dim),
                    "forcing": forcing_expansion(self.shared_embed_dim),
                }
            )
        else:
            self.shared_expansion_map = None

        # --- Module Initialization ---
        common_args = {
            "H": height,
            "W": width,
            "latent_dim": latent_dim,
            "hiddens": hidden_dims,
            "block_size": block_size,
            "kernel_size": kernel_size,
            "use_checkpoint": use_checkpoint,
            "spectral": spectral,
            **conv_kwargs,
        }

        self.encoder = ConvEncoder(
            C=self.total_input_channels,
            cond_type=self.cond_type,
            cond_expansion_type=cond_expansion_type,
            shared_expansion_map=self.shared_expansion_map,
            use_attention=self.use_attention,
            **common_args,
        )

        self.history_encoder = None
        if self.input_frames > 1:
            assert (
                transformer_config is not None
            ), f"Expected valid transformer config but got {type(transformer_config)} instead"

            # Pass the SHARED self.encoder as the backbone.
            # HistoryEncoder now uses composition (stores backbone) instead of inheritance.
            self.history_encoder = HistoryEncoder(
                backbone=self.encoder,
                transformer_config=transformer_config,
                use_positional_encoding=True,  # defaulting to True as per previous logic
            )

        self.decoder = ConvDecoder(C=self.total_input_channels, **common_args)

        if cond_type is not None:
            assert isinstance(
                cond_embedding_dim, int
            ), f"expected 'cond_embedding_dim' to be int but got {type(cond_embedding_dim)} instead."
        self.koopman_operator = KoopmanOperator(
            latent_dim=latent_dim,
            cond_embedding_dim=cond_embedding_dim,  # Force sync
            mode=operator_mode,
            use_checkpoint=use_checkpoint,
            is_continuous=is_continuous,
            rank=rank,
            cond_expansion_type=cond_expansion_type,
            shared_expansion_map=self.shared_expansion_map,  # Force sync
        )
        self.re_predictor = (
            Re(latent_dim=latent_dim, use_checkpoint=use_checkpoint)
            if predict_cond
            else None
        )

    def re_norm(self, re_value: Optional[Tensor]) -> Optional[Tensor]:
        """Applies Z-score normalization to a Reynolds number tensor."""
        if re_value is None:
            return None
        # RE_MEAN, RE_STD = 550.0, 261.1513
        # return (re_value - RE_MEAN) / RE_STD
        return re_value

    def present_encoding(
        self, x: TensorDict, cond_input: Optional[Tensor] = None
    ) -> Tensor:
        """Encodes present data into the latent space."""
        present_list = [x[var][:, -1] for var in self.data_variables]
        for i, (var, num_channels) in enumerate(self.data_variables.items()):
            if num_channels == 1 and present_list[i].ndim == 3:
                present_list[i] = present_list[i].unsqueeze(1)
        stacked_present = torch.cat(present_list, dim=1)
        if cond_input is not None:
            cond_input = cond_input[
                ..., -1:
            ]  # Get condition for the last (present) frame
        return self.encoder(stacked_present, cond=cond_input)

    def history_encoding(
        self, x: TensorDict, cond_input: Optional[Tensor] = None
    ) -> Tensor:
        """Encodes history data into the latent space."""

        assert self.history_encoder is not None, "history_encoder was not initialized"

        history_list = [x[var][:, :-1] for var in self.data_variables]
        for i, (var, num_channels) in enumerate(self.data_variables.items()):
            if num_channels == 1 and history_list[i].ndim == 4:
                history_list[i] = history_list[i].unsqueeze(2)
        stacked_history = torch.cat(history_list, dim=2)
        cond_history = cond_input[:, :-1] if cond_input is not None else None
        latent_history = self.history_encoder(stacked_history, cond=cond_history)
        return latent_history

    def encode(self, x: TensorDict, cond_input: Optional[Tensor] = None) -> Tensor:
        """Encodes input data (history + present) into the latent space."""
        latent_present = self.present_encoding(x=x, cond_input=cond_input)

        if self.input_frames > 1 and self.history_encoder is not None:
            latent_history = self.history_encoding(x=x, cond_input=cond_input)
            return (latent_history + latent_present) / 2
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

        cond_tensor = x.get("cond_input")

        # Check for configuration mismatch
        if self.cond_type in ["late_fusion", "adaln"] and cond_tensor is None:
            raise ValueError(
                f"Model is configured with cond_type='{self.cond_type}', "
                "but no 'cond_input' tensor was found in the batch. "
                "Ensure 'data.selection_param' is set in your config."
            )

        cond_input = self.re_norm(cond_tensor)

        # Dynamically find all metadata keys to exclude from x_data
        keys_to_exclude = [
            k
            for k in x.keys()
            if k not in self.data_variables and k in x  # k in x is redundant but safe
        ]
        if "obstacle_mask" not in keys_to_exclude and "obstacle_mask" in x:
            keys_to_exclude.append("obstacle_mask")
        if "cond_input" not in keys_to_exclude and "cond_input" in x:
            keys_to_exclude.append("cond_input")

        # x_data = x.exclude(*keys_to_exclude)

        seq_len_int = (
            int(seq_length.view(-1)[0].item())
            if isinstance(seq_length, Tensor)
            else int(seq_length)
        )
        # return obstacle_mask, cond_input, x_data, seq_len_int
        return obstacle_mask, cond_input, x, seq_len_int

    def _encode_initial_states(self, x_data: TensorDict, cond_input: Optional[Tensor]):
        """Encodes initial states and creates a perturbed version if needed."""
        z0 = self.encode(x_data, cond_input=cond_input)
        z0_disturbed = None
        if self.disturb_std is not None:
            noise = torch.randn_like(z0) * self.disturb_std
            z0_disturbed = z0 + noise
        return z0, z0_disturbed

    def _autoregressive_rollout(
        self, z_init: Tensor, seq_length: int, cond_for_prediction: Optional[Tensor]
    ):
        """Performs autoregressive rollout in the latent space."""
        if seq_length <= 0:
            return torch.empty(z_init.size(0), 0, self.latent_dim, device=z_init.device)

        z_preds_list = []
        z_current = z_init
        for frame in range(seq_length):
            if self.disturb_std is not None:
                z_current = z_current + torch.randn_like(z_current) * self.disturb_std
            else:
                pass
            # Use the condition for the *specific* frame
            frame_cond = (
                cond_for_prediction[:, frame]
                if cond_for_prediction is not None
                else None
            )

            z_current = self.koopman_operator(z_current, cond=frame_cond)
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

    def forward(
        self,
        x: TensorDict,
        seq_length: Union[int, Tensor],
        cond_future: Optional[Tensor] = None,  # This arg is specific, let's keep it
    ) -> KoopmanOutput:
        """Forward pass: Encode, roll out predictions, and decode."""
        total_start, total_end = cuda_timer()
        total_start.record()

        # 1. Prepare inputs
        obstacle_mask, cond_input, x_data, seq_length_int = self._prepare_inputs(
            x, seq_length
        )

        # 2. Encode initial states (original and optionally disturbed)
        start, end = cuda_timer()
        start.record()
        z0, z0_disturbed = self._encode_initial_states(x_data, cond_input)
        end.record()
        torch.cuda.synchronize()
        self.timings["encode"] = elapsed_time(start, end)

        # 3. Get conditioning tensor for the prediction phase
        # We need to find the *target* condition, not the input one
        cond_for_prediction = x.get("cond_target")

        if cond_for_prediction is None:
            # Fallback: repeat the last *input* condition
            # This is less ideal but maintains behavior
            last_input_cond = (
                cond_input[:, -1]
                if (cond_input is not None and cond_input.ndim == 2)
                else cond_input
            )
            if last_input_cond is not None:
                cond_for_prediction = last_input_cond.view(-1, 1).repeat(
                    1, seq_length_int
                )

        # Ensure it's normalized if it exists
        if cond_for_prediction is not None:
            cond_for_prediction = self.re_norm(cond_for_prediction)

        # 4. Perform autoregressive rollout for the main trajectory
        start, end = cuda_timer()
        start.record()
        z_preds_stacked = self._autoregressive_rollout(
            z0, seq_length_int, cond_for_prediction
        )
        end.record()
        torch.cuda.synchronize()
        self.timings["rollout"] = elapsed_time(start, end)

        # 5. Perform rollout for the disturbed trajectory if needed for stability loss
        disturbed_latents = None
        dz_dt, dz_dt_disturbed = None, None
        if z0_disturbed is not None:
            disturbed_latents = self._autoregressive_rollout(
                z0_disturbed, seq_length_int, cond_for_prediction
            )
            # We detach to ensure this loss only affects the operator, not the encoder
            if self.koopman_operator.is_continuous and hasattr(
                self.koopman_operator.dynamics, "_get_derivative"
            ):
                cond_encoded = self.koopman_operator.dynamics._encode_cond(
                    cond=cond_for_prediction[..., 0]
                )
                dz_dt = self.koopman_operator.dynamics._get_derivative(
                    z0.detach(), cond_encoded=cond_encoded
                )
                dz_dt_disturbed = self.koopman_operator.dynamics._get_derivative(
                    z0_disturbed.detach(), cond_encoded=cond_encoded
                )

        # 6. Decode outputs for the main trajectory
        start, end = cuda_timer()
        start.record()
        if self.training:
            # Add small noise to z before decoding to force decoder robustness
            z0 = z0 + torch.randn_like(z0) * 0.01

        x_recon, x_preds = self._decode_outputs(
            z0, z_preds_stacked, seq_length_int, obstacle_mask
        )
        end.record()
        torch.cuda.synchronize()
        self.timings["decode"] = elapsed_time(start, end)

        # 7. Predict Reynolds number from the main trajectory (optional)
        reynolds = None
        if self.predict_cond and self.re_predictor is not None:
            z_all = torch.cat([z0.unsqueeze(1), z_preds_stacked], dim=1)
            z_for_re = z_all if self.cond_grad_enabled else z_all.detach()
            reynolds = self.re_predictor(z_for_re)

        total_end.record()
        torch.cuda.synchronize()
        self.timings["total"] = elapsed_time(total_start, total_end)
        # np.save("kae_times", np.array(self.timings["total"]))

        # 8. Return the final output structure
        return KoopmanOutput(
            x_recon=x_recon,
            x_preds=x_preds,
            z_preds=z_preds_stacked,
            reynolds=reynolds,
            disturbed_latents=disturbed_latents,
            dz_dt=dz_dt,
            dz_dt_disturbed=dz_dt_disturbed,
        )

    def compute_theoretical_evolution(
        self,
        z0: Tensor,
        num_frames: Union[int, Tensor],
        cond: Optional[Tensor] = None,
        dt: float = 0.1,
    ) -> Tensor:
        """
        Computes the theoretical state evolution using the matrix exponential.
        """
        # 1. Access the dynamics internal module
        dyn = self.koopman_operator.dynamics

        # 2. Encode conditioning
        cond_encoded = dyn._encode_cond(cond) if cond is not None else None

        # 3. Extract the Koopman Matrix A
        # A: [D, D] or [Batch, D, D]
        A = dyn._get_effective_linear_map(cond_encoded)

        # 4. Handle num_frames (Convert Tensor to int if necessary)
        if isinstance(num_frames, Tensor):
            num_frames = int(num_frames.view(-1)[0].item())

        # 5. Pre-compute the time grid
        # arange(1, num_frames + 1) matches autoregressive indices [1, 2, ..., T]
        t_steps = torch.arange(1, num_frames + 1, device=z0.device).float() * dt

        z_theoretical = []
        for t in t_steps:
            # Analytical solution: z(t) = exp(A * t) @ z0
            exp_At = torch.matrix_exp(A * t)

            if A.ndim == 2:  # Shared matrix across batch
                zt = torch.matmul(z0, exp_At.t())
            else:  # Unique matrix per batch element (LoRA/Conditioned)
                zt = torch.bmm(exp_At, z0.unsqueeze(-1)).squeeze(-1)

            z_theoretical.append(zt)

        # Stack to [Batch, num_frames, Latent_dim] and ensure it's real
        return torch.stack(z_theoretical, dim=1).real

    def forward_theoretical(
        self,
        x: TensorDict,
        seq_length: Union[int, Tensor],
        dt: float = 0.1,
        cond_future: Optional[Tensor] = None,
    ) -> KoopmanOutput:
        """
        Forward pass using the analytical matrix exponential solution
        instead of autoregressive rollout.
        """
        # 1. Prepare inputs (standard procedure)
        obstacle_mask, cond_input, x_data, seq_length_int = self._prepare_inputs(
            x, seq_length
        )

        # 2. Encode initial state z0
        # We don't usually disturb for theoretical analysis, but we keep logic consistent
        z0, _ = self._encode_initial_states(x_data, cond_input)

        # 3. Handle Conditioning
        cond_for_prediction = x.get("cond_target")
        if cond_for_prediction is None and cond_input is not None:
            # Use the last known condition if target isn't provided
            cond_for_prediction = (
                cond_input[:, -1].view(-1, 1).repeat(1, seq_length_int)
            )

        if cond_for_prediction is not None:
            cond_for_prediction = self.re_norm(cond_for_prediction)

        # We pass the average condition if the operator is condition-dependent
        # Alternatively, if A changes per step, the matrix exponential is more complex.
        # Here we assume A is constant over the rollout period for the 'theoretical' base.
        mean_cond = (
            cond_for_prediction.mean(dim=1) if cond_for_prediction is not None else None
        )

        z_preds_theoretical = self.compute_theoretical_evolution(
            z0=z0, num_frames=seq_length, cond=mean_cond
        )

        # 5. Decode outputs
        x_recon, x_preds = self._decode_outputs(
            z0, z_preds_theoretical, seq_length_int, obstacle_mask
        )

        # 6. Predict Reynolds (Optional)
        reynolds = None
        if self.predict_cond and self.re_predictor is not None:
            z_all = torch.cat([z0.unsqueeze(1), z_preds_theoretical], dim=1)
            reynolds = self.re_predictor(z_all.detach())

        return KoopmanOutput(
            x_recon=x_recon,
            x_preds=x_preds,
            z_preds=z_preds_theoretical,
            reynolds=reynolds,
            disturbed_latents=None,
            dz_dt=None,
            dz_dt_disturbed=None,
        )

    def compute_stabilized_evolution(
        self,
        z0: Tensor,
        num_frames: Union[int, Tensor],
        cond: Optional[Tensor] = None,
        stabilization_gamma: float = 0.017,
        dt: float = 0.1,
        method: Literal["euler", "rk4"] = "euler",
    ) -> Tensor:
        """
        Integrates the latent dynamics while injecting energy ('gamma')
        to counteract the model's learned damping.

        Args:
            z0: Initial latent state [Batch, Latent]
            num_frames: Number of steps to roll out
            cond: Physics condition [Batch, Cond_Dim] (optional)
            stabilization_gamma: Energy injection rate (default 0.017 matches your spectrum)
            dt: Time step
            method: Integration scheme ('euler' is usually sufficient for stabilization)
        """
        # 1. Setup
        if isinstance(num_frames, Tensor):
            num_frames = int(num_frames.view(-1)[0].item())

        # Ensure we can access the dynamics module for derivatives
        if not (
            self.koopman_operator.is_continuous
            and hasattr(self.koopman_operator.dynamics, "_get_derivative")
        ):
            # Fallback for discrete or MLP operators: simple multiplicative boost
            # z_{t+1} = Op(z_t) * (1 + gamma)
            z_current = z0
            preds = []
            for _ in range(num_frames):
                # We assume cond is static for the rollout here; if dynamic, logic changes slightly
                z_current = self.koopman_operator(z_current, cond=cond)
                z_current = z_current * (1.0 + stabilization_gamma)
                preds.append(z_current)
            return torch.stack(preds, dim=1)

        # 2. Continuous Logic (Physics Derivative)
        dyn = self.koopman_operator.dynamics
        cond_encoded = dyn._encode_cond(cond) if cond is not None else None

        z_current = z0
        z_preds = []

        for _ in range(num_frames):
            if method == "euler":
                # k1 = f(z)
                dz = dyn._get_derivative(z_current, cond_encoded=cond_encoded)

                # --- THE FIX: Gamma Injection ---
                # z_dot_stabilized = z_dot + gamma * z
                dz_stabilized = dz + (stabilization_gamma * z_current)

                # Step
                z_next = z_current + dz_stabilized * dt

            elif method == "rk4":
                # slightly more expensive, usually not needed just for stabilization
                # Note: We apply gamma injection to the final update, or inside the derivative function.
                # Applying inside usually consistent: f_stable(z) = f(z) + gamma*z

                def f_stable(z_in):
                    d = dyn._get_derivative(z_in, cond_encoded=cond_encoded)
                    return d + (stabilization_gamma * z_in)

                k1 = f_stable(z_current)
                k2 = f_stable(z_current + dt * k1 / 2)
                k3 = f_stable(z_current + dt * k2 / 2)
                k4 = f_stable(z_current + dt * k3)

                z_next = z_current + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            z_preds.append(z_next)
            z_current = z_next

        return torch.stack(z_preds, dim=1)

    def forward_stabilized(
        self,
        x: TensorDict,
        seq_length: Union[int, Tensor],
        stabilization_gamma: float = 0.017,
        dt: float = 0.1,
        cond_future: Optional[Tensor] = None,
    ) -> KoopmanOutput:
        """
        Forward pass that manually injects energy to prevent long-term diffusion.
        Use this for validation/inference on long rollouts.
        """
        # 1. Prepare inputs
        obstacle_mask, cond_input, x_data, seq_length_int = self._prepare_inputs(
            x, seq_length
        )

        # 2. Encode initial state z0
        # We don't disturb latents during stable inference usually
        z0, _ = self._encode_initial_states(x_data, cond_input)

        # 3. Handle Conditioning
        # We need a single condition vector for the rollout (assuming constant physics params like Re)
        cond_for_prediction = x.get("cond_target")

        # Fallback to input condition if target missing
        if cond_for_prediction is None and cond_input is not None:
            # Take the last frame's condition
            cond_vec = cond_input[:, -1]
        elif cond_for_prediction is not None:
            # If target provided (e.g. sequence of Re), take mean or first
            # Usually Re is constant over time, so mean/first is same
            cond_vec = cond_for_prediction[:, 0]
        else:
            cond_vec = None

        if cond_vec is not None:
            cond_vec = self.re_norm(cond_vec)

        # 4. Compute Stabilized Evolution (The Core Logic)
        z_preds_stable = self.compute_stabilized_evolution(
            z0=z0,
            num_frames=seq_length_int,
            cond=cond_vec,
            stabilization_gamma=stabilization_gamma,
            dt=dt,
        )

        # 5. Decode outputs
        # Note: Decoder condition needs to be repeated for time dimension
        x_recon = self.decode(z0, obstacle_mask=obstacle_mask)

        x_preds = TensorDict({}, batch_size=[z0.size(0), 0])
        if seq_length_int > 0:
            # Flatten B, T -> B*T for decoding
            B, T, D = z_preds_stable.shape
            z_flat = z_preds_stable.reshape(B * T, D)

            decoded_flat = self.decode(z_flat, obstacle_mask=obstacle_mask)

            # Reshape back to [B, T, C, H, W]
            x_preds = decoded_flat.apply(
                lambda t: t.view(B, T, *t.shape[1:]),
                batch_size=[B, T],
            )

        # 6. Predict Reynolds (Optional)
        reynolds = None
        if self.predict_cond and self.re_predictor is not None:
            z_all = torch.cat([z0.unsqueeze(1), z_preds_stable], dim=1)
            reynolds = self.re_predictor(z_all.detach())

        return KoopmanOutput(
            x_recon=x_recon,
            x_preds=x_preds,
            z_preds=z_preds_stable,
            reynolds=reynolds,
            disturbed_latents=None,
            dz_dt=None,
            dz_dt_disturbed=None,
        )
