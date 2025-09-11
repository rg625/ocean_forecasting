# models/modules.py

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import Tensor
from einops import rearrange
import abc
from typing import Optional, Literal
from dataclasses import dataclass

# Note: The following import assumes that 'networks.py' and 'modules.py' are in the same
# directory and 'networks.py' contains the definitions for ConvEncoder and AdaLNMLP.
from .networks import ConvEncoder, AdaLNMLP

# ======================================================================================
# Helper and Encoder Modules
# ======================================================================================


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
    """Configuration dataclass for the TransformerEncoder in HistoryEncoder."""

    num_layers: int = 4  # Number of transformer encoder layers
    nhead: int = 8  # Number of attention heads
    ff_mult: int = 4  # Multiplier for the feed-forward layer dimension
    max_len: int = 1000  # Maximum sequence length for positional encoding
    dropout: float = 0.1  # Dropout rate


class HistoryEncoder(ConvEncoder):
    """
    Encodes a sequence of images into a single latent vector.

    This module processes a time-series of images (e.g., video frames) by first
    encoding each image into a feature vector using a convolutional encoder, and then
    aggregating these features over time using a Transformer to produce a single
    vector representing the initial state of the system.
    """

    def __init__(
        self,
        latent_dim: int,
        use_positional_encoding: bool = True,
        transformer_config: TransformerConfig = TransformerConfig(),
        **kwargs,
    ):
        # Initialize the parent ConvEncoder with all provided arguments.
        super().__init__(latent_dim=latent_dim, **kwargs)

        # Initialize positional encoding if requested.
        self.pos_enc = (
            PositionalEncoding(latent_dim, max_len=transformer_config.max_len)
            if use_positional_encoding
            else nn.Identity()
        )
        # Initialize the TransformerEncoder, which will process the sequence of features.
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=transformer_config.nhead,
                dim_feedforward=latent_dim * transformer_config.ff_mult,
                dropout=transformer_config.dropout,
                batch_first=True,  # Important: expects input shape (B, T, D)
            ),
            num_layers=transformer_config.num_layers,
        )

    def forward(self, x: Tensor, re: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for the HistoryEncoder.

        Args:
            x (Tensor): Input tensor of image frames. Shape: (B, T, C, H, W).
            re (Optional[Tensor]): Optional Reynolds number conditioning. Shape: (B, T).

        Returns:
            Tensor: A single latent vector representing the sequence. Shape: (B, D).
        """
        B, T, C, H, W = x.shape
        # Flatten the batch and time dimensions to process all images at once.
        # Shape: (B, T, C, H, W) -> (B*T, C, H, W)
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")

        # Prepare the Reynolds number tensor for batch processing if provided.
        re_expanded = None
        if self.re_cond_type is not None:
            if re is None:
                raise ValueError(
                    f"re tensor must be provided for conditioning type '{self.re_cond_type}'"
                )
            if re.ndim != 2 or re.shape != (B, T):
                raise ValueError(
                    f"Expected Re tensor of shape (B, T) = ({B}, {T}), but got {re.shape}"
                )
            # Flatten Re from (B, T) to (B*T,)
            re_expanded = re.reshape(-1)

        # Pass the flattened images through the parent ConvEncoder to get features.
        # Output shape: (B*T, D)
        features = super().forward(x_flat, re=re_expanded)

        # Un-flatten the features back into a sequence for the Transformer.
        # Shape: (B*T, D) -> (B, T, D)
        features = rearrange(features, "(b t) d -> b t d", t=T)

        # Add positional information and process with the Transformer.
        features = self.pos_enc(features)
        out = self.transformer(features)

        # Pool the features over the time dimension to get a single vector per sequence.
        return out.mean(dim=1)


# ======================================================================================
# ROBUST KOOPMAN OPERATOR IMPLEMENTATION
# ======================================================================================


class BaseKoopmanOperator(nn.Module, abc.ABC):
    """
    Abstract base class for Koopman operators.

    This class defines a common interface and handles shared logic (like parameter
    conditioning) for all Koopman operator implementations. Using an abstract base
    class ensures API consistency and reduces code duplication.
    """

    def __init__(
        self,
        latent_dim: int,
        re_embedding_dim: int,
        mode: Literal["linear", "eigen", "mlp"],
        assume_orthogonal_eigenvectors: bool,
        use_checkpoint: bool,
    ):
        super().__init__()
        if mode not in ["linear", "eigen", "mlp"]:
            raise ValueError(f"Mode '{mode}' is not supported.")

        self.latent_dim = latent_dim
        self.mode = mode
        self.assume_orthogonal = assume_orthogonal_eigenvectors
        self.use_checkpoint = use_checkpoint
        self.conditioner = AdaLNMLP(latent_dim, re_embedding_dim)

    def _apply_conditioning(self, z: Tensor, re: Optional[Tensor]) -> Tensor:
        """
        Applies AdaLNMLP conditioning to the output tensor if `re` is provided.
        This models a parameter-dependent forcing or adjustment term.
        """
        if re is not None:
            return self.conditioner(z, re)
        return z

    @abc.abstractmethod
    def forward(self, z: Tensor, re: Optional[Tensor], dt: Optional[float]) -> Tensor:
        """Abstract method to evolve the state `z` by one step."""
        raise NotImplementedError


class DiscreteKoopmanOperator(BaseKoopmanOperator):
    """
    Koopman operator for discrete-time systems.

    This model learns a one-step prediction in latent space, representing the
    dynamics as z_{t+1} = K(z_t, Re).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.mode == "linear":
            self.K = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        elif self.mode == "eigen":
            # --- Parameters for Eigendecomposition ---
            # Unconstrained log-magnitude, mapped to <= 0 by softplus for stability.
            self.unconstrained_log_magnitude = nn.Parameter(
                torch.randn(self.latent_dim)
            )
            # Unconstrained angle (phase) of the eigenvalues.
            self.angle = nn.Parameter(torch.randn(self.latent_dim))
            # Eigenvectors, initialized to be an orthogonal matrix.
            eigenvectors_init = torch.randn(self.latent_dim, self.latent_dim)
            self.eigenvectors = nn.Parameter(torch.linalg.qr(eigenvectors_init).Q)
        elif self.mode == "mlp":
            self.K = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim // 8),
                nn.ReLU(),
                nn.Linear(self.latent_dim // 8, self.latent_dim),
            )

    @property
    def eigenvalues(self) -> Optional[Tensor]:
        """
        Complex eigenvalues with magnitude <= 1, ensuring discrete-time stability.
        """
        if self.mode != "eigen":
            return None
        # -softplus(x) maps any real number to (-inf, 0], ensuring log(magnitude) is non-positive.
        log_magnitude = -F.softplus(self.unconstrained_log_magnitude)
        magnitude = torch.exp(log_magnitude)  # Magnitude is now in (0, 1]
        # Combine magnitude and angle to form complex eigenvalues in polar form.
        return torch.polar(magnitude, self.angle)

    def _forward_impl(self, z: Tensor) -> Tensor:
        """The core, unconditioned, one-step operator."""
        if self.mode in ["linear", "mlp"]:
            return self.K(z)
        elif self.mode == "eigen":
            # Implements the evolution: z_{t+1} = P * diag(λ) * P_inv * z_t
            P = self.eigenvectors
            P_inv = P.T if self.assume_orthogonal else torch.linalg.pinv(P)
            z_c, P_c, P_inv_c = (
                z.to(torch.complex64),
                P.to(torch.complex64),
                P_inv.to(torch.complex64),
            )
            Lambda = torch.diag(self.eigenvalues)
            # Project z into the eigenvector basis.
            z_eig = P_inv_c @ z_c.T
            # Evolve in the eigenvector basis by scaling with eigenvalues.
            # Reconstruct the state in the original basis.
            z_recomposed = (P_c @ Lambda @ z_eig).T
            # Return the real part, as the physical state is real.
            return z_recomposed.real

    def forward(
        self, z: Tensor, re: Optional[Tensor] = None, dt: Optional[float] = None
    ) -> Tensor:
        """Evolves the state by one discrete step, ignoring dt."""
        # --- REVERTED: Conditioning is applied to the INPUT state `z` ---
        z_conditioned = self._apply_conditioning(z, re)

        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, z_conditioned, use_reentrant=True)
        else:
            return self._forward_impl(z_conditioned)


class ContinuousKoopmanOperator(BaseKoopmanOperator):
    """
    Koopman operator for continuous-time systems.

    This model learns the time derivative of the state, dz/dt = f(z, Re),
    and uses a numerical integrator (or analytical solution) to evolve the state.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The network now represents the derivative function, f(z).
        if self.mode == "linear":
            self.K = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        elif self.mode == "mlp":
            self.K = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim // 8),
                nn.SiLU(),
                nn.Linear(self.latent_dim // 8, self.latent_dim),
            )
        elif self.mode == "eigen":
            # --- Parameters for Eigendecomposition of the derivative operator ---
            # Unconstrained real part, mapped to <= 0 for stability.
            self.unconstrained_real_parts = nn.Parameter(torch.randn(self.latent_dim))
            # Unconstrained imaginary part (frequency).
            self.imaginary_parts = nn.Parameter(torch.randn(self.latent_dim))
            # Eigenvectors of the derivative operator.
            eigenvectors_init = torch.randn(self.latent_dim, self.latent_dim)
            self.eigenvectors = nn.Parameter(torch.linalg.qr(eigenvectors_init).Q)

    @property
    def eigenvalues(self) -> Optional[Tensor]:
        """
        Complex eigenvalues with real part <= 0, ensuring continuous-time stability.
        """
        if self.mode != "eigen":
            return None
        # -softplus(x) ensures the real part is non-positive (stable decay or oscillation).
        real_part = -F.softplus(self.unconstrained_real_parts)
        return torch.complex(real_part, self.imaginary_parts)

    def _get_derivative(self, z: Tensor, re: Optional[Tensor]) -> Tensor:
        """
        Computes the derivative dz/dt = f(z), then applies conditioning.
        This models f(z, Re) where Re acts as a forcing term.
        """
        dz_dt = self.K(z)
        return self._apply_conditioning(dz_dt, re)

    def _forward_eigen(self, z: Tensor, dt: float, re: Optional[Tensor]) -> Tensor:
        """
        Applies the exact analytical solution for the linear ODE:
        z(t+dt) = P * exp(diag(λ)*dt) * P_inv * z(t), then applies conditioning.
        """
        assert self.eigenvalues is not None, "Eigenvalues must exist in 'eigen' mode."

        P = self.eigenvectors
        P_inv = P.T if self.assume_orthogonal else torch.linalg.pinv(P)
        z_c, P_c, P_inv_c = (
            z.to(torch.complex64),
            P.to(torch.complex64),
            P_inv.to(torch.complex64),
        )

        # Calculate the matrix exponential term: exp(Lambda * dt).
        # For a diagonal matrix, this is the exponential of each diagonal element.
        exp_lambda_dt = torch.exp(self.eigenvalues * dt)
        Exp_Lambda_t = torch.diag(exp_lambda_dt)

        # Project z into the eigenvector basis.
        z_eig = P_inv_c @ z_c.T
        # Evolve in the eigenvector basis.
        z_evolved_eig = Exp_Lambda_t @ z_eig
        # Reconstruct the state in the original basis.
        z_evolved = (P_c @ z_evolved_eig).T

        return self._apply_conditioning(z_evolved.real, re)

    def _forward_rk4(self, z: Tensor, dt: float, re: Optional[Tensor]) -> Tensor:
        """
        Integrates the learned derivative using the Runge-Kutta 4th Order method.
        This is a highly stable and accurate explicit numerical integration scheme.
        """
        # k1 is the slope at the beginning of the interval.
        k1 = self._get_derivative(z, re)
        # k2 is the slope at the midpoint, using k1 to step.
        k2 = self._get_derivative(z + 0.5 * dt * k1, re)
        # k3 is the slope at the midpoint, using k2 to step.
        k3 = self._get_derivative(z + 0.5 * dt * k2, re)
        # k4 is the slope at the end of the interval, using k3 to step.
        k4 = self._get_derivative(z + dt * k3, re)
        # The final state is a weighted average of the slopes.
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(
        self, z: Tensor, re: Optional[Tensor] = None, dt: Optional[float] = None
    ) -> Tensor:
        """Evolves the state by one continuous step of size dt."""
        if dt is None:
            raise ValueError("`dt` must be provided for ContinuousKoopmanOperator.")

        # Define the function to be checkpointed. This lambda captures the non-tensor `dt`.
        def step_fn(z_arg, re_arg):
            if self.mode == "eigen":
                return self._forward_eigen(z_arg, dt, re_arg)
            else:  # 'linear' or 'mlp'
                return self._forward_rk4(z_arg, dt, re_arg)

        if self.use_checkpoint and self.training:
            return checkpoint(step_fn, z, re, use_reentrant=True)
        else:
            return step_fn(z, re)


# --- FINAL WRAPPER: The user-facing class ---
class KoopmanOperator(nn.Module):
    """
    A robust, user-facing wrapper that selects and steps through a Koopman operator.
    This is the main class to be used in your model.
    """

    def __init__(
        self,
        latent_dim: int,
        re_embedding_dim: int,
        mode: Literal["linear", "eigen", "mlp"] = "linear",
        assume_orthogonal_eigenvectors: bool = True,
        use_checkpoint: bool = False,
        is_continuous: Optional[bool] = False,
    ):
        super().__init__()

        # Collect constructor arguments to pass to the appropriate operator.
        operator_kwargs = {
            "latent_dim": latent_dim,
            "re_embedding_dim": re_embedding_dim,
            "mode": mode,
            "assume_orthogonal_eigenvectors": assume_orthogonal_eigenvectors,
            "use_checkpoint": use_checkpoint,
        }

        # Store a default training timestep.
        self.dt_train = 0.1
        self.is_continuous = is_continuous

        # 1. Declare the attribute and its type ONCE.
        self.dynamics: BaseKoopmanOperator

        # 2. Assign the value in the conditional blocks WITHOUT the type hint.
        if is_continuous:
            self.dynamics = ContinuousKoopmanOperator(**operator_kwargs)
        else:
            self.dynamics = DiscreteKoopmanOperator(**operator_kwargs)

    def forward(
        self, z: Tensor, re: Optional[Tensor] = None, dt: Optional[float] = None
    ) -> Tensor:
        """
        Performs a single, well-defined prediction step.

        Args:
            z (Tensor): The current latent state, z_t.
            re (Optional[Tensor]): The Reynolds number for conditioning.
            dt (Optional[float]): The time step. Required if `is_continuous=True`.
                                 Defaults to a fixed value during training if not provided.

        Returns:
            Tensor: The next latent state, z_{t+1}.
        """
        # Default the time step `dt` during training if it's not provided.
        # This ensures a consistent step size for the continuous model's training phase.
        # During evaluation, `dt` MUST be explicitly provided for the continuous model.
        if dt is None:
            dt = self.dt_train
        return self.dynamics(z, re=re, dt=dt)


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
