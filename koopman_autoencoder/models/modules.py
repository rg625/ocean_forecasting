import abc
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

# Setup logger for industry-standard monitoring
logger = logging.getLogger(__name__)

# --- Optional Dependencies Handling ---
try:
    from .rbf import re_expansion, ma_expansion, forcing_expansion
    from .adaptive_layers import AdaLNMLP
except ImportError:
    # Silent fallback for portability
    class IdentityExpansion(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return x

    re_expansion = ma_expansion = forcing_expansion = IdentityExpansion


# --- Configuration ---
class KoopmanMode(str, Enum):
    LINEAR = "linear"
    EIGEN = "eigen"
    MLP = "mlp"


@dataclass
class KoopmanConfig:
    """Centralized configuration for Koopman Operator."""

    latent_dim: int
    cond_embedding_dim: Optional[int] = None
    mode: KoopmanMode = KoopmanMode.LINEAR
    assume_orthogonal_eigenvectors: bool = True
    use_checkpoint: bool = False
    cond_expansion_type: Optional[str] = None
    rank: Optional[int] = 32  # For Low-Rank Adaptation (Linear Mode)
    hidden_dim: int = 32  # Width for Hypnets and MLPs


# --- Components ---


class KoopmanHypnet(nn.Module):
    """
    Hypernetwork: Predicts DYNAMICS PARAMETERS (weights/eigenvalues) based on physics.
    Used for Linear and Eigen modes.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize last layer to zero -> Start with Base Dynamics
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond: Tensor) -> Tensor:
        return self.net(cond)


class BaseKoopmanOperator(nn.Module, abc.ABC):
    """
    Abstract base class handling common logic:
    - Condition encoding
    - Hypnet creation
    - Low-Rank updates
    """

    def __init__(self, config: KoopmanConfig):
        super().__init__()
        self.config = config

        # FIX: Initialize an empty ModuleDict by default so standalone usage doesn't crash.
        # The Wrapper (KoopmanOperator) will overwrite this with the shared map.
        if (
            config.cond_expansion_type is not None
            and config.cond_embedding_dim is not None
            and config.rank is not None
        ):
            self.expansion_map = nn.ModuleDict()

            # We also provide default fallbacks if not overwritten, for safety
            dim_to_use = config.cond_embedding_dim
            if not hasattr(self, "expansion_map") or len(self.expansion_map) == 0:
                self.expansion_map = nn.ModuleDict(
                    {
                        "Re": re_expansion(dim_to_use),
                        "Ma": ma_expansion(dim_to_use),
                        "forcing": forcing_expansion(dim_to_use),
                    }
                )

        if config.mode == KoopmanMode.MLP and config.cond_embedding_dim is not None:
            self.conditioner = AdaLNMLP(config.latent_dim, config.cond_embedding_dim)

    def _build_hypnet(self, output_dim: int) -> Optional[KoopmanHypnet]:
        assert (
            self.config.cond_embedding_dim is not None
        ), f"Need integer conditioning dimension, not {type(self.config.cond_embedding_dim)}"
        return KoopmanHypnet(
            self.config.cond_embedding_dim, output_dim, self.config.hidden_dim
        )

    def _encode_cond(self, cond: Optional[Tensor]) -> Optional[Tensor]:
        if cond is None:
            assert (
                self.config.cond_embedding_dim is None
            ), f"Broken logic, cannot have condition {type(cond)} and coditioning dim {type(self.config.cond_embedding_dim)}"
            return None

        # 1. Shape Normalization
        if cond.ndim == 1:
            cond = cond.unsqueeze(-1)
        elif (
            cond.ndim == 2
            and cond.shape[1] != 1
            and self.config.cond_embedding_dim is not None
        ):
            if cond.shape[1] != self.config.cond_embedding_dim:
                # Assume sequence (B, T) -> Average to (B, 1) if not embedding
                # But if it matches embedding dim, keep it.
                pass

        # 2. Physics Expansion
        if (
            self.config.cond_expansion_type
            and self.config.cond_expansion_type in self.expansion_map
        ):
            return self.expansion_map[self.config.cond_expansion_type](cond)

        return cond

    def _apply_conditioning_mlp(self, z: Tensor, cond: Optional[Tensor]) -> Tensor:
        """
        Applies AdaLNMLP conditioning to the output tensor if `re` is provided.
        This models a parameter-dependent forcing or adjustment term.
        """
        if cond is not None:
            return self.conditioner(z, cond)
        return z

    @staticmethod
    def _compute_lora_update(
        base_weight: Tensor, uv_tensor: Tensor, rank: int
    ) -> Tensor:
        """
        Shared LoRA computation.
        Args:
            base_weight: [D, D]
            uv_tensor: [B, 2 * rank * D]
        Returns:
            Effective Weight [B, D, D]
        """
        B = uv_tensor.shape[0]
        D = base_weight.shape[0]

        # Reshape to [B, 2*rank, D]
        uv = uv_tensor.view(B, rank * 2, D)
        u, v = uv.chunk(2, dim=1)  # [B, Rank, D]

        # Update = U @ V.T -> [B, D, D]
        update = torch.einsum("brd, bre -> bde", u, v)

        # Apply scaling factor (standard LoRA practice)
        scale = 1.0 / rank
        return base_weight.unsqueeze(0) + (update * scale)

    @abc.abstractmethod
    def forward(self, z: Tensor, cond: Optional[Tensor], dt: Optional[float]) -> Tensor:
        raise NotImplementedError


class ContinuousKoopmanOperator(BaseKoopmanOperator):
    """
    Continuous Time Dynamics: dz/dt = K(c) * z
    Models the derivative and integrates it.
    """

    def __init__(self, config: KoopmanConfig):
        super().__init__(config)
        self.hypnet = None
        self.net = None

        # --- Eigen Mode ---
        if config.mode == KoopmanMode.EIGEN:
            # Base parameters (unconstrained)
            self.unconstrained_real_parts = nn.Parameter(torch.randn(config.latent_dim))
            self.imaginary_parts = nn.Parameter(torch.randn(config.latent_dim))

            # Basis
            q, _ = torch.linalg.qr(torch.randn(config.latent_dim, config.latent_dim))
            self.eigenvectors = nn.Parameter(q)

            if config.rank is not None:
                # Hypnet predicts deltas for real/imag
                self.hypnet = self._build_hypnet(config.latent_dim * 2)

        # --- Stable Linear Mode ---
        if config.mode == KoopmanMode.LINEAR:
            # 1. Rotational Part (Skew-Symmetric Base)
            self.W_param = nn.Linear(config.latent_dim, config.latent_dim, bias=False)

            # 2. Growth/Decay Part (Symmetric Base)
            # We treat this as a free symmetric matrix, NOT a PSD matrix.
            self.D_param = nn.Linear(config.latent_dim, config.latent_dim, bias=False)

            if config.rank is not None:
                # 3. Hypernetworks
                # Output dim accommodates updates for both W and D
                output_dim = config.latent_dim * config.latent_dim * 2
                if config.rank > 0:
                    output_dim = config.latent_dim * 2 * config.rank * 2

                self.hypnet = self._build_hypnet(output_dim)

            self._init_stable_weights()

        # --- MLP Mode ---
        elif config.mode == KoopmanMode.MLP:
            # # 1. Rotational Part (Skew-Symmetric Base)
            # self.W_param = nn.Linear(config.latent_dim, config.latent_dim, bias=False)

            # # 2. Growth/Decay Part (Symmetric Base)
            # # We treat this as a free symmetric matrix, NOT a PSD matrix.
            # self.D_param = nn.Linear(config.latent_dim, config.latent_dim, bias=False)

            # if config.rank is not None:

            #     # 3. Hypernetworks
            #     # Output dim accommodates updates for both W and D
            #     output_dim = config.latent_dim * config.latent_dim * 2
            #     if config.rank > 0:
            #         output_dim = config.latent_dim * 2 * config.rank * 2

            #     self.hypnet = self._build_hypnet(output_dim)

            #     self._init_stable_weights()

            self.K_mlp = nn.Sequential(
                nn.Linear(config.latent_dim, config.latent_dim, bias=False),
                nn.SiLU(),
                nn.Linear(config.latent_dim, config.latent_dim, bias=False),
            )
            for layer in self.K_mlp[:-1]:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu"
                    )

            nn.init.orthogonal_(self.K_mlp[-1].weight, gain=0.1)

    def _init_stable_weights(self):
        # 1. Initialize W as Orthogonal (Pure Rotation, Energy Conserved)
        nn.init.orthogonal_(self.W_param.weight, gain=1.0)

        # 2. Initialize D to ZERO (Neutral Stability)
        # We start with NO dissipation. The model learns where to damp.
        nn.init.zeros_(self.D_param.weight)

    @property
    def base_eigenvalues(self) -> Optional[Tensor]:
        """
        Returns the BASE (unconditioned) eigenvalues of the system.
        Real part <= 0 for stability.
        """
        if self.config.mode != KoopmanMode.EIGEN:
            return None
        real_part = -F.softplus(self.unconstrained_real_parts)
        return torch.complex(real_part, self.imaginary_parts)

    def _get_derivative(self, z: Tensor, cond_encoded: Optional[Tensor]) -> Tensor:
        """
        Computes dz/dt = f(z, cond)
        Exposed for analysis, though internal forward uses efficient steps.
        """
        if self.config.mode == KoopmanMode.MLP:
            # K = self._get_effective_linear_map(cond_encoded)
            # if K.ndim == 3:
            #     return torch.bmm(K, z.unsqueeze(-1)).squeeze(-1) + self.K_mlp(z)
            dz_dt = self.K_mlp(z)
            # dz_dt = self.K(z)
            # residual_z = self._apply_conditioning_mlp(dz_dt, cond_encoded)
            # return residual_z + F.linear(z, K)
            return self._apply_conditioning_mlp(dz_dt, cond_encoded)

        # For Linear/Eigen, we construct K_eff and mul
        else:
            K = self._get_effective_linear_map(cond_encoded)
            if K.ndim == 3:
                return torch.bmm(K, z.unsqueeze(-1)).squeeze(-1)
            return F.linear(z, K)

    def _get_effective_linear_map(self, cond_encoded: Optional[Tensor]):
        """
        Returns K = Skew(W) + Sym(D).
        Allows Limit Cycles (Re(lambda) ~ 0) and prevents wash-out.
        """
        assert (
            self.config.mode == KoopmanMode.LINEAR
        ), f"Can only apply LoRA in Linear mode, not {self.config.mode}"
        W = self.W_param.weight
        D = self.D_param.weight

        # --- LoRA / Hypernetwork Update ---
        if self.hypnet is not None and cond_encoded is not None:
            assert (
                self.config.rank is not None
            ), f"Can only apply LoRA without specified rank, got {self.config.rank} instead"
            uv_all = self.hypnet(cond_encoded)

            # Split features for W and D
            uv_w, uv_d = uv_all.chunk(2, dim=1)

            # Apply Low-Rank Updates
            W = self._compute_lora_update(W, uv_w, self.config.rank)
            D = self._compute_lora_update(D, uv_d, self.config.rank)

        # --- Construct Operator ---

        # 1. Skew-Symmetric Part (Advection/Oscillation)
        # Re(lambda) = 0. Preserves Energy.
        if W.ndim == 3:  # Batched
            W_T = W.transpose(1, 2)
        else:
            W_T = W.T
        K_skew = 0.5 * (W - W_T)

        # 2. Symmetric Part (Growth/Decay)
        # Allows Re(lambda) != 0.
        if D.ndim == 3:
            D_T = D.transpose(1, 2)
        else:
            D_T = D.T

        # Note: We use (D + D_T) instead of (D^T @ D).
        # This allows negative eigenvalues (dissipation) AND positive ones (energy injection).
        # We scale it down significantly so dynamics are dominated by rotation initially.
        K_sym = 0.5 * (D + D_T)

        # Combine: K = Rotation + Deformation
        K = K_skew + K_sym

        return K

    def _forward_eigen(
        self, z: Tensor, dt: float, cond_encoded: Optional[Tensor]
    ) -> Tensor:
        lambdas = self._get_effective_linear_map(cond_encoded)  # [B, D] or [D]

        P = self.eigenvectors
        P_inv = (
            P.T if self.config.assume_orthogonal_eigenvectors else torch.linalg.pinv(P)
        )

        z_c = z.to(torch.complex64)
        z_eig = (P_inv.to(torch.complex64) @ z_c.T).T

        evolution = torch.exp(lambdas * dt)
        z_eig_evolved = z_eig * evolution

        return (P.to(torch.complex64) @ z_eig_evolved.T).T.real

    def _rk4_step(self, z: Tensor, dt: float, cond_encoded: Optional[Tensor]) -> Tensor:
        # Define the derivative function based on mode
        if self.config.mode == KoopmanMode.MLP:

            def f(s: Tensor) -> Tensor:
                return self._get_derivative(s, cond_encoded)

        else:
            # For Linear, pre-compute K once for the step if possible,
            # but K depends on cond, which is constant over the step.
            K = self._get_effective_linear_map(cond_encoded)

            def f(s):
                if K.ndim == 3:
                    return torch.bmm(K, s.unsqueeze(-1)).squeeze(-1)
                return F.linear(s, K)

        k1 = f(z)
        k2 = f(z + 0.5 * dt * k1)
        k3 = f(z + 0.5 * dt * k2)
        k4 = f(z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(self, z: Tensor, cond: Optional[Tensor], dt: Optional[float]) -> Tensor:
        cond_encoded = self._encode_cond(cond)
        assert dt is not None, "Continuous operator needs time step"
        if self.config.mode == KoopmanMode.EIGEN:
            return self._forward_eigen(z, dt, cond_encoded)
        else:
            # Both Linear and MLP use RK4 for stability/accuracy
            return self._rk4_step(z, dt, cond_encoded)


class DiscreteKoopmanOperator(BaseKoopmanOperator):
    """
    Discrete Time Dynamics: z_{t+1} = K(c) * z
    Adjusted to mimic the Continuous operator's structure (Skew/Sym decomposition).
    """

    def __init__(self, config: KoopmanConfig):
        super().__init__(config)
        self.hypnet = None

        # --- Eigen Mode ---
        if config.mode == KoopmanMode.EIGEN:
            self.unconstrained_log_magnitude = nn.Parameter(
                torch.randn(config.latent_dim)
            )
            self.angle = nn.Parameter(torch.randn(config.latent_dim))
            q, _ = torch.linalg.qr(torch.randn(config.latent_dim, config.latent_dim))
            self.eigenvectors = nn.Parameter(q)

            if config.rank is not None:
                self.hypnet = self._build_hypnet(config.latent_dim * 2)

        # --- Linear Mode (Mimicking Continuous Skew/Sym) ---
        elif config.mode == KoopmanMode.LINEAR:
            # Rotational Part
            self.W_param = nn.Linear(config.latent_dim, config.latent_dim, bias=False)
            # Growth/Decay Part
            self.D_param = nn.Linear(config.latent_dim, config.latent_dim, bias=False)

            if config.rank is not None:
                output_dim = config.latent_dim * 2 * config.rank * 2
                self.hypnet = self._build_hypnet(output_dim)

            self._init_stable_weights()

        elif config.mode == KoopmanMode.MLP:
            self.K_mlp = nn.Sequential(
                nn.Linear(config.latent_dim, config.latent_dim, bias=False),
                nn.SiLU(),
                nn.Linear(config.latent_dim, config.latent_dim, bias=False),
            )
            nn.init.orthogonal_(self.K_mlp[-1].weight, gain=0.1)

    def _get_effective_linear_map(self, cond_encoded: Optional[Tensor]):
        """
        Mirroring the Continuous logic: K = Skew(W) + Sym(D)
        In discrete terms, this acts as the transition matrix.
        """
        W = self.W_param.weight
        D = self.D_param.weight

        if self.hypnet is not None and cond_encoded is not None:
            assert (
                self.config.rank is not None
            ), f"Can only apply LoRA without specified rank, got {self.config.rank} instead"
            uv_all = self.hypnet(cond_encoded)
            uv_w, uv_d = uv_all.chunk(2, dim=1)
            W = self._compute_lora_update(W, uv_w, self.config.rank)
            D = self._compute_lora_update(D, uv_d, self.config.rank)

        # 1. Skew-Symmetric (Unitary/Rotation)
        W_T = W.transpose(-1, -2) if W.ndim == 3 else W.T
        K_skew = 0.5 * (W - W_T)

        # 2. Symmetric (Scaling)
        D_T = D.transpose(-1, -2) if D.ndim == 3 else D.T
        K_sym = 0.5 * (D + D_T)

        # Discrete Transition: Identity + (Rotation + Deformation)
        # This approximates z_next = z + (K_skew + K_sym)z
        identity = torch.eye(self.config.latent_dim, device=W.device)
        return identity + K_skew + K_sym

    def _init_linear_weights(self):
        nn.init.eye_(self.K_base.weight)
        with torch.no_grad():
            self.K_base.weight.add_(torch.randn_like(self.K_base.weight) * 0.01)

    @property
    def base_eigenvalues(self) -> Optional[Tensor]:
        """
        Returns BASE eigenvalues. Magnitude <= 1.
        """
        if self.config.mode != KoopmanMode.EIGEN:
            return None
        log_mag = -F.softplus(self.unconstrained_log_magnitude)
        return torch.polar(torch.exp(log_mag), self.angle)

    def forward(
        self, z: Tensor, cond: Optional[Tensor] = None, dt: Optional[float] = None
    ) -> Tensor:
        cond_encoded = self._encode_cond(cond)

        if self.config.mode == KoopmanMode.EIGEN:
            mag_logits, angle = self.unconstrained_log_magnitude, self.angle
            if self.hypnet is not None and cond_encoded is not None:
                delta = self.hypnet(cond_encoded)
                d_mag, d_ang = delta.chunk(2, dim=1)
                mag_logits = mag_logits.unsqueeze(0) + d_mag
                angle = angle.unsqueeze(0) + d_ang

            lambdas = torch.polar(torch.exp(-F.softplus(mag_logits)), angle)
            P = self.eigenvectors
            P_inv = (
                P.T
                if self.config.assume_orthogonal_eigenvectors
                else torch.linalg.pinv(P)
            )

            z_eig = (P_inv.to(torch.complex64) @ z.to(torch.complex64).T).T
            z_eig_next = z_eig * lambdas
            return (P.to(torch.complex64) @ z_eig_next.T).T.real

        elif self.config.mode == KoopmanMode.LINEAR:
            K = self._get_effective_linear_map(cond_encoded)
            if K.ndim == 3:
                return torch.bmm(K, z.unsqueeze(-1)).squeeze(-1)
            return F.linear(z, K)

        elif self.config.mode == KoopmanMode.MLP:
            z_conditioned = self._apply_conditioning_mlp(z, cond_encoded)
            return z + self.K_mlp(z_conditioned)

        else:
            raise RuntimeError(f"Unsupported Koopman mode: {self.config.mode}")


class KoopmanOperator(nn.Module):
    """
    Unified entry point for Koopman Operators.
    Wraps the internal logic but exposes a flat argument structure for initialization.
    """

    def __init__(
        self,
        latent_dim: int,
        cond_embedding_dim: Optional[int] = None,
        mode: Union[KoopmanMode, str] = KoopmanMode.LINEAR,
        assume_orthogonal_eigenvectors: bool = True,
        use_checkpoint: bool = False,
        is_continuous: bool = False,
        rank: int = 4,
        cond_expansion_type: Optional[str] = None,
        shared_expansion_map: Optional[nn.ModuleDict] = None,
    ):
        super().__init__()

        # Convert string to Enum
        if isinstance(mode, str):
            mode = KoopmanMode(mode.lower())

        self.config = KoopmanConfig(
            latent_dim=latent_dim,
            cond_embedding_dim=cond_embedding_dim,
            mode=mode,
            assume_orthogonal_eigenvectors=assume_orthogonal_eigenvectors,
            use_checkpoint=use_checkpoint,
            cond_expansion_type=cond_expansion_type,
            rank=rank,
        )

        self.is_continuous = is_continuous
        self.dt_train = 0.1

        if is_continuous:
            self.dynamics = ContinuousKoopmanOperator(self.config)
        else:
            self.dynamics = DiscreteKoopmanOperator(self.config)

        # Inject the shared physics brain if provided
        if shared_expansion_map is not None:
            self.dynamics.expansion_map = shared_expansion_map

    def forward(
        self, z: Tensor, cond: Optional[Tensor] = None, dt: Optional[float] = None
    ) -> Tensor:
        """
        Args:
            z: Latent state [B, D]
            cond: Condition (Reynolds/Mach etc) [B, C] or [B]
            dt: Time step (required for continuous mode)
        """
        # Alias 're' to 'cond' if user passes it via kwargs (implicit support)
        # But explicitly, we use 'cond' now.

        if dt is None:
            dt = self.dt_train
        if self.is_continuous and dt is None:
            raise ValueError("Continuous dynamics require a `dt` value.")

        return self.dynamics(z, cond=cond, dt=dt)


class Re(nn.Module):
    """
    Auxiliary Reynolds Number Predictor.
    Useful for enforcing physical consistency in the latent space.
    """

    def __init__(self, latent_dim: int, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.latent_dim = latent_dim
        self.re_predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 8),
            nn.SiLU(),
            nn.Linear(latent_dim // 8, 1),
            nn.Softplus(),  # Force positive Reynolds number
        )

    def _forward_impl(self, z: Tensor) -> Tensor:
        original_shape = z.shape
        if z.ndim > 2:
            z = z.reshape(-1, self.latent_dim)
        reynolds = self.re_predictor(z)
        if len(original_shape) > 2:
            reynolds = reynolds.view(*original_shape[:-1], 1)
        return reynolds

    def forward(self, z: Tensor) -> Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, z, use_reentrant=False)
        return self._forward_impl(z)
