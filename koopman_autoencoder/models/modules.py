import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import Tensor
from typing import Optional, Literal
import abc

try:
    from .rbf import re_expansion, ma_expansion, forcing_expansion
except ImportError:
    pass


class KoopmanHypnet(nn.Module):
    """
    Predicts DYNAMICS PARAMETERS based on physics conditions (Reynolds #).
    Used to modulate eigenvalues or matrix weights directly.
    """

    def __init__(self, cond_dim: int, output_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Initialize last layer to zero so training starts with Base Dynamics
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond: Tensor) -> Tensor:
        return self.net(cond)


class BaseKoopmanOperator(nn.Module, abc.ABC):
    """
    Abstract base class handling parameter conditioning encoding.
    """

    def __init__(
        self,
        latent_dim: int,
        cond_embedding_dim: int,
        mode: Literal["linear", "eigen", "mlp"],
        assume_orthogonal_eigenvectors: bool,
        use_checkpoint: bool,
        cond_expansion_type: Optional[str] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.mode = mode
        self.use_checkpoint = use_checkpoint
        self.assume_orthogonal = assume_orthogonal_eigenvectors
        self.cond_embedding_dim = cond_embedding_dim
        self.cond_expansion_type = cond_expansion_type

        # Setup Expansion Map for raw inputs
        dim_to_use = cond_embedding_dim if cond_embedding_dim is not None else 64
        self.expansion_map = nn.ModuleDict(
            {
                "Re": re_expansion(dim_to_use),
                "Ma": ma_expansion(dim_to_use),
                "forcing": forcing_expansion(dim_to_use),
            }
        )

    def _encode_cond(self, cond: Optional[Tensor]) -> Optional[Tensor]:
        """
        Prepares the condition tensor.
        1. Handles dimensionality (unsqueezing/flattening).
        2. Applies physics expansion (Re->HighDim) if configured.
        """
        if cond is None:
            return None

        # 1. Shape Normalization
        # Expecting (Batch, Dim) or (Batch, 1)
        if cond.ndim == 1:
            cond = cond.unsqueeze(-1)
        elif (
            cond.ndim == 2
            and cond.shape[1] != 1
            and self.cond_embedding_dim is not None
        ):
            if cond.shape[1] != self.cond_embedding_dim:
                # Ambiguous case: Input is (B, T) but expected (B, Emb)?
                # Assuming simple averaging for sequence inputs or keeping as is
                pass

        # 2. Expansion
        if self.cond_expansion_type in self.expansion_map:
            return self.expansion_map[self.cond_expansion_type](cond)

        return cond

    def get_effective_linear_map(
        self, K_base: Tensor, cond_encoded: Optional[Tensor], rank: int = 4
    ):
        """Helper to compute K_eff = K_base + U@V.T from hypnet."""
        K = K_base
        if hasattr(self, "hypnet") and cond_encoded is not None:
            # Low-Rank Adaptation
            uv = self.hypnet(cond_encoded)  # [B, 2*D*Rank]
            B_batch = uv.shape[0]

            uv = uv.view(B_batch, rank * 2, self.latent_dim)
            u, v = uv.chunk(2, dim=1)  # [B, Rank, D]

            # Rank-k update: sum(u_r outer v_r)
            # einsum 'brd, bre -> bde' computes batch outer products summed over rank
            update = torch.einsum("brd, bre -> bde", u, v)

            # Scale update to be small initially
            K = K.unsqueeze(0) + update * 0.1
        return K

    @abc.abstractmethod
    def forward(self, z: Tensor, cond: Optional[Tensor], dt: Optional[float]) -> Tensor:
        raise NotImplementedError


# --- Continuous Dynamics ---
class ContinuousKoopmanOperator(BaseKoopmanOperator):
    """
    Continuous Time Dynamics: dz/dt = K(c) * z
    Guaranteed stability via parameter constraints.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # --- Eigen Mode Setup ---
        if self.mode == "eigen":
            # Learnable Base Eigenvalues
            self.base_real = nn.Parameter(torch.randn(self.latent_dim))
            self.base_imag = nn.Parameter(torch.randn(self.latent_dim))

            # Learnable Basis
            eigenvectors_init = torch.randn(self.latent_dim, self.latent_dim)
            self.eigenvectors = nn.Parameter(torch.linalg.qr(eigenvectors_init).Q)

            # Hypernetwork for Eigenvalues (Modulates Real and Imag parts)
            if self.cond_embedding_dim:
                # Output 2x latent_dim (one delta for real, one for imag)
                self.hypnet = KoopmanHypnet(
                    self.cond_embedding_dim, self.latent_dim * 2
                )

        # --- Linear/LoRA Mode Setup ---
        elif self.mode == "linear":
            # Base Matrix
            self.K_base = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

            # Initialize close to rotation (skew-symmetric)
            W = torch.randn(self.latent_dim, self.latent_dim)
            self.K_base.weight.data = (W - W.T) * 0.1 - 0.01 * torch.eye(
                self.latent_dim
            )

            # Hypernetwork for Low-Rank Update (Rank-1 or Rank-2)
            if self.cond_embedding_dim:
                # Predict vectors U and V for K = K_base + U @ V.T
                # Rank 4 update
                rank = 4
                self.hypnet = KoopmanHypnet(
                    self.cond_embedding_dim, self.latent_dim * 2 * rank
                )
                self.rank = rank

        # --- MLP Mode (Residual) ---
        elif self.mode == "mlp":
            self.net = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.SiLU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            nn.init.zeros_(self.net[-1].weight)
            # MLP usually harder to condition purely parametrically without huge hypernets
            # For MLP, we might stick to Input Modulation but ONLY inside the residual
            if self.cond_embedding_dim:
                self.cond_proj = nn.Linear(self.cond_embedding_dim, self.latent_dim)

    def get_effective_parameters(self, cond_encoded: Optional[Tensor]):
        """Calculates K or Lambda ensuring stability constraints."""

        if self.mode == "eigen":
            real = self.base_real
            imag = self.base_imag

            if self.hypnet is not None and cond_encoded is not None:
                delta = self.hypnet(cond_encoded)  # [B, 2*D]
                d_real, d_imag = delta.chunk(2, dim=1)

                # Broadcasting parameters: (D,) + (B, D)
                real = real.unsqueeze(0) + d_real
                imag = imag.unsqueeze(0) + d_imag

            # --- STABILITY CONSTRAINT ---
            # Real part must be negative for stability in continuous time
            # We apply Softplus AFTER modulation to guarantee this property
            constrained_real = -F.softplus(real)
            return torch.complex(constrained_real, imag)  # [B, D] or [D]

        elif self.mode == "linear":
            K = self.K_base.weight  # [D, D]

            if hasattr(self, "hypnet") and cond_encoded is not None:
                # Low-Rank Adaptation: K_eff = K_base + sum(u_i * v_i^T)
                uv = self.hypnet(cond_encoded)  # [B, 2*D*Rank]
                B_batch = uv.shape[0]

                uv = uv.view(B_batch, self.rank * 2, self.latent_dim)
                u, v = uv.chunk(2, dim=1)  # [B, Rank, D]

                # Compute update: U @ V^T -> [B, D, D]
                # u: [B, R, D] -> transpose last two for matmul? No, outer product logic
                # update = torch.bmm(u.transpose(1, 2), v) # [B, D, D]
                # Let's simplify: Rank 1
                update = torch.einsum("brd, bre -> bde", u, v)

                # Scale update to be small initially
                K = K.unsqueeze(0) + update * 0.1

            return K

        return None

    def _forward_eigen(
        self, z: Tensor, dt: float, cond_encoded: Optional[Tensor]
    ) -> Tensor:
        lambdas = self.get_effective_parameters(cond_encoded)  # [B, D]

        # 1. Project to Eigenbasis
        P = self.eigenvectors
        P_inv = P.T if self.assume_orthogonal else torch.linalg.pinv(P)

        z_c = z.to(torch.complex64)
        P_c, P_inv_c = P.to(torch.complex64), P_inv.to(torch.complex64)

        z_eig = (P_inv_c @ z_c.T).T  # [B, D]

        # 2. Evolve
        # exp(lambda * dt)
        evolution = torch.exp(lambdas * dt)
        z_eig_evolved = z_eig * evolution

        # 3. Reconstruct
        z_evolved = (P_c @ z_eig_evolved.T).T
        return z_evolved.real

    def _forward_rk4(
        self, z: Tensor, dt: float, cond_encoded: Optional[Tensor]
    ) -> Tensor:
        # Get effective Matrix K
        K = self.get_effective_parameters(cond_encoded)  # [B, D, D] or [D, D]

        def f(state):
            if K.ndim == 3:  # Batch-specific K
                # state: [B, D], K: [B, D, D] -> [B, D, 1]
                return torch.bmm(K, state.unsqueeze(-1)).squeeze(-1)
            return F.linear(state, K)  # Shared K

        k1 = f(z)
        k2 = f(z + 0.5 * dt * k1)
        k3 = f(z + 0.5 * dt * k2)
        k4 = f(z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(
        self, z: Tensor, cond: Optional[Tensor] = None, dt: Optional[float] = None
    ) -> Tensor:
        if dt is None:
            raise ValueError("Continuous operator requires `dt`.")
        cond_encoded = self._encode_cond(cond)

        if self.mode == "eigen":
            return self._forward_eigen(z, dt, cond_encoded)
        elif self.mode == "linear":
            return self._forward_rk4(z, dt, cond_encoded)
        else:  # MLP (Residual)
            # Basic residual dynamics: z_new = z + f(z, cond) * dt
            # This is technically forward Euler, ok for small dt
            res = self.net(z)
            if cond_encoded is not None:
                # Modulate residual, not input
                gamma = self.cond_proj(cond_encoded)
                res = res * (1 + torch.tanh(gamma))
            return z + res * dt


# --- Discrete Dynamics ---
class DiscreteKoopmanOperator(BaseKoopmanOperator):
    """
    Discrete Time Dynamics: z_{t+1} = K(c) * z
    Stability via unit circle constraints.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.mode == "eigen":
            self.base_mag_logits = nn.Parameter(torch.randn(self.latent_dim))
            self.base_angle = nn.Parameter(torch.randn(self.latent_dim))
            eigenvectors_init = torch.randn(self.latent_dim, self.latent_dim)
            self.eigenvectors = nn.Parameter(torch.linalg.qr(eigenvectors_init).Q)

            if self.cond_embedding_dim:
                self.hypnet = KoopmanHypnet(
                    self.cond_embedding_dim, self.latent_dim * 2
                )

        elif self.mode == "linear":
            self.K_base = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
            # Initialize close to identity (no change) + small noise
            nn.init.eye_(self.K_base.weight)
            self.K_base.weight.data += torch.randn_like(self.K_base.weight) * 0.01

            if self.cond_embedding_dim:
                self.rank = 4
                self.hypnet = KoopmanHypnet(
                    self.cond_embedding_dim, self.latent_dim * 2 * self.rank
                )

        elif self.mode == "mlp":
            # Residual Network: z_{t+1} = z_t + Net(z_t)
            self.net = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.SiLU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

            if self.cond_embedding_dim:
                self.cond_proj = nn.Linear(self.cond_embedding_dim, self.latent_dim)

    def forward(
        self, z: Tensor, cond: Optional[Tensor] = None, dt: Optional[float] = None
    ) -> Tensor:
        cond_encoded = self._encode_cond(cond)

        if self.mode == "eigen":
            mag_logits = self.base_mag_logits
            angle = self.base_angle

            if hasattr(self, "hypnet") and cond_encoded is not None:
                delta = self.hypnet(cond_encoded)
                d_mag, d_ang = delta.chunk(2, dim=1)
                mag_logits = mag_logits.unsqueeze(0) + d_mag
                angle = angle.unsqueeze(0) + d_ang

            # Stability: Magnitude <= 1
            magnitude = torch.sigmoid(mag_logits)
            lambdas = torch.polar(magnitude, angle)

            P = self.eigenvectors
            P_inv = P.T if self.assume_orthogonal else torch.linalg.pinv(P)

            z_c = z.to(torch.complex64)
            P_c, P_inv_c = P.to(torch.complex64), P_inv.to(torch.complex64)
            z_eig = (P_inv_c @ z_c.T).T

            z_eig_next = z_eig * lambdas
            return (P_c @ z_eig_next.T).T.real

        elif self.mode == "linear":
            # Use LoRA updated K if condition exists
            K = self.get_effective_linear_map(
                self.K_base.weight, cond_encoded, self.rank
            )

            if K.ndim == 3:  # Batch-specific K
                return torch.bmm(K, z.unsqueeze(-1)).squeeze(-1)
            return F.linear(z, K)

        elif self.mode == "mlp":
            # Residual step
            res = self.net(z)
            if cond_encoded is not None:
                gamma = self.cond_proj(cond_encoded)
                res = res * (1 + torch.tanh(gamma))
            return z + res

        else:
            raise NotImplementedError(
                "Modes need to be in ['linear', 'mlp' or 'eigen']"
            )


# --- Main Wrapper ---
class KoopmanOperator(nn.Module):
    """
    Unified entry point for Koopman Operators.
    """

    def __init__(
        self,
        latent_dim: int,
        cond_embedding_dim: int,
        mode: Literal["linear", "eigen", "mlp"] = "linear",
        assume_orthogonal_eigenvectors: bool = True,
        use_checkpoint: bool = False,
        is_continuous: Optional[bool] = False,
        cond_expansion_type: Optional[str] = None,
    ):
        super().__init__()

        kwargs = {
            "latent_dim": latent_dim,
            "cond_embedding_dim": cond_embedding_dim,
            "mode": mode,
            "assume_orthogonal_eigenvectors": assume_orthogonal_eigenvectors,
            "use_checkpoint": use_checkpoint,
            "cond_expansion_type": cond_expansion_type,
        }

        self.is_continuous = is_continuous
        self.dt_train = 0.1

        if is_continuous:
            self.dynamics = ContinuousKoopmanOperator(**kwargs)
        else:
            self.dynamics = DiscreteKoopmanOperator(**kwargs)

    def forward(
        self, z: Tensor, cond: Optional[Tensor] = None, dt: Optional[float] = None
    ) -> Tensor:
        if dt is None:
            dt = self.dt_train
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
            nn.Softplus(),  # Ensure positive Re
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
