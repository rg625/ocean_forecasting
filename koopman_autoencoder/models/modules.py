import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import Tensor
from typing import Optional, Literal
import abc

# Import conditioning layers
try:
    from .rbf import re_expansion, ma_expansion, forcing_expansion
except ImportError:
    pass


class BaseKoopmanOperator(nn.Module, abc.ABC):
    """
    Abstract base class handling parameter conditioning.
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
        dim_to_use = cond_embedding_dim if cond_embedding_dim is not None else 64

        # We keep the expansion map to handle raw inputs (Re, Ma)
        self.expansion_map = nn.ModuleDict(
            {
                "Re": re_expansion(dim_to_use),
                "Ma": ma_expansion(dim_to_use),
                "forcing": forcing_expansion(dim_to_use),
            }
        )

        # We remove AdaLN conditioning on 'z' itself to strictly preserve
        # the linearity assumption of the coordinate system.
        # Conditioning is now used ONLY to generate the operator K.

    def _encode_cond(self, cond: Optional[Tensor]) -> Optional[Tensor]:
        if cond is None:
            return None

        if cond.ndim == 1:
            cond = cond.unsqueeze(-1)
        elif (
            cond.ndim == 2
            and cond.shape[1] != 1
            and cond.shape[1] != self.cond_embedding_dim
        ):
            cond = cond.mean(dim=1, keepdim=True)

        if self.cond_expansion_type in self.expansion_map:
            return self.expansion_map[self.cond_expansion_type](cond)

        return cond

    @abc.abstractmethod
    def forward(self, z: Tensor, cond: Optional[Tensor], dt: Optional[float]) -> Tensor:
        raise NotImplementedError


class ContinuousKoopmanOperator(BaseKoopmanOperator):
    """
    General Parametric Continuous Koopman Operator.

    Dynamics: dz/dt = K(p) * z
    Integration: Runge-Kutta 4 (RK4)

    Structure:
        K(p) = K_base + HyperNet(p) - Stability_Bias

    This allows the model to learn completely different dynamics structures
    for different regimes (e.g. Diffusion vs. Advection) without hard constraints,
    while Spectral Norm and Stability Bias prevent explosion.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.mode == "linear":
            # 1. Base Dynamics (Average behavior across all regimes)
            self.K_base = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

            # Init: Stable but active. Negative diagonal (decay) + Random (mixing)
            with torch.no_grad():
                self.K_base.weight.data = -0.1 * torch.eye(self.latent_dim)
                self.K_base.weight.data += 0.01 * torch.randn(
                    self.latent_dim, self.latent_dim
                )

            # 2. HyperNetwork (The "Parametric" part)
            # Predicts the perturbation matrix Delta_K based on condition
            cond_dim = self.cond_embedding_dim if self.cond_embedding_dim else 64

            # STABILITY FIX 1: Spectral Normalization
            # Prevents the HyperNetwork from outputting massive perturbations
            self.hyper_k = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(cond_dim, 64)),
                nn.SiLU(),
                nn.utils.spectral_norm(
                    nn.Linear(64, self.latent_dim * self.latent_dim)
                ),
            )

            # STABILITY FIX 2: Hyper-Scale
            # Learnable scalar initialized small to throttle initial perturbations
            self.hyper_scale = nn.Parameter(torch.tensor(0.01))

            # STABILITY FIX 3: Stability Bias
            # Shifts the diagonal of K to ensure baseline stability (Real < 0)
            self.stability_bias = nn.Parameter(torch.tensor(0.1))

            # Init Hypernet small but non-zero
            nn.init.normal_(self.hyper_k[-1].weight, std=0.001)
            nn.init.zeros_(self.hyper_k[-1].bias)

        elif self.mode == "mlp":
            self.K = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim // 8),
                nn.SiLU(),
                nn.Linear(self.latent_dim // 8, self.latent_dim),
            )

        elif self.mode == "eigen":
            self.unconstrained_real_parts = nn.Parameter(torch.randn(self.latent_dim))
            self.imaginary_parts = nn.Parameter(torch.randn(self.latent_dim))
            eigenvectors_init = torch.randn(self.latent_dim, self.latent_dim)
            self.eigenvectors = nn.Parameter(torch.linalg.qr(eigenvectors_init).Q)

    @property
    def eigenvalues(self) -> Optional[Tensor]:
        if self.mode != "eigen":
            return None
        real_part = -F.softplus(self.unconstrained_real_parts)
        return torch.complex(real_part, self.imaginary_parts)

    def _get_parametric_K(self, cond_encoded: Optional[Tensor]) -> Tensor:
        """
        Returns the full dynamics matrix K for the given condition.
        Shape: (B, D, D)
        """
        # Base matrix (D, D) -> (1, D, D)
        K = self.K_base.weight.unsqueeze(0)

        if cond_encoded is not None:
            # Predict Delta K: (B, D*D)
            delta_k_flat = self.hyper_k(cond_encoded)
            # Reshape: (B, D, D)
            delta_k = delta_k_flat.view(-1, self.latent_dim, self.latent_dim)

            # Combine with throttling scale
            K = K + delta_k * self.hyper_scale

        # Apply Stability Bias to Diagonal (Shift spectrum left)
        # K_eff = K - gamma * I
        eye = torch.eye(self.latent_dim, device=K.device).unsqueeze(0)
        K = K - torch.abs(self.stability_bias) * eye

        return K

    def _get_derivative(self, z: Tensor, K: Tensor) -> Tensor:
        # dz/dt = K * z
        if self.mode == "linear":
            # Batched Matrix Vector Multiply: (B, D, D) @ (B, D, 1)
            # z: (B, D) -> (B, D, 1)
            z_unsqueezed = z.unsqueeze(2)
            dz = torch.bmm(K, z_unsqueezed).squeeze(2)
            return dz

        elif self.mode == "mlp":
            return self.K(z)

        return z

    def _forward_rk4(self, z: Tensor, dt: float, K: Optional[Tensor] = None) -> Tensor:
        """Standard RK4 integration."""

        # Define closure to capture K
        def f(state):
            if self.mode == "linear":
                # K is (B, D, D) or (1, D, D)
                return self._get_derivative(state, K)
            elif self.mode == "mlp":
                return self._get_derivative(state, None)

        k1 = f(z)
        k2 = f(z + 0.5 * dt * k1)
        k3 = f(z + 0.5 * dt * k2)
        k4 = f(z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _forward_eigen(self, z: Tensor, dt: float) -> Tensor:
        """Analytical solution for diagonal Eigen mode."""
        P = self.eigenvectors
        P_inv = P.T if self.assume_orthogonal else torch.linalg.pinv(P)
        z_c = z.to(torch.complex64)
        P_c, P_inv_c = P.to(torch.complex64), P_inv.to(torch.complex64)

        assert self.eigenvalues is not None, "Cannot multiply None type with float"
        exp_lambda_dt = torch.exp(self.eigenvalues * dt)

        # Diagonal multiply
        z_eig = P_inv_c @ z_c.T  # (D, B)
        z_evolved_eig = (z_eig.T * exp_lambda_dt).T  # (D, B)

        z_evolved = (P_c @ z_evolved_eig).T  # (B, D)
        return z_evolved.real

    def forward(
        self, z: Tensor, cond: Optional[Tensor] = None, dt: Optional[float] = None
    ) -> Tensor:
        if dt is None:
            raise ValueError("`dt` must be provided.")

        cond_encoded = self._encode_cond(cond)

        # Get K once per step (Linear Parameter Varying approximation)
        K_matrix = None
        if self.mode == "linear":
            K_matrix = self._get_parametric_K(cond_encoded)

        def step_fn(z_curr, K_curr):
            if self.mode == "eigen":
                return self._forward_eigen(z_curr, dt)
            else:
                # Passes K_curr to RK4
                return self._forward_rk4(z_curr, dt, K_curr)

        if self.use_checkpoint and self.training:
            return checkpoint(step_fn, z, K_matrix, use_reentrant=True)
        else:
            return step_fn(z, K_matrix)


class DiscreteKoopmanOperator(BaseKoopmanOperator):
    """
    General Parametric Discrete Koopman Operator.

    Dynamics: z_{t+1} = z_t + K(p) * z_t
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.mode == "linear":
            # 1. Base Matrix
            self.K_base = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
            nn.init.uniform_(self.K_base.weight, -0.01, 0.01)

            # 2. HyperNetwork
            cond_dim = self.cond_embedding_dim if self.cond_embedding_dim else 64

            # Stabilized HyperNetwork
            self.hyper_k = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(cond_dim, 64)),
                nn.SiLU(),
                nn.utils.spectral_norm(
                    nn.Linear(64, self.latent_dim * self.latent_dim)
                ),
            )

            self.hyper_scale = nn.Parameter(torch.tensor(0.01))
            self.stability_bias = nn.Parameter(torch.tensor(0.01))

            nn.init.normal_(self.hyper_k[-1].weight, std=0.001)
            nn.init.zeros_(self.hyper_k[-1].bias)

        elif self.mode == "eigen":
            self.unconstrained_log_magnitude = nn.Parameter(
                torch.randn(self.latent_dim)
            )
            self.angle = nn.Parameter(torch.randn(self.latent_dim))
            eigenvectors_init = torch.randn(self.latent_dim, self.latent_dim)
            self.eigenvectors = nn.Parameter(torch.linalg.qr(eigenvectors_init).Q)
        elif self.mode == "mlp":
            self.K = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim // 8),
                nn.SiLU(),
                nn.Linear(self.latent_dim // 8, self.latent_dim),
            )
            nn.init.zeros_(self.K[-1].weight)
            nn.init.zeros_(self.K[-1].bias)

    @property
    def eigenvalues(self) -> Optional[Tensor]:
        if self.mode != "eigen":
            return None
        log_magnitude = -F.softplus(self.unconstrained_log_magnitude)
        magnitude = torch.exp(log_magnitude)
        return torch.polar(magnitude, self.angle)

    def _get_parametric_K(self, cond_encoded: Optional[Tensor]) -> Tensor:
        K = self.K_base.weight.unsqueeze(0)
        if cond_encoded is not None:
            delta_k = self.hyper_k(cond_encoded).view(
                -1, self.latent_dim, self.latent_dim
            )
            K = K + delta_k * self.hyper_scale

        # Apply Stability Bias (shifts eigenvalues towards contraction)
        eye = torch.eye(self.latent_dim, device=K.device).unsqueeze(0)
        K = K - torch.abs(self.stability_bias) * eye

        return K

    def _forward_impl(self, z: Tensor, cond_encoded: Optional[Tensor] = None) -> Tensor:
        if self.mode == "linear":
            K = self._get_parametric_K(cond_encoded)
            # z_next = z + Kz
            z_unsqueezed = z.unsqueeze(2)
            update = torch.bmm(K, z_unsqueezed).squeeze(2)
            return z + update

        elif self.mode == "mlp":
            return z + self.K(z)
        elif self.mode == "eigen":
            # Eigen implementation
            P = self.eigenvectors
            P_inv = P.T if self.assume_orthogonal else torch.linalg.pinv(P)
            z_c, P_c, P_inv_c = (
                z.to(torch.complex64),
                P.to(torch.complex64),
                P_inv.to(torch.complex64),
            )
            Lambda = torch.diag(self.eigenvalues)
            z_eig = P_inv_c @ z_c.T
            z_recomposed = (P_c @ Lambda @ z_eig).T
            return z_recomposed.real

    def forward(
        self, z: Tensor, cond: Optional[Tensor] = None, dt: Optional[float] = None
    ) -> Tensor:
        cond_encoded = self._encode_cond(cond)
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, z, cond_encoded, use_reentrant=True)
        else:
            return self._forward_impl(z, cond_encoded)


class KoopmanOperator(nn.Module):
    """
    Main Wrapper Class.
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
