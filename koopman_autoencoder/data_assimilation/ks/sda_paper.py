"""Score-based Data Assimilation, following Rozet & Louppe (NeurIPS 2023).

Implements the method of `2306.10574v2` rather than a generic diffusion-guided DA scheme.
The four components that define it are:

1. a **local** score network with a bounded temporal Markov blanket, trained on short
   segments (Sec. 3.1, Eq. 14, Alg. 1);
2. **Algorithm 2** composition of those local scores into the score of an arbitrarily long
   trajectory, so one trained model serves any length L;
3. the **Tweedie + Gamma** likelihood approximation of Sec. 3.2 / Eq. 15, whose covariance is
   ``Sigma_y + (sigma^2/mu^2) A Gamma A^T`` with Gamma built from the data covariance
   (App. B) -- not the plain DPS covariance ``Sigma_y``;
4. **predictor-corrector** sampling (Alg. 4): an exponential-integrator predictor (Eq. 16)
   with C Langevin Monte Carlo corrections (Eq. 17) between predictor steps.

The observation model is decoupled from training throughout: the score network is trained
once, unconditionally, and observations enter only at sampling time. Changing the mask, the
observation times, the noise level or the operator requires no retraining.

Conventions follow the paper exactly:

    p(x(t) | x) = N(mu(t) x, sigma(t)^2 I),
    mu(t)    = cos(omega t)^2,  omega = arccos(sqrt(1e-3)),
    sigma(t) = sqrt(1 - mu(t)^2),   t in [0, 1].

The network is parameterised as ``eps_phi`` and the score is ``s = -eps_phi / sigma(t)``
(Sec. 2, below Eq. 6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Diffusion process: variance-preserving SDE with the cosine schedule (Sec. 4)
# ---------------------------------------------------------------------------
@dataclass
class CosineVPSchedule:
    """mu(t) = cos(omega t)^2, sigma(t) = sqrt(1 - mu(t)^2), omega = arccos(sqrt(eps))."""

    eps: float = 1e-3

    @property
    def omega(self) -> float:
        return math.acos(math.sqrt(self.eps))

    def mu(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.omega * t) ** 2

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((1.0 - self.mu(t) ** 2).clamp_min(1e-12))

    def perturb(
        self, x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """x(t) = mu(t) x + sigma(t) eps, broadcasting t over the trailing axes."""
        shape = (-1,) + (1,) * (x.ndim - 1)
        return self.mu(t).view(shape) * x + self.sigma(t).view(shape) * eps


# ---------------------------------------------------------------------------
# 2. Local score network with a bounded temporal blanket
# ---------------------------------------------------------------------------
def _gn(c: int) -> nn.GroupNorm:
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(1e4) * torch.arange(half, device=t.device).float() / half
        )
        a = t.float()[:, None] * freqs[None] * 1000.0
        return torch.cat([a.sin(), a.cos()], dim=-1)


class ResBlock1d(nn.Module):
    """Circular in space; the KS domain is periodic."""

    def __init__(self, c_in: int, c_out: int, t_dim: int, kernel: int = 3):
        super().__init__()
        p = kernel // 2
        self.norm1, self.norm2 = _gn(c_in), _gn(c_out)
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=p, padding_mode="circular")
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=p, padding_mode="circular")
        self.emb = nn.Linear(t_dim, c_out)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class LocalScoreUNet(nn.Module):
    """eps-prediction over one segment ``x_{i-k:i+k}``, shaped ``[B, 2k+1, X]``.

    The 2k+1 states of the segment are carried as **channels** of a purely spatial 1-D
    U-Net. The temporal receptive field is therefore exactly the segment -- 2k+1 frames --
    by construction, which is precisely the bounded pseudo-Markov blanket the paper's
    Algorithm 2 composition requires. No temporal convolution is involved, so there is no
    way for the blanket to silently widen.
    """

    def __init__(
        self, k: int = 4, hidden=(96, 192, 384), t_dim: int = 128, blocks: int = 3
    ):
        super().__init__()
        self.k = k
        self.window = 2 * k + 1
        dims = list(hidden)
        self.t_embed = nn.Sequential(
            SinusoidalEmbedding(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.stem = nn.Conv1d(
            self.window, dims[0], 3, padding=1, padding_mode="circular"
        )
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        for i in range(len(dims) - 1):
            self.down.append(
                nn.ModuleList(
                    [ResBlock1d(dims[i], dims[i], t_dim) for _ in range(blocks)]
                )
            )
            self.pool.append(
                nn.Conv1d(
                    dims[i],
                    dims[i + 1],
                    4,
                    stride=2,
                    padding=1,
                    padding_mode="circular",
                )
            )
        self.mid = nn.ModuleList(
            [ResBlock1d(dims[-1], dims[-1], t_dim) for _ in range(blocks)]
        )
        self.up, self.upb = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(len(dims) - 1)):
            self.up.append(
                nn.ConvTranspose1d(dims[i + 1], dims[i], 4, stride=2, padding=1)
            )
            self.upb.append(
                nn.ModuleList(
                    [
                        ResBlock1d(dims[i] * 2 if b == 0 else dims[i], dims[i], t_dim)
                        for b in range(blocks)
                    ]
                )
            )
        self.out_norm = _gn(dims[0])
        self.out = nn.Conv1d(
            dims[0], self.window, 3, padding=1, padding_mode="circular"
        )
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: [B, 2k+1, X], t: [B] in [0,1] -> eps of the same shape."""
        emb = self.t_embed(t)
        h = self.stem(x)
        skips = []
        for blocks, pool in zip(self.down, self.pool):
            for b in blocks:
                h = b(h, emb)
            skips.append(h)
            h = pool(h)
        for b in self.mid:
            h = b(h, emb)
        for up, blocks in zip(self.up, self.upb):
            h = up(h)
            s = skips.pop()
            if h.shape[-1] != s.shape[-1]:
                h = F.interpolate(h, size=s.shape[-1], mode="nearest")
            h = torch.cat([h, s], dim=1)
            for b in blocks:
                h = b(h, emb)
        return self.out(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# 3. Gamma from the data covariance (Appendix B)
# ---------------------------------------------------------------------------
def build_gamma_circulant(
    train_states: torch.Tensor, ridge: float = 0.0
) -> Tuple[torch.Tensor, Dict]:
    """Gamma = Q Lambda (Lambda + I)^-1 Q^-1 for a spatially homogeneous state.

    KS lives on a periodic domain and is statistically homogeneous in space, so the
    single-state covariance ``Sigma_x`` is **circulant**: its eigenvectors are the DFT
    basis and its eigenvalues are the spatial power spectrum. The Appendix-B construction
    is then available in closed form -- no dense eigendecomposition, no approximation --
    and Gamma is itself circulant with eigenvalues ``lam_j / (lam_j + 1)``.

    train_states : [N, X] normalised states from the TRAINING split only.
    returns      : (Gamma [X, X] symmetric PSD, metadata)
    """
    x = train_states - train_states.mean(0, keepdim=True)
    N, X = x.shape
    # circulant (spatially averaged) covariance via the power spectrum
    F_x = torch.fft.rfft(x.double(), dim=-1)
    psd = (F_x.abs() ** 2).mean(0) / X  # eigenvalues of Sigma_x on the DFT basis
    lam = torch.zeros(X, dtype=torch.float64, device=x.device)
    lam[0] = psd[0]
    n_pos = (X - 1) // 2
    lam[1 : n_pos + 1] = psd[1 : n_pos + 1]
    lam[X - n_pos :] = psd[1 : n_pos + 1].flip(0)
    if X % 2 == 0:
        lam[X // 2] = psd[-1]
    lam = lam.clamp_min(0.0) + ridge
    g_eig = lam / (lam + 1.0)  # Lambda (Lambda + I)^-1
    # circulant matrix with these DFT eigenvalues: first column is the inverse FFT
    col = torch.fft.ifft(g_eig.to(torch.complex128)).real
    idx = (
        torch.arange(X, device=x.device)[:, None]
        - torch.arange(X, device=x.device)[None, :]
    ) % X
    Gamma = col[idx]
    Gamma = 0.5 * (Gamma + Gamma.T)  # symmetrise away round-off
    # Gamma is PSD by construction (lam/(lam+1) >= 0), but the ifft -> matrix round trip
    # leaves eigenvalues of order -1e-8. Those are harmless on their own and catastrophic
    # once multiplied by sigma^2/mu^2, which reaches ~1e6 at high noise levels: the
    # likelihood covariance Sigma_y + (sigma^2/mu^2) A Gamma A^T then becomes *indefinite*
    # and the Gaussian solve returns garbage. Project onto the PSD cone exactly.
    # Clamping the eigenvalues is not enough on its own: reconstructing
    # (evecs*evals) @ evecs.T reintroduces negatives of order 1e-8 through matmul
    # round-off. With a *full* observation mask and ratio = sigma^2/mu^2 ~ 1e6 that
    # becomes ratio*|lam_min| ~ 2e-2, which swamps Sigma_y = 4e-4 and makes the
    # likelihood covariance indefinite -- the Cholesky then fails outright. Shift the
    # whole spectrum up instead, by an amount set by the observed round-off. The shift is
    # ~2e-8 against a spectrum whose maximum is ~0.94, so it is numerically decisive and
    # physically irrelevant.
    evals = torch.linalg.eigvalsh(Gamma)
    n_neg = int((evals < 0).sum())
    lam_min, lam_max = float(evals.min()), float(evals.max())
    shift = max(0.0, -lam_min) + 1e-7 * max(lam_max, 1.0)
    Gamma = Gamma + shift * torch.eye(X, device=Gamma.device, dtype=Gamma.dtype)
    Gamma = 0.5 * (Gamma + Gamma.T)
    psd_shift = shift
    meta = {
        "n_states": int(N),
        "grid": int(X),
        "psd_negative_eigenvalues": n_neg,
        "psd_shift": psd_shift,
        "sigma_x_eig_min": float(lam.min()),
        "sigma_x_eig_max": float(lam.max()),
        "gamma_eig_min": float(g_eig.min()),
        "gamma_eig_max": float(g_eig.max()),
        "gamma_trace_over_X": float(Gamma.diagonal().mean()),
        "ridge": ridge,
        "construction": (
            "circulant closed form of Q Lam (Lam+I)^-1 Q^-1; DFT "
            "eigenbasis; eigenvalues are the spatial power spectrum "
            "of the TRAINING states"
        ),
    }
    return Gamma.to(train_states.dtype), meta


# ---------------------------------------------------------------------------
# 4. Observation operator
# ---------------------------------------------------------------------------
@dataclass
class LinearObservation:
    """y_i = A_i x(t_i) + eta,  A_i a selection (masking) operator.

    ``frames``  : [m] indices into the trajectory
    ``mask``    : [m, X] 1 where observed
    ``sigma_y`` : scalar observation-noise std
    Linear and diagonal-in-space, so ``A Gamma A^T`` is the submatrix of Gamma on the
    observed indices -- formed exactly, never approximated.
    """

    frames: torch.Tensor
    mask: torch.Tensor
    sigma_y: float

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, X] -> [B, m, X], zeroed outside the mask."""
        return x[:, self.frames] * self.mask[None]


def _psd_solve(C: torch.Tensor, r: torch.Tensor, max_tries: int = 6) -> torch.Tensor:
    """Solve ``C z = r`` for a covariance that should be PD, with a jitter fallback.

    ``C = Sigma_y + (sigma^2/mu^2) A Gamma A^T`` is PD in exact arithmetic. Round-off in
    Gamma, amplified by ``sigma^2/mu^2`` (which reaches ~1e6 at t -> 1), can still tip it
    indefinite -- most easily when the mask is full, so that the whole near-singular
    spectrum of Gamma is in play. Retry with growing jitter rather than crashing; the
    jitter needed is many orders below Sigma_y whenever it is needed at all.
    """
    jitter = 0.0
    eye = torch.eye(C.shape[-1], device=C.device, dtype=C.dtype)
    scale = float(C.diagonal().mean())
    for i in range(max_tries):
        try:
            L = torch.linalg.cholesky(C + jitter * eye)
            return torch.cholesky_solve(r.unsqueeze(-1), L).squeeze(-1)
        except Exception:  # noqa: BLE001
            jitter = (1e-12 if jitter == 0.0 else jitter * 100.0) * max(scale, 1.0)
    return torch.linalg.lstsq(C + jitter * eye, r.unsqueeze(-1)).solution.squeeze(-1)


# ---------------------------------------------------------------------------
# 5. The method
# ---------------------------------------------------------------------------
class SDA:
    """Score-based data assimilation over a trajectory ``x_{1:L}``.

    ``k`` is fixed by the trained network; ``L`` is free at inference thanks to the
    Algorithm-2 composition.
    """

    name = "SDA"

    def __init__(
        self,
        net: LocalScoreUNet,
        sched: CosineVPSchedule,
        device,
        gamma: Optional[torch.Tensor] = None,
        gamma_scale: float = 1.0,
        gamma_floor: float = 0.0,
    ):
        self.net = net
        self.sched = sched
        self.device = device
        self.k = net.k
        self.window = net.window
        # Gamma of App. B; None means the paper's practical fallback Gamma = gamma_scale * I
        self.Gamma = gamma
        self.gamma_scale = gamma_scale
        # Eigenvalue floor on Gamma. Appendix B gives Gamma = Lam(Lam+I)^-1, whose
        # eigenvalues follow the data spectrum. For KS the spatial power spectrum spans
        # about six decades, so Gamma is near-singular in the high-wavenumber directions --
        # and there the likelihood covariance collapses to Sigma_y, making the model
        # maximally confident exactly where the Tweedie estimate is least reliable (at
        # t -> 1, where mu = 1e-3 and x_hat is amplified a thousandfold). Flooring the
        # spectrum bounds that confidence. floor = 0 recovers the raw Appendix-B matrix;
        # a large floor approaches the paper's own practical fallback Gamma = c*I.
        self.gamma_floor = gamma_floor
        self.reset_counters()

    def reset_counters(self):
        self.nfe = 0  # score-network evaluations (segments are batched)
        self.n_backward = 0  # backward passes for the likelihood score
        self.n_segment_evals = 0  # individual segment evaluations, the real work

    # -- Algorithm 2: compose local scores into the full-trajectory score ----
    def prior_score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """s_phi(x_{1:L}(t), t) for x: [B, L, X] -> [B, L, X].

        Follows Algorithm 2 exactly: the first k+1 elements come from the leading segment,
        the last k+1 from the trailing segment, and every interior element from the centre
        of its own segment. All segments are evaluated in one batched call.
        """
        B, L, X = x.shape
        k, W = self.k, self.window
        assert L >= W, f"trajectory length {L} shorter than the blanket {W}"

        starts = list(range(0, L - W + 1))  # every window position
        seg = torch.stack([x[:, s : s + W] for s in starts], dim=1)  # [B, S, W, X]
        S = seg.shape[1]
        flat = seg.reshape(B * S, W, X)
        t_rep = t.repeat_interleave(S) if t.numel() == B else t.expand(B * S)
        eps = self.net(flat, t_rep).reshape(B, S, W, X)
        self.nfe += 1
        self.n_segment_evals += S

        out = x.new_zeros(B, L, X)
        out[:, : k + 1] = eps[:, 0, : k + 1]  # leading segment
        out[:, L - k - 1 :] = eps[:, S - 1, k:]  # trailing segment
        if L > W:  # interior centres
            # element i (k+1 <= i <= L-k-2) is the centre of the window starting at i-k
            centre = eps[:, 1 : S - 1, k]  # [B, S-2, X]
            out[:, k + 1 : L - k - 1] = centre
        sigma = self.sched.sigma(t).view(-1, 1, 1)
        return -out / sigma  # s = -eps / sigma

    # -- Algorithm 3: posterior score ----------------------------------------
    def posterior_score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor],
        obs: Optional[LinearObservation],
    ) -> torch.Tensor:
        """grad log p(x(t) | y) = s_x + s_y, per Algorithm 3.

        s_y is the gradient of the Gaussian of Eq. 15, whose covariance is
        ``Sigma_y + (sigma^2/mu^2) A Gamma A^T`` -- **not** ``Sigma_y`` alone. The gradient
        is taken through the Tweedie estimate and therefore through the score network.
        """
        if y is None or obs is None:
            with torch.no_grad():
                return self.prior_score(x, t)

        x = x.detach().requires_grad_(True)
        s_x = self.prior_score(x, t)
        mu = self.sched.mu(t).view(-1, 1, 1)
        sig = self.sched.sigma(t).view(-1, 1, 1)
        x_hat = (x + sig**2 * s_x) / mu  # Eq. 10

        resid = obs.apply(x_hat) - y * obs.mask[None]  # [B, m, X]
        ratio = float((sig[0, 0, 0] ** 2 / mu[0, 0, 0] ** 2).item())
        quad = self._mahalanobis(resid, obs, ratio)
        log_lik = -0.5 * quad.sum()
        s_y = torch.autograd.grad(log_lik, x)[0]
        self.n_backward += 1
        return (s_x + s_y).detach()

    def _mahalanobis(
        self, resid: torch.Tensor, obs: LinearObservation, ratio: float
    ) -> torch.Tensor:
        """r^T (Sigma_y + ratio * A Gamma A^T)^-1 r, summed over observation times."""
        m = obs.mask
        total = resid.new_zeros(resid.shape[0])
        for j in range(resid.shape[1]):
            idx = torch.nonzero(m[j] > 0, as_tuple=True)[0]
            r = resid[:, j][:, idx]  # [B, n_obs_j]
            if self.Gamma is None:
                cov_diag = obs.sigma_y**2 + ratio * self.gamma_scale
                total = total + (r**2).sum(-1) / cov_diag
            else:
                # A Gamma A^T is exactly the submatrix of Gamma on the observed indices,
                # because A is a selection operator. The solve is done in float64: with
                # Sigma_y = 0.05^2 and sigma^2/mu^2 up to ~1e6 the condition number of C
                # reaches ~4e8, which float32 cannot carry.
                G = self.Gamma[idx][:, idx].double()  # A Gamma A^T
                if self.gamma_floor > 0:
                    G = G + self.gamma_floor * torch.eye(
                        len(idx), device=G.device, dtype=G.dtype
                    )
                C = (obs.sigma_y**2) * torch.eye(
                    len(idx), device=r.device, dtype=torch.float64
                ) + ratio * G
                sol = _psd_solve(C, r.double()).to(r.dtype)
                total = total + (r * sol).sum(-1)
        return total

    # -- Algorithm 4: predictor-corrector sampling ---------------------------
    @torch.no_grad()
    def sample(
        self,
        L: int,
        X: int,
        B: int,
        *,
        y: Optional[torch.Tensor] = None,
        obs: Optional[LinearObservation] = None,
        n_steps: int = 256,
        corrections: int = 1,
        tau: float = 0.5,
        seed: int = 0,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Draw ``n_samples`` posterior trajectories -> [n_samples, B, L, X].

        Exponential-integrator predictor (Eq. 16) with ``corrections`` Langevin steps
        (Eq. 17) after each predictor step, exactly as Algorithm 4. Each sample starts from
        an independent draw ``x(1) ~ N(0, Sigma(1))``; nothing about the observations or the
        truth enters the initialisation.
        """
        g = torch.Generator(device=self.device).manual_seed(seed)
        ts = torch.linspace(1.0, 0.0, n_steps + 1, device=self.device)
        out = []
        for _ in range(n_samples):
            sig1 = self.sched.sigma(ts[:1])
            x = torch.randn((B, L, X), device=self.device, generator=g) * sig1
            for i in range(n_steps):
                t_i = ts[i].expand(B)
                t_prev = ts[i + 1].expand(B)
                mu_i, sig_i = self.sched.mu(t_i), self.sched.sigma(t_i)
                mu_p, sig_p = self.sched.mu(t_prev), self.sched.sigma(t_prev)

                # --- predictor (Eq. 16) ---
                with torch.enable_grad():
                    s = self.posterior_score(x, t_i, y, obs)
                r_mu = (mu_p / mu_i).view(-1, 1, 1)
                r_sg = (sig_p / sig_i).view(-1, 1, 1)
                x = r_mu * x + (r_mu - r_sg) * (sig_i**2).view(-1, 1, 1) * s

                # --- corrector: C Langevin steps (Eq. 17, Alg. 4 lines 5-9) ---
                for _c in range(corrections):
                    with torch.enable_grad():
                        s_c = self.posterior_score(x, t_prev, y, obs)
                    norm2 = (s_c**2).flatten(1).sum(1).clamp_min(1e-12)
                    delta = tau * s_c[0].numel() / norm2  # tau * dim(s) / ||s||^2
                    d = delta.view(-1, 1, 1)
                    noise = torch.randn(x.shape, device=self.device, generator=g)
                    x = x + d * s_c + (2 * d).sqrt() * noise
            out.append(x.detach())
        return torch.stack(out)

    # -- training objective (Eq. 6 / Alg. 1) ---------------------------------
    def loss(self, segments: torch.Tensor, gen=None) -> torch.Tensor:
        """eps-matching on segments ``x_{i-k:i+k}``: [B, 2k+1, X]."""
        B = segments.shape[0]
        t = torch.rand(B, device=segments.device, generator=gen)
        eps = torch.randn(segments.shape, device=segments.device, generator=gen)
        xt = self.sched.perturb(segments, t, eps)
        return F.mse_loss(self.net(xt, t), eps)
