"""Score-based data assimilation for KS (Rozet & Louppe, 2023), P2 baseline.

This is a *posterior* method, not a forecaster with observations bolted on.  It has two
distinct pieces, which is what separates it from the unconditional climatology model:

1. **Trajectory prior.**  A denoising score model ``s_phi(x_{0:L}, sigma)`` trained on
   windows of ``L`` consecutive KS frames from the training split.  It knows nothing
   about observations and defines ``p(x_{0:L})``.

2. **Observation likelihood.**  With ``y_i = H_i x(t_i) + eps_i``, ``eps_i ~ N(0, sigma_y^2)``,
   the posterior score is ``grad log p(x|y) = grad log p(x) + grad log p(y|x)``.  The second
   term is intractable, and is approximated as in the SDA paper by a Gaussian around the
   Tweedie posterior mean ``xhat(x_t)``:

       log p(y | x_t)  ~=  -|| y - H xhat(x_t) ||^2 / (2 (sigma_y^2 + gamma_t))

   whose gradient is backpropagated through the score network.  Sampling the reverse
   diffusion with this corrected score draws from an approximate ``p(x_{0:L} | y)``.

The observations, masks, noise and trajectories are exactly those produced by
``data_assimilation.ks.protocol``, so the comparison against the two 4D-Var methods is like for like.
Because the method is stochastic, several posterior samples are drawn per problem and
both the ensemble-mean error and the ensemble spread are reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Score network: a 2-D U-Net over (time, space) windows
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
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        a = t.float()[:, None] * freqs[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class ResBlock2d(nn.Module):
    """Circular in space, replicate in time (a window is not periodic in time)."""

    def __init__(self, c_in: int, c_out: int, t_dim: int):
        super().__init__()
        self.norm1, self.norm2 = _gn(c_in), _gn(c_out)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=0)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=0)
        self.emb = nn.Linear(t_dim, c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    @staticmethod
    def _pad(x):
        x = F.pad(x, (1, 1, 0, 0), mode="circular")  # space: periodic
        return F.pad(x, (0, 0, 1, 1), mode="replicate")  # time: replicate

    def forward(self, x, emb):
        h = self.conv1(self._pad(F.silu(self.norm1(x))))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(self._pad(F.silu(self.norm2(h))))
        return h + self.skip(x)


class ScoreUNet2d(nn.Module):
    """``eps``-prediction network over a window ``[B, 1, L, X]``."""

    def __init__(self, hidden=(64, 128, 256), t_dim: int = 128, blocks: int = 2):
        super().__init__()
        self.t_embed = nn.Sequential(
            SinusoidalEmbedding(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        dims = list(hidden)
        self.stem = nn.Conv2d(1, dims[0], 3, padding=1, padding_mode="circular")
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        for i in range(len(dims) - 1):
            self.down.append(
                nn.ModuleList(
                    [ResBlock2d(dims[i], dims[i], t_dim) for _ in range(blocks)]
                )
            )
            self.pool.append(nn.Conv2d(dims[i], dims[i + 1], 4, stride=2, padding=1))
        self.mid = nn.ModuleList(
            [ResBlock2d(dims[-1], dims[-1], t_dim) for _ in range(blocks)]
        )
        self.up, self.upb = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(len(dims) - 1)):
            self.up.append(
                nn.ConvTranspose2d(dims[i + 1], dims[i], 4, stride=2, padding=1)
            )
            self.upb.append(
                nn.ModuleList(
                    [
                        ResBlock2d(dims[i] * 2 if b == 0 else dims[i], dims[i], t_dim)
                        for b in range(blocks)
                    ]
                )
            )
        self.out_norm = _gn(dims[0])
        self.out = nn.Conv2d(dims[0], 1, 3, padding=1, padding_mode="circular")
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
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
            if h.shape[-2:] != s.shape[-2:]:
                h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = torch.cat([h, s], dim=1)
            for b in blocks:
                h = b(h, emb)
        return self.out(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# Variance-preserving diffusion
# ---------------------------------------------------------------------------
@dataclass
class VPSchedule:
    n_steps: int = 1000
    beta_min: float = 1e-4
    beta_max: float = 2e-2

    def build(self, device):
        betas = torch.linspace(
            self.beta_min, self.beta_max, self.n_steps, device=device
        )
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        return betas, alphas, abar


# ---------------------------------------------------------------------------
class ScoreBasedDA:
    """Posterior sampling of a KS trajectory window conditioned on sparse observations."""

    name = "SDA"
    control_kind = "trajectory posterior"

    def __init__(
        self,
        net: ScoreUNet2d,
        sched: VPSchedule,
        window: int,
        device,
        guidance: float = 1.0,
        n_sample_steps: int = 256,
    ):
        self.net = net
        self.sched = sched
        self.L = window
        self.device = device
        self.guidance = guidance
        self.n_sample_steps = n_sample_steps
        self.betas, self.alphas, self.abar = sched.build(device)
        self.nfe = 0

    # -- training helper -----------------------------------------------------
    def loss(self, x0: torch.Tensor, gen=None) -> torch.Tensor:
        """Standard eps-matching loss. ``x0``: [B, 1, L, X]."""
        B = x0.shape[0]
        t = torch.randint(0, self.sched.n_steps, (B,), device=x0.device, generator=gen)
        a = self.abar[t][:, None, None, None]
        eps = torch.randn(x0.shape, device=x0.device, generator=gen)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        return F.mse_loss(self.net(xt, t.float()), eps)

    # -- posterior sampling --------------------------------------------------
    def sample(
        self,
        obs_idx: np.ndarray,
        y: torch.Tensor,
        mask: torch.Tensor,
        sigma_y: float,
        n_samples: int,
        B: int,
        seed: int = 0,
        guidance: Optional[float] = None,
        return_prior: bool = False,
    ) -> torch.Tensor:
        """Draw ``n_samples`` posterior trajectory windows given the observations.

        Reverse DDIM sampling with the posterior score

            grad log p(x_t | y) = grad log p(x_t) + grad log p(y | x_t),

        where the first term is the learned prior score, ``-eps_phi/sqrt(1-abar_t)``, and the
        second is the SDA/DPS Gaussian approximation of the observation likelihood taken
        around the Tweedie posterior mean ``xhat_0(x_t)``:

            log p(y | x_t) ~= -|| M o (y - H xhat_0(x_t)) ||^2 / (2 (sigma_y^2 + gamma (1-abar_t)/abar_t)).

        Its gradient is obtained by differentiating through the score network.  The
        correction enters as an adjusted noise prediction

            eps_post = eps_phi - sqrt(1-abar_t) grad_x log p(y | x_t),

        which is then used in the ordinary DDIM update, so the conditional and
        unconditional samplers differ only by that one term.  Setting
        ``return_prior=True`` disables the likelihood entirely and returns prior samples
        from the same seed and schedule, which is what makes the two directly comparable.

        obs_idx : [n_obs] frame index of each observation within the window
        y       : [n_obs, B, X] observed values (normalised)
        mask    : [n_obs, X] 1 where observed
        returns : [n_samples, B, L, X]
        """
        g = torch.Generator(device=self.device).manual_seed(seed)
        gamma = self.guidance if guidance is None else guidance
        self.nfe = 0
        self.n_backward = 0
        idx = torch.as_tensor(obs_idx, device=self.device, dtype=torch.long)
        y_bnx = y.permute(1, 0, 2).contiguous()  # [B, n_obs, X]
        ts = (
            torch.linspace(
                self.sched.n_steps - 1, 0, self.n_sample_steps, device=self.device
            )
            .round()
            .long()
        )
        out = []
        for s_i in range(n_samples):
            x = torch.randn(
                (B, 1, self.L, y.shape[-1]), device=self.device, generator=g
            )
            for k in range(len(ts)):
                t = ts[k]
                a_t = self.abar[t]
                a_prev = (
                    self.abar[ts[k + 1]]
                    if k + 1 < len(ts)
                    else torch.tensor(1.0, device=self.device)
                )
                if return_prior:
                    with torch.no_grad():
                        eps = self.net(x, t.float().expand(B))
                    self.nfe += 1
                else:
                    x = x.detach().requires_grad_(True)
                    eps = self.net(x, t.float().expand(B))
                    self.nfe += 1
                    x0_hat = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
                    resid = (x0_hat[:, 0][:, idx] - y_bnx) * mask[None]
                    var = sigma_y**2 + gamma * (1 - a_t) / a_t
                    log_lik = -(resid**2).sum() / (2 * var)
                    grad = torch.autograd.grad(log_lik, x)[0]
                    self.n_backward += 1
                    eps = (eps - (1 - a_t).sqrt() * grad).detach()
                    x = x.detach()
                x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).detach()
                x = (a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps).detach()
            out.append(x[:, 0])  # [B, L, X]
        return torch.stack(out)
