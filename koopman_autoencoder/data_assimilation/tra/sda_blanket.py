"""Score-based DA over an ACDM denoiser, using its own 3-frame window as the blanket.

Why this is possible at all.  ``acdm-r20`` is trained with ``conditioningIntegration =
"noisy"``: the two conditioning frames and the target frame are ALL noised to the same
diffusion level, concatenated along channels, and passed to one U-Net whose output has the
same channel count as its input (15 = 3 frames x 5).  turbpred then discards the two
conditioning channels ("discard prediction of conditioning").  Keeping them instead gives
epsilon over the whole 3-frame window, i.e. a LOCAL JOINT SCORE over a Markov blanket of
k = 1 -- exactly the object Algorithm 2 of Rozet & Louppe (2023) composes.  No retraining,
no architecture change, nothing in either codebase modified.

``acdm-r20_ncn`` is trained with ``"clean"`` conditioning (ncn = no conditioning noise), so
its context is never noised during training and a noised window is out of distribution for
it.  It is nevertheless run through **exactly this same path**, by request: identical
treatment is the more defensible protocol, and whatever difference appears between the two
is itself the measurement.  Nothing is special-cased for either model.

Two things are taken from ACDM rather than assumed:
  * the noise schedule (betas / alphasCumprod) -- using any other schedule would evaluate
    the denoiser at noise levels it never saw;
  * the timestep grid, which is only ``timesteps`` long (20 for r20), so the sampler runs
    on that grid rather than a finer invented one.
"""

from __future__ import annotations


import torch


class ACDMBlanketScore:
    """Local joint score over ACDM's own conditioning window."""

    def __init__(self, adapter, n_fields_total: int):
        """`adapter` is a TurbpredAdapter holding a diffusion checkpoint."""
        dm = adapter.model.modelDecoder
        assert adapter.is_diffusion, f"{adapter.name} is not a diffusion model"
        self.ad = adapter
        self.dm = dm
        # the U-Net sees prevSteps CONDITIONING frames plus the target frame, so the
        # window is prevSteps + 1 (= 3 for "+Prev", matching its 15 = 3 x 5 channels)
        self.window = adapter.n_control_frames + 1
        self.k = self.window // 2  # blanket half-width: 2k+1 = window
        self.C = n_fields_total  # per-frame channels incl. params
        self.timesteps = int(dm.timesteps)
        # ACDM's own schedule -- never a substitute
        self.sqrt_ac = dm.sqrtAlphasCumprod
        self.sqrt_1mac = dm.sqrtOneMinusAlphasCumprod
        self.betas = dm.betas
        self.sqrt_recip_alphas = dm.sqrtRecipAlphas
        self.sqrt_post_var = dm.sqrtPosteriorVariance
        self.nfe = 0

    def eps_window(self, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """epsilon over a full window.  w: [B, W, C, H, W2] -> same shape.

        The U-Net consumes the window as concatenated channels; its output has the SAME
        channel count, so it is reshaped straight back into per-frame epsilon.
        """
        B, W, C, H, W2 = w.shape
        flat = w.reshape(B, W * C, H, W2)
        out = self.dm.unet(flat, t)
        self.nfe += 1
        return out.reshape(B, W, C, H, W2)

    def prior_score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Algorithm 2 composition of local scores.  x: [B, L, C, H, W2].

        Leading k+1 frames come from the first window, trailing k+1 from the last, and
        every interior frame from the CENTRE of its own window -- the same composition the
        KS campaign uses, transposed to 2-D fields.
        """
        B, L, C, H, W2 = x.shape
        k, W = self.k, self.window
        assert L >= W, f"trajectory length {L} shorter than the blanket window {W}"
        starts = list(range(0, L - W + 1))
        seg = torch.stack([x[:, s : s + W] for s in starts], dim=1)  # [B,S,W,C,H,W2]
        S = seg.shape[1]
        t_rep = t.repeat_interleave(S) if t.numel() == B else t.expand(B * S)
        eps = self.eps_window(seg.reshape(B * S, W, C, H, W2), t_rep).reshape(
            B, S, W, C, H, W2
        )
        out = x.new_zeros(B, L, C, H, W2)
        out[:, : k + 1] = eps[:, 0, : k + 1]
        out[:, L - k - 1 :] = eps[:, S - 1, k:]
        if L > W:
            for i in range(k + 1, L - k - 1):
                out[:, i] = eps[:, i - k, k]
        return out

    # -- Tweedie ------------------------------------------------------------
    def x_hat(self, x: torch.Tensor, eps: torch.Tensor, ti: int) -> torch.Tensor:
        """E[x(0) | x(t)] from epsilon, using ACDM's schedule."""
        return (x - self.sqrt_1mac[ti] * eps) / self.sqrt_ac[ti]


class BlanketSDA:
    """Posterior sampling with the ACDM blanket score and a Gaussian observation term."""

    def __init__(
        self,
        score: ACDMBlanketScore,
        sigma_y: float = 0.05,
        gamma: float = 1e-2,
        corrections: int = 1,
        tau: float = 0.5,
    ):
        self.s = score
        self.sigma_y = float(sigma_y)
        self.gamma = float(gamma)
        self.corrections = int(corrections)
        self.tau = float(tau)

    def _guided(self, x, ti, obs_frames, y, mask, param_ch):
        """grad log p(x(t) | y): prior score plus the observation term through Tweedie."""
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            t = torch.full((xg.shape[0],), ti, device=xg.device, dtype=torch.long)
            eps = self.s.prior_score(xg, t)
            xh = self.s.x_hat(xg, eps, ti)
            r = (xh[:, obs_frames] - y) * mask
            # Sigma_y + (sigma^2/mu^2) Gamma, the paper's covariance, scalar Gamma here
            ratio = float((self.s.sqrt_1mac[ti] / self.s.sqrt_ac[ti]) ** 2)
            cov = self.sigma_y**2 + ratio * self.gamma
            nll = 0.5 * (r**2).sum() / cov
            g = torch.autograd.grad(nll, xg)[0]
        prior = -eps / self.s.sqrt_1mac[ti]
        return prior - g

    @torch.no_grad()
    def sample(
        self, shape, obs_frames, y, mask, param_ch=None, seed=0, n_samples: int = 1
    ):
        """Draw posterior trajectories.  shape = (B, L, C, H, W2)."""
        dev = y.device
        g = torch.Generator(device=dev).manual_seed(seed)
        out = []
        for s_i in range(n_samples):
            x = torch.randn(shape, device=dev, generator=g)
            for ti in reversed(range(self.s.timesteps)):
                sc = self._guided(x, ti, obs_frames, y, mask, param_ch)
                eps = -sc * self.s.sqrt_1mac[ti]
                mean = self.s.sqrt_recip_alphas[ti] * (
                    x - self.s.betas[ti] * eps / self.s.sqrt_1mac[ti]
                )
                x = mean
                if ti != 0:
                    x = x + self.s.sqrt_post_var[ti] * torch.randn(
                        shape, device=dev, generator=g
                    )
                for _ in range(self.corrections):  # Langevin corrector
                    sc = self._guided(x, ti, obs_frames, y, mask, param_ch)
                    d = self.tau * sc.numel() / sc.pow(2).sum().clamp_min(1e-12)
                    x = (
                        x
                        + d * sc
                        + (2 * d).sqrt() * torch.randn(shape, device=dev, generator=g)
                    )
            out.append(x)
        return torch.stack(out)
