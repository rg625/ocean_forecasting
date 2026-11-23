"""Gaussian latent background and representation error for KAE 4D-Var on KS.

The default KAE objective since the ICLR 2027 revision is

    J(z) = sum_i ||M_i (D(e^{K tau_i} z) - y_i)||^2 / (2 s2) + 1/2 (z - mu)^T B^{-1} (z - mu),
    s2 = sigma_obs^2 + sigma_m^2,

with (mu, B) from encoded TRAINING states and sigma_m the RMS misfit base 4D-Var reaches on
clean, fully observed VALIDATION problems with the same schedule.  Pass the result of
``kae_background`` as ``SolveConfig(background=...)``.  Same estimator as the revision's
R1 experiment (revision_42477/experiments/rev_ks_prior_noise.py).
"""
from __future__ import annotations

import numpy as np
import torch

from data_assimilation.ks.init_utils import encode_field, informed_init
from data_assimilation.ks.protocol import DT, build_problem


@torch.no_grad()
def latent_prior(kae, train, n: int = 4096, seed: int = 0, floor: float = 1e-3):
    g = np.random.default_rng(seed)
    sim = torch.as_tensor(g.integers(0, train.n_sim, n), device=train.device)
    t = torch.as_tensor(g.integers(100, train.n_t, n), device=train.device)
    z = torch.cat(
        [encode_field(kae, train.frames(sim[i : i + 256], t[i : i + 256])) for i in range(0, n, 256)]
    )
    mu = z.mean(0)
    zc = z - mu
    cov = zc.T @ zc / (n - 1)
    # layer-normalised latents span a low-dimensional subspace: floor the eigenvalues
    ev, V = torch.linalg.eigh(cov.double())
    ev = ev.clamp_min(floor * float(ev.max()))
    return mu, ((V / ev) @ V.T).to(zc.dtype)


def representation_error(kae, K, val, frames, iters: int, lr: float, n: int = 24, seed: int = 7) -> float:
    from data_assimilation.ks.methods import KAE4DVar, SolveConfig

    prob = build_problem(val, name="sigma_m", n_problems=n, taus=np.asarray(frames) * DT, seed=seed)
    m = KAE4DVar(kae, K, int(K.shape[-1]), 1.0, "expm")
    z0 = informed_init(kae, K, prob, val.device)
    out = m.solve(prob, val, SolveConfig(iters=iters, lr=lr, init="given", init_value=z0))
    with torch.no_grad():
        pred = m.predict_at(out["control"], prob.taus, out["prep"])
        y = torch.as_tensor(prob.y, device=val.device)
        mask = torch.as_tensor(prob.mask, device=val.device)[:, None, :]
        return float(((((pred - y) * mask) ** 2).sum() / mask.expand_as(y).sum()).sqrt())


def kae_background(kae, K, train, val, frames, noise_std: float, iters: int, lr: float, sigma_m=None):
    mu, Binv = latent_prior(kae, train)
    if sigma_m is None:
        sigma_m = representation_error(kae, K, val, frames, iters, lr)
    return {"mu": mu, "Binv": Binv, "s2": float(noise_std) ** 2 + float(sigma_m) ** 2, "sigma_m": float(sigma_m)}
