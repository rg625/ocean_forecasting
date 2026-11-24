"""First guess for KAE 4D-Var from the nearest observation.

Standard variational assimilation starts from a background state, not from nothing. The
campaign started the latent at zero, which is the model's mean field and a poor basin once
the first observation is far from t_0. This builds z0 = e^{-K tau_1} enc(y_1): the nearest
observation, encoded and propagated back to t_0.

Only observed quantities are used. With partial coverage the unobserved points are filled
with the dataset mean (zero in normalised units) before encoding, which is the same
uninformed value the old initialisation used everywhere.
"""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict


def encode_field(kae, u: torch.Tensor) -> torch.Tensor:
    """[B, X] normalised field -> [B, D] latent."""
    return kae.present_encoding(
        TensorDict({"u": u.unsqueeze(1).unsqueeze(-1)}, batch_size=[u.shape[0], 1]),
        cond_input=None,
    )


def informed_init(kae, K: torch.Tensor, problem, device) -> torch.Tensor:
    """z0 from the nearest observation, propagated back to t_0."""
    i = int(np.argmin(problem.taus))
    y = torch.as_tensor(problem.y[i], device=device)  # [B, X]
    m = torch.as_tensor(problem.mask[i], device=device)  # [X]
    if float(m.min()) < 1.0:  # partial coverage: fill the gaps
        y = y * m
    with torch.no_grad():
        z = encode_field(kae, y)
        return z @ torch.matrix_exp(-K * float(problem.taus[i])).T


def informed_state(kae, K: torch.Tensor, problem, device) -> torch.Tensor:
    """decode(e^{-K tau_1} enc(y_1)): the pulled-back first guess, in data space."""
    with torch.no_grad():
        return kae.decode(informed_init(kae, K, problem, device))["u"].squeeze(-1)


def init_kwargs(mode: str, kae, K, problem, device) -> dict:
    """SolveConfig kwargs for the requested initialisation."""
    if mode == "obs":
        return {"init": "given", "init_value": informed_init(kae, K, problem, device)}
    return {"init": mode}


def informed_init_reg(kae, K: torch.Tensor, problem, device, lam: float = 1e-2, mu=None) -> torch.Tensor:
    """Regularised pull-back (ICLR 2027 revision, R15): argmin ||e^{K tau_1} z - E(y_1)||^2 + lam ||z - mu||^2.

    Same first guess as ``informed_init`` for well-conditioned generators, but modes the forward map
    damps below sqrt(lam) are set to ``mu`` (0 if None) instead of being amplified by e^{|Re lambda| tau_1}.
    """
    i = int(np.argmin(problem.taus))
    y = torch.as_tensor(problem.y[i], device=device)
    m = torch.as_tensor(problem.mask[i], device=device)
    if float(m.min()) < 1.0:
        y = y * m
    with torch.no_grad():
        z1 = encode_field(kae, y).double()
        A = torch.matrix_exp(K.double() * float(problem.taus[i]))
        D = A.shape[0]
        mu_ = torch.zeros(D, device=device, dtype=torch.float64) if mu is None else mu.double().to(device)
        lhs = A.T @ A + lam * torch.eye(D, device=device, dtype=torch.float64)
        rhs = z1 @ A + lam * mu_  # rows: (A^T z1)^T
        return torch.linalg.solve(lhs, rhs.T).T.float()
