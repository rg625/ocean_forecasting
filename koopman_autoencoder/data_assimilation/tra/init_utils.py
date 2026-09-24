"""First guess for TRA 4D-Var from the nearest observation.

The campaign starts every control at climatology (z = 0 for the KAE, a zero conditioning
window for the autoregressive models).  That is the model's mean field, and it is a poor
basin once the first observation is several frames away from t_0: on KS the same change
moved the analysis error by 2.5x at delta_f = 1 and 10x at delta_f = 9, because the solve
was settling into a latent whose cost was LOWER than the truth's.  This builds the
analogous informed start here.

Only observed quantities are used.  Unobserved points -- both the sensor mask and the
obstacle interior -- are filled with climatology, which is the same uninformed value the
old initialisation used everywhere, so nothing about the truth leaks in through the gaps.

Observations on TRA are in PHYSICAL units, and climatology is zero only in NORMALISED
units, so the fill is done by mapping a zero control through the adapter rather than by
writing zeros into the physical field.
"""

from __future__ import annotations

import numpy as np
import torch


def _filled(ad, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Physical observation with its gaps at climatology. y [B,C,H,W], m [B,1,H,W]."""
    clim = ad.to_physical(torch.zeros_like(ad.to_model(y)))
    return y * m + clim * (1.0 - m)


def nearest_observation(ad, prob, dev) -> tuple:
    """(filled physical field at the first observation, its lead time in frames)."""
    i = int(np.argmin(prob.offsets))
    y = torch.as_tensor(prob.y[i], device=dev).float()  # [B, C, H, W]
    m = torch.as_tensor(prob.mask[i], device=dev).float()  # [B, 1, H, W]
    return _filled(ad, y, m), float(prob.offsets_eval[i])


def informed_window(ad, prob, dev) -> torch.Tensor:
    """The conditioning window for an autoregressive control: [B, k, C, H, W].

    The window holds k consecutive frames ending at t_0 and only one of them is observed,
    so the observation is repeated across it.  That is wrong as a trajectory, but it is a
    starting point, not an answer: 4D-Var is free to move every frame of it.
    """
    y, _ = nearest_observation(ad, prob, dev)
    return y.unsqueeze(1).expand(-1, ad.n_control_frames, -1, -1, -1).contiguous()


def informed_latent(ad, data, prob, dev) -> torch.Tensor:
    """z_0 = e^{-K tau_1} Enc(y_1): the nearest observation, encoded and pulled back.

    K is conditioned on the simulation parameter, so it is per-problem and the backward
    map is applied with a batched matmul when it comes back rank-3.
    """
    w = informed_window(ad, prob, dev)
    par = data.params_for(torch.as_tensor(prob.sim, device=dev))
    _, tau1 = nearest_observation(ad, prob, dev)
    with torch.no_grad():
        z = ad.encode(w, par)  # [B, D] at t_0 + tau_1
        K = ad.generator(par).detach()
        back = torch.matrix_exp(-K * (tau1 * ad.dt_train))
        return (
            torch.bmm(back, z.unsqueeze(-1)).squeeze(-1)
            if back.dim() == 3
            else z @ back.T
        )
