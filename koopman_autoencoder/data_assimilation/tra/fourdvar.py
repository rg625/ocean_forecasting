# mypy: disable-error-code="var-annotated"
"""4D-Var over a frozen surrogate, in whatever control variable that surrogate exposes.

    J(c) = (1/N) sum_i || M_i . (G(c, tau_i) - y_i) ||^2

``G`` is the model's own forward map and every weight in it is frozen; only ``c`` moves.
The control differs by method and that difference is the point:

    KAE        c = z0, the latent state.  Reaching tau_i is ONE matrix product
               e^{K tau_i} z0, so the cost of an iteration does not grow with the horizon.
    U-Net/FNO  c = the conditioning window ending at t0, in that model's normalised space.
               Reaching tau_i needs tau_i autoregressive steps, so cost is linear in the
               horizon and gradients traverse the whole chain.

The analysis is the state AT t0 -- for the KAE the decode of z0, for the autoregressive
models the last frame of the control window -- so all methods are scored on the same
quantity even though they optimise different variables.
"""

from __future__ import annotations

import time
from typing import Dict

import torch

from data_assimilation.tra.bridge import rel_l2
from data_assimilation.tra.init_utils import informed_latent, informed_window
from data_assimilation.tra.protocol import Problem


def _to(x, dev):
    return torch.as_tensor(x, device=dev)


def solve_kae(
    ad,
    data,
    prob: Problem,
    *,
    iters: int,
    lr: float,
    seed: int,
    track_every: int = 0,
    init: str = "climatology",
) -> Dict:
    """4D-Var on the KAE latent.

    ``init`` MUST be uninformed by default.  Starting from the encoded true window leaks
    the answer: the analysis error at iteration 0 is then the autoencoder floor, and the
    optimiser can only move away from it, so the run measures nothing.  ``'climatology'``
    starts from z = 0, whose decode is the model's mean field.  ``'encode'`` is retained
    only for diagnostics and is never the default.
    """
    dev = ad.dev
    sim, t0 = _to(prob.sim, dev), _to(prob.t0, dev)
    par = data.params_for(sim)
    mask = _to(prob.mask, dev)
    y = _to(prob.y, dev)
    truth_t0 = data.frames(sim, t0)
    om = data.mask_for(sim)
    # e^{K tau} is defined for any real tau, so an off-grid observation is evaluated
    # where it actually is; every other method must snap to prob.offsets
    taus = prob.offsets_eval * ad.dt_train

    torch.manual_seed(seed)
    if init == "encode":  # diagnostic only: leaks truth
        w = data.window(sim, t0 - (ad.n_control_frames - 1), ad.n_control_frames)
        z = ad.encode(w, par).detach().clone()
    elif init == "obs":  # informed: the nearest observation, pulled back
        z = informed_latent(ad, data, prob, dev).detach().clone()
    elif init == "random":
        z = 0.1 * torch.randn(len(prob.sim), ad.latent_dim, device=dev)
    elif init == "climatology":
        z = torch.zeros(len(prob.sim), ad.latent_dim, device=dev)
    else:
        raise ValueError(f"unknown init {init!r}")
    z.requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    K = ad.generator(par).detach()
    phis = [torch.matrix_exp(K * float(t)) for t in taus]  # precomputed once
    hist = {"iter": [], "loss": [], "rel_t0": []}
    t_start = time.perf_counter()
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for j, phi in enumerate(phis):
            zt = (
                torch.bmm(phi, z.unsqueeze(-1)).squeeze(-1)
                if phi.dim() == 3
                else z @ phi.T
            )
            pred = ad.decode(zt)
            loss = loss + ((pred - y[j]) * mask[j]).pow(2).mean()
        loss = loss / len(phis)
        loss.backward()
        opt.step()
        if track_every and (it % track_every == 0 or it == iters - 1):
            with torch.no_grad():
                hist["iter"].append(it)
                hist["loss"].append(float(loss))
                hist["rel_t0"].append(float(rel_l2(ad.decode(z), truth_t0, om).mean()))
    with torch.no_grad():
        analysis = ad.decode(z)
        rel = rel_l2(analysis, truth_t0, om)
    return {
        "rel": rel.cpu().numpy(),
        "analysis": analysis.detach().cpu().numpy(),
        "hist": hist,
        "wall_s": time.perf_counter() - t_start,
        "control": z.detach(),
        "iters": iters,
        "lr": lr,
    }


def solve_autoregressive(
    ad,
    data,
    prob: Problem,
    *,
    iters: int,
    lr: float,
    seed: int,
    track_every: int = 0,
    init: str = "climatology",
) -> Dict:
    """4D-Var on the conditioning window of a U-Net / FNO surrogate.

    As for the KAE, the default start is uninformed.  In normalised units zero IS the
    dataset mean, so ``'climatology'`` is the natural uninformed control.  ``'truth'``
    is retained for diagnostics only and leaks the answer.
    """
    dev = ad.dev
    sim, t0 = _to(prob.sim, dev), _to(prob.t0, dev)
    par = data.params_for(sim)
    mask, y = _to(prob.mask, dev), _to(prob.y, dev)
    truth_t0 = data.frames(sim, t0)
    om = data.mask_for(sim)
    k = ad.n_control_frames
    n_steps = int(prob.offsets.max())
    idx = torch.as_tensor(prob.offsets + (k - 1), device=dev)  # rows of the rollout

    torch.manual_seed(seed)
    w = data.window(sim, t0 - (k - 1), k)
    if init == "truth":  # diagnostic only: leaks truth
        c = ad.to_model(w).detach().clone()
    elif init == "obs":  # informed: the nearest observation across the window
        c = ad.to_model(informed_window(ad, prob, dev)).detach().clone()
    elif init == "random":
        c = 0.1 * torch.randn_like(ad.to_model(w))
    elif init == "climatology":
        c = torch.zeros_like(ad.to_model(w))
    else:
        raise ValueError(f"unknown init {init!r}")
    c.requires_grad_(True)
    opt = torch.optim.Adam([c], lr=lr)
    hist = {"iter": [], "loss": [], "rel_t0": []}
    t_start = time.perf_counter()
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        traj = ad.rollout(c, n_steps, par)  # physical [B, k+n, C,H,W]
        pred = traj[:, idx].transpose(0, 1)  # [N, B, C, H, W]
        loss = ((pred - y) * mask).pow(2).mean()
        loss.backward()
        opt.step()
        if track_every and (it % track_every == 0 or it == iters - 1):
            with torch.no_grad():
                a = ad.to_physical(c)[:, -1]
                hist["iter"].append(it)
                hist["loss"].append(float(loss))
                hist["rel_t0"].append(float(rel_l2(a, truth_t0, om).mean()))
    with torch.no_grad():
        analysis = ad.to_physical(c)[:, -1]  # the frame AT t0
        rel = rel_l2(analysis, truth_t0, om)
    return {
        "rel": rel.cpu().numpy(),
        "analysis": analysis.detach().cpu().numpy(),
        "hist": hist,
        "wall_s": time.perf_counter() - t_start,
        "control": c.detach(),
        "iters": iters,
        "lr": lr,
    }
