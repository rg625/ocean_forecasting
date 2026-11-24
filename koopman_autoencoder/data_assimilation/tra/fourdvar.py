# mypy: disable-error-code="var-annotated"
"""4D-Var over a frozen surrogate, in whatever control variable that surrogate exposes.

    J(c) = (1/N) sum_i || M_i . (G(c, tau_i) - y_i) ||^2                       (objective "base")

For the KAE the default objective (since the ICLR 2027 revision) is the full Bayesian one,

    J(z) = sum_i || M_i . (D(e^{K tau_i} z) - y_i) ||^2 / (2 s2) + 1/2 (z - mu)^T B^{-1} (z - mu),
    s2   = sigma_obs^2 + sigma_m^2                                                (objective "bg")

with (mu, B) the mean/covariance of encoded TRAINING windows and sigma_m the RMS misfit that
base 4D-Var reaches on clean VALIDATION problems (representation error).  It is attached to
the adapter as ``ad.background`` by ``kae_background``; without it solve_kae runs "base".

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
from typing import Dict, Optional

import torch

from data_assimilation.tra.bridge import rel_l2
from data_assimilation.tra.init_utils import informed_latent, informed_window
from data_assimilation.tra.protocol import Problem


def _to(x, dev):
    return torch.as_tensor(x, device=dev)


@torch.no_grad()
def latent_background(ad, train_data, n: int = 4096, seed: int = 0, floor: float = 1e-3):
    """(mu, B^{-1}) of the KAE latent over encoded training windows.

    The encoder output is layer-normalised and spans a low-dimensional subspace, so the raw
    covariance is numerically singular; its eigenvalues are floored at ``floor`` x the
    largest before inversion (the same rule as the KS revision experiments).
    """
    import numpy as np

    dev = ad.dev
    g = np.random.default_rng(seed)
    k = ad.n_control_frames
    sim = torch.as_tensor(g.integers(0, train_data.n_sim, n), device=dev)
    t = torch.as_tensor(g.integers(k - 1, train_data.n_t, n), device=dev)
    zs = []
    for i in range(0, n, 128):
        s, tt = sim[i : i + 128], t[i : i + 128]
        w = train_data.window(s, tt - (k - 1), k)
        zs.append(ad.encode(w, train_data.params_for(s)).float())
    z = torch.cat(zs)
    mu = z.mean(0)
    zc = z - mu
    cov = (zc.T @ zc / (n - 1)).double()
    ev, V = torch.linalg.eigh(cov)
    ev = ev.clamp_min(floor * float(ev.max()))
    Binv = ((V / ev) @ V.T).float()
    return mu, Binv


def representation_error(ad, val_prob: Problem, val_data, *, iters: int, lr: float, init: str):
    """sigma_m: RMS observation misfit base 4D-Var reaches on clean, fully observed
    validation problems with the same schedule (never the test regime)."""
    r = solve_kae(ad, val_data, val_prob, iters=iters, lr=lr, seed=7, init=init, objective="base")
    return float(r["misfit_rms"].pow(2).mean().sqrt())


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
    objective: Optional[str] = None,
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
    elif init == "obs_reg":  # informed, regularised pull-back (no amplification of damped modes)
        from data_assimilation.tra.init_utils import informed_latent_reg

        z = informed_latent_reg(ad, data, prob, dev).detach().clone()
    elif init == "random":
        z = 0.1 * torch.randn(len(prob.sim), ad.latent_dim, device=dev)
    elif init == "climatology":
        z = torch.zeros(len(prob.sim), ad.latent_dim, device=dev)
    else:
        raise ValueError(f"unknown init {init!r}")
    z.requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    K = ad.generator(par)
    if hasattr(K, "apply"):  # structured operator (CT-KAE detail latent), never dense
        steps = [lambda z, t=float(t): K.apply(z, t) for t in taus]
    else:
        K = K.detach()
        phis = [torch.matrix_exp(K * float(t)) for t in taus]  # precomputed once
        steps = [
            (
                (lambda z, p=p: torch.bmm(p, z.unsqueeze(-1)).squeeze(-1))
                if p.dim() == 3
                else (lambda z, p=p: z @ p.T)
            )
            for p in phis
        ]
    bg = getattr(ad, "background", None)
    if objective is None:
        objective = "bg" if bg is not None else "base"
    if objective == "bg":
        assert bg is not None, "objective 'bg' needs ad.background (see kae_background)"
        mu, Binv = bg["mu"].to(dev), bg["Binv"].to(dev)
        s2 = float(prob.noise_std) ** 2 + float(bg["sigma_m"]) ** 2
    # observed entries per problem (masks broadcast over channels)
    n_ent = float(sum((mask[j] * torch.ones_like(y[j])).sum() for j in range(len(steps)))) / len(prob.sim)

    def _sse(zz):
        out = 0.0
        for j, step in enumerate(steps):
            out = out + ((ad.decode(step(zz)) - y[j]) * mask[j]).pow(2).flatten(1).sum(1)
        return out  # [B]

    hist = {"iter": [], "loss": [], "rel_t0": []}
    t_start = time.perf_counter()
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        if objective == "base":
            loss = 0.0
            for j, step in enumerate(steps):
                pred = ad.decode(step(z))
                loss = loss + ((pred - y[j]) * mask[j]).pow(2).mean()
            loss = loss / len(steps)
        elif objective == "bg":
            dz = z - mu
            prior = 0.5 * ((dz @ Binv) * dz).sum(1)
            loss = (_sse(z) / (2 * s2) + prior).sum() / (len(prob.sim) * n_ent)
        else:
            raise ValueError(f"unknown objective {objective!r}")
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
        misfit_rms = (_sse(z) / n_ent).sqrt()
    return {
        "rel": rel.cpu().numpy(),
        "misfit_rms": misfit_rms,
        "objective": objective,
        "analysis": analysis.detach().cpu().numpy(),
        "hist": hist,
        "wall_s": time.perf_counter() - t_start,
        "control": z.detach(),
        "iters": iters,
        "lr": lr,
    }


def solve_kae_data(
    ad,
    data,
    prob: Problem,
    *,
    iters: int,
    lr: float,
    seed: int,
    track_every: int = 0,
    init: str = "climatology",
    control_frames: Optional[int] = None,
    score: str = "control",
) -> Dict:
    """Ablation: 4D-Var on the KAE's conditioning window, in data space.

    Everything but the forward map is ``solve_autoregressive``: the control is the
    normalised window ending at t0, started by the same rule, scored on its last frame
    and fitted with the same objective.  The map is the KAE's own,
    decode(e^{K tau} encode(window)), so against ``solve_kae`` it isolates the control
    variable (physical window vs latent z0) with the model held fixed.

    ``control_frames`` sets how many frames ending at t0 are optimised; pass the U-Net's
    own window length so both controls have the same dimension.  When it is shorter than
    the encoder's window, the earliest control frame is repeated to fill it -- the same
    construction ``informed_window`` already uses for the obs first guess.  ``None``
    optimises the encoder's full window.

    ``score='reconstruction'`` reports decode(encode(control)) as the analysis instead of
    the raw control frame; the optimisation is unchanged.
    """
    assert score in ("control", "reconstruction"), score
    dev = ad.dev
    sim, t0 = _to(prob.sim, dev), _to(prob.t0, dev)
    par = data.params_for(sim)
    mask, y = _to(prob.mask, dev), _to(prob.y, dev)
    truth_t0 = data.frames(sim, t0)
    om = data.mask_for(sim)
    k = ad.n_control_frames
    kc = control_frames or k
    assert 1 <= kc <= k, f"control_frames={kc} must be in [1, {k}]"
    # the KAE's map is defined at any real tau, as in solve_kae; on the integer
    # schedules of the sweeps this equals the U-Net's prob.offsets exactly
    taus = prob.offsets_eval * ad.dt_train

    torch.manual_seed(seed)
    w = data.window(sim, t0 - (kc - 1), kc)
    if init == "truth":  # diagnostic only: leaks truth
        c = ad.to_model(w).detach().clone()
    elif init == "obs":  # informed: the nearest observation across the window
        c = ad.to_model(informed_window(ad, prob, dev)[:, -kc:]).detach().clone()
    elif init == "pullback":  # the KAE's own warm start: decode(e^{-K tau_1} enc(y_1))
        with torch.no_grad():
            x = ad.decode(informed_latent(ad, data, prob, dev))  # [B, C, H, W]
        c = ad.to_model(x.unsqueeze(1).expand(-1, kc, *x.shape[1:])).contiguous()
    elif init == "random":
        c = 0.1 * torch.randn_like(ad.to_model(w))
    elif init == "climatology":
        c = torch.zeros_like(ad.to_model(w))
    else:
        raise ValueError(f"unknown init {init!r}")
    c.requires_grad_(True)
    opt = torch.optim.Adam([c], lr=lr)
    K = ad.generator(par).detach()
    phis = [torch.matrix_exp(K * float(t)) for t in taus]  # precomputed once

    def encode(c):
        win = torch.cat([c[:, :1].expand(-1, k - kc, *c.shape[2:]), c], dim=1)
        return ad.encode(ad.to_physical(win), par)

    def analysis_of(c):
        if score == "reconstruction":
            return ad.decode(encode(c))
        return ad.to_physical(c)[:, -1]  # the frame AT t0

    def predict(c):
        z0 = encode(c)
        return torch.stack(
            [
                ad.decode(
                    torch.bmm(phi, z0.unsqueeze(-1)).squeeze(-1)
                    if phi.dim() == 3
                    else z0 @ phi.T
                )
                for phi in phis
            ]
        )  # [N, B, C, H, W]

    hist = {"iter": [], "loss": [], "rel_t0": []}
    t_start = time.perf_counter()
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        pred = predict(c)
        loss = ((pred - y) * mask).pow(2).mean()
        loss.backward()
        opt.step()
        if track_every and (it % track_every == 0 or it == iters - 1):
            with torch.no_grad():
                a = analysis_of(c)
                hist["iter"].append(it)
                hist["loss"].append(float(loss))
                hist["rel_t0"].append(float(rel_l2(a, truth_t0, om).mean()))
    with torch.no_grad():
        analysis = analysis_of(c)
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
    kae=None,
    bgx: Optional[Dict] = None,
) -> Dict:
    """4D-Var on the conditioning window of a U-Net / FNO surrogate.

    ICLR 2027 revision (R20): ``init='pullback'`` starts from the KAE's closed-form first guess
    decode(e^{-K tau_1} enc(y_1)) (needs ``kae``), repeated over the window; ``bgx`` = {"mu", "var",
    "s2"} adds a diagonal Gaussian background on the control (pixel-wise climatological mean and
    variance of training frames, model units) with R = s2 I, weighted as the KAE's "bg" objective:
    J = sum_obs (pred - y)^2 / (2 s2) + 1/2 sum (c - mu)^2 / var, divided by (B * observed entries).

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
    elif init == "pullback":  # the KAE's closed-form first guess, repeated over the window
        assert kae is not None, "init='pullback' needs the KAE adapter"
        with torch.no_grad():
            x = kae.decode(informed_latent(kae, data, prob, dev))  # [B, C, H, W]
        c = ad.to_model(x.unsqueeze(1).expand(-1, k, *x.shape[1:])).contiguous()
    elif init == "random":
        c = 0.1 * torch.randn_like(ad.to_model(w))
    elif init == "climatology":
        c = torch.zeros_like(ad.to_model(w))
    else:
        raise ValueError(f"unknown init {init!r}")
    c.requires_grad_(True)
    opt = torch.optim.Adam([c], lr=lr)
    hist = {"iter": [], "loss": [], "rel_t0": []}
    if bgx is not None:
        bmu, bvar, bs2 = bgx["mu"].to(dev), bgx["var"].to(dev), float(bgx["s2"])
        n_ent = float(mask.expand_as(y)[:, 0].sum()) / len(prob.sim) * y.shape[0]
    t_start = time.perf_counter()
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        traj = ad.rollout(c, n_steps, par)  # physical [B, k+n, C,H,W]
        pred = traj[:, idx].transpose(0, 1)  # [N, B, C, H, W]
        if bgx is None:
            loss = ((pred - y) * mask).pow(2).mean()
        else:
            sse = ((pred - y) * mask).pow(2).sum()
            prior = 0.5 * ((c - bmu).pow(2) / bvar).sum()
            loss = (sse / (2 * bs2) + prior) / (len(prob.sim) * n_ent)
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


def solve_unet_latent(
    hy,
    data,
    prob: Problem,
    *,
    iters: int,
    lr: float,
    seed: int,
    track_every: int = 0,
    init: str = "obs",
) -> Dict:
    """Ablation (ICLR 2027 revision, R14): U-Net dynamics, KAE latent control.

    The control is one KAE latent per U-Net conditioning frame, z = [z_{-k+1}, ..., z_0];
    the U-Net's window is their decodes through the frozen KAE decoder, x_j = D(z_j), and
    the U-Net rolls it forward to the observation times.  Against ``solve_kae`` (same latent
    control, KAE dynamics) it isolates the DYNAMICS; against ``solve_autoregressive`` (same
    U-Net, physical window) it isolates the CONTROL VARIABLE.  Objective: the unweighted
    misfit of ``solve_autoregressive``.  Warm start: the KAE's own, z_0 = e^{-K tau_1} E(y_1),
    and z_{-j} = e^{-K j dt} z_0 (the KAE dynamics are used for the first guess only).
    The analysis is D(z_0).
    """
    un, kae = hy.unet, hy.kae
    dev = un.dev
    sim, t0 = _to(prob.sim, dev), _to(prob.t0, dev)
    par = data.params_for(sim)
    mask, y = _to(prob.mask, dev), _to(prob.y, dev)
    truth_t0 = data.frames(sim, t0)
    om = data.mask_for(sim)
    k = un.n_control_frames
    n_steps = int(prob.offsets.max())
    idx = torch.as_tensor(prob.offsets + (k - 1), device=dev)
    torch.manual_seed(seed)
    with torch.no_grad():
        if init == "obs":
            z0 = informed_latent(kae, data, prob, dev)
        elif init == "climatology":
            z0 = torch.zeros(len(prob.sim), kae.latent_dim, device=dev)
        else:
            raise ValueError(f"unknown init {init!r}")
        K = kae.generator(par).detach()
        zs = []
        for j in range(k - 1, -1, -1):  # oldest frame first
            back = torch.matrix_exp(-K * (j * kae.dt_train))
            zs.append(torch.bmm(back, z0.unsqueeze(-1)).squeeze(-1) if back.dim() == 3 else z0 @ back.T)
        z = torch.stack(zs, dim=1).contiguous()  # [B, k, D]
    z.requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)

    def window(zz):
        return torch.stack([kae.decode(zz[:, j]) for j in range(k)], dim=1)  # physical [B,k,C,H,W]

    hist = {"iter": [], "loss": [], "rel_t0": []}
    t_start = time.perf_counter()
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        traj = un.rollout(un.to_model(window(z)), n_steps, par)
        pred = traj[:, idx].transpose(0, 1)
        loss = ((pred - y) * mask).pow(2).mean()
        loss.backward()
        opt.step()
        if track_every and (it % track_every == 0 or it == iters - 1):
            with torch.no_grad():
                hist["iter"].append(it)
                hist["loss"].append(float(loss))
                hist["rel_t0"].append(float(rel_l2(kae.decode(z[:, -1]), truth_t0, om).mean()))
    with torch.no_grad():
        analysis = kae.decode(z[:, -1])
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
