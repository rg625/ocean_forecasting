"""How well do the observations constrain the analysis, per model? (diagnostic; leaks truth)

On the df = 9 default problem (full-field, noise-free observations at t0 + [9, 12, 15, 19, 25]
frames, test regime; the problem of da_repr_check.py), linearise each model around the
encoded true state y* and compare two Jacobians, both in physical units:

    J_obs   = d [decode(e^{K tau_j} y)]_j / dy      (what 4D-Var fits)
    J_state = d [decode(y)] / dy, fluid points only  (what the analysis is scored on)

Reported, averaged over problems:
  gain_*        generalised singular values of (J_obs, J_state): observation change per
                unit of t0-state change, along each latent direction. Invariant to any
                invertible reparametrisation of the latent, so models with different latent
                coordinates are comparable. Small gains are t0-state directions the
                observations barely see; noise or model error in them goes uncorrected.
  cond_latent   condition number of J_obs^T J_obs in the model's own latent coordinates,
                which is what the 4D-Var optimiser (Adam on y) has to cope with.
  decay         |e^{K tau} y*| / |y*| at the first and last observation lead, and the share
                of generator modes whose e-folding time is shorter than the first lead.
  J_true/J_rec  observation misfit at y* and at the 4D-Var analysis (climatology start,
                tuned lr): J_rec < J_true means the cost prefers a wrong state to the
                model's best representation of the truth.

Jacobians are central finite differences (step 1e-2 of |y*|/sqrt(D)) with TF32 disabled
(TF32 convolutions limit the differences to about 1e-3 relative accuracy); the result is
checked against an autograd Jacobian-vector product (relative error about 2e-4).

    python -m data_assimilation.tra.da_conditioning --kae-kind legacy|ctkae --kae-run RUN \
        [--kae-ckpt FILE] --lr LR --json out.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_assimilation.tra.bridge import REGIMES, PhysicalData
from data_assimilation.tra.fourdvar import solve_kae
from data_assimilation.tra.protocol import build_problem

ap = argparse.ArgumentParser()
ap.add_argument("--kae-kind", choices=["legacy", "ctkae"], required=True)
ap.add_argument("--kae-run", required=True)
ap.add_argument("--kae-ckpt", default=None, help="as in exp_tra.py")
ap.add_argument("--lr", type=float, required=True)
ap.add_argument("--iters", type=int, default=500)
ap.add_argument("--n-problems", type=int, default=8)
ap.add_argument("--chunk", type=int, default=32)
ap.add_argument("--json", default=None)
a = ap.parse_args()
dev = torch.device("cuda")
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
if a.kae_kind == "ctkae":
    from data_assimilation.tra.ctkae_adapter import CTKAEAdapter

    ad = CTKAEAdapter(a.kae_run, REGIMES["tra"], dev, ckpt=a.kae_ckpt or "best.pth")
else:
    from data_assimilation.tra.kae_adapter import KAEAdapter

    ck = Path(a.kae_run) / "checkpoints" / a.kae_ckpt if a.kae_ckpt else None
    ad = KAEAdapter(a.kae_run, REGIMES["tra"], dev, ckpt=ck)
data = PhysicalData("data/acdm/128_tra/gt_interp.nc", "tra", dev)
offsets = np.array([9, 12, 15, 19, 25])
prob = build_problem(
    data, name="cond_df9", n_problems=a.n_problems, offsets=offsets, seed=200
)
sim = torch.as_tensor(prob.sim, device=dev)
t0 = torch.as_tensor(prob.t0, device=dev)
par = data.params_for(sim)
om = data.mask_for(sim)
y_obs = torch.as_tensor(prob.y, device=dev)
taus = prob.offsets_eval * ad.dt_train
k = ad.n_control_frames

with torch.no_grad():
    ystar = ad.encode(data.window(sim, t0 - (k - 1), k), par)
rec = solve_kae(ad, data, prob, iters=a.iters, lr=a.lr, seed=1, init="climatology")
y_rec = rec["control"]


def phis_for(b: int) -> tuple:
    K = ad.generator(par[b : b + 1]).detach()  # also pins the CT-KAE decode condition
    K = K[0] if K.dim() == 3 else K
    return K, [torch.matrix_exp(K * float(t)) for t in taus]


def obs_map(Y: torch.Tensor, phis: list) -> torch.Tensor:
    return torch.cat([ad.decode(Y @ p.T).flatten(1) for p in phis], 1)


def state_map(Y: torch.Tensor, fluid: torch.Tensor) -> torch.Tensor:
    return (ad.decode(Y) * fluid).flatten(1)


def jac(f, y: torch.Tensor, h: float) -> torch.Tensor:
    """Central-difference Jacobian [n_out, D] of f at y [D]."""
    D = y.numel()
    cols = []
    for i in range(0, D, a.chunk):
        E = torch.eye(D, device=dev)[i : i + a.chunk] * h
        with torch.no_grad():
            cols.append(((f(y + E) - f(y - E)) / (2 * h)).double().T)
    return torch.cat(cols, 1)


def misfit(y: torch.Tensor, phis: list, b: int) -> float:
    with torch.no_grad():
        pred = obs_map(y[None], phis).view(len(phis), -1)
        return float((pred - y_obs[:, b].flatten(1)).pow(2).mean())


rows = []
for b in range(len(prob.sim)):
    K, phis = phis_for(b)
    fluid = om[b] if om.dim() > 2 else om
    y0 = ystar[b]
    h = 1e-2 * float(y0.norm()) / y0.numel() ** 0.5
    Jo = jac(lambda Y: obs_map(Y, phis), y0, h)
    Js = jac(lambda Y: state_map(Y, fluid), y0, h)
    if b == 0:  # finite differences vs an autograd Jacobian-vector product
        v = torch.randn_like(y0)
        _, jv = torch.autograd.functional.jvp(
            lambda y: obs_map(y[None], phis)[0], (y0,), (v,)
        )
        fd_check = float((Jo @ v.double() - jv.double()).norm() / jv.norm())
    Go, Gs = Jo.T @ Jo, Js.T @ Js
    ridge = 1e-10 * torch.trace(Gs) / Gs.shape[0]
    L = torch.linalg.cholesky(
        Gs + ridge * torch.eye(Gs.shape[0], device=dev, dtype=Gs.dtype)
    )
    Li = torch.linalg.inv(L)
    gain = torch.linalg.eigvalsh(Li @ Go @ Li.T).clamp_min(0).sqrt().cpu().numpy()
    ev = torch.linalg.eigvalsh(Go).clamp_min(0).cpu().numpy()
    lam = torch.linalg.eigvals(K.double()).real.cpu().numpy()
    with torch.no_grad():
        d_first = float((y0 @ phis[0].T).norm() / y0.norm())
        d_last = float((y0 @ phis[-1].T).norm() / y0.norm())
    rows.append(
        {
            "gain_min": float(gain.min()),
            "gain_p10": float(np.percentile(gain, 10)),
            "gain_median": float(np.median(gain)),
            "gain_max": float(gain.max()),
            "share_gain_below_0.1_median": float(np.mean(gain < 0.1 * np.median(gain))),
            "cond_latent": float(ev.max() / max(ev.min(), 1e-300)),
            "decay_first_obs": d_first,
            "decay_last_obs": d_last,
            "share_modes_efold_lt_first_lead": float(np.mean(-lam * taus[0] > 1.0)),
            "share_modes_growing": float(np.mean(lam > 1e-6)),
            "J_true": misfit(y0, phis, b),
            "J_rec": misfit(y_rec[b], phis, b),
        }
    )
    print(b, {kk: round(v, 4) for kk, v in rows[-1].items()})

res = {
    "model": repr(ad),
    "lr": a.lr,
    "fd_vs_autograd_rel_err": fd_check,
    "analysis_rel": float(np.mean(rec["rel"])),
    "mean": {kk: float(np.mean([r[kk] for r in rows])) for kk in rows[0]},
    "share_problems_J_rec_below_J_true": float(
        np.mean([r["J_rec"] < r["J_true"] for r in rows])
    ),
    "per_problem": rows,
}
print(json.dumps({kk: v for kk, v in res.items() if kk != "per_problem"}, indent=1))
if a.json:
    Path(a.json).write_text(json.dumps(res, indent=1))
