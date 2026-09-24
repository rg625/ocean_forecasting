"""The predictability timescale of the transonic flow, so the temporal axes mean something.

Every temporal quantity in this campaign is quoted in FRAMES -- delta_f = 1, delta_l = 25,
a window of W = 32.  Those are arbitrary units until they are divided by a timescale of the
flow itself, and without one there is no way to say whether a 25-frame horizon is long.

The KS campaign measured this with a twin experiment on its own simulator: perturb a state,
integrate the true dynamics twice, fit the exponential separation.  **That route is closed
here.**  The transonic data came from an external CFD code that is not in this repository,
so the true dynamics cannot be integrated at all.  Running the twin experiment on a
surrogate instead would measure the surrogate's exponent, not the flow's, which is exactly
the quantity that must not be assumed.

So lambda_1 is estimated from the DATA, by Rosenstein's method:

  1. treat each stored state as a point (they are already 32,768-dimensional, so no delay
     embedding is needed);
  2. for each reference point find its nearest neighbour that is far enough away IN TIME
     to be a genuine recurrence rather than the same eddy one frame later (Theiler window);
  3. follow each pair forward and average ln(separation) over pairs;
  4. lambda_1 is the slope of that curve while it is still straight -- before the pairs
     saturate at the attractor diameter.

Two honest limits, both reported in the output:
  * a data-driven exponent is a lower bound when the sampling is coarse, because the
    fastest growth happens between stored frames;
  * with 4-6 trajectories the neighbour pool is small, so the fit interval is chosen by
    an R^2 criterion and the residual is reported rather than hidden.

    python -m data_assimilation.tra.lyapunov_tra --regime long
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from data_assimilation.tra.bridge import PhysicalData

DATA = {
    "test": "data/acdm/128_tra/gt_interp.nc",
    "long": "data/acdm/128_tra/gt_longer.nc",
    "val": "data/acdm/128_tra/gt_extrap.nc",
}


def divergence_curve(u, theiler, horizon, max_ref, rng):
    """Mean ln separation of nearest-neighbour pairs, followed forward.

    ``u`` is [S, T, D] already flattened over channels and space.  Neighbours may come
    from a different trajectory; within a trajectory they must be at least ``theiler``
    frames apart, which is what stops "nearest neighbour" from meaning "the next frame".
    """
    S, T, D = u.shape
    idx = [(s, t) for s in range(S) for t in range(T - horizon)]
    rng.shuffle(idx)
    idx = idx[:max_ref]
    flat = u.reshape(S * T, D)
    nrm = (flat**2).sum(1)
    curves = []
    for s, t in idx:
        i = s * T + t
        d2 = nrm + nrm[i] - 2.0 * (flat @ flat[i])
        d2[i] = np.inf
        # exclude temporal neighbours within the same trajectory
        lo, hi = s * T + max(0, t - theiler), s * T + min(T, t + theiler + 1)
        d2[lo:hi] = np.inf
        # the partner must have `horizon` frames left in its own trajectory
        for ss in range(S):
            d2[ss * T + (T - horizon) : (ss + 1) * T] = np.inf
        j = int(torch.argmin(d2))
        sj, tj = divmod(j, T)
        if not np.isfinite(float(d2[j])) or float(d2[j]) <= 0:
            continue
        a = u[s, t : t + horizon + 1]
        b = u[sj, tj : tj + horizon + 1]
        sep = torch.linalg.norm(a - b, dim=1)
        if float(sep[0]) <= 0:
            continue
        curves.append(torch.log(sep).cpu().numpy())
    return np.mean(np.stack(curves), axis=0), len(curves)


def best_linear_fit(y, dt, min_len=4):
    """The straightest leading window of the divergence curve, and its slope."""
    x = np.arange(len(y)) * dt
    best = None
    for a in range(0, max(1, len(y) // 2)):
        for bnd in range(a + min_len, len(y) + 1):
            xx, yy = x[a:bnd], y[a:bnd]
            A = np.vstack([xx, np.ones_like(xx)]).T
            (m, c), res, *_ = np.linalg.lstsq(A, yy, rcond=None)
            ss = float(((yy - yy.mean()) ** 2).sum())
            r2 = 1.0 - (float(res[0]) / ss if res.size and ss > 0 else 0.0)
            if m <= 0:
                continue
            score = r2 * np.sqrt(bnd - a)  # prefer straight AND long
            if best is None or score > best[0]:
                best = (score, float(m), float(c), a, bnd, r2)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--regime", default="long", choices=list(DATA))
    ap.add_argument("--out-dir", type=Path, default=Path("da_results_tra"))
    ap.add_argument("--theiler", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--max-ref", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()

    d = PhysicalData(DATA[a.regime], "tra", "cpu")
    S, T = d.n_sim, d.n_t
    horizon = min(a.horizon, T // 3)
    u = d.u.reshape(S, T, -1).double()
    # per-channel scaling would change the metric; the states are already in physical
    # units and the norm is taken over the whole field, as rel_l2 does
    rng = np.random.default_rng(0)
    y, n_pairs = divergence_curve(u, a.theiler, horizon, a.max_ref, rng)
    fit = best_linear_fit(y, dt=1.0)
    if fit is None:
        print("no positive-slope region found; the sampling may be too coarse")
        return 1
    _, lam_per_frame, _, i0, i1, r2 = fit

    # bootstrap the slope over the fitted window by resampling the curve's own residuals
    x = np.arange(i0, i1, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    (m0, c0), *_ = np.linalg.lstsq(A, y[i0:i1], rcond=None)
    resid = y[i0:i1] - (m0 * x + c0)
    boots = []
    for _ in range(a.n_boot):
        yy = m0 * x + c0 + rng.choice(resid, size=len(x), replace=True)
        (mb, _cb), *_ = np.linalg.lstsq(A, yy, rcond=None)
        boots.append(float(mb))
    lo, hi = np.percentile(boots, [2.5, 97.5])

    T_L = 1.0 / lam_per_frame
    out = {
        "regime": DATA[a.regime],
        "method": "Rosenstein on the stored trajectories",
        "why_not_twin": (
            "the transonic solver is not in this repository, so the true "
            "dynamics cannot be integrated; a twin experiment on a "
            "surrogate would measure the surrogate, not the flow"
        ),
        "lambda_1_per_frame": lam_per_frame,
        "lambda_1_ci95": [float(lo), float(hi)],
        "lyapunov_time_frames": T_L,
        "lyapunov_time_ci95_frames": [1.0 / float(hi), 1.0 / float(lo)],
        "fit_window_frames": [int(i0), int(i1)],
        "fit_r2": float(r2),
        "n_pairs": int(n_pairs),
        "theiler_window": a.theiler,
        "n_sim": int(S),
        "n_frames": int(T),
        "caveat": (
            "a data-driven exponent is a LOWER bound at coarse sampling: the "
            "fastest separation happens between stored frames"
        ),
        "campaign_horizons_in_lyapunov_times": {
            "delta_f = 1": 1.0 / T_L,
            "delta_f = 4": 4.0 / T_L,
            "delta_l = 25 (canonical)": 25.0 / T_L,
            "delta_l = 50": 50.0 / T_L,
            "window W = 32": 32.0 / T_L,
            "post-DA forecast 50 frames": 50.0 / T_L,
        },
    }
    (a.out_dir).mkdir(parents=True, exist_ok=True)
    (a.out_dir / "lyapunov.json").write_text(json.dumps(out, indent=2))
    print(
        f"regime {a.regime}: {S} trajectories x {T} frames, {n_pairs} neighbour pairs"
    )
    print(
        f"  lambda_1 = {lam_per_frame:.4f} / frame  "
        f"[{lo:.4f}, {hi:.4f}]   (fit frames {i0}-{i1}, R^2 {r2:.3f})"
    )
    print(f"  Lyapunov time T_L = {T_L:.1f} frames")
    print("\n  the campaign's horizons, in Lyapunov times:")
    for k, v in out["campaign_horizons_in_lyapunov_times"].items():
        print(f"    {k:28s} {v:6.2f} T_L")
    print(f"\nsaved -> {a.out_dir / 'lyapunov.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
