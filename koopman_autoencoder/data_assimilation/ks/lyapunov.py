# mypy: disable-error-code="assignment"
"""Measure the largest Lyapunov exponent of the KS system used in this study.

Twin experiment on the SAME simulator that produced the data (simulations.KSSimulation,
L=22, 64 grid points, dt=0.1): perturb a state on the attractor, integrate the perturbed
and unperturbed trajectories, and fit the exponential growth of the separation, with
periodic renormalisation so the perturbation stays in the tangent regime.

The Lyapunov time is 1/lambda_1, and all assimilation horizons in the paper are reported
in those units as well as in raw time.

    python -m data_assimilation.ks.lyapunov --n-traj 12
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[3]
_sys.path.insert(0, str(_ROOT))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

logger = logging.getLogger("lyap")

L, N_X, DT = 22.0, 64, 0.1


def rhs_factory():
    from simulations import KSSimulation

    sim = KSSimulation(N_traj=1, forcing="none", L=L, n_x=N_X, t_max=DT, dt=DT)
    return sim._rhs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=12)
    ap.add_argument("--t-total", type=float, default=200.0)
    ap.add_argument("--renorm-every", type=float, default=2.0)
    ap.add_argument("--eps", type=float, default=1e-7)
    ap.add_argument("--spinup", type=float, default=50.0)
    ap.add_argument(
        "--out", type=Path, default=Path("da_results_sda_paper/lyapunov.json")
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    from scipy.integrate import solve_ivp

    rhs = rhs_factory()

    def integrate(u, T):
        s = solve_ivp(
            rhs, (0.0, T), u, method="RK45", rtol=1e-9, atol=1e-11, t_eval=[T]
        )
        return s.y[:, -1]

    rng = np.random.default_rng(20270101)
    lams = []
    for j in range(args.n_traj):
        # start on the attractor
        ic = np.zeros(N_X // 2 + 1, dtype=complex)
        nm = N_X // 8
        ic[1 : nm + 1] = rng.uniform(0, 1, nm) * np.exp(
            1j * rng.uniform(0, 2 * np.pi, nm)
        )
        u = np.fft.irfft(ic, n=N_X)
        u = u / max(np.abs(u).max(), 1e-12)
        u = integrate(u, args.spinup)

        d0 = rng.standard_normal(N_X)
        d0 *= args.eps / np.linalg.norm(d0)
        v = u + d0
        logs, n_steps = [], int(args.t_total / args.renorm_every)
        for _ in range(n_steps):
            u = integrate(u, args.renorm_every)
            v = integrate(v, args.renorm_every)
            d = v - u
            nd = np.linalg.norm(d)
            logs.append(np.log(nd / args.eps))
            v = u + d * (args.eps / nd)  # renormalise into the tangent regime
        lam = float(np.sum(logs) / (n_steps * args.renorm_every))
        lams.append(lam)
        logger.info(
            f"  trajectory {j:2d}: lambda_1 = {lam:.5f}  "
            f"Lyapunov time = {1 / lam:.2f}"
        )
    lams = np.array(lams)
    lam = float(lams.mean())
    sem = float(lams.std(ddof=1) / np.sqrt(lams.size))
    out = {
        "system": {
            "equation": "u_t = -u u_x - u_xx - u_xxxx",
            "L": L,
            "n_x": N_X,
            "dt_frames": DT,
            "forcing": "none",
        },
        "method": (
            "twin experiment with periodic renormalisation; scipy RK45, "
            f"rtol 1e-9, atol 1e-11; perturbation {args.eps}, renormalised "
            f"every {args.renorm_every} time units, {args.t_total} total, "
            f"{args.spinup} spin-up"
        ),
        "n_trajectories": int(lams.size),
        "lambda_1_mean": lam,
        "lambda_1_sem": sem,
        "lambda_1_all": lams.tolist(),
        "lyapunov_time": 1.0 / lam,
        "lyapunov_time_sem": sem / lam**2,
        "lyapunov_time_in_frames": (1.0 / lam) / DT,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    logger.info(
        f"lambda_1 = {lam:.5f} +/- {sem:.5f}  ->  Lyapunov time "
        f"T_L = {1/lam:.2f} time units = {(1/lam)/DT:.1f} frames"
    )
    logger.info(
        f"the study's canonical horizon delta_l = 2.5 is "
        f"{2.5 * lam:.2f} Lyapunov times"
    )
    logger.info(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
