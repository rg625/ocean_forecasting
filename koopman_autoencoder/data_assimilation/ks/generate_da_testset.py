# mypy: disable-error-code="arg-type"
"""Generate a fresh, fully independent KS test set for the data-assimilation campaign.

``--t-max`` controls the record length.  The default 99.9 reproduces the 1000-frame set;
the long-horizon experiments use a much longer record because the 1000-frame records cap
the reachable recovery horizon at about 3.9 Lyapunov times.

The existing splits (train/val/test.nc) come from one 100-trajectory batch split 80/10/10.
`val.nc` was used by the original DA study; `test.nc` has only 10 simulations, which is a
thin basis for trajectory-level statistics.  This script draws N *new* trajectories from the
identical simulator configuration (verified to reproduce the training distribution) under a
DA-specific seed, so every assimilation problem sits on its own independent trajectory.

Simulator settings are those of ``simulate.py`` used to build ``data/ks/*.nc``:
    L = 22.0, n_x = 64, dt = 0.1, t_max = 99.9 (1000 frames), forcing = 'none'.

Usage:
    python data_assimilation/ks/generate_da_testset.py --n 64 --seed 20270101 --workers 16 \
        --out data/ks/da_test.nc
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

L = 22.0
N_X = 64
DT = 0.1
T_MAX = 99.9  # overridden from the command line for the long-horizon set


def _one(args):
    """Simulate a single trajectory under its own seed -> (T, n_x) float32."""
    idx, seed, t_max = args
    np.random.seed(seed)
    from simulations import KSSimulation  # imported inside the worker

    sim = KSSimulation(N_traj=1, forcing="none", L=L, n_x=N_X, t_max=t_max, dt=DT)
    return idx, sim.x.numpy()[0], sim.t.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20270101)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("data/ks/da_test.nc"))
    ap.add_argument(
        "--t-max",
        type=float,
        default=T_MAX,
        help="record length in time units; 99.9 gives the 1000-frame default",
    )
    args = ap.parse_args()

    # one distinct, recorded seed per trajectory
    seeds = (args.seed + np.arange(args.n)).tolist()
    fields = [None] * args.n
    t_coord = None

    t_start = time.time()
    jobs = [(i, sd, args.t_max) for i, sd in enumerate(seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, (idx, traj, t) in enumerate(ex.map(_one, jobs)):
            fields[idx] = traj
            t_coord = t
            if (k + 1) % 4 == 0 or k == args.n - 1:
                print(
                    f"  [{k + 1}/{args.n}] done ({time.time() - t_start:.0f}s)",
                    flush=True,
                )

    u = np.stack(fields, axis=0)[:, :, :, None].astype("float32")  # [sim, t, x, y]
    ds = xr.Dataset(
        data_vars={"u": (["sim", "t", "x", "y"], u)},
        coords={
            "sim": np.arange(args.n),
            "t": t_coord,
            "x": np.linspace(-1, 1, N_X),
            "y": np.array([0.0]),
        },
        attrs={
            "description": "Independent KS trajectories held out for the DA campaign.",
            "L": L,
            "n_x": N_X,
            "dt": DT,
            "t_max": args.t_max,
            "forcing": "none",
            "base_seed": args.seed,
            "seeds": str(seeds),
            "generator": "simulations.KSSimulation (scipy RK45, rtol=1e-6, atol=1e-8)",
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.out)
    print(f"Saved {u.shape} -> {args.out}  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
