# mypy: disable-error-code="assignment, call-overload"
"""Per-iteration 4D-Var cost vs assimilation horizon, measured ONE METHOD AT A TIME.

Why this exists.  ``exp_B_cost`` times the three methods interleaved, with the stated
intention of sharing GPU contention fairly.  That backfires: contention adds a roughly
CONSTANT number of milliseconds per step, so it inflates a 9.5 ms method by ~90% and a
4425 ms method by ~0.2%.  Interleaving is therefore systematically biased against the fast
method, and it made KAE-expm appear to grow 9.07x over an 80x horizon range when measured
in isolation it is flat to 0.2%.

Measured here one method at a time, with warm-up and CUDA synchronisation around every
step, and the median over many steps.  Both numbers are reported in the notebook: the
isolated one is the correct measure of each method's cost, and the difference between them
is itself worth stating.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from data_assimilation.ks.protocol import DT, build_problem
from data_assimilation.ks.da_ks_experiments_3way import Bench, add_common_args

logger = logging.getLogger("cost")


def time_steps(step, n: int, warmup: int, cuda: bool) -> float:
    for _ in range(warmup):
        step()
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        step()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=Path("da_results_long"))
    add_common_args(ap)
    ap.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=[0.5, 1, 2, 5, 10, 20, 40, 100, 250, 400],
    )
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument(
        "--unet-max-steps",
        type=int,
        default=1000,
        help="beyond this the U-Net timing costs more than it informs",
    )
    a = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    a.out_dir.mkdir(parents=True, exist_ok=True)

    b = Bench(a)
    dev, cuda = b.dev, b.dev.type == "cuda"
    TL = json.loads(Path("da_results_sda_paper/lyapunov.json").read_text())[
        "lyapunov_time"
    ]
    rows = []
    for Ht in a.horizons:
        o = int(round(Ht / DT))
        if o + 120 >= b.data.n_t:
            logger.info(
                f"  horizon {Ht} needs {o} frames; record has {b.data.n_t} — skipped"
            )
            continue
        prob = build_problem(
            b.data, name=f"CI{Ht}", n_problems=a.batch, taus=np.array([o * DT]), seed=7
        )
        y = torch.as_tensor(prob.y, device=dev)
        mask = torch.as_tensor(prob.mask, device=dev)
        row = {"horizon_t": Ht, "steps": o, "T_L": Ht / TL}
        for m in ["KAE-expm", "KAE-rk4", "UNet"]:
            if m == "UNet" and o > a.unet_max_steps:
                row[m] = None
                continue
            mm = b.method(m)
            prep = mm.prepare(prob.taus)
            c = torch.zeros(
                tuple(mm.control_shape(a.batch)), device=dev, requires_grad=True
            )
            opt = torch.optim.Adam([c], lr=1e-2)

            def step(mm=mm, c=c, opt=opt, prep=prep):
                opt.zero_grad(set_to_none=True)
                p = mm.predict_at(c, prob.taus, prep)
                ((p - y) * mask[:, None, :]).pow(2).mean().backward()
                opt.step()

            reps = a.reps if o <= 400 else max(8, a.reps // 8)
            row[m] = time_steps(step, reps, warmup=min(20, reps), cuda=cuda)
            del c, opt
            if cuda:
                torch.cuda.empty_cache()
        logger.info(
            f"  tau={Ht:6.1f} ({o:5d} steps, {Ht / TL:5.2f} T_L) | "
            + " | ".join(
                f"{m}={row[m]:9.3f} ms" if row.get(m) is not None else f"{m}=  not run"
                for m in ["KAE-expm", "KAE-rk4", "UNet"]
            )
        )
        rows.append(row)

    ratios = {}
    for m in ["KAE-expm", "KAE-rk4", "UNet"]:
        v = [(r["horizon_t"], r[m]) for r in rows if r.get(m) is not None]
        if len(v) >= 2:
            ratios[m] = {
                "horizon_range": v[-1][0] / v[0][0],
                "cost_ratio": v[-1][1] / v[0][1],
                "first_ms": v[0][1],
                "last_ms": v[-1][1],
                "first_horizon": v[0][0],
                "last_horizon": v[-1][0],
            }
    rep = {
        "method": "one method at a time, warm-up + cudaSynchronize, median of many steps",
        "batch": a.batch,
        "device": str(dev),
        "data": str(a.test),
        "lyapunov_time": TL,
        "rows": rows,
        "scaling": ratios,
        "why": (
            "exp_B_cost interleaves the methods, which shares a roughly constant "
            "per-step contention and therefore inflates the cheap method far more "
            "than the expensive one. Those numbers overstate KAE-expm's growth with "
            "the horizon and must not be used for the scaling claim."
        ),
    }
    (a.out_dir / "cost_isolated.json").write_text(json.dumps(rep, indent=2))
    logger.info("")
    for m, v in ratios.items():
        logger.info(
            f"  {m:9s} {v['first_ms']:8.3f} -> {v['last_ms']:9.3f} ms over a "
            f"{v['horizon_range']:.0f}x horizon range = {v['cost_ratio']:6.2f}x"
        )
    logger.info(f"saved -> {a.out_dir / 'cost_isolated.json'}")


if __name__ == "__main__":
    main()
