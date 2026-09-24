# mypy: disable-error-code="arg-type, no-any-return"
"""Pick the first guess and the learning rate for TRA 4D-Var, on validation only.

The same protocol as the KS tuning (``data_assimilation.ks.tune_ks_kae_init``): a grid over
{climatology, observation} x {learning rates}, scored at delta_f = 1 AND delta_f = 9 so
the choice is not made at one lead and applied at another, then frozen and replayed on the
test regime.  Selection is by the geometric mean over the two leads, which refuses a
setting that wins at one and collapses at the other.

The validation regime is gt_extrap (Mach 0.50-0.52); the test regime is never touched.

    python -m data_assimilation.tra.tune_tra_init --out da_results_tra/tuning_init.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from data_assimilation.tra.exp_tra import FOURDVAR, Bench
from data_assimilation.tra.protocol import build_problem

logger = logging.getLogger("tune_tra_init")


def schedule(df: int, dl: int, n: int) -> np.ndarray:
    """N observation frames spanning [df, dl], both endpoints pinned. As on KS."""
    g = np.unique(np.round(np.geomspace(df, dl, n)).astype(int))
    return np.unique(np.concatenate([[df], g, [dl]]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--kae-run",
        default=("model_outputs_tra/continous_linear_128/run-20260821_031121"),
    )
    ap.add_argument("--out", type=Path, default=Path("da_results_tra/tuning_init.json"))
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 9])
    ap.add_argument("--delta-l", type=int, default=25)
    ap.add_argument("--n-obs", type=int, default=5)
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.03, 0.1])
    ap.add_argument("--inits", nargs="+", default=["climatology", "obs"])
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--n-tune", type=int, default=6)
    ap.add_argument("--methods", nargs="+", default=FOURDVAR, choices=FOURDVAR)
    ap.add_argument("--seed", type=int, default=1234)
    # Bench reads these off the namespace
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=2)
    a = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    a.kae_run = Path(a.kae_run)

    b = Bench(a)
    data = b.data("val")
    probs = {
        df: build_problem(
            data,
            name=f"tune_df{df}",
            n_problems=a.n_tune,
            offsets=schedule(df, a.delta_l, a.n_obs),
            seed=a.seed,
        )
        for df in a.deltas
    }
    for df, p in probs.items():
        logger.info(f"  delta_f={df}: frames {p.offsets.tolist()}")

    grid, best = [], {}
    for name in a.methods:
        ad = b.model(name)
        cells = {}
        for init in a.inits:
            for lr in a.lrs:
                per = {}
                for df in a.deltas:
                    r = b.solve(
                        name,
                        ad,
                        data,
                        probs[df],
                        iters=a.iters,
                        lr=lr,
                        seed=1,
                        init=init,
                    )
                    per[df] = float(np.mean(r["rel"]))
                    grid.append(
                        {
                            "method": name,
                            "delta_f": df,
                            "init": init,
                            "lr": lr,
                            "rel": per[df],
                            "wall_s": r["wall_s"],
                        }
                    )
                    logger.info(
                        f"  {name:5s} df={df:2d} init={init:11s} lr={lr:<6g} "
                        f"rel-L2 {per[df]:.4f}  [{r['wall_s']:.0f}s]"
                    )
                    a.out.parent.mkdir(parents=True, exist_ok=True)
                    a.out.write_text(
                        json.dumps(
                            {
                                "meta": vars(a)
                                | {
                                    "out": str(a.out),
                                    "kae_run": str(a.kae_run),
                                    "val": "gt_extrap.nc",
                                    "partial": True,
                                },
                                "grid": grid,
                            },
                            indent=2,
                            default=str,
                        )
                    )
                # geometric mean: a setting must hold up at BOTH leads
                cells[(init, lr)] = float(
                    np.exp(np.mean(np.log(np.maximum(list(per.values()), 1e-12))))
                )
        pick = min(cells, key=cells.get)
        best[name] = {"init": pick[0], "lr": pick[1], "score": cells[pick]}
        logger.info(
            f"  -> {name}: init={pick[0]} lr={pick[1]} "
            f"(geometric mean {cells[pick]:.4f})"
        )
        del ad
        torch.cuda.empty_cache()

    a.out.write_text(
        json.dumps(
            {
                "meta": {
                    "val": "gt_extrap.nc (validation regime)",
                    "iters": a.iters,
                    "n_tune": a.n_tune,
                    "deltas": a.deltas,
                    "delta_l": a.delta_l,
                    "n_obs": a.n_obs,
                    "kae_run": str(a.kae_run),
                    "selection": "geometric mean of the analysis error over the tested "
                    "delta_f, chosen on validation only",
                },
                "grid": grid,
                "best": best,
            },
            indent=2,
        )
    )
    logger.info(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
