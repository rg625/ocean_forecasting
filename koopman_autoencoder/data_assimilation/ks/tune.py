"""Select the DA optimiser hyper-parameters for each method on the VALIDATION split.

Both methods are given the identical search grid, identical iteration budget, identical
observation protocol and identical seeds; the winner is chosen by mean analysis error on
``data/ks/val.nc``.  The held-out DA trajectories (``data/ks/da_test.nc``) are never
touched here.  This is what makes it legitimate to use a different numerical learning
rate for a 128-dimensional latent control and a 64-dimensional physical control.

    python data_assimilation/ks/tune.py --out da_results_v2/tuning.json
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from data_assimilation.ks.evaluate import evaluate_solution, summarize
from data_assimilation.ks.methods import KAE4DVar, SolveConfig, UNet4DVar
from data_assimilation.ks.models_io import load_kae, load_unet
from data_assimilation.ks.protocol import KSData, build_problem

logger = logging.getLogger("data_assimilation.ks.tune")

CANONICAL_TAUS = np.array([0.1, 0.3, 0.7, 1.5, 2.5])
GRID_LR = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
GRID_INIT = ["random", "zero"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kae-run",
        type=Path,
        default=Path("model_outputs_ks/continous_linear_128/rollout_10"),
    )
    ap.add_argument(
        "--unet-ckpt",
        type=Path,
        default=Path("model_outputs_ks/unet1d/rollout10_base/best_model.pth"),
    )
    ap.add_argument("--val", type=Path, default=Path("data/ks/val.nc"))
    ap.add_argument("--train", type=Path, default=Path("data/ks/train.nc"))
    ap.add_argument("--n-problems", type=int, default=24)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--init-scales", type=Path, default=Path("da_results_v2/init_scales.json")
    )
    ap.add_argument("--out", type=Path, default=Path("da_results_v2/tuning.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    dev = torch.device(args.device)
    data = KSData(args.val, dev)

    kae, K, D = load_kae(args.kae_run, None, dev)
    unet = load_unet(args.unet_ckpt, dev)
    # initialisation scales frozen from the training split (see data_assimilation/ks/init_scales.py)
    with open(args.init_scales) as f:
        scales = json.load(f)
    z_scale = float(scales["kae_latent_init_scale"])
    x_scale = float(scales["unet_state_init_scale"])
    logger.info(f"frozen init scales: KAE z0 {z_scale:.4f}, U-Net x0 {x_scale:.4f}")

    # Validation problems: the *realistic* canonical setting (sparse + noisy + irregular)
    prob = build_problem(
        data,
        name="tune",
        n_problems=args.n_problems,
        taus=CANONICAL_TAUS,
        obs_frac=0.25,
        noise_std=0.05,
        seed=args.seed,
    )

    methods = {
        "KAE-expm": KAE4DVar(kae, K, D, z_scale, "expm"),
        "UNet-4DVar": UNet4DVar(unet, init_scale=x_scale, checkpoint_every=-1),
    }

    results = {}
    for mname, m in methods.items():
        rows = []
        for init in GRID_INIT:
            for lr in GRID_LR:
                cfg = SolveConfig(iters=args.iters, lr=lr, init=init, seed=args.seed)
                res = m.solve(prob, data, cfg)
                ev = evaluate_solution(m, res["control"], prob, data)
                s = summarize(ev["init_rel_l2"])
                so = summarize(ev["obs_rel_l2"])
                rows.append(
                    {
                        "lr": lr,
                        "init": init,
                        "init_rel_l2_mean": s["mean"],
                        "init_rel_l2_median": s["median"],
                        "obs_rel_l2_mean": so["mean"],
                        "s_per_solve": res["total_s"],
                    }
                )
                logger.info(
                    f"{mname:12s} init={init:6s} lr={lr:<6g} "
                    f"analysis={s['mean']:.4f} obs={so['mean']:.4f} "
                    f"({res['total_s']:.1f}s)"
                )
        best = min(rows, key=lambda r: r["init_rel_l2_mean"])
        results[mname] = {"grid": rows, "best": best}
        logger.info(
            f"--> {mname}: best lr={best['lr']} init={best['init']} "
            f"analysis={best['init_rel_l2_mean']:.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "val_file": str(args.val),
        "n_problems": args.n_problems,
        "iters": args.iters,
        "seed": args.seed,
        "taus": CANONICAL_TAUS.tolist(),
        "obs_frac": 0.25,
        "noise_std": 0.05,
        "grid_lr": GRID_LR,
        "grid_init": GRID_INIT,
        "kae_run": str(args.kae_run),
        "unet_ckpt": str(args.unet_ckpt),
        "init_scales": scales,
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
