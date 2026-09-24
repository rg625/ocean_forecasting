"""Sanity check: is the U-Net a competent KS forecaster in its own right?

The data-assimilation comparison is only meaningful if the U-Net baseline is a strong
forecaster.  This script measures plain conditional forecast skill for both frozen
models on the held-out DA trajectories, starting from the *true* state (no assimilation
involved), and writes the curve so it can be reported alongside the DA results.

    python -m data_assimilation.ks.forecast_check --out da_results_v2/forecast_check.json
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

from data_assimilation.ks.methods import KAE4DVar, UNet4DVar
from data_assimilation.ks.models_io import kae_latent_scale, load_kae, load_unet
from data_assimilation.ks.protocol import DT, KSData, rel_l2

logger = logging.getLogger("data_assimilation.ks.forecast_check")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, default=Path("data/ks/da_test.nc"))
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
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-tau", type=float, default=20.0)
    ap.add_argument(
        "--out", type=Path, default=Path("da_results_v2/forecast_check.json")
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.getLogger("models").setLevel(logging.WARNING)
    dev = torch.device(args.device)
    data = KSData(args.test, dev)

    kae, K, D = load_kae(args.kae_run, None, dev)
    unet = load_unet(args.unet_ckpt, dev)
    zscale = kae_latent_scale(kae, data)

    rng = np.random.default_rng(0)
    n_max = int(round(args.max_tau / DT))
    sim = torch.as_tensor(np.arange(args.n) % data.n_sim, device=dev)
    t0 = torch.as_tensor(
        rng.integers(100, data.n_t - n_max - 1, size=args.n), device=dev
    )
    x0 = data.frames(sim, t0)  # [B, X]

    from tensordict import TensorDict

    z0 = kae.present_encoding(
        TensorDict({"u": x0.unsqueeze(1).unsqueeze(-1)}, batch_size=[args.n, 1]), None
    )

    taus = np.round(np.arange(1, n_max + 1) * DT, 4)
    m_kae = KAE4DVar(kae, K, D, zscale, "expm")
    m_unet = UNet4DVar(unet)

    curves = {}
    for name, m, c in [("KAE-expm", m_kae, z0), ("UNet", m_unet, x0)]:
        preds = []
        for i in range(0, len(taus), 40):  # chunked to bound memory
            preds.append(m.forecast(c, taus[i : i + 40]).cpu())
        pred = torch.cat(preds)  # [n_tau, B, X]
        true = torch.stack(
            [data.frames(sim, t0 + int(round(t / DT))) for t in taus]
        ).cpu()
        e = rel_l2(data.denorm(pred), data.denorm(true))  # [n_tau, B]
        curves[name] = {
            "tau": taus.tolist(),
            "mean": e.mean(1).numpy().tolist(),
            "std": e.std(1).numpy().tolist(),
        }
        for probe in (0.5, 1.0, 2.5, 5.0, 10.0, 20.0):
            j = int(np.argmin(np.abs(taus - probe)))
            logger.info(
                f"{name:9s} tau={taus[j]:5.1f}  rel-L2 = "
                f"{e[j].mean():.4f} +/- {e[j].std():.4f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "n_trajectories": args.n,
                "test_file": str(args.test),
                "kae_run": str(args.kae_run),
                "unet_ckpt": str(args.unet_ckpt),
                "note": (
                    "Plain conditional forecast skill from the TRUE initial state; "
                    "no assimilation. Establishes that the U-Net baseline is a "
                    "competent forecaster before it is used inside 4D-Var."
                ),
                "curves": curves,
            },
            f,
            indent=2,
        )
    logger.info(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
