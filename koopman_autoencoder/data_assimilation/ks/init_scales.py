"""Freeze the uninformed initialisation scales for both methods, from TRAINING data only.

Neither method may look at the analysis state it is asked to recover, nor at the held-out
trajectories, when choosing how to initialise its control variable.  Both scales are
therefore computed once on ``data/ks/train.nc``, written to a JSON file, and read from
that file by the tuning and campaign scripts.  Nothing recomputes them at test time.

    python -m data_assimilation.ks.init_scales --out da_results_v2/init_scales.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from data_assimilation.ks.models_io import kae_latent_scale, load_kae
from data_assimilation.ks.protocol import KSData

logger = logging.getLogger("data_assimilation.ks.init_scales")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/ks/train.nc"))
    ap.add_argument(
        "--kae-run",
        type=Path,
        default=Path("model_outputs_ks/continous_linear_128/rollout_10"),
    )
    ap.add_argument("--out", type=Path, default=Path("da_results_v2/init_scales.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.getLogger("models").setLevel(logging.WARNING)
    dev = torch.device(args.device)
    train = KSData(args.train, dev)

    # U-Net control: the physical state itself, so the natural uninformed scale is the
    # per-element standard deviation of the normalised training field.
    unet_scale = float(train.u.std())

    # KAE control: a latent code, so the matching quantity is the per-element standard
    # deviation of encoder latents over the training set.
    kae, _K, _D = load_kae(args.kae_run, None, dev)
    kae_scale = kae_latent_scale(kae, train)

    out = {
        "source": str(args.train),
        "kae_run": str(args.kae_run),
        "kae_latent_init_scale": kae_scale,
        "unet_state_init_scale": unet_scale,
        "note": (
            "Both scales are aggregate statistics of the TRAINING split only. They are "
            "frozen here and read from this file by data_assimilation/ks/tune.py and run_da_suite.py; "
            "neither is recomputed from validation or held-out data."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"KAE latent init scale  = {kae_scale:.6f}")
    logger.info(f"U-Net state init scale = {unet_scale:.6f}")
    logger.info(f"frozen -> {args.out}")


if __name__ == "__main__":
    main()
