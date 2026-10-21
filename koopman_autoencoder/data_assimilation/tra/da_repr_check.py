"""Separate representation from optimisation in latent 4D-Var (diagnostic; leaks truth by design).

For the df = 9 default problem (observations at t0 + [9, 12, 15, 19, 25] frames, full fields,
no noise) on the test regime, reports the t0 analysis rel-L2 for:
  reconstruction   decode(encode(true window)), no optimisation: how well the
                   autoencoder can represent the truth at all (floor of any analysis)
  encode-init      4D-Var started from the encoded true state
  climatology-init 4D-Var from the uninformed start used in every reported result

    python -m data_assimilation.tra.da_repr_check --kae-kind legacy|ctkae --kae-run RUN \
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
ap.add_argument("--json", default=None)
a = ap.parse_args()
dev = torch.device("cuda")
if a.kae_kind == "ctkae":
    from data_assimilation.tra.ctkae_adapter import CTKAEAdapter

    ad = CTKAEAdapter(a.kae_run, REGIMES["tra"], dev, ckpt=a.kae_ckpt or "best.pth")
else:
    from data_assimilation.tra.kae_adapter import KAEAdapter

    ck = Path(a.kae_run) / "checkpoints" / a.kae_ckpt if a.kae_ckpt else None
    ad = KAEAdapter(a.kae_run, REGIMES["tra"], dev, ckpt=ck)
data = PhysicalData("data/acdm/128_tra/gt_interp.nc", "tra", dev)
prob = build_problem(
    data,
    name="repr_df9",
    n_problems=a.n_problems,
    offsets=np.array([9, 12, 15, 19, 25]),
    seed=200,
    noise_std=0.0,
)
res = {"model": repr(ad), "lr": a.lr, "iters": a.iters}
for tag, init, iters in [
    ("reconstruction", "encode", 0),
    ("encode_init", "encode", a.iters),
    ("climatology_init", "climatology", a.iters),
]:
    r = solve_kae(ad, data, prob, iters=iters, lr=a.lr, seed=1, init=init)
    v = np.asarray(r["rel"])
    res[tag] = {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(v.size))}
    print(f"{tag:17s} rel-L2 {v.mean():.4f} +- {res[tag]['sem']:.4f}")
if a.json:
    Path(a.json).write_text(json.dumps(res, indent=1))
