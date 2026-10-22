"""Test score of a CT-KAE run for the loss-hyperparameter search (selection on the test cases,
at the user's request; see the caveat in EXPERIMENT_RESULTS.md).

Forecasting: RUN/eval/<ckpt>.json from eval_ctkae.py (gt_interp / gt_extrap / gt_longer,
benchmark scorer). Data assimilation, as the established protocol: 4D-Var learning rate tuned
on gt_extrap (canonical schedule, 6 problems, 400 iterations), then the df = 9 default problem
on gt_interp (observations at t0 + [9, 12, 15, 19, 25], full fields, noise-free, 16 problems,
500 iterations).

    python test_score.py RUN_DIR [--ckpt last.pth] [--json out.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder")
sys.path.insert(0, str(REPO))
from data_assimilation.tra.bridge import REGIMES, PhysicalData  # noqa: E402
from data_assimilation.tra.ctkae_adapter import CTKAEAdapter  # noqa: E402
from data_assimilation.tra.fourdvar import solve_kae  # noqa: E402
from data_assimilation.tra.protocol import build_problem, canonical  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--ckpt", default="last.pth")
ap.add_argument("--lrs", type=float, nargs="+", default=[0.3, 1.0, 3.0, 10.0])
ap.add_argument("--json", default=None)
args = ap.parse_args()
dev = torch.device("cuda")
run = Path(args.run_dir)
ev = json.loads((run / "eval" / f"{Path(args.ckpt).stem}.json").read_text())
out: dict = {"run": run.name, "ckpt": args.ckpt}
for c in ("interp", "extrap", "longer"):
    out[f"{c}_mse"] = ev[c]["mse_pred"]
    out[f"{c}_lead1"] = ev[c]["mse_lead"][0]
    out[f"{c}_E58"] = ev[c]["energy_lead"][57]
    out[f"{c}_acc58"] = ev[c]["acc_lead"][57]

ad = CTKAEAdapter(run, REGIMES["tra"], dev, ckpt=args.ckpt)
D = REPO / "data/acdm/128_tra"
val = PhysicalData(D / "gt_extrap.nc", "tra", dev)
tune_prob = build_problem(
    val, name="tune", n_problems=6, offsets=canonical(), seed=1234
)
rows = []
for lr in args.lrs:
    r = solve_kae(ad, val, tune_prob, iters=400, lr=lr, seed=1)
    rows.append({"lr": lr, "rel": float(np.mean(r["rel"]))})
lr = min(rows, key=lambda r: r["rel"])["lr"]
test = PhysicalData(D / "gt_interp.nc", "tra", dev)
prob = build_problem(
    test, name="df9", n_problems=16, offsets=np.array([9, 12, 15, 19, 25]), seed=200
)
r = solve_kae(ad, test, prob, iters=500, lr=lr, seed=1)
out.update(da_df9=float(np.mean(r["rel"])), da_lr=lr, da_tune=rows)
print(json.dumps({k: v for k, v in out.items() if k != "da_tune"}))
if args.json:
    Path(args.json).write_text(json.dumps(out, indent=1))
