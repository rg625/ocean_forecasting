"""Validation-only score of a CT-KAE run for the loss-hyperparameter search (no test data).

On data/acdm/128_tra/val_heldout.nc (training simulations Ma 0.60 / 0.75 held out of the
sweep runs; make_val_nc.py):
  fc_mse60    60-frame forecast MSE x1e-3, physical units, obstacle zeroed, frames 2..59
              from frames 0, 1 (benchmark convention)
  fc_lead1, fc_lead10, fc_E58   per-lead MSE and fluctuation-energy ratio at step 58
  da_df9      4D-Var analysis rel-L2 on the df = 9 default problem (observations at t0 +
              [9, 12, 15, 19, 25], full fields, noise-free; 16 problems, 500 iterations),
              best over the learning-rate grid (chosen on these same validation problems)
plus val_mse240 / val_mse60 of the run's own final validation log entry.

    python val_score.py RUN_DIR [--ckpt last.pth] [--json out.json]
"""

from _paths import REPO_ROOT
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = REPO_ROOT / "koopman_autoencoder"
sys.path.insert(0, str(REPO))
from data_assimilation.tra.bridge import REGIMES, PhysicalData  # noqa: E402
from data_assimilation.tra.ctkae_adapter import CTKAEAdapter  # noqa: E402
from data_assimilation.tra.fourdvar import solve_kae  # noqa: E402
from data_assimilation.tra.protocol import build_problem  # noqa: E402
from eval_ctkae import load_model, rollout  # noqa: E402
from models.ct_data import denormalise, load_eval_case  # noqa: E402

VAL = REPO / "data/acdm/128_tra/val_heldout.nc"
ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--ckpt", default="last.pth")
ap.add_argument("--lrs", type=float, nargs="+", default=[0.3, 1.0, 3.0, 10.0])
ap.add_argument("--json", default=None)
args = ap.parse_args()
dev = torch.device("cuda")
run = Path(args.run_dir)
out: dict = {"run": run.name, "ckpt": args.ckpt}

# forecasting
m, _ = load_model(run, args.ckpt, dev)
x, ma, mask = load_eval_case(str(VAL), dev)
T = x.shape[1]
taus = 0.1 * torch.arange(1, T - 1, device=dev, dtype=torch.float32)
with torch.no_grad():
    pred = rollout(m, x, ma, mask, 1, taus)  # frames 2..T-1
    p, t = denormalise(pred) * mask, denormalise(x[:, 2:]) * mask
    lead = ((p - t) ** 2).mean(dim=(0, 2, 3, 4)) * 1e3
    fp = p - t.mean(1, keepdim=True)
    ft = t - t.mean(1, keepdim=True)
    e_lead = fp.pow(2).mean(dim=(0, 2, 3, 4)) / ft.pow(2).mean(dim=(0, 2, 3, 4))
out.update(
    fc_mse60=float(lead.mean()),
    fc_lead1=float(lead[0]),
    fc_lead10=float(lead[9]),
    fc_E58=float(e_lead[-1]),
)
del m
torch.cuda.empty_cache()

# data assimilation (df = 9 default problem)
ad = CTKAEAdapter(run, REGIMES["tra"], dev, ckpt=args.ckpt)
data = PhysicalData(VAL, "tra", dev)
prob = build_problem(
    data, name="val_df9", n_problems=16, offsets=np.array([9, 12, 15, 19, 25]), seed=300
)
rows = []
for lr in args.lrs:
    r = solve_kae(ad, data, prob, iters=500, lr=lr, seed=1)
    rows.append({"lr": lr, "rel": float(np.mean(r["rel"]))})
best = min(rows, key=lambda r: r["rel"])
out.update(da_df9=best["rel"], da_lr=best["lr"], da_sweep=rows)

# the run's own final validation entry (held-out windows, 60 and 240 steps)
vals = [
    json.loads(line)["val"]
    for line in (run / "log.jsonl").read_text().splitlines()
    if '"val"' in line
]
if vals and "val_mse60" in vals[-1]:
    out.update(
        log_val_mse60=vals[-1]["val_mse60"], log_val_mse240=vals[-1]["val_mse240"]
    )
print(json.dumps({k: v for k, v in out.items() if k != "da_sweep"}))
if args.json:
    Path(args.json).write_text(json.dumps(out, indent=1))
