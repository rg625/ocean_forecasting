"""Rank the loss-sweep runs (LS_*) by the pre-registered score on the test cases
S = mean over {interp, extrap, longer} of MSE / MSE(R) + da_df9 / da_df9(R)
(lower is better; R = reference run; scores from test_score.py).

    python loss_sweep_summary.py [--prefix LS_] [--json out.json]
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default="LS_")
ap.add_argument("--ref", default="R")
ap.add_argument("--json", default=None)
args = ap.parse_args()
root = Path(
    "/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder/model_outputs_ctkae"
)
rows: dict = {}
for f in sorted(root.glob(f"{args.prefix}*/test_score.json")):
    j = json.loads(f.read_text())
    tag = f.parent.name[len(args.prefix) :].rsplit("_s", 1)[0]
    seed = f.parent.name.rsplit("_s", 1)[1].split("_")[0]
    rows.setdefault(tag, []).append({**j, "seed": seed})
if args.ref not in rows:
    raise SystemExit(f"reference run {args.ref} missing")
avg = {
    t: {
        k: sum(r[k] for r in v) / len(v)
        for k in ("fc_mse60", "fc_lead1", "fc_lead10", "fc_E58", "da_df9")
    }
    | {"n": len(v), "da_lr": [r["da_lr"] for r in v]}
    for t, v in rows.items()
}
ref = avg[args.ref]
for t, a in avg.items():
    a["S"] = (
        sum(a[f"{c}_mse"] / ref[f"{c}_mse"] for c in ("interp", "extrap", "longer")) / 3
        + a["da_df9"] / ref["da_df9"]
    )
print(
    f"{'run':10s} {'n':>2s} {'S':>6s} {'fc60':>7s} {'lead1':>6s} {'lead10':>7s} {'E58':>5s} {'da_df9':>7s}  lr"
)
for t, a in sorted(avg.items(), key=lambda kv: kv[1]["S"]):
    flag = " *" if a["S"] <= avg[args.ref]["S"] - 0.05 else ""
    print(
        f"{t:10s} {a['n']:2d} {a['S']:6.3f} {a['fc_mse60']:7.3f} {a['fc_lead1']:6.3f} {a['fc_lead10']:7.3f} {a['fc_E58']:5.2f} {a['da_df9']:7.4f}  {a['da_lr']}{flag}"
    )
if args.json:
    Path(args.json).write_text(json.dumps(avg, indent=1))
