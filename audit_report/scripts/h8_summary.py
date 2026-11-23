"""Summarise the fair-horizon H8 runs: test forecast (mean +- std over seeds, last.pth), DA
(mean rel-L2 over each sweep's points) for every model in da_v2 (lr grid 0.03-1) and da_v3
(widened grids), and the DA conditioning diagnostic (da_v3/conditioning).

    python h8_summary.py [--json out.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(
    "/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder/model_outputs_ctkae"
)
SWEEPS = {
    "df9": [
        "G4_delta_l",
        "G2_noise",
        "G3_sparsity",
        "G19_misspec",
        "G5_n_obs",
        "G20_joint_sparsity",
    ],
    "df1": [
        "G1_delta_f",
        "G6_single_obs",
        "G4_delta_l",
        "G2_noise",
        "G3_sparsity",
        "G19_misspec",
        "G5_n_obs",
        "G20_joint_sparsity",
    ],
}
ap = argparse.ArgumentParser()
ap.add_argument("--json", default=None)
args = ap.parse_args()


def da(d: Path) -> dict:
    out: dict = {}
    if (d / "df9/tuning.json").exists():
        t = json.loads((d / "df9/tuning.json").read_text())["KAE"]
        out["lr"] = t["lr"]
        out["tune"] = {s["lr"]: round(s["rel"], 4) for s in t["sweep"]}
    if (d / "repr_check.json").exists():
        r = json.loads((d / "repr_check.json").read_text())
        out["repr"] = {
            k: round(r[k]["mean"], 4)
            for k in ("reconstruction", "encode_init", "climatology_init")
        }
    for sch, secs in SWEEPS.items():
        for s in secs:
            p = d / sch / f"{s}.json"
            if p.exists():
                rows = [
                    r["KAE"]["mean"]
                    for r in json.loads(p.read_text()).get("rows", [])
                    if "KAE" in r
                ]
                if rows:
                    out[f"{sch}/{s}"] = round(float(np.mean(rows)), 4)
    return out


res: dict = {"forecast": {}, "da": {}}
for d in ("H8_d128", "H8_d256", "H8ms_d256"):
    per = []
    for run in sorted(ROOT.glob(f"{d}_s*_*")):
        e = run / "eval/last.json"
        if not e.exists():
            continue
        j = json.loads(e.read_text())
        row: dict = {"run": run.name}
        for c in ("interp", "extrap", "longer"):
            row[c] = j[c]["mse_pred"]
            row[f"{c}_lead1"] = j[c]["mse_lead"][0]
            row[f"{c}_lead10"] = j[c]["mse_lead"][9]
            row[f"{c}_E58"] = j[c]["energy_lead"][57]
        per.append(row)
    if not per:
        continue
    keys = [k for k in per[0] if k != "run"]
    res["forecast"][d] = {
        "runs": per,
        "mean": {k: round(float(np.mean([r[k] for r in per])), 3) for k in keys},
        "std": {
            k: (
                round(float(np.std([r[k] for r in per], ddof=1)), 3)
                if len(per) > 1
                else None
            )
            for k in keys
        },
    }
for v in ("da_v2", "da_v3"):
    for dd in sorted((ROOT / v).iterdir()):
        if dd.is_dir() and dd.name != "conditioning":
            res["da"][f"{v}/{dd.name}"] = da(dd)
res["conditioning"] = {}
for f in sorted((ROOT / "da_v3/conditioning").glob("*.json")):
    j = json.loads(f.read_text())
    res["conditioning"][f.stem] = {
        "model": j["model"],
        "lr": j["lr"],
        "analysis_rel": round(j["analysis_rel"], 4),
        "fd_vs_autograd_rel_err": j["fd_vs_autograd_rel_err"],
        "share_problems_J_rec_below_J_true": j["share_problems_J_rec_below_J_true"],
        **{k: round(v, 4) for k, v in j["mean"].items()},
    }
print(json.dumps(res, indent=1))
if args.json:
    Path(args.json).write_text(json.dumps(res, indent=1))
