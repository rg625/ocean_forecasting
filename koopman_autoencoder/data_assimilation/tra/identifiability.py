# ruff: noqa: F841
"""Is a poor analysis an optimisation failure or an identifiability failure?

Compare the 4D-Var objective at the RECOVERED x_0 with its value at the TRUE x_0:

    J(true) <  J(recovered)   ->  the cost prefers the truth and the optimiser did not get
                                  there: ILL-CONDITIONING, more/better optimisation helps.
    J(true) >= J(recovered)   ->  the optimiser found a point the cost likes at least as
                                  much as the truth: UNIDENTIFIABILITY, no amount of extra
                                  optimisation helps and the observations are the problem.

This is the only measurement that separates the two, and it decides whether a bad number
is a bug, a budget, or a property of the problem.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import numpy as np
import torch

from data_assimilation.tra.bridge import REGIMES, PhysicalData, REPO
from data_assimilation.tra.adapters import TurbpredAdapter
from data_assimilation.tra.protocol import build_problem, canonical
from data_assimilation.tra.fourdvar import solve_autoregressive
from data_assimilation.tra.verify_adapters import MODELS, DIFF_OPTS


def obs_loss_autoreg(ad, data, prob, control, par):
    idx = torch.as_tensor(prob.offsets + (ad.n_control_frames - 1), device=ad.dev)
    y = torch.as_tensor(prob.y, device=ad.dev)
    mask = torch.as_tensor(prob.mask, device=ad.dev)
    with torch.no_grad():
        traj = ad.rollout(control, int(prob.offsets.max()), par, checkpoint_every=0)
        pred = traj[:, idx].transpose(0, 1)
        return float(((pred - y) * mask).pow(2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kae-run",
        default=("model_outputs_tra/continous_linear_128/" "run-20260821_031121"),
    )
    ap.add_argument("--n-problems", type=int, default=8)
    ap.add_argument("--iters", type=int, default=2000)
    # This diagnostic is not a --sections entry of exp_tra, so the per-regime campaigns
    # never ran it and only gt_interp had a result: the cross-regime figure drew one panel
    # and blanked the other two. It takes a regime and an output directory for the same
    # reason every other section does.
    ap.add_argument("--regime", default="test", choices=["test", "long", "val"])
    ap.add_argument("--out-dir", default="da_results_tra")
    a = ap.parse_args()
    dev = torch.device("cuda")
    reg = REGIMES["tra"]
    DATA = {
        "test": "data/acdm/128_tra/gt_interp.nc",
        "long": "data/acdm/128_tra/gt_longer.nc",
        "val": "data/acdm/128_tra/gt_extrap.nc",
    }
    print(f"regime: {a.regime} -> {DATA[a.regime]}   out: {a.out_dir}")
    data = PhysicalData(DATA[a.regime], "tra", dev)
    prob = build_problem(
        data, name="ID", n_problems=a.n_problems, offsets=canonical(), seed=100
    )
    tuning = json.loads(open("da_results_tra/tuning.json").read())
    base = REPO / "autoreg_pde_diffusion" / "pretrained_models" / "models_tra"
    sim = torch.as_tensor(prob.sim, device=dev)
    t0 = torch.as_tensor(prob.t0, device=dev)
    par = data.params_for(sim)
    om = data.mask_for(sim)

    print(
        f"{'method':6s} {'J(recovered)':>14s} {'J(true x0)':>12s} {'ratio':>8s}  verdict"
    )
    out = {}
    for m in ["UNet", "FNO"]:
        ad = TurbpredAdapter(
            base / MODELS[m] / "Model.pth", reg, dev, DIFF_OPTS.get(m), name=m
        )
        k = ad.n_control_frames
        r = solve_autoregressive(
            ad, data, prob, iters=a.iters, lr=tuning[m]["lr"], seed=100
        )
        c_rec = r["control"]
        c_true = ad.to_model(data.window(sim, t0 - (k - 1), k))
        j_rec = obs_loss_autoreg(ad, data, prob, c_rec, par)
        j_true = obs_loss_autoreg(ad, data, prob, c_true, par)
        verdict = (
            "ILL-CONDITIONED (cost prefers the truth)"
            if j_true < j_rec
            else "UNIDENTIFIABLE (cost does not prefer the truth)"
        )
        print(
            f"{m:6s} {j_rec:14.4e} {j_true:12.4e} {j_rec / max(j_true, 1e-30):8.1f}x  "
            f"{verdict}"
        )
        out[m] = {
            "J_recovered": j_rec,
            "J_true": j_true,
            "ratio": j_rec / max(j_true, 1e-30),
            "analysis_rel": float(np.mean(r["rel"])),
            "ill_conditioned": bool(j_true < j_rec),
        }
        del ad
        torch.cuda.empty_cache()
    f = Path(a.out_dir) / "identifiability.json"
    json.dump(out, open(f, "w"), indent=2)
    print(f"\nsaved -> {f}")


if __name__ == "__main__":
    sys.exit(main())
