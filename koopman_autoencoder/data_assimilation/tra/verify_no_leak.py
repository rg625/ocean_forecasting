"""Gate 4: the frame being recovered is never an input to any method.

If the target state at t_0 reached a solver -- as an observation, as an initialisation, or
through the sampler's trajectory -- then every number in this campaign would be measuring
memorisation rather than inference.  Reading the code says it cannot happen: `build_problem`
asserts every observation lead is strictly positive, and `_solve_diffusion` reads the
analysis at trajectory index `back` while the observations sit at `offsets + back`, so the
analysis index is never an observation index.

Reading the code is not a measurement.  This gate does the decisive experiment instead:

    CORRUPT the ground truth at t_0 with large noise, re-solve with the identical seed,
    and require every method's analysis to come back BIT-IDENTICAL.

A method that had used the target in any way could not possibly return the same answer.
The score against truth naturally changes -- that is only the yardstick moving -- so the
comparison is made on the recovered FIELD, not on the error.

Run: ``python -m data_assimilation.tra.verify_no_leak``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from data_assimilation.tra.bridge import REGIMES, PhysicalData, REPO
from data_assimilation.tra.adapters import TurbpredAdapter
from data_assimilation.tra.kae_adapter import KAEAdapter
from data_assimilation.tra.protocol import build_problem, canonical
from data_assimilation.tra.fourdvar import solve_kae, solve_autoregressive
from data_assimilation.tra.verify_adapters import MODELS, DIFF_OPTS

DATA = {
    "test": "data/acdm/128_tra/gt_interp.nc",
    "long": "data/acdm/128_tra/gt_longer.nc",
    "val": "data/acdm/128_tra/gt_extrap.nc",
}


def analyses(b_model, data, prob, tuning, iters, n_samples, chunk, out_dir):
    """Every method's recovered field at t_0, as numpy."""
    from data_assimilation.tra.sda_blanket import ACDMBlanketScore, BlanketSDA

    res = {}
    dev = data.dev
    sim = torch.as_tensor(prob.sim, device=dev)
    for m in ["KAE", "UNet", "FNO", "ACDM", "ACDM-ncn"]:
        ad = b_model(m)
        if m in ("ACDM", "ACDM-ncn"):
            C = ad.n_fields + ad.n_params
            sc = ACDMBlanketScore(ad, C)
            td = json.loads((out_dir / "tuning_diffusion.json").read_text())[m]
            sda = BlanketSDA(
                sc,
                sigma_y=max(prob.noise_std, 0.05),
                gamma=td["gamma"],
                corrections=td["corrections"],
                tau=0.5,
            )
            L = max(int(prob.offsets.max()) + 1, sc.window)
            obs_idx = torch.as_tensor(prob.offsets, device=dev)
            assert (
                0 not in prob.offsets.tolist()
            ), "the analysis index 0 must never be an observation index"
            y = ad.to_model(torch.as_tensor(prob.y, device=dev)).transpose(0, 1)
            msk = (
                torch.as_tensor(prob.mask, device=dev).transpose(0, 1).transpose(-1, -2)
            )
            par = data.params_for(sim)
            H, W = y.shape[-2], y.shape[-1]
            N = len(prob.offsets)
            pch = ad.param_channel(par, (H, W))
            yf = torch.cat([y, pch.unsqueeze(1).expand(-1, N, -1, -1, -1)], dim=2)
            mf = torch.cat(
                [
                    msk.expand(-1, -1, ad.n_fields, -1, -1),
                    torch.ones_like(pch).unsqueeze(1).expand(-1, N, -1, -1, -1),
                ],
                dim=2,
            )
            smp = sda.sample(
                (len(prob.sim), L, C, H, W),
                obs_idx,
                yf,
                mf,
                seed=100,
                n_samples=n_samples,
            )
            a = ad.to_physical(smp[:, :, 0, : ad.n_fields]).mean(0)
        else:
            fn = solve_kae if m == "KAE" else solve_autoregressive
            a = torch.as_tensor(
                fn(ad, data, prob, iters=iters, lr=tuning[m]["lr"], seed=100)[
                    "analysis"
                ]
            )
        res[m] = a.detach().cpu().numpy().copy()
        del ad
        torch.cuda.empty_cache()
    return res


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--regime", default="test", choices=list(DATA))
    ap.add_argument("--out-dir", type=Path, default=Path("da_results_tra"))
    ap.add_argument("--n-problems", type=int, default=3)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument(
        "--kae-run",
        default=("model_outputs_tra/continous_linear_128/" "run-20260821_031121"),
    )
    a = ap.parse_args()
    dev = torch.device("cuda")
    reg = REGIMES["tra"]
    base = REPO / "autoreg_pde_diffusion" / "pretrained_models" / "models_tra"
    tuning = json.loads((a.out_dir / "tuning.json").read_text())

    def model(m):
        if m == "KAE":
            return KAEAdapter(a.kae_run, reg, dev)
        return TurbpredAdapter(
            base / MODELS[m] / "Model.pth", reg, dev, DIFF_OPTS.get(m), name=m
        )

    data = PhysicalData(DATA[a.regime], "tra", dev)
    prob = build_problem(
        data, name="LEAK", n_problems=a.n_problems, offsets=canonical(), seed=100
    )
    print(
        f"regime {a.regime} | offsets {prob.offsets.tolist()} "
        f"(all > 0, so t_0 is never observed)"
    )

    clean = analyses(model, data, prob, tuning, a.iters, a.n_samples, 1, a.out_dir)

    # --- corrupt the ground truth AT THE TARGET, then re-solve identically -----
    sim_i = torch.as_tensor(prob.sim)
    t0_i = torch.as_tensor(prob.t0)
    g = torch.Generator().manual_seed(0)
    before = data.u[sim_i, t0_i].clone()
    data.u[sim_i, t0_i] = torch.randn(before.shape, generator=g) * before.std() * 5
    moved = float((data.u[sim_i, t0_i] - before).abs().max())
    print(f"corrupted the truth at t_0 by max |delta| = {moved:.3f}\n")

    dirty = analyses(model, data, prob, tuning, a.iters, a.n_samples, 1, a.out_dir)
    data.u[sim_i, t0_i] = before  # restore

    print(f"{'method':10s} {'max |clean - corrupted|':>24s}   verdict")
    bad = 0
    for m in clean:
        d = float(np.abs(clean[m] - dirty[m]).max())
        ok = d == 0.0
        bad += not ok
        print(
            f"{m:10s} {d:24.3e}   {'IDENTICAL - no leak' if ok else 'CHANGED - LEAK!'}"
        )
    print(
        "\n"
        + (
            "NO METHOD USES THE TARGET FRAME"
            if not bad
            else f"{bad} METHOD(S) LEAK THE TARGET"
        )
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
