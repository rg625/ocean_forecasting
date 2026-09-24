"""Choose the KAE's learning rate and initialisation on VALIDATION, at the reported budget.

The campaign's lr was picked at 2000 iterations and its initialisation was never a choice at
all. Both are re-selected here on val.nc at the budget the report uses, then frozen before
any test number is computed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_assimilation.ks.exp_geometry import schedule
from data_assimilation.ks.init_utils import init_kwargs
from data_assimilation.ks.methods import KAE4DVar, SolveConfig
from data_assimilation.ks.models_io import load_kae
from data_assimilation.ks.protocol import DT, KSData, build_problem, rel_l2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--kae-run", default="model_outputs_ks_latentdim/dz_512/run-20260906_140222"
    )
    ap.add_argument("--val", type=Path, default=Path("data/ks/val.nc"))
    ap.add_argument(
        "--out", type=Path, default=Path("da_results_geometry_df9/kae_tuning.json")
    )
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 9])
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    ap.add_argument("--inits", nargs="+", default=["zero", "obs"])
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--n-problems", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    data = KSData(a.val, dev)
    kae, K, D = load_kae(Path(a.kae_run), None, dev)
    rows = []
    for df in a.deltas:
        fr = schedule(df, 25, 5)
        prob = build_problem(
            data, name=f"T{df}", n_problems=a.n_problems, taus=fr * DT, seed=a.seed
        )
        u0 = data.frames(
            torch.as_tensor(prob.sim, device=dev), torch.as_tensor(prob.t0, device=dev)
        )
        for init in a.inits:
            ik = init_kwargs(init, kae, K, prob, dev)
            for lr in a.lrs:
                m = KAE4DVar(kae, K, D, 1.0, "expm")
                r = m.solve(
                    prob, data, SolveConfig(iters=a.iters, lr=lr, seed=a.seed, **ik)
                )
                with torch.no_grad():
                    v = (
                        rel_l2(
                            data.denorm(m.analysis_field(r["control"])), data.denorm(u0)
                        )
                        .cpu()
                        .numpy()
                    )
                rows.append(
                    {
                        "delta_f": df,
                        "init": init,
                        "lr": lr,
                        "mean": float(v.mean()),
                        "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
                    }
                )
                print(
                    f"val df={df:2d} init={init:5s} lr={lr:<6g} {v.mean():.4f}",
                    flush=True,
                )
                a.out.parent.mkdir(parents=True, exist_ok=True)
                a.out.write_text(
                    json.dumps(
                        {
                            "meta": {
                                "val": str(a.val),
                                "iters": a.iters,
                                "n_problems": a.n_problems,
                                "seed": a.seed,
                                "kae_run": a.kae_run,
                            },
                            "grid": rows,
                        },
                        indent=2,
                    )
                )
    # pick one setting for every schedule: the best mean rank across the tested delta_f
    best, best_score = None, np.inf
    for init in a.inits:
        for lr in a.lrs:
            sel = [r["mean"] for r in rows if r["init"] == init and r["lr"] == lr]
            score = float(np.mean([np.log(v) for v in sel]))  # geometric mean
            if score < best_score:
                best, best_score = {"init": init, "lr": lr}, score
    doc = json.loads(a.out.read_text())
    doc["best"] = best
    doc["selection"] = (
        "geometric mean of the analysis error over the tested delta_f, "
        "chosen on validation only"
    )
    a.out.write_text(json.dumps(doc, indent=2))
    print("BEST (frozen):", best)


if __name__ == "__main__":
    main()
