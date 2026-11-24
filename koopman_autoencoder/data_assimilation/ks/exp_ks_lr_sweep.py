# ruff: noqa: E731, F841
"""Does a larger step size help, once the budget is long?

The campaign's learning rate was chosen on validation at 2000 iterations. The loss curves
run for 8000 and keep descending, so the natural question is whether the optimiser is simply
under-stepping. Each configuration is tracked along the way, so one run per learning rate
gives the whole budget curve and the 8k / 50k answers come from the same trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_assimilation.ks.exp_geometry import schedule
from data_assimilation.ks.methods import KAE4DVar, SolveConfig
from data_assimilation.ks.models_io import load_kae
from data_assimilation.ks.protocol import DT, KSData, build_problem

SDA = {1: 0.0094, 9: 0.0192}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--kae-run", default="model_outputs_ks_latentdim/dz_512/run-20260906_140222"
    )
    ap.add_argument("--test", type=Path, default=Path("data/ks/da_test.nc"))
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    ap.add_argument(
        "--json", type=Path, default=Path("da_results_geometry_df9/lr_sweep.json")
    )
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 9])
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--iters", type=int, default=50000)
    ap.add_argument("--track-every", type=int, default=250)
    ap.add_argument("--n-problems", type=int, default=16)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    data = KSData(a.test, dev)
    kae, K, D = load_kae(Path(a.kae_run), None, dev)
    out = {
        "meta": {
            "d_z": int(D),
            "iters": a.iters,
            "n_problems": a.n_problems,
            "kae_run": a.kae_run,
            "sda_reference": SDA,
        },
        "runs": {},
    }
    for df in a.deltas:
        fr = schedule(df, 25, 5)
        prob = build_problem(
            data, name=f"LR{df}", n_problems=a.n_problems, taus=fr * DT, seed=a.seed
        )
        u0 = data.denorm(
            data.frames(
                torch.as_tensor(prob.sim, device=dev),
                torch.as_tensor(prob.t0, device=dev),
            )
        )
        for lr in a.lrs:
            m = KAE4DVar(kae, K, D, 1.0, "expm")
            res = m.solve(
                prob,
                data,
                SolveConfig(
                    iters=a.iters,
                    lr=lr,
                    init="zero",
                    seed=a.seed,
                    track_every=a.track_every,
                ),
            )
            h = res["hist"]
            it = np.asarray(h["iter"], dtype=float)
            err = np.asarray(h["init_rel_l2"], dtype=float)
            at = lambda n: float(err[int(np.argmin(np.abs(it - n)))])
            out["runs"][f"df{df}_lr{lr}"] = {
                "iter": it.tolist(),
                "analysis": err.tolist(),
                "cost": list(map(float, h["loss"])),
            }
            print(
                f"df={df} lr={lr:<5g} 2k {at(2000):.4f} | 8k {at(8000):.4f} | "
                f"50k {err[-1]:.4f} | best {err.min():.4f} at "
                f"{int(it[int(np.argmin(err))])}",
                flush=True,
            )
            a.json.parent.mkdir(parents=True, exist_ok=True)
            a.json.write_text(json.dumps(out, indent=2))
    fig, ax = plt.subplots(
        1, len(a.deltas), figsize=(6.4 * len(a.deltas), 4.6), squeeze=False
    )
    cmap = plt.cm.plasma(np.linspace(0, 0.8, len(a.lrs)))
    for c, df in enumerate(a.deltas):
        for col, lr in zip(cmap, a.lrs):
            r = out["runs"][f"df{df}_lr{lr}"]
            ax[0, c].loglog(
                r["iter"], r["analysis"], color=col, lw=1.7, label=f"lr={lr}"
            )
        ax[0, c].axhline(
            SDA[df], color="#762a83", ls="--", lw=1.6, label=f"SDA ({SDA[df]})"
        )
        ax[0, c].axvline(8000, color="0.6", ls=":", lw=1.2)
        ax[0, c].set(
            xlabel="optimisation step",
            ylabel=r"analysis rel-$L_2$ at $t_0$",
            title=f"$\\delta_f={df}$, KAE $d_z$={int(D)}",
        )
        ax[0, c].grid(alpha=0.3, which="both")
        ax[0, c].legend(fontsize=8)
    fig.suptitle(
        "Does a larger step size help at a long budget? "
        "(dotted line: the campaign's 8000 iterations)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    o = a.out_dir / "ks_lr_sweep.png"
    fig.savefig(o, dpi=150, bbox_inches="tight")
    print("wrote", o)


if __name__ == "__main__":
    main()
