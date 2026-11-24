"""How far does more optimisation take the d_z=512 KAE at delta_f=9?

One long solve with the analysis error tracked along the way, so the whole budget curve
comes from a single run. The question is whether the 4D-Var analysis ever reaches SDA's
level, which SDA attains with no optimisation at all.
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

SDA_DF9 = 0.0192  # measured, same schedule, no optimiser


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--kae-run", default="model_outputs_ks_latentdim/dz_512/run-20260906_140222"
    )
    ap.add_argument("--test", type=Path, default=Path("data/ks/da_test.nc"))
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    ap.add_argument(
        "--json", type=Path, default=Path("da_results_geometry_df9/kae512_budget.json")
    )
    ap.add_argument("--delta-f", type=int, default=9)
    ap.add_argument("--iters", type=int, default=200_000)
    ap.add_argument("--track-every", type=int, default=250)
    ap.add_argument("--n-problems", type=int, default=48)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    data = KSData(a.test, dev)
    fr = schedule(a.delta_f, 25, 5)
    prob = build_problem(
        data, name="BUD", n_problems=a.n_problems, taus=fr * DT, seed=a.seed
    )
    kae, K, D = load_kae(Path(a.kae_run), None, dev)
    m = KAE4DVar(kae, K, D, 1.0, "expm")
    print(
        f"d_z={D} schedule={fr.tolist()} n={a.n_problems} iters={a.iters}", flush=True
    )
    res = m.solve(
        prob,
        data,
        SolveConfig(
            iters=a.iters, lr=a.lr, init="zero", seed=a.seed, track_every=a.track_every
        ),
    )
    h = res["hist"]
    it = np.asarray(h["iter"], dtype=float)
    err = np.asarray(h["init_rel_l2"], dtype=float)
    cost = np.asarray(h["loss"], dtype=float)
    # power-law fit on the last decade, which is what an extrapolation would use
    tail = it >= it.max() / 10
    slope, inter = np.polyfit(np.log(it[tail]), np.log(err[tail]), 1)
    need = (
        float(np.exp((np.log(SDA_DF9) - inter) / slope)) if slope < 0 else float("inf")
    )
    ms_per_iter = res["ms_per_iter"]
    out = {
        "meta": {
            "kae_run": a.kae_run,
            "d_z": int(D),
            "delta_f": a.delta_f,
            "frames": fr.tolist(),
            "n_problems": a.n_problems,
            "lr": a.lr,
            "iters": a.iters,
            "ms_per_iter": ms_per_iter,
            "sda_reference": SDA_DF9,
        },
        "iter": it.tolist(),
        "analysis_rel_l2": err.tolist(),
        "cost": cost.tolist(),
        "tail_power_law": {
            "slope": float(slope),
            "intercept": float(inter),
            "iters_to_reach_sda": need,
            "hours_to_reach_sda": need * ms_per_iter / 1e3 / 3600,
        },
    }
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(out, indent=2))
    print(
        f"final analysis {err[-1]:.4f} at {int(it[-1])} iters "
        f"({ms_per_iter:.2f} ms/iter, {a.iters * ms_per_iter / 1e3 / 60:.1f} min)"
    )
    print(
        f"tail power law: err ~ iters^{slope:.3f}; reaching SDA ({SDA_DF9}) needs "
        f"{need:.3g} iterations = {need * ms_per_iter / 1e3 / 3600:.3g} GPU-hours"
    )

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))
    ax[0].loglog(it, err, color="#1b7837", lw=1.9, label=f"KAE $d_z$={D}")
    ax[0].axhline(
        SDA_DF9,
        color="#762a83",
        ls="--",
        lw=1.7,
        label=f"SDA, no optimiser ({SDA_DF9})",
    )
    xf = np.logspace(np.log10(it[tail][0]), np.log10(max(need, it.max()) * 1.2), 50)
    ax[0].loglog(
        xf,
        np.exp(inter) * xf**slope,
        ":",
        color="0.4",
        lw=1.4,
        label=f"tail fit, slope {slope:.2f}",
    )
    ax[0].set(
        xlabel="optimisation step",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title=f"$\\delta_f={a.delta_f}$: does more optimisation close the gap?",
    )
    ax[1].loglog(it, cost, color="#1b7837", lw=1.9)
    ax[1].set(
        xlabel="optimisation step", ylabel="4D-Var cost", title="the objective itself"
    )
    for x_ in ax:
        x_.grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    out_png = a.out_dir / "ks_kae512_budget.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("wrote", out_png)


if __name__ == "__main__":
    main()
