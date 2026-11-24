"""All three methods at ONE wall-clock budget, set by the KAE at a stated iteration count.

Iterations, rollout steps and sampler steps buy different things, so the only honest way to
ask "who is best for the same compute" is to fix seconds. The budget is whatever the KAE
takes for --kae-iters; the U-Net is then given as many 4D-Var iterations as fit, and SDA as
many predictor steps as fit, each calibrated on a short timing run on the same GPU.

A method that cannot run at all inside the budget is reported as such: SDA has a floor cost
of one full reverse diffusion, and below that floor it has no answer, which is a property
worth stating rather than hiding.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_assimilation.ks.exp_geometry import schedule
from data_assimilation.ks.protocol import DT, build_problem
from data_assimilation.ks.da_ks_experiments_3way import Bench, add_common_args


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    ap.add_argument(
        "--json",
        type=Path,
        default=Path("da_results_geometry_df9/equal_wallclock.json"),
    )
    ap.add_argument("--delta-f", type=int, nargs="+", default=[1, 9])
    ap.add_argument("--kae-iters", type=int, default=8000)
    ap.add_argument("--n-problems", type=int, default=16)
    ap.add_argument("--calib-iters", type=int, default=100)
    ap.add_argument("--calib-steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=43)
    a = ap.parse_args()
    b = Bench(a)
    rows = []
    for df in a.delta_f:
        fr = schedule(df, 25, 5)
        prob = build_problem(
            b.data, name=f"EW{df}", n_problems=a.n_problems, taus=fr * DT, seed=a.seed
        )
        # 1. the budget: whatever the KAE costs at the stated iteration count
        r, T = timed(lambda: b.run("KAE-expm", prob, iters=a.kae_iters, seed=a.seed))
        v = np.asarray(r["rel"], dtype=float)
        rows.append(
            {
                "delta_f": df,
                "method": "KAE",
                "budget": a.kae_iters,
                "budget_kind": "iterations",
                "wall_s": T,
                "mean": float(v.mean()),
                "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
            }
        )
        print(
            f"df={df} BUDGET = {T:.1f}s (KAE {a.kae_iters} iters), rel {v.mean():.4f}",
            flush=True,
        )

        # 2. U-Net: as many iterations as fit, from a short calibration
        _, t_cal = timed(lambda: b.run("UNet", prob, iters=a.calib_iters, seed=a.seed))
        per_iter = t_cal / a.calib_iters
        n_it = max(1, int(T / per_iter))
        r, t_u = timed(lambda: b.run("UNet", prob, iters=n_it, seed=a.seed))
        v = np.asarray(r["rel"], dtype=float)
        rows.append(
            {
                "delta_f": df,
                "method": "UNet",
                "budget": n_it,
                "budget_kind": "iterations",
                "wall_s": t_u,
                "mean": float(v.mean()),
                "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
                "calib_s_per_iter": per_iter,
            }
        )
        print(
            f"df={df} U-Net fits {n_it} iterations in the budget ({t_u:.1f}s), "
            f"rel {v.mean():.4f}",
            flush=True,
        )

        # 3. SDA: as many predictor steps as fit
        base = b.sda_cfg["n_steps"]
        b.sda_cfg["n_steps"] = a.calib_steps
        _, t_cal = timed(lambda: b.run("SDA", prob, iters=0, seed=a.seed))
        per_step = t_cal / a.calib_steps
        n_steps = int(T / per_step)
        if n_steps < 4:
            rows.append(
                {
                    "delta_f": df,
                    "method": "SDA",
                    "budget": None,
                    "budget_kind": "predictor steps",
                    "wall_s": None,
                    "mean": None,
                    "sem": None,
                    "calib_s_per_step": per_step,
                    "note": "cannot run inside the budget: fewer than 4 predictor steps",
                }
            )
            print(
                f"df={df} SDA does NOT fit in the budget ({per_step:.2f}s per step)",
                flush=True,
            )
        else:
            b.sda_cfg["n_steps"] = n_steps
            r, t_s = timed(lambda: b.run("SDA", prob, iters=0, seed=a.seed))
            v = np.asarray(r["rel"], dtype=float)
            rows.append(
                {
                    "delta_f": df,
                    "method": "SDA",
                    "budget": n_steps,
                    "budget_kind": "predictor steps",
                    "wall_s": t_s,
                    "mean": float(np.nanmean(v)),
                    "sem": float(np.nanstd(v, ddof=1) / np.sqrt(v.size)),
                    "n_diverged": int((~np.isfinite(v)).sum()),
                    "calib_s_per_step": per_step,
                }
            )
            print(
                f"df={df} SDA fits {n_steps} predictor steps ({t_s:.1f}s), "
                f"rel {np.nanmean(v):.4f}",
                flush=True,
            )
        b.sda_cfg["n_steps"] = base
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(
            json.dumps(
                {
                    "meta": {
                        "kae_iters": a.kae_iters,
                        "n_problems": a.n_problems,
                        "seed": a.seed,
                        "corrections": b.sda_cfg["corrections"],
                        "draws": a.n_samples,
                        "note": "budget = KAE wall clock at kae_iters; run with the GPU "
                        "otherwise idle",
                    },
                    "rows": rows,
                },
                indent=2,
            )
        )

    # figure: one group of bars per delta_f
    fig, ax = plt.subplots(
        1, len(a.delta_f), figsize=(5.6 * len(a.delta_f), 4.4), squeeze=False
    )
    for c, df in enumerate(a.delta_f):
        sel = [r for r in rows if r["delta_f"] == df]
        names = [r["method"] for r in sel]
        vals = [r["mean"] if r["mean"] is not None else np.nan for r in sel]
        cols = {"KAE": "#1b7837", "UNet": "#d6604d", "SDA": "#762a83"}
        bars = ax[0, c].bar(
            names, vals, color=[cols[n] for n in names], alpha=0.9, edgecolor="0.3"
        )
        for bar, r in zip(bars, sel):
            lab = (
                "does not fit"
                if r["mean"] is None
                else f"{r['mean']:.4f}\n{r['budget']} {r['budget_kind'].split()[0]}\n"
                f"{r['wall_s']:.0f}s"
            )
            ax[0, c].text(
                bar.get_x() + bar.get_width() / 2,
                (r["mean"] if r["mean"] else 1e-3) * 1.05,
                lab,
                ha="center",
                fontsize=7.5,
            )
        budget = sel[0]["wall_s"]
        ax[0, c].set(
            yscale="log",
            ylabel=r"analysis rel-$L_2$ at $t_0$",
            title=f"$\\delta_f={df}$: equal wall clock ({budget:.0f}s)",
        )
        ax[0, c].grid(alpha=0.3, axis="y", which="both")
    fig.suptitle(
        f"Same compute for every method: the budget is the KAE at "
        f"{a.kae_iters} iterations, $n={a.n_problems}$",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = a.out_dir / "ks_equal_wallclock.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
