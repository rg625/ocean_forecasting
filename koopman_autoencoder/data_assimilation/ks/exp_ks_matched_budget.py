# mypy: disable-error-code="var-annotated"
"""KAE 4D-Var against SDA at MATCHED WALL CLOCK, at delta_f = 9.

Iterations and sampler steps are not comparable units, so both methods are swept over their
own quality knob and plotted against measured seconds: the 4D-Var budget is the iteration
count, SDA's is the number of predictor steps (its corrections and draws are held fixed).
This is the fair version of "give the KAE more compute": if SDA still wins at equal cost,
the gap is not a budget artefact.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    ap.add_argument(
        "--json", type=Path, default=Path("da_results_geometry_df9/matched_budget.json")
    )
    ap.add_argument("--delta-f", type=int, default=9)
    ap.add_argument(
        "--kae-iters", type=int, nargs="+", default=[1000, 4000, 16000, 64000, 200000]
    )
    ap.add_argument("--sda-steps", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--n-problems", type=int, default=16)
    ap.add_argument("--seed", type=int, default=43)
    a = ap.parse_args()
    b = Bench(a)
    fr = schedule(a.delta_f, 25, 5)
    prob = build_problem(
        b.data, name="MB", n_problems=a.n_problems, taus=fr * DT, seed=a.seed
    )
    rows = []

    def flush():
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(
            json.dumps(
                {
                    "meta": {
                        "delta_f": a.delta_f,
                        "frames": fr.tolist(),
                        "n_problems": a.n_problems,
                        "seed": a.seed,
                        "sda_corrections": b.sda_cfg["corrections"],
                        "sda_draws": a.n_samples,
                        "note": "wall clock measured on the same GPU; other jobs may share it",
                    },
                    "rows": rows,
                },
                indent=2,
            )
        )

    for it in a.kae_iters:
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = b.run("KAE-expm", prob, iters=it, seed=a.seed)
        torch.cuda.synchronize()
        v = np.asarray(r["rel"], dtype=float)
        rows.append(
            {
                "method": "KAE",
                "budget": it,
                "budget_kind": "iterations",
                "wall_s": time.perf_counter() - t,
                "mean": float(v.mean()),
                "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
            }
        )
        print(
            f"KAE  {it:7d} iters  {rows[-1]['wall_s']:7.1f}s  rel {v.mean():.4f}",
            flush=True,
        )
        flush()

    base = b.sda_cfg["n_steps"]
    for ns in a.sda_steps:
        b.sda_cfg["n_steps"] = ns
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = b.run("SDA", prob, iters=0, seed=a.seed)
        torch.cuda.synchronize()
        v = np.asarray(r["rel"], dtype=float)
        rows.append(
            {
                "method": "SDA",
                "budget": ns,
                "budget_kind": "predictor steps",
                "wall_s": time.perf_counter() - t,
                "mean": float(np.nanmean(v)),
                "sem": float(np.nanstd(v, ddof=1) / np.sqrt(v.size)),
                "n_diverged": int((~np.isfinite(v)).sum()),
            }
        )
        print(
            f"SDA  {ns:7d} steps  {rows[-1]['wall_s']:7.1f}s  rel {np.nanmean(v):.4f}",
            flush=True,
        )
        flush()
    b.sda_cfg["n_steps"] = base

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for meth, col in (("KAE", "#1b7837"), ("SDA", "#762a83")):
        pts = [r for r in rows if r["method"] == meth]
        ax.errorbar(
            [p["wall_s"] for p in pts],
            [p["mean"] for p in pts],
            yerr=[p["sem"] for p in pts],
            color=col,
            lw=1.9,
            marker="o",
            ms=5,
            capsize=3,
            label=meth,
        )
        for p in pts:
            ax.annotate(
                str(p["budget"]),
                (p["wall_s"], p["mean"]),
                fontsize=6.5,
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                color=col,
            )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="wall clock for the whole batch (s)",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title=f"Matched budget at $\\delta_f={a.delta_f}$ ($n={a.n_problems}$)\n"
        "annotations: 4D-Var iterations / SDA predictor steps",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = a.out_dir / "ks_matched_budget.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
