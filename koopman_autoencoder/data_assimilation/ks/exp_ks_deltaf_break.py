# mypy: disable-error-code="arg-type, index"
"""Push delta_f until assimilation stops working.

The window AFTER the first observation is held fixed (16 frames, N = 5), so the only thing
that changes is how far the first observation sits from t_0. Two references are recorded
with every point: climatology (predict zero, rel-L2 = 1 by construction) and the error of
simply copying the nearest observed state. A method is "broken" once it is no better than
those.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_assimilation.ks.exp_geometry import schedule
from data_assimilation.ks.protocol import DT, build_problem, rel_l2
from data_assimilation.ks.da_ks_experiments_3way import ALL, Bench, add_common_args

STYLE = {
    "KAE-expm": dict(color="#1b7837", label="KAE"),
    "KAE-rk4": dict(color="#7fbc41", label="KAE / RK4"),
    "UNet": dict(color="#d6604d", label="U-Net 4D-Var"),
    "SDA": dict(color="#762a83", label="SDA"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    ap.add_argument(
        "--json", type=Path, default=Path("da_results_geometry_df9/deltaf_break.json")
    )
    ap.add_argument(
        "--deltas", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    )
    ap.add_argument(
        "--span",
        type=int,
        default=16,
        help="frames between the first and last observation, held fixed",
    )
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--n-problems", type=int, default=24)
    ap.add_argument("--seed", type=int, default=91)
    a = ap.parse_args()
    b = Bench(a)
    rows, store = [], {}
    for df in a.deltas:
        fr = schedule(df, df + a.span, 5)
        prob = build_problem(
            b.data, name=f"B{df}", n_problems=a.n_problems, taus=fr * DT, seed=a.seed
        )
        sim = torch.as_tensor(prob.sim, device=b.dev)
        t0 = torch.as_tensor(prob.t0, device=b.dev)
        u0 = b.data.denorm(b.data.frames(sim, t0))
        near = b.data.denorm(b.data.frames(sim, t0 + int(fr[0])))
        row = {
            "delta_f": int(df),
            "frames": list(map(int, fr)),
            "N": int(len(fr)),
            "copy_nearest": float(rel_l2(near, u0).mean()),
            "climatology": float(rel_l2(torch.zeros_like(u0), u0).mean()),
        }
        per_problem = {}
        for m in ALL:
            r = b.run(m, prob, iters=a.iters, seed=a.seed)
            v = np.asarray(r["rel"], dtype=float)
            per_problem[f"df{df}__{m}"] = v
            row[m] = {
                "mean": float(np.nanmean(v)),
                "sem": float(np.nanstd(v, ddof=1) / np.sqrt(v.size)),
                "n_diverged": int((~np.isfinite(v)).sum()),
            }
        store.update(per_problem)
        np.savez_compressed(a.json.with_suffix(".npz"), **store)
        print(
            f"df={df:3d} frames={list(map(int, fr))} copy={row['copy_nearest']:.3f} "
            + " ".join(f"{m}={row[m]['mean']:.4f}" for m in ALL),
            flush=True,
        )
        rows.append(row)
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(
            json.dumps(
                {
                    "meta": {
                        "span": a.span,
                        "iters": a.iters,
                        "n_problems": a.n_problems,
                        "seed": a.seed,
                    },
                    "rows": rows,
                },
                indent=2,
            )
        )

    x = [r["delta_f"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for m, st in STYLE.items():
        ax.errorbar(
            x,
            [r[m]["mean"] for r in rows],
            yerr=[r[m]["sem"] for r in rows],
            lw=1.9,
            marker="o",
            ms=4.5,
            capsize=3,
            **st,
        )
    ax.plot(
        x,
        [r["copy_nearest"] for r in rows],
        color="0.45",
        ls="-.",
        lw=1.5,
        label="copy the nearest observation",
    )
    ax.axhline(1.0, color="crimson", ls=":", lw=1.6, label="climatology")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"$\delta_f$ (frames to the first observation)",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title=f"Pushing $\\delta_f$ until assimilation breaks\n"
        f"window after the first observation fixed at {a.span} frames, $N=5$, "
        f"{a.iters} iterations, $n={a.n_problems}$",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = a.out_dir / "ks_deltaf_break.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
