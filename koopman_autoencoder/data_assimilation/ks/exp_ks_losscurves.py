"""The 4D-Var objective against optimisation step, per method and per schedule.

Both the cost being minimised and the analysis error are recorded, because they do not have
to move together: a cost that keeps falling while the analysis gets worse is the signature
of an objective whose minimum is in the wrong place. SDA has no optimisation loop and is
drawn as a horizontal reference instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_assimilation.ks.exp_geometry import schedule
from data_assimilation.ks.protocol import DT, build_problem
from data_assimilation.ks.da_ks_experiments_3way import Bench, add_common_args

STYLE = {
    "KAE-expm": dict(color="#1b7837", label="KAE"),
    "UNet": dict(color="#d6604d", label="U-Net 4D-Var"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    ap.add_argument(
        "--json", type=Path, default=Path("da_results_geometry_df9/loss_curves.json")
    )
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 9])
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--n-problems", type=int, default=16)
    ap.add_argument("--track-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=43)
    a = ap.parse_args()
    b = Bench(a)
    out, sda_ref = {}, {}
    for df in a.deltas:
        fr = schedule(df, 25, 5)
        prob = build_problem(
            b.data, name=f"L{df}", n_problems=a.n_problems, taus=fr * DT, seed=a.seed
        )
        for m in STYLE:
            r = b.run(m, prob, iters=a.iters, seed=a.seed, track_every=a.track_every)
            h = r["hist"]
            out[f"df{df}_{m}"] = {
                k: list(map(float, h[k]))
                for k in ("iter", "loss", "init_rel_l2", "obs_rel_l2")
            }
            print(
                f"df={df} {m:9s} loss {h['loss'][0]:.4g} -> {h['loss'][-1]:.4g} | "
                f"analysis {h['init_rel_l2'][0]:.4f} -> {h['init_rel_l2'][-1]:.4f} | "
                f"best {min(h['init_rel_l2']):.4f} at iter "
                f"{h['iter'][int(np.argmin(h['init_rel_l2']))]}",
                flush=True,
            )
        s = b.run("SDA", prob, iters=a.iters, seed=a.seed)
        sda_ref[df] = float(np.mean(s["rel"]))
        print(f"df={df} SDA (no optimisation) analysis {sda_ref[df]:.4f}", flush=True)
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(
        json.dumps(
            {
                "curves": out,
                "sda": sda_ref,
                "meta": {"iters": a.iters, "n": a.n_problems},
            },
            indent=2,
        )
    )

    fig, ax = plt.subplots(
        2, len(a.deltas), figsize=(6.2 * len(a.deltas), 7), squeeze=False
    )
    for c, df in enumerate(a.deltas):
        for m, st in STYLE.items():
            d = out[f"df{df}_{m}"]
            ax[0, c].loglog(d["iter"][1:], d["loss"][1:], lw=1.8, **st)
            ax[1, c].loglog(d["iter"][1:], d["init_rel_l2"][1:], lw=1.8, **st)
        ax[1, c].axhline(
            sda_ref[df],
            color="#762a83",
            ls="--",
            lw=1.6,
            label=f"SDA (no optimiser) {sda_ref[df]:.4f}",
        )
        ax[0, c].set(
            title=f"$\\delta_f={df}$: 4D-Var objective",
            xlabel="optimisation step",
            ylabel="cost",
        )
        ax[1, c].set(
            title=f"$\\delta_f={df}$: analysis error",
            xlabel="optimisation step",
            ylabel=r"rel-$L_2$ at $t_0$",
        )
        for r in (0, 1):
            ax[r, c].grid(alpha=0.3, which="both")
            ax[r, c].legend(fontsize=8)
    fig.tight_layout()
    out_png = a.out_dir / "ks_loss_curves.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("wrote", out_png)


if __name__ == "__main__":
    main()
