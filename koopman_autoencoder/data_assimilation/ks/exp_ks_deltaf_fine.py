"""delta_f from 1 to 20, one frame at a time, everything else fixed.

The C1 protocol: delta_l = 25 frames and N = 5 stay pinned, so the only thing that moves is
how far the first observation sits from t_0. Every method runs at the configuration frozen
on validation -- the KAE and the U-Net each with their own tuned learning rate and
initialisation, SDA at its frozen sampler settings -- so no method is tuned on these points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_assimilation.ks.exp_geometry import schedule
from data_assimilation.ks.init_utils import init_kwargs
from data_assimilation.ks.methods import KAE4DVar, SolveConfig, UNet4DVar
from data_assimilation.ks.protocol import DT, build_problem, rel_l2
from data_assimilation.ks.da_ks_experiments_3way import Bench, add_common_args

STYLE = {
    "KAE": ("#1b7837", "KAE"),
    "UNet": ("#d6604d", "U-Net 4D-Var"),
    "SDA": ("#762a83", "SDA"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument(
        "--kae-run-512", default="model_outputs_ks_latentdim/dz_512/run-20260906_140222"
    )
    ap.add_argument(
        "--kae-tuning",
        type=Path,
        default=Path("da_results_geometry_df9/kae_tuning.json"),
    )
    ap.add_argument(
        "--unet-tuning",
        type=Path,
        default=Path("da_results_geometry_df9/unet_tuning.json"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    ap.add_argument(
        "--json", type=Path, default=Path("da_results_geometry_df9/deltaf_fine.json")
    )
    ap.add_argument("--deltas", type=int, nargs="+", default=list(range(1, 21)))
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["KAE", "SDA", "UNet"],
        help="run these in this order; results merge into the same json, so "
        "one method at a time still produces a usable file and figure",
    )
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--n-problems", type=int, default=24)
    ap.add_argument("--seed", type=int, default=101)
    a = ap.parse_args()
    b = Bench(a)
    dev = b.dev
    from data_assimilation.ks.models_io import load_kae

    kae, K, D = load_kae(Path(a.kae_run_512), None, dev)
    kae_hp = {"lr": 0.01, "init": "zero"}
    if a.kae_tuning.is_file():
        kae_hp.update(json.loads(a.kae_tuning.read_text())["best"])
    unet_hp = {"lr": 0.01, "init": "zero"}
    if a.unet_tuning.is_file():
        unet_hp.update(json.loads(a.unet_tuning.read_text())["best"])
    print(f"frozen: KAE {kae_hp} | U-Net {unet_hp} | methods {a.methods}", flush=True)
    by_df = {}
    if a.json.is_file():  # merge with earlier legs of the same sweep
        for r in json.loads(a.json.read_text())["rows"]:
            by_df[int(r["delta_f"])] = r

    def save():
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(
            json.dumps(
                {
                    "meta": {
                        "iters": a.iters,
                        "n_problems": a.n_problems,
                        "seed": a.seed,
                        "kae_hp": kae_hp,
                        "unet_hp": unet_hp,
                        "delta_l_frames": 25,
                        "N": 5,
                    },
                    "rows": [by_df[k] for k in sorted(by_df)],
                },
                indent=2,
            )
        )

    for meth in a.methods:  # one method at a time, in the order given
        for df in a.deltas:
            fr = schedule(df, 25, 5)
            prob = build_problem(
                b.data,
                name=f"F{df}",
                n_problems=a.n_problems,
                taus=fr * DT,
                seed=a.seed,
            )
            sim = torch.as_tensor(prob.sim, device=dev)
            t0 = torch.as_tensor(prob.t0, device=dev)
            u0 = b.data.frames(sim, t0)
            row = by_df.setdefault(
                int(df),
                {"delta_f": int(df), "frames": list(map(int, fr)), "N": int(len(fr))},
            )
            if "copy_nearest" not in row:
                near = b.data.frames(sim, t0 + int(fr[0]))
                row["copy_nearest"] = float(
                    rel_l2(b.data.denorm(near), b.data.denorm(u0)).mean()
                )
            if meth == "KAE":
                m = KAE4DVar(kae, K, D, 1.0, "expm")
                ik = init_kwargs(kae_hp["init"], kae, K, prob, dev)
                r = m.solve(
                    prob,
                    b.data,
                    SolveConfig(iters=a.iters, lr=kae_hp["lr"], seed=a.seed, **ik),
                )
                with torch.no_grad():
                    v = (
                        rel_l2(
                            b.data.denorm(m.analysis_field(r["control"])),
                            b.data.denorm(u0),
                        )
                        .cpu()
                        .numpy()
                    )
            elif meth == "UNet":
                mu = UNet4DVar(b.unet, init_scale=b.x_scale, checkpoint_every=-1)
                i = int(np.argmin(prob.taus))
                uik = (
                    {
                        "init": "given",
                        "init_value": torch.as_tensor(prob.y[i], device=dev),
                    }
                    if unet_hp["init"] == "obs"
                    else {"init": unet_hp["init"]}
                )
                r = mu.solve(
                    prob,
                    b.data,
                    SolveConfig(iters=a.iters, lr=unet_hp["lr"], seed=a.seed, **uik),
                )
                with torch.no_grad():
                    v = (
                        rel_l2(
                            b.data.denorm(mu.analysis_field(r["control"])),
                            b.data.denorm(u0),
                        )
                        .cpu()
                        .numpy()
                    )
            else:
                v = np.asarray(
                    b.run("SDA", prob, iters=a.iters, seed=a.seed)["rel"], dtype=float
                )
            row[meth] = {
                "mean": float(np.nanmean(v)),
                "sem": float(np.nanstd(v, ddof=1) / np.sqrt(v.size)),
                "n_diverged": int((~np.isfinite(v)).sum()),
            }
            print(
                f"{meth:5s} df={df:2d} {row[meth]['mean']:.4f} "
                f"(copy {row['copy_nearest']:.4f})",
                flush=True,
            )
            save()
        print(f"=== {meth} leg complete ===", flush=True)
    rows = [by_df[k] for k in sorted(by_df)]

    x = [r["delta_f"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    for k, (col, lab) in STYLE.items():
        pts = [r for r in rows if k in r]
        if not pts:
            continue
        ax.errorbar(
            [r["delta_f"] for r in pts],
            [r[k]["mean"] for r in pts],
            yerr=[r[k]["sem"] for r in pts],
            color=col,
            label=lab,
            lw=1.9,
            marker="o",
            ms=4,
            capsize=2.5,
        )
    ax.plot(
        x,
        [r["copy_nearest"] for r in rows],
        color="0.45",
        ls="-.",
        lw=1.5,
        label="copy the nearest observation",
    )
    ax.axhline(1.0, color="crimson", ls=":", lw=1.5, label="climatology")
    ax.set(
        yscale="log",
        xlabel=r"$\delta_f$  (frames to the first observation)",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        xticks=x[::2],
        title=f"$\\delta_f$ one frame at a time, $\\delta_l=25$ and $N=5$ pinned\n"
        f"{a.iters} iterations, $n={a.n_problems}$, settings frozen on validation",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = a.out_dir / "ks_deltaf_fine.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
