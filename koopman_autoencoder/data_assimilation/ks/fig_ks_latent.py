"""Recovered u(t_0) on KS for every trained KAE latent width, at two observation schedules.

Same problems and the same budget for every row; only the checkpoint's latent width changes.
SDA is drawn for reference because it is the method the KAE has to beat once the first
observation moves away from t_0.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_assimilation.ks.evaluate import ae_reconstruction_floor
from data_assimilation.ks.methods import KAE4DVar, SolveConfig
from data_assimilation.ks.models_io import load_kae
from data_assimilation.ks.protocol import DT, build_problem, rel_l2
from data_assimilation.ks.da_ks_experiments_3way import Bench, add_common_args

SCHEDULES = {1: [1, 3, 7, 15, 25], 9: [9, 12, 15, 19, 25]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    ap.add_argument("--n-problems", type=int, default=6)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--latent-root", default="model_outputs_ks_latentdim")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    b = Bench(a)
    x = b.data.x
    # only runs that actually finished: an aborted run has no final_model.pth, and
    # load_kae would silently fall back to the default architecture and then fail
    runs = {
        int(p.split("dz_")[1].split("/")[0]): Path(p)
        for p in sorted(glob.glob(f"{a.latent_root}/dz_*/run-*"))
        if (Path(p) / "final_model.pth").is_file()
        and (Path(p) / "checkpoints" / "best_model.pth").is_file()
    }
    dzs = sorted(runs)
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(dzs)))

    for df, frames in SCHEDULES.items():
        prob = build_problem(
            b.data,
            name=f"L{df}",
            n_problems=a.n_problems,
            taus=np.array(frames) * DT,
            seed=a.seed,
        )
        sim = torch.as_tensor(prob.sim, device=b.dev)
        t0 = torch.as_tensor(prob.t0, device=b.dev)
        truth = b.data.denorm(b.data.frames(sim, t0)).cpu().numpy()
        curves, rels, floors = {}, {}, {}
        for dz in dzs:
            kae, K, D = load_kae(runs[dz], None, b.dev)
            m = KAE4DVar(kae, K, D, 1.0, "expm")
            r = m.solve(
                prob,
                b.data,
                SolveConfig(iters=a.iters, lr=0.01, init="zero", seed=a.seed),
            )
            with torch.no_grad():
                ana = b.data.denorm(m.analysis_field(r["control"]))
                curves[dz] = ana.cpu().numpy()
                rels[dz] = (
                    rel_l2(ana, b.data.denorm(b.data.frames(sim, t0))).cpu().numpy()
                )
                floors[dz] = float(ae_reconstruction_floor(kae, b.data, prob).mean())
            print(
                f"df={df} d_z={dz:5d}  rel-L2 {rels[dz].mean():.4f}  "
                f"(AE round trip {floors[dz]:.4f})",
                flush=True,
            )
            del kae, K, m
            torch.cuda.empty_cache()
        sda = b.run("SDA", prob, iters=a.iters, seed=a.seed)
        print(f"df={df} SDA          rel-L2 {np.mean(sda['rel']):.4f}", flush=True)

        n = a.n_problems
        ncol = 3
        nrow = int(np.ceil(n / ncol))
        fig, ax = plt.subplots(
            nrow, ncol, figsize=(4.4 * ncol, 2.7 * nrow), sharex=True
        )
        ax = np.atleast_1d(ax).ravel()
        for j in range(n):
            ax[j].plot(
                x,
                truth[j],
                lw=6,
                color="k",
                alpha=0.18,
                solid_capstyle="round",
                zorder=1,
            )
            for c, dz in zip(cmap, dzs):
                ax[j].plot(x, curves[dz][j], lw=1.4, color=c, zorder=3)
            ax[j].plot(
                x, sda["analysis"][j], lw=1.3, color="crimson", ls="--", zorder=4
            )
            ax[j].set_title(
                "  ".join(f"{dz}:{rels[dz][j]:.3f}" for dz in dzs)
                + f"   SDA:{sda['rel'][j]:.3f}",
                fontsize=7.5,
            )
            ax[j].set_xlabel("x")
        for j in range(n, len(ax)):
            ax[j].axis("off")
        h = [plt.Line2D([], [], color="k", lw=5, alpha=0.3, label="truth $u(t_0)$")]
        h += [
            plt.Line2D([], [], color=c, lw=1.6, label=f"KAE $d_z$={dz}")
            for c, dz in zip(cmap, dzs)
        ]
        h += [plt.Line2D([], [], color="crimson", lw=1.5, ls="--", label="SDA")]
        fig.legend(
            handles=h,
            loc="lower center",
            ncol=len(h),
            frameon=False,
            bbox_to_anchor=(0.5, -0.03),
        )
        means = "   ".join(f"{dz}: {rels[dz].mean():.3f}" for dz in dzs)
        fig.suptitle(
            f"Recovered $u(t_0)$ by KAE latent width, observations at "
            f"$t_0+{frames}$ ($\\delta_f={df}$)\n"
            f"panel titles are rel-$L_2$ per $d_z$;  means:  {means}   "
            f"SDA: {np.mean(sda['rel']):.3f}",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.04, 1, 0.92])
        out = a.out_dir / f"ks_latent_recovery_df{df}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
