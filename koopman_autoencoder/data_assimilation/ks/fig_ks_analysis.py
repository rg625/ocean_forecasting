# ruff: noqa: E731
"""KS qualitative figure: the chaotic trajectory as context, the analysis as the claim.

Two things have to be shown at once and they need different time axes.

  The REGIME is only visibly chaotic over hundreds of frames: over one assimilation
  window KS barely turns (path length within 1.3x a straight line for every candidate),
  so a panel restricted to the window looks laminar and says nothing about the problem
  being hard.

  The RESULT is a single frame.  Analysis error is defined at t_0 and nowhere else, and a
  long panel is dominated by free-running forecast skill -- which is the axis the U-Net
  wins and the one this paper argues is the wrong one to rank assimilation by.  Drawn over
  250 frames the U-Net's panel is indistinguishable from the truth while its analysis is
  the worst of the three.

So: the truth over a long window on the left, with t_0 and the observation times marked,
and the analysis at t_0 on the right, where the methods are actually separated.

Trajectories are chosen by state-space path length over the long window -- a criterion
that never looks at which method wins -- and the per-example errors are printed so the
figure can be checked against the sweep it is supposed to illustrate.

    python -m data_assimilation.ks.fig_ks_analysis --delta-f 9 --k 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_assimilation.ks.exp_geometry import schedule
from data_assimilation.ks.protocol import DT, Problem, build_problem
from data_assimilation.ks.da_ks_experiments_3way import Bench, add_common_args

STYLE = {
    "KAE-expm": ("#1b7837", "KAE"),
    "UNet": ("#d6604d", "U-Net 4D-Var"),
    "SDA": ("#762a83", "SDA (one draw)"),
}


def problem_for(data, sims, t0s, taus, name):
    """A Problem on CHOSEN (trajectory, t_0) pairs, clean and fully observed.

    build_problem draws its own trajectories, so the selected ones are assembled here
    instead; the observations are read off the record exactly as it would.
    """
    taus = np.asarray(taus, dtype=float)
    sim, t0 = np.asarray(sims), np.asarray(t0s)
    sim_t = torch.as_tensor(sim, device=data.device)
    y = np.zeros((len(taus), len(sim), data.X), dtype=np.float32)
    for i, tau in enumerate(taus):
        k = int(round(tau / DT))
        y[i] = (
            data.frames(sim_t, torch.as_tensor(t0 + k, device=data.device))
            .cpu()
            .numpy()
        )
    return Problem(
        name=name,
        data_path=data.path,
        sim=sim,
        t0=t0,
        taus=taus,
        obs_frac=1.0,
        noise_std=0.0,
        noise_dist="gaussian",
        seed=0,
        forecast_taus=np.array([]),
        on_grid=True,
        mask=np.ones((len(taus), data.X), dtype=np.float32),
        y=y,
    )


def pick_chaotic(b, window, n_cand, max_tau, seed, k):
    """The k trajectories whose state-space path turns most over `window` frames."""
    prob = build_problem(
        b.data, name="cand", n_problems=n_cand, taus=np.array([max_tau]) * DT, seed=seed
    )
    ok = np.where(prob.t0 + window - 1 < b.data.n_t)[0]
    sim = torch.as_tensor(prob.sim[ok], device=b.dev)
    t0 = torch.as_tensor(prob.t0[ok], device=b.dev)
    u = np.stack(
        [
            b.data.denorm(b.data.frames(sim, t0 + j)).cpu().numpy()
            for j in range(window)
        ],
        axis=1,
    )
    nrm = lambda a: np.linalg.norm(a, axis=-1)
    drift = nrm(u[:, -1] - u[:, 0]) / nrm(u[:, 0])
    wig = nrm(np.diff(u, axis=1)).sum(1) / nrm(u[:, 0]) / np.maximum(drift, 1e-6)
    order = np.argsort(-wig)[:k]
    return (prob.sim[ok][order], prob.t0[ok][order], wig[order])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    ap.add_argument("--stem", default="ks_analysis_df")
    ap.add_argument("--delta-f", type=int, default=9)
    ap.add_argument("--delta-l", type=int, default=25)
    ap.add_argument("--n-obs", type=int, default=5)
    ap.add_argument(
        "--window",
        type=int,
        default=250,
        help="frames of truth drawn as context (NOT rolled out by any method)",
    )
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument(
        "--kae-tuning",
        type=Path,
        default=Path("da_results_geometry_df9/kae_tuning.json"),
    )
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    b = Bench(a)
    if a.kae_tuning and a.kae_tuning.exists():
        b.hp["KAE-expm"].update(json.loads(a.kae_tuning.read_text())["best"])
        print(f"KAE settings: {b.hp['KAE-expm']}", flush=True)
    x = b.data.x
    fr = schedule(a.delta_f, a.delta_l, a.n_obs)
    sims, t0s, wig = pick_chaotic(b, a.window, a.candidates, a.window - 1, a.seed, a.k)
    print(
        f"schedule {fr.tolist()}; picked {list(zip(sims.tolist(), t0s.tolist()))} "
        f"path/straight-line {np.round(wig, 1).tolist()}",
        flush=True,
    )

    prob = problem_for(b.data, sims, t0s, fr * DT, "Q")

    res, store = {}, {"frames": fr, "sim": sims, "t0": t0s, "wiggle": wig}
    for m in STYLE:
        r = b.run(m, prob, iters=a.iters, seed=a.seed)
        res[m] = (np.asarray(r["analysis"]), np.asarray(r["rel"], dtype=float))
        store[f"{m}__analysis"], store[f"{m}__rel"] = res[m]
        print(
            f"  {m:9s} rel at t0 " + "  ".join(f"{v:.4f}" for v in res[m][1]),
            flush=True,
        )

    sim_t = torch.as_tensor(sims, device=b.dev)
    t0_t = torch.as_tensor(t0s, device=b.dev)
    truth_win = (
        b.data.denorm(
            torch.stack([b.data.frames(sim_t, t0_t + j) for j in range(a.window)])
        )
        .cpu()
        .numpy()
    )
    truth0 = truth_win[0]
    store["truth_window"] = truth_win
    np.savez_compressed(a.out_dir / f"{a.stem}{a.delta_f}.npz", **store)

    taus = np.arange(a.window) * DT
    for j in range(a.k):
        fig, ax = plt.subplots(
            1, 3, figsize=(15.2, 3.6), gridspec_kw={"width_ratios": [1.5, 1.15, 1.15]}
        )
        v = float(np.abs(truth_win[:, j]).max())
        im = ax[0].imshow(
            truth_win[:, j].T,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-v,
            vmax=v,
            extent=[0, taus[-1], float(x.min()), float(x.max())],
        )
        ax[0].axvline(0, color="k", lw=2.0)
        for f_ in fr:
            ax[0].axvline(f_ * DT, color="w", ls=":", lw=1.1)
        ax[0].set(
            xlabel="time",
            ylabel="x",
            title=f"truth, {a.window} frames "
            f"(path {wig[j]:.0f}$\\times$ a straight line)",
        )
        fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02)

        ax[1].plot(x, truth0[j], color="k", lw=2.4, label="truth", zorder=5)
        for m, (col, lab) in STYLE.items():
            ax[1].plot(
                x, res[m][0][j], color=col, lw=1.5, label=f"{lab}  {res[m][1][j]:.3f}"
            )
        ax[1].set(xlabel="x", ylabel="$u(x,t_0)$", title="analysis at $t_0$")
        ax[1].legend(fontsize=7.5, loc="best")
        ax[1].grid(alpha=0.3)

        for m, (col, lab) in STYLE.items():
            ax[2].plot(x, res[m][0][j] - truth0[j], color=col, lw=1.5, label=lab)
        ax[2].axhline(0, color="k", lw=0.8)
        ax[2].set(
            xlabel="x", ylabel="analysis $-$ truth", title="analysis error at $t_0$"
        )
        ax[2].grid(alpha=0.3)
        fig.tight_layout()
        p = a.out_dir / f"{a.stem}{a.delta_f}_ex{j}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
