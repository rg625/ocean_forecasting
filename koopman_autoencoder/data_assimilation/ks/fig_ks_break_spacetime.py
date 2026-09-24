# ruff: noqa: E731
"""The break sweep as fields: one chaotic trajectory, several delta_f, all three methods.

`exp_ks_deltaf_break` reports how the analysis error grows as the first observation moves
away from t_0. This shows the same thing as trajectories, so the failure can be seen rather
than read off a curve. The SAME trajectory and t_0 are used at every delta_f -- the problem
is built directly for that pair instead of redrawing -- so the panels are comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_assimilation.ks.exp_geometry import schedule
from data_assimilation.ks.protocol import DT, Problem, build_problem
from data_assimilation.ks.da_ks_experiments_3way import Bench, add_common_args

STYLE = {
    "KAE-expm": ("#1b7837", "KAE"),
    "UNet": ("#d6604d", "U-Net 4D-Var"),
    "SDA": ("#762a83", "SDA"),
}


def problem_for(data, sim_i, t0_i, taus, name):
    """build_problem for ONE fixed (trajectory, t_0), clean and fully observed."""
    taus = np.asarray(taus, dtype=float)
    sim = np.array([sim_i])
    t0 = np.array([t0_i])
    sim_t = torch.as_tensor(sim, device=data.device)
    y = np.zeros((len(taus), 1, data.X), dtype=np.float32)
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


def pick_chaotic(b, window, n_cand, max_tau, seed):
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
    i = int(np.argmax(wig))
    return int(prob.sim[ok][i]), int(prob.t0[ok][i]), float(drift[i]), float(wig[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    ap.add_argument(
        "--npz", type=Path, default=Path("da_results_geometry_df9/break_spacetime.npz")
    )
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 8, 24, 48])
    ap.add_argument("--span", type=int, default=16)
    ap.add_argument(
        "--pin-delta-l",
        type=int,
        default=0,
        help="pin delta_l to this many frames (the C1 protocol) instead of "
        "holding the span after the first observation fixed",
    )
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument(
        "--kae-tuning",
        type=Path,
        default=Path("da_results_geometry_df9/kae_tuning.json"),
        help="frozen KAE hyper-parameters (lr and first guess) to draw the "
        "panels with, so they match the tables; '' keeps the campaign's",
    )
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    b = Bench(a)
    if a.kae_tuning and Path(a.kae_tuning).exists():
        b.hp["KAE-expm"].update(json.loads(Path(a.kae_tuning).read_text())["best"])
        print(
            f"KAE hyper-parameters from {a.kae_tuning}: {b.hp['KAE-expm']}", flush=True
        )
    x = b.data.x
    sched = (
        (lambda df: schedule(df, a.pin_delta_l, 5))
        if a.pin_delta_l
        else (lambda df: schedule(df, df + a.span, 5))
    )
    max_tau = a.pin_delta_l if a.pin_delta_l else max(a.deltas) + a.span
    sim_i, t0_i, drift, wig = pick_chaotic(b, a.window, a.candidates, max_tau, a.seed)
    print(
        f"trajectory {sim_i}, t0 frame {t0_i} (drift {drift:.2f}, path {wig:.0f}x)",
        flush=True,
    )
    sim = torch.as_tensor([sim_i], device=b.dev)
    t0 = torch.as_tensor([t0_i], device=b.dev)
    W = a.window
    tru = (
        b.data.denorm(torch.stack([b.data.frames(sim, t0 + k) for k in range(W)]))[:, 0]
        .cpu()
        .numpy()
    )
    taus_plot = np.arange(W) * DT
    store = {
        "taus": taus_plot,
        "truth": tru,
        "sim": np.array(sim_i),
        "t0": np.array(t0_i),
    }

    for df in a.deltas:
        fr = sched(df)
        prob = problem_for(b.data, sim_i, t0_i, fr * DT, f"BS{df}")
        fields = {}
        for m in STYLE:
            r = b.run(m, prob, iters=a.iters, seed=a.seed, window_frames=W)
            st = r["draws"][0][0] if "draws" in r else r["spacetime"][:, 0]
            fields[m] = np.asarray(st)
            store[f"df{df}__{m}"] = fields[m]
            print(
                f"df={df:3d} {m:9s} rel at t0 {float(np.asarray(r['rel'])[0]):.4f}",
                flush=True,
            )
        errs = {m: np.abs(f - tru) for m, f in fields.items()}
        emax = max(
            float(
                np.percentile(np.concatenate([e.ravel() for e in errs.values()]), 99.5)
            ),
            1e-6,
        )
        vmax = float(np.abs(tru).max())
        ext = [0, taus_plot[-1], float(x.min()), float(x.max())]
        fig, ax = plt.subplots(2, 4, figsize=(16.5, 5.8))
        ax[0, 0].imshow(
            tru.T,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=ext,
        )
        ax[0, 0].set(title="TRUTH", ylabel="x")
        for c, m in enumerate(STYLE, start=1):
            col, lab = STYLE[m]
            ax[0, c].imshow(
                fields[m].T,
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                extent=ext,
            )
            ax[0, c].set_title(lab + (" (one draw)" if m == "SDA" else ""), color=col)
            im = ax[1, c].imshow(
                errs[m].T,
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=0,
                vmax=emax,
                extent=ext,
            )
            rel = np.linalg.norm(fields[m] - tru, axis=-1) / np.linalg.norm(
                tru, axis=-1
            )
            ax[1, c].set(
                title=f"|error|, rel-$L_2$ at $t_0$ = {rel[0]:.3f}", xlabel="time"
            )
            fig.colorbar(im, ax=ax[1, c], fraction=0.046, pad=0.02)
            ax[1, 0].semilogy(
                taus_plot, np.maximum(rel, 1e-9), color=col, lw=1.6, label=lab
            )
            store[f"df{df}__{m}__rel"] = rel
        for f_ in fr:
            for r_ in range(2):
                for c_ in range(4):
                    if ax[r_, c_].has_data():
                        ax[r_, c_].axvline(f_ * DT, color="w", ls=":", lw=1.0)
        for c_ in range(1, 4):
            ax[0, c_].axvline(fr[-1] * DT, color="lime", lw=1.5)
            ax[1, c_].axvline(fr[-1] * DT, color="lime", lw=1.5)
        ax[1, 0].set(
            xlabel="time",
            ylabel=r"rel-$L_2$ per frame",
            title="error against time (log)",
        )
        ax[1, 0].grid(alpha=0.3, which="both")
        ax[1, 0].legend(fontsize=7.5)
        for c_ in range(4):
            ax[0, c_].set_xlabel("time")
        fig.suptitle(
            f"$\\delta_f={df}$, observations at $t_0+{list(map(int, fr))}$ "
            f"(trajectory {sim_i}, $t_0$ frame {t0_i}, path {wig:.0f}$\\times$ a "
            f"straight line). Green line: last observation; error panels share one "
            f"scale (0 to {emax:.2f})",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        tag = "fine" if a.pin_delta_l else "break"
        out = a.out_dir / f"ks_{tag}_spacetime_df{df}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out, flush=True)
    a.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.npz, **store)
    print("saved ->", a.npz)


if __name__ == "__main__":
    main()
