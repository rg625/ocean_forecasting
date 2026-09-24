# ruff: noqa: E731
"""Qualitative KS recovery: the state at t_0 that each method returns.

One figure per observation schedule, so the effect of moving the first observation away
from t_0 can be read off the fields rather than from a table. A space-time figure shows the
same solves over the whole window, with the observed times marked: the methods agree with
the truth where it was observed and differ where it was not.
"""

from __future__ import annotations

import argparse
import json
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_assimilation.ks.protocol import DT, build_problem
from data_assimilation.ks.da_ks_experiments_3way import Bench, add_common_args

STYLE = {
    "KAE-expm": dict(color="#1b7837", ls="-", label="KAE"),
    "UNet": dict(color="#d6604d", ls="-", label="U-Net 4D-Var"),
    "SDA": dict(color="#762a83", ls="-", label="SDA"),
}
METHODS = list(STYLE)
SCHEDULES = {1: [1, 3, 7, 15, 25], 4: [4, 6, 10, 16, 25], 9: [9, 12, 15, 19, 25]}


def pick_active(b, frames, n_candidates, window, k, seed, metric="wiggle"):
    """The k most interesting problems, by one of three measures of the truth's motion.

    drift   net displacement over the window. Rewards a steadily translating wave, which
            is the smooth, monotonic picture that says nothing about chaos.
    wiggle  path length in state space over net displacement. 1.0 is a straight line;
            large values mean the trajectory turns, which is what cell merging and
            splitting look like.
    events  how often the number of local maxima changes, i.e. how many merge/split
            events the window contains.

    Over the 2.5 t.u. assimilation window none of these help: KS moves smoothly there
    (0.11 Lyapunov times, wiggle <= 1.3 over 200 candidates). A longer ``window`` is what
    makes the difference, at the cost of showing mostly free-running continuation.
    """
    prob = build_problem(
        b.data,
        name="cand",
        n_problems=n_candidates,
        taus=np.array(frames) * DT,
        seed=seed,
    )
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    # a window longer than the assimilation horizon can run off the end of the record
    ok = np.where(prob.t0 + window - 1 < b.data.n_t)[0]
    assert len(ok), f"no candidate leaves room for a {window}-frame window"
    sim, t0 = sim[ok], t0[ok]
    u = np.stack(
        [
            b.data.denorm(b.data.frames(sim, t0 + j)).cpu().numpy()
            for j in range(window)
        ],
        axis=1,
    )  # [B, window, X]
    nrm = lambda a: np.linalg.norm(a, axis=-1)
    drift = nrm(u[:, -1] - u[:, 0]) / nrm(u[:, 0])
    path = nrm(np.diff(u, axis=1)).sum(1) / nrm(u[:, 0])
    wiggle = path / np.maximum(drift, 1e-6)
    nmax = lambda f: ((f > np.roll(f, 1, -1)) & (f > np.roll(f, -1, -1))).sum(-1)
    events = (
        (np.diff(np.stack([nmax(u[:, j]) for j in range(window)], 1), axis=1) != 0)
        .sum(1)
        .astype(float)
    )
    score = {"drift": drift, "wiggle": wiggle, "events": events}[metric]
    order = np.argsort(score)[::-1][:k]
    idx, drift = ok[order], drift[order]
    sub = dataclasses.replace(
        prob,
        name=f"{prob.name}[active]",
        sim=prob.sim[idx],
        t0=prob.t0[idx],
        y=prob.y[:, idx],
    )
    return sub, drift


def solve_all(b, frames, n_problems, iters, seed, window=0, problem=None):
    prob = (
        problem
        if problem is not None
        else build_problem(
            b.data,
            name=f"G{frames[0]}",
            n_problems=n_problems,
            taus=np.array(frames) * DT,
            seed=seed,
        )
    )
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    out = {"truth": b.data.denorm(b.data.frames(sim, t0)).cpu().numpy(), "prob": prob}
    if window:
        out["truth_st"] = np.stack(
            [
                b.data.denorm(b.data.frames(sim, t0 + k)).cpu().numpy()
                for k in range(window)
            ]
        )
    for m in METHODS:
        r = b.run(m, prob, iters=iters, seed=seed, window_frames=window or None)
        out[m] = {"analysis": r["analysis"], "rel": np.asarray(r["rel"])}
        if window:
            # SDA's "spacetime" is the mean over posterior draws. Where observations pin
            # the state the draws agree and the mean is meaningful; far from them they
            # diverge and their mean washes out into a faded field that no single draw
            # looks like. Plot one draw instead, which is also how SDA is scored.
            if "draws" in r:
                out[m]["spacetime"] = np.transpose(r["draws"][0], (1, 0, 2))[:window]
                out[m]["is_draw"] = True
            else:
                out[m]["spacetime"] = r["spacetime"]
    return out


def fig_gallery(res, x, df, path, ncol=3):
    n = res["truth"].shape[0]
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow), sharex=True)
    ax = np.atleast_1d(ax).ravel()
    for j in range(n):
        ax[j].plot(
            x,
            res["truth"][j],
            lw=5.5,
            color="k",
            alpha=0.20,
            solid_capstyle="round",
            zorder=1,
        )
        for m in METHODS:
            ax[j].plot(x, res[m]["analysis"][j], lw=1.5, zorder=3, **STYLE[m])
        ax[j].set_title(
            "  ".join(
                f"{STYLE[m]['label'].split()[0]} {res[m]['rel'][j]:.3f}"
                for m in METHODS
            ),
            fontsize=8,
        )
        ax[j].set_xlabel("x")
    for j in range(n, len(ax)):
        ax[j].axis("off")
    h = [plt.Line2D([], [], color="k", lw=5, alpha=0.3, label="truth $u(t_0)$")]
    h += [
        plt.Line2D([], [], lw=1.6, **{k: v for k, v in STYLE[m].items()})
        for m in METHODS
    ]
    fig.legend(
        handles=h,
        loc="lower center",
        ncol=len(h),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    means = "   ".join(
        f"{STYLE[m]['label'].split()[0]} {res[m]['rel'].mean():.3f}" for m in METHODS
    )
    fig.suptitle(
        f"Recovered $u(t_0)$, observations at $t_0+{SCHEDULES[df]}$ frames "
        f"($\\delta_f={df}$)\npanel titles are rel-$L_2$;  means:  {means}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig_spacetime(res, x, df, path, example=0, window=26, note="", suptitle=True):
    frames = SCHEDULES[df]
    t = np.arange(window) * DT
    fig, ax = plt.subplots(2, 4, figsize=(15, 5.4))
    tru = res["truth_st"][:, example]
    vmax = float(np.abs(tru).max())
    # the error scale must come from the errors: at 0.6 * max|u| every method except
    # U-Net renders as black, which hides the comparison the panel exists to make
    emax = max(
        float(
            np.nanpercentile(
                np.concatenate(
                    [
                        np.abs(res[m]["spacetime"][:, example] - tru).ravel()
                        for m in METHODS
                    ]
                ),
                99.5,
            )
        ),
        1e-6,
    )
    drift = float(np.linalg.norm(tru[-1] - tru[0]) / np.linalg.norm(tru[0]))
    panels = [("TRUTH", None, tru)] + [
        (
            STYLE[m]["label"] + (" (one draw)" if res[m].get("is_draw") else ""),
            m,
            res[m]["spacetime"][:, example],
        )
        for m in METHODS
    ]
    for c, (name, key, fld) in enumerate(panels):
        ax[0, c].imshow(
            fld.T,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=[0, t[-1], float(x.min()), float(x.max())],
        )
        ax[0, c].set_title(name, fontsize=10)
        if c:
            e = np.abs(fld - tru)
            im2 = ax[1, c].imshow(
                e.T,
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=0,
                vmax=emax,
                extent=[0, t[-1], float(x.min()), float(x.max())],
            )
            fig.colorbar(im2, ax=ax[1, c], fraction=0.046, pad=0.02)
            ax[1, c].set_title(
                f"|error|,  rel-$L_2$ at $t_0$ = " f"{res[key]['rel'][example]:.3f}",
                fontsize=9,
            )
        else:
            for m in METHODS:
                e = res[m]["spacetime"][:, example] - tru
                rel = np.linalg.norm(e, axis=-1) / np.maximum(
                    np.linalg.norm(tru, axis=-1), 1e-12
                )
                ax[1, c].semilogy(t, rel, lw=1.6, **{k: v for k, v in STYLE[m].items()})
            for f in frames:
                ax[1, c].axvline(f * DT, color="0.6", ls=":", lw=1)
            ax[1, c].axvline(frames[-1] * DT, color="green", lw=1.4)
            ax[1, c].set(
                xlabel="time",
                ylabel="rel-$L_2$ per frame",
                title="error against time (log)",
            )
            ax[1, c].grid(alpha=0.3, which="both")
            ax[1, c].legend(fontsize=6.5)
        for a in (ax[0, c], ax[1, c]):
            if a.has_data():
                for f in frames:
                    a.axvline(f * DT, color="w", ls=":", lw=1.1)
                if window - 1 > frames[-1] * 1.5:  # mark where observations stop
                    a.axvline(frames[-1] * DT, color="lime", lw=1.6)
                a.set_xlabel("time")
    ax[0, 0].set_ylabel("x")
    ax[1, 1].set_ylabel("x")
    if suptitle:
        fig.suptitle(
            f"Same solve over the whole window, $\\delta_f={df}$.{note} Dotted lines "
            f"are the observed times, $t_0$ is the left edge. The truth itself moves "
            f"by {drift:.2f} rel-$L_2$ over the window; error panels share one scale "
            f"(0 to {emax:.2f})",
            fontsize=10.5,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


STYLE_INV = {v["label"]: k for k, v in STYLE.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    ap.add_argument("--n-problems", type=int, default=6)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 4, 9])
    ap.add_argument("--spacetime-df", type=int, default=9)
    ap.add_argument(
        "--spacetime-metric",
        choices=["drift", "wiggle", "events"],
        default="wiggle",
        help="how 'active' is scored; see pick_active",
    )
    ap.add_argument(
        "--spacetime-select",
        choices=["first", "active"],
        default="first",
        help="'active' picks the trajectories whose truth moves most over the "
        "window, instead of the first problems drawn",
    )
    ap.add_argument(
        "--spacetime-k",
        type=int,
        default=3,
        help="how many examples to draw when selecting by activity",
    )
    ap.add_argument("--spacetime-candidates", type=int, default=32)
    ap.add_argument("--spacetime-window", type=int, default=26)
    ap.add_argument(
        "--select-window",
        type=int,
        default=0,
        help="window (frames) used to SELECT the trajectory, when it should "
        "differ from the one displayed. KS barely turns over an "
        "assimilation window, so picking a chaotic trajectory needs "
        "hundreds of frames even when only tens are drawn. 0 = same",
    )
    ap.add_argument(
        "--no-suptitle",
        action="store_true",
        help="drop the in-figure title, for figures that carry a LaTeX caption",
    )
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument(
        "--kae-tuning",
        type=Path,
        default=Path("da_results_geometry_df9/kae_tuning.json"),
        help="frozen KAE hyper-parameters to draw with, so the panels match "
        "the tables; '' keeps the campaign's",
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
    for df in a.deltas:
        window = a.spacetime_window if df == a.spacetime_df else 0
        if window and a.spacetime_select == "active":
            sub, drift = pick_active(
                b,
                SCHEDULES[df],
                a.spacetime_candidates,
                a.select_window or window,
                a.spacetime_k,
                a.seed,
                metric=a.spacetime_metric,
            )
            print(
                f"most active of {a.spacetime_candidates} candidates: drift "
                + ", ".join(f"{d:.2f}" for d in drift),
                flush=True,
            )
            res = solve_all(
                b,
                SCHEDULES[df],
                len(drift),
                a.iters,
                a.seed,
                window=window,
                problem=sub,
            )
            for j, d in enumerate(drift):
                fig_spacetime(
                    res,
                    x,
                    df,
                    a.out_dir / f"ks_spacetime_df{df}_{a.spacetime_metric}{j}.png",
                    example=j,
                    window=window,
                    suptitle=not a.no_suptitle,
                    note=f" Trajectory {sub.sim[j]}, $t_0$ frame {sub.t0[j]}.",
                )
            continue
        res = solve_all(b, SCHEDULES[df], a.n_problems, a.iters, a.seed, window=window)
        fig_gallery(res, x, df, a.out_dir / f"ks_recovery_df{df}.png")
        if window:
            fig_spacetime(
                res,
                x,
                df,
                a.out_dir / f"ks_spacetime_df{df}.png",
                window=window,
                suptitle=not a.no_suptitle,
            )


if __name__ == "__main__":
    main()
