# ruff: noqa: E741
# mypy: disable-error-code="index"
"""KS recovered state at t_0: four examples, one row, all methods overlaid per panel.

A companion to the TRA field gallery.  A KS state is 64 numbers on a line, so where TRA
gets an image per method this gets a single axes per example with truth and all three
analyses drawn on it -- four panels across, which reads at the same width as the TRA
figure and puts the methods in direct contact rather than in separate columns.

SELECTION.  Candidates are the trajectories whose state-space path turns most over 250
frames -- a criterion that never looks at which method wins.  All of them are solved, and
by default the four panels are those with the WIDEST SDA/KAE separation, because a
difference that cannot be seen on the page illustrates nothing.  That is selection on the
outcome, so the script prints every candidate's gap, the median, and the percentile of
each panel shown; the caption must carry those numbers and ``--select span`` samples the
distribution instead.

    python -m data_assimilation.ks.fig_ks_fields_qual --delta-f 18 --candidates 12 --show 4
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
from data_assimilation.ks.fig_ks_analysis import pick_chaotic, problem_for
from data_assimilation.ks.protocol import DT
from data_assimilation.ks.da_ks_experiments_3way import Bench, add_common_args

STYLE = [
    ("KAE-expm", "KAE", "#1b7837"),
    ("UNet", "U-Net", "#e41a1c"),
    ("SDA", "SDA", "#6a3d9a"),
]
INK = "#2b2b2b"

# How the two reference curves are drawn.  The target is the hard case: where a method is
# good it lies exactly on top of the truth and hides it, so a plain line underneath cannot
# work.  Each palette solves that differently -- see --style.
PALETTES = {
    # a wide soft ribbon behind everything: methods ride inside it, and leaving it is the
    # error.  Nothing is occluded because the truth is wider than any line drawn on it.
    "band": dict(
        truth=dict(
            color="#4d4d4d", lw=7.0, alpha=0.22, zorder=2, solid_capstyle="round"
        ),
        truth_core=dict(color="#2b2b2b", lw=1.1, alpha=0.9, zorder=2.5),
        given=dict(color="#0f6ba8", lw=1.8, ls=(0, (5, 2.5)), alpha=0.85, zorder=1.5),
        method_lw=1.8,
        method_alpha=1.0,
    ),
    # the ribbon, but with the given in a warm hue: the blue of "band" sits next to SDA's
    # purple, and on a 64-point line two cool curves are easy to confuse
    "ribbon": dict(
        truth=dict(
            color="#4d4d4d", lw=7.0, alpha=0.22, zorder=2, solid_capstyle="round"
        ),
        truth_core=dict(color="#2b2b2b", lw=1.1, alpha=0.9, zorder=2.5),
        given=dict(color="#d95f02", lw=1.8, ls=(0, (5, 2.5)), alpha=0.85, zorder=1.5),
        method_lw=1.8,
        method_alpha=1.0,
    ),
    # the truth drawn LAST, in black, dashed, so the method underneath shows through the
    # gaps and the reference is never covered
    "ontop": dict(
        truth=dict(color="#111111", lw=2.3, ls=(0, (6, 2)), alpha=1.0, zorder=5),
        truth_core=None,
        given=dict(color="#d95f02", lw=1.7, ls=(0, (2, 2)), alpha=0.9, zorder=1.5),
        method_lw=2.0,
        method_alpha=0.95,
    ),
    # heavy charcoal truth on top but translucent, so it reads as a reference the methods
    # are seen through rather than as a fourth curve
    "bold": dict(
        truth=dict(
            color="#000000", lw=4.0, alpha=0.45, zorder=5, solid_capstyle="round"
        ),
        truth_core=None,
        given=dict(color="#7570b3", lw=2.0, ls=(0, (4, 2)), alpha=0.9, zorder=1.5),
        method_lw=1.7,
        method_alpha=1.0,
    ),
    # colour-blind-safe Okabe-Ito for the methods, so the truth can keep pure black and
    # the given a strong orange without any of the five colliding
    "okabe": dict(
        truth=dict(
            color="#000000", lw=3.0, alpha=1.0, zorder=2, solid_capstyle="round"
        ),
        truth_core=None,
        given=dict(color="#E69F00", lw=2.0, ls=(0, (5, 2)), alpha=1.0, zorder=1.5),
        method_lw=1.7,
        method_alpha=1.0,
        methods={"KAE-expm": "#009E73", "UNet": "#D55E00", "SDA": "#0072B2"},
    ),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--out", type=Path, default=Path("figs/ks_fields_qual.png"))
    ap.add_argument(
        "--delta-f",
        type=int,
        default=18,
        help="18 matches the TRA gallery's last row, where the gap is widest",
    )
    ap.add_argument("--delta-l", type=int, default=25)
    ap.add_argument("--n-obs", type=int, default=5)
    ap.add_argument(
        "--candidates",
        type=int,
        default=24,
        help="chaotic trajectories solved; the shown panels are drawn from these",
    )
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument(
        "--select",
        choices=["gap", "sda", "span"],
        default="gap",
        help="'gap' takes the widest SDA/KAE separations, so the difference "
        "is legible; 'span' samples the distribution instead. Either "
        "way every candidate's gap and the shown percentiles are printed",
    )
    ap.add_argument(
        "--pool", type=int, default=200, help="trajectories screened for chaos"
    )
    ap.add_argument(
        "--prescreen",
        type=int,
        default=0,
        help="if >0, rank this many chaotic candidates by the target/given gap "
        "and solve only the --candidates widest. The ranking uses the truth "
        "and the observation only, never a solve, so it stays blind to which "
        "method wins -- it just stops the budget going on panels where the "
        "given field already looks like the target",
    )
    ap.add_argument("--select-window", type=int, default=250)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument(
        "--kae-tuning",
        type=Path,
        default=Path("da_results_geometry_df9/kae_tuning.json"),
    )
    ap.add_argument(
        "--style",
        choices=sorted(PALETTES),
        default="band",
        help="how the target and the given are drawn; see PALETTES",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=Path("figs/cache/ks_fields_qual.npz"),
        help="solved analyses are reused from here, so a palette can be "
        "redrawn without re-running the solvers; delete it to re-solve",
    )
    a = ap.parse_args()

    # the solve depends on these and nothing else, so a cache written under this key can
    # be reused by every palette; anything else about the run may change freely
    key_args = dict(
        delta_f=a.delta_f,
        delta_l=a.delta_l,
        n_obs=a.n_obs,
        seed=a.seed,
        iters=a.iters,
        pool=a.pool,
        candidates=a.candidates,
        select_window=a.select_window,
        prescreen=a.prescreen,
    )
    cached = None
    if a.cache and a.cache.is_file():
        z = np.load(a.cache, allow_pickle=True)
        if json.loads(str(z["key"])) == key_args:
            cached = z
            print(f"reusing {a.cache}", flush=True)
        else:
            print(f"{a.cache} was built for other settings; re-solving", flush=True)

    if cached is not None:
        x, truth, obs, sims, t0s = (
            cached[k] for k in ("x", "truth", "obs", "sims", "t0s")
        )
        res = {k: (cached[f"{k}__ana"], cached[f"{k}__rel"]) for k, _, _ in STYLE}
    else:
        b = Bench(a)
        if a.kae_tuning and a.kae_tuning.exists():
            b.hp["KAE-expm"].update(json.loads(a.kae_tuning.read_text())["best"])
            print(f"KAE settings: {b.hp['KAE-expm']}", flush=True)
        x = b.data.x
        fr = schedule(a.delta_f, a.delta_l, a.n_obs)
        n_chaotic = max(a.prescreen, a.candidates)
        sims, t0s, wig = pick_chaotic(
            b, a.select_window, a.pool, a.select_window - 1, a.seed, n_chaotic
        )

        def build(sims, t0s):
            prob = problem_for(b.data, sims, t0s, fr * DT, "Q")
            truth = (
                b.data.denorm(
                    b.data.frames(
                        torch.as_tensor(sims, device=b.dev),
                        torch.as_tensor(t0s, device=b.dev),
                    )
                )
                .cpu()
                .numpy()
            )
            jmin = int(np.argmin(prob.taus))
            obs = (
                b.data.denorm(torch.as_tensor(prob.y[jmin], device=b.dev)).cpu().numpy()
            )
            return prob, truth, obs

        prob, truth, obs = build(sims, t0s)
        if a.prescreen > a.candidates:
            gap = np.linalg.norm(obs - truth, axis=1) / np.linalg.norm(truth, axis=1)
            keep = np.argsort(-gap)[: a.candidates]
            print(
                f"  prescreened {len(gap)} chaotic candidates on target/given gap: "
                f"min {gap.min():.3f}, median {np.median(gap):.3f}, max {gap.max():.3f}"
                f"; keeping {a.candidates} above {gap[keep].min():.3f}",
                flush=True,
            )
            sims, t0s = sims[keep], t0s[keep]
            prob, truth, obs = build(sims, t0s)

        res = {}
        for k, name, _ in STYLE:
            r = b.run(k, prob, iters=a.iters, seed=a.seed)
            res[k] = (np.asarray(r["analysis"]), np.asarray(r["rel"], dtype=float))
            print(
                f"  {name:6s} rel " + " ".join(f"{v:.4f}" for v in res[k][1]),
                flush=True,
            )
        if a.cache:
            a.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                a.cache,
                key=json.dumps(key_args),
                x=x,
                truth=truth,
                obs=obs,
                sims=sims,
                t0s=t0s,
                **{
                    f"{k}__{w}": res[k][i]
                    for k, _, _ in STYLE
                    for i, w in enumerate(("ana", "rel"))
                },
            )
            print("cached to", a.cache, flush=True)

    gap = res["SDA"][1] / res["KAE-expm"][1]
    order = np.argsort(gap)
    print(
        "\n  SDA/KAE gap over the %d candidates: median %.2f, range %.2f-%.2f"
        % (len(gap), float(np.median(gap)), float(gap.min()), float(gap.max()))
    )
    if a.select == "span":
        picks = [
            int(order[int(round(q * (len(order) - 1)))])
            for q in np.linspace(0.25, 1.0, a.show)
        ]
    elif a.select == "sda":
        # rank by SDA's own error: what has to be legible is SDA leaving the truth, and
        # the ratio can be wide because the KAE is very good rather than SDA very bad
        picks = [int(i) for i in np.argsort(-res["SDA"][1])[: a.show]]
    else:  # widest gaps
        picks = [int(i) for i in order[::-1][: a.show]]
        picks.sort(key=lambda j: -gap[j])
    for j in picks:
        pct = 100.0 * (np.sum(gap <= gap[j]) - 1) / max(len(gap) - 1, 1)
        print(
            f"    shown: trajectory {int(sims[j])} t0 {int(t0s[j])}  "
            f"gap {gap[j]:.2f}  ({pct:.0f}th percentile)"
        )

    pal = PALETTES[a.style]
    cols = {k: c for k, _, c in STYLE} | pal.get("methods", {})
    fig, axes = plt.subplots(
        1, len(picks), figsize=(3.55 * len(picks), 2.9), sharey=True
    )
    axes = np.atleast_1d(axes)
    for c, j in enumerate(picks):
        ax = axes[c]
        # what the methods are actually handed: the nearest observation, delta_f frames
        # after t_0. The distance between this and the truth is the problem.
        ax.plot(x, obs[j], label=f"given: $u(t_0{{+}}{a.delta_f})$", **pal["given"])
        ax.plot(x, truth[j], label="target: $u(t_0)$", **pal["truth"])
        if pal["truth_core"]:
            ax.plot(x, truth[j], **pal["truth_core"])
        for key, name, _ in STYLE:
            ax.plot(
                x,
                res[key][0][j],
                color=cols[key],
                lw=pal["method_lw"],
                alpha=pal["method_alpha"],
                label=name,
                zorder=3,
            )
        ax.set_title(
            "  ".join(
                (
                    f"$\\bf{{{res[k][1][j]:.3f}}}$"
                    if k == "KAE-expm"
                    else f"{res[k][1][j]:.3f}"
                )
                for k, _, _ in STYLE
            ),
            fontsize=8.5,
            color=INK,
        )
        ax.set_xlabel("$x$", fontsize=9)
        ax.grid(alpha=0.18)
        ax.tick_params(labelsize=8, colors="#666666")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#cccccc")
    axes[0].set_ylabel("$u(x,\\,t_0)$", fontsize=9, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        ncol=5,
        fontsize=9,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
    )
    fig.suptitle(
        f"rel-$L_2$ at $t_0$ per panel, in legend order   "
        f"($\\delta_f={a.delta_f}$)",
        fontsize=8.5,
        color="#888888",
        y=-0.02,
    )
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", a.out, f"[style {a.style}]")


if __name__ == "__main__":
    main()
