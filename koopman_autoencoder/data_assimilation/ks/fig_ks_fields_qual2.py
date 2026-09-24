# ruff: noqa: E731
"""KS recovered state at t_0: four examples, two rows.

Top row -- the GIVEN field, the earliest observation u(t_0 + delta_f), which is the only
thing any method was handed, drawn over the TARGET u(t_0).  That pair states the problem:
the distance between the two curves is what the inverse has to cover.  Bottom row -- the
same target with all three analyses on it, so the methods are in direct contact rather
than in separate columns, and so the answer is read after the question.

SELECTION.  Candidates are the trajectories whose state-space path turns most over 250
frames, a criterion that never looks at which method wins; they are the same pool and the
same solves as ``fig_ks_fields_qual`` and are read from its cache.  By default the four
panels are those with the LARGEST target/given gap, because a panel where the given field
already looks like the target illustrates nothing -- the reader cannot see that any
inference was required.  That criterion is still blind to which method wins.  ``--select
diverse`` restores farthest-point sampling over (gap, KAE error, sign changes,
best-baseline/KAE ratio), and ``--examples`` forces a choice by hand; either way the
script prints every candidate's numbers and each shown panel's percentile.

    python -m data_assimilation.ks.fig_ks_fields_qual2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = [
    ("KAE-expm", "KAE", "#1b7837"),
    ("UNet", "U-Net 4D-Var", "#e41a1c"),
    ("SDA", "SDA", "#6a3d9a"),
]
INK = "#2b2b2b"
TRUTH = dict(color="#4d4d4d", lw=7.0, alpha=0.22, zorder=2, solid_capstyle="round")
CORE = dict(color="#2b2b2b", lw=1.1, alpha=0.9, zorder=2.5)
GIVEN = dict(color="#d95f02", lw=2.0, ls=(0, (5, 2.5)), alpha=0.95, zorder=3)


def features(rel, truth, obs):
    """Standardised coordinates a 'behaviour' is judged different in."""
    gap = np.linalg.norm(obs - truth, axis=1) / np.linalg.norm(truth, axis=1)
    nz = (np.diff(np.sign(truth), axis=1) != 0).sum(1).astype(float)
    ratio = np.minimum(rel["UNet"], rel["SDA"]) / rel["KAE-expm"]
    f = np.stack([np.log(gap), np.log(rel["KAE-expm"]), nz, np.log(ratio)], 1)
    return gap, nz, ratio, (f - f.mean(0)) / f.std(0).clip(1e-9)


def farthest_point(f, k, start):
    """Greedy farthest-point sampling: maximally unlike each other, start fixed."""
    picks = [int(start)]
    while len(picks) < k:
        d = np.min(np.linalg.norm(f[:, None] - f[picks][None], axis=2), axis=1)
        d[picks] = -1.0
        picks.append(int(np.argmax(d)))
    return picks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("figs/cache/ks_fields_qual.npz"))
    ap.add_argument("--out", type=Path, default=Path("figs/ks_fields_qual2.png"))
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument(
        "--select",
        choices=["gap", "diverse"],
        default="gap",
        help="'gap' takes the widest target/given separations so the inference "
        "being asked for is visible; 'diverse' samples the pool instead",
    )
    ap.add_argument(
        "--examples",
        type=int,
        nargs="+",
        default=None,
        help="force these candidate indices instead of the diverse pick",
    )
    a = ap.parse_args()

    z = np.load(a.cache, allow_pickle=True)
    key = json.loads(str(z["key"]))
    df = int(key["delta_f"])
    x, truth, obs = z["x"], z["truth"], z["obs"]
    ana = {k: z[f"{k}__ana"] for k, _, _ in STYLE}
    rel = {k: z[f"{k}__rel"] for k, _, _ in STYLE}
    n = len(truth)

    gap, nz, ratio, f = features(rel, truth, obs)
    beats = int(np.sum(rel["KAE-expm"] < np.minimum(rel["UNet"], rel["SDA"])))
    print(
        f"  KAE beats BOTH baselines on {beats}/{n} candidates "
        f"(means: KAE {rel['KAE-expm'].mean():.4f}, SDA {rel['SDA'].mean():.4f}, "
        f"U-Net {rel['UNet'].mean():.4f})"
    )

    if a.examples is not None:
        picks = list(a.examples)
    elif a.select == "gap":
        picks = [int(j) for j in np.argsort(-gap)[: a.show]]
    else:
        picks = farthest_point(f, a.show, int(np.argmax(rel["KAE-expm"])))
    # left to right the given field drifts further from the target, widest last
    picks = sorted(picks, key=lambda j: gap[j])
    print(
        f"\n  {'i':>3} {'sim':>4} {'t0':>5} {'KAE':>7} {'SDA':>7} {'U-Net':>7}"
        f" {'ratio':>6} {'obsgap':>7} {'signs':>6}"
    )
    for i in range(n):
        print(
            f"  {i:>3} {int(z['sims'][i]):>4} {int(z['t0s'][i]):>5} "
            f"{rel['KAE-expm'][i]:7.4f} {rel['SDA'][i]:7.4f} {rel['UNet'][i]:7.4f} "
            f"{ratio[i]:6.2f} {gap[i]:7.3f} {int(nz[i]):>6}"
            + ("   <- shown" if i in picks else "")
        )
    for j in picks:
        q = lambda v: 100.0 * np.sum(v <= v[j]) / len(v)
        print(
            f"    panel {j}: KAE error at the {q(rel['KAE-expm']):.0f}th percentile "
            f"of the pool, observation gap at the {q(gap):.0f}th"
        )

    fig, axes = plt.subplots(
        2,
        len(picks),
        figsize=(3.40 * len(picks), 4.75),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    # the question is asked on the top row and answered on the bottom one
    for c, j in enumerate(picks):
        top, bot = axes[0][c], axes[1][c]
        for ax in (top, bot):
            ax.plot(x, truth[j], **TRUTH)
            ax.plot(x, truth[j], **CORE)
            ax.grid(alpha=0.18)
            ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
        top.plot(x, obs[j], label=f"given: $u(t_0{{+}}{df})$", **GIVEN)
        top.set_title(f"gap to target  {gap[j]:.3f}", fontsize=8.5, color="#a24a02")
        for k, name, col in STYLE:
            bot.plot(x, ana[k][j], color=col, lw=1.9, label=name, zorder=3)
        # one coloured number per curve, so a value cannot be read off the wrong method
        for i, (k, _, col) in enumerate(STYLE):
            bot.text(
                0.5 + (i - 1) * 0.30,
                1.045,
                (
                    f"$\\bf{{{rel[k][j]:.3f}}}$"
                    if k == "KAE-expm"
                    else f"{rel[k][j]:.3f}"
                ),
                transform=bot.transAxes,
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=col,
            )
        bot.set_xlabel("$x$", fontsize=9)
        top.tick_params(labelsize=8)
        bot.tick_params(labelsize=8)
    axes[0][0].set_ylabel(f"given: $u(t_0{{+}}{df})$", fontsize=9.5, color=INK)
    axes[1][0].set_ylabel("analysis at $t_0$", fontsize=9.5, color=INK)

    h = [
        plt.Line2D([], [], **{**TRUTH, "alpha": 0.45}),
        plt.Line2D([], [], **GIVEN),
    ] + [plt.Line2D([], [], color=c, lw=1.9) for _, _, c in STYLE]
    lab = ["target $u(t_0)$", f"given $u(t_0{{+}}{df})$"] + [n for _, n, _ in STYLE]
    fig.legend(
        h,
        lab,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.035),
    )
    fig.tight_layout(h_pad=2.6)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
