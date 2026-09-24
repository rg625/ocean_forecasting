# ruff: noqa: F841
# mypy: disable-error-code="var-annotated"
"""Cost against accuracy for every method, both regimes, one pair of axes.

Each marker is an AVERAGE OVER EVERY SWEEP POINT of that regime -- the five axes delta_f,
delta_l, N, observation noise and sensor coverage -- not a single operating point.  A
single point misleads here: at the canonical schedule ACDM beats the KAE on TRA, but that
is the one lead (delta_f = 1) where it does.  For the same reason every conditional axis is
counted at BOTH pinned leads: weighting the average toward delta_f = 1, as the original
campaign does, hides the very axis this paper is about.

The two regimes differ by orders of magnitude in both quantities, so they share one
log-log plane rather than two y-scales.  Shape carries the regime, colour the method, and
every point is labelled, so nothing depends on telling two colours apart.

Cost is WALL CLOCK PER SOLVE: the wall of the batch divided by the problems in it.  4D-Var
solves a batch in one graph and the samplers draw a batch at once, so this is an amortised
per-problem cost.

SPREAD.  These errors are right-skewed over three decades, so the centre is the geometric
mean and the spread is one standard deviation of log10, which is symmetric on a log axis.
The ellipses are drawn at a COMMON FRACTION of that spread (``--spread-scale``, 0.5 by
default): comparable with each other, deliberately not to scale against the axes, which is
what keeps eight overlapping bands legible.  The printed numbers are unscaled.

``--stat mean`` gives the arithmetic mean instead.  It cannot be drawn against 1/error for
this data -- four of the eight series have a standard deviation larger than their mean, so
the band has no positive lower edge -- and the two statistics disagree about who wins in
both regimes.  Both are reported rather than one quietly chosen.

    python -m data_assimilation.ks.fig_cost_accuracy_both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse

# re-stepped from the report's figure palette so that adjacent pairs stay apart under
# simulated protanopia/deuteranopia; the original green/red pair was dE 3.4 and unreadable
COLOR = {
    "KAE": "#1b7837",
    "U-Net": "#e41a1c",
    "FNO": "#ff7f00",
    "SDA": "#6a3d9a",
    "ACDM $+$ SDA": "#6a3d9a",
    "ACDM-ncn": "#4393c3",
}
INK = "#2b2b2b"


def _points(path, mmap):
    """Every usable sweep point as (rel, wall_s, n), dropping diverged ones."""
    out = {}
    for r in json.load(open(path))["rows"]:
        for src, dst in mmap.items():
            v = r.get(src)
            if not isinstance(v, dict) or v.get("mean") is None:
                continue
            m = v["mean"]
            # a sampler that blew up on part of a point returns a finite mean with a
            # standard error its own size; that is not a measurement
            if m != m or (v.get("sem") and m > 0 and v["sem"] / m > 0.5):
                continue
            out.setdefault(dst, []).append((m, v.get("wall_s"), v.get("n")))
    return out


def axes_data():
    """{regime: {axis: {method: [(rel, wall_s, n), ...]}}} over the shared axes.

    The conditional axes (delta_l, N, noise, sensor coverage) were each run at BOTH pinned
    leads, and both are included.  Taking only the delta_f = 1 campaign put 24 of 30 TRA
    points and 47 of 67 KS points at a single lead -- the one where ACDM is at its best and
    the whole delta_f argument is invisible -- so the "average over conditions" was really
    an average at delta_f = 1 and it reversed the TRA ordering.
    """
    tuned = json.load(open("da_results_geometry_df9/kae512_tuned.json"))
    ks = {
        "delta_f": _points(
            "da_results_geometry_df9/deltaf_fine.json",
            {"KAE": "KAE", "SDA": "SDA", "UNet": "U-Net"},
        )
    }
    for df, dirn in ((1, "da_results_geometry"), (9, "da_results_geometry_df9")):
        for sec in ("C2_delta_l", "C3_n_obs", "noise_law", "joint_sparsity"):
            d = _points(f"{dirn}/{sec}.json", {"SDA": "SDA", "UNet": "U-Net"})
            # the KAE column is the checkpoint the KS tables quote (d_z = 512, informed
            # start), so its points come from the replay, not the original campaign
            d["KAE"] = [
                (r["mean"], r["wall_s"], r["n"])
                for r in tuned["sweeps"]
                if r["section"] == sec and int(r["delta_f"]) == df
            ]
            ks[f"{sec}_df{df}"] = d

    tm = {
        "KAE": "KAE",
        "UNet": "U-Net",
        "FNO": "FNO",
        "ACDM": "ACDM $+$ SDA",
        "ACDM-ncn": "ACDM-ncn",
    }
    tra = {"delta_f": _points("da_results_tra/G1_delta_f.json", tm)}
    for df, dirn in ((1, "da_results_tra"), (9, "da_results_tra_df9")):
        for ax, f in (
            ("noise", "G2_noise"),
            ("sparsity", "G3_sparsity"),
            ("delta_l", "G4_delta_l"),
            ("n_obs", "G5_n_obs"),
        ):
            tra[f"{ax}_df{df}"] = _points(f"{dirn}/{f}.json", tm)
    return {"KS": ks, "TRA": tra}


def load(stat):
    """Per method: log10 centre and log10 spread on both axes, plus the raw numbers."""
    rows = []
    for regime, axd in axes_data().items():
        for m in sorted({m for d in axd.values() for m in d}):
            raw = [p for d in axd.values() for p in d.get(m, [])]
            err = np.array([p[0] for p in raw], dtype=float)
            sec = np.array([p[1] / p[2] for p in raw if p[1] and p[2]], dtype=float)
            if stat == "mean":
                centre, sd = float(err.mean()), float(err.std(ddof=1))
                if centre - sd <= 0:
                    print(
                        f"    note: {regime} {m}: s.d. {sd:.4f} exceeds the mean "
                        f"{centre:.4f}; the band has no positive lower edge"
                    )
            else:
                centre = float(np.exp(np.log(err).mean()))
                sd = float(np.exp(np.log(err).std(ddof=1)))
            rows.append(
                {
                    "regime": regime,
                    "method": m,
                    "n": len(err),
                    "x": float(np.log10(sec.mean())),
                    "sx": float(np.log10(sec).std(ddof=1)),
                    "y": float(-np.log10(centre)),
                    "sy": float(np.log10(err).std(ddof=1)),
                    "centre": centre,
                    "sd": sd,
                    "sec": float(sec.mean()),
                }
            )
    return rows


def decades(ax, which):
    """Integer decade ticks labelled 10^n, for axes drawn in log10 units."""
    lo, hi = ax.get_xlim() if which == "x" else ax.get_ylim()
    t = np.arange(np.ceil(lo), np.floor(hi) + 1)
    lab = [f"$10^{{{int(v)}}}$" for v in t]
    (ax.set_xticks if which == "x" else ax.set_yticks)(t)
    (ax.set_xticklabels if which == "x" else ax.set_yticklabels)(lab)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figs/cost_accuracy_both.png"))
    ap.add_argument("--stat", choices=["mean", "geo"], default="geo")
    ap.add_argument(
        "--spread-scale",
        type=float,
        default=0.5,
        help="common fraction of 1 s.d. drawn, identical for every method",
    )
    ap.add_argument(
        "--spread",
        choices=["both", "y", "none"],
        default="both",
        help="'y' drops the wall-clock spread, which is set by the delta_l "
        "axis by design rather than by run-to-run noise; 'none' plots "
        "bare markers",
    )
    ap.add_argument(
        "--with-ncn",
        action="store_true",
        help="include ACDM-ncn; it sits a decade below everything and only "
        "stretches the axis",
    )
    a = ap.parse_args()

    rows = load(a.stat)
    if not a.with_ncn:
        rows = [r for r in rows if r["method"] != "ACDM-ncn"]
    k = a.spread_scale
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.set_facecolor("#fdfdfc")
    by = {(r["regime"], r["method"]): r for r in rows}

    for r in rows:
        c = COLOR[r["method"]]
        if a.spread == "both":
            ax.add_patch(
                Ellipse(
                    (r["x"], r["y"]),
                    2 * k * r["sx"],
                    2 * k * r["sy"],
                    facecolor=c,
                    alpha=0.14,
                    edgecolor=c,
                    linewidth=1.1,
                    zorder=2,
                )
            )
        elif a.spread == "y":
            ax.errorbar(
                r["x"],
                r["y"],
                yerr=k * r["sy"],
                fmt="none",
                ecolor=c,
                elinewidth=2.4,
                capsize=5,
                capthick=2.0,
                alpha=0.55,
                zorder=2,
            )
    off = {
        ("KS", "KAE"): (-2, 25),
        ("KS", "SDA"): (0, 25),
        ("KS", "U-Net"): (0, -30),
        ("TRA", "KAE"): (0, -30),
        ("TRA", "ACDM $+$ SDA"): (0, 25),
        ("TRA", "ACDM-ncn"): (0, 25),
        ("TRA", "FNO"): (-52, -4),
        ("TRA", "U-Net"): (0, 25),
    }
    for r in rows:
        c, regime = COLOR[r["method"]], r["regime"]
        ax.scatter(
            r["x"],
            r["y"],
            s=210,
            marker="o" if regime == "KS" else "s",
            facecolor="white",
            edgecolor="white",
            linewidth=0,
            zorder=3,
        )
        ax.scatter(
            r["x"],
            r["y"],
            s=165,
            marker="o" if regime == "KS" else "s",
            facecolor=c if regime == "KS" else "white",
            edgecolor=c,
            linewidth=2.4,
            zorder=4,
        )
        ax.annotate(
            r["method"],
            (r["x"], r["y"]),
            textcoords="offset points",
            xytext=off[(regime, r["method"])],
            ha="center",
            fontsize=10,
            color=INK,
            zorder=5,
            path_effects=[pe.withStroke(linewidth=3.0, foreground="#fdfdfc")],
        )

    ax.set_xlabel("wall clock per solve  (s)", fontsize=11, color=INK, labelpad=7)
    ax.set_ylabel(
        r"accuracy,  $1\,/\,$rel-$L_2$ at $t_0$", fontsize=11, color=INK, labelpad=7
    )
    ax.margins(x=0.17, y=0.15)
    decades(ax, "x")
    decades(ax, "y")
    ax.tick_params(colors="#777777", labelsize=9.5, length=0, pad=5)
    ax.grid(alpha=0.16, which="major", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d8d8d8")

    h = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=9.5,
            mfc="#666666",
            mec="#666666",
            label="KS  (1-D)",
        ),
        plt.Line2D(
            [],
            [],
            marker="s",
            ls="none",
            ms=9.5,
            mfc="white",
            mec="#666666",
            mew=2.2,
            label=r"TRA  (2-D)",
        ),
    ]
    leg = ax.legend(
        handles=h,
        fontsize=10,
        loc="lower left",
        frameon=False,
        handletextpad=0.7,
        borderpad=0.2,
    )
    leg.set_zorder(6)
    ax.text(
        0.012,
        0.978,
        "$\\nwarrow$  better",
        transform=ax.transAxes,
        fontsize=10,
        color="#9a9a9a",
        va="top",
        ha="left",
    )
    note = {
        "both": f"shaded: $\\pm{k:g}\\,$s.d. over sweep points",
        "y": f"bars: $\\pm{k:g}\\,$s.d. in accuracy over sweep points",
        "none": "",
    }[a.spread]
    if note:
        ax.text(
            0.99,
            0.018,
            note,
            transform=ax.transAxes,
            ha="right",
            fontsize=8.5,
            color="#a5a5a5",
        )
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", a.out, f"[{a.stat}, spread drawn at {k:g} s.d.]")
    for r in sorted(rows, key=lambda r: (r["regime"], r["centre"])):
        band = f"x/÷ {r['sd']:.2f}" if a.stat == "geo" else f"± {r['sd']:.4f}"
        print(
            f"  {r['regime']:3s} {r['method']:14s} n={r['n']:3d}  "
            f"rel {r['centre']:.4f} {band}   {r['sec']:7.2f} s/solve"
        )


if __name__ == "__main__":
    main()
