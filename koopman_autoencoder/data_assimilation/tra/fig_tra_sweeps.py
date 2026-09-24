# ruff: noqa: E741
# mypy: disable-error-code="var-annotated"
"""The four TRA conditional sweeps as curves, at both pinned leads.

The companion to ``data_assimilation.ks.fig_ks_sweeps``: same layout, same convention, so the two regimes
can be read side by side.  One panel per axis, delta_f = 1 faint and dashed, delta_f = 9
solid, so the effect of the lead shows up inside each panel rather than across two tables.

ACDM-ncn is excluded throughout -- trained with clean conditioning, its score is invalid
for assimilation and it sits near 4 at every point.

Read from the same files as the tables, so the curves and the tables cannot drift apart.
Sweep points are sorted by their swept value first: the campaign appended sigma = 0.05 and
0.15 after 0.3, so file order is not sweep order.

    python -m data_assimilation.tra.fig_tra_sweeps
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
    ("KAE", "KAE", "#1b7837"),
    ("ACDM", "ACDM $+$ SDA", "#6a3d9a"),
    ("FNO", "FNO", "#ff7f00"),
    ("UNet", "U-Net", "#e41a1c"),
]
INK = "#2b2b2b"

# stem -> (x label, title, log-x, invert-x)
SPEC = {
    "G4_delta_l": (
        r"recovery horizon $\delta_l$  (frames)",
        "recovery horizon",
        True,
        False,
    ),
    "G5_n_obs": (r"observations $N$", "number of observations", True, False),
    "G3_sparsity": ("fraction of grid points observed", "sensor coverage", True, True),
    "G2_noise": (r"observation noise $\sigma$", "observation noise", False, False),
}


def series(stem, df):
    """{method: (x, mean, sem)} for one axis at one pinned lead."""
    dirn = "da_results_tra" if df == 1 else "da_results_tra_df9"
    path = Path(dirn) / f"{stem}.json"
    if not path.is_file():
        return {}
    rows = sorted(json.load(open(path))["rows"], key=lambda r: float(r["value"]))
    out = {}
    for r in rows:
        for key, _, _ in STYLE:
            v = r.get(key)
            if not isinstance(v, dict) or v.get("mean") is None:
                continue
            m = v["mean"]
            if m != m or (v.get("sem") and m > 0 and v["sem"] / m > 0.5):
                continue
            out.setdefault(key, []).append((float(r["value"]), m, v.get("sem") or 0.0))
    return {k: tuple(np.array(z) for z in zip(*v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figs/tra_sweeps.png"))
    ap.add_argument("--only-df9", action="store_true")
    a = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))
    for ax, stem in zip(axes.ravel(), SPEC):
        xlab, title, logx, invert = SPEC[stem]
        for df, alpha, lw, ls in ((1, 0.32, 1.4, (0, (3, 2))), (9, 1.0, 2.0, "-")):
            if a.only_df9 and df == 1:
                continue
            s = series(stem, df)
            for key, name, col in STYLE:
                if key not in s:
                    continue
                x, m, e = s[key]
                ax.errorbar(
                    x,
                    m,
                    yerr=e,
                    color=col,
                    lw=lw,
                    ls=ls,
                    alpha=alpha,
                    marker="o" if df == 9 else None,
                    ms=4,
                    capsize=2.5,
                    label=name if df == 9 else None,
                    zorder=3 if df == 9 else 2,
                )
        ax.set(yscale="log", xlabel=xlab, title=title)
        if logx:
            ax.set_xscale("log")
        if invert:
            ax.invert_xaxis()
        ax.grid(alpha=0.2, which="both")
        ax.tick_params(labelsize=9, colors="#666666")
        ax.title.set(fontsize=10, color=INK)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#d0d0d0")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"analysis rel-$L_2$ at $t_0$", fontsize=10, color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    extra = (
        []
        if a.only_df9
        else [
            plt.Line2D(
                [],
                [],
                color="#888888",
                lw=1.4,
                ls=(0, (3, 2)),
                alpha=0.6,
                label=r"$\delta_f=1$ (faint)",
            )
        ]
    )
    fig.legend(
        h + extra,
        l + [e.get_label() for e in extra],
        ncol=5,
        fontsize=10,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.045),
    )
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)
    for stem in SPEC:
        s9 = series(stem, 9)
        print(
            f"  {stem:13s} df=9 "
            + "  ".join(f"{k} {v[1][0]:.4f}->{v[1][-1]:.4f}" for k, v in s9.items())
        )


if __name__ == "__main__":
    main()
