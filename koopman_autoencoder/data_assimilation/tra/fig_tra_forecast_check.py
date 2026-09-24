"""Forecast sanity row: every surrogate predicts this flow well.

This figure exists to close an obvious line of attack.  The assimilation gallery shows the
U-Net and FNO recovering the state at t_0 poorly, and a reader is entitled to suspect that
the baselines were simply trained badly.  They were not: rolled out from the EXACT initial
state, with no assimilation and no optimisation anywhere in the loop, the U-Net is the most
accurate forecaster of the five and the FNO is second.  The KAE is the WEAKEST of them.

That is the paper's point in one row.  Forecast skill and assimilation skill are close to
anti-correlated here, so the models that extrapolate best from a known state are not the
models one should invert to find an unknown one.

Drawn from ``rollout_from_true_ic.npz``, which stores each method's free rollout, so no
model is re-run and nothing here depends on a tuning choice.

    python -m data_assimilation.tra.fig_tra_forecast_check --lead 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ACDM-ncn is excluded throughout: its score is invalid for conditioning and it is noise
ORDER = [
    ("KAE", "KAE", "#1b7837"),
    ("UNet", "U-Net", "#e41a1c"),
    ("FNO", "FNO", "#ff7f00"),
    ("ACDM", "ACDM", "#6a3d9a"),
]
INK = "#2b2b2b"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--npz", type=Path, default=Path("da_results_tra/rollout_from_true_ic.npz")
    )
    ap.add_argument("--out", type=Path, default=Path("figs/tra_forecast_check.png"))
    ap.add_argument("--lead", type=int, default=20, help="rollout frame to display")
    ap.add_argument("--example", type=int, default=0)
    ap.add_argument(
        "--channel", type=int, default=3, help="3 = density, shows the wake"
    )
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    i = a.lead - 1
    truth = d["truth"][a.example, i, a.channel]
    n_lead = d["truth"].shape[1]

    fig, axes = plt.subplots(
        1,
        len(ORDER) + 2,
        figsize=(3.0 * (len(ORDER) + 2), 2.35),
        gridspec_kw={"width_ratios": [1] * (len(ORDER) + 1) + [1.25]},
    )
    lo, hi = np.nanpercentile(truth, [1, 99])
    axes[0].imshow(
        truth, origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi, aspect="auto"
    )
    axes[0].set_title("TRUTH", fontsize=9, color=INK)
    for c, (key, name, col) in enumerate(ORDER, start=1):
        f = d[f"{key}__roll"][a.example, i, a.channel]
        axes[c].imshow(
            f, origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi, aspect="auto"
        )
        axes[c].set_title(
            f"{name}\n{float(d[f'{key}__err'][i]):.4f}", fontsize=8.5, color=col
        )
    for ax in axes[:-1]:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#dddddd")

    ax = axes[-1]
    # frame 1 IS the given state, so every error there is exactly 0 and drags a log axis
    # down eight decades; the curve starts at the first frame that is a prediction
    lead = np.arange(2, n_lead + 1)
    sl = slice(1, None)
    for key, name, col in ORDER:
        ax.plot(lead, d[f"{key}__err"][sl], color=col, lw=1.8, label=name)
    ax.plot(
        lead, d["persistence"][sl], color="0.55", ls="-.", lw=1.4, label="persistence"
    )
    ax.axvline(a.lead, color="#999999", ls=":", lw=1.2)
    ax.set(
        yscale="log",
        xlabel="rollout frame",
        ylabel="rel-$L_2$",
        xlim=(1, n_lead),
        ylim=(3e-3, 1.0),
    )
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.22, which="both")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower right", ncol=2)
    ax.set_title("free rollout from the true state", fontsize=8.5, color=INK)

    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)
    print(
        f"  at lead {a.lead}: "
        + "  ".join(f"{n} {float(d[k + '__err'][i]):.4f}" for k, n, _ in ORDER)
        + f"  persistence {float(d['persistence'][i]):.4f}"
    )


if __name__ == "__main__":
    main()
