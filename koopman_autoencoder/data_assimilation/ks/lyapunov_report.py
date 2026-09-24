"""Convert every horizon in the campaign to Lyapunov times and plot the growth curve.

The point of this module is a units correction.  The brief asks to "extend beyond the
current 2.5 Lyapunov-time limit", but 2.5 is the canonical delta_l in TIME UNITS.  With
the measured lambda_1 the canonical horizon is about 0.11 T_L, so the campaign has been
running two orders of magnitude inside the predictability limit, not at it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DT = 0.1
OUT = Path("da_results_sda_paper")


def main():
    d = json.loads((OUT / "lyapunov.json").read_text())
    lam, sem, TL = d["lambda_1_mean"], d["lambda_1_sem"], d["lyapunov_time"]
    TL_lo, TL_hi = 1.0 / (lam + sem), 1.0 / (lam - sem)

    horizons = [
        ("canonical delta_l (sections A, D-H)", 2.5),
        ("section I space-time window", 7.9),
        ("C2 sweep, longest", 70.0),
        ("post-DA forecast (section B)", 40.0),
        ("full test record", 999 * DT),
    ]
    rows = [
        {
            "name": n,
            "t_units": t,
            "frames": round(t / DT),
            "lyapunov_times": t / TL,
            "lyapunov_times_lo": t / TL_hi,
            "lyapunov_times_hi": t / TL_lo,
        }
        for n, t in horizons
    ]
    targets = [
        {
            "lyapunov_times": q,
            "t_units": q * TL,
            "frames": round(q * TL / DT),
            "fits_in_1000_frame_record": q * TL / DT < 900,
        }
        for q in (0.5, 1.0, 2.0, 2.5, 4.0, 5.0)
    ]

    rep = {
        "lambda_1": lam,
        "lambda_1_sem": sem,
        "n_trajectories": d["n_trajectories"],
        "lyapunov_time_t_units": TL,
        "lyapunov_time_frames": TL / DT,
        "horizons": rows,
        "targets": targets,
        "note": (
            "The canonical delta_l = 2.5 is in TIME UNITS. In Lyapunov units it is "
            f"{2.5 / TL:.3f} T_L. Reaching 2.5 T_L needs delta_l = {2.5 * TL:.1f} "
            f"t.u. = {round(2.5 * TL / DT)} frames; the stored records are 1000 "
            "frames, so 2.5 T_L is reachable but 5 T_L is not."
        ),
    }
    (OUT / "lyapunov_report.json").write_text(json.dumps(rep, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    all_l = np.asarray(d["lambda_1_all"])
    ax[0].plot(
        np.arange(len(all_l)),
        all_l,
        "o",
        color="steelblue",
        ms=5,
        label="per trajectory",
    )
    ax[0].axhline(
        lam, color="k", lw=1.8, label=rf"$\lambda_1={lam:.4f}\pm{sem:.4f}$ (SEM)"
    )
    ax[0].axhspan(lam - sem, lam + sem, color="k", alpha=0.15)
    ax[0].set(
        xlabel="test trajectory",
        ylabel=r"$\lambda_1$",
        title=rf"(a) twin-experiment $\lambda_1$   ($T_L={TL:.1f}$ t.u.)",
    )
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    names = [r["name"] for r in rows]
    vals = [r["lyapunov_times"] for r in rows]
    y = np.arange(len(names))[::-1]
    ax[1].barh(y, vals, color=["#4c72b0"] * 4 + ["#999"], height=0.6)
    for yy, v in zip(y, vals):
        ax[1].text(v * 1.12, yy, f"{v:.2f}", va="center", fontsize=8)
    ax[1].axvline(2.5, color="crimson", ls="--", lw=1.4)
    ax[1].text(2.5, len(names) - 0.3, "  2.5 $T_L$", color="crimson", fontsize=8)
    ax[1].set(
        yticks=y,
        xscale="log",
        xlabel=r"horizon in Lyapunov times $\delta_l/T_L$",
        title="(b) where the campaign actually sits",
    )
    ax[1].set_yticklabels(names, fontsize=7.5)
    ax[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "lyapunov_report.png", dpi=150)

    print(
        f"lambda_1 = {lam:.5f} +/- {sem:.5f}   T_L = {TL:.2f} t.u. = {TL / DT:.0f} frames"
    )
    print(f"\n{'horizon':40s} {'t.u.':>7s} {'frames':>7s} {'T_L':>7s}")
    for r in rows:
        print(
            f"{r['name']:40s} {r['t_units']:7.1f} {r['frames']:7d} {r['lyapunov_times']:7.2f}"
        )
    print(
        f"\n{'target':>10s} {'t.u.':>8s} {'frames':>8s}  fits in a 1000-frame record?"
    )
    for t in targets:
        print(
            f"{t['lyapunov_times']:9.1f} {t['t_units']:8.1f} {t['frames']:8d}  "
            f"{'yes' if t['fits_in_1000_frame_record'] else 'NO'}"
        )


if __name__ == "__main__":
    main()
