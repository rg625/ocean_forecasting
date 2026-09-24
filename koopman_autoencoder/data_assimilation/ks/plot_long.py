# ruff: noqa: F841
"""Figures for the long-rollout and long-baseline experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_assimilation.ks.plot_geometry import STYLE, ORDER

DT = 0.1
LONG = Path("da_results_long")

from data_assimilation.ks.plot_geometry import require, NotRunYet  # noqa: E402

# measured U-Net 4D-Var cost: 2224.5 ms/iter at a 100-frame rollout, i.e. 22.2 ms per
# iteration per rollout frame. Used only to annotate how far outside budget a skipped
# configuration is; never to stand in for a number that was not run.
MS_PER_ITER_PER_FRAME = 2224.47 / 100


def _ref(ax, TL, train_tu=1.0):
    """Reference lines. Call AFTER the axis scales are set.

    Anchoring a text to get_ylim() while the axis is still linear can return a negative
    y, which has no finite position once the axis is switched to log -- and with
    bbox_inches="tight" (the inline backend's default) that inflates the saved figure to
    tens of thousands of pixels. Guarding here makes the helper safe in either order.
    """
    ax.axvline(train_tu / TL, color="0.25", ls="-", lw=1.3)
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        lo = max(lo, 1e-12)
    elif lo <= 0:
        lo = hi * 1e-3 if hi > 0 else 1e-12
    ax.text(
        train_tu / TL,
        hi,
        " training\n rollout",
        fontsize=7,
        color="0.25",
        va="top",
        ha="left",
    )
    ax.axvline(1.0, color="crimson", ls="--", lw=1.2)
    ax.text(1.0, lo, r" $1\,T_L$", fontsize=7.5, color="crimson", va="bottom")


# ---------------------------------------------------------------------------
def fig_forecast(out_dir=LONG, ax=None):
    """Free-running forecast skill from an EXACT initial condition."""
    d = np.load(
        require(
            Path(out_dir) / "FC_forecast.npz",
            "python -m data_assimilation.ks.exp_long_horizon --sections FC "
            "--test data/ks/da_test_long.nc",
        )
    )
    TL = float(d["T_L"])
    tau = d["taus"]
    xs = tau / TL
    ax = ax or plt.subplots(figsize=(7.4, 4.6))[1]
    for m in ["KAE-expm", "KAE-rk4", "UNet"]:
        k = f"{m}__rel_mean"
        if k not in d.files:
            continue
        mu, se = d[k], d[f"{m}__rel_sem"]
        st = {kk: v for kk, v in STYLE[m].items() if kk != "marker"}
        ax.plot(xs, mu, lw=1.9, **st)
        ax.fill_between(xs, mu - se, mu + se, color=STYLE[m]["color"], alpha=0.2, lw=0)
    if "kae_roundtrip" in d.files:
        ax.axhline(
            float(d["kae_roundtrip"]),
            color=STYLE["KAE-expm"]["color"],
            ls="--",
            lw=1.2,
            alpha=0.7,
            label=f"KAE round-trip floor ({float(d['kae_roundtrip']):.3f})",
        )
    ax.plot(xs, d["persistence"], color="0.45", ls="-.", lw=1.5, label="persistence")
    ax.plot(
        xs,
        d["climatology"],
        color="0.15",
        ls=":",
        lw=1.5,
        label="climatology ($\\hat u=0$)",
    )
    ax.plot(
        xs,
        d["saturation"],
        color="crimson",
        ls=":",
        lw=2.0,
        label="saturation (independent state)",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"forecast lead time  $\tau / T_L$   (Lyapunov times)",
        ylabel=r"rel-$L_2$",
        title=(
            "Free-running forecast from the EXACT true state\n"
            rf"$n={int(d['n_problems'])}$ trajectories, $T_L={TL:.1f}$ t.u., "
            f"trained on {float(d['train_rollout_tu']):.1f} t.u. rollouts"
        ),
    )
    ax.grid(alpha=0.3, which="both")
    _ref(ax, TL, float(d["train_rollout_tu"]))
    ax.legend(fontsize=7.5, loc="lower right")
    return ax


def table_forecast(out_dir=LONG) -> str:
    d = np.load(
        require(
            Path(out_dir) / "FC_forecast.npz",
            "python -m data_assimilation.ks.exp_long_horizon --sections FC "
            "--test data/ks/da_test_long.nc",
        )
    )
    TL = float(d["T_L"])
    tau = d["taus"]
    marks = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    L = [
        f"free-running forecast, rel-L2 at a given lead time  (T_L = {TL:.1f} t.u.)",
        "",
        f"{'tau/T_L':>8s} {'t.u.':>8s} "
        + " ".join(f"{m:>10s}" for m in ["KAE-expm", "KAE-rk4", "UNet"])
        + f" {'persist':>9s} {'saturate':>9s}",
    ]
    for q in marks:
        t = q * TL
        if t > tau.max():
            continue
        row = f"{q:8.1f} {t:8.1f} "
        for m in ["KAE-expm", "KAE-rk4", "UNet"]:
            k = f"{m}__rel_mean"
            row += (
                f"{np.interp(t, tau, d[k]):10.4f} " if k in d.files else f"{'--':>10s} "
            )
        row += f"{np.interp(t, tau, d['persistence']):9.4f} "
        row += f"{np.interp(t, tau, d['saturation']):9.4f}"
        L.append(row)
    L += ["", "lead time at which the forecast is no better than an independent state:"]
    for m in ["KAE-expm", "KAE-rk4", "UNet"]:
        k = f"{m}__skill_horizon_tu"
        if k in d.files:
            h = float(d[k])
            L.append(
                f"   {m:9s} {h:8.1f} t.u. = {h / TL:5.2f} T_L"
                if np.isfinite(h)
                else f"   {m:9s} never reached"
            )
    return "\n".join(L)


# ---------------------------------------------------------------------------
def fig_post_da(out_dir=LONG, ax=None, show_perfect=True):
    """Forecast skill after assimilation, against forecast skill from the exact state."""
    d = np.load(
        require(
            Path(out_dir) / "PF_post_da_forecast.npz",
            "python -m data_assimilation.ks.exp_long_horizon --sections PF "
            "--test data/ks/da_test_long.nc",
        )
    )
    TL = float(d["T_L"])
    xs = d["taus"] / TL
    ax = ax or plt.subplots(figsize=(7.8, 4.9))[1]
    for m in ["KAE-expm", "KAE-rk4", "UNet"]:
        k = f"{m}__rel_mean"
        if k not in d.files:
            continue
        mu, se = d[k], d[f"{m}__rel_sem"]
        st = {kk: v for kk, v in STYLE[m].items() if kk != "marker"}
        st["label"] = f"{st['label']}  (analysis {float(d[f'{m}__analysis_rel']):.4f})"
        ax.plot(xs, mu, lw=2.0, **st)
        ax.fill_between(xs, mu - se, mu + se, color=STYLE[m]["color"], alpha=0.2, lw=0)
        ok = np.where(np.isfinite(mu))[0]
        if len(ok) and ok[-1] < len(xs) - 1:
            ax.plot(
                xs[ok[-1]],
                mu[ok[-1]],
                marker="x",
                ms=10,
                mew=2.2,
                color=STYLE[m]["color"],
            )
    if show_perfect and (Path(out_dir) / "FC_forecast.npz").is_file():
        fc = np.load(Path(out_dir) / "FC_forecast.npz")
        fx = fc["taus"] / float(fc["T_L"])
        for m in ["KAE-expm", "UNet"]:
            ax.plot(
                fx,
                fc[f"{m}__rel_mean"],
                color=STYLE[m]["color"],
                ls=":",
                lw=1.4,
                alpha=0.75,
                label=f"{m} from the EXACT state" if m == "UNet" else None,
            )
    ax.plot(xs, d["persistence"], color="0.45", ls="-.", lw=1.4, label="persistence")
    ax.plot(
        xs, d["saturation"], color="crimson", ls=":", lw=1.8, label="no-skill level"
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"forecast lead time after $t_0$,  $\tau/T_L$",
        ylabel=r"rel-$L_2$",
        title=(
            "Forecast AFTER assimilation (solid) vs from the EXACT state (dotted)\n"
            rf"observations at {list(map(int, d['obs_frames']))} frames, "
            rf"{int(d['iters'])} iterations, $n={int(d['n_problems'])}$"
        ),
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")
    return ax


def table_post_da(out_dir=LONG) -> str:
    d = np.load(require(Path(out_dir) / "PF_post_da_forecast.npz", "--sections PF"))
    TL = float(d["T_L"])
    tau = d["taus"]
    ms = [m for m in ["KAE-expm", "KAE-rk4", "UNet"] if f"{m}__rel_mean" in d.files]
    L = [
        f"forecast after assimilation  (T_L = {TL:.1f} t.u., "
        f"{int(d['iters'])} iterations, n={int(d['n_problems'])})",
        "",
        f"{'tau/T_L':>8s} {'t.u.':>8s} "
        + " ".join(f"{m:>10s}" for m in ms)
        + f" {'persist':>9s}",
    ]
    L.append(
        f"{'analysis':>8s} {0.0:8.1f} "
        + " ".join(f"{float(d[f'{m}__analysis_rel']):10.4f}" for m in ms)
        + f" {0.0:9.4f}"
    )
    for q in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        t = q * TL
        if t > tau.max():
            continue
        row = f"{q:8.2f} {t:8.1f} "
        for m in ms:
            mu = d[f"{m}__rel_mean"]
            ok = np.isfinite(mu)
            row += (
                f"{np.interp(t, tau[ok], mu[ok]):10.4f} "
                if t <= tau[ok].max()
                else f"{'capped':>10s} "
            )
        row += f"{np.interp(t, tau, d['persistence']):9.4f}"
        L.append(row)
    L += ["", "lead time at which the post-DA forecast has no skill left:"]
    for m in ms:
        h = float(d[f"{m}__skill_horizon_tu"])
        L.append(
            f"   {m:9s} {h:8.1f} t.u. = {h / TL:5.2f} T_L"
            if np.isfinite(h)
            else f"   {m:9s} not reached within the window"
        )
    L += ["", "rollout caps (affordability):"]
    for m in ms:
        L.append(f"   {m:9s} {int(d[f'{m}__cap_frames'])} frames")
    return "\n".join(L)


# ---------------------------------------------------------------------------
def _fmt_tau(v: float) -> str:
    """Format a lead time so a SMALL value never renders as 0.

    tau = 0 would mean observing t_0 itself, which the protocol forbids -- every
    observation is strictly in the future.  The first observation sits one frame ahead,
    which is 0.0044 T_L, and rounding that to "0.0" reads as if t_0 had been observed.
    """
    if v == 0:
        return "0"
    if v >= 0.1:
        return f"{v:.2f}"
    return f"{v:.2g}"


def fig_headline_long(
    out_dir=LONG, tags=("canonical", "medium", "long", "extreme"), axes=None
):
    """Panel (b)+(c) of the headline figure, at several observation baselines.

    Left column: the recovered state at t_0, which is never observed.
    Right column: the fit to the observations, which move progressively further into the
    future -- the extension of A.1(c) beyond delta_l = 2.5.
    """
    missing = [
        t for t in tags if not (Path(out_dir) / f"HL_headline_{t}.npz").is_file()
    ]
    tags = [t for t in tags if (Path(out_dir) / f"HL_headline_{t}.npz").is_file()]
    if not tags:
        raise NotRunYet(
            "no HL_headline_*.npz on disk yet.\n    produce them with:  "
            "python -m data_assimilation.ks.exp_long_horizon --sections HL --test data/ks/da_test_long.nc"
        )
    if missing:
        print(
            f"[note] baselines not on disk yet, omitted from the figure: "
            f"{', '.join(missing)}"
        )
    if axes is None:
        _, axes = plt.subplots(
            len(tags), 2, figsize=(12.6, 3.3 * len(tags)), squeeze=False
        )
    axes = np.atleast_2d(axes)

    for r, tag in enumerate(tags):
        d = np.load(Path(out_dir) / f"HL_headline_{tag}.npz", allow_pickle=True)
        TL = float(d["T_L"])
        x = np.asarray(d["x"]).squeeze()
        offs = np.atleast_1d(d["offsets"])
        a0, a1 = axes[r, 0], axes[r, 1]

        tru = np.asarray(d["u_t0_true"]).reshape(-1)
        a0.plot(
            x,
            tru,
            lw=6.5,
            color="k",
            alpha=0.22,
            solid_capstyle="round",
            label="true $u(t_0)$  (never observed)",
            zorder=1,
        )
        for m in ORDER:
            k = f"{m}__u_t0_recon"
            if k not in d.files:
                continue
            a0.plot(
                x,
                np.asarray(d[k]).reshape(-1),
                lw=1.7,
                color=STYLE[m]["color"],
                zorder=3,
                label=f"{m}  ({float(d[f'{m}__rel_final']):.3f})",
            )
        lim = 1.35 * np.abs(tru).max()
        a0.set(
            ylim=(-lim, lim),
            xlabel="space $x$",
            ylabel="$u$",
            title=(
                f"({tag})  recovered "
                + r"$u(t_0)$    "
                + rf"$\delta_l={offs.max() * DT:.1f}$ t.u. "
                + rf"$={offs.max() * DT / TL:.2f}\,T_L$"
            ),
        )
        a0.legend(fontsize=7)
        a0.grid(alpha=0.3)

        ot = np.asarray(d["obs_true"])
        for i in range(ot.shape[0]):
            a1.plot(
                x,
                ot[i],
                color="k",
                lw=4.5,
                alpha=0.18,
                solid_capstyle="round",
                zorder=1,
                label="observed truth" if i == 0 else None,
            )
        for m in ORDER:
            k = f"{m}__obs_pred"
            if k not in d.files:
                continue
            op = np.asarray(d[k])
            for i in range(op.shape[0]):
                a1.plot(
                    x,
                    op[i],
                    color=STYLE[m]["color"],
                    lw=1.0,
                    alpha=0.85,
                    label=m if i == 0 else None,
                )
        lim = 1.35 * np.abs(ot).max()
        a1.set(
            ylim=(-lim, lim),
            xlabel="space $x$",
            ylabel="$u$",
            title=(
                r"fit to the observations,  $\tau/T_L$ = ["
                + ", ".join(_fmt_tau(float(o) * DT / TL) for o in offs)
                + "]"
            ),
        )
        a1.legend(fontsize=7)
        a1.grid(alpha=0.3)
        miss = [m for m in ORDER if f"{m}__skipped_reason" in d.files]
        if miss:
            n_fr = int(offs.max())
            est_h = MS_PER_ITER_PER_FRAME * n_fr * 2000 / 3.6e6
            a1.text(
                0.99,
                0.015,
                "cost wall: "
                + ", ".join(miss)
                + f"  ({n_fr} fr "
                + r"$\approx$ "
                + f"{est_h:.0f} GPU-h for U-Net; "
                + r"KAE-expm $O(1)$)",
                transform=a1.transAxes,
                fontsize=6.0,
                color="crimson",
                va="bottom",
                ha="right",
                bbox=dict(fc="white", ec="crimson", lw=0.6, alpha=0.85, pad=1.6),
            )
    return axes


def fig_long_baseline(out_dir=LONG, ax=None, fc_saturation=True):
    d = json.loads(
        require(
            Path(out_dir) / "LB_long_baseline.json",
            "python -m data_assimilation.ks.exp_long_horizon --sections LB "
            "--test data/ks/da_test_long.nc",
        ).read_text()
    )
    rows, TL = d["rows"], d["meta"]["T_L"]
    xs = np.array([r["delta_l_TL"] for r in rows])
    ax = ax or plt.subplots(figsize=(7.4, 4.6))[1]
    for m in ORDER:
        mu = np.array(
            [
                r[m]["mean"] if m in r and not r[m].get("skipped") else np.nan
                for r in rows
            ]
        )
        se = np.array(
            [
                r[m]["sem"] if m in r and not r[m].get("skipped") else np.nan
                for r in rows
            ]
        )
        if np.all(np.isnan(mu)):
            continue
        ax.errorbar(xs, mu, yerr=se, capsize=2.5, lw=1.7, ms=5, **STYLE[m])
        ok = np.where(np.isfinite(mu))[0]
        if len(ok) and ok[-1] < len(xs) - 1:
            ax.plot(
                xs[ok[-1]],
                mu[ok[-1]],
                marker="x",
                ms=11,
                mew=2.2,
                color=STYLE[m]["color"],
            )
            ax.annotate(
                "cost wall",
                (xs[ok[-1]], mu[ok[-1]]),
                textcoords="offset points",
                xytext=(6, -12),
                fontsize=7,
                color=STYLE[m]["color"],
            )
    if fc_saturation and (Path(out_dir) / "FC_forecast.npz").is_file():
        fc = np.load(Path(out_dir) / "FC_forecast.npz")
        ax.axhline(
            float(np.median(fc["saturation"])),
            color="crimson",
            ls=":",
            lw=1.8,
            label="no-skill level (independent state)",
        )
    ax.axvline(1.0, color="0.4", ls="--", lw=1.2)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"observation baseline  $\delta_l / T_L$   (Lyapunov times)",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title=(
            "Recovery of $t_0$ from FAR-FUTURE observations\n"
            rf"$\delta_f={d['meta']['fixed']['delta_f_frames'] * DT:.1f}$ t.u., "
            rf"$N={d['meta']['fixed']['N_requested']}$, "
            rf"{d['meta']['iters']} iterations, $n={d['meta']['n_problems']}$"
        ),
    )
    # The 1 T_L marker is placed AFTER the scales are set. Anchoring a text to
    # ax.get_ylim()[0] while the axis is still LINEAR returns a negative value (-0.061
    # here), and once the axis becomes log that text has no finite position: the inline
    # backend saves with bbox_inches="tight", the tight bbox runs away, and the cell
    # renders a 675 x 151,411 pixel, almost entirely empty figure.
    ax.text(1.0, ax.get_ylim()[0], r" $1\,T_L$", fontsize=7.5, color="0.4", va="bottom")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="lower right")
    return ax


def table_long_baseline(out_dir=LONG) -> str:
    d = json.loads(
        require(
            Path(out_dir) / "LB_long_baseline.json",
            "python -m data_assimilation.ks.exp_long_horizon --sections LB "
            "--test data/ks/da_test_long.nc",
        ).read_text()
    )
    rows = d["rows"]
    L = [
        f"{'delta_l':>9s} {'T_L':>7s} {'frames':>7s} {'N':>3s} "
        + " ".join(f"{m:>18s}" for m in ORDER)
    ]
    L.append("-" * len(L[0]))
    for r in rows:
        s = (
            f"{r['delta_l_tu']:9.1f} {r['delta_l_TL']:7.2f} "
            f"{r['delta_l_frames']:7d} {r['N']:3d} "
        )
        for m in ORDER:
            if m not in r or r[m].get("skipped"):
                s += f" {'not run':>18s}"
            else:
                s += f" {r[m]['mean']:9.4f}±{r[m]['sem']:.4f}"
        L.append(s)
    L += ["", "caps (affordability, not capability):"]
    for m, c in d["meta"]["caps"].items():
        L.append(f"   {m:6s} {c if c < 10 ** 8 else 'none'} frames")
    L += ["", d["meta"]["note"]]
    return "\n".join(L)
