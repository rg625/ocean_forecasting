# ruff: noqa: E741
# mypy: disable-error-code="no-any-return"
"""Figures for the temporal-geometry, noise-law and joint-sparsity sections.

Kept out of the notebook so that the notebook stays a thin presentation layer and the
same code produces the paper figures.  Every curve is mean +/- SEM over the assimilation
problems in that cell; the number of problems is annotated on the axes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

DT = 0.1


class NotRunYet(Exception):
    """Raised when a section's results are not on disk yet."""


def require(path: Path, how: str):
    """Fail loudly but informatively when a section has not been run.

    A half-finished campaign should still render every section it CAN, and name the
    command that produces the rest -- rather than aborting the notebook on the first gap.
    """
    if not Path(path).is_file():
        raise NotRunYet(f"{path} is not on disk yet.\n    produce it with:  {how}")
    return Path(path)


# Identical to the STYLE dict in the notebook setup cell, so a method keeps one colour
# and one marker across every figure in the paper.
STYLE = {
    "KAE-expm": dict(
        color="#1b7837", marker="o", label=r"Continuous KAE — exact $e^{K\tau}$"
    ),
    "KAE-rk4": dict(color="#7fbc41", marker="s", label="Continuous KAE — RK4 rollout"),
    "UNet": dict(color="#d6604d", marker="^", label="U-Net 4D-Var"),
    "SDA": dict(color="#762a83", marker="D", label="Score-based DA (per draw)"),
}
ORDER = ["KAE-expm", "KAE-rk4", "UNet", "SDA"]


SECTION_OF = {
    "C1_delta_f": "C1",
    "C2_delta_l": "C2",
    "C3_n_obs": "C3",
    "noise_law": "NL",
    "joint_sparsity": "JS",
}


def load(out_dir: Path, name: str) -> Dict:
    f = require(
        Path(out_dir) / f"{name}.json",
        f"python -m data_assimilation.ks.exp_geometry --out-dir {out_dir} "
        f"--sections {SECTION_OF.get(name, '?')}",
    )
    return json.loads(f.read_text())


def _series(rows: List[Dict], m: str, key="mean"):
    return np.array(
        [r[m][key] if m in r and np.isfinite(r[m]["mean"]) else np.nan for r in rows],
        dtype=float,
    )


def _curve(
    ax, xs, rows, *, floors=True, methods=ORDER, diverged=None, diverged_method="SDA"
):
    """One curve per method.

    KAE-expm and KAE-RK4 agree to ~4 decimals by construction (that parity IS the result),
    so plotted identically the second would hide the first entirely. expm is drawn as a
    thick translucent underlay and RK4 as a thin line on top, so the reader sees both
    curves and sees that they coincide.

    Where a sampler diverged on some problems, the mean over ALL problems is meaningless
    (one runaway dominates it). Those points are drawn from the converged runs only, with
    an open marker and the failure rate printed beside them -- never silently dropped.
    """
    for m in methods:
        mu, se = _series(rows, m), _series(rows, m, "sem")
        if np.all(np.isnan(mu)):
            continue
        st = dict(STYLE[m])
        # the divergence record belongs to ONE method; applying it to the others would
        # overwrite their means with that method's numbers
        dv = (diverged or {}) if m == diverged_method else {}
        bad = np.array([dv.get(r["tag"], {}).get("n_diverged", 0) for r in rows])
        if bad.any():
            mu = np.array(
                [
                    (
                        dv.get(r["tag"], {}).get("mean_converged", v)
                        if dv.get(r["tag"], {}).get("n_diverged", 0)
                        else v
                    )
                    for r, v in zip(rows, mu)
                ]
            )
            se = np.array(
                [
                    (
                        dv.get(r["tag"], {}).get("sem_converged", v)
                        if dv.get(r["tag"], {}).get("n_diverged", 0)
                        else v
                    )
                    for r, v in zip(rows, se)
                ]
            )
            st["label"] = st["label"] + " (converged runs)"
        if m == "KAE-expm":
            ax.errorbar(
                xs,
                mu,
                yerr=se,
                capsize=0,
                lw=5.0,
                alpha=0.30,
                zorder=2,
                color=st["color"],
                marker=st["marker"],
                ms=9,
                label=st["label"],
            )
            continue
        ax.errorbar(xs, mu, yerr=se, capsize=2.5, lw=1.6, ms=5, zorder=3, **st)
        for i, nb in enumerate(bad):
            if nb:
                n = dv[rows[i]["tag"]]["n"]
                ax.plot(
                    xs[i],
                    mu[i],
                    marker="o",
                    ms=11,
                    mfc="none",
                    mew=1.8,
                    color=st["color"],
                    zorder=4,
                )
                ax.annotate(
                    f"{100 * nb / n:.0f}% diverged",
                    (xs[i], mu[i]),
                    textcoords="offset points",
                    xytext=(-14, 11 + 11 * (i % 2)),
                    fontsize=6.5,
                    ha="right",
                    color=st["color"],
                )
    if floors:
        f = np.array([r["ae_floor"] for r in rows])
        ax.plot(xs, f, ls=":", color="0.35", lw=1.4, label="KAE autoencoder floor")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")


def _lyap(ax, TL, ts):
    """Add a secondary top axis in Lyapunov times."""
    sec = ax.secondary_xaxis("top", functions=(lambda t: t / TL, lambda q: q * TL))
    sec.set_xlabel(r"horizon in Lyapunov times  $\delta_l/T_L$", fontsize=8)
    return sec


# ---------------------------------------------------------------------------
def fig_C1(out_dir, ax=None):
    d = load(out_dir, "C1_delta_f")
    rows = d["rows"]
    xs = np.array([r["delta_f"] for r in rows])
    ax = ax or plt.subplots(figsize=(5.2, 3.6))[1]
    _curve(ax, xs, rows, diverged=divergence(out_dir, "C1_delta_f"))
    ax.set(
        xscale="log",
        xlabel=r"$\delta_f = t_1 - t_0$   (time units)",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title=(
            r"C1  lead to the FIRST observation"
            "\n"
            rf"$\delta_l={d['meta']['fixed']['delta_l']:.1f}$, "
            rf"$N={d['meta']['fixed']['N']}$ fixed  "
            rf"($n={d['meta']['n_problems']}$)"
        ),
    )
    ax.legend(fontsize=7)
    return ax


def fig_C2(out_dir, TL, ax=None, xunits="lyapunov"):
    d = load(out_dir, "C2_delta_l")
    # the anchor repeats one horizon at a DIFFERENT budget; joining it to the sweep would
    # draw a spurious line across the panel. Plot it separately, as the offset it measures.
    rows = [r for r in d["rows"] if not r["tag"].endswith("_anchor")]
    anchors = [r for r in d["rows"] if r["tag"].endswith("_anchor")]
    ts = np.array([r["delta_l"] for r in rows])
    xs = ts / TL if xunits == "lyapunov" else ts
    ax = ax or plt.subplots(figsize=(5.6, 3.8))[1]
    _curve(ax, xs, rows, diverged=divergence(out_dir, "C2_delta_l"))
    for a in anchors:
        ax_ = a["delta_l"] / TL if xunits == "lyapunov" else a["delta_l"]
        for m in ORDER:
            if m in a and "mean" in a[m]:
                ax.plot(
                    ax_,
                    a[m]["mean"],
                    marker="*",
                    ms=13,
                    mew=1.1,
                    color=STYLE[m]["color"],
                    mec="k",
                    zorder=6,
                    label=(
                        f"same horizon at {a['iters']} iters (anchor)"
                        if m == ORDER[0]
                        else None
                    ),
                )
    if xunits == "lyapunov":
        ax.axvline(2.5, color="crimson", ls="--", lw=1.3)
        ax.text(
            2.5, ax.get_ylim()[1], r" 2.5 $T_L$", color="crimson", fontsize=8, va="top"
        )
        ax.axvline(2.5 / TL, color="0.4", ls="-.", lw=1.1)
        ax.text(
            2.5 / TL,
            ax.get_ylim()[0],
            "  canonical\n  2.5 t.u.",
            color="0.4",
            fontsize=7,
            va="bottom",
            ha="left",
        )
        xlab = r"recovery horizon  $\delta_l / T_L$   (Lyapunov times)"
    else:
        xlab = r"recovery horizon  $\delta_l$   (time units)"
    ax.set(
        xscale="log",
        xlabel=xlab,
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title=(
            "C2  RECOVERY HORIZON\n"
            rf"$\delta_f={d['meta']['fixed']['delta_f']:.1f}$, "
            rf"$N={d['meta']['fixed']['N']}$ fixed  "
            rf"($n={d['meta']['n_problems']}$, $T_L={TL:.1f}$ t.u.)"
        ),
    )
    ax.legend(fontsize=7)
    return ax


def fig_C3(out_dir, ax=None):
    d = load(out_dir, "C3_n_obs")
    rows = d["rows"]
    xs = np.array([r["N"] for r in rows])  # REALISED N, not requested
    ax = ax or plt.subplots(figsize=(5.2, 3.6))[1]
    _curve(ax, xs, rows, diverged=divergence(out_dir, "C3_n_obs"))
    ax.set(
        xscale="log",
        xlabel=r"$N$  (number of observation times, realised)",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title=(
            "C3  NUMBER of observations\n"
            rf"$\delta_f={d['meta']['fixed']['delta_f']:.1f}$, "
            rf"$\delta_l={d['meta']['fixed']['delta_l']:.1f}$ fixed  "
            rf"($n={d['meta']['n_problems']}$)"
        ),
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(v)) for v in xs])
    ax.legend(fontsize=7)
    return ax


def fig_noise_law(out_dir, axes=None):
    d = load(out_dir, "noise_law")
    rows = d["rows"]
    g = [r for r in rows if r["dist"] == "gaussian"]
    l = [r for r in rows if r["dist"] == "laplace"]
    xs = np.array([r["value"] for r in g])
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    a0, a1 = axes
    for sub, ls, tag in [(g, "-", "Gaussian"), (l, "--", "Laplace")]:
        for m in ORDER:
            mu, se = _series(sub, m), _series(sub, m, "sem")
            if np.all(np.isnan(mu)):
                continue
            st = dict(STYLE[m])
            st["label"] = f"{st['label']} · {tag}"
            a0.errorbar(
                xs,
                mu,
                yerr=se,
                ls=ls,
                capsize=2.5,
                lw=1.5,
                ms=4.5,
                alpha=1.0 if tag == "Gaussian" else 0.65,
                **st,
            )
    a0.set_xscale("symlog", linthresh=1e-2)
    a0.set(
        yscale="log",
        xlabel=r"observation noise std $\sigma_y$  (normalised units)",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title=f"Noise law, matched variance  ($n={d['meta']['n_problems']}$)",
    )
    a0.grid(alpha=0.3, which="both")
    a0.legend(fontsize=6, ncol=2)

    # penalty: Laplace / Gaussian at the same sigma
    for m in ORDER:
        mg, ml = _series(g, m), _series(l, m)
        if np.all(np.isnan(mg)):
            continue
        st = dict(STYLE[m])
        st.pop("label")
        a1.plot(xs, ml / mg, lw=1.6, ms=5, label=STYLE[m]["label"], **st)
    a1.axhline(1.0, color="k", lw=1)
    a1.set_xscale("symlog", linthresh=1e-2)
    a1.set(
        xlabel=r"observation noise std $\sigma_y$",
        ylabel="Laplace / Gaussian error ratio",
        title="Cost of a misspecified (heavy-tailed) error law",
    )
    a1.grid(alpha=0.3)
    a1.legend(fontsize=7)
    return axes


def fig_joint_sparsity(out_dir, methods=("KAE-expm", "UNet", "SDA"), axes=None):
    d = load(out_dir, "joint_sparsity")
    rows = d["rows"]
    fracs = sorted(d["meta"]["axes"]["obs_frac"])  # ascending, so density grows upward
    ns = d["meta"]["axes"]["n_times"]
    grids = {}
    for m in methods:
        G = np.full((len(fracs), len(ns)), np.nan)
        for r in rows:
            i, j = fracs.index(r["obs_frac"]), ns.index(r["n_times"])
            if m in r:
                G[i, j] = r[m]["mean"]
        grids[m] = G
    finite = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
    vmin, vmax = np.nanpercentile(finite, 2), np.nanpercentile(finite, 98)
    if axes is None:
        _, axes = plt.subplots(1, len(methods), figsize=(4.3 * len(methods), 3.9))
    axes = np.atleast_1d(axes)
    from matplotlib.colors import LogNorm

    for ax, m in zip(axes, methods):
        G = grids[m]
        im = ax.imshow(
            G,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            norm=LogNorm(vmin=max(vmin, 1e-4), vmax=vmax),
        )
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                if np.isfinite(G[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{G[i, j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="w",
                    )
        ax.set(
            xticks=range(len(ns)),
            yticks=range(len(fracs)),
            xlabel="observation times $N$ (requested)",
            ylabel="fraction of sensors" if m == methods[0] else "",
            title=STYLE[m]["label"],
        )
        ax.set_xticklabels(ns)
        ax.set_yticklabels(fracs)
        plt.colorbar(im, ax=ax, label=r"rel-$L_2$ at $t_0$" if m == methods[-1] else "")
    return axes


def divergence(out_dir, name, method="SDA", threshold=1.0):
    """Per-point count of assimilations that diverged rather than converged.

    A sampler that blows up produces a non-finite or absurd error; averaging that in
    destroys the statistic, but silently dropping it would be changing the metric to
    flatter the method. Both are therefore reported: the failure COUNT is a first-class
    number, and any statistic computed over the survivors is always printed next to it.
    """
    f = Path(out_dir) / f"{name}.npz"
    if not f.is_file():
        return {}
    d = np.load(f, allow_pickle=True)
    out = {}
    for r in load(out_dir, name)["rows"]:
        k = f"{r['tag']}__{method}__rel"
        if k not in d.files:
            continue
        v = np.asarray(d[k], dtype=float)
        bad = (~np.isfinite(v)) | (v > threshold)
        out[r["tag"]] = {
            "n": int(v.size),
            "n_diverged": int(bad.sum()),
            "mean_converged": float(v[~bad].mean()) if (~bad).any() else float("nan"),
            "sem_converged": (
                float(v[~bad].std(ddof=1) / np.sqrt((~bad).sum()))
                if (~bad).sum() > 1
                else float("nan")
            ),
            "median_all": float(np.nanmedian(v)),
        }
    return out


def table_geometry(out_dir, name, TL=None) -> str:
    d = load(out_dir, name)
    rows = d["rows"]
    hdr = f"{'schedule (frames)':38s} {'d_f':>5s} {'d_l':>6s} {'N':>3s}"
    if TL:
        hdr += f" {'d_l/T_L':>8s}"
    for m in ORDER:
        hdr += f" {m:>18s}"
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        fr = str(r["frames"])
        s = f"{fr[:38]:38s} {r['delta_f']:5.1f} {r['delta_l']:6.1f} {r['N']:3d}"
        if TL:
            s += f" {r['delta_l'] / TL:8.3f}"
        for m in ORDER:
            s += (
                f" {r[m]['mean']:9.4f}±{r[m]['sem']:.4f}"
                if m in r and np.isfinite(r[m]["mean"])
                else f" {'--':>18s}"
            )
        lines.append(s)

    # divergence report, only when something actually diverged
    dv = divergence(out_dir, name)
    if any(v["n_diverged"] for v in dv.values()):
        lines += [
            "",
            "SDA sampler divergence (a diverged run is NOT dropped silently):",
            f"{'schedule':38s} {'n':>4s} {'diverged':>9s} "
            f"{'median(all)':>12s} {'mean(converged)':>18s}",
        ]
        for r in rows:
            v = dv.get(r["tag"])
            if v is None:
                continue
            lines.append(
                f"{str(r['frames'])[:38]:38s} {v['n']:4d} "
                f"{v['n_diverged']:4d} ({100 * v['n_diverged'] / v['n']:4.1f}%) "
                f"{v['median_all']:12.5f} "
                f"{v['mean_converged']:11.5f}±{v['sem_converged']:.5f}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def fig_sweep_t0_fields(
    out_dir,
    sweep="C1_delta_f",
    xlabel=r"$\delta_f$",
    axes=None,
    example: int = 0,
    ncol: int = 3,
):
    """The recovered state at t_0 at every point of a sweep, all methods together.

    The sweep curve says how MUCH accuracy is lost as the swept quantity grows; this says
    WHAT is lost.  One panel per sweep point, in the style of the headline panel: the true
    u(t_0) as a thick underlay, each method's recovered field over it, that method's
    rel-L2 in the legend.
    """
    f = require(
        Path(out_dir) / f"{sweep}_fields.npz",
        f"python -m data_assimilation.ks.sweep_fields --sweep {sweep} --problem-index 16 "
        f"--rollout 0",
    )
    d = np.load(f, allow_pickle=True)
    x = np.asarray(d["x"]).squeeze()
    tags = [str(t) for t in d["tags"]]
    vals = np.asarray(d["values"], dtype=float)
    keep = [(t, v) for t, v in zip(tags, vals) if f"{t}__truth_t0" in d.files]
    n = len(keep)
    nrow = int(np.ceil(n / ncol))
    if axes is None:
        _, axes = plt.subplots(
            nrow, ncol, figsize=(4.4 * ncol, 3.1 * nrow), squeeze=False
        )
    axes = np.asarray(axes).reshape(-1)

    for i, (tag, val) in enumerate(keep):
        a = axes[i]
        tru = np.asarray(d[f"{tag}__truth_t0"])[example].reshape(-1)
        a.plot(
            x,
            tru,
            lw=6.0,
            color="k",
            alpha=0.22,
            solid_capstyle="round",
            label="true $u(t_0)$",
            zorder=1,
        )
        for m in ORDER:
            k = f"{tag}__{m}__analysis"
            if k not in d.files:
                continue
            rec = np.asarray(d[k])[example].reshape(-1)
            e = np.linalg.norm(rec - tru) / np.linalg.norm(tru)
            a.plot(
                x,
                rec,
                lw=1.6,
                color=STYLE[m]["color"],
                zorder=3,
                label=f"{m} ({e:.3f})",
            )
        lim = 1.45 * np.abs(tru).max()
        a.set_ylim(-lim, lim)
        fr = d.get(f"{tag}__frames")
        ttl = f"{xlabel} = {val:g}"
        if fr is not None:
            ttl += f"    obs at {np.asarray(fr).tolist()}"
        a.set_title(ttl, fontsize=8.5)
        a.legend(fontsize=6.5, loc="upper right")
        a.grid(alpha=0.3)
        a.set_xlabel("space $x$")
        if i % ncol == 0:
            a.set_ylabel("$u$")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    return axes
