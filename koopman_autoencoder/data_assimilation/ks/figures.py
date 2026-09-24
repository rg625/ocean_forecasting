# ruff: noqa: F841
"""Publication figures for the KS data-assimilation campaign.

Every figure is produced *only* from the machine-readable artefacts written by
``run_da_suite.py`` (``results.csv``, ``summary.json``, ``raw/*.npz``); nothing is
hard-coded.  Style follows the repository's existing plots (seaborn whitegrid, PDF at
300 dpi).

    python -m data_assimilation.ks.figures --results da_results_v2 --out iclr_2027/figures/da
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

STYLE = {
    "KAE-expm": dict(
        color="#1f77b4", marker="o", label="Continuous KAE (exact $e^{K\\tau}$)"
    ),
    "KAE-rk4": dict(color="#2ca02c", marker="s", label="Continuous KAE (RK4 rollout)"),
    "UNet-4DVar": dict(color="#d62728", marker="^", label="U-Net 4D-Var"),
    "SDA": dict(color="#9467bd", marker="D", label="Score-based DA"),
}
FLOOR = dict(color="0.35", ls=":", lw=1.4, label="Autoencoder reconstruction floor")


def sty(m: str) -> Dict:
    return STYLE.get(m, dict(color="0.4", marker="x", label=m))


class Results:
    def __init__(self, root: Path):
        self.root = root
        self.df = pd.read_csv(root / "results.csv")
        with open(root / "summary.json") as f:
            self.summary = json.load(f)

    def raw(self, name: str):
        return np.load(self.root / "raw" / f"{name}.npz", allow_pickle=True)

    def sel(self, **kw) -> pd.DataFrame:
        d = self.df
        for k, v in kw.items():
            d = d[d[k].astype(str) == str(v)]
        return d


def _errbar(ax, x, mean, std, m, **kw):
    s = sty(m)
    ax.errorbar(
        x,
        mean,
        yerr=std,
        capsize=3,
        lw=1.8,
        ms=5,
        color=s["color"],
        marker=s["marker"],
        label=s["label"],
        **kw,
    )


# ---------------------------------------------------------------------------
def fig_schematic_and_recovery(R: Results, out: Path):
    """Fig. A/B: what the task is, and what each method recovers."""
    d = R.raw("canonical_sparse_noisy")
    taus = d["taus"]
    mask = d["mask"]
    y = d["y"]
    idx = int(np.argsort(d["KAE-expm__init_rel_l2"])[len(d["sim"]) // 2])  # median case

    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)

    # -- (a) the assimilation window: truth, observations, both reconstructions
    ax = fig.add_subplot(gs[0, :])
    wt = d["KAE-expm__window_taus"]
    ax.plot(
        wt,
        d["KAE-expm__window_curve"][:, idx],
        **{k: v for k, v in sty("KAE-expm").items() if k != "marker"},
        lw=2,
    )
    ax.plot(
        wt,
        d["UNet-4DVar__window_curve"][:, idx],
        **{k: v for k, v in sty("UNet-4DVar").items() if k != "marker"},
        lw=2,
    )
    for t in taus:
        ax.axvline(t, color="0.5", ls="--", lw=0.9, alpha=0.8)
    ax.axvline(
        taus[0], color="0.5", ls="--", lw=0.9, alpha=0.8, label="observation times"
    )
    ax.scatter(
        [0],
        [d["KAE-expm__window_curve"][0, idx]],
        marker="*",
        s=140,
        color="k",
        zorder=5,
        label="analysis time $t_0$ (unobserved)",
    )
    ax.set_xlabel(r"$\tau$  (time after the unobserved analysis state)")
    ax.set_ylabel("relative $L_2$ error")
    ax.set_yscale("log")
    ax.set_title(
        "(a) Future-only assimilation: only $\\tau>0$ is observed; "
        "$u(t_0)$ is inferred",
        loc="left",
    )
    ax.legend(ncol=2, loc="upper left", framealpha=0.9)

    # -- (b) recovered initial field
    ax = fig.add_subplot(gs[1, 0])
    X = np.arange(64)
    ax.plot(
        X,
        d["truth_t0"][idx] if "truth_t0" in d else np.full(64, np.nan),
        color="k",
        lw=2,
        label="true $u(t_0)$",
    )
    ax.plot(
        X,
        d["KAE-expm__analysis"][idx],
        color=sty("KAE-expm")["color"],
        lw=1.8,
        label="KAE",
    )
    ax.plot(
        X,
        d["UNet-4DVar__analysis"][idx],
        color=sty("UNet-4DVar")["color"],
        lw=1.8,
        label="U-Net",
    )
    ax.set_xlabel("grid point")
    ax.set_ylabel("$u$")
    ax.set_title("(b) recovered $u(t_0)$", loc="left")
    ax.legend(fontsize=8)

    # -- (c) paired per-trajectory analysis error
    ax = fig.add_subplot(gs[1, 1])
    a, b = d["KAE-expm__init_rel_l2"], d["UNet-4DVar__init_rel_l2"]
    lim = [min(a.min(), b.min()) * 0.8, max(a.max(), b.max()) * 1.2]
    ax.scatter(a, b, s=22, alpha=0.75, color="#4c72b0", edgecolor="w", linewidth=0.4)
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("KAE analysis rel. $L_2$")
    ax.set_ylabel("U-Net analysis rel. $L_2$")
    ax.set_title(f"(c) paired, $N$={a.size}", loc="left")

    # -- (d) distribution
    ax = fig.add_subplot(gs[1, 2])
    parts = ax.violinplot([a, b], showmedians=True, widths=0.8)
    for pc, m in zip(parts["bodies"], ["KAE-expm", "UNet-4DVar"]):
        pc.set_facecolor(sty(m)["color"])
        pc.set_alpha(0.55)
    if "KAE-expm__ae_floor" in d:
        ax.axhline(float(np.mean(d["KAE-expm__ae_floor"])), **FLOOR)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["KAE", "U-Net"])
    ax.set_ylabel("analysis rel. $L_2$")
    ax.set_yscale("log")
    ax.set_title("(d) distribution", loc="left")
    ax.legend(fontsize=8)

    fig.savefig(out / "fig_da_main.pdf")
    plt.close(fig)


def fig_horizon(R: Results, out: Path):
    """Fig. C/D: accuracy and cost as the assimilation window lengthens."""
    d = R.sel(stage="horizon")
    tim = R.sel(stage="timing")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    for m in ["KAE-expm", "KAE-rk4", "UNet-4DVar"]:
        s = d[d.method == m].sort_values("cond_value", key=lambda c: c.astype(float))
        _errbar(
            ax, s.cond_value.astype(float), s.init_rel_l2_mean, s.init_rel_l2_std, m
        )
    if "ae_floor_mean" in d:
        f = d.dropna(subset=["ae_floor_mean"])
        ax.axhline(float(f.ae_floor_mean.mean()), **FLOOR)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"assimilation horizon $\tau_{\max}$")
    ax.set_ylabel("analysis rel. $L_2$")
    ax.set_title("(a) accuracy vs horizon", loc="left")
    ax.legend(fontsize=8)

    ax = axes[1]
    for m in ["KAE-expm", "KAE-rk4", "UNet-4DVar"]:
        s = tim[tim.method == m].sort_values(
            "cond_value", key=lambda c: c.astype(float)
        )
        st = sty(m)
        ax.plot(
            s.cond_value.astype(float),
            s.ms_per_iter,
            marker=st["marker"],
            color=st["color"],
            label=st["label"],
            lw=1.8,
            ms=5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"assimilation horizon $\tau_{\max}$")
    ax.set_ylabel("ms per optimisation iteration")
    ax.set_title("(b) cost per DA iteration", loc="left")
    ax.legend(fontsize=8)

    ax = axes[2]
    for m in ["KAE-expm", "KAE-rk4", "UNet-4DVar"]:
        s = tim[tim.method == m].sort_values(
            "cond_value", key=lambda c: c.astype(float)
        )
        st = sty(m)
        ax.plot(
            s.cond_value.astype(float),
            s.evals_per_iter_propagation_steps,
            marker=st["marker"],
            color=st["color"],
            label=st["label"],
            lw=1.8,
            ms=5,
        )
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel(r"assimilation horizon $\tau_{\max}$")
    ax.set_ylabel("sequential propagation steps / iteration")
    ax.set_title("(c) intermediate integration steps", loc="left")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "fig_da_horizon.pdf")
    plt.close(fig)


def fig_sweeps(R: Results, out: Path):
    """Fig. F/G/H/I: number of observations, noise, sensor fraction, sampling pattern."""
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6))
    specs = [
        ("nobs", "n_obs", "number of observations", axes[0], "(a)", False),
        (
            "noise",
            "noise_std",
            r"observation noise $\sigma$ (normalised)",
            axes[1],
            "(b)",
            False,
        ),
        (
            "sparsity",
            "obs_frac",
            "fraction of grid points observed",
            axes[2],
            "(c)",
            True,
        ),
    ]
    for stage, cond, xlabel, ax, tag, invert in specs:
        d = R.sel(stage=stage)
        for m in ["KAE-expm", "UNet-4DVar"]:
            s = d[d.method == m].sort_values(
                "cond_value", key=lambda c: c.astype(float)
            )
            _errbar(
                ax, s.cond_value.astype(float), s.init_rel_l2_mean, s.init_rel_l2_std, m
            )
        f = d.dropna(subset=["ae_floor_mean"])
        if len(f):
            ax.axhline(float(f.ae_floor_mean.mean()), **FLOOR)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("analysis rel. $L_2$")
        ax.set_yscale("log")
        if stage != "noise":
            ax.set_xscale("log")
        if invert:
            ax.invert_xaxis()
        ax.set_title(f"{tag} {stage}", loc="left")
    axes[0].legend(fontsize=8)

    ax = axes[3]
    d = R.sel(stage="irregular")
    pats = ["regular", "irregular", "regular_offgrid", "irregular_offgrid"]
    xs = np.arange(len(pats))
    w = 0.36
    for i, m in enumerate(["KAE-expm", "UNet-4DVar"]):
        mu, sd = [], []
        for p in pats:
            s = d[(d.method == m) & (d.cond_value == p)]
            mu.append(float(s.init_rel_l2_mean.iloc[0]) if len(s) else np.nan)
            sd.append(float(s.init_rel_l2_std.iloc[0]) if len(s) else np.nan)
        st = sty(m)
        ax.bar(
            xs + (i - 0.5) * w,
            mu,
            w,
            yerr=sd,
            capsize=3,
            color=st["color"],
            alpha=0.85,
            label=st["label"],
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [
            "regular\n(on grid)",
            "irregular\n(on grid)",
            "regular\n(off grid)",
            "irregular\n(off grid)",
        ],
        fontsize=8,
    )
    ax.set_ylabel("analysis rel. $L_2$")
    ax.set_title("(d) observation-time pattern", loc="left")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "fig_da_sweeps.pdf")
    plt.close(fig)


def fig_propagator(R: Results, out: Path):
    """Fig. E: exact exponential vs the model's own RK4 integration."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))

    # (a) per-trajectory parity at the canonical horizon
    ax = axes[0]
    pooled_x, pooled_y = [], []
    for f in sorted((R.root / "raw").glob("propagator_*.npz")):
        d = np.load(f)
        if "KAE-expm__init_rel_l2" in d and "KAE-rk4__init_rel_l2" in d:
            pooled_x.append(d["KAE-expm__init_rel_l2"])
            pooled_y.append(d["KAE-rk4__init_rel_l2"])
    if pooled_x:
        x, y = np.concatenate(pooled_x), np.concatenate(pooled_y)
        lim = [min(x.min(), y.min()) * 0.85, max(x.max(), y.max()) * 1.15]
        ax.scatter(x, y, s=20, alpha=0.7, color="#4c72b0", edgecolor="w", linewidth=0.4)
        ax.plot(lim, lim, "k--", lw=1, label="$y=x$")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("analysis rel. $L_2$ — exact $e^{K\\tau}$")
        ax.set_ylabel("analysis rel. $L_2$ — RK4 rollout")
        ax.set_title(f"(a) accuracy parity, $N$={x.size}", loc="left")
        ax.legend(fontsize=8)

    # (b) operator-level agreement
    ax = axes[1]
    g = R.summary.get("generator", {}).get("operator_agreement", [])
    if g:
        ax.plot(
            [r["tau"] for r in g],
            [r["rel_fro_diff"] for r in g],
            marker="o",
            color="#8c564b",
            lw=1.8,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"$\|e^{K\tau}-\mathrm{RK4}^{n}\|_F/\|e^{K\tau}\|_F$")
        ax.set_title("(b) same continuous generator", loc="left")

    # (c) runtime
    ax = axes[2]
    tim = R.sel(stage="timing")
    for m in ["KAE-expm", "KAE-rk4"]:
        s = tim[tim.method == m].sort_values(
            "cond_value", key=lambda c: c.astype(float)
        )
        st = sty(m)
        ax.plot(
            s.cond_value.astype(float),
            s.ms_per_iter,
            marker=st["marker"],
            color=st["color"],
            label=st["label"],
            lw=1.8,
            ms=5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"assimilation horizon $\tau_{\max}$")
    ax.set_ylabel("ms per optimisation iteration")
    ax.set_title("(c) cost of the two propagators", loc="left")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "fig_da_propagator.pdf")
    plt.close(fig)


def fig_forecast(R: Results, out: Path):
    """Fig. J: does the assimilated state produce a useful forecast?"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, tag in zip(axes, ["clean", "sparse_noisy"]):
        p = R.root / "raw" / f"canonical_{tag}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        ft = d["KAE-expm__forecast_taus"]
        for m in ["KAE-expm", "UNet-4DVar"]:
            key = f"{m}__forecast_curve"
            if key not in d:
                continue
            c = d[key]  # [n_tau, B]
            st = sty(m)
            ax.plot(ft, c.mean(1), color=st["color"], lw=2, label=st["label"])
            ax.fill_between(
                ft,
                np.percentile(c, 25, axis=1),
                np.percentile(c, 75, axis=1),
                color=st["color"],
                alpha=0.18,
            )
        ax.axhline(
            np.sqrt(2.0),
            color="0.35",
            ls="--",
            lw=1.2,
            label=r"climatological error ($\sqrt{2}$)",
        )
        ax.set_xlabel(r"forecast lead time $\tau$ past $t_0$")
        ax.set_title(
            f"({'a' if tag == 'clean' else 'b'}) {tag.replace('_', ' + ')} "
            "observations",
            loc="left",
        )
    axes[0].set_ylabel("relative $L_2$ error")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_da_forecast.pdf")
    plt.close(fig)


def fig_forward_skill(R: Results, out: Path):
    """Forward-model skill of the two frozen models (no assimilation).

    Context for the DA comparison: it establishes how accurate each forward operator is
    on its own, which is the ceiling any variational analysis built on it can approach.
    """
    p = R.root / "forecast_check.json"
    if not p.exists():
        return
    with open(p) as f:
        fc = json.load(f)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for key, m in [("KAE-expm", "KAE-expm"), ("UNet", "UNet-4DVar")]:
        c = fc["curves"].get(key)
        if not c:
            continue
        tau = np.array(c["tau"])
        mu = np.array(c["mean"])
        sd = np.array(c["std"])
        st = sty(m)
        ax.plot(
            tau,
            mu,
            color=st["color"],
            lw=2,
            label=st["label"].replace(" (exact $e^{K\\tau}$)", ""),
        )
        ax.fill_between(
            tau, np.maximum(mu - sd, 1e-6), mu + sd, color=st["color"], alpha=0.18
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"forecast lead time $\tau$")
    ax.set_ylabel("relative $L_2$ error")
    ax.set_title(
        f"Forward-model skill from the true state ($N$={fc['n_trajectories']})",
        loc="left",
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_da_forward_skill.pdf")
    plt.close(fig)


def fig_convergence_and_gradients(R: Results, out: Path):
    """Fig. E/F: how each variational problem optimises, and how its gradients scale.

    Panels (a-c) show the objective, the analysis error and the gradient norm against
    iteration at several assimilation horizons; panel (d) collects the gradient norm at
    the first iteration against horizon, which is the quantity that would reveal
    exponential amplification through a long autoregressive rollout.
    """
    files = sorted(
        (R.root / "raw").glob("convergence_*.npz"),
        key=lambda f: float(f.stem.split("_")[1]),
    )
    if not files:
        return
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(files)))

    for c, f in zip(cmap, files):
        H = float(f.stem.split("_")[1])
        d = np.load(f)
        for m, ls in [("KAE-expm", "-"), ("UNet-4DVar", "--")]:
            if f"{m}__iter" not in d:
                continue
            it = d[f"{m}__iter"]
            axes[0].plot(
                it,
                d[f"{m}__loss"],
                ls,
                color=c,
                lw=1.5,
                label=f"{m.split('-')[0]} $\\tau_{{\\max}}$={H:g}",
            )
            axes[1].plot(it, d[f"{m}__init_rel_l2"], ls, color=c, lw=1.5)
            axes[2].plot(it, d[f"{m}__grad_norm_per_dim"], ls, color=c, lw=1.5)
    for ax, ylab, title in [
        (axes[0], "objective $J$", "(a) objective"),
        (axes[1], "analysis rel. $L_2$", "(b) analysis error"),
        (axes[2], r"$\|\nabla J\|/\sqrt{d}$", "(c) gradient norm"),
    ]:
        ax.set_xlabel("optimisation iteration")
        ax.set_ylabel(ylab)
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].text(
        0.98,
        0.95,
        "solid: KAE\ndashed: U-Net",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )

    ax = axes[3]
    d = R.sel(stage="convergence")
    for m in ["KAE-expm", "UNet-4DVar"]:
        sdf = d[d.method == m].sort_values("cond_value", key=lambda c: c.astype(float))
        if not len(sdf):
            continue
        st = sty(m)
        dims = float(sdf.n_control_dims.iloc[0])
        ax.plot(
            sdf.cond_value.astype(float),
            sdf.grad_norm_first / np.sqrt(dims),
            marker=st["marker"],
            color=st["color"],
            lw=1.8,
            ms=5,
            label=st["label"],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"assimilation horizon $\tau_{\max}$")
    ax.set_ylabel(r"$\|\nabla J\|/\sqrt{d}$ at iteration 0")
    ax.set_title("(d) gradient scale vs horizon", loc="left")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "fig_da_convergence.pdf")
    plt.close(fig)


def fig_cost_scaling(R: Results, out: Path):
    """Fig. C/D: total DA runtime, per-iteration cost and peak memory vs horizon."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))

    ax = axes[0]
    d = R.sel(stage="horizon")
    for m in ["KAE-expm", "KAE-rk4", "UNet-4DVar"]:
        sdf = d[d.method == m].sort_values("cond_value", key=lambda c: c.astype(float))
        if not len(sdf):
            continue
        st = sty(m)
        ax.plot(
            sdf.cond_value.astype(float),
            sdf.total_s,
            marker=st["marker"],
            color=st["color"],
            label=st["label"],
            lw=1.8,
            ms=5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"assimilation horizon $\tau_{\max}$")
    ax.set_ylabel("total DA wall-clock time (s)")
    ax.set_title("(a) end-to-end DA runtime", loc="left")
    ax.legend(fontsize=8)

    ax = axes[1]
    tim = R.sel(stage="timing")
    for m in ["KAE-expm", "KAE-rk4", "UNet-4DVar"]:
        sdf = tim[tim.method == m].sort_values(
            "cond_value", key=lambda c: c.astype(float)
        )
        if not len(sdf):
            continue
        st = sty(m)
        ax.plot(
            sdf.cond_value.astype(float),
            sdf.ms_per_iter,
            marker=st["marker"],
            color=st["color"],
            label=st["label"],
            lw=1.8,
            ms=5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"assimilation horizon $\tau_{\max}$")
    ax.set_ylabel("ms per optimisation iteration")
    ax.set_title("(b) cost per DA iteration", loc="left")

    ax = axes[2]
    mem = R.sel(stage="memory")
    for m, lab in [
        ("KAE-expm", None),
        ("KAE-rk4", None),
        ("UNet-4DVar", None),
        ("UNet-4DVar-nockpt", "U-Net 4D-Var (no checkpointing)"),
    ]:
        sdf = mem[mem.method == m].sort_values(
            "cond_value", key=lambda c: c.astype(float)
        )
        if not len(sdf):
            continue
        st = sty(m if m != "UNet-4DVar-nockpt" else "UNet-4DVar")
        ax.plot(
            sdf.cond_value.astype(float),
            sdf.peak_mem_MiB,
            marker=st["marker"],
            color=st["color"],
            lw=1.8,
            ms=5,
            ls=":" if m.endswith("nockpt") else "-",
            label=lab or st["label"],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"assimilation horizon $\tau_{\max}$")
    ax.set_ylabel("peak GPU memory (MiB)")
    ax.set_title("(c) memory of one DA iteration", loc="left")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out / "fig_da_cost_scaling.pdf")
    plt.close(fig)


def fig_memory(R: Results, out: Path):
    """Peak memory of one 4D-Var iteration vs horizon — a primary result.

    Plain backpropagation through the autoregressive rollout stores every intermediate
    activation, so its footprint grows linearly with the assimilation window until it
    exhausts the device. Activation checkpointing is the standard remedy: it recomputes
    each segment's forward pass during the backward pass, trading roughly a second forward
    pass for a much smaller footprint, and it is what makes the long-window U-Net baseline
    runnable at all. The exact matrix exponential needs neither, because there is no
    rollout to differentiate through.
    """
    mem = R.sel(stage="memory")
    if not len(mem):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    series = [
        (
            "KAE-expm",
            "-",
            sty("KAE-expm")["color"],
            sty("KAE-expm")["marker"],
            "Continuous KAE, exact $e^{K\\tau}$",
        ),
        (
            "KAE-rk4",
            "-",
            sty("KAE-rk4")["color"],
            sty("KAE-rk4")["marker"],
            "Continuous KAE, RK4 rollout",
        ),
        (
            "UNet-4DVar",
            "-",
            sty("UNet-4DVar")["color"],
            sty("UNet-4DVar")["marker"],
            "U-Net 4D-Var, checkpointed BPTT",
        ),
        ("UNet-4DVar-nockpt", ":", "#8c1a1a", "v", "U-Net 4D-Var, plain BPTT"),
    ]
    for col, ax, title in [
        ("peak_mem_MiB", axes[0], "(a) peak allocated memory"),
        ("activation_mem_MiB", axes[1], "(b) memory above resident weights"),
    ]:
        if col not in mem.columns:
            continue
        for m, ls, c, mk, lab in series:
            sdf = mem[mem.method == m].sort_values(
                "cond_value", key=lambda x: x.astype(float)
            )
            if not len(sdf):
                continue
            x = sdf.cond_value.astype(float).values
            y = sdf[col].values.astype(float)
            ok = ~np.isnan(y)
            ax.plot(x[ok], y[ok], ls, color=c, marker=mk, lw=1.9, ms=5.5, label=lab)
            if (~ok).any():  # mark where the run exhausted the device
                ylim = np.nanmax(y) if np.isfinite(np.nanmax(y)) else 1.0
                ax.scatter(
                    x[~ok],
                    np.full((~ok).sum(), ylim),
                    marker="x",
                    s=70,
                    color=c,
                    zorder=6,
                )
                ax.annotate(
                    "out of memory",
                    (x[~ok][0], ylim),
                    textcoords="offset points",
                    xytext=(4, 8),
                    fontsize=7.5,
                    color=c,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"assimilation horizon $\tau_{\max}$")
        ax.set_ylabel("MiB")
        ax.set_title(title, loc="left")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_da_memory.pdf")
    plt.close(fig)


def fig_sda(R: Results, out: Path):
    """Fig. K: score-based DA — prior vs posterior, and the three-way comparison.

    Panel (a) is the gate that makes this a data-assimilation method rather than a
    generative model: error at frames that were *never observed* must fall from prior to
    posterior, which is only possible if the learned trajectory prior propagates the
    observational constraint through time.
    """
    import pandas as _pd

    f = R.root / "sda_final.csv"
    if not f.exists():
        return
    sda = _pd.read_csv(f)
    with open(R.root / "sda_summary.json") as fh:
        ssum = json.load(fh)
    df = R.df
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.7))

    # (a) prior vs posterior at observed and never-observed frames
    ax = axes[0]
    tags = ["clean", "sparse_noisy"]
    xs = np.arange(len(tags))
    w = 0.2
    pv = ssum["prior_vs_posterior"]
    for i, (key, lab, col) in enumerate(
        [
            ("obs_frames_prior", "prior, observed frames", "#bbbbbb"),
            ("obs_frames_posterior", "posterior, observed frames", "#9467bd"),
            ("unobs_frames_prior", "prior, never-observed", "#dddddd"),
            ("unobs_frames_posterior", "posterior, never-observed", "#c5b0d5"),
        ]
    ):
        ax.bar(
            xs + (i - 1.5) * w,
            [pv[t][key] for t in tags],
            w,
            color=col,
            edgecolor="0.3",
            linewidth=0.5,
            label=lab,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(["clean", "sparse+noisy"])
    ax.set_ylabel("relative $L_2$ error")
    ax.set_title("(a) SDA: observations propagate\nto unobserved frames", loc="left")
    ax.legend(fontsize=6.5)

    # (b) analysis accuracy, three methods
    ax = axes[1]
    meths = [("KAE-expm", "KAE"), ("UNet-4DVar", "U-Net"), ("SDA", "SDA")]
    for i, (m, lab) in enumerate(meths):
        vals, errs = [], []
        for t in tags:
            if m == "SDA":
                r = sda[(sda.cond_value == t) & (sda.method == "SDA")]
                vals.append(float(r.analysis_rel_l2_mean.iloc[0]) if len(r) else np.nan)
                errs.append(float(r.analysis_rel_l2_std.iloc[0]) if len(r) else np.nan)
            else:
                r = df[
                    (df.stage == "canonical") & (df.cond_value == t) & (df.method == m)
                ]
                vals.append(float(r.init_rel_l2_mean.iloc[0]) if len(r) else np.nan)
                errs.append(float(r.init_rel_l2_std.iloc[0]) if len(r) else np.nan)
        ax.bar(
            xs + (i - 1) * 0.26,
            vals,
            0.26,
            yerr=errs,
            capsize=3,
            color=sty(m)["color"],
            alpha=0.85,
            label=lab,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(["clean", "sparse+noisy"])
    ax.set_ylabel("analysis rel. $L_2$")
    ax.set_yscale("log")
    ax.set_title("(b) analysis accuracy", loc="left")
    ax.legend(fontsize=8)

    # (c) cost: total wall-clock for the same 64 problems
    ax = axes[2]
    cost = []
    for m, lab in meths:
        if m == "SDA":
            r = sda[(sda.cond_value == "clean") & (sda.method == "SDA")]
            cost.append((lab, float(r.wall_s.iloc[0])))
        else:
            r = df[
                (df.stage == "canonical")
                & (df.cond_value == "clean")
                & (df.method == m)
            ]
            cost.append((lab, float(r.total_s.iloc[0])))
    ax.bar(
        [c[0] for c in cost],
        [c[1] for c in cost],
        color=[sty(m)["color"] for m, _ in meths],
        alpha=0.85,
    )
    for i, (lab, v) in enumerate(cost):
        ax.text(i, v * 1.08, f"{v:.0f}s", ha="center", fontsize=8)
    ax.set_ylabel("total wall-clock (s), 64 problems")
    ax.set_yscale("log")
    ax.set_title("(c) cost", loc="left")

    # (d) SDA ensemble calibration
    ax = axes[3]
    for t, c in zip(tags, ["#9467bd", "#c5b0d5"]):
        r = sda[(sda.cond_value == t) & (sda.method == "SDA")]
        if not len(r):
            continue
        ax.bar(
            t.replace("_", "+"),
            float(r.coverage_95.iloc[0]),
            0.5,
            color=c,
            edgecolor="0.3",
        )
        ax.text(
            t.replace("_", "+"),
            float(r.coverage_95.iloc[0]) + 0.02,
            f"{float(r.coverage_95.iloc[0]):.3f}\nspread/err "
            f"{float(r.spread_error_ratio.iloc[0]):.2f}",
            ha="center",
            fontsize=7.5,
        )
    ax.axhline(0.95, color="k", ls="--", lw=1.2, label="nominal 95%")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("empirical 95% coverage")
    ax.set_title("(d) posterior calibration", loc="left")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "fig_da_sda.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("da_results_v2"))
    ap.add_argument("--out", type=Path, default=Path("../iclr_2027/figures/da"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    R = Results(args.results)

    made = []
    for fn in (
        fig_schematic_and_recovery,
        fig_horizon,
        fig_sweeps,
        fig_propagator,
        fig_forecast,
        fig_forward_skill,
        fig_convergence_and_gradients,
        fig_cost_scaling,
        fig_memory,
        fig_sda,
    ):
        try:
            fn(R, args.out)
            made.append(fn.__name__)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {fn.__name__}: {type(e).__name__}: {e}")
    print("figures written:", made, "->", args.out)


if __name__ == "__main__":
    main()
