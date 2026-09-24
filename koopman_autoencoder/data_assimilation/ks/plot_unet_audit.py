# ruff: noqa: F841
"""Part 1 figure: is the U-Net space-time recovery plausible, or is it a bug?

Four panels, all on the SAME assimilation problems:
  (a) full-field error vs time, observation times marked
  (b) low-wavenumber (k=1..12) error vs time  -- the part that carries the energy
  (c) high-wavenumber (k>=13) error vs time   -- the part that carries ~none
  (d) Fourier error spectrum at t0 and at the first unobserved step after t0
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DT = 0.1
OUT = Path("da_results_sda_paper")
STYLE = {
    "UNet": dict(color="#d6604d", label="U-Net 4D-Var"),
    "KAE-expm": dict(color="#1b7837", label="Continuous KAE"),
    "SDA": dict(color="#762a83", label="Score-based DA (per draw)"),
}


def figure(res=None, npz=None, axes=None):
    res = res or json.loads((OUT / "unet_spacetime_audit.json").read_text())
    npz = npz if npz is not None else np.load(OUT / "unet_audit_fields.npz")
    obs = res["geometry"]["obs_frames"]
    T = len(res["per_frame_rel_l2"]["UNet"])
    t = np.arange(T) * DT
    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(12.4, 7.2))
    axes = np.asarray(axes).ravel()

    def mark(ax):
        for k in obs:
            ax.axvline(k * DT, color="0.72", lw=1.0, zorder=0)
        ax.axvline(0.0, color="k", ls=":", lw=1.2, zorder=0)
        ax.grid(alpha=0.3, which="both")

    for m, st in STYLE.items():
        axes[0].semilogy(t, res["per_frame_rel_l2"][m], lw=1.8, **st)
    mark(axes[0])
    axes[0].set(
        xlabel="time since $t_0$ (t.u.)",
        ylabel=r"rel-$L_2$",
        title="(a) full-field error.  grey = observation times, dotted = $t_0$",
    )
    axes[0].legend(fontsize=8)

    for j, (key, ttl) in enumerate(
        [
            ("low_k1_12", r"(b) low $k=1..12$ (carries the energy)"),
            ("high_k13_32", r"(c) high $k\geq13$ (carries ~0 energy)"),
        ]
    ):
        ax = axes[1 + j]
        for m, st in STYLE.items():
            ax.semilogy(t, res["low_high_vs_time"][m][key], lw=1.8, **st)
        mark(ax)
        ax.set(xlabel="time since $t_0$ (t.u.)", ylabel=r"band rel-$L_2$", title=ttl)
        ax.legend(fontsize=8)

    ax = axes[3]
    tru, X = npz["truth"], npz["truth"].shape[-1]
    kk = np.fft.rfftfreq(X, d=1.0 / X)
    ft = np.abs(np.fft.rfft(tru, axis=-1))
    for m, st in STYLE.items():
        p = npz[f"{m}__traj"]
        for k, ls, a in [(0, "-", 1.0), (1, "--", 0.55)]:
            fe = np.abs(np.fft.rfft(p[:, k] - tru[:, k], axis=-1)).mean(0)
            ax.semilogy(
                kk,
                fe,
                ls=ls,
                lw=1.6,
                alpha=a,
                color=st["color"],
                label=f"{st['label']} @ {'$t_0$' if k == 0 else 'frame 1'}",
            )
    ax.semilogy(
        kk,
        ft[:, 0].mean(0),
        color="k",
        lw=2.4,
        alpha=0.35,
        label="truth spectrum at $t_0$",
    )
    ax.set(
        xlabel="wavenumber $k$",
        ylabel="mean |FFT| of the error",
        title="(d) where the error lives, at $t_0$ and one step later",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5, ncol=2)
    return axes


def summary(res=None) -> str:
    res = res or json.loads((OUT / "unet_spacetime_audit.json").read_text())
    c = res["independent_rollout_check"]
    pf = res["per_frame_rel_l2"]
    obs = res["geometry"]["obs_frames"]
    L = [
        f"rollout check: max|plotted - independently re-rolled| = "
        f"{c['max_abs_difference']:.3e}  (matches: {c['matches']}, "
        f"deterministic: {c['deterministic_given_x0']})",
        "",
        f"{'frame':>5s} {'t':>6s} {'obs':>4s} " + " ".join(f"{m:>10s}" for m in STYLE),
    ]
    for k in sorted({0, 1, 2, 3, 5, 9, 15, 22, 40, 63, 70, len(pf["UNet"]) - 1}):
        if k >= len(pf["UNet"]):
            continue
        L.append(
            f"{k:5d} {k * DT:6.1f} {'Y' if k in obs else '-':>4s} "
            + " ".join(f"{pf[m][k]:10.5f}" for m in STYLE)
        )
    s = res["spectral"]
    L += [
        "",
        "share of TOTAL squared error by band:",
        f"{'frame':>26s} {'method':>9s} "
        + " ".join(f"{b:>16s}" for b in ("k1-4", "k5-12", "k13-32")),
    ]
    for lbl in (
        "t0",
        "first_unobserved_after_t0",
        "first_observation",
        "later_frame_40",
    ):
        for m in STYLE:
            r = s[lbl][m]
            L.append(
                f"{lbl:>26s} {m:>9s} "
                + " ".join(
                    f"{100 * r[b]['share_of_total_sq_error']:8.1f}% {r[b]['rel_err_in_band']:6.3f}"
                    for b in ("k1-4", "k5-12", "k13-32")
                )
            )
    return "\n".join(L)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    fig, ax = plt.subplots(2, 2, figsize=(12.4, 7.2))
    figure(axes=ax)
    fig.tight_layout()
    fig.savefig(OUT / "unet_spacetime_audit.png", dpi=150)
    print(summary())


# ---------------------------------------------------------------------------
def nullspace_figure(rep=None, axes=None):
    """Why 4D-Var stalls at t_0 while nailing every other frame."""
    rep = rep or json.loads((OUT / "unet_nullspace.json").read_text())
    sp = rep["spectrum"]
    k = np.asarray(sp["k"])
    ana = np.asarray(sp["analytic_one_step_multiplier_exp((q^2-q^4)dt)"])
    mea = np.asarray(sp["measured_one_step_multiplier"])
    err = np.asarray(sp["error_amplitude_at_t0"])
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))
    a0, a1 = np.asarray(axes).ravel()[:2]

    a0.semilogy(
        k,
        err,
        color="#d6604d",
        lw=2.0,
        marker="o",
        ms=3.5,
        label=r"$|\widehat{\delta}(k)|$, $\delta = x_0^{\rm rec}-x_0^{\rm true}$",
    )
    a0.set(
        xlabel="wavenumber $k$",
        ylabel="error amplitude at $t_0$",
        title="(a) the analysis error is one narrow band",
    )
    kpk = int(k[np.argmax(err)])
    a0.axvline(kpk, color="0.4", ls="--", lw=1.2)
    a0.text(kpk, err.max(), f"  $k={kpk}$", fontsize=9, va="top")
    a0.grid(alpha=0.3, which="both")
    a0.legend(fontsize=8)

    a1.semilogy(
        k,
        np.maximum(ana, 1e-8),
        color="k",
        lw=2.0,
        label=r"analytic KS  $e^{(q^2-q^4)\Delta t}$,  $q=2\pi k/L$",
    )
    a1.semilogy(
        k,
        mea,
        color="#d6604d",
        lw=2.0,
        marker="o",
        ms=3.5,
        label=r"measured along $\delta$:  $|F_\theta(x_0+\delta)-F_\theta(x_0)|$",
    )
    p = rep["perturbation"]
    a1.axhline(
        p["random_direction_contraction_mean"],
        color="#1b7837",
        ls=":",
        lw=1.8,
        label=f"random direction: {p['random_direction_contraction_mean']:.3f}",
    )
    a1.axvline(kpk, color="0.4", ls="--", lw=1.2)
    a1.set(
        xlabel="wavenumber $k$",
        ylabel="one-step multiplier",
        ylim=(1e-3, 3),
        title="(b) that band is what the model damps hardest",
    )
    a1.grid(alpha=0.3, which="both")
    a1.legend(fontsize=7.5, loc="lower left")
    return axes


def nullspace_summary(rep=None) -> str:
    rep = rep or json.loads((OUT / "unet_nullspace.json").read_text())
    p, c = rep["perturbation"], rep["cost_function_blindness"]
    return "\n".join(
        [
            f"delta = x0_recovered - x0_true,   |delta| / |x0| = {p['rel_norm_at_t0']:.4f}",
            "",
            f"  one step of F_theta contracts delta by      "
            f"{p['one_step_contraction_mean']:.4f} +/- {p['one_step_contraction_std']:.4f}",
            f"  a RANDOM direction of the same norm by      "
            f"{p['random_direction_contraction_mean']:.4f}",
            f"  -> the analysis error is {p['selectivity']:.1f}x more strongly damped than a "
            f"generic direction",
            "",
            f"  4D-Var objective at the RECOVERED x0        {c['J_at_recovered_x0']:.4e}",
            f"  4D-Var objective at the TRUE      x0        {c['J_at_true_x0']:.4e}",
            f"  -> the cost STILL prefers the truth by {c['ratio']:.1f}x "
            f"(optimiser_prefers_recovered_to_truth = {c['optimiser_prefers_recovered_to_truth']})",
        ]
    )


# ---------------------------------------------------------------------------
def fig_zoom(n_frames: int = 10, rep=None, axes=None):
    """The first few steps after t_0, magnified.

    The full-window figure is dominated by the long-time behaviour and hides what happens
    immediately after t_0 -- which is where the U-Net's behaviour is surprising: it is the
    WORST method at t_0 and the best one step later.  This panel shows only the first
    ``n_frames`` frames, on both a log and a linear axis, with the per-frame numbers
    printed so the drop can be read off directly.
    """
    rep = rep or json.loads((OUT / "unet_spacetime_audit.json").read_text())
    pf = rep["per_frame_rel_l2"]
    obs = set(rep["geometry"]["obs_frames"])
    ms = [m for m in ["UNet", "KAE-expm", "SDA"] if m in pf]
    n = min(n_frames, len(pf[ms[0]]))
    x = np.arange(n)
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    a0, a1 = np.asarray(axes).ravel()[:2]

    for ax, logy in ((a0, True), (a1, False)):
        for m in ms:
            st = STYLE.get(m, {})
            ax.plot(
                x,
                np.asarray(pf[m])[:n],
                marker="o",
                ms=4.5,
                lw=1.8,
                color=st.get("color"),
                label=st.get("label", m),
            )
        for f in sorted(obs):
            if f < n:
                ax.axvline(f, color="0.55", lw=1.1, zorder=0)
        ax.axvline(0, color="k", ls=":", lw=1.4, zorder=0)
        ax.set(
            xlabel="frame after $t_0$",
            ylabel=r"rel-$L_2$",
            xticks=x[:: max(1, n // 10)],
        )
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.set_title("(a) log scale" if logy else "(b) linear scale")
    a0.legend(fontsize=7.5)
    a0.text(
        0.02,
        0.02,
        "grey = observation times;  dotted = $t_0$",
        transform=a0.transAxes,
        fontsize=6.8,
        color="0.35",
    )
    return axes


def zoom_table(n_frames: int = 10, rep=None) -> str:
    rep = rep or json.loads((OUT / "unet_spacetime_audit.json").read_text())
    pf = rep["per_frame_rel_l2"]
    obs = set(rep["geometry"]["obs_frames"])
    ms = [m for m in ["UNet", "KAE-expm", "SDA"] if m in pf]
    n = min(n_frames, len(pf[ms[0]]))
    L = [
        f"first {n} frames after t_0  (obs at {sorted(obs)})",
        "",
        f"{'frame':>5s} {'obs':>4s} "
        + " ".join(f"{m:>11s}" for m in ms)
        + f"  {'UNet vs prev':>13s}",
    ]
    prev = None
    for i in range(n):
        r = f"{i:5d} {'Y' if i in obs else '-':>4s} "
        r += " ".join(f"{pf[m][i]:11.5f}" for m in ms)
        cur = pf["UNet"][i]
        r += f"  {('' if prev is None else f'{cur / prev:12.3f}x'):>13s}"
        prev = cur
        L.append(r)
    u = pf["UNet"]
    L += [
        "",
        f"the U-Net's error falls {u[0] / u[1]:.1f}x in ONE step "
        f"({u[0]:.4f} -> {u[1]:.4f}) and {u[0] / u[3]:.0f}x by frame 3.",
        "That is the signature of an analysis error living in a strongly damped",
        "direction: it is destroyed by the dynamics rather than corrected by them.",
    ]
    return "\n".join(L)


# ---------------------------------------------------------------------------
def fig_fields_first_steps(n_frames: int = 10, traj: int = 0, axes=None):
    """The actual KS fields over the first steps: truth vs each method's forecast.

    Every curve after frame 0 is a genuine forecast: each method's ASSIMILATED state at
    $t_0$ rolled forward under its own dynamics.  Nothing is re-observed, so the panels
    show how an analysis error at $t_0$ propagates -- which is the thing the error curve
    can only summarise.
    """
    d = np.load(OUT / "unet_audit_fields.npz")
    obs = set(int(o) for o in d["obs_frames"])
    ms = [m for m in ["UNet", "KAE-expm", "SDA"] if f"{m}__traj" in d.files]
    n = min(n_frames, d["truth"].shape[1])
    x = np.linspace(-1, 1, d["truth"].shape[-1])
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    if axes is None:
        _, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(3.1 * ncol, 2.5 * nrow),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
    axes = np.asarray(axes).reshape(-1)

    tru = d["truth"][traj]
    lim = 1.25 * np.abs(tru[:n]).max()
    for f in range(n):
        a = axes[f]
        a.plot(
            x,
            tru[f],
            lw=5.0,
            color="k",
            alpha=0.20,
            solid_capstyle="round",
            label="truth" if f == 0 else None,
        )
        for m in ms:
            st = STYLE.get(m, {})
            a.plot(
                x,
                d[f"{m}__traj"][traj, f],
                lw=1.5,
                color=st.get("color"),
                label=(st.get("label", m) if f == 0 else None),
            )
        e = {
            m: np.linalg.norm(d[f"{m}__traj"][traj, f] - tru[f])
            / np.linalg.norm(tru[f])
            for m in ms
        }
        a.set_title(
            (
                f"frame {f}"
                + ("  ($t_0$)" if f == 0 else "")
                + ("  [observed]" if f in obs else "")
            )
            + "\n"
            + "  ".join(f"{m.split('-')[0]} {e[m]:.3f}" for m in ms),
            fontsize=7.5,
        )
        a.set_ylim(-lim, lim)
        a.grid(alpha=0.3)
        if f % ncol == 0:
            a.set_ylabel("$u$")
        if f >= n - ncol:
            a.set_xlabel("space $x$")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=6.5, loc="upper right")
    return axes


# ---------------------------------------------------------------------------
def fig_spacetime_zoom(n_frames: int = 10, traj: int = 0, fig=None):
    """Space-time recovery over the FIRST few steps only, in the Part F style.

    Same layout as the full-window space-time figure -- top row the fields, bottom row
    |error| on a shared scale -- but restricted to the frames just after $t_0$, where the
    interesting behaviour is compressed into the first column or two and is invisible at
    full extent.
    """
    d = np.load(OUT / "unet_audit_fields.npz")
    obs = [int(o) for o in d["obs_frames"]]
    ms = [m for m in ["UNet", "KAE-expm", "SDA"] if f"{m}__traj" in d.files]
    n = min(n_frames, d["truth"].shape[1])
    tru = d["truth"][traj, :n]  # [n, X]
    X = tru.shape[-1]
    dt = 0.1
    ext = [-dt / 2, (n - 0.5) * dt, -1.0, 1.0]  # time on x, space on y

    vmin, vmax = float(tru.min()), float(tru.max())
    allerr = np.concatenate(
        [np.abs(d[f"{m}__traj"][traj, :n] - tru).ravel() for m in ms]
    )
    emax = max(float(np.nanpercentile(allerr, 99.0)), 1e-6)

    ncol = 1 + len(ms)
    if fig is None:
        fig, ax = plt.subplots(2, ncol, figsize=(3.4 * ncol, 5.4))
    else:
        ax = fig.subplots(2, ncol)
    ax = np.atleast_2d(ax)

    def _mark(a):
        for o in obs:
            if o < n:
                a.axvline(o * dt, color="k", lw=1.0, alpha=0.55)
        a.axvline(0.0, color="k", ls=":", lw=1.4)
        a.set_xlabel("time after $t_0$")

    im = ax[0, 0].imshow(
        tru.T,
        origin="lower",
        aspect="auto",
        extent=ext,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax[0, 0].set_title("TRUTH", fontsize=10)
    ax[0, 0].set_ylabel("space $x$")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.02)
    _mark(ax[0, 0])
    ax[1, 0].axis("off")
    ax[1, 0].text(
        0.5,
        0.5,
        f"frames 0–{n - 1}\nobservations at {obs}\n"
        r"dotted = $t_0$ (never observed)"
        "\n\nbottom row: |error|, shared scale,\nclipped at the 99th percentile",
        ha="center",
        va="center",
        fontsize=8,
        transform=ax[1, 0].transAxes,
    )

    for c, m in enumerate(ms, start=1):
        pr = d[f"{m}__traj"][traj, :n]
        im = ax[0, c].imshow(
            pr.T,
            origin="lower",
            aspect="auto",
            extent=ext,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax[0, c].set_title(
            STYLE.get(m, {}).get("label", m),
            fontsize=9,
            color=STYLE.get(m, {}).get("color"),
        )
        fig.colorbar(im, ax=ax[0, c], fraction=0.046, pad=0.02)
        _mark(ax[0, c])
        im2 = ax[1, c].imshow(
            np.abs(pr - tru).T,
            origin="lower",
            aspect="auto",
            extent=ext,
            cmap="magma",
            vmin=0,
            vmax=emax,
        )
        e0 = np.linalg.norm(pr[0] - tru[0]) / np.linalg.norm(tru[0])
        ax[1, c].set_title(f"|error|   rel-$L_2$ at $t_0$ = {e0:.3f}", fontsize=8.5)
        fig.colorbar(im2, ax=ax[1, c], fraction=0.046, pad=0.02)
        _mark(ax[1, c])
        if c == 1:
            ax[1, c].set_ylabel("space $x$")
    return fig, ax


# ---------------------------------------------------------------------------
def fig_unet_true_vs_analysis(
    n_frames: int = 10,
    traj: int = 0,
    fig=None,
    unet_ckpt="model_outputs_ks/unet1d/rollout10_extended2/" "best_model.pth",
):
    """U-Net only: what the model costs, versus what the assimilation costs.

    Three fields, all over the same frames:
        TRUTH
        forecast from the TRUE x_0        -- pure MODEL error, perfect initial condition
        forecast from the ASSIMILATED x_0 -- model error PLUS analysis error

    and the two error maps on a shared scale.  The middle column is the floor: no
    assimilation can do better than the model itself.  The difference between the two
    error maps is exactly what the 4D-Var solve costs.
    """
    import torch
    from data_assimilation.ks.models_io import load_unet
    from models.dataloader import KS_MEAN, KS_STD

    d = np.load(OUT / "unet_audit_fields.npz")
    obs = [int(o) for o in d["obs_frames"]]
    n = min(n_frames, d["truth"].shape[1])
    tru = d["truth"][traj, :n]  # physical units
    assim = d["UNet__traj"][traj, :n]  # rollout from the assimilated x0

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_unet(Path(unet_ckpt), dev)
    mu, sd = float(KS_MEAN["u"]), float(KS_STD["u"])
    x = torch.as_tensor(
        (d["truth"][traj, 0] - mu) / sd, device=dev, dtype=torch.float32
    ).unsqueeze(0)
    seq = [x]
    with torch.no_grad():
        for _ in range(n - 1):
            seq.append(net(seq[-1].unsqueeze(1)).squeeze(1))
    perfect = (torch.cat(seq, 0).cpu().numpy() * sd) + mu  # [n, X], physical

    dt = 0.1
    ext = [-dt / 2, (n - 0.5) * dt, -1.0, 1.0]
    vmin, vmax = float(tru.min()), float(tru.max())
    emax = max(
        float(
            np.nanpercentile(
                np.concatenate(
                    [np.abs(perfect - tru).ravel(), np.abs(assim - tru).ravel()]
                ),
                99.5,
            )
        ),
        1e-6,
    )

    if fig is None:
        fig, ax = plt.subplots(2, 3, figsize=(12.6, 6.0))
    else:
        ax = fig.subplots(2, 3)
    ax = np.atleast_2d(ax)

    def _mark(a):
        for o in obs:
            if o < n:
                a.axvline(o * dt, color="k", lw=1.0, alpha=0.6)
        a.axvline(0.0, color="k", ls=":", lw=1.5)
        a.set_xlabel("time after $t_0$")

    panels = [
        (tru, "TRUTH", None),
        (perfect, "U-Net forecast from the TRUE $x_0$", perfect - tru),
        (assim, "U-Net forecast from the ASSIMILATED $x_0$", assim - tru),
    ]
    for c, (field, title, err) in enumerate(panels):
        im = ax[0, c].imshow(
            field.T,
            origin="lower",
            aspect="auto",
            extent=ext,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax[0, c].set_title(title, fontsize=9.5)
        fig.colorbar(im, ax=ax[0, c], fraction=0.046, pad=0.02)
        _mark(ax[0, c])
        if c == 0:
            ax[0, c].set_ylabel("space $x$")
        if err is None:
            ax[1, c].axis("off")
            ax[1, c].text(
                0.5,
                0.5,
                f"frames 0–{n - 1}\nobservations at {obs}\n"
                r"dotted = $t_0$, never observed"
                "\n\nmiddle column = MODEL error alone\n"
                "right column = model + ANALYSIS error\n"
                "(shared colour scale)",
                ha="center",
                va="center",
                fontsize=8.5,
                transform=ax[1, c].transAxes,
            )
            continue
        im2 = ax[1, c].imshow(
            np.abs(err).T,
            origin="lower",
            aspect="auto",
            extent=ext,
            cmap="magma",
            vmin=0,
            vmax=emax,
        )
        r0 = np.linalg.norm(err[0]) / np.linalg.norm(tru[0])
        rl = np.linalg.norm(err[-1]) / np.linalg.norm(tru[-1])
        ax[1, c].set_title(
            f"|error|   rel-$L_2$: {r0:.4f} at $t_0$, " f"{rl:.4f} at frame {n - 1}",
            fontsize=8.5,
        )
        fig.colorbar(im2, ax=ax[1, c], fraction=0.046, pad=0.02)
        _mark(ax[1, c])
        if c == 1:
            ax[1, c].set_ylabel("space $x$")
    return fig, ax, {"perfect": perfect, "assim": assim, "truth": tru}
