"""Field-level figures for the readouts of the collaborator report.

fig_readouts.pdf  Tra_long, first simulation: pressure at t = 60, 120, 240 and the vorticity of
                  the wake at t = 240, for the baselines and every CT-KAE readout (reference),
                  including SDEdit refinement of the single decode with ACDM (t0 = 8).
fig_spectra.pdf   |v| spectra (prediction and truth masked alike) and their ratio to the truth:
                  Tra_int 0-60, Tra_long 0-240, Tra_long 180-240.
fig_da_fields.pdf 4D-Var analyses at t0 of one df1 headline problem (the median-error problem of
                  the plain reference): U-Net, FNO and both CT-KAEs (seed 2) plain, mirror and
                  mirror + LPM; pressure, pressure error and wake vorticity.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).parent
IMPL = HERE.parents[2]
KAE = IMPL / "koopman_autoencoder"
AUTOREG = Path("/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion")
for p in (KAE, IMPL / "audit_report/scripts", AUTOREG / "src"):
    sys.path.insert(0, str(p))

import spectral_analysis as SA  # noqa: E402
from benchmark_suite import UNTIMED, load_gt, load_pred  # noqa: E402
from evaluate_dist_metrics import (  # noqa: E402
    load_ground_truth,
    load_prediction,
    resolve_pred,
)
from make_collab import BLUE, INK, INK2, MUTED, ORANGE  # noqa: E402

mpl.rcParams["pdf.fonttype"] = 42
FIELDS = IMPL / "audit_report/results/paper_ctkae/fields"
SDE = Path(
    "/tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/"
    "cfcfc896-3b58-4c33-b471-e1efcfb360e4/scratchpad/pp3px/sde"
)
DA = KAE / "model_outputs_ctkae"
BASE_DA = Path("/home/rg625/mnt/ocean_forecasting/koopman_autoencoder/da_results_tra")
P = 3  # pressure, ACDM layout [v_x, v_y, rho, p, Ma] x [X, Y]
GREEN, PURPLE, GREY = "#2e9e5b", "#8a4fbf", "#6f6e69"


def vorticity(vx, vy, axis_x, axis_y):
    return np.gradient(vy, axis=axis_x) - np.gradient(vx, axis=axis_y)


def label(ax, txt, bold=False):
    ax.text(
        0.97,
        0.05,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        fontweight="bold" if bold else "normal",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.8),
    )


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ------------------------------------------------------------------------ readout gallery
def fig_readouts():
    case, sim, ts = "longer", 0, [60, 120, 240]
    gt, fluid = load_gt(case)
    S, T = gt.shape[:2]
    src = {
        "ACDM": UNTIMED / case / "acdm-r20.npz",
        "U-Net$_{ut}$": UNTIMED / case / "unet-m8.npz",
        "Reference (single)": FIELDS / case / "ref.npz",
        "+ post-proc.": FIELDS / case / "ref+pp.npz",
        "+ sharp. mirror": FIELDS / case / "ref+sharp.npz",
        "+ LPM mirror": FIELDS / case / "ref+lpm.npz",
        "single + SDEdit": SDE / case / "ref@sde8.npz",
    }
    fr = {"Simulation": gt[sim]}
    for lab, path in src.items():
        fr[lab] = (load_pred(path, S, T, False) * fluid)[0, sim]
    rows = list(fr)
    vmin, vmax = np.percentile(gt[sim, [t - 1 for t in ts], P], [1, 99])
    wz = {
        k: vorticity(v[ts[-1] - 1, 0], v[ts[-1] - 1, 1], 0, 1) * fluid[0]
        for k, v in fr.items()
    }
    wl = np.percentile(np.abs(wz["Simulation"]), 99)
    mse = {
        k: [
            ((v[t - 1, P] - gt[sim, t - 1, P]) ** 2 * fluid[0]).mean() * 1e3 for t in ts
        ]
        for k, v in fr.items()
    }
    fig, axs = plt.subplots(
        len(rows),
        5,
        figsize=(7.2, 8.4),
        gridspec_kw={"width_ratios": [2, 2, 2, 0.08, 1.25]},
    )
    for i, k in enumerate(rows):
        for j, t in enumerate(ts):
            ax = axs[i, j]
            ax.imshow(
                fr[k][t - 1, P].T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax
            )
            if k != "Simulation":
                best = min(mse[q][j] for q in rows[1:])
                label(ax, f"{mse[k][j]:.1f}", round(mse[k][j], 1) == round(best, 1))
            clean(ax)
            if i == 0:
                ax.set_title(f"pressure, $t={t}$", fontsize=7, color=INK)
        axs[i, 3].axis("off")
        ax = axs[i, 4]
        ax.imshow(wz[k][20:84, :].T, origin="lower", cmap="PuOr_r", vmin=-wl, vmax=wl)
        clean(ax)
        if i == 0:
            ax.set_title(f"vorticity (wake), $t={ts[-1]}$", fontsize=7, color=INK)
        axs[i, 0].set_ylabel(k, fontsize=6.5, rotation=0, ha="right", va="center")
    fig.subplots_adjust(
        left=0.15, right=0.99, top=0.95, bottom=0.01, wspace=0.03, hspace=0.05
    )
    fig.savefig(HERE / "fig_readouts.pdf", dpi=250)


# ------------------------------------------------------------------------ spectra
def fig_spectra():
    mods = {
        "ACDM": ("acdm-r20.npz", GREY, "--"),
        "U-Net$_{ut}$": ("unet-m8.npz", GREY, ":"),
        "Reference (single)": ("ref", INK2, "-"),
        "+ post-proc.": ("ref+pp", ORANGE, "-"),
        "+ sharp. mirror": ("ref+sharp", GREEN, "-"),
        "+ LPM mirror": ("ref+lpm", BLUE, "-"),
    }
    panels = [
        ("interp", 0, 60, "Tra$_{int}$, $t\\leq60$"),
        ("longer", 0, 240, "Tra$_{long}$, $t\\leq240$"),
        ("longer", 180, 240, "Tra$_{long}$, $180<t\\leq240$"),
    ]
    fig, axs = plt.subplots(
        2, 3, figsize=(7.2, 3.9), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]}
    )
    cache: dict = {}
    for j, (case, a, b, title) in enumerate(panels):
        if case not in cache:
            gt_full, mask = load_ground_truth(case)
            vp = {}
            for lab, (fn, _, _) in mods.items():
                f = fn if fn.endswith(".npz") else str(FIELDS / case / f"{fn}.npz")
                raw = load_prediction(case, resolve_pred(case, f))
                m_, e, s_ = raw.shape[:3]
                cs, ct = min(gt_full.shape[0], s_), min(gt_full.shape[1], raw.shape[3])
                fl = mask[:cs, None, None]
                pred = (raw[:, :, :cs, :ct] * fl).reshape(
                    m_ * e * cs, ct, *raw.shape[4:]
                )
                vp[lab] = SA.velocity_magnitude(pred)
                del raw, pred
            cs = min(len(v) for v in vp.values())
            vg = SA.velocity_magnitude(gt_full[:cs] * mask[:cs, None, None])
            cache[case] = (vg, vp)
        vg, vp = cache[case]
        k, sg = SA.spectrum(vg[:, a:b])
        ax, axr = axs[0, j], axs[1, j]
        ax.loglog(k[1:], sg[1:], color=INK, lw=1.6, label="Simulation")
        for lab, (_, col, ls) in mods.items():
            _, sp = SA.spectrum(vp[lab][:, a:b])
            n = min(len(sp), len(sg))
            ax.loglog(k[1:n], sp[1:n], color=col, ls=ls, lw=1.0, label=lab)
            axr.semilogx(k[1:n], sp[1:n] / sg[1:n], color=col, ls=ls, lw=1.0)
        axr.axhline(1, color=INK, lw=0.8)
        axr.set_yscale("log")
        axr.set_ylim(0.15, 3)
        axr.set_yticks([0.25, 0.5, 1, 2])
        axr.set_yticklabels(["0.25", "0.5", "1", "2"])
        axr.minorticks_off()
        for kb in (4.5, 12.5):
            ax.axvline(kb, color=MUTED, lw=0.5, ls=":")
            axr.axvline(kb, color=MUTED, lw=0.5, ls=":")
        ax.set_title(title, fontsize=7.5, color=INK)
        axr.set_xlabel("wavenumber $k$")
    axs[0, 0].set_ylabel("$E(k)$ of $|v|$")
    axs[1, 0].set_ylabel("prediction / truth")
    axs[0, 0].legend(fontsize=5.8, loc="lower left")
    fig.tight_layout()
    fig.savefig(HERE / "fig_spectra.pdf")


# ------------------------------------------------------------------------ DA analyses
def fig_da_fields():
    from data_assimilation.tra.bridge import PhysicalData
    from data_assimilation.tra.protocol import build_problem, canonical

    import os

    os.chdir(KAE)
    data = PhysicalData("data/acdm/128_tra/gt_interp.nc", "tra", torch.device("cpu"))
    prob = build_problem(data, name="A", n_problems=12, offsets=canonical(), seed=100)
    truth = data.frames(torch.as_tensor(prob.sim), torch.as_tensor(prob.t0)).numpy()
    mask = data.mask_for(torch.as_tensor(prob.sim)).numpy()
    src = {
        "U-Net 4D-Var": (BASE_DA / "A_headline.npz", "UNet"),
        "FNO 4D-Var": (BASE_DA / "A_headline.npz", "FNO"),
        "Reference": (DA / "da_v4/plain_ref_s2/df1/A_headline.npz", "KAE"),
        "+ mirror": (DA / "da_v4/mirror_ref_s2/df1/A_headline.npz", "KAE"),
        "+ mirror + LPM": (DA / "da_v5/lpm_ref_s2/df1/A_headline.npz", "KAE"),
        "back001": (DA / "da_v4/plain_back001_s2/df1/A_headline.npz", "KAE"),
        "+ mirror ": (DA / "da_v4/mirror_back001_s2/df1/A_headline.npz", "KAE"),
        "+ mirror + LPM ": (DA / "da_v5/lpm_back001_s2/df1/A_headline.npz", "KAE"),
    }
    an, rel = {}, {}
    for lab, (path, m) in src.items():
        d = np.load(path, allow_pickle=True)
        an[lab], rel[lab] = d[f"{m}__analysis"], d[f"{m}__rel"]
        # the stored error must be reproduced from the reconstructed truth
        d2 = (an[lab] - truth) * mask[:, None]
        r = np.sqrt(
            (d2**2).sum((1, 2, 3)) / ((truth * mask[:, None]) ** 2).sum((1, 2, 3))
        )
        assert np.allclose(r, rel[lab], rtol=1e-3), (lab, r[:3], rel[lab][:3])
    i = int(np.argsort(rel["Reference"])[len(rel["Reference"]) // 2])
    pc = 2  # pressure, DA layout [v_x, v_y, p, rho] x [Y, X]
    tr = truth[i] * mask[i]
    vmin, vmax = np.percentile(tr[pc][mask[i] > 0], [1, 99])
    el = 0.5 * (vmax - vmin) * 0.25
    wz_t = vorticity(tr[0], tr[1], 1, 0) * mask[i]
    wl = np.percentile(np.abs(wz_t), 99)
    rows = ["Truth"] + list(src)
    fig, axs = plt.subplots(
        len(rows), 3, figsize=(6.0, 8.6), gridspec_kw={"width_ratios": [2, 2, 1.2]}
    )
    for r, lab in enumerate(rows):
        x = tr if lab == "Truth" else an[lab][i] * mask[i]
        axs[r, 0].imshow(x[pc], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        if lab != "Truth":
            best = min(rel[q][i] for q in src)
            label(axs[r, 0], f"{rel[lab][i]:.3f}", rel[lab][i] == best)
            axs[r, 1].imshow(
                x[pc] - tr[pc], origin="lower", cmap="RdBu_r", vmin=-el, vmax=el
            )
        else:
            axs[r, 1].axis("off")
        wz = vorticity(x[0], x[1], 1, 0) * mask[i]
        axs[r, 2].imshow(wz[:, 20:84], origin="lower", cmap="PuOr_r", vmin=-wl, vmax=wl)
        for ax in axs[r]:
            clean(ax)
        axs[r, 0].set_ylabel(
            lab.strip(), fontsize=6.5, rotation=0, ha="right", va="center"
        )
    for ax, t in zip(
        axs[0], ["pressure at $t_0$", "pressure error", "vorticity (wake)"]
    ):
        ax.set_title(t, fontsize=7, color=INK)
    fig.subplots_adjust(
        left=0.2, right=0.99, top=0.96, bottom=0.01, wspace=0.04, hspace=0.06
    )
    fig.savefig(HERE / "fig_da_fields.pdf", dpi=250)
    print("DA problem", i, {k: round(float(v[i]), 4) for k, v in rel.items()})


if __name__ == "__main__":
    which = sys.argv[1:] or ["readouts", "spectra", "da"]
    if "readouts" in which:
        fig_readouts()
    if "spectra" in which:
        fig_spectra()
    if "da" in which:
        fig_da_fields()
