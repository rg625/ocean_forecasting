"""Tra_ext and Tra_int at t = 15, 30, 45, 60, as in the paper: per-lead MSE table (tab_lead.tex)
and a pressure gallery (fig_short.pdf) in the layout of autoreg_pde_diffusion/src/plot_data.py.

t is the paper's label (GT frame index t - 1; frames 0 and 1 are the inputs). The original
KAE's stored predictions are one frame early (benchmark_suite.EARLY) and are realigned here.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
IMPL = HERE.parents[2]
AUTOREG = Path("/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion")
PP1 = Path(
    "/tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/093163e5-e75d-47aa-b04b-9bb0dda37f82/scratchpad/res/pp"
)
for p in (
    IMPL / "koopman_autoencoder",
    IMPL / "audit_report/scripts",
    PP1,
    AUTOREG / "src",
):
    sys.path.insert(0, str(p))

import pp_analyze as P1  # noqa: E402
from benchmark_suite import (  # noqa: E402
    BUILTIN,
    UNTIMED,
    load_gt,
    load_pred,
    score,
    train_climatology,
)
from plot_color_and_name_mapping import getColormapAndNorm  # noqa: E402

mpl.rcParams["pdf.fonttype"] = 42
CASES = {"extrap": "ext", "interp": "int"}
TS = [15, 30, 45, 60]
FIELDS = IMPL / "audit_report/results/paper_ctkae/fields"
P = 3  # pressure


def sources(case):
    """label -> (path, early)"""
    return {
        "ACDM": (UNTIMED / case / "acdm-r20.npz", False),
        "U-Net$_{ut}$": (UNTIMED / case / "unet-m8.npz", False),
        "U-Net$_{tn}$": (UNTIMED / case / "unet-m2-noise0.01.npz", False),
        "Original KAE": (
            Path(BUILTIN["kae_run20260821_ep199"].format(case=case)),
            True,
        ),
        "Reference + post-proc.": (FIELDS / case / "ref+pp.npz", False),
        "back001 + post-proc.": (FIELDS / case / "back001+pp.npz", False),
        "Reference + sharp. mirror": (FIELDS / case / "ref+sharp.npz", False),
        "back001 + sharp. mirror": (FIELDS / case / "back001+sharp.npz", False),
        "Reference + LPM mirror": (FIELDS / case / "ref+lpm.npz", False),
        "back001 + LPM mirror": (FIELDS / case / "back001+lpm.npz", False),
    }


def load(path, S, T, early):
    a = load_pred(path, S, T, early)
    if (
        early
    ):  # GT frame T-1 is stored frame T-2; load_pred leaves stored frame T-1 there
        raw = np.load(path)["arr_0"]
        a[:, :, -1] = raw.reshape(-1, S, *raw.shape[-4:])[:, :, -2, :4]
    return a


def per_lead(clim):
    out = {}
    for case in CASES:
        gt, fluid = load_gt(case)
        S, T = gt.shape[:2]
        res = {}
        for lab, (path, early) in sources(case).items():
            res[lab] = score(load(path, S, T, early), gt, fluid, clim)["mse_lead"]
        for t, name in [("ref", "Reference"), ("back001", "back001")]:
            res[name] = np.mean(
                [
                    score(
                        P1.to_acdm(case, P1.Ckpt(f"{t}_s{s}").d[f"{case}_pred"][0])[
                            None
                        ],
                        gt,
                        fluid,
                        clim,
                    )["mse_lead"]
                    for s in range(3)
                ],
                0,
            )
        out[case] = {
            k: [float(v[t - 3]) for t in TS] for k, v in res.items()
        }  # lead n = frame n + 2
    return out


def tab_lead(m):
    order = [
        "ACDM",
        "U-Net$_{ut}$",
        "U-Net$_{tn}$",
        "Original KAE",
        "Reference",
        "Reference + post-proc.",
        "Reference + sharp. mirror",
        "Reference + LPM mirror",
        "back001",
        "back001 + post-proc.",
        "back001 + sharp. mirror",
        "back001 + LPM mirror",
    ]
    rows = []
    for k in order:
        cells = []
        for case in CASES:
            col = [m[case][o] for o in order]
            for j, v in enumerate(m[case][k]):
                best = min(c[j] for c in col)
                cells.append(rf"\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}")
        rows.append(
            f"{k} & "
            + " & ".join(cells)
            + r"\\"
            + (r" \midrule" if k == "Original KAE" else "")
        )
    (HERE / "tab_lead.tex").write_text("\n".join(rows) + "\n")


def mse_label(ax, v, best):
    """pressure MSE x1e-3 of this panel, bold for the best model in the column"""
    ax.text(
        0.97,
        0.04,
        f"{v:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        fontweight="bold" if round(v, 2) == round(best, 2) else "normal",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.8),
    )


def gallery():
    rows = [
        "Simulation",
        "ACDM",
        "Original KAE",
        "Reference",
        "Reference + post-proc.",
        "Reference + sharp. mirror",
        "Reference + LPM mirror",
        "back001",
        "back001 + post-proc.",
        "back001 + sharp. mirror",
        "back001 + LPM mirror",
    ]
    fig, axs = plt.subplots(
        len(rows),
        9,
        figsize=(7.0, 8.6),
        gridspec_kw={"width_ratios": [1] * 4 + [0.12] + [1] * 4},
    )
    for k, case in enumerate(CASES):
        gt, fluid = load_gt(case)
        S, T = gt.shape[:2]
        src = sources(case)
        src["Reference"] = (FIELDS / case / "ref.npz", False)
        src["back001"] = (FIELDS / case / "back001.npz", False)
        cmap, norm = getColormapAndNorm(case, "pres")
        frames = {
            lab: (
                gt[0, [t - 1 for t in TS], P]
                if lab == "Simulation"
                else (load(src[lab][0], S, T, src[lab][1]) * fluid)[
                    0, 0, [t - 1 for t in TS], P
                ]
            )
            for lab in rows
        }
        mse = {
            lab: ((v - frames["Simulation"]) ** 2).mean((-1, -2)) * 1e3
            for lab, v in frames.items()
        }
        for i, lab in enumerate(rows):
            for j, t in enumerate(TS):
                ax = axs[i, 5 * k + j]
                ax.imshow(
                    frames[lab][j, 6:70, 0:64].T,
                    cmap=cmap,
                    norm=norm,
                    interpolation="nearest",
                )
                if lab != "Simulation":
                    mse_label(ax, mse[lab][j], min(mse[r][j] for r in rows[1:]))
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"$t={t}$", fontsize=7)
                if j == 0 and k == 0:
                    ax.set_ylabel(lab.replace(" + ", "\n+ "), fontsize=6.5)
        axs[0, 5 * k + 1].text(
            1.0,
            1.35,
            f"Tra$_{{{CASES[case]}}}$",
            transform=axs[0, 5 * k + 1].transAxes,
            ha="center",
            fontsize=8,
        )
    for i in range(len(rows)):
        axs[i, 4].axis("off")
    fig.subplots_adjust(
        left=0.1, right=0.99, top=0.93, bottom=0.01, wspace=0.03, hspace=0.04
    )
    fig.savefig(HERE / "fig_short.pdf", dpi=250)


if __name__ == "__main__":
    m = per_lead(train_climatology())
    for case in CASES:
        for k, v in m[case].items():
            print(case, f"{k:28s}", " ".join(f"{x:6.2f}" for x in v))
    tab_lead(m)
    gallery()
