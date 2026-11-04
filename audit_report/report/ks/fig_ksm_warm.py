"""Figures of make_ksm_warm.py's two comparisons (warm start for every gradient-based method):
fig_ksm_warm9.{pdf,png}  delta_f = 9 DA, four sweeps, mean +- SE (band) over problems;
fig_ksm_fc2.{pdf,png}    free-running forecast from the true state, mean +- SE.
Hue = method family (SDA, KAE, U-Net, CT-KAE ref, CT-KAE back001; validated palette);
line style = variant (CT-KAE: solid 128, dashed 256; KAE: solid control, dotted stored d_z 512).

    python fig_ksm_warm.py
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from make_ksm_warm import COLS, HERE, MAIN, RES, RUNS, table9

STY = {  # colour, linestyle, marker, label
    "SDA": ("#2a78d6", "-", "s", "SDA (stored)"),
    "KAE ctrl": ("#eb6834", "-", "o", "KAE control (matched retrain)"),
    "KAE-512 (stored)": ("#eb6834", ":", "o", "KAE d_z 512 (stored, Sept.)"),
    "U-Net": ("#1baf7a", "-", "D", "U-Net"),
    "ref 128": ("#eda100", "-", "^", "CT-KAE ref 128"),
    "ref 256": ("#eda100", "--", "^", "CT-KAE ref 256"),
    "back001 128": ("#e87ba4", "-", "v", "CT-KAE back001 128"),
    "back001 256": ("#e87ba4", "--", "v", "CT-KAE back001 256"),
    "ref signed 128": ("#008300", "-", "P", "CT-KAE ref 128, signed decay"),
    "ref w2 128": ("#6250d6", "-", "X", "CT-KAE ref 128, widths x2"),
}
mpl.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.5,
        "axes.edgecolor": "#898781",
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
    }
)


def line(ax, x, v, col):
    c, ls, mk, lab = STY[col]
    x = np.asarray(x, float)
    m = np.array([np.nan if e is None else e["mean"] for e in v])
    s = np.array([np.nan if e is None else e["sem"] for e in v])
    ok = np.isfinite(m)
    if not ok.any():
        return
    ax.plot(
        x[ok],
        m[ok],
        color=c,
        ls=ls,
        lw=1.6,
        marker=mk,
        ms=4,
        mfc=c if ls == "-" else "#fcfcfb",
        mec=c,
        label=lab,
    )
    ax.fill_between(x[ok], (m - s)[ok], (m + s)[ok], color=c, alpha=0.15, lw=0)


def da9():
    tab = dict(table9())
    fig, axs = plt.subplots(1, 4, figsize=(10.5, 2.9), sharey=True)
    # (a) last observation
    rr = [(r, v) for r, v in tab["C2_delta_l"] if not r["tag"].endswith("anchor")]
    x = [r["taus"][-1] for r, _ in rr]
    for c in COLS:
        line(axs[0], x, [v[c] for _, v in rr], c)
    axs[0].set_xscale("log")
    axs[0].set_xticks([1.3, 2.5, 5, 10, 20, 40, 70])
    axs[0].set_xticklabels(["1.3", "2.5", "5", "10", "20", "40", "70"])
    axs[0].minorticks_off()
    axs[0].set_xlabel(r"last observation $\delta_l$ (t.u.)")
    axs[0].set_title("(a) window length, N = 5\n(1000 it., 24 problems)", fontsize=8)
    # (b) number of observation times
    rr = tab["C3_n_obs"]
    x = [len(r["taus"]) for r, _ in rr]
    for c in COLS:
        line(axs[1], x, [v[c] for _, v in rr], c)
    axs[1].set_xscale("log")
    axs[1].minorticks_off()
    axs[1].set_xticks([2, 3, 5, 9, 17])
    axs[1].set_xticklabels(["2", "3", "5", "9", "17"])
    axs[1].set_xlabel("observation times N")
    axs[1].set_title(r"(b) N, $\delta_l$ = 2.5", fontsize=8)
    # (c) noise (0 drawn at 0.003)
    rr = tab["noise_law"]
    x = [max(float(r["value"]), 0.003) for r, _ in rr]
    for c in COLS:
        line(axs[2], x, [v[c] for _, v in rr], c)
    axs[2].set_xscale("log")
    axs[2].minorticks_off()
    axs[2].set_xticks([0.003, 0.01, 0.05, 0.1, 0.2, 0.4])
    axs[2].set_xticklabels(["0", "0.01", "0.05", "0.1", "0.2", "0.4"])
    axs[2].set_xlabel("Gaussian noise std")
    axs[2].set_title("(c) noise, N = 5", fontsize=8)
    # (d) coverage at N = 5
    rr = [(r, v) for r, v in tab["joint_sparsity"] if len(r["taus"]) == 5]
    x = [100 * r["value"][0] for r, _ in rr]
    for c in COLS:
        line(axs[3], x, [v[c] for _, v in rr], c)
    axs[3].set_xscale("log")
    axs[3].minorticks_off()
    axs[3].set_xticks([10, 25, 50, 100])
    axs[3].set_xticklabels(["10", "25", "50", "100"])
    axs[3].invert_xaxis()
    axs[3].set_xlabel("spatial coverage (%)")
    axs[3].set_title("(d) coverage, N = 5", fontsize=8)
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"analysis error at $t_0$ (rel. L2)")
    for ax in axs:
        ax.axhline(1, color="#898781", lw=0.6)
    h, lab = [], []
    for ax in axs:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    order = [STY[c][3] for c in STY if STY[c][3] in lab]
    h = [h[lab.index(o)] for o in order]
    fig.suptitle(
        r"KS data assimilation at $\delta_f$ = 0.9 t.u., warm start for every gradient method "
        "(mean $\\pm$ SE over problems; lower is better)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.legend(h, order, loc="lower center", ncol=4, fontsize=7.5)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_ksm_warm9.{ext}", dpi=170)


def fc():
    f = np.load(MAIN / "da_results_long/FC_forecast.npz")
    TL = float(f["T_L"])
    fig, ax = plt.subplots(figsize=(6.8, 3.3))

    def band(t, m, s, c, ls, lab, mk=None):
        x = np.asarray(t) / TL
        k = x <= 3
        ax.plot(x[k], m[k], color=c, ls=ls, lw=1.6, label=lab)
        ax.fill_between(x[k], (m - s)[k], (m + s)[k], color=c, alpha=0.15, lw=0)

    band(
        f["taus"],
        f["UNet__rel_mean"],
        f["UNet__rel_sem"],
        "#1baf7a",
        "-",
        "U-Net (paper ckpt)",
    )
    band(
        f["taus"],
        f["KAE-expm__rel_mean"],
        f["KAE-expm__rel_sem"],
        "#eb6834",
        ":",
        "paper KAE (paper ckpt)",
    )
    lab_of = {
        "KAE ctrl": "KAE control (matched retrain)",
        "ref 128": "CT-KAE ref 128",
        "ref 256": "CT-KAE ref 256",
        "back001 128": "CT-KAE back001 128",
        "back001 256": "CT-KAE back001 256",
        "ref signed 128": "CT-KAE ref 128, signed decay",
        "ref w2 128": "CT-KAE ref 128, widths x2",
    }
    for n, lab in RUNS:
        p = RES / n / "fc_matched.npz"
        if p.exists():
            d = np.load(p)
            c, ls, _, _ = STY[lab]
            band(
                d["taus"],
                d[f"{n}__single__rel_mean"],
                d[f"{n}__single__rel_sem"],
                c,
                ls,
                lab_of[lab],
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1, color="#898781", lw=0.6)
    ax.set_xlabel("lead time (Lyapunov times)")
    ax.set_ylabel("relative L2 error")
    ax.set_title(
        "KS forecast from the true state (32 trajectories, single-frame encoding, mean $\\pm$ SE)",
        fontsize=8.5,
    )
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_ksm_fc2.{ext}", dpi=170)


if __name__ == "__main__":
    da9()
    fc()
    print("fig_ksm_warm9, fig_ksm_fc2 written")
