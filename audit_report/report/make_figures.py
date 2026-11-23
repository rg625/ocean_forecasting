"""Figures for the PDF report, drawn only from committed result files.

python make_figures.py   # writes fig_*.pdf next to this script
"""

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
AR = HERE.parent
RUNS = AR.parent / "koopman_autoencoder" / "model_outputs_ctkae"

# validated categorical slots (dataviz reference palette, light): fixed order, never cycled
C = {"ctkae": "#2a78d6", "acdm": "#eb6834", "unet": "#1baf7a", "legacy": "#eda100"}
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e3e3e0"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "figure.dpi": 150,
    }
)


def load(p):
    return json.loads(Path(p).read_text())


base = load(AR / "results/benchmark_suite_baselines.json")
e0 = load(AR / "results/E0_legacy_run20260821_ep199.json")
ct = [load(p) for p in sorted(glob.glob(str(RUNS / "F_block_s*/eval/best.json")))]


def ct_mean(case, key):
    return np.mean([np.array(d[case][key]) for d in ct], axis=0)


def series(case, key):
    return [
        ("CT-KAE (new, 3 seeds)", ct_mean(case, key), C["ctkae"], "-"),
        ("ACDM", np.array(base[case]["acdm-r20"][key]), C["acdm"], "--"),
        ("U-Net_ut", np.array(base[case]["unet-m8"][key]), C["unet"], "-."),
        ("Legacy KAE (E0)", np.array(e0[case][key]), C["legacy"], ":"),
    ]


def label_end(ax, x, y, text, color):
    ax.annotate(
        text,
        (x, y),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=7.5,
        color=INK,
    )
    ax.plot([x], [y], "o", ms=3.5, color=color)


# --- Figure 1: error vs lead (interp, extrap) --------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
for ax, case, title in zip(
    axes,
    ["interp", "extrap"],
    ["Interpolation (Ma 0.66-0.68)", "Extrapolation (Ma 0.50-0.52)"],
):
    clim = np.array(base[case]["train_climatology"]["mse_lead"])
    ax.axhline(clim.mean(), color=MUTED, lw=1, ls=(0, (2, 2)))
    ax.text(1, clim.mean() * 1.05, "constant mean field", fontsize=7, color=MUTED)
    for name, y, col, ls in series(case, "mse_lead"):
        lead = np.arange(1, len(y) + 1)
        ax.plot(lead, y, color=col, ls=ls, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("lead (steps of 0.1 time units)")
    ax.set_title(title, fontsize=9, color=INK, loc="left")
axes[0].set_ylabel("MSE (x1e-3, physical units)")
axes[1].legend(loc="lower right", fontsize=7)
fig.tight_layout()
fig.savefig(HERE / "fig_error_vs_lead.pdf")
plt.close(fig)

# --- Figure 2: fluctuation energy vs lead on the 240-step case ----------------------
fig, ax = plt.subplots(figsize=(7.2, 2.7))
ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (2, 2)))
ax.text(2, 1.03, "true energy (ratio = 1)", fontsize=7, color=MUTED)
for d in ct:  # individual seeds, thin, behind the mean
    y = np.array(d["longer"]["energy_lead"])
    ax.plot(np.arange(1, len(y) + 1), y, color=C["ctkae"], lw=0.7, alpha=0.45)
for name, y, col, ls in series("longer", "energy_lead"):
    lead = np.arange(1, len(y) + 1)
    ax.plot(
        lead,
        y,
        color=col,
        ls=ls,
        label=name if "CT-KAE" not in name else "CT-KAE (new): mean, thin = seeds",
    )
ax.legend(loc="lower left", fontsize=7, ncol=2)
ax.set_xlim(0, 242)
ax.set_xlabel("lead (steps of 0.1 time units)")
ax.set_ylabel("predicted / true\nfluctuation energy")
ax.set_title(
    "Long rollout (Ma 0.64-0.65, 240 steps): does the forecast keep its energy?",
    fontsize=9,
    color=INK,
    loc="left",
)
fig.tight_layout()
fig.savefig(HERE / "fig_energy_longer.pdf")
plt.close(fig)

# --- Figure 3: pilot ablations (best validation MSE60) ------------------------------
pil = {r["run"].split("_s0_")[0]: r for r in load(AR / "results/pilots_summary.json")}
names = {
    "P2_hurwitz": "P2 Hurwitz generator",
    "P4_block_amp": "P4 + amplitude loss",
    "P0_block": "P0 block generator (reference)",
    "P6_block_strides": "P6 + mixed time steps",
    "P3_block_nocond": "P3 no Mach conditioning",
    "P1_dense": "P1 dense (old-style) generator",
    "P5_block_k10": "P5 10-step training horizon",
}
rows = sorted(
    ((names[k], v["val_mse60"]) for k, v in pil.items() if k in names),
    key=lambda t: -t[1],
)
fig, ax = plt.subplots(figsize=(7.2, 2.6))
y = np.arange(len(rows))
ax.barh(y, [r[1] for r in rows], color=C["ctkae"], height=0.6)
for i, (_, v) in enumerate(rows):
    ax.text(v + 0.08, i, f"{v:.2f}", va="center", fontsize=7.5, color=INK)
pod = load(AR / "results/val_baselines.json")["pod_dmd2_r128"]["val_mse60"]
ax.axvline(pod, color=MUTED, lw=1, ls=(0, (2, 2)))
ax.text(
    pod - 0.1,
    len(rows) - 0.4,
    f"linear POD baseline {pod:.2f}",
    fontsize=7,
    color=MUTED,
    ha="right",
)
ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
ax.set_xlabel("best validation MSE over 60 steps (x1e-3; held-out Ma 0.60 / 0.75)")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(HERE / "fig_pilots.pdf")
plt.close(fig)

# --- Figure 4: Mach sensitivity -------------------------------------------------------
p0 = load(glob.glob(str(RUNS / "P0_block_s0_*/eval/best.json"))[0])["ma_sweep"][
    "output_rms_change_vs_true_ma"
]
fig, ax = plt.subplots(figsize=(3.6, 2.4))
ma, d = zip(*p0)
ax.plot(ma, d, color=C["ctkae"], marker="o", ms=4)
ax.axvline(0.66, color=MUTED, lw=1, ls=(0, (2, 2)))
ax.text(0.665, max(d) * 0.9, "true Ma", fontsize=7, color=MUTED)
ax.set_xlabel("Mach number given to the model")
ax.set_ylabel("rms change of forecast")
ax.set_title("Forecast responds smoothly to Ma", fontsize=9, color=INK, loc="left")
fig.tight_layout()
fig.savefig(HERE / "fig_mach.pdf")
plt.close(fig)

# --- Figure 5 (optional): representation pilots, if available ----------------------
rep = [
    r
    for r in (
        load(AR / "results/repr_summary.json")
        if (AR / "results/repr_summary.json").exists()
        else []
    )
]
if rep:
    lab = {
        "R0_d256_ref": "R0 latent 256 (reference)",
        "R1_d512": "R1 latent 512",
        "R2_d1024": "R2 latent 1024",
        "R3_wrec5": "R3 reconstruction x5",
    }
    rr = [
        (
            lab.get(r["run"].split("_s0_")[0], r["run"]),
            r["val_mse60"],
            r["val_mse60_first10"],
        )
        for r in rep
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2))
    for ax, j, t in zip(
        axes,
        [1, 2],
        ["best validation MSE over 60 steps", "first-10-step error at end of training"],
    ):
        ax.barh(range(len(rr)), [r[j] for r in rr], color=C["ctkae"], height=0.6)
        for i, r in enumerate(rr):
            ax.text(r[j], i, f" {r[j]:.2f}", va="center", fontsize=7.5)
        ax.set_yticks(
            range(len(rr)), [r[0] for r in rr] if j == 1 else [""] * len(rr), fontsize=8
        )
        ax.set_title(t, fontsize=8.5, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_repr.pdf")
    plt.close(fig)
print("figures written to", HERE)
