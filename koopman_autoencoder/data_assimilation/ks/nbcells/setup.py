# ruff: noqa: E741
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS = Path("da_results_3way_sda_paper")

SECTIONS = {
    "A": "A_headline",
    "B": "B_cost_vs_horizon",
    "C": "C_nobs",
    "D": "D_noise",
    "E": "E_sparsity",
    "F": "F_statistics",
    "G": "G_continuous",
    "H": "H_gallery",
    "I": "I_spacetime",
    "J": "J_sparse_recovery",
}

# One identity per method, used in every panel so the eye can track them across sections.
STYLE = {
    "KAE-expm": dict(
        color="#1b7837",
        marker="o",
        ls="-",
        label="Continuous KAE — exact $e^{K\\tau}$",
        short="KAE",
    ),
    "KAE-rk4": dict(
        color="#7fbc41",
        marker="s",
        ls="--",
        label="Continuous KAE — RK4 rollout",
        short="KAE/RK4",
    ),
    "UNet": dict(
        color="#d6604d", marker="^", ls="-", label="U-Net 4D-Var", short="U-Net"
    ),
    "SDA": dict(
        color="#762a83",
        marker="D",
        ls="-",
        label="Score-based DA (Rozet \u0026 Louppe)",
        short="SDA",
    ),
}
METHODS = ["KAE-expm", "KAE-rk4", "UNet", "SDA"]

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "legend.framealpha": 0.9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    }
)


def status():
    """Which sections are on disk yet."""
    if not RESULTS.is_dir():
        print(RESULTS, "does not exist yet. Generate it with:")
        print(
            "    python -m data_assimilation.ks.da_ks_experiments_3way --out-dir",
            RESULTS,
            "\\\n        --sda-config da_results_sda_paper/frozen_config.json",
        )
        return set()
    have = {k for k, v in SECTIONS.items() if (RESULTS / (v + ".npz")).is_file()}
    missing = sorted(set(SECTIONS) - have)
    print("sections available:", "".join(sorted(have)) or "(none)")
    if missing:
        print(
            "still missing:     ",
            "".join(missing),
            "  (the driver writes each section as it finishes)",
        )
    return have


def L(name):
    f = RESULTS / (name + ".npz")
    if not f.is_file():
        raise FileNotFoundError(
            str(f) + " not written yet.\n"
            "Run:  python -m data_assimilation.ks.da_ks_experiments_3way --out-dir "
            + str(RESULTS)
            + " \\\n          --sda-config da_results_sda_paper/frozen_config.json\n"
            "or just this section by adding:  --sections " + name[0]
        )
    return np.load(f, allow_pickle=True)


def present(d, method, suffix):
    return (method + "__" + suffix) in d.files


def avail(d, suffix):
    """Methods present in this file, in canonical order."""
    return [m for m in METHODS if present(d, m, suffix)]


def legend_once(fig, axes, ncol=4):
    """One shared legend for a row of panels, so each panel keeps its data area."""
    h, l = [], []
    for ax in np.atleast_1d(axes).ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(
        h, l, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.06), frameon=False
    )


have = status()

summary = {}
sp = RESULTS / "summary.json"
if sp.is_file():
    summary = json.loads(sp.read_text())
    print("\n=== RUN CONFIGURATION ===")
    for k in (
        "sda_implementation",
        "sda_k",
        "sda_blanket",
        "n_sweep",
        "n_stats",
        "sda_n_samples",
        "unet_ckpt",
        "test_file",
    ):
        if k in summary:
            print(f"  {k:22s} = {summary[k]}")
    if "sda_settings" in summary:
        s_ = summary["sda_settings"]
        print(
            f"  {'SDA sampler':22s} = N={s_['n_steps']} steps, C={s_['corrections']} "
            f"Langevin corrections, tau={s_['tau']}, Gamma={s_['gamma_mode']}"
        )
    print("\n=== HEADLINE NUMBERS ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:34s} = {v:.4f}")
else:
    print("\nsummary.json appears when the whole campaign finishes.")


def truth_ylim(truth, pad=1.9):
    """Y-limits set by the GROUND TRUTH, not by whichever method happens to be worst.

    A single badly-performing method can otherwise compress every other curve onto a flat
    line, which hides exactly the comparison the panel exists to make. Curves leaving the
    frame are annotated rather than silently cropped.
    """
    t = np.asarray(truth)
    lo, hi = float(np.nanmin(t)), float(np.nanmax(t))
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * pad
    return mid - half, mid + half


def note_clipped(ax, curves, lim):
    """Mark any curve that runs outside the truth-set frame."""
    out = [n for n, c in curves if np.nanmin(c) < lim[0] or np.nanmax(c) > lim[1]]
    if out:
        ax.text(
            0.99,
            0.02,
            "off-scale: " + ", ".join(out),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="0.35",
            bbox=dict(fc="w", ec="0.8", alpha=0.85, pad=1.5),
        )
