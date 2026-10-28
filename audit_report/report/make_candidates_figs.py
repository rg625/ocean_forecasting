"""Figures for ctkae_candidates.tex (reference CT-KAE vs back001, 2026-10-28).

python make_candidates_figs.py      -> fig_cand_*.pdf next to this file
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
KAE = ROOT / "koopman_autoencoder"
OUT = KAE / "model_outputs_ctkae"
ST = OUT / "ablation_state"
DA3 = OUT / "da_v3"
BASE_DA = Path(
    "/home/rg625/mnt/ocean_forecasting/koopman_autoencoder/da_results_tra_df9"
)
SUITE = json.loads(
    (ROOT / "audit_report/results/benchmark_suite_baselines.json").read_text()
)
# original KAE, final checkpoint, re-run through the verified pipeline (the stored prediction
# files in the benchmark suite are not reproducible; see report.tex)
E0 = json.loads(
    (ROOT / "audit_report/results/E0_legacy_run20260821_ep199.json").read_text()
)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"ref": BLUE, "back001": ORANGE}
LAB = {"ref": "Reference + mixture", "back001": "back001 + mixture"}
CASES = [("extrap", "Tra_ext"), ("interp", "Tra_int"), ("longer", "Tra_long")]

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "savefig.bbox": "tight",
    }
)


def run(key):
    r = Path(open(ST / f"{key}.done").read().strip())
    return r if r.is_absolute() else KAE / r


def ev(key, mix=True):
    r = run(key)
    if mix:
        sg = json.load(open(r / "mix_sigma.json"))["sigma"]
        return json.load(open(r / "eval" / f"last_mix{sg:g}.json"))
    return json.load(open(r / "eval" / "last.json"))


def lead_curves(tag, case, mix=True):
    return np.array([ev(f"{tag}_s{s}", mix)[case]["mse_lead"] for s in range(3)])


def fig_lead():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, (case, title) in zip(axs, CASES):
        for tag in ["ref", "back001"]:
            a = lead_curves(tag, case)
            k = np.arange(1, a.shape[1] + 1)
            mu, se = a.mean(0), a.std(0, ddof=1) / np.sqrt(3)
            ax.plot(k, mu, color=COL[tag], label=LAB[tag])
            ax.fill_between(k, mu - se, mu + se, color=COL[tag], alpha=0.18, lw=0)
        for key, lab, ls in [
            ("acdm-r20", "ACDM", "--"),
            ("unet-m8", "U-Net$_{ut}$", ":"),
            ("E0", "Original KAE (final ckpt)", "-."),
        ]:
            m = np.array(
                E0[case]["mse_lead"] if key == "E0" else SUITE[case][key]["mse_lead"]
            )
            ax.plot(np.arange(1, len(m) + 1), m, color=MUTED, ls=ls, lw=1.0, label=lab)
        ax.set_yscale("log")
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_xlabel("lead (steps of 0.1)")
    axs[0].set_ylabel("MSE (x1e-3)")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(
        h, lab, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.2)
    )
    fig.savefig(HERE / "fig_cand_lead.pdf")


def fig_delta():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=False)
    for ax, (case, title) in zip(axs, CASES):
        a, b = lead_curves("ref", case), lead_curves("back001", case)
        K = a.shape[1]
        bins = [(0, 1), (1, 5), (5, 15), (15, 30), (30, 58)] + (
            [(58, K)] if K > 58 else []
        )
        d = np.array(
            [(b[:, i:j].mean(1) - a[:, i:j].mean(1)) for i, j in bins]
        )  # [nb, 3]
        mu, se = d.mean(1), d.std(1, ddof=1) / np.sqrt(3)
        x = np.arange(len(bins))
        ax.bar(x, mu, color=INK2, width=0.6)
        ax.errorbar(x, mu, yerr=se, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xticks(x, [f"{i + 1}-{j}" for i, j in bins], rotation=30, fontsize=6.5)
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_xlabel("lead bin")
    axs[0].set_ylabel(
        "back001 - reference\nMSE (x1e-3), with mixture\n(< 0: back001 better)"
    )
    fig.savefig(HERE / "fig_cand_delta.pdf")


SWEEPS = [
    ("G2_noise", "noise std"),
    ("G3_sparsity", "observed fraction"),
    ("G4_delta_l", "window length (frames)"),
    ("G5_n_obs", "number of obs. times"),
    ("G19_misspec", "misspecified noise"),
    ("G20_joint_sparsity", "joint sparsity (N)"),
]


def da_rows(lab, sec):
    d = json.load(open(DA3 / lab / "df9" / f"{sec}.json"))
    return [(r.get("tag"), r["KAE"]["mean"]) for r in d["rows"] if "KAE" in r]


def short(t):
    import re

    t = str(t)
    for k, v in (("gaussian", "G "), ("laplace", "L ")):
        if t.startswith(k):
            return v + t[len(k) :]
    return re.sub(r"^[A-Za-z_]+", "", t) or t


def sort_order(tags):
    """Numeric order for purely numeric sweeps (the stored order is the run order)."""
    try:
        vals = [float(short(t)) for t in tags]
    except ValueError:
        return list(range(len(tags)))
    return (
        list(np.argsort(vals))
        if short(tags[0])[:1].isdigit() and "noise" in str(tags[0])
        else list(range(len(tags)))
    )


def fig_da():
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.0))
    for ax, (sec, xl) in zip(axs.flat, SWEEPS):
        for tag, lab in [("ref", "ctkae_H8_d256"), ("back001", "ctkae_back001")]:
            rows = [da_rows(f"{lab}_s{s}", sec) for s in range(3)]
            tags = [t for t, _ in rows[0]]
            v = np.array([[x for _, x in r] for r in rows])
            order = sort_order(tags)
            tags, v = [tags[i] for i in order], v[:, order]
            mu, se = v.mean(0), v.std(0, ddof=1) / np.sqrt(3)
            x = np.arange(len(tags))
            ax.errorbar(
                x,
                mu,
                yerr=se,
                color=COL[tag],
                marker="o",
                ms=3.5,
                capsize=2,
                lw=1.5,
                label=LAB[tag].replace(" + mixture", ""),
            )
        # original KAE, final checkpoint (epoch 199), same DA code (da_v3); the epoch-40
        # "paper" checkpoint was selected on the DA test trajectories
        ok = [
            r["KAE"]["mean"]
            for r in json.load(open(DA3 / "legacy_ep199/df9" / f"{sec}.json"))["rows"]
            if "KAE" in r
        ]
        ok = [ok[i] for i in order]
        ax.plot(
            np.arange(len(ok)),
            ok,
            color=MUTED,
            ls="-.",
            lw=1.0,
            marker="s",
            ms=2.5,
            label="Original KAE (final ckpt)",
        )
        ax.set_xticks(
            np.arange(len(tags)), [short(t) for t in tags], rotation=35, fontsize=6
        )
        ax.set_xlabel(xl)
    for ax in axs[:, 0]:
        ax.set_ylabel("analysis error (rel.)")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        h, lab, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.03)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "fig_cand_da.pdf")


def fig_sigma():
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.2))
    for ax, (fn, title) in zip(
        axs,
        [
            ("mix_sigma.json", "58-lead train rollouts"),
            ("mix_sigma_long.json", "238-lead train rollouts"),
        ],
    ):
        for tag in ["ref", "back001"]:
            js = [json.load(open(run(f"{tag}_s{s}") / fn)) for s in range(3)]
            g = js[0]["grid"]
            v = np.array([j["train_mse"] for j in js])
            ax.plot(
                g,
                v.mean(0),
                color=COL[tag],
                marker="o",
                ms=3,
                label=LAB[tag].replace(" + mixture", ""),
            )
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_xlabel("mixture width $\\sigma$")
    axs[0].set_ylabel("train MSE (x1e-3)")
    axs[0].legend(fontsize=6.5)
    fig.savefig(HERE / "fig_cand_sigma.pdf")


if __name__ == "__main__":
    fig_lead()
    fig_delta()
    fig_da()
    fig_sigma()
    print("figures written")
