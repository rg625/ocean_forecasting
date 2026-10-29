"""Tables and figures for ctkae_collab.tex (short collaborator report, 2026-10-28).

python make_collab.py      -> tab_*.tex, fig_*.pdf next to this file
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

AUTOREG_RES = Path("/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion/results")

HERE = Path(__file__).parent
IMPL = HERE.parents[2]
RES = IMPL / "audit_report/results"
PC = RES / "paper_ctkae"
DA = IMPL / "koopman_autoencoder/model_outputs_ctkae"
BASE_DA1 = Path("/home/rg625/mnt/ocean_forecasting/koopman_autoencoder/da_results_tra")
BASE_DA9 = Path(
    "/home/rg625/mnt/ocean_forecasting/koopman_autoencoder/da_results_tra_df9"
)
E0 = json.loads((RES / "E0_legacy_run20260821_ep199.json").read_text())
ABL = {
    k: json.loads((IMPL / f"audit_report/report/abl_{k}.json").read_text())
    for k in ("single", "mix")
}
PP2 = json.loads((RES / "postproc2/pp2_test_results.json").read_text())
LSIM = json.loads((PC / "lsim_fields.json").read_text())
CASES = ["extrap", "interp", "longer"]
SWEEPS = [
    "G4_delta_l",
    "G5_n_obs",
    "G3_sparsity",
    "G2_noise",
    "G19_misspec",
    "G20_joint_sparsity",
]

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
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


def f(v, d=2):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


# ------------------------------------------------------------------------- DA helpers
def da_dir(model, variant, seed=None):
    """model: ref | back001; variant: plain | mirror | ens"""
    if variant == "ens":
        return DA / f"da_v4/ens_mirror_{model}"
    if variant == "plain" and seed == 0:
        return (
            DA
            / "da_v3"
            / ("ctkae_H8_d256_s0" if model == "ref" else "ctkae_back001_s0")
        )
    return DA / f"da_v4/{variant}_{model}_s{seed}"


def _tags(d, s):
    return {r["tag"] for r in json.loads((d / f"{s}.json").read_text())["rows"]}


# df1: the baselines' delta_l and n_obs grids differ from the CT-KAE runs'; average over the
# points both have (31 of 37). df9: identical grids (37 points).
COMMON = {
    "df1": {
        s: _tags(BASE_DA1, s) & _tags(DA / "da_v3/ctkae_H8_d256_s0/df1", s)
        for s in SWEEPS
    },
    "df9": {s: _tags(BASE_DA9, s) for s in SWEEPS},
}
NPTS = {k: sum(len(v) for v in c.values()) for k, c in COMMON.items()}


def sweep_mean(d, method="KAE", pin=None):
    pin = pin or ("df1" if d.name == "df1" or d == BASE_DA1 else "df9")
    try:
        v = [
            r[method]["mean"]
            for s in SWEEPS
            for r in json.loads((d / f"{s}.json").read_text())["rows"]
            if r["tag"] in COMMON[pin][s]
            and r.get(method)
            and r[method].get("mean") is not None
        ]
    except FileNotFoundError:
        return None
    return float(np.mean(v)) if len(v) == NPTS[pin] else None


def headline(d, method="KAE"):
    for fn in ("A_headline.json", "A_headline_diffusion.json"):
        p = d / fn
        if p.exists():
            h = json.loads(p.read_text())
            if method in h:
                return h[method]["mean"]
    return None


def delta_f(d, method="KAE"):
    p = d / "G1_delta_f.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text())["rows"]
    return {
        r["value"]: r[method]["mean"]
        for r in rows
        if r.get(method) and r[method].get("mean") is not None
    }


def da_stats(model, variant, seeds=range(3)):
    """(headline df1, df1 sweeps, df9 sweeps): mean over seeds (or the ensemble)."""
    dirs = (
        [da_dir(model, "ens")]
        if variant == "ens"
        else [da_dir(model, variant, s) for s in seeds]
    )
    out = []
    for get in (
        lambda d: headline(d / "df1"),
        lambda d: sweep_mean(d / "df1"),
        lambda d: sweep_mean(d / "df9"),
    ):
        v = [get(d) for d in dirs]
        out.append(float(np.mean(v)) if all(x is not None for x in v) else None)
    return out


# ------------------------------------------------------------------------- tables
def tab_main():
    timing = {}
    tp = PC / "timing.json"
    if tp.exists():
        timing = {r["readout"]: r for r in json.loads(tp.read_text())["rows"]}
    base_t = {
        r["model"]: r
        for r in json.loads(
            Path(
                "/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion/src/results/metrics_dist/inference_cost_baselines.json"
            ).read_text()
        )["rows"]
    }
    kae_t = {
        r["model"]: r
        for r in json.loads(
            Path(
                "/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion/src/results/metrics_dist/inference_cost_kae.json"
            ).read_text()
        )["rows"]
    }
    rows = []

    def row(lab, mse, lsim, da, t):
        cells = [f(v) for v in mse] + [f(v) for v in lsim] + [f(v, 3) for v in da] + [t]
        rows.append(f"{lab} & " + " & ".join(cells) + r"\\")

    import sys

    sys.path.insert(0, str(IMPL / "audit_report/scripts"))
    from ablation_table import baselines

    bl = dict(baselines())
    for lab, key, dk, tk in [
        ("ACDM", "ACDM (official, r20)", "ACDM", "ACDM"),
        ("U-Net$_{ut}$", "U-Net_ut (official, unrolled 8)", "UNet", "U-Net-ut"),
        ("U-Net$_{tn}$", "U-Net_tn (official, training noise 0.01)", None, "U-Net-tn"),
    ]:
        mse = [bl[key][f"{c}_mse_all"] for c in CASES]
        lsim = [bl[key][f"{c}_lsim"] for c in CASES]
        da = [
            headline(BASE_DA1, dk) if dk else None,
            sweep_mean(BASE_DA1, dk, "df1") if dk else None,
            sweep_mean(BASE_DA9, dk, "df9") if dk else None,
        ]
        row(lab, mse, lsim, da, f"{base_t[tk]['ms_per_step']:.2f}")
    leg = DA / "da_v3/legacy_ep199"
    row(
        "Original KAE (final ckpt)",
        [E0[c]["mse_all"] for c in CASES],
        [None] * 3,
        [headline(leg / "df1"), sweep_mean(leg / "df1"), sweep_mean(leg / "df9")],
        f"{kae_t['KAE (RK4 rollout)']['ms_per_step']:.2f}",
    )
    rows[-1] += r" \midrule"
    names = {"ref": "Reference", "back001": "back001"}
    for t in ("ref", "back001"):
        s = ABL["single"]["ref" if t == "ref" else "back001"]
        tm = timing.get("single decode", {}).get("ms_per_step")
        row(
            f"{names[t]}",
            [s[f"{c}_mse_all"][0] for c in CASES],
            [s[f"{c}_lsim"][0] for c in CASES],
            da_stats(t, "plain"),
            f(tm),
        )
        tm = timing.get("3 seeds, mixture + mirror + shrink", {}).get("ms_per_step")
        row(
            f"{names[t]} + post-proc.",
            [PP2[f"mirror+tr|{c}|{t}_ens"] for c in CASES],
            [LSIM[f"{t}+pp"][c] for c in CASES],
            [None] * 3,
            f(tm),
        )
    (HERE / "tab_main.tex").write_text("\n".join(rows) + "\n")


def tab_pp():
    """paired gains of each step (reference, 3 seeds) and inference cost"""
    timing = {}
    tp = PC / "timing.json"
    if tp.exists():
        timing = {r["readout"]: r for r in json.loads(tp.read_text())["rows"]}
    m = ABL["mix"]
    rows = []
    for t in ("ref", "back001"):
        s = ABL["single"][t]
        steps = [
            ("single decode", [s[f"{c}_mse_all"][0] for c in CASES], "single decode"),
            ("+ mixture", [m[t][f"{c}_mse_all"][0] for c in CASES], "mixture"),
            (
                "+ mirror",
                [
                    np.mean([PP2[f"mirror|{c}|{t}_s{i}"] for i in range(3)])
                    for c in CASES
                ],
                "mixture + mirror",
            ),
            (
                "+ shrinkage",
                [
                    np.mean([PP2[f"mirror+tr|{c}|{t}_s{i}"] for i in range(3)])
                    for c in CASES
                ],
                "mixture + mirror + shrink",
            ),
            (
                "+ 3-seed average",
                [PP2[f"mirror+tr|{c}|{t}_ens"] for c in CASES],
                "3 seeds, mixture + mirror + shrink",
            ),
        ]
        for i, (lab, v, tk) in enumerate(steps):
            tr = timing.get(tk, {})
            lead = (
                (r"\multirow{5}{*}{" + ("Reference" if t == "ref" else "back001") + "}")
                if i == 0
                else ""
            )
            rows.append(
                f"{lead} & {lab} & "
                + " & ".join(f(x) for x in v)
                + f" & {tr.get('decodes_per_step', '--')} & {f(tr.get('ms_per_step'))}\\\\"
            )
        rows[-1] += r" \midrule" if t == "ref" else ""
    (HERE / "tab_pp.tex").write_text("\n".join(rows) + "\n")


def tab_da_pp():
    """mirror averaging in 4D-Var, paired on seed 2 of each model"""
    rows = []
    for t in ("ref", "back001"):
        a, b = da_stats(t, "plain", [2]), da_stats(t, "mirror", [2])
        name = "Reference" if t == "ref" else "back001"
        rows.append(f"{name} & plain & " + " & ".join(f(v, 3) for v in a) + r"\\")
        rows.append(" & mirror & " + " & ".join(f(v, 3) for v in b) + r"\\")
        rows.append(
            " & change & "
            + " & ".join(
                "--" if x is None or y is None else f"{100 * (y / x - 1):+.0f}\\%"
                for x, y in zip(a, b)
            )
            + r"\\"
        )
        rows[-1] += r" \midrule" if t == "ref" else ""
    (HERE / "tab_da_pp.tex").write_text("\n".join(rows) + "\n")


def tab_paper():
    sol = json.loads((PC / "solvers.json").read_text())
    rows = []
    for t in ("ref", "back001"):
        for key, lab in [
            ("exact", r"$e^{\tau A}$ (exact)"),
            ("rk4@0.1", "RK4, $\\Delta t=0.1$"),
            ("rk4@0.3", "RK4, $\\Delta t=0.3$"),
            ("rk4@0.5", "RK4, $\\Delta t=0.5$"),
            ("midpoint@0.1", "midpoint, $\\Delta t=0.1$"),
            ("euler@0.05", "Euler, $\\Delta t=0.05$"),
            ("euler@0.1", "Euler, $\\Delta t=0.1$"),
            ("dopri5", "Dopri5 (adaptive)"),
        ]:
            v = sol[t][key]
            cells = ["diverged" if not np.isfinite(v[c]) else f(v[c]) for c in CASES]
            rows.append(
                f"{'Reference' if t == 'ref' else 'back001'} & {lab} & "
                + " & ".join(cells)
                + r"\\"
            )
        rows[-1] += r" \midrule" if t == "ref" else ""
    (HERE / "tab_solvers.tex").write_text("\n".join(rows) + "\n")
    # unseen dt (eval_ctkae --dt_test): MSE at the half-step leads vs the whole-step leads
    rows = []
    for t in ("ref", "back001"):
        vals = []
        for s in range(3):
            r = Path(
                open(
                    IMPL
                    / f"koopman_autoencoder/model_outputs_ctkae/ablation_state/{t}_s{s}.done"
                )
                .read()
                .strip()
            )
            r = r if r.is_absolute() else IMPL / "koopman_autoencoder" / r
            vals.append(json.loads((r / "eval/last.json").read_text())["dt_test"])
        cells = []
        for c in ("extrap", "interp"):
            cells += [
                f(np.mean([v[c]["mse_whole_step_leads"] for v in vals])),
                f(np.mean([v[c]["mse_half_step_leads"] for v in vals])),
            ]
        rows.append(
            f"{'Reference' if t == 'ref' else 'back001'} & " + " & ".join(cells) + r"\\"
        )
    (HERE / "tab_dt.tex").write_text("\n".join(rows) + "\n")
    # spectra
    sp = json.loads((PC / "spectral.json").read_text())
    rows = []
    for name, lab in [
        ("ACDM", "ACDM"),
        ("U-Net-ut", "U-Net$_{ut}$"),
        ("FNO-16", "FNO$_{16}$"),
        ("ref", "Reference"),
        ("ref+pp", "Reference + post-proc."),
        ("back001", "back001"),
        ("back001+pp", "back001 + post-proc."),
    ]:
        cells = []
        for case, win in [("interp", "0-60"), ("longer", "0-240")]:
            sr = next(
                x for x in sp[case]["rows"] if x["model"] == name and x["window"] == win
            )
            cells += [
                f(sr["err_low"]),
                f(sr["err_mid"]),
                f(sr["err_high"]),
                f(sr["slope_pred"]),
            ]
        rows.append(f"{lab} & " + " & ".join(cells) + r"\\")
    gt = [
        next(x for x in sp[c]["rows"] if x["window"] == w)["slope_gt"]
        for c, w in [("interp", "0-60"), ("longer", "0-240")]
    ]
    rows.append(f"\\emph{{Ground truth}} & & & & {f(gt[0])} & & & & {f(gt[1])}\\\\")
    (HERE / "tab_spectral.tex").write_text("\n".join(rows) + "\n")


def tab_sharp():
    """what the post-processing costs in sharpness: long MSE, LSiM, long-rollout spectrum"""
    from ablation_table import baselines

    bl = dict(baselines())
    sp = {
        x["model"]: x
        for x in json.loads((PC / "spectral.json").read_text())["longer"]["rows"]
        if x["window"] == "0-240"
    }
    rows = []
    for lab, mse, lsim, sk in [
        (
            "ACDM",
            bl["ACDM (official, r20)"]["longer_mse_all"],
            [bl["ACDM (official, r20)"][f"{c}_lsim"] for c in CASES],
            "ACDM",
        ),
        (
            "U-Net$_{ut}$",
            bl["U-Net_ut (official, unrolled 8)"]["longer_mse_all"],
            [bl["U-Net_ut (official, unrolled 8)"][f"{c}_lsim"] for c in CASES],
            "U-Net-ut",
        ),
    ] + [
        (lab, mse, lsim, sk)
        for t, name in (("ref", "Reference"), ("back001", "back001"))
        for lab, mse, lsim, sk in (
            (
                name,
                ABL["single"][t]["longer_mse_all"][0],
                [ABL["single"][t][f"{c}_lsim"][0] for c in CASES],
                t,
            ),
            (
                name + " + post-proc.",
                PP2[f"mirror+tr|longer|{t}_ens"],
                [LSIM[f"{t}+pp"][c] for c in CASES],
                t + "+pp",
            ),
        )
    ]:
        r = sp[sk]
        cells = (
            [f(mse)]
            + [f(v) for v in lsim]
            + [f(r[k]) for k in ("err_low", "err_mid", "err_high", "slope_pred")]
        )
        rows.append(
            f"{lab} & "
            + " & ".join(cells)
            + r"\\"
            + (r" \midrule" if lab.startswith("U-Net") else "")
        )
    rows.append(
        f"\\emph{{Ground truth}} & & & & & & & & {f(next(iter(sp.values()))['slope_gt'])}\\\\"
    )
    (HERE / "tab_sharp.tex").write_text("\n".join(rows) + "\n")


# ------------------------------------------------------------------------- figures
def fig_da():
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.4))
    for lab, d, method, ls, col in [
        ("U-Net 4D-Var", BASE_DA1, "UNet", "--", MUTED),
        ("FNO 4D-Var", BASE_DA1, "FNO", ":", MUTED),
        ("ACDM + SDA", BASE_DA1, "ACDM", "-.", MUTED),
        (
            "Original KAE (final ckpt)",
            DA / "da_v3/legacy_ep199/df1",
            "KAE",
            (0, (1, 1)),
            INK2,
        ),
    ]:
        v = delta_f(d, method)
        if v:
            x = sorted(v)
            ax.plot(x, [v[k] for k in x], ls=ls, color=col, lw=1.0, label=lab)
    for t, col in [("ref", BLUE), ("back001", ORANGE)]:
        for variant, ls, suf in [("plain", "-", "")]:
            dirs = (
                [da_dir(t, "ens")]
                if variant == "ens"
                else [da_dir(t, "plain", s) for s in range(3)]
            )
            vs = [delta_f(d / "df1") for d in dirs]
            if any(v is None for v in vs):
                continue
            x = sorted(vs[0])
            ax.plot(
                x,
                [np.mean([v[k] for v in vs]) for k in x],
                ls=ls,
                color=col,
                marker="o",
                ms=2.5,
                label=("Reference" if t == "ref" else "back001") + suf,
            )
    ax.set_yscale("log")
    ax.set_xlabel(r"lead to first observation $\delta_f$ (frames)")
    ax.set_ylabel("analysis error (rel. $L_2$)")
    ax.legend(fontsize=6, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(HERE / "fig_da.pdf")


def fig_long():
    d = json.loads((PC / "long1000.json").read_text())["models"]
    sn = np.load(PC / "long1000_pressure_snapshots.npz")
    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.8], hspace=0.25)
    top = gs[0].subgridspec(1, 2, wspace=0.25)
    sty = {
        "ACDM": (MUTED, "--"),
        "Original KAE": (INK2, ":"),
        "ref": (BLUE, "-"),
        "ref+pp": (BLUE, "--"),
        "back001": (ORANGE, "-"),
        "back001+pp": (ORANGE, "--"),
    }
    lab = {
        "ACDM": "ACDM",
        "Original KAE": "Original KAE",
        "ref": "Reference",
        "ref+pp": "Reference + post-proc.",
        "back001": "back001",
        "back001+pp": "back001 + post-proc.",
    }
    for i, (key, yl) in enumerate(
        [("corr", "$v_x$ spatial correlation"), ("l2", "$v_x$ relative $L_2$ error")]
    ):
        ax = fig.add_subplot(top[i])
        for k, (col, ls) in sty.items():
            a = np.array(d[k][key])[:, 3:]
            ax.plot(
                np.arange(3, 3 + a.shape[1]),
                a.mean(0),
                color=col,
                ls=ls,
                lw=1.0,
                label=lab[k],
            )
        ax.set_xlabel("frame (0.05 s)")
        ax.set_ylabel(yl)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    rows = [
        ("truth", "Truth"),
        ("ACDM", "ACDM"),
        ("Original KAE", "Orig. KAE"),
        ("ref", "Reference"),
        ("ref+pp", "Reference + pp"),
        ("back001", "back001"),
        ("back001+pp", "back001 + pp"),
    ]
    bot = gs[1].subgridspec(len(rows), 4, wspace=0.03, hspace=0.06)
    vmin, vmax = np.percentile(sn["truth"], [1, 99])
    fluid = (
        torch.load(AUTOREG_RES / "longer/groundTruth.dict", weights_only=False)[
            "obsMask"
        ][0].numpy()
        > 0.5
    )
    mse = {
        k: ((sn[k] - sn["truth"]) ** 2 * fluid).mean((-1, -2)) * 1e3 for k, _ in rows
    }
    for r, (k, nm) in enumerate(rows):
        for c, st in enumerate(sn["steps"]):
            ax = fig.add_subplot(bot[r, c])
            ax.imshow(sn[k][c].T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
            if k != "truth":
                v, best = mse[k][c], min(mse[q][c] for q, _ in rows[1:])
                ax.text(
                    0.98,
                    0.05,
                    f"{v:.1f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=5.5,
                    fontweight="bold" if round(v, 1) == round(best, 1) else "normal",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.8),
                )
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            if c == 0:
                ax.set_ylabel(nm, fontsize=6.5, rotation=0, ha="right", va="center")
            if r == 0:
                ax.set_title(f"frame {st + 1}", fontsize=7, color=INK)
    fig.savefig(HERE / "fig_long.pdf")


def fig_eigen():
    e = json.loads((PC / "eigen.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3), sharey=True)
    for ax, t in zip(axs, ("ref", "back001")):
        ma = np.array(e[t]["ma"])
        g, w = np.array(e[t]["gamma"]), np.array(e[t]["omega"])
        sc = None
        for j in range(g.shape[1]):
            sc = ax.scatter(g[:, j], np.abs(w[:, j]), c=ma, cmap="viridis", s=2, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel(r"decay rate $\gamma = -\mathrm{Re}\,\lambda$ (log)")
        ax.set_title("Reference" if t == "ref" else "back001", fontsize=8, color=INK)
    axs[0].set_ylabel(r"Im $\lambda = |\omega|$")
    cb = fig.colorbar(sc, ax=axs, fraction=0.03, pad=0.01)
    cb.set_label("Mach", fontsize=7)
    fig.savefig(HERE / "fig_eigen.pdf")


if __name__ == "__main__":
    tab_main()
    tab_pp()
    tab_da_pp()
    tab_paper()
    tab_sharp()
    fig_da()
    fig_long()
    fig_eigen()
    print("collab tables and figures written")
