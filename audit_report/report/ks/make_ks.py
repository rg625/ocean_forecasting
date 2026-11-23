"""Tables and figures of the KS report (ctkae_ks.tex) from results/ks_ctkae and the main
repository's stored paper results. Run ks_summary.py first.

    python make_ks.py
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
IMPL = HERE.parents[2]
RES = IMPL / "audit_report/results"
KS = RES / "ks_ctkae"
MAIN = Path("/home/rg625/mnt/ocean_forecasting/koopman_autoencoder")
S = json.loads((KS / "summary.json").read_text())
FE = json.loads((KS / "forecast_eval.json").read_text())
PPK = json.loads((RES / "postproc_ks/pp_ks_ctkae.json").read_text())
COLS = ["legacy KAE", "U-Net", "SDA", "CT-KAE ref", "CT-KAE back001"]
HEAD = ["Paper KAE", "U-Net", "SDA", "Reference", "back001"]
BLUE, ORANGE, INK, INK2, MUTED, GREY = (
    "#2a78d6",
    "#eb6834",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#6f6e69",
)
STY = {
    "legacy KAE": (INK2, ":"),
    "U-Net": (GREY, "--"),
    "SDA": (MUTED, "-."),
    "CT-KAE ref": (BLUE, "-"),
    "CT-KAE back001": (ORANGE, "-"),
}
mpl.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    }
)


def f(v, d=4):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def bold_min(vals, d=4):
    ok = [v for v in vals if v is not None and np.isfinite(v)]
    b = min(ok) if ok else None
    return [
        (
            r"\textbf{" + f(v, d) + "}"
            if v is not None and b is not None and v == b
            else f(v, d)
        )
        for v in vals
    ]


def tab_da(camp="geometry", out="tab_ks_da.tex"):
    """one block per sweep: selected rows"""
    pick = {
        "C1_delta_f": ("first observation $\\delta_f$ (t.u.)", None),
        "C2_delta_l": ("last observation $\\delta_l$ (t.u.)", None),
        "C3_n_obs": ("observation times $N$", None),
        "noise_law": (
            "noise std (Gaussian)",
            lambda r: r["tag"].startswith("NL_gaussian"),
        ),
        "joint_sparsity": ("coverage, $N$", lambda r: r["value"][1] in (3, 5, 9, 17)),
        "S1_single_obs": ("single observation at $\\tau$", None),
    }
    rows = []
    for sec, (lab, keep) in pick.items():
        if sec not in S[camp]:
            continue
        rr = [r for r in S[camp][sec] if keep is None or keep(r)]
        rows.append(r"\multicolumn{6}{l}{\emph{" + lab + r"}}\\")
        for r in rr:
            v = r["value"]
            if sec == "C2_delta_l":
                v = f"{r['taus'][-1]:g}" + (
                    " (8000 it.)" if r["tag"].endswith("anchor") else ""
                )
            elif sec == "S1_single_obs":
                v = f"{r['taus'][0]:g}"
            elif sec == "joint_sparsity":
                v = f"{int(100 * v[0])}\\%, {len(r['taus'])}"
            elif sec == "C3_n_obs":
                v = f"{len(r['taus'])}"  # realised N: rounding to the frame grid drops duplicates
            elif sec == "C1_delta_f":
                v = f"{float(v):g}"
            rows.append(
                f"\\quad {v} & " + " & ".join(bold_min([r[c] for c in COLS])) + r"\\"
            )
    (HERE / out).write_text("\n".join(rows) + "\n")


def tab_lb():
    rows = []
    for r in S["long_baseline"]:
        rows.append(
            f"{r['delta_l_tu']:g} & {r['delta_l_TL']:.2f} & "
            + " & ".join(bold_min([r[c] for c in COLS]))
            + r"\\"
        )
    (HERE / "tab_ks_lb.tex").write_text("\n".join(rows) + "\n")


def tab_forecast():
    fc = S["forecast"]
    pts = ["0.1", "0.5", "1.0", "5.0"]
    rows = []
    for k, lab in (("U-Net", "U-Net"), ("legacy KAE", "Paper KAE")):
        r = fc[k]
        rows.append(
            f"{lab} & "
            + " & ".join(f(r["rel_at_TL"][p], 4) for p in pts)
            + f" & {r['skill_horizon_TL']:.2f} & {f(r.get('roundtrip'), 4)}\\\\"
        )
    for k, lab in (("CT-KAE ref", "Reference"), ("CT-KAE back001", "back001")):
        sd = fc[k]["seeds"]
        m = {p: np.mean([x["rel_at_TL"][p] for x in sd]) for p in pts}
        e = {
            p: np.std([x["rel_at_TL"][p] for x in sd], ddof=1) / np.sqrt(len(sd))
            for p in pts
        }
        sh = [x["skill_horizon_TL"] for x in sd]
        rt = [x["roundtrip"] for x in sd]
        rows.append(
            f"{lab} & "
            + " & ".join(f"{m[p]:.3f}$\\pm${e[p]:.3f}" for p in pts)
            + f" & {np.mean(sh):.2f}$\\pm${np.std(sh, ddof=1) / np.sqrt(len(sh)):.2f} & {np.mean(rt):.4f}\\\\"
        )
    (HERE / "tab_ks_forecast.tex").write_text("\n".join(rows) + "\n")
    # one-step and generator facts (test.nc)
    gen = []
    for k in ("ref", "back001"):
        g = [FE["models"][f"{k}_s{s}"]["generator"] for s in range(3)]
        gen.append(
            {
                "model": k,
                "real_max": [x["real_max"] for x in g],
                "n_pos": [x["n_real_gt_0"] for x in g],
            }
        )
    leg = FE["models"]["legacy KAE"]["generator"]
    (HERE / "ks_generator.json").write_text(
        json.dumps(
            {"ctkae": gen, "legacy_current_code": leg, "lambda_1": FE["lambda_1"]},
            indent=1,
        )
    )


def tab_postda():
    pd = S["post_da"]
    rows = []
    for k, lab in (
        ("U-Net", "U-Net"),
        ("legacy KAE", "Paper KAE"),
        ("CT-KAE ref", "Reference"),
        ("CT-KAE back001", "back001"),
    ):
        if k not in pd:
            rows.append(f"{lab} & -- & -- & -- & --\\\\")
            continue
        r = pd[k]
        rows.append(
            f"{lab} & {r['analysis_rel']:.4f} & {r['rel_at_TL']['0.1']:.4f} & {r['rel_at_TL']['1.0']:.4f} & {r['skill_horizon_TL']:.2f}\\\\"
        )
    (HERE / "tab_ks_postda.tex").write_text("\n".join(rows) + "\n")


def tab_pp():
    C = [
        ("single", "single decode"),
        ("mirror", "+ mirror"),
        ("sym8", "+ 4 shifts $\\times$ mirror"),
        ("sharp(sym8)", "\\quad + sharpen"),
        ("lpm(sym8)", "\\quad + LPM"),
    ]
    rows = []
    for cand, lab in (("ref", "Reference"), ("back001", "back001")):
        for i, (c, cl) in enumerate(C):
            cells = []
            for st in ("short", "long"):
                Sx = [PPK[f"{cand}_s{s}"]["test"][st]["stats"] for s in range(3)]
                s0 = np.array([x["single"]["mse"] for x in Sx])
                v = np.array([x[c]["mse"] for x in Sx])
                cells.append(
                    f"{v.mean():.3f}"
                    if c == "single"
                    else f"{(v / s0).mean():.2f}$\\pm${(v / s0).std(ddof=1) / np.sqrt(3):.2f}"
                )
                cells += [
                    f"{np.mean([x[c]['log_' + b] for x in Sx]):+.2f}"
                    for b in ("low", "high")
                ]
            lead = (r"\multirow{5}{*}{" + lab + "}") if i == 0 else ""
            rows.append(
                f"{lead} & {cl} & "
                + " & ".join(cells)
                + r"\\"
                + (r" \midrule" if (i == 4 and cand == "ref") else "")
            )
    T0 = PPK["ref_s0"]["test"]
    rows.append(
        r"\midrule \multicolumn{2}{l}{U-Net} & "
        + f"{T0['short']['stats']['U-Net']['mse']:.1e} & & & {T0['long']['stats']['U-Net']['mse']:.1e} & & \\\\"
    )
    rows.append(
        r"\multicolumn{2}{l}{persistence} & "
        + f"{T0['short']['stats']['persistence']['mse']:.3f} & & & {T0['long']['stats']['persistence']['mse']:.3f} & & \\\\"
    )
    (HERE / "tab_ks_pp.tex").write_text("\n".join(rows) + "\n")


def fig_ks():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    TL = S["forecast"]["lyapunov_time_tu"]
    fc = np.load(MAIN / "da_results_long/FC_forecast.npz")
    ax = axs[0]
    ax.loglog(
        fc["taus"] / TL,
        fc["UNet__rel_mean"],
        color=STY["U-Net"][0],
        ls=STY["U-Net"][1],
        lw=1,
        label="U-Net",
    )
    ax.loglog(
        fc["taus"] / TL,
        fc["KAE-expm__rel_mean"],
        color=STY["legacy KAE"][0],
        ls=STY["legacy KAE"][1],
        lw=1.2,
        label="Paper KAE",
    )
    for k, t in (("CT-KAE ref", "ref"), ("CT-KAE back001", "back001")):
        for s in range(3):
            d = np.load(KS / f"{t}_s{s}/long/FC_forecast.npz")
            ax.loglog(
                d["taus"] / TL,
                d["KAE-expm__rel_mean"],
                color=STY[k][0],
                lw=0.8,
                alpha=0.8,
                label=("Reference" if t == "ref" else "back001") if s == 0 else None,
            )
    ax.axhline(1, color=MUTED, lw=0.5)
    ax.set_xlabel("lead (Lyapunov times)")
    ax.set_ylabel("relative error")
    ax.set_title("(a) forecast from the true state", fontsize=7.5)
    ax.legend(fontsize=5.5, loc="lower right")
    for ax, sec, xl, title in (
        (
            axs[1],
            "C1_delta_f",
            "first observation $\\delta_f$ (t.u.)",
            "(b) DA, $\\delta_l = 2.5$, $N = 5$",
        ),
        (
            axs[2],
            "C2_delta_l",
            "last observation $\\delta_l$ (t.u.)",
            "(c) DA, $\\delta_f = 0.1$, $N = 5$",
        ),
    ):
        rr = [r for r in S["geometry"][sec] if not r["tag"].endswith("anchor")]
        x = [float(r["value"]) for r in rr]
        for c, lab in zip(COLS, HEAD):
            y = [r[c] if r[c] is not None else np.nan for r in rr]
            ax.loglog(
                x, y, color=STY[c][0], ls=STY[c][1], marker="o", ms=2.2, lw=1, label=lab
            )
        ax.set_xlabel(xl)
        ax.set_title(title, fontsize=7.5)
    axs[1].set_ylabel("analysis error at $t_0$")
    h, lab = axs[2].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(
        h, lab, loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.66, 0)
    )
    fig.savefig(HERE / "fig_ks.pdf")


def fig_fields():
    pap = np.load(MAIN / "da_results_geometry/C1_delta_f_fields.npz", allow_pickle=True)
    ct = {
        k: np.load(KS / f"{t}_s0/geometry/C1_delta_f_fields.npz", allow_pickle=True)
        for k, t in (("Reference", "ref"), ("back001", "back001"))
    }
    x = pap["x"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.1), sharey=True)
    for ax, tag, title in (
        (axs[0], "C1_df1", "$\\delta_f = 0.1$"),
        (axs[1], "C1_df8", "$\\delta_f = 0.8$"),
    ):
        tr = pap[f"{tag}__truth_t0"][0]
        assert np.allclose(
            tr, ct["Reference"][f"{tag}__truth_t0"][0]
        ), "different problem"
        ax.plot(x, tr, color=INK, lw=1.8, label="truth")
        for lab, k, (col, ls) in (
            ("Paper KAE", "KAE-expm", STY["legacy KAE"]),
            ("U-Net", "UNet", STY["U-Net"]),
            ("SDA", "SDA", STY["SDA"]),
        ):
            ax.plot(
                x, pap[f"{tag}__{k}__analysis"][0], color=col, ls=ls, lw=1, label=lab
            )
        for lab, d in ct.items():
            ax.plot(
                x,
                d[f"{tag}__KAE-expm__analysis"][0],
                color=BLUE if lab == "Reference" else ORANGE,
                lw=1,
                label=lab,
            )
        ax.set_title(f"analysis at $t_0$, first problem, {title}", fontsize=7.5)
        ax.set_xlabel("$x$ (grid points)")
    axs[0].set_ylabel("$u$ (normalised)")
    axs[1].legend(fontsize=5.5, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "fig_ks_fields.pdf")


if __name__ == "__main__":
    tab_da()
    tab_da("geometry_df9", "tab_ks_da9.tex")
    tab_lb()
    tab_forecast()
    tab_postda()
    tab_pp()
    fig_ks()
    fig_fields()
    print("ks tables and figures written")
