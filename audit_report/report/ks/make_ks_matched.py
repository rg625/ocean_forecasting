"""Tables and figure of the matched-training section of ctkae_ks.tex: the paper KAE
architecture (control) and the CT-KAE candidates trained with the paper's KS pipeline
(scripts/ks_matched), evaluated with the paper's KS suite, next to the paper's stored
results. Models not yet evaluated show as '--'.

    python make_ks_matched.py
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RES = HERE.parents[1] / "results/ks_matched"
MAIN = Path("/home/rg625/mnt/ocean_forecasting/koopman_autoencoder")
MODELS = [
    ("kae_128_s0", "KAE (ctrl)"),
    ("ct_ref_128_s0", "ref 128"),
    ("ct_back001_128_s0", "back001 128"),
    ("ct_ref_256_s0", "ref 256"),
    ("ct_back001_256_s0", "back001 256"),
]
PAPER = [("KAE-expm", "Paper KAE"), ("UNet", "U-Net"), ("SDA", "SDA")]
COL = {
    "KAE (ctrl)": "#52514e",
    "ref 128": "#2a78d6",
    "back001 128": "#eb6834",
    "ref 256": "#7fb2ea",
    "back001 256": "#f3a77f",
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


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def f(v, d=4):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def bold_min(vals, d=4):
    ok = [v for v in vals if v is not None and np.isfinite(v)]
    b = min(ok) if ok else None
    return [
        r"\textbf{" + f(v, d) + "}" if b is not None and v == b else f(v, d)
        for v in vals
    ]


def kae_mean(row):
    v = None if row is None else row.get("KAE-expm")
    return (
        None if not isinstance(v, dict) or v.get("mean") is None else float(v["mean"])
    )


def tab_fc():
    TL = float(np.load(MAIN / "da_results_long/FC_forecast.npz")["T_L"])
    fc = np.load(MAIN / "da_results_long/FC_forecast.npz")
    pts = [0.1, 0.5, 1.0, 5.0]
    rows = []
    for k, lab in (("UNet", "U-Net (paper)"), ("KAE-expm", "Paper KAE (paper)")):
        at = [float(np.interp(p * TL, fc["taus"], fc[f"{k}__rel_mean"])) for p in pts]
        h = float(fc[f"{k}__skill_horizon_tu"]) / TL
        rt = f(float(fc["kae_roundtrip"])) if k == "KAE-expm" else "--"
        rows.append(
            f"{lab} & single & "
            + " & ".join(f(v) for v in at)
            + f" & {h:.2f} & {rt}\\\\"
        )
    rows.append(r"\midrule")
    for n, lab in MODELS:
        d = jl(RES / n / "fc_matched.json")
        for i, enc in enumerate(("single", "window")):
            lead = lab if i == 0 else ""
            if d is None:
                rows.append(f"{lead} & {enc} & " + " & ".join(["--"] * 6) + r"\\")
                continue
            r = d["models"][n][enc]
            rows.append(
                f"{lead} & {enc} & "
                + " & ".join(f(r["rel_at_TL"][str(p)]) for p in pts)
                + f" & {r['skill_horizon_TL']:.2f} & {f(r['roundtrip'])}\\\\"
            )
    (HERE / "tab_ksm_fc.tex").write_text("\n".join(rows) + "\n")


def geo(sec):
    pap = jl(MAIN / f"da_results_geometry/{sec}.json")
    out = []
    for r in pap["rows"]:
        vals = [
            (
                None
                if r.get(k) is None or r[k].get("mean") is None
                else float(r[k]["mean"])
            )
            for k, _ in PAPER
        ]
        for n, _ in MODELS:
            d = jl(RES / n / f"geometry/{sec}.json")
            hit = (
                None
                if d is None
                else next((x for x in d["rows"] if x["tag"] == r["tag"]), None)
            )
            if hit is not None:
                assert list(hit["frames"]) == list(r["frames"]), (sec, r["tag"])
            vals.append(kae_mean(hit))
        out.append((r, vals))
    return out


def tab_da():
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
        rows.append(r"\multicolumn{9}{l}{\emph{" + lab + r"}}\\")
        for r, vals in geo(sec):
            if keep is not None and not keep(r):
                continue
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
                v = f"{len(r['taus'])}"
            else:
                v = f"{float(v):g}"
            rows.append(f"\\quad {v} & " + " & ".join(bold_min(vals)) + r"\\")
    (HERE / "tab_ksm_da.tex").write_text("\n".join(rows) + "\n")


def tab_lb_pf():
    pap = jl(MAIN / "da_results_long/LB_long_baseline.json")
    rows = []
    for r in pap["rows"]:
        vals = [
            (
                None
                if r.get(k) is None or r[k].get("mean") is None
                else float(r[k]["mean"])
            )
            for k, _ in PAPER
        ]
        for n, _ in MODELS:
            d = jl(RES / n / "long/LB_long_baseline.json")
            hit = (
                None
                if d is None
                else next(
                    (
                        x
                        for x in d["rows"]
                        if x["delta_l_frames"] == r["delta_l_frames"]
                    ),
                    None,
                )
            )
            vals.append(kae_mean(hit))
        rows.append(f"{r['delta_l_TL']:.2f} & " + " & ".join(bold_min(vals)) + r"\\")
    (HERE / "tab_ksm_lb.tex").write_text("\n".join(rows) + "\n")
    pf = np.load(MAIN / "da_results_long/PF_post_da_forecast.npz")
    TL = float(pf["T_L"])
    rows = []

    def row(lab, d, m):
        a = float(d[f"{m}__analysis_rel"])
        at = [
            float(np.interp(p * TL, d["taus"], d[f"{m}__rel_mean"])) for p in (0.1, 1.0)
        ]
        h = float(d[f"{m}__skill_horizon_tu"]) / TL
        return f"{lab} & {a:.4f} & {at[0]:.4f} & {at[1]:.4f} & {h:.2f}\\\\"

    rows.append(row("U-Net (paper)", pf, "UNet"))
    rows.append(row("Paper KAE (paper)", pf, "KAE-expm"))
    for n, lab in MODELS:
        p = RES / n / "long/PF_post_da_forecast.npz"
        rows.append(
            row(lab, np.load(p), "KAE-expm")
            if p.exists()
            else f"{lab} & -- & -- & -- & --\\\\"
        )
    (HERE / "tab_ksm_pf.tex").write_text("\n".join(rows) + "\n")


def fig():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    fc = np.load(MAIN / "da_results_long/FC_forecast.npz")
    TL = float(fc["T_L"])
    ax = axs[0]
    ax.loglog(
        fc["taus"] / TL,
        fc["UNet__rel_mean"],
        color="#6f6e69",
        ls="--",
        lw=1,
        label="U-Net (paper)",
    )
    ax.loglog(
        fc["taus"] / TL,
        fc["KAE-expm__rel_mean"],
        color="#0b0b0b",
        ls=":",
        lw=1.2,
        label="Paper KAE (paper)",
    )
    for n, lab in MODELS:
        p = RES / n / "fc_matched.npz"
        if p.exists():
            d = np.load(p)
            ax.loglog(
                d["taus"] / TL,
                d[f"{n}__window__rel_mean"],
                color=COL[lab],
                lw=1,
                label=lab,
            )
    ax.axhline(1, color="#898781", lw=0.5)
    ax.set_xlabel("lead (Lyapunov times)")
    ax.set_ylabel("relative error")
    ax.set_title("(a) forecast from the true state (window)", fontsize=7.5)
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
        g = [(r, v) for r, v in geo(sec) if not r["tag"].endswith("anchor")]
        x = [r["taus"][0] if sec == "C1_delta_f" else r["taus"][-1] for r, _ in g]
        labs = [lab for _, lab in PAPER] + [lab for _, lab in MODELS]
        sty = {
            "Paper KAE": ("#0b0b0b", ":"),
            "U-Net": ("#6f6e69", "--"),
            "SDA": ("#898781", "-."),
        }
        for j, lab in enumerate(labs):
            y = np.array([np.nan if v[j] is None else v[j] for _, v in g], dtype=float)
            if np.isfinite(y).any():
                c, ls = sty.get(lab, (COL.get(lab, "k"), "-"))
                ax.loglog(x, y, color=c, ls=ls, marker="o", ms=2, lw=1, label=lab)
        ax.set_xlabel(xl)
        ax.set_title(title, fontsize=7.5)
    axs[1].set_ylabel("analysis error at $t_0$")
    h, lab = axs[2].get_legend_handles_labels()
    h0, l0 = axs[0].get_legend_handles_labels()
    for hh, ll in zip(h0, l0):
        if ll not in lab:
            h.append(hh)
            lab.append(ll)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=6.5)
    fig.savefig(HERE / "fig_ksm.pdf")


if __name__ == "__main__":
    tab_fc()
    tab_da()
    tab_lb_pf()
    fig()
    print("matched tables and figure written")
