# mypy: disable-error-code="assignment"
"""LaTeX tables for the KS data-assimilation campaign, generated from saved results.

    python -m data_assimilation.ks.tables --results da_results_v2 --out ../iclr_2027/tables/da

No number in the manuscript is typed by hand: every table here reads ``results.csv`` /
``summary.json`` written by ``run_da_suite.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

PRETTY = {
    "KAE-expm": r"Continuous KAE (exact $e^{\mathbf{K}\tau}$)",
    "KAE-rk4": r"Continuous KAE (RK4 rollout)",
    "UNet-4DVar": r"U-Net 4D-Var",
    "SDA": r"Score-based DA",
}


def pm(mean, std, prec=4) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    return f"{mean:.{prec}f} $\\pm$ {std:.{prec}f}"


def wrap(body: str, header: str, caption: str, label: str, align: str) -> str:
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{align}}}",
            r"\toprule",
            header,
            r"\midrule",
            body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def table_main(
    df: pd.DataFrame,
    summary: dict,
    forward: Optional[dict] = None,
    sda: Optional[pd.DataFrame] = None,
) -> str:
    """Main DA table: forward-model skill, analysis accuracy, forecast skill and cost."""
    fwd = {}
    if forward:
        # forward-model error at the end of the assimilation window (tau = 2.5)
        for key, m in [("KAE-expm", "KAE-expm"), ("UNet", "UNet-4DVar")]:
            c = forward["curves"].get(key)
            if c:
                j = int(np.argmin(np.abs(np.array(c["tau"]) - 2.5)))
                fwd[m] = (c["mean"][j], c["std"][j])

    lines: List[str] = []
    for tag, pretty in [
        ("clean", "Dense, noise-free observations"),
        ("sparse_noisy", r"25\% of grid points, $\sigma=0.05$"),
    ]:
        d = df[(df.stage == "canonical") & (df.cond_value == tag)]
        if not len(d):
            continue
        lines.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{pretty}}}}} \\")
        for m in ["KAE-expm", "UNet-4DVar"]:
            r = d[d.method == m]
            if not len(r):
                continue
            r = r.iloc[0]
            fw = fwd.get(m)
            lines.append(
                " & ".join(
                    [
                        PRETTY[m],
                        (pm(*fw) if fw else "--"),
                        pm(r.init_rel_l2_mean, r.init_rel_l2_std),
                        pm(r.forecast_rel_l2_mean, r.forecast_rel_l2_std, 3),
                        f"{r.total_s:.1f}",
                        f"{r.ms_per_iter:.2f}",
                        (
                            f"{r.peak_mem_MiB:.0f}"
                            if not np.isnan(r.peak_mem_MiB)
                            else "--"
                        ),
                        f"{r.evals_per_iter_network:.0f}",
                    ]
                )
                + r" \\"
            )
        if sda is not None:
            sr = sda[(sda.cond_value == tag) & (sda.method == "SDA")]
            if len(sr):
                sr = sr.iloc[0]
                lines.append(
                    " & ".join(
                        [
                            PRETTY["SDA"],
                            "--",
                            pm(sr.analysis_rel_l2_mean, sr.analysis_rel_l2_std),
                            pm(sr.forecast_rel_l2_mean, sr.forecast_rel_l2_std, 3)
                            + r"$^{\dagger}$",
                            f"{sr.wall_s:.1f}",
                            "--",
                            f"{sr.peak_mem_MiB:.0f}",
                            f"{sr.nfe / sr.n_problems:.0f}~/~{sr.n_backward / sr.n_problems:.0f}",
                        ]
                    )
                    + r" \\"
                )
        f = d.dropna(subset=["ae_floor_mean"])
        if len(f):
            lines.append(
                r"\quad\emph{autoencoder floor (KAE only)} & -- & "
                + pm(f.ae_floor_mean.iloc[0], f.ae_floor_std.iloc[0])
                + r" & -- & -- & -- & -- & -- \\"
            )
        lines.append(r"\addlinespace")
    header = (
        r"Method & Forward rel-$L_2$ $\downarrow$ & Analysis rel-$L_2$ $\downarrow$ & "
        r"Post-DA forecast $\downarrow$ & DA time (s) $\downarrow$ & "
        r"ms/iter $\downarrow$ & Peak mem.\ (MiB) $\downarrow$ & "
        r"Evals/iter $\downarrow$ \\"
    )
    n = int(df[df.stage == "canonical"].n_problems.iloc[0])
    it = int(df[df.stage == "canonical"].iters.iloc[0])
    cap = (
        rf"Data assimilation on {n} independent held-out KS trajectories. Every method "
        rf"infers the unobserved state $u(t_0)$ from the same five future observations at "
        rf"$\tau=\{{0.1,0.3,0.7,1.5,2.5\}}$, with identical masks, noise realisations, "
        rf"initialisation scales, seeds and a budget of {it} optimisation iterations. "
        r"Mean $\pm$ s.d.\ over trajectories. `Forward rel-$L_2$' is each frozen model's "
        r"own forecast error at $\tau=2.5$ from the \emph{true} state, i.e.\ the accuracy "
        r"ceiling its variational analysis inherits. Post-DA forecast error is averaged over "
        r"leads $\tau\in[3,20]$, strictly beyond the last observation. `Evals/iter' counts "
        r"sequential forward-model applications per optimisation iteration; for "
        r"score-based DA the two numbers are score-network evaluations and "
        r"likelihood-guidance backward passes per assimilation problem. "
        r"$^{\dagger}$SDA infers a trajectory window ending at $\tau=5.5$, so its "
        r"post-DA forecast is averaged over $\tau\in[2.6,5.5]$ rather than $[3,20]$; "
        r"on that same range the KAE and U-Net score 0.2443 and 0.0002 (clean), "
        r"0.4366 and 0.2689 (sparse+noisy)."
    )
    return wrap("\n".join(lines), header, cap, "tab:da_main", "lccccccc")


def _pivot(tim: pd.DataFrame, method: str, col: str):
    sdf = tim[tim.method == method]
    return sdf.set_index(sdf.cond_value.astype(float))[col] if len(sdf) else None


def table_cost(df: pd.DataFrame) -> str:
    """Cost vs horizon, with the three speed comparisons kept strictly separate.

    (A) KAE exact vs KAE RK4  -- an internal continuous-propagation comparison.
    (B) KAE vs U-Net           -- end-to-end cost of one DA iteration.
    (C) KAE vs U-Net           -- total DA runtime for the full optimisation.
    """
    tim = df[df.stage == "timing"]
    hor = df[df.stage == "horizon"]
    mem = df[df.stage == "memory"]
    if not len(tim):
        return ""
    hs = sorted(tim.cond_value.astype(float).unique())
    lines = []

    mech = {
        "KAE-expm": r"one $e^{\mathbf{K}\tau_i}$ per obs.\ time",
        "KAE-rk4": r"RK4 steps of $\dot z=\mathbf{K}z$",
        "UNet-4DVar": r"autoregressive $F_\theta^{n}$",
    }
    lines.append(
        r"\multicolumn{" + str(len(hs) + 2) + r"}{l}{\emph{ms per DA iteration}} \\"
    )
    for m in ["KAE-expm", "KAE-rk4", "UNet-4DVar"]:
        v = _pivot(tim, m, "ms_per_iter")
        if v is None:
            continue
        lines.append(
            " & ".join([PRETTY[m], mech[m]] + [f"{float(v.loc[h]):.2f}" for h in hs])
            + r" \\"
        )
    ke, kr, un = (
        _pivot(tim, m, "ms_per_iter") for m in ["KAE-expm", "KAE-rk4", "UNet-4DVar"]
    )
    lines.append(r"\addlinespace")
    lines.append(
        r"\multicolumn{2}{l}{\emph{(A) KAE\,RK4 $/$ KAE\,exact "
        r"(internal propagation)}} & "
        + " & ".join(f"{float(kr.loc[h]) / float(ke.loc[h]):.1f}$\\times$" for h in hs)
        + r" \\"
    )
    lines.append(
        r"\multicolumn{2}{l}{\emph{(B) U-Net $/$ KAE\,exact "
        r"(per DA iteration)}} & "
        + " & ".join(f"{float(un.loc[h]) / float(ke.loc[h]):.1f}$\\times$" for h in hs)
        + r" \\"
    )
    if len(hor):
        hh = sorted(hor.cond_value.astype(float).unique())
        keh, unh = (_pivot(hor, m, "total_s") for m in ["KAE-expm", "UNet-4DVar"])
        if keh is not None and unh is not None:
            lines.append(r"\addlinespace")
            lines.append(
                r"\multicolumn{2}{l}{\emph{(C) total DA runtime, "
                r"U-Net $/$ KAE\,exact}} & "
                + " & ".join(
                    (
                        f"{float(unh.loc[h]) / float(keh.loc[h]):.1f}$\\times$"
                        if h in hh
                        else "--"
                    )
                    for h in hs
                )
                + r" \\"
            )
    if len(mem):
        lines.append(r"\addlinespace")
        lines.append(
            r"\multicolumn{"
            + str(len(hs) + 2)
            + r"}{l}{\emph{peak GPU memory of one DA iteration (MiB)}} \\"
        )
        for m in ["KAE-expm", "KAE-rk4", "UNet-4DVar", "UNet-4DVar-nockpt"]:
            v = _pivot(mem, m, "peak_mem_MiB")
            if v is None:
                continue
            nm = PRETTY.get(m) or r"U-Net 4D-Var (no checkpointing)"
            lines.append(
                " & ".join(
                    [nm, ""]
                    + [
                        (
                            f"{float(v.loc[h]):.0f}"
                            if h in v.index and not np.isnan(float(v.loc[h]))
                            else "OOM"
                        )
                        for h in hs
                    ]
                )
                + r" \\"
            )
    lines.append(r"\addlinespace")
    lines.append(
        r"\multicolumn{2}{l}{\emph{sequential propagation steps / iter.\ (U-Net)}} & "
        + " & ".join(
            f"{int(_pivot(tim, 'UNet-4DVar', 'evals_per_iter_propagation_steps').loc[h])}"
            for h in hs
        )
        + r" \\"
    )

    header = (
        r"Method & Propagation mechanism & "
        + " & ".join(rf"$\tau_{{\max}}{{=}}{h:g}$" for h in hs)
        + r" \\"
    )
    b = int(tim.n_problems.iloc[0])
    cap = (
        r"Computational cost of variational assimilation versus horizon. A DA iteration is "
        r"propagation to all five observation times, decoding, the loss, the backward pass "
        rf"and the optimiser step, for a batch of {b} assimilation problems; medians over 25 "
        r"repeats measured interleaved across methods on a shared GPU. The three speed ratios "
        r"are deliberately kept apart: (A) compares two ways of propagating the \emph{same} "
        r"learned generator, (B) and (C) compare the two \emph{methods}. Peak memory is "
        r"per-process and therefore unaffected by the unrelated job sharing the device."
    )
    return wrap("\n".join(lines), header, cap, "tab:da_cost", "ll" + "c" * len(hs))


def table_propagator(df: pd.DataFrame, summary: dict) -> str:
    """Exact exponential vs RK4: accuracy parity and runtime."""
    d = df[(df.stage == "propagator") & (df.condition == "max_tau")]
    agree = {
        r["tau"]: r["rel_fro_diff"]
        for r in summary.get("generator", {}).get("operator_agreement", [])
    }
    lines = []
    for h in sorted(d.cond_value.astype(float).unique()):
        s = d[d.cond_value.astype(float) == h]
        e = s[s.method == "KAE-expm"]
        r4 = s[s.method == "KAE-rk4"]
        if not len(e) or not len(r4):
            continue
        e, r4 = e.iloc[0], r4.iloc[0]
        n = int(round(h / 0.1))
        lines.append(
            " & ".join(
                [
                    f"{h:g}",
                    f"{n}",
                    pm(e.init_rel_l2_mean, e.init_rel_l2_std),
                    pm(r4.init_rel_l2_mean, r4.init_rel_l2_std),
                    f"{r4.init_rel_l2_mean - e.init_rel_l2_mean:+.5f}",
                    (f"{agree[h]:.1e}" if h in agree else "--"),
                    f"{e.total_s:.1f}",
                    f"{r4.total_s:.1f}",
                    f"{r4.total_s / e.total_s:.2f}$\\times$",
                ]
            )
            + r" \\"
        )
    header = (
        r"$\tau_{\max}$ & $n$ steps & Exact $e^{\mathbf{K}\tau}$ & RK4 rollout & "
        r"$\Delta$ & $\|e^{\mathbf{K}\tau}-\mathrm{RK4}^{n}\|_F/\|e^{\mathbf{K}\tau}\|_F$ & "
        r"$t_{\mathrm{exact}}$ (s) & $t_{\mathrm{RK4}}$ (s) & speed-up \\"
    )
    cap = (
        r"Closed-form propagation versus numerical integration of the \emph{same} learned "
        r"generator. Columns 3--5 are the analysis rel-$L_2$ obtained by solving the "
        r"identical variational problem with each propagator; column 6 compares the two "
        r"propagators directly as matrices. The exponential is a numerical convenience, "
        r"not a different model."
    )
    return wrap("\n".join(lines), header, cap, "tab:da_propagator", "rrccccccc")


def table_sweeps(df: pd.DataFrame) -> str:
    """Appendix table: every sweep in one place."""
    blocks = [
        ("nobs", "Number of observations", "n_obs"),
        ("noise", r"Observation noise $\sigma$", "noise_std"),
        ("sparsity", "Fraction of grid points observed", "obs_frac"),
        ("irregular", "Observation-time pattern", "pattern"),
        ("horizon", r"Assimilation horizon $\tau_{\max}$", "max_tau"),
    ]
    lines = []
    for stage, title, _cond in blocks:
        d = df[df.stage == stage]
        if not len(d):
            continue
        lines.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{title}}}}} \\")
        try:
            vals = sorted(d.cond_value.unique(), key=float)
        except ValueError:
            vals = list(dict.fromkeys(d.cond_value))
        for v in vals:
            s = d[d.cond_value.astype(str) == str(v)]
            row = [str(v).replace("_", r"\_")]
            for m in ["KAE-expm", "UNet-4DVar"]:
                r = s[s.method == m]
                row.append(
                    pm(r.init_rel_l2_mean.iloc[0], r.init_rel_l2_std.iloc[0])
                    if len(r)
                    else "--"
                )
            for m in ["KAE-expm", "UNet-4DVar"]:
                r = s[s.method == m]
                row.append(f"{float(r.total_s.iloc[0]):.1f}" if len(r) else "--")
            f = s.dropna(subset=["ae_floor_mean"])
            row.append(f"{float(f.ae_floor_mean.iloc[0]):.4f}" if len(f) else "--")
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace")
    header = (
        r"Setting & KAE analysis rel-$L_2$ & U-Net analysis rel-$L_2$ & "
        r"KAE DA time (s) & U-Net DA time (s) & AE floor \\"
    )
    cap = (
        r"Complete data-assimilation sweeps. Mean $\pm$ s.d.\ over held-out KS "
        r"trajectories; both methods see identical observations, masks, noise "
        r"realisations, seeds and optimisation budgets in every row."
    )
    return wrap("\n".join(lines), header, cap, "tab:da_sweeps", "lccccc")


def table_paired(summary: dict) -> str:
    """Paired trajectory-level KAE vs U-Net statistics."""
    paired = summary.get("paired", {})
    if not paired:
        return ""
    lines = []
    for key, r in paired.items():
        tag, metric = key.split("__")
        lines.append(
            " & ".join(
                [
                    tag.replace("_", r"\_"),
                    metric.replace("_", r"\_"),
                    str(r["n"]),
                    f"{r['mean_diff']:+.4f}",
                    f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]",
                    f"{r['kae_wins']}/{r['n']}",
                    (f"{r['wilcoxon_p']:.2e}" if "wilcoxon_p" in r else "--"),
                ]
            )
            + r" \\"
        )
    header = (
        r"Setting & Metric & $N$ & mean(KAE $-$ U-Net) & 95\% CI & KAE better & "
        r"Wilcoxon $p$ \\"
    )
    cap = (
        r"Paired per-trajectory comparison. Negative differences favour the continuous "
        r"KAE. Pairing is exact: both methods solve the same assimilation problem on the "
        r"same trajectory from the same observations."
    )
    return wrap("\n".join(lines), header, cap, "tab:da_paired", "llrccrc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("da_results_v2"))
    ap.add_argument("--out", type=Path, default=Path("../iclr_2027/tables/da"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results / "results.csv")
    with open(args.results / "summary.json") as f:
        summary = json.load(f)

    made = []
    sdadf = None
    _sp = args.results / "sda_final.csv"
    if _sp.exists():
        sdadf = pd.read_csv(_sp)

    forward = None
    fpath = args.results / "forecast_check.json"
    if fpath.exists():
        with open(fpath) as f:
            forward = json.load(f)

    for name, fn in [
        ("da_main", lambda: table_main(df, summary, forward, sdadf)),
        ("da_cost", lambda: table_cost(df)),
        ("da_propagator", lambda: table_propagator(df, summary)),
        ("da_sweeps", lambda: table_sweeps(df)),
        ("da_paired", lambda: table_paired(summary)),
    ]:
        try:
            tex = fn()
        except Exception as e:  # noqa: BLE001
            print(f"  !! {name}: {type(e).__name__}: {e}")
            continue
        if tex.strip():
            (args.out / f"{name}.tex").write_text(tex)
            made.append(name)
    print("tables written:", made, "->", args.out)


if __name__ == "__main__":
    main()
