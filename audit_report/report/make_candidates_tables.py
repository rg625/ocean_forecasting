"""Tables for ctkae_candidates.tex, regenerated from the ablation runs (partial results fine).

    python make_candidates_tables.py   -> tab_abl_single.tex, tab_abl_mix.tex, tab_main.tex

Ablation tables: per variant the paired change vs the reference run of the same seed (mean
over seeds +- SE of the paired change); the reference row gives absolute values. Bold: an
improvement larger than 2 SE; a dagger: a deterioration larger than 2 SE.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
SCR = ROOT / "audit_report/scripts"
PY = "/home/rg625/anaconda3/envs/samudra/bin/python"
sys.path.insert(0, str(SCR))
from ablation_table import GROUPS, KEYS, LABELS, baselines  # noqa: E402

DA3 = ROOT / "koopman_autoencoder/model_outputs_ctkae/da_v3"
E0 = json.loads(
    (ROOT / "audit_report/results/E0_legacy_run20260821_ep199.json").read_text()
)


def tex_label(s):
    return (
        s.replace("->", r"$\to$")
        .replace("mu(Ma)", r"$\mu$(Ma)")
        .replace("_", r"\_")
        .replace(" x rms", r" $\times$ rms")
    )


def cell(d, se, k):
    dd = 3 if k == "da" else 2
    if d is None:
        return "--"
    s = f"{d:+.{dd}f}" + (f"$\\pm${se:.{dd}f}" if se else "")
    if se and abs(d) > 2 * se:
        s = f"\\textbf{{{s}}}" if d < 0 else s + "$^\\dagger$"
    return s


def ablation(js, out):
    s = json.loads(js.read_text())
    rows = []
    ref = s["ref"]
    rows.append(
        r"\emph{Reference (absolute)} & 3 & "
        + " & ".join(f"{ref[k][0]:.{3 if k == 'da' else 2}f}" for k, _ in KEYS)
        + r"\\ \midrule"
    )
    for title, tags in GROUPS[1:]:
        present = [t for t in tags if t in s]
        if not present:
            continue
        rows.append(f"\\multicolumn{{{len(KEYS) + 2}}}{{l}}{{\\emph{{{title}}}}}\\\\")
        for t in present:
            o = s[t]
            cells = [cell(o.get(f"d_{k}"), o.get(f"dse_{k}"), k) for k, _ in KEYS]
            rows.append(
                f"{tex_label(LABELS[t])} & {o['n']} & " + " & ".join(cells) + r"\\"
            )
    sens = sorted(t for t in s if t.startswith("S_"))
    if sens:
        rows.append(
            f"\\multicolumn{{{len(KEYS) + 2}}}{{l}}{{\\emph{{Weight sensitivity (seed 0)}}}}\\\\"
        )
        for t in sens:
            o = s[t]
            cells = [cell(o.get(f"d_{k}"), None, k) for k, _ in KEYS]
            rows.append(
                f"{tex_label(t.replace('S_', 'weight: ').replace('_', ' '))} & 1 & "
                + " & ".join(cells)
                + r"\\"
            )
    out.write_text("\n".join(rows) + "\n")


def da6(lab):
    secs = [
        "G4_delta_l",
        "G5_n_obs",
        "G3_sparsity",
        "G2_noise",
        "G19_misspec",
        "G20_joint_sparsity",
    ]
    v = [
        r["KAE"]["mean"]
        for s in secs
        for r in json.loads((DA3 / lab / "df9" / f"{s}.json").read_text())["rows"]
        if "KAE" in r
    ]
    return float(np.mean(v))


def main_table(single, mix, out):
    s, m = json.loads(single.read_text()), json.loads(mix.read_text())
    base = dict(baselines())
    base_da6 = {}
    bd = Path(
        "/home/rg625/mnt/ocean_forecasting/koopman_autoencoder/da_results_tra_df9"
    )
    for meth in ["ACDM", "UNet", "KAE"]:
        v = []
        for sec in [
            "G4_delta_l",
            "G5_n_obs",
            "G3_sparsity",
            "G2_noise",
            "G19_misspec",
            "G20_joint_sparsity",
        ]:
            v += [
                r[meth]["mean"]
                for r in json.loads((bd / f"{sec}.json").read_text())["rows"]
                if meth in r
            ]
        base_da6[meth] = float(np.mean(v))
    ref_da = [da6(f"ctkae_H8_d256_s{i}") for i in range(3)]
    b_da = [da6(f"ctkae_back001_s{i}") for i in range(3)]

    def f(v, d=2):
        return "--" if v is None else f"{v:.{d}f}"

    def pm(o, k):
        return f"{o[k][0]:.2f}$\\pm${o[k][1]:.2f}"

    rows = []
    for lab, key, da in [
        ("ACDM (official, r20)", "ACDM (official, r20)", base_da6["ACDM"]),
        (
            "U-Net$_{ut}$ (official, unrolled 8)",
            "U-Net_ut (official, unrolled 8)",
            base_da6["UNet"],
        ),
        (
            "U-Net$_{tn}$ (official, train noise 0.01)",
            "U-Net_tn (official, training noise 0.01)",
            None,
        ),
    ]:
        o = base[key]
        rows.append(
            f"{lab} & "
            + " & ".join(f(o[f"{c}_mse_all"]) for c in ["extrap", "interp", "longer"])
            + " & "
            + " & ".join(f(o[f"{c}_lsim"]) for c in ["extrap", "interp", "longer"])
            + f" & {f(da, 3)}\\\\"
        )
    rows.append(
        "Original KAE, final checkpoint (ep.\\ 199) & "
        + " & ".join(f(E0[c]["mse_all"]) for c in ["extrap", "interp", "longer"])
        + f" & -- & -- & -- & {da6('legacy_ep199'):.3f}\\\\"
    )
    rows.append(
        f"Original KAE, selected checkpoint (ep.\\ 40)$^\\ast$ & -- & -- & -- & -- & -- & -- & {base_da6['KAE']:.3f}\\\\ \\midrule"
    )
    for lab, o, das in [
        ("CT-KAE reference, single decode", s["ref"], ref_da),
        ("CT-KAE reference + mixture", m["ref"], ref_da),
        ("back001, single decode", s["back001"], b_da),
        ("\\textbf{back001 + mixture}", m["back001"], b_da),
    ]:
        rows.append(
            f"{lab} & "
            + " & ".join(pm(o, f"{c}_mse_all") for c in ["extrap", "interp", "longer"])
            + " & "
            + " & ".join(pm(o, f"{c}_lsim") for c in ["extrap", "interp", "longer"])
            + f" & {np.mean(das):.3f}$\\pm${np.std(das, ddof=1):.3f}\\\\"
        )
    out.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    for tag, extra in [("single", []), ("mix", ["--mix"])]:
        subprocess.run(
            [
                PY,
                str(SCR / "ablation_table.py"),
                *extra,
                "--out",
                str(HERE / f"abl_{tag}"),
            ],
            check=True,
            capture_output=True,
        )
        ablation(HERE / f"abl_{tag}.json", HERE / f"tab_abl_{tag}.tex")
    main_table(HERE / "abl_single.json", HERE / "abl_mix.json", HERE / "tab_main.tex")
    print("tables written")
