"""Post-processing tables for ctkae_candidates.tex (\\S6), from the archived lab results.

    python make_pp_tables.py   -> tab_pp_delta.tex, tab_pp_ens.tex

Source: audit_report/results/postproc/pp_results.json (scripts/postproc/pp_analyze.py; all-frames
MSE x1e-3 per variant, case and checkpoint or seed ensemble). Paired deltas are against the
calibrated mixture readout of the same checkpoint (mean over 3 seeds +- SE).
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
R = json.loads((HERE.parent / "results/postproc/pp_results.json").read_text())
CASES = ["extrap", "interp", "longer"]
MODELS = ["ref", "back001"]
LABELS = {
    "mix": "mixture (current protocol)",
    "single": "single decode",
    "mixlead": r"mixture, lead-dependent $\sigma(\tau)$",
    "mix+sself": "mixture + scalar shrink to own mean",
    "mix+wself": "mixture + Wiener to own mean",
    "mix+sclim": "mixture + scalar shrink to climatology",
    "mix+wclim": "mixture + Wiener to climatology",
    "mixlead+sself": r"$\sigma(\tau)$ + scalar shrink to own mean",
    "mixlead+wself": r"$\sigma(\tau)$ + Wiener to own mean",
    "mixlead+wclim": r"$\sigma(\tau)$ + Wiener to climatology",
    "single+wself": "single decode + Wiener to own mean",
}


def v(var, case, key):
    return R[f"{var}|{case}|{key}"]


def cell(d, se):
    s = f"{d:+.2f}$\\pm${se:.2f}"
    if abs(d) > 2 * se and abs(d) >= 0.005:
        s = f"\\textbf{{{s}}}" if d < 0 else s + "$^\\dagger$"
    return s


def delta_table():
    rows = []
    for var in LABELS:
        cells = []
        for t in MODELS:
            for case in CASES:
                a = np.array([v(var, case, f"{t}_s{s}") for s in range(3)])
                if var == "mix":
                    cells.append(f"{a.mean():.2f}")
                    continue
                b = np.array([v("mix", case, f"{t}_s{s}") for s in range(3)])
                d = a - b
                cells.append(cell(d.mean(), d.std(ddof=1) / np.sqrt(3)))
        tail = r"\\ \midrule" if var == "mix" else r"\\"
        rows.append(f"{LABELS[var]} & " + " & ".join(cells) + tail)
    (HERE / "tab_pp_delta.tex").write_text("\n".join(rows) + "\n")


def ens_table():
    rows = []
    for var in ["single", "mix", "mix+sclim", "mix+wclim"]:
        cells = []
        for key in ["ref_ens", "back001_ens", "both_ens"]:
            cells += [f"{v(var, c, key):.2f}" for c in CASES]
        rows.append(f"{LABELS[var]} & " + " & ".join(cells) + r"\\")
    (HERE / "tab_pp_ens.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    delta_table()
    ens_table()
    print("pp tables written")
