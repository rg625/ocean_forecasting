"""Build the CT-KAE ablation table from finished run_ablation.sh jobs (partial results fine).

Per run: Tra_ext / Tra_int / Tra_long MSE on all frames (ACDM-paper convention, x1e-3) and
LSiM (x1e-1) from test_score.json; DA = mean analysis error over every point of the four df9
sweeps of the paper's Table 11 (window length, number of observations, sensor coverage,
noise), from model_outputs_ctkae/da_abl. Per row: mean +- s.d. over seeds, and the paired
change vs the reference run of the same seed (mean over seeds).

    python ablation_table.py [--out audit_report/results/ablation_table]
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/rg625/mnt/ocean_forecasting_impl")
OUT = ROOT / "koopman_autoencoder/model_outputs_ctkae"
STATE = OUT / "ablation_state"
SWEEPS = ("G4_delta_l", "G5_n_obs", "G3_sparsity", "G2_noise")
LABELS = {
    "ref": "Full model (2 -> 8, latent 256, block generator, FiLM Ma)",
    "nocond": "- Mach conditioning (encoder, decoder, generator)",
    "dense": "Generator: dense, unconstrained",
    "hurwitz": "Generator: Hurwitz (J - R)",
    "d128": "Latent 128",
    "d512": "Latent 512",
    "frames1": "1 input frame (no history)",
    "onestep": "One-step training (2 -> 1)",
    "mu": "+ conditional fixed point mu(Ma)",
    "norec": "- reconstruction loss",
    "nopred": "- prediction loss",
    "nolat": "- forward latent consistency",
    "noamp": "- amplitude loss",
    "back": "+ backward latent consistency (0.1)",
    "cos": "+ directional cosine (0.1)",
    "energy": "+ latent energy (0.1)",
    "sobt": "+ temporal Sobolev (0.1)",
    "sobx": "+ spatial Sobolev (0.1)",
    "spec": "+ spectral loss (0.1)",
    "cosw": "Prediction weighting: cosine over leads (original KAE)",
}
GROUPS = [
    ("Reference", ["ref"]),
    (
        "Architecture and training",
        ["nocond", "dense", "hurwitz", "d128", "d512", "frames1", "onestep", "mu"],
    ),
    ("Loss terms used (removed)", ["norec", "nopred", "nolat", "noamp"]),
    (
        "Loss terms of the original KAE (added)",
        ["back", "cos", "energy", "sobt", "sobx", "spec", "cosw"],
    ),
]
KEYS = [
    ("extrap_mse_all", "Tra_ext MSE"),
    ("interp_mse_all", "Tra_int MSE"),
    ("longer_mse_all", "Tra_long MSE"),
    ("extrap_lsim", "Tra_ext LSiM"),
    ("interp_lsim", "Tra_int LSiM"),
    ("longer_lsim", "Tra_long LSiM"),
    ("da", "DA df9"),
]


def da_mean(run: Path):
    d = OUT / "da_abl" / f"ctkae_{run.name}" / "df9"
    vals = []
    for s in SWEEPS:
        p = d / f"{s}.json"
        if not p.exists():
            return None
        vals += [
            r["KAE"]["mean"] for r in json.loads(p.read_text())["rows"] if "KAE" in r
        ]
    return float(np.mean(vals)) if vals else None


def collect():
    rows: dict = {}
    for f in sorted(STATE.glob("*.done")):
        key = f.stem
        tag, seed = key.rsplit("_s", 1)
        run = Path(f.read_text().strip())
        run = run if run.is_absolute() else OUT.parent / run
        ts = json.loads((run / "test_score.json").read_text())
        rec = {k: ts.get(k) for k, _ in KEYS if k != "da"}
        rec["da"] = da_mean(run)
        rows.setdefault(tag, {})[int(seed)] = rec
    return rows


def fmt(v, sd=None, digits=2):
    if v is None:
        return "-"
    return f"{v:.{digits}f}" + ("" if sd is None else f" ± {sd:.{digits}f}")


def summarise(rows):
    ref = rows.get("ref", {})
    out = {}
    for tag, seeds in rows.items():
        o = {"n": len(seeds), "seeds": sorted(seeds)}
        for k, _ in KEYS:
            v = [seeds[s][k] for s in seeds if seeds[s][k] is not None]
            o[k] = (
                (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None)
                if v
                else (None, None)
            )
            d = [
                seeds[s][k] - ref[s][k]
                for s in seeds
                if s in ref and seeds[s][k] is not None and ref[s][k] is not None
            ]
            o[f"d_{k}"] = float(np.mean(d)) if d and tag != "ref" else None
        out[tag] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "audit_report/results/ablation_table"))
    a = ap.parse_args()
    s = summarise(collect())
    dig = {"da": 3}
    md = [
        "| Variant | seeds | " + " | ".join(n for _, n in KEYS) + " |",
        "|---|---|" + "---|" * len(KEYS),
    ]
    tex = []
    sens = sorted(t for t in s if t.startswith("S_"))
    for title, tags in GROUPS + [("Weight sensitivity (seed 0)", sens)]:
        present = [t for t in tags if t in s]
        if not present:
            continue
        md.append(f"| **{title}** |" + " |" * (len(KEYS) + 1))
        tex.append(f"\\multicolumn{{{len(KEYS) + 2}}}{{l}}{{\\emph{{{title}}}}}\\\\")
        for t in present:
            o = s[t]
            cells = []
            for k, _ in KEYS:
                m, sd = o[k]
                c = fmt(m, sd, dig.get(k, 2))
                if o.get(f"d_{k}") is not None:
                    c += f" ({o[f'd_{k}']:+.{dig.get(k, 2)}f})"
                cells.append(c)
            lab = LABELS.get(t, t.replace("S_", "weight: "))
            md.append(f"| {lab} | {o['n']} | " + " | ".join(cells) + " |")
            tex.append(
                lab.replace("->", "$\\to$").replace("mu", "$\\mu$")
                + f" & {o['n']} & "
                + " & ".join(c.replace("±", "$\\pm$") for c in cells)
                + "\\\\"
            )
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    Path(a.out + ".tex").write_text("\n".join(tex) + "\n")
    Path(a.out + ".json").write_text(json.dumps(s, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
