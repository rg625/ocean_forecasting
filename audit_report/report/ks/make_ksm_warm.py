"""The KS comparison with a warm start for every gradient-based method, delta_f = 9 DA and the
free-running forecast, one protocol per table (ctkae_ks.tex, matched-training section).

DA (exp_geometry at its default pin delta_f = 9: C2 C3 NL JS on the campaign's seeded
problems; mean +- SE over problems):
  SDA                      stored (da_results_geometry_df9); the sampler has no first guess
  KAE-512 (stored, warm)   the main repository's kae512_tuned.json: d_z = 512, init 'obs'
  U-Net (warm)             results/ks_matched/unet_warm: nearest observation as first guess
  KAE ctrl, CT-KAE ...     results/ks_matched/<run>/geometry_df9, init 'obs'
Rows are paired by tag and frames (asserted). Partial runs show their finished rows.

    python make_ksm_warm.py            # writes tab_ksm_warm9.tex, tab_ksm_fc2.tex, prints both
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RES = HERE.parents[1] / "results/ks_matched"
MAIN = Path("/home/rg625/mnt/ocean_forecasting/koopman_autoencoder")
CAMP = MAIN / "da_results_geometry_df9"
SECS = ["C2_delta_l", "C3_n_obs", "noise_law", "joint_sparsity"]
RUNS = [
    ("kae_128_s0", "KAE ctrl"),
    ("ct_ref_128_s0", "ref 128"),
    ("ct_back001_128_s0", "back001 128"),
    ("ct_ref_256_s0", "ref 256"),
    ("ct_back001_256_s0", "back001 256"),
    ("ct_ref_signed_128_s0", "ref signed 128"),
]
COLS = ["SDA", "KAE-512 (stored)", "U-Net"] + [lab for _, lab in RUNS]


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def ms(v):
    if not isinstance(v, dict) or v.get("mean") is None or not np.isfinite(v["mean"]):
        return None
    return v


def kae512_rows():
    """kae512_tuned sweeps keyed by (section, frames, noise, frac)"""
    out = {}
    for r in jl(CAMP / "kae512_tuned.json")["sweeps"]:
        if r.get("delta_f") != 9:
            continue
        key = (
            r["section"],
            tuple(r["frames"]),
            r.get("noise_dist"),
            r.get("noise_std"),
            r.get("obs_frac"),
            r["iters"],
        )
        out[key] = r
    return out


def key_of(sec, r):
    nd = ns = fr = None
    if sec == "noise_law":
        nd, ns = r["tag"].split("_")[1], float(r["tag"].split("_")[2])
    if sec == "joint_sparsity":
        fr = float(r["value"][0])
    return (sec, tuple(r["frames"]), nd, ns, fr, r["iters"])


def unet_file(sec):
    for sub in ("geometry_df9_a", "geometry_df9_b", "geometry_df9_c"):
        d = jl(RES / "unet_warm" / sub / f"{sec}.json")
        if d is not None:
            return d
    return None


def table9():
    k512 = kae512_rows()
    out = []
    for sec in SECS:
        camp = jl(CAMP / f"{sec}.json")
        files = {lab: jl(RES / n / f"geometry_df9/{sec}.json") for n, lab in RUNS}
        files["U-Net"] = unet_file(sec)
        rows = []
        for r in camp["rows"]:
            if sec == "noise_law" and not r["tag"].startswith("NL_gaussian"):
                continue
            vals = {"SDA": ms(r.get("SDA"))}
            h = k512.get(key_of(sec, r))
            vals["KAE-512 (stored)"] = h
            for lab, d in files.items():
                hit = (
                    None
                    if d is None
                    else next((x for x in d["rows"] if x["tag"] == r["tag"]), None)
                )
                if hit is not None:
                    assert list(hit["frames"]) == list(r["frames"]), (sec, r["tag"])
                key = "UNet" if lab == "U-Net" else "KAE-expm"
                vals[lab] = None if hit is None else ms(hit.get(key))
            rows.append((r, vals))
        out.append((sec, rows))
    return out


def label(sec, r):
    if sec == "C2_delta_l":
        return f"dl={r['taus'][-1]:g}" + (
            " (8000 it.)" if r["tag"].endswith("anchor") else ""
        )
    if sec == "C3_n_obs":
        return f"N={len(r['taus'])}"
    if sec == "noise_law":
        return f"noise {float(r['value']):g}"
    return f"{int(100 * r['value'][0])}%, N={len(r['taus'])}"


HEAD = {
    "C2_delta_l": "last observation delta_l (t.u.); delta_f = 0.9, N = 5; 1000 it., 24 problems",
    "C3_n_obs": "observation times N; delta_f = 0.9, delta_l = 2.5",
    "noise_law": "Gaussian noise std; N = 5",
    "joint_sparsity": "coverage, N",
}


def fmt(v, tex=False):
    if v is None:
        return "--"
    if tex:
        return f"{v['mean']:.4f}"
    return f"{v['mean']:.4f}±{v['sem']:.4f}"


def write9(tab):
    md, tex = [], []
    md.append("| setting | " + " | ".join(COLS) + " |")
    md.append("|---" * (len(COLS) + 1) + "|")
    for sec, rows in tab:
        md.append(f"| **{HEAD[sec]}** |" + " |" * len(COLS))
        h = (
            HEAD[sec]
            .replace("delta_l", r"$\delta_l$")
            .replace("delta_f", r"$\delta_f$")
        )
        tex.append(r"\multicolumn{" + str(len(COLS) + 1) + r"}{l}{\emph{" + h + r"}}\\")
        for r, v in rows:
            if sec == "joint_sparsity" and len(r["taus"]) not in (3, 5, 9):
                continue
            ok = [v[c]["mean"] for c in COLS if v[c] is not None]
            best = min(ok) if ok else None
            cells_md, cells_tex = [], []
            for c in COLS:
                s, t = fmt(v[c]), fmt(v[c], True)
                if v[c] is not None and v[c]["mean"] == best:
                    s, t = f"**{s}**", r"\textbf{" + t + "}"
                cells_md.append(s)
                cells_tex.append(t)
            md.append(f"| {label(sec, r)} | " + " | ".join(cells_md) + " |")
            tex.append(
                rf"\quad {label(sec, r).replace('%', chr(92) + '%').replace('dl=', '')} & "
                + " & ".join(cells_tex)
                + r"\\"
            )
    (HERE / "tab_ksm_warm9.tex").write_text("\n".join(tex) + "\n")
    return "\n".join(md)


def forecast():
    fc = np.load(MAIN / "da_results_long/FC_forecast.npz")
    TL = float(fc["T_L"])
    pts = [0.1, 0.25, 0.5, 1.0]
    rows = []

    def row(lab, taus, mu, se, rt):
        at = [float(np.interp(p * TL, taus, mu)) for p in pts]
        sa = [float(np.interp(p * TL, taus, se)) for p in pts]
        return lab, at, sa, rt

    rows.append(
        row(
            "U-Net (paper ckpt)",
            fc["taus"],
            fc["UNet__rel_mean"],
            fc["UNet__rel_sem"],
            None,
        )
    )
    rows.append(
        row(
            "Paper KAE (paper ckpt)",
            fc["taus"],
            fc["KAE-expm__rel_mean"],
            fc["KAE-expm__rel_sem"],
            float(fc["kae_roundtrip"]),
        )
    )
    for n, lab in RUNS:
        p = RES / n / "fc_matched.npz"
        if not p.exists():
            continue
        d = np.load(p)
        js = jl(RES / n / "fc_matched.json")["models"][n]["single"]
        rows.append(
            row(
                lab,
                d["taus"],
                d[f"{n}__single__rel_mean"],
                d[f"{n}__single__rel_sem"],
                js["roundtrip"],
            )
        )
    md = ["| model | round trip | " + " | ".join(f"{p:g} T_L" for p in pts) + " |"]
    md.append("|---" * (len(pts) + 2) + "|")
    tex = []
    best = [min(r[1][i] for r in rows) for i in range(len(pts))]
    for lab, at, sa, rt in rows:
        cm = [
            (f"**{a:.4f}±{s:.4f}**" if a == b else f"{a:.4f}±{s:.4f}")
            for a, s, b in zip(at, sa, best)
        ]
        ct = [
            (r"\textbf{" + f"{a:.4f}" + "}" if a == b else f"{a:.4f}")
            for a, b in zip(at, best)
        ]
        rts = "--" if rt is None else f"{rt:.4f}"
        md.append(f"| {lab} | {rts} | " + " | ".join(cm) + " |")
        tex.append(f"{lab} & {rts} & " + " & ".join(ct) + r"\\")
    (HERE / "tab_ksm_fc2.tex").write_text("\n".join(tex) + "\n")
    return "\n".join(md)


if __name__ == "__main__":
    print(write9(table9()))
    print()
    print(forecast())
