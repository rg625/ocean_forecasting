"""Score pp2_apply.py outputs on test with the benchmark metric (all-frames MSE x1e-3).

Variants: mix (enc:cal), mirror (enc:cal:mirror), each optionally with scalar climatology
shrinkage (per lead and channel) from
  tr   train-calibrated gains of round 1 (res/pp/<key>.npz, fitted for the mixture forecast)
  px   proxy held-out gains (PX_<model>.json, tag enc|s0.06 or mirror|s0.06; leads 1-58 pooled
       over the short and long sets, later leads from the long set)
Per model: mean over seeds +- SE of the paired change vs mix, and the 3-seed average;
plus the 6-checkpoint average.

    python pp2_score.py [--lsim VARIANT ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).parent
sys.path.insert(0, str(D.parent / "pp"))
import pp_analyze as P1  # noqa: E402

CASES = ["extrap", "interp", "longer"]
H, W = 64, 128


def smooth(a, n=9):
    k = np.ones(n) / n
    ap = np.pad(a, ((n // 2, n // 2), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(ap[:, c], k, mode="valid") for c in range(a.shape[1])], 1
    )


def proxy_gain(model, tag):
    r = json.load(open(D / f"PX_{model}.json"))["sets"]
    s, lo = r["short"]["stats"][tag], r["long"]["stats"][tag]
    num, den = np.array(lo["num"]), np.array(lo["den"])
    num[:58] += np.array(s["num"])
    den[:58] += np.array(s["den"])
    g = np.where(den > 0, num / np.maximum(den, 1e-12), 1.0)
    return np.clip(smooth(g), 0, 1)  # [238, C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lsim", nargs="*", default=[])
    a = ap.parse_args()
    keys = {t: [f"{t}_s{s}" for s in range(3)] for t in ["ref", "back001"]}
    fields = {}  # (variant, case, key) -> [S, K, C, H, W] float32
    for t, ks in keys.items():
        g_px = {
            "mix": proxy_gain(t, "enc|s0.06"),
            "mirror": proxy_gain(t, "mirror|s0.06"),
        }
        for k in ks:
            d = np.load(D / f"apply_{k}.npz")
            ck = P1.Ckpt(k)
            g_tr = ck.gains("clim", np.full(238, ck.icar), per_band=False)[
                ..., 0
            ]  # [K, C]
            for case in CASES:
                clim = ck.d[f"{case}_clim"][:, None]  # [S, 1, C, H, W]
                for v, src in [("mix", "enc:cal"), ("mirror", "enc:cal:mirror")]:
                    x = d[f"{case}|{src}"].astype(np.float32)
                    K = x.shape[1]
                    mask = (np.abs(x).sum((0, 1, 2)) > 0).astype(np.float32)
                    fields[(v, case, k)] = x
                    for gk, g in [("tr", g_tr), ("px", g_px[v])]:
                        gg = g[:K][None, :, :, None, None]
                        fields[(f"{v}+{gk}", case, k)] = (clim + gg * (x - clim)) * mask
    variants = sorted(
        {v for v, _, _ in fields}, key=lambda s: (s.split("+")[0] != "mix", s)
    )
    res = {}
    for (v, case, k), x in fields.items():
        res[(v, case, k)] = P1.mse_all(case, x)
    for v in variants:
        for case in CASES:
            for t, ks in keys.items():
                res[(v, case, f"{t}_ens")] = P1.mse_all(
                    case, np.mean([fields[(v, case, k)] for k in ks], 0)
                )
            res[(v, case, "all6")] = P1.mse_all(
                case,
                np.mean([fields[(v, case, k)] for ks in keys.values() for k in ks], 0),
            )
    json.dump(
        {"|".join(k): val for k, val in res.items()},
        open(D / "pp2_test_results.json", "w"),
        indent=0,
    )
    for t, ks in keys.items():
        print(f"\n== {t}: mean over seeds [paired d vs mix +- SE] | 3-seed average")
        for v in variants:
            row = f"  {v:12s}"
            for case in CASES:
                vals = np.array([res[(v, case, k)] for k in ks])
                d = vals - np.array([res[("mix", case, k)] for k in ks])
                row += f" | {case[:3]} {vals.mean():6.2f} [{d.mean():+.2f}±{d.std(ddof=1) / np.sqrt(3):.2f}] avg {res[(v, case, f'{t}_ens')]:6.2f}"
            print(row)
    print("\n== average of all 6 checkpoints")
    for v in variants:
        print(
            f"  {v:12s}"
            + "".join(f" | {c[:3]} {res[(v, c, 'all6')]:6.2f}" for c in CASES)
        )
    if a.lsim:
        from lsim_score import lsim

        print("\n== LSiM x1e-1: mean over seeds | 3-seed average")
        for v in a.lsim:
            for t, ks in keys.items():
                row = f"  {v:12s} {t:8s}"
                for case in CASES:
                    per = [
                        lsim(P1.to_acdm(case, fields[(v, case, k)])[None], case)[0]
                        for k in ks
                    ]
                    ens = lsim(
                        P1.to_acdm(
                            case, np.mean([fields[(v, case, k)] for k in ks], 0)
                        )[None],
                        case,
                    )[0]
                    row += f" | {case[:3]} {np.mean(per):.2f} avg {ens:.2f}"
                print(row, flush=True)


if __name__ == "__main__":
    main()
