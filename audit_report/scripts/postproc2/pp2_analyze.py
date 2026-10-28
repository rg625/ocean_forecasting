"""Read pp2_lab.py outputs (proxy held-out sets) and report every variant.

Per variant: mean MSE (x1e-3, mean over windows and leads) on the short set split into the
ext-like (Ma < 0.6) and int-like held-out Mach numbers, and on the long set; then the effect of
scalar climatology shrinkage (per lead and channel, gain = num/den clipped to [0, 1] and
smoothed over 9 leads), fitted and evaluated on the same held-out set (in-sample: an upper bound).

    python pp2_analyze.py OUT1.json [OUT2.json ...]
"""

import json
import sys

import numpy as np

H, W = 64, 128


def smooth(a, n=9):
    k = np.ones(n) / n
    ap = np.pad(a, ((n // 2, n // 2), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(ap[:, c], k, mode="valid") for c in range(a.shape[1])], 1
    )


def shrink_delta(st, nwin):
    num, den = np.array(st["num"]), np.array(st["den"])
    g = np.clip(smooth(np.where(den > 0, num / np.maximum(den, 1e-12), 1.0)), 0, 1)
    d = ((g**2 - 1) * den - 2 * (g - 1) * num).sum(1)  # [K]
    return float((d / (nwin * H * W * 4) * 1e3).mean()), g


def report(path):
    r = json.load(open(path))
    print(f"\n######## {path}")
    for name, d in r["sets"].items():
        ma = np.array(d["ma"])
        groups: dict[str, np.ndarray] = {"all": np.ones(len(ma), bool)}
        if name == "short":
            groups = {"ext-like": ma < 0.6, "int-like": ma >= 0.6}
        print(
            f"\n== {name} ({len(ma)} windows); input-frame rec:",
            {k: round(v, 5) for k, v in r["input_rec"][name].items()},
        )
        hdr = "".join(f"{g:>10s}" for g in groups) + "   shrink(all, in-sample)"
        print(f"{'variant':24s}{hdr}")
        rows = []
        for tag, st in d["stats"].items():
            w = np.array(st["mse_win"])
            vals = [w[msk].mean() for msk in groups.values()]
            dsh, _ = shrink_delta(st, len(w))
            rows.append((tag, vals, dsh))
        base = next(v for t, v, _ in rows if t == "enc|s0")
        for tag, vals, dsh in rows:
            rel = "".join(f"{v:10.3f}" for v in vals)
            dd = "".join(f"{v - b:+8.3f}" for v, b in zip(vals, base))
            print(f"{tag:24s}{rel}  [{dd} ]  {dsh:+.3f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
