"""Summarise pp3_proxy.py outputs and apply its pre-registered selection rule.

Per proxy and set: MSE and LSiM as ratio to the single decode (paired; SE over the 7 held-out
simulations, windows of one simulation averaged first), log10 power ratio per band, slope.

    python pp3_proxy_summary.py PX3_ref.json PX3_back001.json
"""

import json
import sys
from pathlib import Path

import numpy as np

TOL = 0.05


def per_sim(vals, ma):
    ma = np.round(np.array(ma), 4)
    return np.array([np.mean(np.array(vals)[ma == u]) for u in np.unique(ma)])


def main():
    runs = {Path(p).stem.split("_", 1)[-1]: json.load(open(p)) for p in sys.argv[1:]}
    names = list(next(iter(runs.values()))["sets"]["short"]["stats"])
    summ: dict = {n: {"ok": True, "mse_ratio": []} for n in names}
    for model, r in runs.items():
        for st, d in r["sets"].items():
            S, ma = d["stats"], d["ma"]
            s0 = S["single"]
            print(
                f"\n== {model} {st} (single: MSE {s0['mse']:.3f}, LSiM {s0['lsim']:.3f})"
            )
            print(
                f"  {'candidate':30s} {'MSE/single':>16s} {'dLSiM':>16s}  log_low log_mid log_high slope"
            )
            for n in names:
                c = S[n]
                rm = per_sim(c["mse_win"], ma) / per_sim(s0["mse_win"], ma)
                dl = per_sim(c["lsim_win"], ma) - per_sim(s0["lsim_win"], ma)
                se = lambda v: v.std(ddof=1) / np.sqrt(len(v))  # noqa: E731
                print(
                    f"  {n:30s} {rm.mean():7.3f}+-{se(rm):.3f} {dl.mean():+7.3f}+-{se(dl):.3f}"
                    f"  {c['log_low']:+.3f}  {c['log_mid']:+.3f}  {c['log_high']:+.3f}  {c['slope']:.2f}"
                )
                summ[n]["mse_ratio"].append(rm.mean())
                if (
                    dl.mean() > 0
                    or min(c["log_mid"] - s0["log_mid"], c["log_high"] - s0["log_high"])
                    < -TOL
                ):
                    summ[n]["ok"] = False
    print(
        "\n== selection (admissible: dLSiM <= 0 and band log ratio >= single - 0.05 everywhere)"
    )
    for n in sorted(names, key=lambda n: np.mean(summ[n]["mse_ratio"])):
        print(
            f"  {n:30s} mean MSE/single {np.mean(summ[n]['mse_ratio']):.3f}  {'ADMISSIBLE' if summ[n]['ok'] else '-'}"
        )


if __name__ == "__main__":
    main()
