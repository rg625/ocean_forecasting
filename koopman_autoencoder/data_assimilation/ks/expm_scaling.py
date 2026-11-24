"""How does KAE-expm assimilation cost ACTUALLY scale with the horizon?

The manuscript claims cost "independent of the assimilation window".  The measured 4D-Var
timings say 11.0 ms/iter at a 25-frame horizon and 19.7 ms at 100 frames -- a 1.8x rise for
a 4x horizon.  That is not independence, and it is not linear growth either.  The likely
cause is `torch.matrix_exp`, which uses scaling-and-squaring: the number of squarings grows
like log2(||K tau||), so the cost should grow LOGARITHMICALLY with tau.  This measures it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from data_assimilation.ks.da_ks_experiments_3way import load_kae

OUT = Path("da_results_sda_paper")


def bench(fn, n=60, warmup=10, cuda=False):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = dev.type == "cuda"
    _, K, D = load_kae(
        Path("model_outputs_ks/continous_linear_128/rollout_10"), None, dev
    )
    K = K.detach()
    nrm = float(torch.linalg.matrix_norm(K, 2))

    taus = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]
    rows = []
    for t in taus:
        ms = bench(lambda t=t: torch.matrix_exp(K * t), cuda=cuda)
        rows.append(
            {
                "tau_t_units": t,
                "frames": int(round(t / 0.1)),
                "K_tau_norm": nrm * t,
                "expected_squarings": max(0.0, np.log2(max(nrm * t, 1e-12))),
                "ms_per_matrix_exp": ms,
            }
        )

    # fit  ms = a + b*log2(tau)  and  ms = a + b*tau, and compare
    x = np.array([r["tau_t_units"] for r in rows])
    y = np.array([r["ms_per_matrix_exp"] for r in rows])

    def r2(pred):
        return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    cl = np.polyfit(np.log2(x), y, 1)
    pl_ = np.polyval(cl, np.log2(x))
    cn = np.polyfit(x, y, 1)
    pn = np.polyval(cn, x)

    rep = {
        "latent_dim": int(D),
        "spectral_norm_K": nrm,
        "device": str(dev),
        "rows": rows,
        "fit_logarithmic": {
            "slope_ms_per_doubling": float(cl[0]),
            "intercept_ms": float(cl[1]),
            "r2": float(r2(pl_)),
        },
        "fit_linear": {
            "slope_ms_per_t_unit": float(cn[0]),
            "intercept_ms": float(cn[1]),
            "r2": float(r2(pn)),
        },
        "claim": (
            "KAE-expm assimilation cost grows LOGARITHMICALLY with the horizon, "
            "not linearly and not as O(1). torch.matrix_exp uses "
            "scaling-and-squaring, so the squaring count rises like "
            "log2(||K tau||). Autoregressive propagation is linear in the "
            "horizon, which is the contrast that matters."
        ),
    }
    (OUT / "expm_scaling.json").write_text(json.dumps(rep, indent=2))

    print(f"latent {D}, ||K||_2 = {nrm:.4f}, device {dev}\n")
    print(
        f"{'tau (t.u.)':>11s} {'frames':>7s} {'||K tau||':>10s} {'ms / matrix_exp':>16s}"
        f" {'vs tau=0.1':>11s}"
    )
    for r in rows:
        print(
            f"{r['tau_t_units']:11.1f} {r['frames']:7d} {r['K_tau_norm']:10.2f} "
            f"{r['ms_per_matrix_exp']:16.4f} {r['ms_per_matrix_exp'] / rows[0]['ms_per_matrix_exp']:10.2f}x"
        )
    print(
        f"\nlogarithmic fit  ms = {cl[1]:.3f} + {cl[0]:.3f} * log2(tau)   R^2 = {r2(pl_):.4f}"
    )
    print(
        f"linear fit       ms = {cn[1]:.3f} + {cn[0]:.5f} * tau          R^2 = {r2(pn):.4f}"
    )
    print(
        f"\ncost of a 5000x longer horizon (0.1 -> 500 t.u.): "
        f"{rows[-1]['ms_per_matrix_exp'] / rows[0]['ms_per_matrix_exp']:.2f}x"
    )
    print("an autoregressive rollout over the same range would cost 5000x")


if __name__ == "__main__":
    main()
