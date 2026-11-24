"""Why does the KAE forecast lose skill so early?  Look at the generator itself.

The continuous KAE propagates z(tau) = exp(K tau) z0 with K = skew(W) + sym(D).  The
eigenvalues of K are therefore the whole story of its free-running behaviour: their real
parts set growth or decay, their imaginary parts set oscillation, and the spread of the
real parts sets how quickly an arbitrary initial latent collapses onto the slowest modes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data_assimilation.ks.da_ks_experiments_3way import load_kae

OUT = Path("da_results_sda_paper")


def main():
    dev = torch.device("cpu")
    _, K, D = load_kae(
        Path("model_outputs_ks/continous_linear_128/rollout_10"), None, dev
    )
    K = K.detach().cpu().numpy().astype(np.float64)
    ev = np.linalg.eigvals(K)
    re, im = ev.real, ev.imag
    TL = json.loads((OUT / "lyapunov.json").read_text())["lyapunov_time"]

    order = np.argsort(-re)
    rep = {
        "latent_dim": int(D),
        "K_asymmetry": float(np.abs(K + K.T).max() / np.abs(K).max()),
        "eigenvalues": {
            "real_max": float(re.max()),
            "real_min": float(re.min()),
            "real_mean": float(re.mean()),
            "n_unstable_re_gt_0": int((re > 0).sum()),
            "n_marginal_abs_re_lt_1e-3": int((np.abs(re) < 1e-3).sum()),
            "imag_absmax": float(np.abs(im).max()),
        },
        "timescales": {
            "slowest_decay_time_1_over_re": (
                float(-1.0 / re[re < 0].max()) if (re < 0).any() else None
            ),
            "fastest_decay_time": float(-1.0 / re.min()) if (re < 0).any() else None,
        },
        "lyapunov_time": TL,
        # A finite-dimensional LINEAR autonomous system cannot have a chaotic attractor:
        # its growth rates are fixed by Re(eig K) and its orbits are sums of exponentials
        # times sinusoids. The largest real part is therefore the fastest error growth the
        # KAE can express, and it can be compared directly with the measured lambda_1 of
        # the true system.
        "growth_rate_comparison": {
            "kae_max_growth_rate_max_Re_eig_K": float(re.max()),
            "true_lambda_1": float(1.0 / TL),
            "ratio_true_over_kae": (
                float((1.0 / TL) / re.max()) if re.max() > 0 else None
            ),
        },
    }
    # how much of an arbitrary latent survives to a given horizon
    horizons = [1.0, 2.5, 5.0, 10.0, TL, 2 * TL, 5 * TL]
    rep["mode_survival"] = []
    for t in horizons:
        g = np.exp(re * t)
        rep["mode_survival"].append(
            {
                "t_units": float(t),
                "T_L": float(t / TL),
                "max_gain": float(g.max()),
                "median_gain": float(np.median(g)),
                "frac_modes_below_1pct": float((g < 1e-2).mean()),
            }
        )
    (OUT / "koopman_spectrum.json").write_text(json.dumps(rep, indent=2))

    print(
        f"latent dimension {D};  K is {rep['K_asymmetry']:.3f} away from skew-symmetric "
        f"(0 = purely oscillatory, no decay)\n"
    )
    print(
        f"eigenvalues of K: Re in [{re.min():.4f}, {re.max():.4f}], "
        f"|Im| <= {np.abs(im).max():.4f}"
    )
    print(f"  modes with Re > 0 (growing):        {(re > 0).sum():3d} / {D}")
    print(f"  modes with |Re| < 1e-3 (marginal):  {(np.abs(re) < 1e-3).sum():3d} / {D}")
    print(f"  modes with Re < 0 (decaying):       {(re < 0).sum():3d} / {D}")
    if (re < 0).any():
        print(
            f"\n  slowest decay time 1/|Re|  {-1 / re[re < 0].max():8.2f} t.u. "
            f"= {-1 / re[re < 0].max() / TL:.3f} T_L"
        )
        print(f"  fastest decay time         {-1 / re.min():8.2f} t.u.")
    print("\nfraction of latent modes decayed below 1% of their initial amplitude:")
    print(
        f"{'t.u.':>8s} {'T_L':>7s} {'max gain':>10s} {'median gain':>12s} {'frac < 1%':>10s}"
    )
    for r in rep["mode_survival"]:
        print(
            f"{r['t_units']:8.1f} {r['T_L']:7.2f} {r['max_gain']:10.4f} "
            f"{r['median_gain']:12.2e} {100 * r['frac_modes_below_1pct']:9.1f}%"
        )
    g = rep["growth_rate_comparison"]
    print(
        f"\nfastest error growth the KAE can express : Re(eig K)max = "
        f"{g['kae_max_growth_rate_max_Re_eig_K']:+.5f}"
    )
    print(
        f"measured growth rate of the true system  : lambda_1     = "
        f"{g['true_lambda_1']:+.5f}"
    )
    print(
        f"  -> the linear generator is {g['ratio_true_over_kae']:.1f}x too slow to "
        f"sustain the system's own error growth"
    )
    print("\ntop 8 eigenvalues by real part:")
    for i in order[:8]:
        print(
            f"   {re[i]:+.5f} {im[i]:+.5f}i   "
            f"{'GROWING' if re[i] > 0 else 'decay time %.1f t.u.' % (-1 / re[i])}"
        )


if __name__ == "__main__":
    main()
