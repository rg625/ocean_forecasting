"""Spectral audit of a trained continuous-time Koopman generator (CPU only).

Reads only the state_dict of a checkpoint; never writes next to it.

Reports, for A = skew(W) + sym(D) (ContinuousKoopmanOperator, LINEAR mode):
  * whether the LoRA hypernetwork that injects the condition is still at its
    zero initialisation (dead conditioning path),
  * eigenvalues of A, spectral abscissa, frequencies,
  * numerical abscissa (max eig of sym part) -> bound on transient growth,
  * departure from normality and eigenvector conditioning,
  * how far the RK4 one-step map P4(dt A) used in training is from exp(dt A),
  * ||P4^k|| vs ||exp(k dt A)|| over a long horizon.

Usage:
    python operator_spectrum.py CKPT [--dt 0.1] [--json OUT.json]
"""

import argparse
import json
import math

import numpy as np
import torch


def rk4_poly(M: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(M.shape[0], dtype=M.dtype)
    M2 = M @ M
    M3 = M2 @ M
    return eye + M + M2 / 2 + M3 / 6 + M3 @ M / 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    pre = "koopman_operator.dynamics."
    out = {"ckpt": args.ckpt}

    # --- conditioning path -------------------------------------------------
    hyp = {k: v for k, v in sd.items() if k.startswith(pre + "hypnet")}
    if hyp:
        last_w = [k for k in hyp if k.endswith("weight")][-1]
        last_b = [k for k in hyp if k.endswith("bias")][-1]
        out["hypnet_last_weight_absmax"] = float(hyp[last_w].abs().max())
        out["hypnet_last_bias_absmax"] = float(hyp[last_b].abs().max())
        out["hypnet_first_weight_absmax"] = float(
            hyp[[k for k in hyp if k.endswith("weight")][0]].abs().max()
        )

    W = sd[pre + "W_param.weight"].double()
    D = sd[pre + "D_param.weight"].double()
    A = 0.5 * (W - W.T) + 0.5 * (D + D.T)
    n = A.shape[0]
    dt = args.dt

    ev, V = torch.linalg.eig(A)
    re, im = ev.real.numpy(), ev.imag.numpy()
    out["latent_dim"] = n
    out["spectral_abscissa"] = float(re.max())
    out["n_unstable_eigs"] = int((re > 0).sum())
    out["real_part_quantiles"] = np.quantile(re, [0, 0.1, 0.5, 0.9, 1]).tolist()
    out["max_abs_imag"] = float(np.abs(im).max())
    out["max_|lambda|*dt"] = float(np.abs(ev.numpy()).max() * dt)
    out["sym_part_eig_max"] = float(torch.linalg.eigvalsh(0.5 * (A + A.T)).max())
    out["sym_part_eig_min"] = float(torch.linalg.eigvalsh(0.5 * (A + A.T)).min())
    out["fro_skew"] = float(torch.linalg.norm(0.5 * (W - W.T)))
    out["fro_sym"] = float(torch.linalg.norm(0.5 * (D + D.T)))
    fro2 = float(torch.linalg.norm(A) ** 2)
    out["henrici_departure_rel"] = math.sqrt(
        max(fro2 - float((ev.abs() ** 2).sum()), 0.0)
    ) / math.sqrt(fro2)
    out["eigvec_cond"] = float(torch.linalg.cond(V))

    # --- RK4 vs exponential --------------------------------------------------
    P = rk4_poly(A * dt)
    E = torch.matrix_exp(A * dt)
    out["||P4(dtA)-exp(dtA)||_2"] = float(torch.linalg.matrix_norm(P - E, 2))
    out["||exp(dtA)||_2"] = float(torch.linalg.matrix_norm(E, 2))
    mu_p = torch.linalg.eigvals(P).abs().numpy()
    mu_e = np.exp(re * dt)
    out["spectral_radius_P4"] = float(mu_p.max())
    out["spectral_radius_exp"] = float(mu_e.max())
    # per-eigenvalue amplification of RK4 vs exact
    z = ev.numpy() * dt
    R = 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    ratio = np.abs(R) / np.abs(np.exp(z))
    out["rk4_amp_ratio_min_max"] = [float(ratio.min()), float(ratio.max())]
    # implied extra damping per 250 steps from RK4 on the fastest mode
    out["rk4_amp_ratio_pow250_min"] = float(ratio.min() ** 250)

    growth = {}
    Pk = torch.eye(n, dtype=A.dtype)
    for k in range(1, 1001):
        Pk = P @ Pk
        if k in (1, 2, 5, 10, 20, 50, 100, 250, 500, 1000):
            growth[k] = {
                "rk4": float(torch.linalg.matrix_norm(Pk, 2)),
                "exp": float(torch.linalg.matrix_norm(torch.matrix_exp(A * dt * k), 2)),
            }
    out["norm_of_propagator_vs_steps"] = growth
    # frequencies resolved: |Im lambda| in rad per time unit, period in steps
    pos = im[im > 1e-8]
    if pos.size:
        out["periods_in_steps_min_median_max"] = [
            float(2 * np.pi / pos.max() / dt),
            float(2 * np.pi / np.median(pos) / dt),
            float(2 * np.pi / pos.min() / dt),
        ]
    out["eigs_sorted_by_real"] = sorted(
        [[float(a), float(b)] for a, b in zip(re, im)], key=lambda t: -t[0]
    )[:12]

    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
