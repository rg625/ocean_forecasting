"""Is the U-Net's analysis error a real failure, or an invisible direction?

The audit shows the U-Net recovers every frame of the trajectory to ~1e-4 while being the
WORST method at t_0 itself.  The hypothesis is that 4D-Var has landed on an x_0 that
differs from the truth by a perturbation lying in the near-nullspace of the one-step map
F_theta -- a direction the cost function cannot see, because the cost only ever evaluates
frames at and after the first observation.

This script tests that directly, and against the analytic prediction.  Linearising KS
about zero, mode q = 2 pi k / L grows at q^2 - q^4, so over one stored step dt the mode is
multiplied by exp((q^2 - q^4) dt).  If the U-Net has learned that dissipation, the measured
contraction along the recovered error direction should match the analytic curve, and the
error's own spectrum should peak where the contraction is strongest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data_assimilation.ks.da_ks_experiments_3way import load_unet
from data_assimilation.ks.protocol import KSData, DT

OUT = Path("da_results_sda_paper")
L_DOMAIN = 22.0


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz = np.load(OUT / "unet_audit_fields.npz")
    data = KSData("data/ks/da_test.nc", dev)
    unet = load_unet(
        Path("model_outputs_ks/unet1d/rollout10_extended2/best_model.pth"), dev
    )

    truth = npz["truth"]  # [B, T, X] denormalised
    rec = npz["UNet__traj"]  # [B, T, X] denormalised
    B, T, X = truth.shape
    k = np.fft.rfftfreq(X, d=1.0 / X)
    q = 2 * np.pi * k / L_DOMAIN
    analytic = np.exp((q**2 - q**4) * DT)  # one-step linear KS multiplier

    # ---- normalised fields, because the network sees normalised inputs ----------
    mu, sd = float(data.mean), float(data.std)
    x0_true = torch.as_tensor((truth[:, 0] - mu) / sd, device=dev, dtype=torch.float32)
    x0_rec = torch.as_tensor((rec[:, 0] - mu) / sd, device=dev, dtype=torch.float32)
    delta = x0_rec - x0_true

    @torch.no_grad()
    def step(x):
        return unet(x.unsqueeze(1)).squeeze(1) if x.dim() == 2 else unet(x)

    with torch.no_grad():
        f_true, f_rec = step(x0_true), step(x0_rec)
    d_out = f_rec - f_true

    n_in = delta.norm(dim=-1)
    n_out = d_out.norm(dim=-1)
    ratio = (n_out / n_in).cpu().numpy()

    # ---- per-wavenumber contraction along the SAME perturbation -----------------
    Fi = np.abs(np.fft.rfft(delta.cpu().numpy(), axis=-1)).mean(0)
    Fo = np.abs(np.fft.rfft(d_out.cpu().numpy(), axis=-1)).mean(0)
    measured = Fo / np.maximum(Fi, 1e-12)

    # ---- control: a RANDOM perturbation of the same norm ------------------------
    g = torch.Generator(device=dev).manual_seed(0)
    r = torch.randn(delta.shape, device=dev, generator=g)
    r = r / r.norm(dim=-1, keepdim=True) * n_in[:, None]
    with torch.no_grad():
        d_rand = step(x0_true + r) - f_true
    ratio_rand = (d_rand.norm(dim=-1) / n_in).cpu().numpy()

    # ---- the decisive number: what the 4D-Var cost actually sees -----------------
    # observation loss evaluated at the RECOVERED x0 vs at the TRUE x0
    obs_frames = npz["obs_frames"].tolist()

    def rollout_loss(x0):
        x, tot = x0, 0.0
        for n in range(1, max(obs_frames) + 1):
            x = step(x)
            if n in obs_frames:
                tgt = torch.as_tensor(
                    (truth[:, n] - mu) / sd, device=dev, dtype=torch.float32
                )
                tot += float(((x - tgt) ** 2).mean())
        return tot / len(obs_frames)

    with torch.no_grad():
        J_rec, J_true = rollout_loss(x0_rec), rollout_loss(x0_true)

    rep = {
        "perturbation": {
            "definition": "delta = x0_recovered - x0_true, in normalised units",
            "rel_norm_at_t0": float((n_in / x0_true.norm(dim=-1)).mean()),
            "one_step_contraction_mean": float(ratio.mean()),
            "one_step_contraction_std": float(ratio.std(ddof=1)),
            "random_direction_contraction_mean": float(ratio_rand.mean()),
            "selectivity": float(ratio_rand.mean() / ratio.mean()),
        },
        "cost_function_blindness": {
            "J_at_recovered_x0": J_rec,
            "J_at_true_x0": J_true,
            "ratio": J_rec / J_true,
            "note": (
                "The 4D-Var objective is evaluated only at the observation frames. "
                "If J(x0_recovered) <= J(x0_true) then the optimiser has found a "
                "point the cost prefers to the truth, and no amount of extra "
                "optimisation would move it towards the truth: the analysis error "
                "is not an optimisation failure but an identifiability failure."
            ),
            "optimiser_prefers_recovered_to_truth": bool(J_rec <= J_true),
        },
        "spectrum": {
            "k": k.tolist(),
            "analytic_one_step_multiplier_exp((q^2-q^4)dt)": analytic.tolist(),
            "measured_one_step_multiplier": measured.tolist(),
            "error_amplitude_at_t0": Fi.tolist(),
        },
    }
    (OUT / "unet_nullspace.json").write_text(json.dumps(rep, indent=2))

    print(
        f"perturbation delta = x0_recovered - x0_true, |delta|/|x0| = "
        f"{rep['perturbation']['rel_norm_at_t0']:.4f}\n"
    )
    print(
        f"one application of F_theta contracts it by "
        f"{ratio.mean():.4f} +/- {ratio.std(ddof=1):.4f}"
    )
    print(f"a RANDOM direction of the same norm contracts by {ratio_rand.mean():.4f}")
    print(
        f"  -> the recovered error is {rep['perturbation']['selectivity']:.1f}x more "
        f"strongly damped than a generic direction\n"
    )
    print(f"4D-Var objective at the RECOVERED x0 : {J_rec:.4e}")
    print(f"4D-Var objective at the TRUE      x0 : {J_true:.4e}")
    print(
        f"  -> the optimiser prefers its own answer to the truth: "
        f"{rep['cost_function_blindness']['optimiser_prefers_recovered_to_truth']}"
        f"  (ratio {J_rec / J_true:.3f})\n"
    )
    print(f"{'k':>3s} {'|err(t0)|':>11s} {'measured mult':>14s} {'analytic mult':>14s}")
    for i in range(0, len(k)):
        if i % 2 == 0 or 6 <= i <= 10:
            print(f"{k[i]:3.0f} {Fi[i]:11.4f} {measured[i]:14.4f} {analytic[i]:14.4f}")


if __name__ == "__main__":
    main()
