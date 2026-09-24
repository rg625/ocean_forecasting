"""Forecast from a PERTURBED true state: does the model damp the perturbation, and its energy?

No assimilation anywhere. The true state at t_0 is perturbed with Gaussian noise of a given
amplitude and rolled forward by each model. Two things are measured per frame: the error
against the truth, and the field energy relative to the truth. A model that contracts every
perturbation also contracts the signal, which shows up as an energy ratio below 1 within the
first few steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_assimilation.ks.protocol import DT, build_problem
from data_assimilation.ks.da_ks_experiments_3way import Bench, add_common_args

SIGMAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    ap.add_argument(
        "--json", type=Path, default=Path("da_results_geometry_df9/noise_forecast.json")
    )
    ap.add_argument("--n-problems", type=int, default=24)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--seed", type=int, default=77)
    a = ap.parse_args()
    b = Bench(a)
    dev = b.dev
    # the draw must leave room for the whole rollout, not just one frame
    prob = build_problem(
        b.data,
        name="NF",
        n_problems=a.n_problems,
        taus=np.array([a.frames - 1]) * DT,
        seed=a.seed,
    )
    sim = torch.as_tensor(prob.sim, device=dev)
    t0 = torch.as_tensor(prob.t0, device=dev)
    T = a.frames
    truth = torch.stack(
        [b.data.frames(sim, t0 + k) for k in range(T)]
    )  # [T,B,X] normalised
    taus = np.arange(T) * DT
    g = torch.Generator(device=dev).manual_seed(a.seed)
    rows = []
    curves = {}
    for name in ("UNet", "KAE-expm"):
        m = b.method(name)
        for s in SIGMAS:
            x0 = truth[0] + (
                s * torch.randn(truth[0].shape, device=dev, generator=g)
                if s > 0
                else 0.0
            )
            with torch.no_grad():
                if name == "KAE-expm":
                    from tensordict import TensorDict

                    z0 = b.kae.present_encoding(
                        TensorDict(
                            {"u": x0.unsqueeze(1).unsqueeze(-1)},
                            batch_size=[x0.shape[0], 1],
                        ),
                        cond_input=None,
                    )
                    pred = m.forecast(z0, taus)
                else:
                    pred = m.forecast(x0, taus)
            d = b.data.denorm(pred) - b.data.denorm(truth)
            rel = (
                (d.norm(dim=-1) / b.data.denorm(truth).norm(dim=-1))
                .mean(1)
                .cpu()
                .numpy()
            )
            energy = (
                (b.data.denorm(pred).norm(dim=-1) / b.data.denorm(truth).norm(dim=-1))
                .mean(1)
                .cpu()
                .numpy()
            )
            curves[(name, s)] = (rel, energy)
            rows.append(
                {
                    "model": name,
                    "sigma": s,
                    "rel_t0": float(rel[0]),
                    "rel_final": float(rel[-1]),
                    "energy_t0": float(energy[0]),
                    "energy_min": float(energy.min()),
                    "energy_at_frame5": float(energy[min(5, T - 1)]),
                    "energy_final": float(energy[-1]),
                }
            )
            print(
                f"{name:9s} sigma={s:<5g} rel {rel[0]:.4f} -> {rel[-1]:.4f} | "
                f"energy {energy[0]:.3f} -> min {energy.min():.3f} -> {energy[-1]:.3f}",
                flush=True,
            )
        del m
        torch.cuda.empty_cache()
    a.json.parent.mkdir(parents=True, exist_ok=True)
    # the curves themselves, so the figure can be redrawn without re-running the rollouts
    np.savez_compressed(
        a.json.with_suffix(".npz"),
        taus=taus,
        **{
            f"{n}__sigma{s:g}__{q}": v
            for (n, s), (rel, en) in curves.items()
            for q, v in (("rel", rel), ("energy", en))
        },
    )
    a.json.write_text(
        json.dumps(
            {
                "meta": {
                    "frames": T,
                    "n_problems": a.n_problems,
                    "sigmas": SIGMAS,
                    "seed": a.seed,
                    "note": "forecast only, no assimilation; noise in "
                    "normalised units on the true state at t0",
                },
                "rows": rows,
            },
            indent=2,
        )
    )

    t = taus
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 7.4), sharex=True)
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(SIGMAS)))
    for col, name in enumerate(("UNet", "KAE-expm")):
        for c, s in zip(cmap, SIGMAS):
            rel, en = curves[(name, s)]
            ax[0, col].semilogy(
                t, np.maximum(rel, 1e-8), color=c, lw=1.6, label=f"$\\sigma$={s}"
            )
            ax[1, col].plot(t, en, color=c, lw=1.6)
        ax[0, col].set_title(f"{name}: error against truth", fontsize=10)
        ax[1, col].axhline(1.0, color="k", lw=1, ls="--")
        ax[1, col].set(
            xlabel="time",
            ylabel=r"$\|u_{\rm pred}\|/\|u_{\rm true}\|$",
            title=f"{name}: energy ratio",
        )
        ax[0, col].grid(alpha=0.3, which="both")
        ax[1, col].grid(alpha=0.3)
    ax[0, 0].set_ylabel(r"rel-$L_2$")
    ax[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Free-running forecast from a perturbed true state (no assimilation)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = a.out_dir / "ks_noise_forecast.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
