# ruff: noqa: E731
"""Space-time trajectories of a forecast started from a PERTURBED true state.

Same layout as the chaotic assimilation panels: truth and each model's rollout on top, the
per-frame error and the |error| maps below. One figure per noise amplitude, all on the same
chaotic trajectory, so the effect of the perturbation can be read off the fields rather than
from an error curve alone. No assimilation is involved: the only input is u(t_0) + sigma*eps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensordict import TensorDict

from data_assimilation.ks.protocol import DT, build_problem
from data_assimilation.ks.da_ks_experiments_3way import Bench, add_common_args


def pick_chaotic(b, window, n_cand, seed):
    """The trajectory whose state-space path turns most over the window (see fig_ks_gallery)."""
    prob = build_problem(
        b.data,
        name="cand",
        n_problems=n_cand,
        taus=np.array([window - 1]) * DT,
        seed=seed,
    )
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    u = np.stack(
        [
            b.data.denorm(b.data.frames(sim, t0 + j)).cpu().numpy()
            for j in range(window)
        ],
        axis=1,
    )
    nrm = lambda a: np.linalg.norm(a, axis=-1)
    drift = nrm(u[:, -1] - u[:, 0]) / nrm(u[:, 0])
    wiggle = nrm(np.diff(u, axis=1)).sum(1) / nrm(u[:, 0]) / np.maximum(drift, 1e-6)
    i = int(np.argmax(wiggle))
    return prob.sim[i], prob.t0[i], float(drift[i]), float(wiggle[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--out-dir", type=Path, default=Path("figs"))
    ap.add_argument(
        "--npz", type=Path, default=Path("da_results_geometry_df9/noise_spacetime.npz")
    )
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.05, 0.2, 0.4])
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=43)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    b = Bench(a)
    dev, x = b.dev, b.data.x
    sim_i, t0_i, drift, wig = pick_chaotic(b, a.window, a.candidates, a.seed)
    print(
        f"trajectory {sim_i}, t0 frame {t0_i} (drift {drift:.2f}, path {wig:.0f}x)",
        flush=True,
    )
    sim = torch.as_tensor([sim_i], device=dev)
    t0 = torch.as_tensor([t0_i], device=dev)
    W = a.window
    truth = torch.stack([b.data.frames(sim, t0 + k) for k in range(W)])  # [W,1,X]
    tru = b.data.denorm(truth)[:, 0].cpu().numpy()
    taus = np.arange(W) * DT
    g = torch.Generator(device=dev).manual_seed(a.seed)
    store = {"taus": taus, "truth": tru, "sim": np.array(sim_i), "t0": np.array(t0_i)}
    models = {"U-Net": b.method("UNet"), "KAE": b.method("KAE-expm")}

    for s in a.sigmas:
        x0 = truth[0] + (
            s * torch.randn(truth[0].shape, device=dev, generator=g) if s > 0 else 0.0
        )
        fields = {}
        for name, m in models.items():
            with torch.no_grad():
                if name == "KAE":
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
            fields[name] = b.data.denorm(pred)[:, 0].cpu().numpy()
            store[f"sigma{s:g}__{name}"] = fields[name]
        store[f"sigma{s:g}__input"] = b.data.denorm(x0)[0].cpu().numpy()

        errs = {n: np.abs(f - tru) for n, f in fields.items()}
        emax = max(
            float(
                np.percentile(np.concatenate([e.ravel() for e in errs.values()]), 99.5)
            ),
            1e-6,
        )
        vmax = float(np.abs(tru).max())
        fig, ax = plt.subplots(2, 3, figsize=(13.2, 5.6))
        ext = [0, taus[-1], float(x.min()), float(x.max())]
        ax[0, 0].imshow(
            tru.T,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=ext,
        )
        ax[0, 0].set(title="TRUTH", ylabel="x")
        for c, name in enumerate(models, start=1):
            ax[0, c].imshow(
                fields[name].T,
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                extent=ext,
            )
            ax[0, c].set_title(name)
            im = ax[1, c].imshow(
                errs[name].T,
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=0,
                vmax=emax,
                extent=ext,
            )
            rel = np.linalg.norm(fields[name] - tru, axis=-1) / np.linalg.norm(
                tru, axis=-1
            )
            ax[1, c].set(
                title=f"|error|, rel-$L_2$ at $t_0$ = {rel[0]:.3f}", xlabel="time"
            )
            fig.colorbar(im, ax=ax[1, c], fraction=0.046, pad=0.02)
        for name, col in (("U-Net", "#d6604d"), ("KAE", "#1b7837")):
            rel = np.linalg.norm(fields[name] - tru, axis=-1) / np.linalg.norm(
                tru, axis=-1
            )
            ax[1, 0].semilogy(
                taus, np.maximum(rel, 1e-9), color=col, lw=1.7, label=name
            )
            store[f"sigma{s:g}__{name}__rel"] = rel
        ax[1, 0].set(
            xlabel="time",
            ylabel=r"rel-$L_2$ per frame",
            title="error against time (log)",
        )
        ax[1, 0].grid(alpha=0.3, which="both")
        ax[1, 0].legend(fontsize=8)
        for a_ in ax[0]:
            a_.set_xlabel("time")
        fig.suptitle(
            f"Forecast from $u(t_0)+\\sigma\\epsilon$, $\\sigma={s:g}$ "
            f"(trajectory {sim_i}, $t_0$ frame {t0_i}). No assimilation; "
            f"error panels share one scale (0 to {emax:.2f})",
            fontsize=10.5,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = a.out_dir / f"ks_noise_spacetime_sigma{s:g}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out, flush=True)
    a.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.npz, **store)
    print("saved ->", a.npz)


if __name__ == "__main__":
    main()
