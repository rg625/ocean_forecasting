"""Roll every model forward from the TRUE initial condition and show the fields.

No assimilation anywhere in this script.  Each model is handed the exact state and asked
to predict forward, which is the cleanest possible test of the model itself: if a
"surprisingly bad" DA number comes from a model that cannot predict, the fix is the model;
if the model predicts well, the fix is the inversion.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_assimilation.tra.bridge import REGIMES, PhysicalData, REPO, rel_l2
from data_assimilation.tra.adapters import TurbpredAdapter
from data_assimilation.tra.kae_adapter import KAEAdapter
from data_assimilation.tra.verify_adapters import MODELS, DIFF_OPTS

ORDER = ["KAE", "UNet", "FNO", "ACDM", "ACDM-ncn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--example", type=int, default=0)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument(
        "--kae-run",
        default=("model_outputs_tra/continous_linear_128/" "run-20260821_031121"),
    )
    ap.add_argument("--out", default="da_results_tra/rollout_from_true_ic.png")
    a = ap.parse_args()

    dev = torch.device("cuda")
    reg = REGIMES["tra"]
    d = PhysicalData("data/acdm/128_tra/gt_interp.nc", "tra", dev)
    sim = torch.arange(a.batch)
    t0 = torch.full((a.batch,), 4, dtype=torch.long)
    par, om = d.params_for(sim), d.mask_for(sim)
    T = a.frames
    truth = torch.stack([d.frames(sim, t0 + k) for k in range(T)], dim=1)
    base = REPO / "autoreg_pde_diffusion" / "pretrained_models" / "models_tra"

    rolls, errs = {}, {}
    for m in ORDER:
        if m == "KAE":
            ad = KAEAdapter(a.kae_run, reg, dev)
            w = d.window(sim, t0 - (ad.n_control_frames - 1), ad.n_control_frames)
            with torch.no_grad():
                z = ad.encode(w, par)
                K = ad.generator(par)
                roll = torch.stack(
                    [
                        ad.decode(ad.propagate(z, k * ad.dt_train, K=K))
                        for k in range(T)
                    ],
                    dim=1,
                )
        else:
            ad = TurbpredAdapter(
                base / MODELS[m] / "Model.pth", reg, dev, DIFF_OPTS.get(m), name=m
            )
            k0 = ad.n_control_frames
            w = d.window(sim, t0 - (k0 - 1), k0)
            with torch.no_grad():
                tr = ad.rollout(ad.to_model(w), T, par, checkpoint_every=0)
            roll = tr[:, k0 - 1 : k0 - 1 + T]
        rolls[m] = roll.cpu().numpy()
        errs[m] = (
            torch.stack([rel_l2(roll[:, k], truth[:, k], om) for k in range(T)])
            .mean(1)
            .cpu()
            .numpy()
        )
        print(
            f"{m:9s} rel-L2  1 frame {errs[m][1]:.5f} | "
            f"{T//2} frames {errs[m][T//2]:.5f} | {T-1} frames {errs[m][-1]:.5f}"
        )
        del ad
        torch.cuda.empty_cache()

    pers = (
        torch.stack([rel_l2(truth[:, 0], truth[:, k], om) for k in range(T)])
        .mean(1)
        .cpu()
        .numpy()
    )
    print(
        f"{'persistence':9s} rel-L2  1 frame {pers[1]:.5f} | "
        f"{T//2} frames {pers[T//2]:.5f} | {T-1} frames {pers[-1]:.5f}"
    )

    # ---- figure: space-time fields, truth then each model, with |error| beneath -----
    tr0 = truth[a.example, :, a.channel].cpu().numpy()
    mask = None if om is None else om[a.example].cpu().numpy()
    show = (
        (lambda f: np.where(mask > 0, f, np.nan)) if mask is not None else (lambda f: f)
    )
    ncol = 1 + len(ORDER)
    fig, ax = plt.subplots(2, ncol, figsize=(3.0 * ncol, 5.6))
    v = np.nanpercentile(np.abs(show(tr0)), 99)
    emax = max(
        np.nanpercentile(
            np.concatenate(
                [
                    np.abs(show(rolls[m][a.example, :, a.channel] - tr0)).ravel()
                    for m in ORDER
                ]
            ),
            99,
        ),
        1e-6,
    )
    # time on x, one spatial line through the domain on y
    mid = tr0.shape[1] // 2
    ext = [0, T, 0, tr0.shape[2]]
    ax[0, 0].imshow(
        show(tr0)[:, mid].T,
        origin="lower",
        aspect="auto",
        extent=ext,
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
    )
    ax[0, 0].set_title("TRUTH", fontsize=9)
    ax[0, 0].set_ylabel("y (slice at mid-x)")
    ax[1, 0].axis("off")
    ax[1, 0].text(
        0.5,
        0.5,
        f"rollout from the\nTRUE initial condition\n\n{T} frames, "
        f"no assimilation\n\nbottom row: |error|,\nshared scale",
        ha="center",
        va="center",
        fontsize=8,
        transform=ax[1, 0].transAxes,
    )
    for c, m in enumerate(ORDER, start=1):
        r = rolls[m][a.example, :, a.channel]
        ax[0, c].imshow(
            show(r)[:, mid].T,
            origin="lower",
            aspect="auto",
            extent=ext,
            cmap="RdBu_r",
            vmin=-v,
            vmax=v,
        )
        ax[0, c].set_title(m, fontsize=9)
        ax[1, c].imshow(
            np.abs(show(r - tr0))[:, mid].T,
            origin="lower",
            aspect="auto",
            extent=ext,
            cmap="magma",
            vmin=0,
            vmax=emax,
        )
        ax[1, c].set_title(f"|error|  {errs[m][-1]:.4f} @ {T-1} fr", fontsize=8)
        for row in (0, 1):
            ax[row, c].set_xlabel("frame")
    fig.suptitle(
        "Free-running rollout from the TRUE initial condition "
        "(no assimilation anywhere)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(a.out, dpi=115, bbox_inches="tight")
    np.savez_compressed(
        "da_results_tra/rollout_from_true_ic.npz",
        truth=truth.cpu().numpy(),
        persistence=pers,
        **{f"{m}__roll": rolls[m] for m in ORDER},
        **{f"{m}__err": errs[m] for m in ORDER},
    )
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
