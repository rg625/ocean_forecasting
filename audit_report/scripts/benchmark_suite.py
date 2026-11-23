"""Unified TRA benchmark: score every prediction source on identical cases and frames.

Ground truth: autoreg_pde_diffusion/src/results/sampling/<case>/groundTruth.dict
(ACDM test protocol: interp/extrap 6 sequences x 60 frames, longer 4 x 240; frames 0,1
are the conditioning window, frames 2..T-1 are predicted).

Prediction files are npz with arr_0 shaped [..., S, T, C>=4, 128, 64] in physical units,
fields [v_x, v_y, rho, p]; leading dims are model seeds x samples. Files written by
evaluate.py between a5f4796 and the BUG-01 fix are stored one frame early and are
realigned (see EARLY).

Metrics per case and source (mean over leading dims, plus spread over model seeds):
  mse_pred      MSE x1e-3, physical units, obstacle zeroed, mean over 4 fields, H, W and
                predicted frames 2..T-1
  mse_all       same including the 2 conditioning frames (plot_loss_all convention)
  mse_lead      per-lead MSE x1e-3
  acc_lead      anomaly correlation vs the training climatology (normalised units,
                fluid cells)
  energy_lead   predicted / true fluctuation energy about the true time mean
  first_acc_below_0.6, energy_ratio at leads 30/58/238

Usage:
    python benchmark_suite.py --sources KEY=PATH_PATTERN ... --json out.json
    PATH_PATTERN may contain {case}. Built-in sources are always included.
"""

from _paths import MAIN_ROOT
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = MAIN_ROOT
SAMPLING = ROOT / "autoreg_pde_diffusion/src/results/sampling"
UNTIMED = ROOT / "autoreg_pde_diffusion/results_untimed"
TRAIN = ROOT / "koopman_autoencoder/data/acdm/128_tra/train.nc"
GT_STD = np.array([0.216987, 0.216987, 0.145391, 0.119944])  # v_x, v_y, rho, p
CASES = ["interp", "extrap", "longer"]
EARLY = {"kae_run20260821_ep199", "kae_azencot"}  # stored one frame early

BUILTIN = {
    "kae_run20260821_ep199": str(SAMPLING / "{case}/continous_linear_128_tra.npz"),
    "kae_azencot": str(SAMPLING / "{case}/azencot_linear_128_tra.npz"),
    "kae_jan_rk4": str(SAMPLING / "{case}/continous_linear_128_rk4.npz"),
    "acdm-r20_second_run": str(SAMPLING / "{case}/acdm-r20.npz"),
}
ACDM_BASELINES = [
    "acdm-r20",
    "acdm-r20_ncn",
    "unet-m2",
    "unet-m8",
    "unet-m2-noise0.01",
    "fno-16modes-m2",
    "fno-32modes-m2",
    "resnet-m2",
    "dil-resnet-m2",
    "tf-enc",
    "tf-mgn",
    "tf-vae",
    "refiner-r4_std0.000001",
]


def load_gt(case):
    gd = torch.load(SAMPLING / case / "groundTruth.dict", weights_only=False)
    mask = np.asarray(gd["obsMask"])[0]
    gt = np.asarray(gd["data"])[:, :, :4] * mask[None, None, None]
    return gt.astype(np.float32), mask > 0.5


def train_climatology():
    tr = xr.open_dataset(TRAIN)
    return np.stack(
        [
            tr[v].isel(t=slice(0, None, 10)).values.mean((0, 1)).T
            for v in ["v_x", "v_y", "rho", "p"]
        ]
    ).astype(np.float32)


def score(pred, gt, fluid, clim):
    """pred: [n, S, T, 4, H, W] aligned with gt [S, T, 4, H, W] (frames 0,1 = inputs)."""
    T = gt.shape[1]
    std = GT_STD[None, None, :, None, None]
    tgt = gt[:, 2:]
    p = pred[:, :, 2:] * fluid
    err = (p - tgt[None]) ** 2  # [n, S, N, 4, H, W]
    mse_lead = err.mean(axis=(0, 1, 3, 4, 5)) * 1e3
    per_model = err.mean(axis=(1, 2, 3, 4, 5)) * 1e3
    mse_all = (((pred * fluid - gt[None]) ** 2).mean()) * 1e3
    # normalised anomalies on fluid cells
    pa = ((p - clim[None, None]) / std[None])[..., fluid]  # [n, S, N, 4, F]
    ta = ((tgt - clim[None]) / std)[..., fluid]
    pa = pa.reshape(*pa.shape[:3], -1)
    ta = ta.reshape(*ta.shape[:2], -1)
    pa = pa - pa.mean(-1, keepdims=True)
    ta = ta - ta.mean(-1, keepdims=True)
    acc = (pa * ta[None]).sum(-1) / (
        np.linalg.norm(pa, axis=-1) * np.linalg.norm(ta, axis=-1)[None] + 1e-12
    )
    acc_lead = acc.mean(axis=(0, 1))
    om = tgt.mean(1, keepdims=True)
    fe_p = (
        (((p - om[None]) / std[None]) ** 2)[..., fluid].mean(-1).mean(-1)
    )  # [n, S, N]
    fe_t = (((tgt - om) / std) ** 2)[..., fluid].mean(-1).mean(-1)  # [S, N]
    energy_lead = (fe_p / fe_t[None]).mean(axis=(0, 1))
    leads = {30: 29, 58: 57, 238: 237}
    out = {
        "mse_pred": float(per_model.mean()),
        "mse_pred_seed_std": (
            float(per_model.std(ddof=1)) if len(per_model) > 1 else None
        ),
        "n_models_or_samples": int(pred.shape[0]),
        "mse_all": float(mse_all),
        "first_acc_below_0.6": (
            int(np.argmax(acc_lead < 0.6) + 1) if (acc_lead < 0.6).any() else None
        ),
        "energy_ratio_at": {
            k: float(energy_lead[v]) for k, v in leads.items() if v < T - 2
        },
        "acc_at": {k: float(acc_lead[v]) for k, v in leads.items() if v < T - 2},
        "mse_lead": [float(x) for x in mse_lead],
        "acc_lead": [float(x) for x in acc_lead],
        "energy_lead": [float(x) for x in energy_lead],
    }
    return out


def load_pred(path, S, T, early, seeds_axis=True):
    a = np.load(path)["arr_0"]
    if a.shape[-4] != T:
        return None
    a = a.reshape(-1, S, *a.shape[-4:])[:, :, :, :4].astype(np.float32)
    if early:  # stored frame j = ground-truth frame j+1
        b = np.empty_like(a)
        b[:, :, 1:] = a[:, :, :-1]
        b[:, :, 0] = np.nan  # never scored (frames 0,1 are inputs)
        b[:, :, -1:] = a[:, :, -1:]  # last GT frame has no prediction; see note
        a = b
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=[], help="KEY=PATTERN with {case}")
    ap.add_argument("--no_builtin", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    clim = train_climatology()
    sources = {} if args.no_builtin else dict(BUILTIN)
    if not args.no_builtin:
        for b in ACDM_BASELINES:
            sources[b] = str(UNTIMED / "{case}" / f"{b}.npz")
    for s in args.sources:
        k, v = s.split("=", 1)
        sources[k] = v
    res = {}
    for case in CASES:
        gt, fluid = load_gt(case)
        S, T = gt.shape[:2]
        rc = {}
        rc["persistence"] = score(
            np.repeat(gt[None, :, 1:2], T, axis=2), gt, fluid, clim
        )
        rc["train_climatology"] = score(
            np.broadcast_to(
                clim[None, None, None] * fluid, (1, S, T, 4, *fluid.shape)
            ).copy(),
            gt,
            fluid,
            clim,
        )
        for key, pat in sources.items():
            path = Path(pat.format(case=case))
            if not path.exists():
                continue
            a = load_pred(path, S, T, early=key in EARLY)
            if a is None:
                rc[key] = {"skipped": "frame count mismatch"}
                continue
            if (
                key in EARLY
            ):  # the final GT frame has no prediction: score frames 2..T-2
                a = a[:, :, :-1]
                rc[key] = score(a, gt[:, :-1], fluid, clim)
                rc[key]["note"] = "realigned; last frame not available"
            else:
                rc[key] = score(a, gt, fluid, clim)
        res[case] = rc
        print(f"== {case}")
        for k, v in sorted(rc.items(), key=lambda kv: kv[1].get("mse_pred", 1e9)):
            if "mse_pred" in v:
                sd = (
                    f"+-{v['mse_pred_seed_std']:.2f}"
                    if v["mse_pred_seed_std"] is not None
                    else ""
                )
                print(
                    f"  {k:28s} mse {v['mse_pred']:7.2f}{sd:8s} acc<0.6@{v['first_acc_below_0.6']}  "
                    f"E {json.dumps({kk: round(vv, 2) for kk, vv in v['energy_ratio_at'].items()})}"
                )
    if args.json:
        json.dump(res, open(args.json, "w"))


if __name__ == "__main__":
    main()
