"""Score saved TRA predictions under ONE convention (read-only on result files).

Convention: physical units, obstacle cells zeroed in both, mean over all pixels, the
four fields and the PREDICTED frames 2..T-1 only (the two conditioning frames are
excluded), reported x1e-3. KAE files written by evaluate.py after a5f4796 are stored
one frame early (frame j holds the prediction for ground-truth frame j+1) and are
realigned here; the "as_scored" column reproduces the misaligned comparison.

Also scores persistence (last conditioning frame) and the training climatology, and
merges the POD baselines from linear_baselines.py.

Usage:
    python benchmark_table.py --pod results/linear_baselines_r128.json --json out.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

SAMPLING = Path(
    "/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion/src/results/sampling"
)
TRAIN = Path(
    "/home/rg625/mnt/ocean_forecasting/koopman_autoencoder/data/acdm/128_tra/train.nc"
)
MODELS = {  # file -> stored-one-frame-early?
    "continous_linear_128_tra.npz": True,
    "azencot_linear_128_tra.npz": True,
    "continous_linear_128_rk4.npz": False,
    "continous_linear_128_exp.npz": False,
    "acdm-r20.npz": False,
    "acdm-r20_ncn.npz": False,
    "uncond-ddpm.npz": False,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    pod = json.load(open(args.pod)) if args.pod else {}
    tr = xr.open_dataset(TRAIN)
    clim = np.stack(  # GT dict layout: fields [v_x, v_y, rho, p], spatial (128, 64)
        [
            tr[v].isel(t=slice(0, None, 10)).values.mean((0, 1)).T
            for v in ["v_x", "v_y", "rho", "p"]
        ]
    )
    out = {}
    for case in ["interp", "extrap", "longer"]:
        gd = torch.load(SAMPLING / case / "groundTruth.dict", weights_only=False)
        mask = np.asarray(gd["obsMask"])[0][None, None, None]
        gt = np.asarray(gd["data"])[:, :, :4] * mask  # [S, T, 4, H, W]
        S, T = gt.shape[:2]
        tgt = gt[:, 2:]

        def mse(pred):
            return float(((pred - tgt) ** 2).mean() * 1e3)

        res = {
            "persistence": mse(np.repeat(gt[:, 1:2], T - 2, axis=1)),
            "train_climatology": mse(
                np.broadcast_to(clim[None, None] * mask, tgt.shape)
            ),
        }
        for f, early in MODELS.items():
            p = SAMPLING / case / f
            if not p.exists():
                continue
            a = np.load(p)["arr_0"]
            if a.shape[-4] != T:  # e.g. saved at the 0.05 (subsample_1) resolution
                res[f + " (skipped)"] = float(a.shape[-4])
                continue
            a = (
                a.reshape(-1, S, *a.shape[-4:])[:, :, :, :4] * mask
            )  # [n, S, T, 4, H, W]
            if early:
                res[f + " (realigned)"] = mse(a[:, :, 1 : T - 1])
                res[f + " (as_scored)"] = mse(a[:, :, 2:])
            else:
                res[f] = mse(a[:, :, 2:])
        key = f"gt_{case}"
        if key in pod:
            for m in ["pod_dmd", "pod_dmd2"]:
                res[f"{m}_r{pod['rank']}"] = pod[key][m]["phys_mse_1e-3_pred_frames"]
        out[case] = res
        print(case, json.dumps({k: round(v, 2) for k, v in res.items()}, indent=1))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
