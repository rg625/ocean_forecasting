"""POD + linear-dynamics baselines for the TRA benchmark (no neural networks).

Fits, on the TRAINING file only (train.nc, subsample 2 -> dt = 0.1 as in training):
  * POD basis of the masked, normalised fields (rank r, about the training mean),
  * "pod_dmd":    a_{t+1} = K a_t                     (least squares, all train sims),
  * "pod_dmd2":   a_{t+1} = K1 a_t + K2 a_{t-1}        (2-frame history, like the KAE),
  * "*_unit":     same, with eigenvalues of the companion map clipped to |mu| <= 1.

Evaluates on gt_interp / gt_extrap / gt_longer with the protocol of
tra_rollout_diagnostics.py (input frames 0,1; predict frames 2..T-1; normalised
units; fluid cells only), reporting RMSE, ACC w.r.t. the training climatology,
fluctuation-energy ratio and the POD projection floor.

Usage:
    python linear_baselines.py --rank 128 --json out.json
"""

from _paths import MAIN_ROOT
import argparse
import json
import sys
import time

import numpy as np
import torch
import xarray as xr

REPO = MAIN_ROOT / "koopman_autoencoder"
sys.path.insert(0, str(REPO))
from models.dataloader import TRA_MEAN, TRA_STD  # noqa: E402

VARS = ["v_x", "v_y", "p", "rho"]
DATA = REPO / "data/acdm/128_tra"


def load(path, t_slice, stride):
    ds = xr.open_dataset(path).isel(t=slice(t_slice[0], t_slice[1], stride))
    mask = ds.obstacle_mask.values
    while mask.ndim > 2:
        mask = mask[0]
    x = np.stack(
        [((ds[v].values - TRA_MEAN[v]) / (TRA_STD[v] + 1e-8)) * mask for v in VARS],
        axis=2,
    ).astype(
        np.float32
    )  # [S, T, C, H, W]
    return x, mask > 0.5


def fit_ls(Y, X, ridge=1e-6):
    """Solve Y = K X in least squares with a tiny ridge; X: [n, N], Y: [m, N]."""
    G = X @ X.T
    G += ridge * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
    return np.linalg.solve(G, X @ Y.T).T


def clip_unit(M):
    """Clip eigenvalues of the (companion) map to the closed unit disc."""
    mu, V = np.linalg.eig(M)
    mu = np.where(np.abs(mu) > 1, mu / np.abs(mu), mu)
    return np.real(V @ np.diag(mu) @ np.linalg.inv(V))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    r = args.rank
    tic = time.time()

    xtr, fluid = load(DATA / "train.nc", (0, None), 2)  # dt = 0.1 like training
    S, T = xtr.shape[:2]
    flat = torch.from_numpy(xtr.reshape(S * T, -1))
    mean = flat.mean(0, keepdim=True)
    U, s, V = torch.svd_lowrank(flat - mean, q=r + 20, niter=4)
    Phi = V[:, :r].numpy()  # [n_pix, r]
    energy = float((s[:r] ** 2).sum() / ((flat - mean) ** 2).sum())
    a = ((flat - mean).numpy() @ Phi).reshape(S, T, r)
    mean = mean.numpy()[0]
    clim = mean.reshape(xtr.shape[2:])

    # one-step and two-step linear maps fitted on all training transitions
    X1 = a[:, :-1].reshape(-1, r).T
    Y1 = a[:, 1:].reshape(-1, r).T
    K = fit_ls(Y1, X1)
    X2 = np.concatenate([a[:, 1:-1], a[:, :-2]], axis=-1).reshape(-1, 2 * r).T
    Y2 = a[:, 2:].reshape(-1, r).T
    K12 = fit_ls(Y2, X2)
    comp = np.zeros((2 * r, 2 * r))
    comp[:r] = K12
    comp[r:, :r] = np.eye(r)
    models = {
        "pod_dmd": ("one", K),
        "pod_dmd_unit": ("one", clip_unit(K)),
        "pod_dmd2": ("two", comp),
        "pod_dmd2_unit": ("two", clip_unit(comp)),
    }
    spectra = {
        "pod_dmd_max|mu|": float(np.abs(np.linalg.eigvals(K)).max()),
        "pod_dmd2_max|mu|": float(np.abs(np.linalg.eigvals(comp)).max()),
    }
    del flat, xtr
    fit_s = time.time() - tic

    out: dict = {
        "rank": r,
        "pod_energy_fraction_train": energy,
        "fit_seconds": fit_s,
        **spectra,
    }
    for case in ["gt_interp", "gt_extrap", "gt_longer"]:
        x, _ = load(DATA / f"{case}.nc", (0, None), 1)
        Sx, Tx = x.shape[:2]
        N = Tx - 2
        truth = x[:, 2:]
        ax = (x.reshape(Sx, Tx, -1) - mean) @ Phi  # [S, T, r]
        proj = (ax[:, 2:] @ Phi.T + mean).reshape(truth.shape)

        def metrics(pred):
            e = ((pred - truth) ** 2)[..., fluid].mean(-1).mean(-1)  # [S, N]
            pa = (pred - clim)[..., fluid].reshape(Sx, N, -1)
            ta = (truth - clim)[..., fluid].reshape(Sx, N, -1)
            pa = pa - pa.mean(-1, keepdims=True)
            ta = ta - ta.mean(-1, keepdims=True)
            acc = (pa * ta).sum(-1) / (
                np.linalg.norm(pa, axis=-1) * np.linalg.norm(ta, axis=-1) + 1e-12
            )
            om = truth.mean(1, keepdims=True)
            fe = ((pred - om) ** 2)[..., fluid].mean((-1, -2)) / ((truth - om) ** 2)[
                ..., fluid
            ].mean((-1, -2))
            return np.sqrt(e).mean(0), acc.mean(0), fe.mean(0)

        std_ = np.array([TRA_STD[v] for v in VARS])[None, None, :, None, None]
        res: dict = {
            "projection_floor_rmse": float(metrics(proj)[0].mean()),
            "train_climatology_phys_mse_1e-3": float(
                (((clim[None, None] - truth) * std_) ** 2).mean() * 1e3
            ),
        }
        for name, (kind, M) in models.items():
            if kind == "one":
                z, zs = ax[:, 1], []
                for _ in range(N):
                    z = z @ M.T
                    zs.append(z)
            else:
                z, zs = np.concatenate([ax[:, 1], ax[:, 0]], -1), []
                for _ in range(N):
                    z = z @ M.T
                    zs.append(z[:, :r])
            pred = (np.stack(zs, 1) @ Phi.T + mean).reshape(truth.shape)
            rm, acc, fe = metrics(pred)
            # paper / plot_loss_all convention: physical units, obstacle zeroed, mean over
            # all pixels, fields and the predicted frames (x1e-3)
            std = np.array([TRA_STD[v] for v in VARS])[None, None, :, None, None]
            phys_mse = float((((pred - truth) * std) ** 2).mean() * 1e3)
            leads = [
                lead
                for lead in [1, 2, 5, 10, 20, 30, 50, 58, 100, 150, 200, 238]
                if lead <= N
            ]
            res[name] = {
                "by_lead[rmse,acc,energy_ratio]": {
                    lead: [
                        float(rm[lead - 1]),
                        float(acc[lead - 1]),
                        float(fe[lead - 1]),
                    ]
                    for lead in leads
                },
                "first_lead_acc_below_0.6": (
                    int(np.argmax(acc < 0.6) + 1) if (acc < 0.6).any() else None
                ),
                "mean_rmse_all_leads": float(rm.mean()),
                "phys_mse_1e-3_pred_frames": phys_mse,
                "phys_mse_1e-3_incl_2_cond_frames": phys_mse * N / (N + 2),
            }
        out[case] = res
    print(json.dumps(out, indent=1))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
