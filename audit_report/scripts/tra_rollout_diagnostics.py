"""Long-horizon rollout diagnostics for the TRA continuous KAE (read-only on checkpoints).

For each held-out trajectory file and each start time t0 it measures:
  * RMSE vs lead time (normalised units, fluid cells only, channel-averaged) for
    the KAE, persistence, the training-set mean field ("climatology") and the
    oracle time-mean of the true trajectory,
  * anomaly correlation (ACC) w.r.t. the training climatology,
  * predicted / true fluctuation energy (does the forecast regress to the mean?),
  * reconstruction error D(E(x)) on the same frames (representation floor),
  * Koopman consistency ||K^k z0 - E(x_k)|| / ||E(x_k)|| and latent norms,
  * whether changing the Mach-number input changes the output at all.

Usage:
    python tra_rollout_diagnostics.py --run RUN_DIR --ckpt epoch_199.pth \
        --data FILE.nc --steps 600 --t0 20 150 300 --json out.json
"""

from _paths import MAIN_ROOT
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import xarray as xr
import yaml
from tensordict import TensorDict

REPO = MAIN_ROOT / "koopman_autoencoder"
sys.path.insert(0, str(REPO))
from models.autoencoder import KoopmanAutoencoder  # noqa: E402
from models.dataloader import TRA_MEAN, TRA_STD  # noqa: E402
from models.networks import TransformerConfig  # noqa: E402

VARS = ["v_x", "v_y", "p", "rho"]  # order of configs/data/tra.yaml


def build(run_dir: Path, ckpt: str, device):
    mcfg = yaml.safe_load(open(next(run_dir.glob("*.yaml"))))
    model = KoopmanAutoencoder(
        data_variables={v: 1 for v in VARS},
        input_frames=2,
        height=mcfg["height"],
        width=mcfg["width"],
        latent_dim=mcfg["latent_dim"],
        cond_embedding_dim=mcfg["cond_embedding_dim"],
        cond_type=mcfg["cond_type"],
        operator_mode=mcfg["operator_mode"],
        hidden_dims=mcfg["hidden_dims"],
        transformer_config=TransformerConfig(**mcfg["transformer"]),
        predict_cond=mcfg["predict_cond"],
        cond_grad_enabled=mcfg["cond_grad_enabled"],
        disturb_std=None,
        is_continuous=mcfg["is_continuous"],
        operator_type=mcfg.get("operator_type", "continuous"),
        rank=mcfg["rank"],
        spectral=mcfg["spectral"],
        cond_expansion_type="Ma",
        use_attention=mcfg["use_attention"],
        **mcfg["conv_kwargs"],
    )
    sd = torch.load(
        run_dir / "checkpoints" / ckpt, map_location="cpu", weights_only=False
    )
    model.load_state_dict(sd["model_state_dict"], strict=True)
    return model.to(device).eval()


def load_traj(path: Path, stride: int = 1):
    # NOTE: gt_longer_1000.nc and gt_*_subsample_1.nc label frames 0.1 apart but are
    # really 0.05 apart (shedding period ~50 frames vs ~25 at training dt); use stride 2.
    ds = xr.open_dataset(path).isel(t=slice(0, None, stride))
    mask = torch.from_numpy(ds.obstacle_mask.values).float()
    while mask.ndim > 2:  # some gt_* files store it per sim/time; dataset uses [0]
        mask = mask[0]
    x = {
        v: (torch.from_numpy(ds[v].values).float() - TRA_MEAN[v]) / (TRA_STD[v] + 1e-8)
        for v in VARS
    }
    x = {v: t * mask for v, t in x.items()}  # same as dataset.apply_mask
    return x, mask, torch.from_numpy(ds.Ma.values).float()


def train_climatology(path: Path, mask, dev):
    tr = xr.open_dataset(path)
    fields = []
    for v in VARS:
        m = (
            torch.from_numpy(tr[v].isel(t=slice(0, None, 10)).values)
            .float()
            .mean((0, 1))
        )
        fields.append(((m - TRA_MEAN[v]) / (TRA_STD[v] + 1e-8)) * mask)
    return torch.stack(fields).to(dev)


def stack(xd: Dict[str, torch.Tensor], idx) -> torch.Tensor:
    return torch.stack([xd[v][idx] for v in VARS], dim=-3)  # [..., C, H, W]


def window(x, t_end, dev):
    """Two-frame input window ending at t_end (inclusive), as the dataset builds it."""
    return TensorDict(
        {v: x[v][:, t_end - 1 : t_end + 1].to(dev) for v in VARS},
        batch_size=[x[VARS[0]].shape[0], 2],
    )


@torch.inference_mode()
def run_one(model, x, mask, ma, clim, t0, steps, dev):
    S = x["v_x"].shape[0]
    N = min(steps, x["v_x"].shape[1] - t0 - 1)
    fluid = mask.to(dev).bool()
    inp = window(x, t0, dev)
    inp["obstacle_mask"] = mask.to(dev).expand(S, 2, -1, -1)
    inp["cond_input"] = ma.view(S, 1).expand(S, 2).to(dev)
    truth = stack(x, (slice(None), slice(t0 + 1, t0 + 1 + N))).to(dev)  # [S,N,C,H,W]
    x_now = stack(x, (slice(None), t0)).to(dev)

    tic = time.time()
    z0 = model.encode(inp, cond_input=inp["cond_input"])
    z, z_list = z0, []
    for _ in range(N):
        z = model.koopman_operator(z, cond=inp["cond_input"][:, -1], dt=0.1)
        z_list.append(z)
    zs = torch.stack(z_list, 1)
    preds = []
    for k in range(0, N, 100):
        zz = zs[:, k : k + 100].reshape(-1, zs.shape[-1])
        d = model.decode(zz, obstacle_mask=inp["obstacle_mask"])
        preds.append(
            torch.stack([d[v] for v in VARS], 1).view(S, -1, len(VARS), *mask.shape)
        )
    pred = torch.cat(preds, 1)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - tic

    def rmse(a, b):  # [S, N]
        return ((a - b) ** 2)[..., fluid].mean(-1).mean(-1).sqrt()

    def acc(a, b):  # anomaly correlation wrt training climatology, [S, N]
        pa, ta = (a - clim)[..., fluid].flatten(2), (b - clim)[..., fluid].flatten(2)
        pa, ta = pa - pa.mean(-1, keepdim=True), ta - ta.mean(-1, keepdim=True)
        return (pa * ta).sum(-1) / (pa.norm(dim=-1) * ta.norm(dim=-1) + 1e-12)

    om = truth.mean(1, keepdim=True)
    out = {
        "kae_rmse": rmse(pred, truth),
        "persistence_rmse": rmse(x_now[:, None].expand_as(truth), truth),
        "train_climatology_rmse": rmse(clim[None, None].expand_as(truth), truth),
        "oracle_timemean_rmse": rmse(om.expand_as(truth), truth),
        "kae_acc": acc(pred, truth),
        "persistence_acc": acc(x_now[:, None].expand_as(truth), truth),
        "fluct_energy_ratio": ((pred - om) ** 2)[..., fluid].mean((-1, -2))
        / ((truth - om) ** 2)[..., fluid].mean((-1, -2)),
    }
    out = {k: v.cpu().numpy() for k, v in out.items()}  # each [S, N]

    rec, lat = {}, {}
    for k in sorted({1, 10, 50, 100, 200, 500, N} & set(range(1, N + 1))):
        ze = model.encode(window(x, t0 + k, dev))
        dk = model.decode(ze, obstacle_mask=inp["obstacle_mask"])
        rk = torch.stack([dk[v] for v in VARS], 1)
        rec[k] = float(rmse(rk[:, None], truth[:, k - 1 : k]).mean())
        zp = zs[:, k - 1]
        lat[k] = [
            float(((zp - ze).norm(dim=-1) / ze.norm(dim=-1)).mean()),
            float(zp.norm(dim=-1).mean()),
            float(ze.norm(dim=-1).mean()),
        ]

    z_a = model.koopman_operator(z0, cond=torch.full((S,), 0.50, device=dev), dt=0.1)
    z_b = model.koopman_operator(z0, cond=torch.full((S,), 0.89, device=dev), dt=0.1)
    e_a = model.encode(inp, cond_input=torch.full((S, 2), 0.50, device=dev))
    e_b = model.encode(inp, cond_input=torch.full((S, 2), 0.89, device=dev))
    extra = {
        "seconds_rollout_plus_decode": secs,
        "reconstruction_rmse_at_lead": rec,
        "latent_[relerr,|zpred|,|zenc|]_at_lead": lat,
        "mach_sensitivity_operator_maxabs": float((z_a - z_b).abs().max()),
        "mach_sensitivity_encoder_maxabs": float((e_a - e_b).abs().max()),
    }
    return N, out, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="epoch_199.pth")
    ap.add_argument("--data", default=str(REPO / "data/acdm/128_tra/gt_longer_1000.nc"))
    ap.add_argument("--train_data", default=str(REPO / "data/acdm/128_tra/train.nc"))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--t0", type=int, nargs="+", default=[20])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build(Path(args.run), args.ckpt, dev)
    x, mask, ma = load_traj(Path(args.data), args.stride)
    clim = train_climatology(Path(args.train_data), mask, dev)

    runs = [run_one(model, x, mask, ma, clim, t0, args.steps, dev) for t0 in args.t0]
    N = min(r[0] for r in runs)
    agg = {k: np.concatenate([r[1][k][:, :N] for r in runs], 0) for k in runs[0][1]}
    n_cases = agg["kae_rmse"].shape[0]
    leads = [
        lead
        for lead in [
            1,
            2,
            5,
            10,
            20,
            30,
            50,
            60,
            100,
            150,
            200,
            240,
            300,
            400,
            500,
            600,
        ]
        if lead <= N
    ]
    mean = {k: v.mean(0) for k, v in agg.items()}
    sem = {
        k: v.std(0, ddof=1) / np.sqrt(n_cases) if n_cases > 1 else 0 * v[0]
        for k, v in agg.items()
    }
    worse_clim = mean["kae_rmse"] > mean["train_climatology_rmse"]
    res = {
        "run": args.run,
        "ckpt": args.ckpt,
        "data": args.data,
        "stride": args.stride,
        "t0": args.t0,
        "n_cases": int(n_cases),
        "steps": int(N),
        "device": str(dev),
        "by_lead_mean_sem": {
            lead: {k: [float(mean[k][lead - 1]), float(sem[k][lead - 1])] for k in mean}
            for lead in leads
        },
        "first_lead_kae_rmse_above_train_climatology": (
            int(np.argmax(worse_clim) + 1) if worse_clim.any() else None
        ),
        "first_lead_kae_acc_below_0.6": (
            int(np.argmax(mean["kae_acc"] < 0.6) + 1)
            if (mean["kae_acc"] < 0.6).any()
            else None
        ),
        "first_lead_persistence_acc_below_0.6": (
            int(np.argmax(mean["persistence_acc"] < 0.6) + 1)
            if (mean["persistence_acc"] < 0.6).any()
            else None
        ),
        "per_t0_extra": {t0: r[2] for t0, r in zip(args.t0, runs)},
    }
    print(json.dumps(res, indent=1))
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
