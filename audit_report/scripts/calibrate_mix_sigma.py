"""Calibrate the time-scale mixture readout (models/ct_kae.mixture_nodes) of one CT-KAE run on
TRAINING simulations only: 58-lead rollouts (stride 2, dt 0.1, the test-case length) from two
windows per training simulation of train.nc, MSE in physical units (x1e-3) for each sigma of a
grid; the minimiser is written to RUN/mix_sigma.json. No test file is read.
--leads 238 --starts 300 500 --out mix_sigma_long.json calibrates for the Tra_long horizon
(238 leads) instead; the training data then stay on the CPU and windows go to the GPU in
chunks, so memory does not grow with the horizon.

    python calibrate_mix_sigma.py RUN_DIR [--ckpt last.pth] [--device cuda] [--leads K]
        [--starts S ...] [--out FILE]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path("/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder")
sys.path.insert(0, str(REPO))
from eval_ctkae import load_model  # noqa: E402
from models.ct_data import TraWindows, denormalise  # noqa: E402
from models.ct_kae import mixture_nodes  # noqa: E402

GRID = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
STARTS = [300, 700]  # raw-frame indices (0.05 apart); the rollouts span 116 raw frames
K = 58


@torch.no_grad()
def mixture_mse(m, win, tgt, taus, ma, mask, sigma, chunk=256):
    """Mean squared error (physical units, x1e-3) of the mixture forecast; win / tgt may live
    on the CPU (they are moved to the model's device in groups of 8 windows)."""
    dev = next(m.parameters()).device
    num, cnt = 0.0, 0
    for i in range(0, len(win), 8):
        sl = slice(i, i + 8)
        e = _mixture_se(
            m,
            win[sl].to(dev),
            tgt[sl].to(dev),
            taus[sl].to(dev),
            ma[sl].to(dev),
            mask.to(dev),
            sigma,
            chunk,
        )
        num, cnt = num + e[0], cnt + e[1]
    return num / cnt * 1e3


def _mixture_se(m, win, tgt, taus, ma, mask, sigma, chunk):
    z0 = torch.cat(
        [
            m.encode(win[i : i + 64], mask, ma[i : i + 64])
            for i in range(0, len(win), 64)
        ]
    )
    pred = 0.0
    for sq, wq in mixture_nodes(sigma):
        zs = m.forecast(z0, taus, ma, sq).flatten(0, 1)  # [N K, d]
        mk = ma.repeat_interleave(taus.shape[1])
        dec = []
        for i in range(0, len(zs), chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=zs.is_cuda):
                dec.append(m.decode(zs[i : i + chunk], mask, mk[i : i + chunk]).float())
        pred = pred + wq * torch.cat(dec).view(*tgt.shape)
    se = (denormalise(pred) - denormalise(tgt)) ** 2
    return float(se.double().sum()), se.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="last.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--leads", type=int, default=K)
    ap.add_argument("--starts", type=int, nargs="+", default=STARTS)
    ap.add_argument("--out", default="mix_sigma.json")
    a = ap.parse_args()
    dev = torch.device(a.device)
    run = Path(a.run_dir)
    m, _ = load_model(run, a.ckpt, dev)
    big = a.leads > K  # long horizon: keep the data on the CPU
    tw = TraWindows(
        str(REPO / "data/acdm/128_tra/train.nc"),
        None,
        torch.device("cpu") if big else dev,
    )
    win, tgt, taus, ma = tw.fixed_windows(m.cfg.n_frames, a.leads, 2, a.starts)
    mse = [mixture_mse(m, win, tgt, taus, ma, tw.mask, s) for s in GRID]
    best = GRID[min(range(len(GRID)), key=mse.__getitem__)]
    out = {
        "sigma": best,
        "grid": GRID,
        "train_mse": mse,
        "ckpt": a.ckpt,
        "windows": {
            "n": len(win),
            "starts_raw": a.starts,
            "leads": a.leads,
            "stride": 2,
        },
        "note": "calibrated on train.nc rollouts only; no test data",
    }
    (run / a.out).write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {"run": run.name, "sigma": best, "train_mse": [round(v, 3) for v in mse]}
        )
    )


if __name__ == "__main__":
    main()
