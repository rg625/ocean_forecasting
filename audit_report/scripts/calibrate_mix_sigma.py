"""Calibrate the time-scale mixture readout (models/ct_kae.mixture_nodes) of one CT-KAE run on
TRAINING simulations only: 58-lead rollouts (stride 2, dt 0.1, the test-case length) from two
windows per training simulation of train.nc, MSE in physical units (x1e-3) for each sigma of a
grid; the minimiser is written to RUN/mix_sigma.json. No test file is read.

    python calibrate_mix_sigma.py RUN_DIR [--ckpt last.pth] [--device cuda]
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
    return float(((denormalise(pred) - denormalise(tgt)) ** 2).mean() * 1e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="last.pth")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    run = Path(a.run_dir)
    m, _ = load_model(run, a.ckpt, dev)
    tw = TraWindows(str(REPO / "data/acdm/128_tra/train.nc"), None, dev)
    win, tgt, taus, ma = tw.fixed_windows(m.cfg.n_frames, K, 2, STARTS)
    mse = [mixture_mse(m, win, tgt, taus, ma, tw.mask, s) for s in GRID]
    best = GRID[min(range(len(GRID)), key=mse.__getitem__)]
    out = {
        "sigma": best,
        "grid": GRID,
        "train_mse": mse,
        "ckpt": a.ckpt,
        "windows": {"n": len(win), "starts_raw": STARTS, "leads": K, "stride": 2},
        "note": "calibrated on train.nc rollouts only; no test data",
    }
    (run / "mix_sigma.json").write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {"run": run.name, "sigma": best, "train_mse": [round(v, 3) for v in mse]}
        )
    )


if __name__ == "__main__":
    main()
