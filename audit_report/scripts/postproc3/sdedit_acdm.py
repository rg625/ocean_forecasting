"""SDEdit refinement (Meng et al. 2022) of CT-KAE test forecasts with the trained ACDM prior.

ACDM (Kohl et al. 2023, direct-ddpm+Prev, 20-step linear schedule, 'noisy' conditioning) predicts
frame i from frames i-2, i-1. Here every forecast frame i >= 2 is noised to diffusion step t0,
x_t0 = sqrt(ab_t0) x + sqrt(1 - ab_t0) eps, and denoised by the ACDM reverse process from t0 to 0,
conditioned on the (unrefined) forecast frames i-2 and i-1; all frames in parallel, so no
autoregressive drift. Obstacle cells are filled from the last input frame (known at forecast
time); the Mach channel is reset after denoising. One ACDM seed (00), one sample.
t0 is a fixed grid reported as a curve (ACDM saw the proxies' held-out Mach in training, so no
held-out selection is possible).

    python sdedit_acdm.py FIELDS_DIR OUT_DIR NAME [NAME ...] [--t0 1 2 3 5]
writes OUT_DIR/<case>/<NAME>@sde<t0>.npz (ACDM layout, physical, like FIELDS_DIR) and
OUT_DIR/sdedit_scores.json (MSE x1e-3, LSiM x1e-1, |v| spectrum band metrics)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ACDM = Path("/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion")
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ACDM / "src"))
from benchmark_suite import SAMPLING, load_gt, score, train_climatology  # noqa: E402
from lsim_score import lsim  # noqa: E402
from turbpred.model import PredictionModel  # noqa: E402

from pp3_proxy import spec_metrics  # noqa: E402

CASES = ["extrap", "interp", "longer"]
MEAN = np.array([0.560642, -0.000129, 0.903352, 0.637941, 0.7], np.float32)
STD = np.array([0.216987, 0.216987, 0.145391, 0.119944, 0.118322], np.float32)
dev = torch.device("cuda")


@torch.no_grad()
def refine(dm, x, t0, gen, batch=64):
    """x [N, T, 5, H, W] normalised; returns refined copy (frames >= 2)"""
    out = x.clone()
    N, T = x.shape[:2]
    idx = [(n, i) for n in range(N) for i in range(2, T)]
    for b in range(0, len(idx), batch):
        nb, ib = (torch.tensor(v) for v in zip(*idx[b : b + batch]))
        cond = torch.cat([x[nb, ib - 2], x[nb, ib - 1]], 1).to(dev)
        d = x[nb, ib].to(dev)
        cnoise = torch.randn(cond.shape, generator=gen, device=dev)
        d = dm.sqrtAlphasCumprod[t0] * d + dm.sqrtOneMinusAlphasCumprod[
            t0
        ] * torch.randn(d.shape, generator=gen, device=dev)
        for i in reversed(range(0, t0 + 1)):
            t = torch.full((len(d),), i, device=dev, dtype=torch.long)
            cn = (
                dm.sqrtAlphasCumprod[t] * cond
                + dm.sqrtOneMinusAlphasCumprod[t] * cnoise
            )
            dn = torch.cat([cn, d], 1)
            eps = dm.unet(dn, t)
            mean = dm.sqrtRecipAlphas[t] * (
                dn - dm.betas[t] * eps / dm.sqrtOneMinusAlphasCumprod[t]
            )
            d = mean[:, cond.shape[1] :]
            if i > 0:
                d = d + dm.sqrtPosteriorVariance[t] * torch.randn(
                    d.shape, generator=gen, device=dev
                )
        out[nb, ib] = d.cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fields")
    ap.add_argument("out")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--t0", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--cases", nargs="+", default=CASES)
    a = ap.parse_args()
    fields, out = Path(a.fields), Path(a.out)
    model = PredictionModel.load(
        str(ACDM / "pretrained_models/models_tra/128_acdm-r20_00/Model.pth"),
        useGPU=True,
    )
    model.eval()
    dm = model.modelDecoder
    clim = train_climatology()
    sp = out / "sdedit_scores.json"
    res = json.loads(sp.read_text()) if sp.exists() else {}
    for case in a.cases:
        gt, fluid = load_gt(case)
        raw = np.asarray(
            torch.load(SAMPLING / case / "groundTruth.dict", weights_only=False)["data"]
        )
        obst = ~fluid
        for name in a.names:
            x = np.load(fields / case / f"{name}.npz")["arr_0"][0, 0].astype(
                np.float32
            )  # [S, T, 5, H, W]
            x[:, :, :, obst] = raw[
                :, 1:2, :, obst
            ]  # obstacle cells from the last input frame
            xn = torch.from_numpy((x - MEAN[:, None, None]) / STD[:, None, None])
            for t0 in [0] + a.t0:
                if t0 == 0:
                    y = x
                else:
                    gen = torch.Generator(device=dev).manual_seed(t0)
                    y = (
                        refine(dm, xn, t0, gen).numpy() * STD[:, None, None]
                        + MEAN[:, None, None]
                    )
                    y[:, :, 4] = x[:, :, 4]
                    (out / case).mkdir(parents=True, exist_ok=True)
                    np.savez(
                        out / case / f"{name}@sde{t0}.npz",
                        arr_0=y[None, None].astype(np.float16),
                    )
                p = y[None, :, :, :4]
                r = {
                    "mse_all": float(score(p, gt, fluid, clim)["mse_all"]),
                    "lsim": float(lsim(p, case)[0]),
                    **spec_metrics(gt, p[0]),
                }
                res.setdefault(case, {})[f"{name}@sde{t0}"] = r
                print(
                    case,
                    f"{name}@sde{t0}",
                    " ".join(f"{k}={v:.3f}" for k, v in r.items()),
                    flush=True,
                )
                json.dump(res, open(sp, "w"), indent=1)


if __name__ == "__main__":
    main()
