"""Post-processing lab, round 2: honest calibration on held-out Mach numbers.

On a PROXY checkpoint (trained with data.val_ma held out) the held-out simulations of train.nc
play the role of test: windows shaped like the test cases (2 input frames; 58 leads from raw
starts 200/320/500/620, 238 leads from 0/480; dt 0.1). On a FULL checkpoint, --set test runs
the same variants on the real test cases (used only once the settings are fixed on the proxy).

Variants (each scored per lead):
  z0       encoder | refined: Adam on the two INPUT frames (decode at tau = -0.1, 0), lr x
           steps grid, no prior (early stopping is the regulariser)
  sigma    time-scale mixture width (0 = single decode)
  extras   Mach mixture (Gauss-Hermite over Ma +- delta in propagate/decode), mirror average
           (y-flip with v_y sign change, averaged with the plain forecast)
Shrinkage statistics per variant: per lead and channel, num = <p - M, t - M>, den = |p - M|^2,
anchor M = Mach-interpolated time-mean of the checkpoint's own training simulations.

    python pp2_lab.py RUN_DIR OUT.json [--set proxy|test]
"""

import argparse
import time
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder")
sys.path.insert(0, str(REPO))
from eval_ctkae import load_model  # noqa: E402
from models.ct_data import TraWindows, load_eval_case, train_val_intervals  # noqa: E402
from models.ct_kae import mixture_nodes  # noqa: E402
from models.dataloader import TRA_STD  # noqa: E402

VARS = ["v_x", "v_y", "p", "rho"]
dev = torch.device("cuda")
REFINE = [(0.05, [25, 50, 100, 200, 400]), (0.2, [25, 50, 100, 200, 400])]
SIGMAS = [0.0, 0.04, 0.06, 0.08, 0.10]
MA_DELTAS = [0.005, 0.01, 0.02]


@torch.no_grad()
def decode(m, z, ma, mask, chunk=32):
    """z [N, K, d] -> [N, K, C, H, W]"""
    N, K, d = z.shape
    flat, mk = z.reshape(-1, d), ma.repeat_interleave(K)
    out = []
    for i in range(0, len(flat), chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(m.decode(flat[i : i + chunk], mask, mk[i : i + chunk]).float())
    return torch.cat(out).view(N, K, *out[0].shape[1:]) * mask


@torch.no_grad()
def forecast(m, z0, taus, ma, mask, sigma, ma_delta=0.0):
    acc = 0.0
    mnodes = mixture_nodes(ma_delta / 1.0) if ma_delta > 0 else [(1.0, 1.0)]
    for mq, mw in mnodes:
        mac = (
            ma if ma_delta == 0 else ma + (mq - 1.0)
        )  # mixture_nodes(delta): 1 + delta*xi
        for sq, wq in mixture_nodes(sigma):
            z = m.forecast(z0, taus, mac, sq)
            acc = acc + mw * wq * decode(m, z, mac, mask)
    return acc


def rec_loss(m, z, x2, mask, ma):
    taus = torch.tensor([-0.1, 0.0], device=dev).expand(z.shape[0], -1)
    d = m.decode(m.forecast(z, taus, ma), mask, ma)
    return ((d - x2) ** 2 * mask).sum((-1, -2, -3)).mean(-1) / (mask.sum() * 4)


def refine_all(m, z0, x2, mask, ma):
    """-> {(lr, steps): z}"""
    out = {}
    for lr, steps in REFINE:
        z = z0.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        for i in range(1, max(steps) + 1):
            opt.zero_grad()
            rec_loss(m, z, x2, mask, ma).sum().backward()
            opt.step()
            if i in steps:
                out[(lr, i)] = z.detach().clone()
    return out


def mirror(x):
    y = x.flip(-2).clone()
    y[..., 1, :, :] *= -1
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out")
    ap.add_argument("--set", default="proxy", choices=["proxy", "test"])
    ap.add_argument("--mem_gb", type=float, default=3.0)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(
        a.mem_gb / (torch.cuda.get_device_properties(0).total_memory / 2**30)
    )
    run = Path(a.run_dir)
    m, _ = load_model(run, "last.pth", dev)
    for p in m.parameters():
        p.requires_grad_(False)
    cfg = json.load(open(run / "run_metadata.json"))["config"]
    std2 = torch.tensor([TRA_STD[v] ** 2 for v in VARS], device=dev)

    # climatology of the checkpoint's own training simulations
    val_ma = cfg["data"]["val_ma"]
    tr_int = train_val_intervals(val_ma)[0] if val_ma else None
    trw = TraWindows(
        str(REPO / "data/acdm/128_tra/train.nc"), tr_int, torch.device("cpu")
    )
    tmean, tma = trw.x[:, 100:].float().mean(1), trw.ma.cpu().numpy()
    order = np.argsort(tma)
    tmean, tma = tmean[order], tma[order]

    def clim_at(v):
        if v <= tma[0]:
            return tmean[0]
        if v >= tma[-1]:
            return tmean[-1]
        j = int(np.searchsorted(tma, v))
        w = (v - tma[j - 1]) / (tma[j] - tma[j - 1])
        return (1 - w) * tmean[j - 1] + w * tmean[j]

    # evaluation sets: name -> (x2 [N,2,C,H,W], tgt [N,K,C,H,W], ma [N], mask)
    T0 = time.time()
    print("loaded train/clim", flush=True)
    sets = {}
    if a.set == "proxy":
        va = TraWindows(
            str(REPO / "data/acdm/128_tra/train.nc"),
            train_val_intervals(val_ma)[1],
            torch.device("cpu"),
        )
        for name, K, starts in [
            ("short", 58, [200, 320, 500, 620]),
            ("long", 238, [0, 480]),
        ]:
            win, tgt, _, ma = va.fixed_windows(2, K, 2, starts)
            sets[name] = (win, tgt, ma, va.mask)
        del va
    else:
        for case in ["extrap", "interp", "longer"]:
            x, ma, mask = load_eval_case(
                str(REPO / f"data/acdm/128_tra/gt_{case}.nc"), torch.device("cpu")
            )
            sets[case] = (x[:, :2], x[:, 2:], ma, mask)

    res = {
        "sets": {},
        "config": {"refine": REFINE, "sigmas": SIGMAS, "ma_deltas": MA_DELTAS},
    }
    for name, (win, tgt, ma, mask) in sets.items():
        mask = mask.to(dev)
        N, K = tgt.shape[:2]
        taus = 0.1 * torch.arange(1, K + 1, device=dev, dtype=torch.float32)
        clim = torch.stack([clim_at(float(v)) for v in ma])  # [N, C, H, W]
        stats = {}

        def score(tag, fc_fn):
            """per-window per-lead MSE (phys) and shrink stats, accumulated over chunks"""
            se = torch.zeros(N, K, device=dev, dtype=torch.float64)
            num = torch.zeros(K, 4, device=dev, dtype=torch.float64)
            den = torch.zeros_like(num)
            for i in range(0, N, 2):
                sl = slice(i, i + 2)
                t_ = tgt[sl].to(dev).float() * mask
                p = fc_fn(sl)
                M = clim[sl].to(dev)[:, None] * mask
                se[sl] = (((p - t_) ** 2).mean((-1, -2)) * std2).mean(-1)
                num += ((p - M) * (t_ - M)).sum((-1, -2)).sum(0) * std2
                den += ((p - M) ** 2).sum((-1, -2)).sum(0) * std2
            stats[tag] = {
                "mse_lead": (se.mean(0) * 1e3).tolist(),
                "mse_win": (se.mean(1) * 1e3).tolist(),
                "num": num.tolist(),
                "den": den.tolist(),
            }
            print(name, tag, f"{se.mean().item() * 1e3:.3f}", flush=True)

        # z0 variants for all windows
        x2_all, ma_all = win.to(dev), ma.to(dev)
        with torch.no_grad():
            z_enc = m.encode(x2_all, mask, ma_all)
        zs = {"enc": z_enc}
        print(name, "sets ready", N, K, f"{time.time() - T0:.0f}s", flush=True)
        for i in range(0, N, 8):
            print(name, "refine chunk", i, f"{time.time() - T0:.0f}s", flush=True)
            sl = slice(i, i + 8)
            for k, z in refine_all(m, z_enc[sl], x2_all[sl], mask, ma_all[sl]).items():
                zs.setdefault(f"ref{k[0]:g}_{k[1]}", []).append(z)
        zs = {k: (v if torch.is_tensor(v) else torch.cat(v)) for k, v in zs.items()}
        with torch.no_grad():
            res.setdefault("input_rec", {})[name] = {
                k: rec_loss(m, z, x2_all, mask, ma_all).mean().item()
                for k, z in zs.items()
            }

        def tf(sl):
            return taus.expand(len(range(N)[sl]), -1)

        for zk, z in zs.items():
            for sg in SIGMAS:
                score(
                    f"{zk}|s{sg:g}",
                    lambda sl: forecast(m, z[sl], tf(sl), ma_all[sl], mask, sg),
                )
        for zk in ["enc", "ref0.2_100"]:
            z = zs[zk]
            for dl in MA_DELTAS:
                score(
                    f"{zk}|s0.06|ma{dl:g}",
                    lambda sl: forecast(m, z[sl], tf(sl), ma_all[sl], mask, 0.06, dl),
                )
        # mirror average (encoder z0 of the mirrored input)
        mmask = mask.flip(-2)
        with torch.no_grad():
            z_mir = m.encode(mirror(x2_all), mmask, ma_all)
        for sg in [0.0, 0.06]:
            score(
                f"mirror|s{sg:g}",
                lambda sl: 0.5
                * (
                    forecast(m, z_enc[sl], tf(sl), ma_all[sl], mask, sg)
                    + mirror(forecast(m, z_mir[sl], tf(sl), ma_all[sl], mmask, sg))
                    * mask
                ),
            )
        res["sets"][name] = {"ma": ma.tolist(), "stats": stats}
        json.dump(res, open(a.out, "w"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
