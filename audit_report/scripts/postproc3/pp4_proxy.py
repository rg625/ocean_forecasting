"""Post-processing round 4: literature families, held-out validation on the proxy checkpoints.

Same proxies, windows, members (sigma = 0.04, 5 nodes, mirror) and metrics as pp3_proxy.py.
x* = mix+mirror (the MSE-best average), anchor M = Mach-interpolated training mean.
Candidates
  pdot(A,a)      perception-distortion transport (Blau & Michaeli 2018; Freirich et al. 2021):
                 x = (1 - a) A + a T(A), T rescales the anomaly per (frame, channel, radial
                 wavenumber) to the TRUTH anomaly power (training simulations, Mach-interpolated;
                 no forecasts used); a in {0.25, 0.5, 0.75, 1}
  noise(A,tgt)   stochastic sharpening (STEPS / S-PROG style noise injection): add a random-phase
                 field carrying the missing band power max(0, E_tgt - E_A); tgt = single | truth
  flow(A)        field-alignment averaging (Ravela et al. 2007): members warped onto the single
                 decode by dense optical flow (Farneback, guided by pressure), averaged there, the
                 average moved by the mean displacement. A = mirror (2 singles) | mix+mirror (10)
  lpm(A)         localized probability-matched mean (Ebert 2001; Clark 2017): rank-preserving
                 remap of A onto the pooled single-decode values in 16x16 patches (stride 8,
                 Hann-weighted overlap)
  blend(A)       member-by-member style: 0.5 A + 0.5 single
  nudge(kc)      scale-selective blend (spectral nudging, von Storch et al. 2000; STEPS cascade
                 blending, Bowler et al. 2006): mix+mirror for radial k <= kc, sharp(mirror) above;
                 kc in {2, 4} (a priori)
  lpm+sharp      lpm(mix+mirror) re-sharpened to the single-decode power; sharp+lpm the reverse
Spectral metrics mask the prediction like the truth (pp3_proxy.spec_metrics).
  sharp(..)      round-3 winner sharp(mirror,a1,c5) and sharp(flow(mirror)) for reference
Selection rule as pp3_proxy.py (fixed before scoring).

    python pp4_proxy.py RUN_DIR OUT.json [--mem_gb 3]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from pp3_proxy import (  # noqa: E402
    REPO,
    RB,
    decode,
    dev,
    lsim_win,
    load_model,
    mirror,
    mixture_nodes,
    power,
    regain,
    spec_metrics,
    to_acdm_layout,
    train_val_intervals,
    TraWindows,
)

SIGMA = 0.04
H, W = 64, 128
ALPHAS = (0.25, 0.5, 0.75, 1.0)
P = 2  # pressure channel (VARS order v_x, v_y, p, rho)


def pdot(A, M, e_tgt, a):
    g = np.sqrt(e_tgt / np.maximum(power(A, M), 1e-30))
    return regain(A, M, (1 - a) + a * g)


def noise_inject(A, M, e_tgt, rng):
    """add random-phase anomaly with band power max(0, e_tgt - E_A)"""
    miss = np.maximum(e_tgt - power(A, M), 0.0)  # [K, C, NB]
    z = np.fft.rfft2(rng.standard_normal(A.shape))
    zp = np.abs(z) ** 2
    pz = np.zeros(miss.shape)
    for b in range(miss.shape[-1]):
        pz[..., b] = zp[..., RB == b].sum(-1)
    g = np.sqrt(miss / np.maximum(pz, 1e-30))
    return A + np.fft.irfft2(z * g[..., RB], s=(H, W))


def _flow(ref, mov):
    """dense flow f with ref(p) ~ mov(p + f(p)) (pixels), Farneback on a joint 8-bit scaling"""
    lo, hi = min(ref.min(), mov.min()), max(ref.max(), mov.max())
    sc = 255.0 / max(hi - lo, 1e-12)
    a = ((ref - lo) * sc).astype(np.uint8)
    b = ((mov - lo) * sc).astype(np.uint8)
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.1, 0)


def _warp(x, f):
    """x [C, H, W] sampled at p + f(p)"""
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    mx, my = gx + f[..., 0], gy + f[..., 1]
    return np.stack(
        [
            cv2.remap(
                c.astype(np.float32),
                mx,
                my,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            for c in x
        ]
    )


def flow_average(members, ws, ref, fl):
    """members [n][K, C, H, W]; average after alignment onto ref, moved by the mean flow"""
    out = np.empty_like(ref)
    for k in range(ref.shape[0]):
        acc, fbar = 0.0, 0.0
        for m, w in zip(members, ws):
            f = _flow(ref[k, P], m[k, P])
            acc = acc + w * _warp(m[k], f)
            fbar = fbar + w * f
        out[k] = _warp(acc, -fbar)  # approximate forward move by the mean displacement
    return out * fl


def lpm(a, singles, fl, ps=16, st=8):
    """localized PMM over patches, Hann-weighted"""
    win = np.outer(np.hanning(ps + 2)[1:-1], np.hanning(ps + 2)[1:-1])
    num = np.zeros_like(a)
    den = np.zeros((H, W))
    n = len(singles)
    for i in range(0, H - ps + 1, st):
        for j in range(0, W - ps + 1, st):
            sl = (..., slice(i, i + ps), slice(j, j + ps))
            m = fl[sl[1:]]
            if m.sum() < 8:
                continue
            va = a[sl][..., m]  # [K, C, F]
            pool = np.sort(np.concatenate([s[sl][..., m] for s in singles], -1), -1)
            rank = va.argsort(-1).argsort(-1)
            idx = rank[..., None] * n + np.arange(n)
            mapped = (
                np.take_along_axis(pool, idx.reshape(*idx.shape[:-2], -1), -1)
                .reshape(idx.shape)
                .mean(-1)
            )
            patch = np.zeros(a.shape[:-2] + (ps, ps))
            patch[..., m] = mapped
            num[sl] += patch * win
            den[i : i + ps, j : j + ps] += win * m
    out = np.where(den > 0, num / np.maximum(den, 1e-12), a)
    return out * fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out")
    ap.add_argument("--mem_gb", type=float, default=3.0)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(
        a.mem_gb / (torch.cuda.get_device_properties(0).total_memory / 2**30)
    )
    run = Path(a.run_dir)
    m, _ = load_model(run, "last.pth", dev)
    for p in m.parameters():
        p.requires_grad_(False)
    val_ma = json.load(open(run / "run_metadata.json"))["config"]["data"]["val_ma"]
    tr_int, va_int = train_val_intervals(val_ma)
    trw = TraWindows(
        str(REPO / "data/acdm/128_tra/train.nc"), tr_int, torch.device("cpu")
    )
    xs = trw.x[:, 100:].float().numpy()  # [S, T, C, H, W]
    tmean, tma = xs.mean(1), trw.ma.numpy()
    # truth anomaly band power per training simulation (about its own time mean)
    tpow = np.stack([power(xs[s, ::4], tmean[s][None]).mean(0) for s in range(len(xs))])
    order = np.argsort(tma)
    tmean, tpow, tma = tmean[order], tpow[order], tma[order]
    del trw, xs

    def interp(arr, v):
        j = int(np.clip(np.searchsorted(tma, v), 1, len(tma) - 1))
        w = float(np.clip((v - tma[j - 1]) / (tma[j] - tma[j - 1]), 0, 1))
        return (1 - w) * arr[j - 1] + w * arr[j]

    va = TraWindows(
        str(REPO / "data/acdm/128_tra/train.nc"), va_int, torch.device("cpu")
    )
    mask = va.mask.to(dev)
    mmask = mask.flip(-2)
    fl = va.mask.numpy() > 0.5
    fluid = fl.T.astype(np.float32)
    nodes = mixture_nodes(SIGMA)
    ws = [w for _, w in nodes]
    c0 = min(range(len(nodes)), key=lambda i: abs(nodes[i][0] - 1.0))
    rng = np.random.default_rng(0)
    T0 = time.time()
    res: dict = {"sigma": SIGMA, "sets": {}}
    for name, K, starts in [
        ("short", 58, [200, 320, 500, 620]),
        ("long", 238, [0, 480]),
    ]:
        win, tgt, _, ma = va.fixed_windows(2, K, 2, starts)
        N = len(win)
        taus = 0.1 * torch.arange(1, K + 1, device=dev, dtype=torch.float32)[None]
        cands: dict = {}
        for i in range(N):
            x2, mai = win[i : i + 1].to(dev), ma[i : i + 1].to(dev)
            with torch.no_grad():
                zp = [
                    m.forecast(m.encode(x2, mask, mai), taus, mai, s) for s, _ in nodes
                ]
                zm = [
                    m.forecast(m.encode(mirror(x2), mmask, mai), taus, mai, s)
                    for s, _ in nodes
                ]
                Pn = [
                    decode(m, z, mai, mask)[0].cpu().numpy().astype(np.float64)
                    for z in zp
                ]
                Mn = [
                    (mirror(decode(m, z, mai, mmask)) * mask)[0]
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                    for z in zm
                ]
            single, smir = Pn[c0], Mn[c0]
            mixm = 0.5 * (
                sum(w * p for w, p in zip(ws, Pn)) + sum(w * p for w, p in zip(ws, Mn))
            )
            mirr = 0.5 * (single + smir)
            v = float(ma[i])
            M = interp(tmean, v)[None].astype(np.float64) * fl
            e_truth = np.broadcast_to(interp(tpow, v)[None], (K,) + tpow.shape[1:])
            e_single = 0.5 * (power(single, M) + power(smir, M))
            g1 = np.clip(
                np.sqrt(e_single / np.maximum(power(mirr, M), 1e-30)), 1.0, 5.0
            )
            fmir = flow_average([single, smir], [0.5, 0.5], single, fl)
            c = {
                "single": single,
                "mirror": mirr,
                "mix+mirror": mixm,
                "sharp(mirror,a1,c5)": regain(mirr, M, g1) * fl,
                "flow(mirror)": fmir,
                "flow(mix+mirror)": flow_average(
                    Pn + Mn, [w / 2 for w in ws] * 2, single, fl
                ),
                "sharp(flow(mirror))": regain(
                    fmir,
                    M,
                    np.clip(
                        np.sqrt(e_single / np.maximum(power(fmir, M), 1e-30)), 1.0, 5.0
                    ),
                )
                * fl,
                "noise(mix+mirror,single)": noise_inject(mixm, M, e_single, rng) * fl,
                "noise(mix+mirror,truth)": noise_inject(mixm, M, e_truth, rng) * fl,
                "noise(mirror,single)": noise_inject(mirr, M, e_single, rng) * fl,
                "lpm(mix+mirror)": lpm(mixm, [single, smir], fl),
                "lpm(mirror)": lpm(mirr, [single, smir], fl),
                "blend(mix+mirror)": 0.5 * (mixm + single),
            }
            for kc in (2, 4):
                lo = (RB <= kc).astype(np.float64)
                Fm = np.fft.rfft2(mixm - M)
                Fs = np.fft.rfft2(c["sharp(mirror,a1,c5)"] - M)
                c[f"nudge(k{kc})"] = (
                    M + np.fft.irfft2(lo * Fm + (1 - lo) * Fs, s=(H, W))
                ) * fl
            lp = c["lpm(mix+mirror)"]
            c["lpm+sharp"] = (
                regain(
                    lp,
                    M,
                    np.clip(
                        np.sqrt(e_single / np.maximum(power(lp, M), 1e-30)), 1.0, 5.0
                    ),
                )
                * fl
            )
            c["sharp+lpm"] = lpm(c["sharp(mirror,a1,c5)"], [single, smir], fl)
            for al in ALPHAS:
                c[f"pdot(mix+mirror,a{al:g})"] = pdot(mixm, M, e_truth, al) * fl
            for al in (0.5, 1.0):
                c[f"pdot(mirror,a{al:g})"] = pdot(mirr, M, e_truth, al) * fl
            for k, x in c.items():
                cands.setdefault(k, []).append(x.astype(np.float32))
            print(name, "window", i, f"{time.time() - T0:.0f}s", flush=True)
        gt = to_acdm_layout(torch.cat([win, tgt], 1), ma)[:, :, :4] * fluid
        stats = {}
        for k, x in cands.items():
            p = to_acdm_layout(torch.cat([win, torch.from_numpy(np.stack(x))], 1), ma)[
                :, :, :4
            ]
            mse_w = (((p * fluid - gt) ** 2).mean((1, 2, 3, 4)) * 1e3).tolist()
            stats[k] = {
                "mse": float(np.mean(mse_w)),
                "mse_win": mse_w,
                "lsim_win": lsim_win(gt, p, fluid),
                **spec_metrics(gt, p),
            }
            stats[k]["lsim"] = float(np.mean(stats[k]["lsim_win"]))
            r = stats[k]
            print(
                name,
                f"{k:28s} mse={r['mse']:.3f} lsim={r['lsim']:.3f} "
                + " ".join(
                    f"{q}={r[q]:.3f}"
                    for q in ["log_low", "log_mid", "log_high", "slope"]
                ),
                flush=True,
            )
        res["sets"][name] = {"ma": ma.numpy().tolist(), "stats": stats}
        del cands
        json.dump(res, open(a.out, "w"))
    print("saved", a.out, f"{time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
