"""Post-processing round 3: held-out validation of non-blurring averages on the proxy checkpoints.

Proxy = CT-KAE trained with Mach 0.53-0.54 and 0.73-0.77 held out (PX_<model>_s0_*). Its 7
held-out simulations of train.nc stand in for test, windowed as in pp2_lab.py: short = 28
windows x 58 leads (raw starts 200/320/500/620), long = 7 windows x 238 leads (starts 0/480),
2 input frames, dt 0.1. One seed per proxy, so seed averaging is not tested here; the form of
the post-processing is.

Members (sigma = 0.04, the modal train-calibrated width of the full checkpoints; 5 Gauss-Hermite
nodes; mirror = y-flip with v_y -> -v_y): node decodes P_q, mirrored node decodes M_q, and the
latent average D(sum_q w_q z_q) (plain and mirrored).
Candidates
  single         P at s = 1                         mirror       (P + M) / 2 at s = 1
  mix            sum_q w_q P_q                      mix+mirror   the adopted average (one seed)
  latent         D(sum_q w_q z_q)                   latent+mirror
  sharp(A,a,c)   A with its anomaly about the train climatology rescaled per (frame, channel,
                 radial wavenumber) by g = clip((E_single / E_A)^(a/2), 1, c), phases of A kept;
                 E_single = mean band power of the two single decodes (plain, mirrored).
                 A in {mirror, mix+mirror}; a in {0.5, 1}; c in {2, 5}
  ampmean(A)     per Fourier mode of the anomaly: weighted mean of the member amplitudes with
                 the phase of the weighted mean (members: 2 singles | 10 node decodes)
  qmap           mix+mirror, values remapped per (window, lead, channel) over fluid cells to the
                 pooled quantiles of the two single decodes (rank kept)
Selection rule (fixed before scoring): admissible = paired mean LSiM <= single and mid/high band
log10 power ratio no lower than single - 0.05, on both proxies and both sets; among those, the
lowest mean MSE ratio to single. Within 1 SE, the cheaper one.
Metrics as on test (pp3_lab.py): all-frames MSE x1e-3 in the ACDM layout (inputs included),
LSiM x1e-1, |v| spectrum band errors and signed mean log10 power ratio, mid-band slope.
Per-window MSE and LSiM are saved for paired SE over simulations.

    python pp3_proxy.py RUN_DIR OUT.json [--mem_gb 3]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "postproc2"))
sys.path.insert(0, "/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion/src")
from pp2_lab import REPO, decode, dev, mirror  # noqa: E402

from eval_ctkae import load_model  # noqa: E402
from lsim_score import _model as lsim_model  # noqa: E402
from models.ct_data import TraWindows, to_acdm_layout, train_val_intervals  # noqa: E402
from models.ct_kae import mixture_nodes  # noqa: E402
from spectral_analysis import (  # noqa: E402
    BANDS,
    fitted_slope,
    spectrum,
    velocity_magnitude,
)
from turbpred.loss import loss_lsim  # noqa: E402

SIGMA = 0.04
H, W = 64, 128
SHARP = [(a, c) for a in (0.5, 1.0) for c in (2.0, 5.0)]


def radial_onehot():
    ky = np.fft.fftfreq(H, d=1.0 / H)
    kx = np.arange(W // 2 + 1)
    rb = np.sqrt(ky[:, None] ** 2 + kx[None] ** 2).round().astype(int)
    return rb, np.eye(rb.max() + 1)[rb.ravel()]  # [H, Wr], [H*Wr, NB]


RB, ONEHOT = radial_onehot()


def power(x, M):
    """band power of the anomaly: [..., C, NB]"""
    F = np.abs(np.fft.rfft2(x - M)) ** 2
    return F.reshape(*F.shape[:-2], -1) @ ONEHOT


def regain(x, M, g):
    return M + np.fft.irfft2(np.fft.rfft2(x - M) * g[..., RB], s=(H, W))


def ampmean(xs, ws, M):
    F = [np.fft.rfft2(x - M) for x in xs]
    amp = sum(w * np.abs(f) for w, f in zip(ws, F))
    ph = np.angle(sum(w * f for w, f in zip(ws, F)))
    return M + np.fft.irfft2(amp * np.exp(1j * ph), s=(H, W))


def qmap(a, singles, fluid):
    """rank-preserving remap of a [K, C, H, W] onto the pooled single-decode quantiles"""
    out = a.copy()
    va = a[..., fluid]  # [K, C, F]
    pool = np.sort(
        np.concatenate([s[..., fluid] for s in singles], -1), -1
    )  # [K, C, nF]
    n = len(singles)
    rank = va.argsort(-1).argsort(-1)
    # quantile (r + 0.5) / F of the pooled sample -> mean of its n nearest order statistics
    idx = rank[..., None] * n + np.arange(n)
    out[..., fluid] = (
        np.take_along_axis(pool, idx.reshape(*idx.shape[:-2], -1), -1)
        .reshape(idx.shape)
        .mean(-1)
    )
    return out


def spec_metrics(gt, p):
    """|v| spectrum metrics; gt is obstacle-zeroed and p is masked the same way (the CT-KAE
    fields carry the normalisation mean in the obstacle, a spurious step at its edge)"""
    T = gt.shape[1]
    fluid = (np.abs(gt).sum((0, 1, 2)) > 0).astype(np.float32)
    k, sg = spectrum(velocity_magnitude(torch.from_numpy(gt[:, 2:T])))
    _, sp = spectrum(velocity_magnitude(torch.from_numpy(p[:, 2:T] * fluid)))
    n = min(len(sg), len(sp))
    k, sg, sp = k[:n], sg[:n], sp[:n]
    out = {}
    for name, (lo, hi) in BANDS.items():
        sel = (k >= lo) & (k <= hi)
        out[f"err_{name}"] = float((np.abs(sp - sg) / sg)[sel].mean())
        out[f"log_{name}"] = float(np.log10(sp / sg)[sel].mean())
    out["slope"] = fitted_slope(k, sp)
    out["slope_gt"] = fitted_slope(k, sg)
    return out


@torch.no_grad()
def lsim_win(gt, p, fluid):
    vals = []
    for s in range(len(gt)):
        pp = torch.from_numpy(np.nan_to_num(p[s]) * fluid).cuda()[None]
        g = torch.from_numpy(gt[s]).cuda()[None]
        vals.append(10 * float(loss_lsim(lsim_model(), g, pp).mean()))
    return vals


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
    tmean, tma = trw.x[:, 100:].float().mean(1).numpy(), trw.ma.numpy()
    order = np.argsort(tma)
    tmean, tma = tmean[order], tma[order]
    del trw

    def clim_at(v):
        j = int(np.clip(np.searchsorted(tma, v), 1, len(tma) - 1))
        w = float(np.clip((v - tma[j - 1]) / (tma[j] - tma[j - 1]), 0, 1))
        return (1 - w) * tmean[j - 1] + w * tmean[j]

    va = TraWindows(
        str(REPO / "data/acdm/128_tra/train.nc"), va_int, torch.device("cpu")
    )
    mask = va.mask.to(dev)
    mmask = mask.flip(-2)
    fl = va.mask.numpy() > 0.5  # [H, W]
    fluid = fl.T.astype(np.float32)  # ACDM layout [128, 64]
    nodes = mixture_nodes(SIGMA)
    ws = [w for _, w in nodes]
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
                P = [decode(m, z, mai, mask)[0].cpu().numpy() for z in zp]
                Mi = [
                    (mirror(decode(m, z, mai, mmask)) * mask)[0].cpu().numpy()
                    for z in zm
                ]
                lat = (
                    decode(m, sum(w * z for w, z in zip(ws, zp)), mai, mask)[0]
                    .cpu()
                    .numpy()
                )
                latm = (
                    (
                        mirror(
                            decode(m, sum(w * z for w, z in zip(ws, zm)), mai, mmask)
                        )
                        * mask
                    )[0]
                    .cpu()
                    .numpy()
                )
            P = [p.astype(np.float64) for p in P]
            Mi = [p.astype(np.float64) for p in Mi]
            c0 = nodes.index(min(nodes, key=lambda n: abs(n[0] - 1.0)))
            single, smir = P[c0], Mi[c0]
            mix = sum(w * p for w, p in zip(ws, P))
            mixm = 0.5 * (mix + sum(w * p for w, p in zip(ws, Mi)))
            mirr = 0.5 * (single + smir)
            M = clim_at(float(ma[i]))[None].astype(np.float64) * fl
            e_single = 0.5 * (power(single, M) + power(smir, M))
            c = {
                "single": single,
                "mirror": mirr,
                "mix": mix,
                "mix+mirror": mixm,
                "latent": lat,
                "latent+mirror": 0.5 * (lat + latm),
            }
            for an, A in [("mirror", mirr), ("mix+mirror", mixm)]:
                ratio = e_single / np.maximum(power(A, M), 1e-30)
                for al, cap in SHARP:
                    g = np.clip(ratio ** (al / 2), 1.0, cap)
                    c[f"sharp({an},a{al:g},c{cap:g})"] = regain(A, M, g) * fl
            c["ampmean(mirror)"] = ampmean([single, smir], [0.5, 0.5], M) * fl
            c["ampmean(mix+mirror)"] = ampmean(P + Mi, [w / 2 for w in ws] * 2, M) * fl
            c["qmap(mix+mirror)"] = qmap(mixm, [single, smir], fl)
            for k, v in c.items():
                cands.setdefault(k, []).append(v.astype(np.float32))
            print(name, "window", i, f"{time.time() - T0:.0f}s", flush=True)
        manp = ma.numpy()
        gt = to_acdm_layout(torch.cat([win, tgt], 1), ma)[:, :, :4] * fluid
        stats = {}
        for k, v in cands.items():
            p = to_acdm_layout(torch.cat([win, torch.from_numpy(np.stack(v))], 1), ma)[
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
                f"{k:30s} mse={r['mse']:.3f} lsim={r['lsim']:.3f} "
                + " ".join(
                    f"{q}={r[q]:.3f}"
                    for q in ["log_low", "log_mid", "log_high", "slope"]
                ),
                flush=True,
            )
        res["sets"][name] = {"ma": manp.tolist(), "stats": stats}
        del cands
        json.dump(res, open(a.out, "w"))
    print("saved", a.out, f"{time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
