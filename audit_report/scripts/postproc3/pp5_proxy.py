"""Post-processing round 5: fitted combinations (superensemble / MOS / STEPS cascade weights).

Same proxies, windows, members and metrics as pp4_proxy.py. Members (anomalies about the
Mach-interpolated training mean M): single, mirrored single, plain mixture mean, mirrored
mixture mean. Weights are fitted by least squares on the held-out windows with
leave-one-simulation-out cross-validation (each simulation is scored with weights fitted on the
other six), separately per set (short, long) and per lead third.
Candidates
  stack(band)    per (channel, radial wavenumber) weights over the 4 members: spectral
                 superensemble (Krishnamurti et al. 1999) / fitted STEPS cascade blend
                 (Bowler et al. 2006); includes MOS-type damping (Glahn & Lowry 1972)
  stack(global)  one weight vector per channel (classic superensemble / stacking)
  stack(band)+sharp  stack(band) re-sharpened to the single-decode band power (round-3 regain,
                 gain in [1, 5]): fitted low-k blend, member power at every k
  lpm(stack(band))   localized PMM of stack(band) onto the single decodes (round-4 lpm)
  single, mirror, sharp(mirror,a1,c5), lpm(mirror) for reference
Selection rule as pp3_proxy.py.

    python pp5_proxy.py RUN_DIR OUT.json [--mem_gb 3]
"""

import argparse
import json
import sys
import time
from pathlib import Path

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
from pp4_proxy import lpm  # noqa: E402

SIGMA = 0.04
H, W = 64, 128
NB = int(RB.max()) + 1
# rfft2 half-plane weights (Parseval)
CW = np.full(W // 2 + 1, 2.0)
CW[0] = CW[-1] = 1.0


def gram(X, y, lead_bins):
    """X [n, K, C, H, W] member anomalies, y [K, C, H, W] truth anomaly ->
    per (lead bin, channel, band) G [nl, C, NB, n, n] and r [nl, C, NB, n] (also global, band -1)
    """
    F = np.fft.rfft2(X) * np.sqrt(CW)
    Fy = np.fft.rfft2(y) * np.sqrt(CW)
    n, K, C = X.shape[:3]
    G = np.zeros((len(lead_bins), C, NB + 1, n, n))
    r = np.zeros((len(lead_bins), C, NB + 1, n))
    for li, (a, b) in enumerate(lead_bins):
        f, fy = F[:, a:b], Fy[a:b]
        for i in range(n):
            ri = np.real(f[i] * fy.conj())  # [k, C, H, Wr]
            for bnd in range(NB):
                r[li, :, bnd, i] = ri[..., RB == bnd].sum((0, -1))
            for j in range(i, n):
                gij = np.real(f[i] * f[j].conj())
                for bnd in range(NB):
                    G[li, :, bnd, i, j] = G[li, :, bnd, j, i] = gij[..., RB == bnd].sum(
                        (0, -1)
                    )
    G[:, :, NB] = G[:, :, :NB].sum(2)
    r[:, :, NB] = r[:, :, :NB].sum(2)
    return G, r


def solve(G, r, lam=1e-6):
    n = G.shape[-1]
    reg = lam * np.trace(G, axis1=-2, axis2=-1)[..., None, None] / n * np.eye(n)
    return np.linalg.solve(G + reg + 1e-30 * np.eye(n), r[..., None])[..., 0]


def apply(X, w, lead_bins, M, fl, glob):
    """w [nl, C, NB+1, n]"""
    F = np.fft.rfft2(X)
    out = np.zeros(X.shape[1:])
    for li, (a, b) in enumerate(lead_bins):
        # wk [n, C, H, Wr], broadcast over leads
        if glob:
            wk = np.moveaxis(w[li, :, NB], -1, 0)[:, :, None, None]
        else:
            wk = np.moveaxis(w[li][:, RB], -1, 0)
        out[a:b] = np.fft.irfft2((F[:, a:b] * wk[:, None]).sum(0), s=(H, W))
    return (M + out) * fl


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
    xs = trw.x[:, 100:].float().numpy()
    tmean, tma = xs.mean(1), trw.ma.numpy()
    order = np.argsort(tma)
    tmean, tma = tmean[order], tma[order]
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
    T0 = time.time()
    res: dict = {"sigma": SIGMA, "sets": {}}
    for name, K, starts in [
        ("short", 58, [200, 320, 500, 620]),
        ("long", 238, [0, 480]),
    ]:
        win, tgt, _, ma = va.fixed_windows(2, K, 2, starts)
        N = len(win)
        lb = [(0, K // 3), (K // 3, 2 * K // 3), (2 * K // 3, K)]
        taus = 0.1 * torch.arange(1, K + 1, device=dev, dtype=torch.float32)[None]
        sim = np.round(ma.numpy(), 4)
        mem, Ms, grams = [], [], []
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
            M = interp(tmean, float(ma[i]))[None].astype(np.float64) * fl
            X = np.stack(
                [
                    Pn[c0],
                    Mn[c0],
                    sum(w * p for w, p in zip(ws, Pn)),
                    sum(w * p for w, p in zip(ws, Mn)),
                ]
            )
            X = (X - M) * fl
            y = (tgt[i].numpy().astype(np.float64) - M) * fl
            grams.append(gram(X, y, lb))
            mem.append(X.astype(np.float32))
            Ms.append(M)
            print(name, "window", i, f"{time.time() - T0:.0f}s", flush=True)
        cands: dict = {}
        wts: dict = {}
        for i in range(N):
            tr = [j for j in range(N) if sim[j] != sim[i]]
            G = sum(grams[j][0] for j in tr)
            r = sum(grams[j][1] for j in tr)
            w = solve(G, r)
            wts.setdefault(str(sim[i]), w[:, :, NB].tolist())
            X, M = mem[i].astype(np.float64), Ms[i]
            single, smir = M + X[0], M + X[1]
            mirr = 0.5 * (single + smir)
            e_single = 0.5 * (power(single, M) + power(smir, M))

            def sharp(A):
                return (
                    regain(
                        A,
                        M,
                        np.clip(
                            np.sqrt(e_single / np.maximum(power(A, M), 1e-30)), 1.0, 5.0
                        ),
                    )
                    * fl
                )

            sb = apply(X, w, lb, M, fl, glob=False)
            c = {
                "single": single * fl,
                "mirror": mirr * fl,
                "sharp(mirror,a1,c5)": sharp(mirr),
                "lpm(mirror)": lpm(mirr * fl, [single * fl, smir * fl], fl),
                "stack(global)": apply(X, w, lb, M, fl, glob=True),
                "stack(band)": sb,
                "stack(band)+sharp": sharp(sb),
                "lpm(stack(band))": lpm(sb, [single * fl, smir * fl], fl),
            }
            for k, x in c.items():
                cands.setdefault(k, []).append(x.astype(np.float32))
        del mem, grams
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
        res["sets"][name] = {
            "ma": ma.numpy().tolist(),
            "stats": stats,
            "global_weights": wts,
        }
        del cands
        json.dump(res, open(a.out, "w"))
    print("saved", a.out, f"{time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
