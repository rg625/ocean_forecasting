"""Post-processing lab for one CT-KAE checkpoint (research; scratchpad).

Pass 1 (TRAIN only, train.nc, 238-lead rollouts from raw starts 300/500, stride 2): for each
mixture width sigma in SIG, per-lead physical MSE and per-(lead, channel, radial-wavenumber)
Wiener statistics of the forecast anomaly about two anchors:
   self: the forecast's own mean over leads 1..58 (no extra data)
   clim: Mach-interpolated time-mean field of the training sims (leave-one-sim-out on train)
   num = sum Re(conj(F(xhat - M)) F(x - M)),  den = sum |F(xhat - M)|^2
Pass 2 (TEST cases): mixture forecasts for every sigma, saved fp16 for offline combination.

    python pp_lab.py RUN_DIR OUT.npz [--starts 300 500] [--mem_gb 1.6]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder")
sys.path.insert(0, str(REPO))
from eval_ctkae import load_model  # noqa: E402
from models.ct_data import TraWindows, load_eval_case  # noqa: E402
from models.ct_kae import mixture_nodes  # noqa: E402
from models.dataloader import TRA_STD  # noqa: E402

SIG = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]
VARS = ["v_x", "v_y", "p", "rho"]
K = 238
H, W = 64, 128


def radial_bins(dev):
    ky = torch.fft.fftfreq(H, d=1.0 / H, device=dev)
    kx = torch.arange(W // 2 + 1, device=dev, dtype=torch.float32)
    r = torch.sqrt(ky[:, None] ** 2 + kx[None] ** 2).round().long()
    return r, int(r.max()) + 1


@torch.no_grad()
def decode_all(m, zs, ma, mask, chunk=32):
    """zs [N, K, d] -> [N, K, C, H, W] (masked)"""
    N, Kk, d = zs.shape
    flat = zs.reshape(-1, d)
    mk = ma.repeat_interleave(Kk)
    out = []
    for i in range(0, len(flat), chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=flat.is_cuda):
            out.append(m.decode(flat[i : i + chunk], mask, mk[i : i + chunk]).float())
    return torch.cat(out).view(N, Kk, *out[0].shape[1:]) * mask


@torch.no_grad()
def mixtures(m, z0, taus, ma, mask):
    """Yields (sigma, forecast) for every width, sharing the centre node (s = 1); one
    forecast is alive at a time besides the centre."""
    centre = decode_all(m, m.forecast(z0, taus, ma, 1.0), ma, mask)
    for sg in SIG:
        if sg == 0:
            yield sg, centre
            continue
        acc = 0.0
        for sq, wq in mixture_nodes(sg):
            dec = (
                centre
                if abs(sq - 1.0) < 1e-12
                else decode_all(m, m.forecast(z0, taus, ma, sq), ma, mask)
            )
            acc = acc + wq * dec
            del dec
        yield sg, acc
        del acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out")
    ap.add_argument("--starts", type=int, nargs="+", default=[300, 500])
    ap.add_argument("--mem_gb", type=float, default=1.6)
    ap.add_argument("--chunk_windows", type=int, default=2)
    a = ap.parse_args()
    dev = torch.device("cuda")
    tot = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(a.mem_gb / tot)
    m, _ = load_model(Path(a.run_dir), "last.pth", dev)
    std2 = torch.tensor([TRA_STD[v] ** 2 for v in VARS], device=dev)
    rb, nb = radial_bins(dev)
    onehot = torch.nn.functional.one_hot(rb.flatten(), nb).float()  # [H*(W/2+1), nb]

    # ---------------- pass 1: train
    tw = TraWindows(str(REPO / "data/acdm/128_tra/train.nc"), None, torch.device("cpu"))
    S = tw.S
    tmean = tw.x[:, 100:].float().mean(1)  # [S, C, H, W] per-sim time mean
    tma = tw.ma.cpu().numpy()

    def clim_at(ma_val, exclude=None):
        idx = [i for i in range(S) if i != exclude]
        mas = tma[idx]
        order = np.argsort(mas)
        mas, idx = mas[order], [idx[i] for i in order]
        if ma_val <= mas[0]:
            return tmean[idx[0]]
        if ma_val >= mas[-1]:
            return tmean[idx[-1]]
        j = int(np.searchsorted(mas, ma_val))
        w = (ma_val - mas[j - 1]) / (mas[j] - mas[j - 1])
        return (1 - w) * tmean[idx[j - 1]] + w * tmean[idx[j]]

    win, tgt, taus, ma = tw.fixed_windows(m.cfg.n_frames, K, 2, a.starts)
    sim_of = np.repeat(np.arange(S), len(a.starts))  # fixed_windows order: sim-major
    mask = tw.mask.to(dev)
    nS = len(SIG)
    mse = torch.zeros(nS, K, device=dev, dtype=torch.float64)
    num = torch.zeros(2, nS, K, 4, nb, device=dev, dtype=torch.float64)
    den = torch.zeros_like(num)
    tvar = torch.zeros(2, K, 4, nb, device=dev, dtype=torch.float64)  # |F(x - M)|^2
    cnt = 0
    for i in range(0, len(win), a.chunk_windows):
        sl = slice(i, i + a.chunk_windows)
        w_, t_, ta_, m_ = (
            win[sl].to(dev),
            tgt[sl].to(dev).float() * mask,
            taus[sl].to(dev),
            ma[sl].to(dev),
        )
        z0 = m.encode(w_, mask, m_)
        clim = (
            torch.stack(
                [
                    clim_at(float(ma[j]), exclude=int(sim_of[j]))
                    for j in range(i, i + len(w_))
                ]
            ).to(dev)[:, None]
            * mask
        )
        for si, (sg, p) in enumerate(mixtures(m, z0, ta_, m_, mask)):
            mse[si] += (((p - t_) ** 2).mean(dim=(-1, -2)) * std2).mean(-1).sum(0)
            anchors = [p[:, :58].mean(1, keepdim=True), clim]
            for ai, M in enumerate(anchors):
                Fp = torch.fft.rfft2(p - M).flatten(-2)  # [n, K, C, F]
                Ft = torch.fft.rfft2(t_ - M).flatten(-2)
                num[ai, si] += ((Fp.conj() * Ft).real @ onehot).sum(0)
                den[ai, si] += ((Fp.abs() ** 2) @ onehot).sum(0)
                if si == 0:
                    tvar[ai] += ((Ft.abs() ** 2) @ onehot).sum(0)
        cnt += len(w_)
        print(f"train {cnt}/{len(win)}", flush=True)
    res = {
        "sig": np.array(SIG),
        "train_mse": (mse / cnt * 1e3).cpu().numpy(),
        "train_num": num.cpu().numpy(),
        "train_den": den.cpu().numpy(),
        "train_tvar": tvar.cpu().numpy(),
        "nbins": nb,
    }
    del tw, win, tgt

    # ---------------- pass 2: test cases
    for case in ["interp", "extrap", "longer"]:
        x, mac, maskc = load_eval_case(
            str(REPO / f"data/acdm/128_tra/gt_{case}.nc"), dev
        )
        T = x.shape[1]
        tt = 0.1 * torch.arange(1, T - 1, device=dev, dtype=torch.float32)
        z0 = m.encode(x[:, :2], maskc, mac)
        res[f"{case}_pred"] = np.stack(
            [
                p.cpu().numpy().astype(np.float16)
                for _, p in mixtures(m, z0, tt.expand(x.shape[0], -1), mac, maskc)
            ]
        )
        res[f"{case}_clim"] = np.stack(
            [(clim_at(float(v)).to(dev) * maskc).cpu().numpy() for v in mac]
        ).astype(np.float32)
        print(case, "done", flush=True)
    np.savez(a.out, **res)
    print("saved", a.out)


if __name__ == "__main__":
    main()
