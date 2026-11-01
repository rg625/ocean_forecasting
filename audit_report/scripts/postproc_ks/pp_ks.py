"""The Tra post-processing readouts on Kuramoto-Sivashinsky (L = 22, 64 points, periodic).

Model: the KS Koopman autoencoder of the DA campaign (continuous generator, exact e^{K tau}),
one input frame; baseline: the autoregressive 1D U-Net. KS is symmetric under every periodic
shift and under the reflection u(x) -> -u(-x); both give extra members, as the y-flip does
on Tra.
Readouts (all at inference, no fitting):
  single         one decode
  mirror         average with the reflected forecast (2 decodes)
  sym8           average over 4 shifts (0, 16, 32, 48 points) x reflection (8)
  mix+mirror     mixture over time scales 1 + sigma xi (5 Gauss-Hermite nodes, sigma = 0.04)
                 with reflection (10): the Tra post-processing without seeds and shrinkage
  sharp(A)       anomaly of A rescaled per frame and wavenumber to the members' mean power,
                 gain in [1, 5] (postproc3/sharp.py)
  lpm(A)         localized probability-matched mean of A onto its members, periodic patches
                 of 16 points, stride 8, Hann overlap (postproc3: lpm)
Metrics per horizon (short: 50 frames = 0.22 T_L from 10 starts per simulation; long: 400 frames
= 1.8 T_L from 3 starts): MSE (normalised units), relative L2, and the power spectrum of u in
bands low (k <= 3), mid (4-6), high (7-10; above, the truth has < 1e-5 of the peak power): mean
log10 power ratio prediction / truth and
relative error. Selection on val.nc with the Tra rule (band power at most 0.05 dex below the
single decode in mid and high, on both horizons; then lowest MSE), scored on test.nc.

    python pp_ks.py OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

IMPL = Path(__file__).resolve().parents[3]
MAIN = Path("/home/rg625/mnt/ocean_forecasting/koopman_autoencoder")
sys.path.insert(0, str(IMPL / "koopman_autoencoder"))
sys.path.insert(0, str(IMPL / "audit_report/scripts/postproc3"))
from data_assimilation.ks.models_io import load_kae, load_unet  # noqa: E402
from data_assimilation.ks.protocol import KSData  # noqa: E402
from models.ct_kae import mixture_nodes  # noqa: E402

dev = torch.device("cuda")
SHIFTS = (0, 16, 32, 48)
SIGMA = 0.04
BANDS = {
    "low": (1, 3),
    "mid": (4, 6),
    "high": (7, 10),
}  # k > 10 carries < 1e-5 of the peak power
SETS = {
    "short": (50, list(range(2, 950, 100))),
    "long": (400, [2, 300, 590]),
}


def reflect(u):
    """u(x) -> -u(-x) on the periodic grid (physical units), last axis"""
    return -torch.flip(u, [-1])


class KAE:
    def __init__(self):
        self.m, self.K, _ = load_kae(
            MAIN / "model_outputs_ks/continous_linear_128/rollout_10", None, dev
        )

    @torch.no_grad()
    def forecast(self, x0n, taus):
        """x0n [B, X] normalised -> [F, B, X] normalised"""
        z0 = self.m.present_encoding(
            TensorDict({"u": x0n[:, None, :, None]}, batch_size=[len(x0n), 1]), None
        )
        out = []
        for t in taus:
            z = z0 @ torch.matrix_exp(self.K * float(t)).T
            out.append(self.m.decode(z)["u"].squeeze(-1))
        return torch.stack(out)


@torch.no_grad()
def unet_forecast(unet, x0n, n):
    x, out = x0n, []
    for _ in range(n):
        x = unet(x.unsqueeze(1)).squeeze(1)
        out.append(x)
    return torch.stack(out)


def power(u):
    """per-wavenumber power of u [..., X] -> [..., X//2+1]"""
    return torch.fft.rfft(u.double(), dim=-1).abs() ** 2


def sharpen(avg, e_mem, M, gmax=5.0):
    F = torch.fft.rfft((avg - M).double(), dim=-1)
    g = torch.clamp(torch.sqrt(e_mem / (F.abs() ** 2).clamp_min(1e-30)), 1.0, gmax)
    g[..., 0] = 1.0
    return (M + torch.fft.irfft(F * g, n=avg.shape[-1], dim=-1)).to(avg.dtype)


def lpm(avg, members, ps=16, st=8):
    """1D localized probability-matched mean, periodic patches"""
    X = avg.shape[-1]
    n = len(members)
    win = torch.hann_window(
        ps + 2, periodic=False, dtype=torch.float64, device=avg.device
    )[1:-1]
    num = torch.zeros_like(avg, dtype=torch.float64)
    den = torch.zeros(X, dtype=torch.float64, device=avg.device)
    ar = torch.arange(n, device=avg.device)
    for s in range(0, X, st):
        idx = (torch.arange(ps, device=avg.device) + s) % X
        a = avg[..., idx].double()
        pool = torch.cat([m[..., idx].double() for m in members], -1).sort(-1).values
        rank = a.argsort(-1).argsort(-1)
        mapped = (
            pool.gather(-1, (rank[..., None] * n + ar).flatten(-2))
            .unflatten(-1, (ps, n))
            .mean(-1)
        )
        num[..., idx] += mapped * win
        den[idx] += win
    return (num / den).to(avg.dtype)


def spec_metrics(gt, p):
    """band log ratio and relative error of the time-mean power spectrum; gt, p [F, B, X]"""
    sg, sp = power(gt).mean((0, 1)), power(p).mean((0, 1))
    out = {}
    for b, (lo, hi) in BANDS.items():
        r = torch.log10(sp[lo : hi + 1] / sg[lo : hi + 1])
        out[f"log_{b}"] = float(r.mean())
        out[f"err_{b}"] = float(
            ((sp[lo : hi + 1] - sg[lo : hi + 1]).abs() / sg[lo : hi + 1]).mean()
        )
    return out


def run(split, kae, unet):
    data = KSData(IMPL / f"koopman_autoencoder/data/ks/{split}.nc", dev)
    nodes = mixture_nodes(SIGMA)
    res = {}
    for name, (n, starts) in SETS.items():
        taus = 0.1 * np.arange(1, n + 1)
        sim = torch.arange(data.n_sim, device=dev).repeat_interleave(len(starts))
        t0 = torch.tensor(starts, device=dev).repeat(data.n_sim)
        x0 = data.frames(sim, t0)  # [B, X] normalised
        gt = torch.stack(
            [data.frames(sim, t0 + k) for k in range(1, n + 1)]
        )  # [F, B, X]
        M = torch.tensor(
            -data.mean / (data.std + 1e-8), device=dev
        )  # u = 0, normalised

        def norm(u):
            return (u - data.mean) / (data.std + 1e-8)

        def den(u):
            return data.denorm(u)

        mem = {}  # (shift, refl) -> forecast, normalised
        for s in SHIFTS:
            for r in (False, True):
                xp = torch.roll(den(x0), s, -1)
                xp = reflect(xp) if r else xp
                f = den(kae.forecast(norm(xp), taus))
                f = reflect(f) if r else f
                mem[(s, r)] = norm(torch.roll(f, -s, -1))
        mix = []
        for r in (False, True):
            acc = 0.0
            xp = reflect(den(x0)) if r else den(x0)
            for c, w in nodes:
                f = den(kae.forecast(norm(xp), taus * c))
                acc = acc + w * (reflect(f) if r else f)
            mix.append(norm(acc))
        single = mem[(0, False)]
        mir = [mem[(0, False)], mem[(0, True)]]
        sym = list(mem.values())
        cands = {
            "U-Net": unet_forecast(unet, x0, n),
            "persistence": x0[None].expand_as(gt),
            "single": single,
            "mirror": sum(mir) / 2,
            "sym8": sum(sym) / 8,
            "mix+mirror": sum(mix) / 2,
        }
        e2 = sum(power(m - M) for m in mir) / 2
        e8 = sum(power(m - M) for m in sym) / 8
        cands["sharp(mirror)"] = sharpen(cands["mirror"], e2, M)
        cands["sharp(sym8)"] = sharpen(cands["sym8"], e8, M)
        cands["lpm(mirror)"] = lpm(cands["mirror"], mir)
        cands["lpm(sym8)"] = lpm(cands["sym8"], sym)
        cands["sharp+lpm(sym8)"] = lpm(cands["sharp(sym8)"], sym)
        stats = {}
        for k, p in cands.items():
            d2 = ((p - gt) ** 2).mean(-1)  # [F, B]
            rel = torch.sqrt(((den(p) - den(gt)) ** 2).sum(-1) / (den(gt) ** 2).sum(-1))
            stats[k] = {
                "mse": float(d2.mean()),
                "mse_win": d2.mean(0).tolist(),
                "rel": float(rel.mean()),
                "mse_lead": d2.mean(1).tolist(),
                **spec_metrics(gt, p),
            }
            print(
                split,
                name,
                f"{k:18s}",
                " ".join(
                    f"{q}={v:.4f}"
                    for q, v in stats[k].items()
                    if not isinstance(v, list)
                ),
                flush=True,
            )
        res[name] = {"sim": sim.tolist(), "stats": stats}
    return res


def main():
    out = Path(sys.argv[1])
    kae = KAE()
    unet = load_unet(
        MAIN / "model_outputs_ks/unet1d/rollout10_extended2/best_model.pth", dev
    )
    res = {s: run(s, kae, unet) for s in ("val", "test")}
    json.dump(res, open(out, "w"))
    print("saved", out)


if __name__ == "__main__":
    main()
