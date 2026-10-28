"""Offline scoring of post-processing variants from pp_lab.py outputs (train-calibrated only).

Variants per checkpoint (all forecasts normalised; scored with benchmark_suite.score, mse_all
= the table's all-frames MSE):
  single       sigma = 0
  mix          sigma calibrated on 58-lead train rollouts (RUN/mix_sigma.json, the protocol)
  mixlead      per-lead sigma(tau): argmin of the train MSE (smoothed over 15 leads)
  +wself/wclim per-lead, per-channel, per-radial-wavenumber Wiener gain on the anomaly about
               the forecast's own lead-1..58 mean (self) or the Mach-interpolated training
               climatology (clim); gains = train num/den, clipped to [0, 1], smoothed over leads
  +sself/sclim same with one gain per lead and channel (no wavenumber dependence)
Ensembles: mean of a variant over the 3 seeds of a model, and over all 6 checkpoints.

    python pp_analyze.py [--lsim VARIANT ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

P = Path(__file__).parent
REPO = Path("/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / "audit_report/scripts"))
from benchmark_suite import load_gt, score, train_climatology  # noqa: E402
from models.ct_data import load_eval_case, to_acdm_layout  # noqa: E402

ST = REPO / "model_outputs_ctkae/ablation_state"
CASES = ["extrap", "interp", "longer"]
H, W = 64, 128


def smooth(a, n, axis):
    k = np.ones(n) / n
    pad = [(0, 0)] * a.ndim
    pad[axis] = (n // 2, n // 2)
    ap_ = np.pad(a, pad, mode="edge")
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), axis, ap_)


def radial_bins():
    ky = np.fft.fftfreq(H, d=1.0 / H)
    kx = np.arange(W // 2 + 1)
    return np.sqrt(ky[:, None] ** 2 + kx[None] ** 2).round().astype(int)


RB = radial_bins()


def apply_gain(x, M, g):
    """x [S, K, C, H, W], M broadcastable anchor, g [K, C, nb] -> M + F^-1(g F(x - M))."""
    F = np.fft.rfft2(x - M)
    F = F * g[None][..., RB]
    return M + np.fft.irfft2(F, s=(H, W))


class Ckpt:
    def __init__(self, key):
        self.key = key
        self.d = np.load(P / f"{key}.npz")
        self.sig = list(self.d["sig"])
        run = Path(open(ST / f"{key}.done").read().strip())
        if not run.is_absolute():
            run = REPO / run
        self.sig_cal = json.load(open(run / "mix_sigma.json"))["sigma"]
        tm = self.d["train_mse"]  # [nS, K]
        tms = smooth(tm, 15, 1)
        self.sig_lead = np.array(self.sig)[tms.argmin(0)]  # [K]
        self.ilead = tms.argmin(0)
        self.icar = self.sig.index(self.sig_cal)

    def gains(self, anchor, idx_per_lead, per_band=True):
        a = {"self": 0, "clim": 1}[anchor]
        num, den = self.d["train_num"][a], self.d["train_den"][a]  # [nS, K, C, nb]
        Kk = num.shape[1]
        n = num[idx_per_lead, np.arange(Kk)]
        dd = den[idx_per_lead, np.arange(Kk)]
        if not per_band:
            g = n.sum(-1) / np.maximum(dd.sum(-1), 1e-12)
            g = np.repeat(g[..., None], num.shape[-1], -1)
        else:
            g = n / np.maximum(dd, 1e-12)
            g = np.where(dd > 0, g, 1.0)
        return np.clip(smooth(g, 9, 0), 0.0, 1.0)

    def variant(self, case, name):
        pred = self.d[f"{case}_pred"]  # [nS, S, K, C, H, W] fp16
        K = pred.shape[2]
        base, _, post = name.partition("+")
        if base == "single":
            idx = np.zeros(238, int)
        elif base == "mix":
            idx = np.full(238, self.icar)
        elif base == "mixlead":
            idx = self.ilead
        else:
            raise ValueError(name)
        x = pred[idx[:K], :, np.arange(K)].transpose(1, 0, 2, 3, 4).astype(np.float32)
        if post:
            kind, anchor = post[0], post[1:]
            g = self.gains(anchor, idx, per_band=(kind == "w"))[:K]
            if anchor == "self":
                M = x[:, :58].mean(1, keepdims=True)
            else:
                M = self.d[f"{case}_clim"][:, None]
            mask = (np.abs(x).sum((0, 1, 2)) > 0).astype(np.float32)
            x = apply_gain(x, M, g) * mask
        return x


_GT = {}


def gt_case(case):
    if case not in _GT:
        x, ma, _ = load_eval_case(
            str(REPO / f"data/acdm/128_tra/gt_{case}.nc"), torch.device("cpu")
        )
        gt, fluid = load_gt(case)
        _GT[case] = (x[:, :2], ma, gt, fluid)
    return _GT[case]


CLIM = None


def to_acdm(case, x):
    xin, ma, _, _ = gt_case(case)
    full = torch.cat(
        [xin.float(), torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))], 1
    )
    return to_acdm_layout(full, ma)[:, :, :4]


def mse_all(case, x):
    global CLIM
    if CLIM is None:
        CLIM = train_climatology()
    _, _, gt, fluid = gt_case(case)
    return score(to_acdm(case, x)[None], gt, fluid, CLIM)["mse_all"]


VARIANTS = [
    "single",
    "mix",
    "mixlead",
    "mix+sself",
    "mix+wself",
    "mix+sclim",
    "mix+wclim",
    "mixlead+sself",
    "mixlead+wself",
    "mixlead+wclim",
    "single+wself",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lsim", nargs="*", default=[])
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    a = ap.parse_args()
    keys = {
        t: [f"{t}_s{s}" for s in range(3) if (P / f"{t}_s{s}.npz").exists()]
        for t in ["ref", "back001"]
    }
    ck = {k: Ckpt(k) for t in keys for k in keys[t]}
    for k, c in ck.items():
        print(
            k,
            "sigma_cal",
            c.sig_cal,
            "sigma(tau) at leads 1/10/30/58/120/238:",
            [float(c.sig_lead[i]) for i in [0, 9, 29, 57, 119, 237]],
        )
    res = {}
    for v in a.variants:
        for case in CASES:
            fields = {}
            for k, c in ck.items():
                fields[k] = c.variant(case, v)
                res[(v, case, k)] = mse_all(case, fields[k])
            for t in keys:
                if len(keys[t]) > 1:
                    res[(v, case, f"{t}_ens")] = mse_all(
                        case, np.mean([fields[k] for k in keys[t]], 0)
                    )
            if all(len(keys[t]) > 1 for t in keys):
                res[(v, case, "both_ens")] = mse_all(
                    case, np.mean(list(fields.values()), 0)
                )
            del fields
        print("scored", v, flush=True)
    if res:
        json.dump(
            {"|".join(k): v for k, v in res.items()},
            open(P / "pp_results.json", "w"),
            indent=0,
        )
    # report: per model, mean over seeds and paired delta vs "mix"
    for t in keys:
        n = len(keys[t])
        print(
            f"\n== {t} ({n} seeds): all-frames MSE x1e-3, mean over seeds [paired d vs mix +- SE] | {t} 3-seed ensemble"
        )
        for v in a.variants:
            row = f"  {v:16s}"
            for case in CASES:
                vals = np.array([res[(v, case, k)] for k in keys[t]])
                ref = np.array([res[("mix", case, k)] for k in keys[t]])
                d = vals - ref
                se = d.std(ddof=1) / np.sqrt(n) if n > 1 else 0
                ens = res.get((v, case, f"{t}_ens"))
                row += f" | {case[:3]} {vals.mean():6.2f} [{d.mean():+.2f}±{se:.2f}] ens {ens if ens is None else round(ens, 2)}"
            print(row)
    if all(len(keys[t]) > 1 for t in keys):
        print("\n== ensemble of all 6 checkpoints")
        for v in a.variants:
            print(
                f"  {v:16s}"
                + "".join(f" | {c[:3]} {res[(v, c, 'both_ens')]:6.2f}" for c in CASES)
            )
    if a.lsim:
        from lsim_score import lsim

        print("\n== LSiM x1e-1 (lower better)")
        for v in a.lsim:
            for t in list(keys) + ["both"]:
                ks = keys[t] if t in keys else [k for t2 in keys for k in keys[t2]]
                row = f"  {v:16s} {t:8s} seeds:"
                for case in CASES:
                    per = (
                        [
                            lsim(to_acdm(case, ck[k].variant(case, v))[None], case)[0]
                            for k in ks
                        ]
                        if t in keys
                        else []
                    )
                    ens = lsim(
                        to_acdm(case, np.mean([ck[k].variant(case, v) for k in ks], 0))[
                            None
                        ],
                        case,
                    )[0]
                    row += f" | {case[:3]} {np.mean(per) if per else float('nan'):.2f} ens {ens:.2f}"
                print(row, flush=True)


if __name__ == "__main__":
    main()
