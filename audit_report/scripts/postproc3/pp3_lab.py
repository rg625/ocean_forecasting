"""Post-processing round 3: improve on the single decode WITHOUT blurring.

Members per model: single decodes of seeds 0-2 (sigma = 0) and the adopted average
AVG = 3-seed mean of mixture + mirror (no shrinkage). Candidates:
  medoid     per (sequence, lead): the single decode closest (MSE) to AVG; one real forecast
  align      each single shifted along x (integer, |s| <= 6) to best match AVG, then averaged
  sharp      AVG with its anomaly (about the train climatology) rescaled per (frame, channel,
             radial wavenumber) to the mean single-decode anomaly power; phases of AVG kept
  sharp+tr   as sharp, target power times the train ratio truth/single-forecast (per lead,
             channel, wavenumber; from pp_lab's train rollouts, seeds averaged)
Nothing is tuned on test (gain cap 5 and shift range fixed a priori).
Metrics: MSE (all frames, x1e-3), LSiM (x1e-1), |v| spectrum band errors and signed mean
log10 power ratio per band (pred / truth), mid-band slope.

    python pp3_lab.py MODEL [--no-lsim]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).parents[1]
sys.path.insert(0, str(R / "pp"))
sys.path.insert(0, "/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion/src")
import pp_analyze as P1  # noqa: E402
from pp_analyze import RB, H, W, smooth  # noqa: E402
from spectral_analysis import (  # noqa: E402
    BANDS,
    fitted_slope,
    spectrum,
    velocity_magnitude,
)

CASES = ["extrap", "interp", "longer"]
GMAX, SHIFT = 5.0, 6
NB = RB.max() + 1


def power(x, M):
    """band power of the anomaly: [S, K, C, NB]"""
    F = np.abs(np.fft.rfft2(x - M)) ** 2
    out = np.zeros(x.shape[:3] + (NB,), np.float64)
    for b in range(NB):
        out[..., b] = F[..., RB == b].sum(-1)
    return out


def regain(x, M, g):
    """M + F^-1(g F(x - M)) with g [S, K, C, NB] per frame"""
    F = np.fft.rfft2(x - M)
    F = F * g[..., RB]
    return M + np.fft.irfft2(F, s=(H, W))


def spec_metrics(case, x):
    gt = torch.from_numpy(P1.gt_case(case)[2])  # ACDM layout [S, T, 4, 128, 64]
    p = torch.from_numpy(P1.to_acdm(case, x).astype(np.float32))
    T = gt.shape[1]
    k, sg = spectrum(velocity_magnitude(gt[:, 2:T]))
    _, sp = spectrum(velocity_magnitude(p[:, 2:T]))
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


def main():
    model = sys.argv[1]
    do_lsim = "--no-lsim" not in sys.argv
    if do_lsim:
        from lsim_score import lsim
    cks = [P1.Ckpt(f"{model}_s{s}") for s in range(3)]
    aps = [np.load(R / "pp2" / f"apply_{model}_s{s}.npz") for s in range(3)]
    # train ratio truth / single-forecast anomaly power (anchor: clim), per lead
    ratio = np.mean(
        [
            ck.d["train_tvar"][1] / np.maximum(ck.d["train_den"][1, 0], 1e-30)
            for ck in cks
        ],
        0,
    )
    ratio = smooth(ratio, 9, 0)  # [238, C, NB]
    res: dict = {}
    for case in CASES:
        singles = [ck.d[f"{case}_pred"][0].astype(np.float32) for ck in cks]
        avg = np.mean(
            [ap[f"{case}|enc:cal:mirror"].astype(np.float32) for ap in aps], 0
        )
        mask = (np.abs(singles[0]).sum((0, 1, 2)) > 0).astype(np.float32)
        M = cks[0].d[f"{case}_clim"][:, None]  # [S, 1, C, H, W]
        K = avg.shape[1]

        # medoid
        d = np.stack([((s - avg) ** 2).mean((2, 3, 4)) for s in singles])  # [3, S, K]
        pick = d.argmin(0)
        medoid = np.choose(pick[..., None, None, None], singles)

        # align: integer x-shift per (member, sequence, lead), channels shared
        shifts = range(-SHIFT, SHIFT + 1)
        aligned = []
        for s in singles:
            err = np.stack(
                [
                    ((np.roll(s, sh, -1) * mask - avg) ** 2).mean((2, 3, 4))
                    for sh in shifts
                ]
            )
            best = np.array(list(shifts))[err.argmin(0)]  # [S, K]
            a = np.empty_like(s)
            for sh in shifts:
                sel = best == sh
                a[sel] = np.roll(s, sh, -1)[sel]
            aligned.append(a * mask)
        align = np.mean(aligned, 0)

        # spectral re-sharpening
        e_avg = power(avg, M)
        e_single = np.mean([power(s, M) for s in singles], 0)
        g = np.clip(np.sqrt(e_single / np.maximum(e_avg, 1e-30)), 1.0, GMAX)
        sharp = regain(avg, M, g) * mask
        g_tr = np.clip(
            np.sqrt(e_single * ratio[None, :K] / np.maximum(e_avg, 1e-30)), 1.0, GMAX
        )
        sharp_tr = regain(avg, M, g_tr) * mask

        variants = {
            "single (seed mean)": None,
            "3-seed single mean": np.mean(singles, 0),
            "avg (mix+mirror, 3 seeds)": avg,
            "medoid": medoid,
            "align": align,
            "sharp": sharp,
            "sharp+tr": sharp_tr,
        }
        res[case] = {}
        for name, x in variants.items():
            if x is None:  # mean of per-seed metrics
                ms = [P1.mse_all(case, s) for s in singles]
                sp = [spec_metrics(case, s) for s in singles]
                r = {
                    "mse": float(np.mean(ms)),
                    **{k: float(np.mean([q[k] for q in sp])) for k in sp[0]},
                }
                if do_lsim:
                    r["lsim"] = float(
                        np.mean(
                            [lsim(P1.to_acdm(case, s)[None], case)[0] for s in singles]
                        )
                    )
            else:
                r = {"mse": float(P1.mse_all(case, x)), **spec_metrics(case, x)}
                if do_lsim:
                    r["lsim"] = float(lsim(P1.to_acdm(case, x)[None], case)[0])
            res[case][name] = r
            print(
                case,
                f"{name:28s}",
                " ".join(f"{k}={v:.3f}" for k, v in r.items()),
                flush=True,
            )
        del singles, avg, medoid, aligned, align, sharp, sharp_tr
    json.dump(res, open(Path(__file__).parent / f"pp3_{model}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
