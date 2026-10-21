"""Is late-lead error phase drift, amplitude loss, or decorrelation? (test sets, analysis only)

Per test simulation, the two leading POD modes of the true fluctuations (about the true
time mean, fields scaled by GT_STD, fluid points) span the vortex-shedding pair. Truth and
forecast are projected onto them, a(t) = a1 + i a2, and per lead:
  phase_err   |angle(a_pred conj(a_true))| in degrees (mean over sims and samples)
  amp_ratio   |a_pred| / |a_true|
  resid_frac  share of the forecast error NOT in the shedding plane
  band_err    forecast error energy (scaled units, per fluid value) in bands of the truth's
              own POD modes: 1-2, 3-10, 11+, and outside the true trajectory's span
plus the mean shedding frequency of truth and forecast (unwrapped phase slope) and the
share of true fluctuation energy in the two modes.

    python phase_drift.py --ctkae RUN_DIR[:CKPT] ... [--baselines acdm-r20 unet-m8] [--json out.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
REPO = Path("/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from benchmark_suite import GT_STD, UNTIMED, load_gt, load_pred  # noqa: E402
from eval_ctkae import load_model, rollout  # noqa: E402
from models.ct_data import load_eval_case, to_acdm_layout  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ctkae", nargs="*", default=[])
ap.add_argument("--baselines", nargs="*", default=["acdm-r20", "unet-m8"])
ap.add_argument("--json", default=None)
args = ap.parse_args()
dev = torch.device("cuda")
LEADS = [1, 10, 20, 30, 45, 58]
DT = 0.1


def ctkae_pred(spec, case):
    run, _, ck = spec.partition(":")
    m, _ = load_model(Path(run), ck or "last.pth", dev)
    x, ma, mask = load_eval_case(str(REPO / f"data/acdm/128_tra/gt_{case}.nc"), dev)
    T = x.shape[1]
    taus = DT * torch.arange(1, T - 1, device=dev, dtype=torch.float32)
    full = torch.cat([x[:, :2], rollout(m, x, ma, mask, 1, taus)], 1)
    return to_acdm_layout(full, ma)[None, :, :, :4]  # [1, S, T, 4, H, W]


def analyse(pred, gt, fluid):
    """pred [n, S, T, 4, H, W]; gt [S, T, 4, H, W] -> per-lead statistics."""
    std = GT_STD[:, None, None]
    n, S, T = pred.shape[:3]
    ph, amp, res, w_t, w_p, e2 = [], [], [], [], [], []
    bands: dict = {"1-2": [], "3-10": [], "11+": [], "outside": []}
    for s in range(S):
        g = (gt[s, 2:] / std)[..., fluid]  # [N, 4, F]
        mean = g.mean(0, keepdims=True)
        X = (g - mean).reshape(g.shape[0], -1)
        _, sv, Vt = np.linalg.svd(X, full_matrices=False)
        V = Vt[:2].T
        e2.append(float((sv[:2] ** 2).sum() / (sv**2).sum()))
        at = X @ V
        at = at[:, 0] + 1j * at[:, 1]
        w_t.append(np.mean(np.diff(np.unwrap(np.angle(at)))) / DT)
        for j in range(n):
            P = ((pred[j, s, 2:] / std)[..., fluid] - mean).reshape(g.shape[0], -1)
            ap_ = P @ V
            ap_ = ap_[:, 0] + 1j * ap_[:, 1]
            ph.append(np.abs(np.angle(ap_ * np.conj(at))) * 180 / np.pi)
            amp.append(np.abs(ap_) / np.abs(at))
            err = P - X
            inplane = (err @ V) @ V.T
            res.append(1 - (inplane**2).sum(1) / np.maximum((err**2).sum(1), 1e-12))
            c = err @ Vt.T  # coefficients on all true POD modes [N, r]
            nf = err.shape[1]
            bands["1-2"].append((c[:, :2] ** 2).sum(1) / nf)
            bands["3-10"].append((c[:, 2:10] ** 2).sum(1) / nf)
            bands["11+"].append((c[:, 10:] ** 2).sum(1) / nf)
            bands["outside"].append(((err**2).sum(1) - (c**2).sum(1)) / nf)
            w_p.append(np.mean(np.diff(np.unwrap(np.angle(ap_)))) / DT)
    pha, ampa, resa = np.array(ph), np.array(amp), np.array(res)

    def at_leads(a):
        return {
            L: round(float(a[:, L - 1].mean()), 3) for L in LEADS if L - 1 < a.shape[1]
        }

    return {
        "phase_err_deg": at_leads(pha),
        "amp_ratio": at_leads(ampa),
        "resid_frac": at_leads(resa),
        "freq_true": round(float(np.mean(w_t)), 4),
        "freq_pred": round(float(np.mean(w_p)), 4),
        "freq_rel_err": round(float(np.mean(w_p) / np.mean(w_t) - 1), 4),
        "shedding_pair_energy": round(float(np.mean(e2)), 3),
        "band_err": {k: at_leads(np.array(v)) for k, v in bands.items()},
    }


out: dict = {}
for case in ("interp", "extrap"):
    gt, fluid = load_gt(case)
    S, T = gt.shape[:2]
    srcs = {}
    for spec in args.ctkae:
        srcs[Path(spec.split(":")[0]).name] = lambda spec=spec: ctkae_pred(spec, case)
    for b in args.baselines:
        p = UNTIMED / case / f"{b}.npz"
        srcs[b] = lambda p=p: load_pred(p, S, T, early=False)
    out[case] = {}
    for k, f in srcs.items():
        r = analyse(f(), gt, fluid)
        out[case][k] = r
        print(case, k, json.dumps(r))
if args.json:
    Path(args.json).write_text(json.dumps(out, indent=1))
