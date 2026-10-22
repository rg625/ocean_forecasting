"""Inference-only ablations of a trained block-generator CT-KAE (no retraining; analysis only).

  --gamma_scale s   multiply every oscillator's damping gamma(Ma) by s (0 = neutral modes)
  --snap            snap oscillator frequencies to integer multiples of the dominant one
                    (the mode carrying most latent energy on the test case), modes with
                    |omega| below half the fundamental left unchanged
Scored with benchmark_suite.score on gt_interp / gt_extrap / gt_longer, last.pth.

    python infer_ablate.py RUN_DIR [--ckpt last.pth] [--gamma_scale 1 0.5 0] [--snap] [--json out]
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
from benchmark_suite import load_gt, score, train_climatology  # noqa: E402
from eval_ctkae import load_model, rollout  # noqa: E402
from models.ct_data import load_eval_case, to_acdm_layout  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--ckpt", default="last.pth")
ap.add_argument("--gamma_scale", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.0])
ap.add_argument("--snap", action="store_true")
ap.add_argument("--json", default=None)
args = ap.parse_args()
dev = torch.device("cuda")
m, _ = load_model(Path(args.run_dir), args.ckpt, dev)
gen = m.gen
orig_params = gen.params
clim = train_climatology()
STATE: dict = {"scale": 1.0, "w_override": None}


def patched(e, batch, cn=None):
    g, w = orig_params(e, batch, cn)
    if STATE["w_override"] is not None:
        w = STATE["w_override"](w, g)
    return g * STATE["scale"], w


gen.params = patched


@torch.no_grad()
def fundamental(x, ma, mask):
    """Index of the oscillator pair with the largest mean latent amplitude."""
    z = m.encode(x[:, :2], mask, ma) - m.fixed_point(m.embed_cond(ma), x.shape[0])
    amp = z[:, : m.slow_dim].view(z.shape[0], -1, 2).norm(dim=-1).mean(0)
    return int(amp.argmax())


def snap_fn(k0):
    def f(w, g):
        w1 = w[:, k0 : k0 + 1].abs()
        n = torch.round(w.abs() / w1)
        snapped = torch.sign(w) * n * w1
        return torch.where(n >= 1, snapped, w)

    return f


@torch.no_grad()
def run_case(case):
    x, ma, mask = load_eval_case(str(REPO / f"data/acdm/128_tra/gt_{case}.nc"), dev)
    T = x.shape[1]
    taus = 0.1 * torch.arange(1, T - 1, device=dev, dtype=torch.float32)
    full = torch.cat([x[:, :2], rollout(m, x, ma, mask, 1, taus)], 1)
    gt, fluid = load_gt(case)
    r = score(to_acdm_layout(full, ma)[None, :, :, :4], gt, fluid, clim)
    lead = r["mse_lead"]
    return {
        "mse": round(r["mse_pred"], 3),
        "lead": {
            L: round(lead[L - 1], 3) for L in (1, 10, 20, 30, 58) if L - 1 < len(lead)
        },
        "energy58": round(r["energy_lead"][57], 3),
        "acc58": round(r["acc_lead"][57], 3),
        "_x": (x, ma, mask),
    }


out: dict = {"run": args.run_dir}
variants = [(f"gamma_x{s}", s, False) for s in args.gamma_scale]
if args.snap:
    variants += [("snap_harmonics", 1.0, True), ("snap_harmonics_gamma_x0", 0.0, True)]
for name, s, snap in variants:
    out[name] = {}
    for case in ("interp", "extrap", "longer"):
        STATE["scale"], STATE["w_override"] = 1.0, None
        if snap:
            x, ma, mask = load_eval_case(
                str(REPO / f"data/acdm/128_tra/gt_{case}.nc"), dev
            )
            k0 = fundamental(x, ma, mask)
            STATE["w_override"] = snap_fn(k0)
        STATE["scale"] = s
        r = run_case(case)
        r.pop("_x")
        out[name][case] = r
    print(name, json.dumps(out[name]))

# frequency ratios to the fundamental, amplitude-weighted (interp case, true Ma)
x, ma, mask = load_eval_case(str(REPO / "data/acdm/128_tra/gt_interp.nc"), dev)
STATE["scale"], STATE["w_override"] = 1.0, None
with torch.no_grad():
    e = m.embed_cond(ma)
    g, w = orig_params(e, x.shape[0])
    z = m.encode(x[:, :2], mask, ma) - m.fixed_point(e, x.shape[0])
    amp = z[:, : m.slow_dim].view(z.shape[0], -1, 2).norm(dim=-1).mean(0)
    k0 = int(amp.argmax())
    ratio = (w.abs() / w[:, k0 : k0 + 1].abs()).mean(0).cpu().numpy()
    a = (amp / amp.sum()).cpu().numpy()
    gg = g.mean(0).cpu().numpy()
order = np.argsort(-a)[:20]
out["top_modes_interp"] = [
    {
        "amp_share": round(float(a[i]), 4),
        "omega_ratio": round(float(ratio[i]), 3),
        "gamma": round(float(gg[i]), 4),
    }
    for i in order
]
out["fundamental_omega_interp"] = float(w[:, k0].abs().mean())
print(json.dumps(out["top_modes_interp"][:12]))
if args.json:
    Path(args.json).write_text(json.dumps(out, indent=1))
