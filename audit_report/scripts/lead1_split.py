"""Split a CT-KAE's one-step error into reconstruction and linear-propagation parts (test sets).

For every window (x_{t-1}, x_t) of gt_interp / gt_extrap, with z_t = E(x_{t-1}, x_t) and
P = e^{A tau}, tau = one frame:
  rec       |D(z_t) - x_t|^2                 encode-decode of the current frame
  pred1     |D(P z_t) - x_{t+1}|^2           one-step forecast (the scorer's lead 1 at t = 1)
  rec_next  |D(z_{t+1}) - x_{t+1}|^2         best the decoder can do on the target frame
  dyn       |D(P z_t) - D(z_{t+1})|^2        what the linear step adds, inside the decoder's range
  lat_res   |P z_t - z_{t+1}|^2 / |z_{t+1}|^2
MSE x1e-3 in physical units, obstacle zeroed, as benchmark_suite.

    python lead1_split.py RUN_DIR [--ckpt last.pth] [--json out.json]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path("/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder")
sys.path.insert(0, str(REPO))
from eval_ctkae import load_model  # noqa: E402
from models.ct_data import denormalise, load_eval_case  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--ckpt", default="last.pth")
ap.add_argument("--json", default=None)
args = ap.parse_args()
dev = torch.device("cuda")
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
m, _ = load_model(Path(args.run_dir), args.ckpt, dev)


def mse(a, b, mask):
    return ((denormalise(a) - denormalise(b)) * mask).pow(2).mean(
        dim=(-1, -2, -3)
    ) * 1e3


out: dict = {"run": args.run_dir, "ckpt": args.ckpt}
with torch.no_grad():
    for case in ("interp", "extrap"):
        x, ma, mask = load_eval_case(str(REPO / f"data/acdm/128_tra/gt_{case}.nc"), dev)
        S, T = x.shape[:2]
        acc: dict = {k: [] for k in ("rec", "pred1", "rec_next", "dyn", "lat_res")}
        first: dict = {}
        for s in range(S):
            win = torch.stack([x[s, :-1], x[s, 1:]], 1)  # windows ending at t = 1..T-1
            c = ma[s : s + 1].expand(win.shape[0])
            z = torch.cat(
                [m.encode(w, mask, cc) for w, cc in zip(win.split(64), c.split(64))]
            )
            tau = torch.full((z.shape[0], 1), 0.1, device=dev)
            pz = m.propagate(z, tau, c)[:, 0]
            dz = torch.cat(
                [m.decode(v, mask, cc) for v, cc in zip(z.split(64), c.split(64))]
            )
            dp = torch.cat(
                [m.decode(v, mask, cc) for v, cc in zip(pz.split(64), c.split(64))]
            )
            xt = x[s, 1:]
            r = {
                "rec": mse(dz[:-1], xt[:-1], mask),
                "pred1": mse(dp[:-1], xt[1:], mask),
                "rec_next": mse(dz[1:], xt[1:], mask),
                "dyn": mse(dp[:-1], dz[1:], mask),
                "lat_res": (pz[:-1] - z[1:]).pow(2).sum(-1) / z[1:].pow(2).sum(-1),
            }
            for k, v in r.items():
                acc[k].append(v)
                first.setdefault(k, []).append(float(v[0]))
        out[case] = {k: float(torch.cat(v).mean()) for k, v in acc.items()}
        out[case]["pred1_at_t1"] = sum(first["pred1"]) / S
        print(case, {k: round(v, 4) for k, v in out[case].items()})
if args.json:
    Path(args.json).write_text(json.dumps(out, indent=1))
