"""Reference baselines on the CT-KAE validation windows (held-out Ma 0.60 / 0.75), CPU only.

Same windows and metric as train_ctkae.validate: stride 2 (dt = 0.1), 60-step rollouts from
raw starts [100, 300, 500, 700, 860] and 240-step rollouts from [100, 300, 500]; MSE x1e-3 in
physical units over predicted frames. Baselines fitted on the 30 training simulations only:
persistence, training climatology, POD(r) + 2-frame linear map (POD-DMD2).

    python val_baselines.py --rank 128 --json out.json
"""

from _paths import REPO_ROOT
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(REPO_ROOT / "koopman_autoencoder"))
from models.ct_data import TraWindows, denormalise, train_val_intervals  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, default=128)
ap.add_argument("--json", default=None)
args = ap.parse_args()
cpu = torch.device("cpu")
path = str(REPO_ROOT / "koopman_autoencoder/data/acdm/128_tra/train.nc")
tr_int, va_int = train_val_intervals([0.60, 0.75])
train = TraWindows(path, tr_int, cpu, dtype=torch.float32)
val = TraWindows(path, va_int, cpu, dtype=torch.float32)

xs = train.x[:, ::2]  # dt = 0.1
S, T = xs.shape[:2]
flat = xs.reshape(S * T, -1)
mean = flat.mean(0, keepdim=True)
U, s, V = torch.svd_lowrank(flat - mean, q=args.rank + 20, niter=4)
Phi = V[:, : args.rank]
a = ((flat - mean) @ Phi).reshape(S, T, -1).double()
X = torch.cat([a[:, 1:-1], a[:, :-2]], -1).reshape(-1, 2 * args.rank)
Y = a[:, 2:].reshape(-1, args.rank)
G = X.T @ X
G += 1e-6 * torch.trace(G) / G.shape[0] * torch.eye(G.shape[0], dtype=G.dtype)
K = torch.linalg.solve(G, X.T @ Y).T  # [r, 2r]
clim = mean.view(1, *xs.shape[2:])

res: dict = {}
for Kh, starts in ((60, [100, 300, 500, 700, 860]), (240, [100, 300, 500])):
    win, tgt, _, _ = val.fixed_windows(2, Kh, 2, starts)
    tt = denormalise(tgt)
    preds = {
        "persistence": win[:, -1:].expand_as(tgt),
        "train_climatology": clim.expand_as(tgt),
    }
    aw = ((win.reshape(win.shape[0], 2, -1) - mean) @ Phi).double()
    z = torch.cat([aw[:, 1], aw[:, 0]], -1)
    outs = []
    for _ in range(Kh):
        nxt = z @ K.T
        outs.append(nxt)
        z = torch.cat([nxt, z[:, : args.rank]], -1)
    preds[f"pod_dmd2_r{args.rank}"] = (
        torch.stack(outs, 1).float() @ Phi.T + mean
    ).view_as(tgt) * val.mask
    for k, p in preds.items():
        e = ((denormalise(p) - tt) ** 2).mean(dim=(-1, -2)).mean(-1)  # [N, K]
        res.setdefault(k, {})[f"val_mse{Kh}"] = float(e.mean() * 1e3)
        res[k][f"val_mse{Kh}_first10"] = float(e[:, :10].mean() * 1e3)
print(json.dumps(res, indent=1))
if args.json:
    Path(args.json).write_text(json.dumps(res, indent=1))
