"""Diagnostic (inputs only): reconstruction error of the two INPUT frames from the encoder's z0
and after refining z0 on them (Adam, no prior). No future frames are used.

    python z0_diag.py RUN_DIR
"""

import sys
from pathlib import Path

import torch

REPO = Path("/home/rg625/mnt/ocean_forecasting_impl/koopman_autoencoder")
sys.path.insert(0, str(REPO))
from eval_ctkae import load_model  # noqa: E402
from models.ct_data import TraWindows, load_eval_case  # noqa: E402

dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(3.0 / 23.5)
m, _ = load_model(Path(sys.argv[1]), "last.pth", dev)
for p in m.parameters():
    p.requires_grad_(False)


def rec_err(z, x2, mask, ma):
    """mean over the two input frames of the masked MSE (normalised units)"""
    taus = torch.tensor([-0.1, 0.0], device=dev).expand(z.shape[0], -1)
    d = m.decode(m.forecast(z, taus, ma), mask, ma)
    return ((d - x2) ** 2 * mask).sum((-1, -2, -3)).mean(-1) / (mask.sum() * 4)


def refine(z0, x2, mask, ma, steps, lr=1e-2):
    z = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    out = {}
    for i in range(steps + 1):
        if i in (0, 25, 50, 100, 200, 400):
            out[i] = rec_err(z.detach(), x2, mask, ma).mean().item()
        if i == steps:
            break
        opt.zero_grad()
        rec_err(z, x2, mask, ma).sum().backward()
        opt.step()
    return out


for case in ["extrap", "interp", "longer"]:
    x, ma, mask = load_eval_case(str(REPO / f"data/acdm/128_tra/gt_{case}.nc"), dev)
    x2 = x[:, :2]
    with torch.no_grad():
        z0 = m.encode(x2, mask, ma)
    print(
        case,
        "|z0| rms",
        z0.pow(2).mean().sqrt().item(),
        refine(z0, x2, mask, ma, 400),
        flush=True,
    )

tw = TraWindows(str(REPO / "data/acdm/128_tra/train.nc"), None, torch.device("cpu"))
win, tgt, taus, mas = tw.fixed_windows(2, 2, 2, [500])
idx = torch.arange(0, len(win), 4)
x2, mas = win[idx].to(dev), mas[idx].to(dev)
mask = tw.mask.to(dev)
with torch.no_grad():
    z0 = m.encode(x2, mask, mas)
print("train", refine(z0, x2, mask, mas, 400))
