"""Check that CTKAEAdapter (shifted linear coordinates) reproduces the CT-KAE's own forecast
and runs through solve_kae.

    python -m data_assimilation.tra.verify_ctkae_adapter RUN_DIR [CKPT]
"""

import sys

import numpy as np
import torch

from data_assimilation.tra.bridge import REGIMES, PhysicalData
from data_assimilation.tra.ctkae_adapter import CTKAEAdapter
from data_assimilation.tra.fourdvar import solve_kae
from data_assimilation.tra.protocol import build_problem, canonical
from models.ct_data import load_eval_case

run, ckpt = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "best.pth")
dev = torch.device("cuda")
ad = CTKAEAdapter(run, REGIMES["tra"], dev, ckpt=ckpt)
print(ad)
data = PhysicalData("data/acdm/128_tra/gt_interp.nc", "tra", dev)
x, ma, mask = load_eval_case("data/acdm/128_tra/gt_interp.nc", dev)
with torch.no_grad():
    # native model: frames 0,1 -> leads 1..5 (frames 2..6), normalised
    taus = 0.1 * torch.arange(1, 6, device=dev, dtype=torch.float32)
    z0 = ad.model.encode(x[:, :2], mask, ma)
    native = ad.model.decode(
        ad.model.propagate(z0, taus.expand(x.shape[0], -1), ma),
        mask,
        ma.repeat_interleave(5),
    )
    native = ad.to_physical(native.view(-1, 4, *mask.shape)).view(
        x.shape[0], 5, 4, *mask.shape
    )
    # adapter: physical window -> y0 -> exp(A tau) y0 -> decode
    win = ad.to_physical(x[:, :2].reshape(-1, 4, *mask.shape)).view(
        x.shape[0], 2, 4, *mask.shape
    )
    y0 = ad.encode(win, ma)
    K = ad.generator(ma)
    outs = [ad.decode(ad.propagate(y0, float(t), K=K)) for t in taus]
    via = torch.stack(outs, 1)
d = float((via - native).abs().max())
print(f"adapter vs native forecast, max |diff| (physical): {d:.2e}")
assert d < 1e-3, d
prob = build_problem(
    data, name="verify", n_problems=4, offsets=canonical(), seed=7, noise_std=0.0
)
r = solve_kae(ad, data, prob, iters=50, lr=1e-2, seed=1)
print("solve_kae ok: rel-L2", np.round(r["rel"], 4), f"[{r['wall_s']:.1f}s]")
