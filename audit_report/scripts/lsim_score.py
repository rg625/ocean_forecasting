"""LSiM on the Tra test cases, replicating the ACDM evaluation (plot_loss_all.py, metric LSIM).

Per sequence and channel, prediction and ground truth (obstacle zeroed, fields 0:4, ALL frames
including the two inputs) are min-max scaled to [0, 255] over (time, H, W) and compared by the
pretrained LSiM network; the per-sequence value is the mean over frames and channels, and the
reported number is mean +- std over (models, samples, sequences), x10 (the paper's x10^-1 unit).

    from lsim_score import lsim            # lsim(pred [n,S,T,4,H,W], case) -> (mean, std)
    python lsim_score.py --baselines acdm-r20 unet-m8      # validation against the ACDM paper
"""

from _paths import MAIN_ROOT
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ACDM = MAIN_ROOT / "autoreg_pde_diffusion/src"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ACDM))
from benchmark_suite import UNTIMED, load_gt, load_pred  # noqa: E402
from turbpred.loss import loss_lsim  # noqa: E402
from turbpred.lsim.distance_model import DistanceModel  # noqa: E402

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = DistanceModel(baseType="lsim", isTrain=False, useGPU=True)
        _MODEL.load(str(ACDM / "turbpred/lsim/models/LSiM.pth"))
        _MODEL.eval()
        for p in _MODEL.parameters():
            p.requires_grad_(False)
        _MODEL = _MODEL.cuda()
    return _MODEL


@torch.no_grad()
def lsim(pred: np.ndarray, case: str) -> tuple:
    """pred [n, S, T, 4, H, W] in the ACDM layout (frames 0, 1 = inputs)."""
    gt, fluid = load_gt(case)  # gt already obstacle-zeroed
    mask = fluid.astype(np.float32)
    vals = []
    for j in range(pred.shape[0]):
        for s in range(pred.shape[1]):
            p = torch.from_numpy(np.nan_to_num(pred[j, s]) * mask).cuda()[None]
            g = torch.from_numpy(gt[s]).cuda()[None]
            d = loss_lsim(_model(), g, p)  # [1, T, C]
            vals.append(float(d.mean()))
    v = np.array(vals)
    return 10 * float(v.mean()), 10 * float(v.std())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", default=["acdm-r20", "unet-m8"])
    a = ap.parse_args()
    for case in ("extrap", "interp", "longer"):
        gt, _ = load_gt(case)
        S, T = gt.shape[:2]
        for b in a.baselines:
            p = load_pred(UNTIMED / case / f"{b}.npz", S, T, early=False)
            m, sd = lsim(p, case)
            print(f"{case:7s} {b:12s} LSiM x10^-1 = {m:.2f} +- {sd:.2f}")
