"""Gate: every surrogate must beat persistence as a one-step forward model.

Why this exists.  ``verify_adapters`` checks that the adapter reproduces turbpred's OWN
forward pass.  That is necessary but not sufficient: it compares two computations on the
SAME input, so any error in what is fed to the model is invisible to it.  Exactly that
happened -- the .nc files store (x=64, y=128) while the checkpoints are trained on
``dataSize=[128, 64]``, and convolutions accept either layout silently.  The one-step error
was 0.1456 instead of 0.0025 and every baseline came out worse than doing nothing.

This gate is a SANITY check on the physics, not on the code: a trained autoregressive
surrogate must predict the next frame better than simply repeating the current one.  Any
model that fails it is being fed something wrong, and no downstream DA number from it means
anything.
"""

from __future__ import annotations

import argparse
import sys

import torch

from data_assimilation.tra.bridge import REGIMES, PhysicalData, REPO, rel_l2
from data_assimilation.tra.adapters import TurbpredAdapter
from data_assimilation.tra.verify_adapters import MODELS, DIFF_OPTS, NC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="+", default=["tra"])
    ap.add_argument("--models", nargs="+", default=["UNet", "FNO"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = REPO / "autoreg_pde_diffusion" / "pretrained_models"
    fails = 0
    for reg_name in a.regimes:
        reg = REGIMES[reg_name]
        d = PhysicalData(REPO / "koopman_autoencoder" / NC[reg_name], reg_name, dev)
        sim = torch.arange(min(a.batch, d.n_sim))
        t0 = torch.full((len(sim),), 4, dtype=torch.long)
        par, om = d.params_for(sim), d.mask_for(sim)
        pers = float(rel_l2(d.frames(sim, t0), d.frames(sim, t0 + 1), om).mean())
        print(f"\n=== {reg_name}: persistence one-step = {pers:.4f} ===")
        print(f"{'model':9s} {'1-step':>9s} {'vs persistence':>15s} {'verdict':>9s}")
        for m in a.models:
            ck = base / f"models_{reg_name}" / MODELS[m] / "Model.pth"
            if not ck.is_file():
                continue
            ad = TurbpredAdapter(ck, reg, dev, DIFF_OPTS.get(m), name=m)
            k = ad.n_control_frames
            with torch.no_grad():
                tr = ad.rollout(
                    ad.to_model(d.window(sim, t0 - (k - 1), k)),
                    a.steps,
                    par,
                    checkpoint_every=0,
                )
            e1 = float(rel_l2(tr[:, k], d.frames(sim, t0 + 1), om).mean())
            ok = e1 < pers
            fails += not ok
            print(
                f"{m:9s} {e1:9.4f} {pers / max(e1, 1e-12):14.1f}x "
                f"{'PASS' if ok else 'FAIL':>9s}"
            )
            del ad
            torch.cuda.empty_cache()
    print(f"\n{'ALL MODELS BEAT PERSISTENCE' if not fails else str(fails) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
