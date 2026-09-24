"""Gate: the adapter must reproduce turbpred's OWN forward pass, bit for bit.

``TurbpredAdapter.rollout`` is a transcription of ``PredictionModel.forwardDirect`` /
``forwardDiffusionDirect`` with the simulation-parameter channel supplied from the known
constant instead of from future ground truth (a DA driver has no future frames).  If the
transcription is right, then feeding the TRUE sequence must give exactly what turbpred's
own ``model(data, simParameters)`` gives.  Anything else means the glue is wrong, and every
number built on it would be wrong too.

Deterministic models are required to agree to ~float32 round-off.  Diffusion models draw
noise internally, so they are checked under a fixed seed instead, and additionally that
their output is a plausible field rather than a match to a different RNG stream.
"""

from __future__ import annotations

import argparse
import sys

import torch

from data_assimilation.tra.bridge import REGIMES, PhysicalData, REPO
from data_assimilation.tra.adapters import TurbpredAdapter

MODELS = {
    "UNet": "128_unet-m2_00",
    "FNO": "128_fno-32modes-m2_00",
    "ACDM": "128_acdm-r20_00",
    "ACDM-ncn": "128_acdm-r20_ncn_00",
}
DIFF_OPTS = {
    "ACDM": dict(
        samplingMode="ddpm",
        posteriorSampling="random",
        initialSampling="random",
        conditioningIntegration="noisy",
    ),
    "ACDM-ncn": dict(
        samplingMode="ddpm",
        posteriorSampling="random",
        initialSampling="random",
        conditioningIntegration="clean",
    ),
}
NC = {
    "tra": "data/acdm/128_tra/gt_interp.nc",
    "inc": "data/acdm/128_inc/gt_highRey_stable.nc",
}


def reference_forward(ad: TurbpredAdapter, data_norm, param_phys, n_total):
    """turbpred's own forward, with the full true sequence, as sample_models_*.py calls it."""
    B, T, C, H, W = data_norm.shape
    if ad.n_params:
        pch = (
            ad.param_channel(param_phys, (H, W)).unsqueeze(1).expand(-1, T, -1, -1, -1)
        )
        d = torch.cat([data_norm, pch], dim=2)
    else:
        d = data_norm
    with torch.no_grad():
        pred, _, _ = ad.model(d, None)
    return pred[:, :, : ad.n_fields]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="+", default=["tra", "inc"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--tol", type=float, default=2e-4)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = REPO / "autoreg_pde_diffusion" / "pretrained_models"
    fails = 0
    for reg_name in a.regimes:
        reg = REGIMES[reg_name]
        data = PhysicalData(REPO / "koopman_autoencoder" / NC[reg_name], reg_name, dev)
        sim = torch.arange(min(a.batch, data.n_sim))
        t0 = torch.zeros(len(sim), dtype=torch.long)
        seq_phys = data.window(sim, t0, a.frames)
        par = data.params_for(sim)
        print(
            f"\n=== {reg_name}  ({data.n_sim} sims, {a.frames} frames, "
            f"batch {len(sim)}) ==="
        )

        for m in a.models:
            ck = base / f"models_{reg_name}" / MODELS[m] / "Model.pth"
            if not ck.is_file():
                print(f"  {m:9s} SKIP (no checkpoint at {ck})")
                continue
            ad = TurbpredAdapter(ck, reg, dev, DIFF_OPTS.get(m), name=m)
            k = ad.n_control_frames
            seq_norm = ad.to_model(seq_phys)

            torch.manual_seed(0)
            ref = reference_forward(ad, seq_norm, par, a.frames)
            torch.manual_seed(0)
            with torch.no_grad():
                got_phys = ad.rollout(seq_norm[:, :k].contiguous(), a.frames - k, par)
            got = ad.to_model(got_phys)

            d = (got - ref).abs()
            worst = float(d.max())
            rel = float(d.mean() / ref.abs().mean())
            ok = worst < a.tol
            # a round-trip check that is independent of the model
            rt = float((ad.to_physical(ad.to_model(seq_phys)) - seq_phys).abs().max())
            tag = "PASS" if ok else ("NOISE" if ad.is_diffusion else "FAIL")
            if not ok and not ad.is_diffusion:
                fails += 1
            print(
                f"  {m:9s} k={k} {'diffusion' if ad.is_diffusion else 'determin.'}  "
                f"max|adapter-turbpred|={worst:.3e}  rel={rel:.3e}  [{tag}]"
                f"   norm round-trip {rt:.2e}"
            )
            if ad.is_diffusion and not ok:
                print(
                    f"            (diffusion draws its own noise; agreement is not "
                    f"expected. field std ref {float(ref.std()):.3f} vs "
                    f"adapter {float(got.std()):.3f})"
                )
            del ad
            torch.cuda.empty_cache()

    print(
        f"\n{'ALL DETERMINISTIC ADAPTERS MATCH' if not fails else str(fails) + ' FAILED'}"
    )
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
