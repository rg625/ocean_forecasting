"""Inference cost of the CT-KAE readouts, under the baseline protocol of
autoreg_pde_diffusion/src/bench_inference.py: Tra_long, one trajectory (batch 1), 240 predicted
steps, CUDA events around the whole forecast (encode, propagate, decode, readout), 2 untimed
warm-up repeats, median of 5 timed repeats, peak VRAM from torch.cuda.max_memory_allocated.

Readouts: single decode; mixture (5 decodes per step); mixture + mirror (10); and the same for
the 3-seed average (3 models, run one after another). Climatology shrinkage is one fused
multiply-add per frame and is included where named. Round 3: mirror average of single decodes,
optionally re-sharpened (postproc3/sharp.py: band powers of every single decode, one FFT pair
on the average). Round 4: mirror average re-mapped by the localized probability-matched mean
onto the single decodes (sharp.lpm, on the GPU).

    python bench_ctkae_inference.py OUT.json RUN_s0 RUN_s1 RUN_s2 [--only SUBSTRING]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "postproc3"))
from pp2_lab import REPO, dev, forecast, mirror  # noqa: E402
from sharp import band_power, lpm, radial_bins, sharpen  # noqa: E402

from eval_ctkae import load_model  # noqa: E402
from models.ct_data import load_eval_case  # noqa: E402

STEPS, WARMUP, REPS = 240, 2, 5


def run_once(models, x2, ma, mask, sigma, use_mirror, shrink, post=None):
    taus = 0.1 * torch.arange(1, STEPS + 1, device=dev, dtype=torch.float32)[None]
    mmask = mask.flip(-2)
    out, e, singles = 0.0, 0.0, []
    clim = torch.zeros_like(x2[:, :1])
    H, W = mask.shape[-2:]
    rb = radial_bins(H, W, dev)
    with torch.no_grad():
        for m in models:
            z = m.encode(x2, mask, ma)
            p = forecast(m, z, taus, ma, mask, sigma)
            if use_mirror:
                zm = m.encode(mirror(x2), mmask, ma)
                pm = mirror(forecast(m, zm, taus, ma, mmask, sigma)) * mask
                if post == "sharp":
                    e = e + band_power(p, clim, rb) + band_power(pm, clim, rb)
                if post == "lpm":
                    singles += [p, pm]
                p = 0.5 * (p + pm)
            out = out + p / len(models)
        if shrink is not None:
            g, clim = shrink
            out = clim + g * (out - clim)
        if post == "sharp":
            out = sharpen(out, e / (2 * len(models)), clim, mask, rb)
        if post == "lpm":
            out = lpm(out, singles, mask.reshape(H, W))
    return out


def time_cfg(models, x2, ma, mask, sigma, use_mirror, shrink, post=None):
    for _ in range(WARMUP):
        run_once(models, x2, ma, mask, sigma, use_mirror, shrink, post)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ms = []
    for _ in range(REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        torch.cuda.synchronize()
        a.record()
        run_once(models, x2, ma, mask, sigma, use_mirror, shrink, post)
        b.record()
        torch.cuda.synchronize()
        ms.append(a.elapsed_time(b))
    peak = (
        torch.cuda.max_memory_allocated() / 2**20
    )  # weights included, as for the baselines
    med = float(np.median(ms))
    return {
        "rollout_ms": med,
        "ms_per_step": med / STEPS,
        "peak_vram_MB": peak,
        "all_ms": ms,
    }


def main():
    args = sys.argv[1:]
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = args[i + 1]
        args = args[:i] + args[i + 2 :]
    out, runs = args[0], args[1:]
    models = []
    for r in runs:
        m, _ = load_model(
            Path(REPO / r) if not Path(r).is_absolute() else Path(r), "last.pth", dev
        )
        models.append(m)
    sig = json.load(open((REPO / runs[0]) / "mix_sigma.json"))["sigma"]
    x, ma, mask = load_eval_case(str(REPO / "data/acdm/128_tra/gt_longer.nc"), dev)
    x2, ma = x[:1, :2], ma[:1]
    shrink = (
        torch.full((1, STEPS, 4, 1, 1), 0.9, device=dev),
        torch.zeros_like(x2[:, :1]),
    )
    weights_MB = (
        sum(p.numel() * p.element_size() for p in models[0].parameters()) / 2**20
    )
    cfgs: list = [
        ("single decode", models[:1], 0.0, False, None),
        ("mixture", models[:1], sig, False, None),
        ("mixture + mirror", models[:1], sig, True, None),
        ("mixture + mirror + shrink", models[:1], sig, True, shrink),
        ("3 seeds, single decode", models, 0.0, False, None),
        ("3 seeds, mixture", models, sig, False, None),
        ("3 seeds, mixture + mirror + shrink", models, sig, True, shrink),
        ("mirror", models[:1], 0.0, True, None),
        ("mirror + sharpen", models[:1], 0.0, True, None, "sharp"),
        ("3 seeds, mirror", models, 0.0, True, None),
        ("3 seeds, mirror + sharpen", models, 0.0, True, None, "sharp"),
        ("mirror + lpm", models[:1], 0.0, True, None, "lpm"),
        ("3 seeds, mirror + lpm", models, 0.0, True, None, "lpm"),
    ]
    if only:
        cfgs = [c for c in cfgs if only in c[0]]
    rows = []
    for name, ms_, sg, mir, sh, *shp in cfgs:
        row = {
            "readout": name,
            "decodes_per_step": len(ms_) * (5 if sg > 0 else 1) * (2 if mir else 1),
            **time_cfg(ms_, x2, ma, mask, sg, mir, sh, shp[0] if shp else None),
        }
        rows.append(row)
        print(
            f"{name:36s} {row['rollout_ms']:9.1f} ms  {row['ms_per_step']:6.2f} ms/step  "
            f"{row['peak_vram_MB']:7.1f} MB",
            flush=True,
        )
    json.dump(
        {
            "protocol": {
                "steps": STEPS,
                "batch": 1,
                "warmup": WARMUP,
                "reps": REPS,
                "statistic": "median",
                "gpu": torch.cuda.get_device_name(0),
                "testset": "longer (transonic Ma 0.64-0.65)",
                "decode": "bf16 autocast, 32 frames per decoder call",
                "vram": "peak allocated, all loaded models included",
                "weights_MB_per_model": weights_MB,
                "runs": runs,
                "sigma": sig,
            },
            "rows": rows,
        },
        open(out, "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
