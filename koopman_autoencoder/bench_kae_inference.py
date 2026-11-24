"""Inference cost of the continuous KAE under the same protocol as the baselines.

Mirrors ``autoreg_pde_diffusion/src/bench_inference.py`` exactly: transonic ``longer``
test case, 240 predicted timesteps, batch 1, CUDA events with synchronisation, a warm-up
followed by timed repeats, median reported, peak VRAM from ``max_memory_allocated``.

Times both ways of advancing the same learned generator:
  * ``rk4``  -- the model's own fourth-order Runge-Kutta latent rollout (240 steps),
  * ``expm`` -- closed-form ``exp(K tau)`` evaluated at each queried output time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate import EVAL_CASES, EvalConfig, KoopmanEvaluator

STEPS = 240


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "../autoreg_pde_diffusion/src/results/metrics_dist/"
            "inference_cost_kae.json"
        ),
    )
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    cfg = EvalConfig(
        model_arch="linear",
        model_type="continous",
        dimension=128,
        regime="tra",
        ckpt_index=199,
        base_output_dir=Path("./model_outputs_tra"),
        eval_case="longer",
    )
    ev = KoopmanEvaluator(cfg)
    model, dev = ev.model, ev.device
    assert model is not None

    steps = EVAL_CASES["longer"]["rollout_steps"]
    input_seq, _, meta = ev.val_dataset[0, steps]
    input_seq["obstacle_mask"] = (
        meta["obstacle_mask"][0].repeat(*input_seq.batch_size, 1, 1).to(dev)
    )
    input_seq["cond_input"] = (
        meta["cond_input"][0].repeat(*input_seq.batch_size).to(dev)
    )
    batch = input_seq.unsqueeze(0).to(dev)

    rows = []
    for label, fn in [
        ("KAE (RK4 rollout)", lambda: model(batch, seq_length=steps)),
        (
            "KAE (exact expm)",
            lambda: model.forward_theoretical(batch, seq_length=steps),
        ),
    ]:
        times = []
        torch.cuda.reset_peak_memory_stats(dev)
        with torch.no_grad():
            for i in range(a.warmup + a.reps):
                torch.cuda.synchronize()
                s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                s.record()
                fn()
                e.record()
                torch.cuda.synchronize()
                if i >= a.warmup:
                    times.append(s.elapsed_time(e))
        peak = torch.cuda.max_memory_allocated(dev) / 1024**2
        med = float(np.median(times))
        rows.append(
            {
                "model": label,
                "rollout_ms": med,
                "ms_per_step": med / steps,
                "peak_vram_MB": float(peak),
                "reps": a.reps,
                "all_ms": [float(t) for t in times],
            }
        )
        print(
            f"  {label:<20} {med:10.1f} ms / {steps} steps = "
            f"{med / steps:8.3f} ms/step   peak {peak:8.1f} MB"
        )

    n_par = sum(p.numel() for p in model.parameters())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            {
                "protocol": {
                    "steps": steps,
                    "batch": 1,
                    "warmup": a.warmup,
                    "reps": a.reps,
                    "statistic": "median",
                    "gpu": torch.cuda.get_device_name(0),
                    "testset": "longer (transonic Ma 0.64-0.65)",
                    "checkpoint": str(cfg.checkpoint_path),
                },
                "parameters": int(n_par),
                "rows": rows,
            },
            indent=1,
        )
    )
    print(f"parameters: {n_par:,}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
