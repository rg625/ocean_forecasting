# ruff: noqa: F841
# mypy: disable-error-code="dict-item"
"""Error against WALL-CLOCK for all three methods, and a high-sample uncertainty run.

Comparing "1000 optimisation iterations" to "8 posterior samples" is not a comparison: the
two budgets buy different things and cost different amounts. The honest version is an
error-versus-time curve, where each method is swept over its own natural budget knob and
both axes are measured:

    KAE / U-Net : optimisation iterations
    SDA         : predictor steps N, and number of posterior samples

Also draws a large posterior ensemble so that coverage and calibration are estimated from
enough samples to mean something.

    python -m data_assimilation.ks.cost_accuracy --out da_results_sda_paper/cost_accuracy.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from data_assimilation.ks.protocol import DT, build_problem, rel_l2
from data_assimilation.ks.sda_paper import LinearObservation
from data_assimilation.ks.da_ks_experiments_3way import Bench, CANON

logger = logging.getLogger("cost_accuracy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, default=Path("data/ks/da_test.nc"))
    ap.add_argument("--n-problems", type=int, default=32)
    ap.add_argument("--L", type=int, default=56)
    ap.add_argument(
        "--big-samples",
        type=int,
        default=96,
        help="ensemble size for the calibration run",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("da_results_sda_paper/cost_accuracy.json")
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--kae-run",
        type=Path,
        default=Path("model_outputs_ks/continous_linear_128/rollout_10"),
    )
    ap.add_argument(
        "--kae-tuning",
        type=Path,
        default=None,
        help="frozen KAE settings (lr and first guess) to measure with, so "
        "the curve matches the tables rather than the old campaign",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    import argparse as _a

    ns = _a.Namespace(
        test=args.test,
        kae_run=args.kae_run,
        unet_ckpt=Path("model_outputs_ks/unet1d/rollout10_extended2/best_model.pth"),
        sda_config=Path("da_results_sda_paper/frozen_config.json"),
        tuning=Path("da_results_v2/tuning.json"),
        init_scales=Path("da_results_v2/init_scales.json"),
        device=args.device,
        n_samples=1,
        chunk=4,
        iters_scale=1.0,
        iters_override=0,
        iters_kae=0,
        iters_unet=0,
    )
    b = Bench(ns)
    if args.kae_tuning and args.kae_tuning.exists():
        b.hp["KAE-expm"].update(json.loads(args.kae_tuning.read_text())["best"])
        logger.info(f"KAE settings from {args.kae_tuning}: {b.hp['KAE-expm']}")
    prob = build_problem(
        b.data,
        name="cost",
        n_problems=args.n_problems,
        taus=np.array(CANON) * DT,
        seed=123,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def flush(o):
        """Persist after every measurement. A crash then costs one point, not the run."""
        args.out.write_text(json.dumps(o, indent=2))

    out = {
        "protocol": {
            "n_problems": args.n_problems,
            "taus": (np.array(CANON) * DT).tolist(),
            "note": (
                "wall-clock measured on the same GPU, same batch of "
                "assimilation problems, for every point"
            ),
            "kae_run": str(args.kae_run),
            "kae_tuning": str(args.kae_tuning),
        },
        "curves": {},
    }

    # ---- 4D-Var: error vs iterations, with time measured ----------------------
    for m in ["KAE-expm", "UNet"]:
        pts = []
        grid = (
            [250, 500, 1000, 2000, 4000, 8000, 16000]
            if m == "KAE-expm"
            else [250, 500, 1000, 2000, 4000, 8000]
        )
        for it in grid:
            r = b.run(m, prob, iters=it, seed=123)
            pts.append(
                {
                    "budget": it,
                    "wall_s": float(r["wall_s"]),
                    "rel": float(r["rel"].mean()),
                    "sem": float(r["rel"].std(ddof=1) / np.sqrt(len(r["rel"]))),
                }
            )
            logger.info(
                f"{m:9s} iters={it:6d}  {r['wall_s']:7.1f}s  "
                f"rel {pts[-1]['rel']:.4f} +/- {pts[-1]['sem']:.4f}"
            )
            out["curves"][m] = pts
            flush(out)

    # ---- SDA: error vs predictor steps, and vs ensemble size ------------------
    dev = b.dev
    frames = torch.as_tensor(
        np.round(np.array(CANON) * DT / DT).astype(int), device=dev
    )
    mask = torch.as_tensor(prob.mask, device=dev)
    y = torch.as_tensor(prob.y, device=dev).permute(1, 0, 2).contiguous()
    sig = prob.noise_std if prob.noise_std > 0 else b.sda_cfg["sigma_y_clean"]
    obs = LinearObservation(frames, mask, sig)
    sim = torch.as_tensor(prob.sim, device=dev)
    t0 = torch.as_tensor(prob.t0, device=dev)
    truth0 = b.data.frames(sim, t0)
    B = len(prob.sim)

    def sda_run(n_steps, n_samples, chunk=4):
        torch.cuda.synchronize()
        t = time.perf_counter()
        ch = []
        for i in range(0, B, chunk):
            sl = slice(i, min(i + chunk, B))
            ch.append(
                b.sda.sample(
                    args.L,
                    b.data.X,
                    sl.stop - sl.start,
                    y=y[sl],
                    obs=LinearObservation(frames, mask, sig),
                    n_steps=n_steps,
                    corrections=b.sda_cfg["corrections"],
                    tau=b.sda_cfg["tau"],
                    seed=123 + 1000 * i,
                    n_samples=n_samples,
                )
            )
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
        smp = torch.cat(ch, dim=1)  # [S, B, L, X]
        per = torch.stack(
            [
                rel_l2(b.data.denorm(smp[s][:, 0]), b.data.denorm(truth0))
                for s in range(smp.shape[0])
            ]
        )
        return smp, wall, per

    pts = []
    for N in [16, 32, 64, 128, 256]:
        smp, wall, per = sda_run(N, 8)
        v = float(per.mean())
        pts.append(
            {
                "budget": N,
                "kind": "predictor steps (8 draws)",
                "wall_s": wall,
                "rel": v,
                "sem": float(per.mean(0).cpu().numpy().std(ddof=1) / np.sqrt(B)),
            }
        )
        logger.info(f"SDA N={N:4d} (8 draws)  {wall:7.1f}s  rel/draw {v:.4f}")
        out["curves"]["SDA"] = pts
        flush(out)

    # ---- large ensemble for calibration ---------------------------------------
    logger.info(f"drawing {args.big_samples} posterior samples for calibration ...")
    smp, wall, per = sda_run(b.sda_cfg["n_steps"], args.big_samples, chunk=2)
    truthW = torch.stack([b.data.frames(sim, t0 + k) for k in range(args.L)], dim=1)
    dn = b.data.denorm
    S = smp.shape[0]
    q = {}
    for lvl in [50, 75, 90, 95]:
        lo = np.percentile(dn(smp).cpu().numpy(), (100 - lvl) / 2, axis=0)
        hi = np.percentile(dn(smp).cpu().numpy(), 100 - (100 - lvl) / 2, axis=0)
        t_ = dn(truthW).cpu().numpy()
        q[str(lvl)] = float(((t_ >= lo) & (t_ <= hi)).mean())
        logger.info(f"   nominal {lvl}%  ->  empirical {100 * q[str(lvl)]:.1f}%")
    out["calibration"] = {
        "n_samples": S,
        "wall_s": wall,
        "nominal_vs_empirical": q,
        "per_draw_analysis_rel": float(per.mean()),
        "ensemble_mean_analysis_rel": float(
            rel_l2(dn(smp.mean(0)[:, 0]), dn(truth0)).mean()
        ),
        "n_problems": B,
    }
    flush(out)
    logger.info(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
