# mypy: disable-error-code="arg-type, return-value"
"""Score-based DA evaluation on the frozen 64-trajectory held-out DA benchmark.

Runs only after the trained score model has passed every gate in ``data_assimilation/ks/test_sda.py``.
The assimilation problems, observation times, spatial masks, observation values and noise
realisations are reconstructed with the *same* seeds and the same ``data_assimilation.ks.protocol`` code the
two 4D-Var methods used, so all three methods receive identical information.

Because SDA infers a whole trajectory window rather than an initial condition, the window
must cover both the assimilation times and the post-assimilation forecast leads that are
to be reported; leads beyond the window are not evaluated and are reported as such rather
than extrapolated.

    python -m data_assimilation.ks.run_sda --ckpt model_outputs_ks/sda/win56/best_model.pth \
        --out-dir da_results_v2 --n-samples 16
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from data_assimilation.ks.evaluate import summarize
from data_assimilation.ks.protocol import DT, KSData, build_problem, rel_l2
from data_assimilation.ks.test_sda import build

logger = logging.getLogger("data_assimilation.ks.run_sda")

CANONICAL_TAUS = np.array([0.1, 0.3, 0.7, 1.5, 2.5])
# Identical to run_da_suite.FORECAST_TAUS. It must be passed to build_problem so
# that sample_analysis_points draws exactly the same (sim, t0) pairs the two
# 4D-Var methods were evaluated on; omitting it silently changes the problems.
FORECAST_TAUS = np.round(np.arange(3.0, 20.01, 0.5), 3)


def _rel(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Relative L2 over the last axis."""
    return rel_l2(pred, true)


def evaluate_setting(
    sda,
    data: KSData,
    prob,
    L: int,
    n_samples: int,
    sigma_y: float,
    chunk: int,
    seed: int,
) -> Dict:
    """Posterior and prior statistics for one observation setting."""
    dev = data.device
    sim = torch.as_tensor(prob.sim, device=dev)
    t0 = torch.as_tensor(prob.t0, device=dev)
    B = len(prob.sim)

    obs_frames = np.round(prob.taus / DT).astype(int)
    assert obs_frames.max() < L, "window too short for these observation times"
    unobs_frames = np.array([i for i in range(L) if i not in set(obs_frames.tolist())])

    y = torch.as_tensor(prob.y, device=dev)  # [n_obs, B, X]
    mask = torch.as_tensor(prob.mask, device=dev)  # [n_obs, X]
    truth = torch.stack(
        [data.frames(sim, t0 + k) for k in range(L)], dim=1
    )  # [B, L, X]

    out: Dict = {}
    for tag, prior in [("posterior", False), ("prior", True)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        chunks: List[torch.Tensor] = []
        nfe = nbw = 0
        for i in range(0, B, chunk):
            sl = slice(i, min(i + chunk, B))
            s = sda.sample(
                obs_frames,
                y[:, sl],
                mask,
                sigma_y,
                n_samples,
                sl.stop - sl.start,
                seed=seed + i,
                return_prior=prior,
            )
            chunks.append(s)  # [N, b, L, X]
            nfe += sda.nfe
            nbw += sda.n_backward
        torch.cuda.synchronize()
        wall = time.perf_counter() - t_start
        peak = torch.cuda.max_memory_allocated() / 2**20
        samples = torch.cat(chunks, dim=1)  # [N, B, L, X]

        mean = samples.mean(0)  # [B, L, X]
        spread = samples.std(0)  # [B, L, X]
        dn_mean, dn_truth = data.denorm(mean), data.denorm(truth)

        r = {
            "analysis_rel_l2": _rel(dn_mean[:, 0], dn_truth[:, 0]).cpu().numpy(),
            "spacetime_rel_l2": _rel(dn_mean, dn_truth).mean(1).cpu().numpy(),
            "obs_frame_rel_l2": _rel(dn_mean[:, obs_frames], dn_truth[:, obs_frames])
            .mean(1)
            .cpu()
            .numpy(),
            "unobs_frame_rel_l2": _rel(
                dn_mean[:, unobs_frames], dn_truth[:, unobs_frames]
            )
            .mean(1)
            .cpu()
            .numpy(),
            "wall_s": wall,
            "peak_mem_MiB": peak,
            "nfe": nfe,
            "n_backward": nbw,
            "n_samples": n_samples,
            "time_per_sample_s": wall / max(1, n_samples * int(np.ceil(B / chunk))),
        }
        # observation-space fit at the observed entries only
        pred_obs = dn_mean[:, obs_frames].permute(1, 0, 2)  # [n_obs, B, X]
        dy = (pred_obs - data.denorm(y)) * mask[:, None, :]
        num = torch.linalg.vector_norm(dy, dim=-1)
        den = torch.linalg.vector_norm(
            data.denorm(y) * mask[:, None, :], dim=-1
        ).clamp_min(1e-12)
        r["obs_rel_l2"] = (num / den).mean(0).cpu().numpy()

        # post-assimilation forecast: frames strictly beyond the last observation.
        # The window ends at tau = (L-1)*DT, so SDA's forecast is only defined on that
        # overlap with the 4D-Var methods' forecast range; leads beyond it are not
        # evaluated rather than extrapolated.
        fcf = np.arange(int(obs_frames.max()) + 1, L)
        if fcf.size:
            r["forecast_rel_l2"] = (
                _rel(dn_mean[:, fcf], dn_truth[:, fcf]).mean(1).cpu().numpy()
            )
            r["forecast_tau_lo"] = float(fcf.min() * DT)
            r["forecast_tau_hi"] = float(fcf.max() * DT)

        # per-sample (not ensemble-mean) error, for an honest single-draw comparison
        per = torch.stack(
            [
                _rel(data.denorm(samples[s][:, 0]), dn_truth[:, 0])
                for s in range(samples.shape[0])
            ]
        )  # [N, B]
        r["analysis_rel_l2_per_sample"] = per.mean(0).cpu().numpy()

        # ensemble spread and its calibration against the actual error
        err = (dn_mean - dn_truth).abs()
        sp = data.std * spread
        r["mean_spread"] = float(sp.mean())
        r["mean_abs_error"] = float(err.mean())
        r["spread_error_ratio"] = float(sp.mean() / max(err.mean(), 1e-12))
        # fraction of points inside the ensemble +/- 1.96 sigma interval
        cover = (err <= 1.96 * sp.clamp_min(1e-12)).float().mean()
        r["coverage_95"] = float(cover)
        out[tag] = r
        logger.info(
            f"  {tag:9s} | analysis {r['analysis_rel_l2'].mean():.4f} "
            f"| obs-frames {r['obs_frame_rel_l2'].mean():.4f} "
            f"| unobserved-frames {r['unobs_frame_rel_l2'].mean():.4f} "
            f"| forecast {r.get('forecast_rel_l2', np.array([np.nan])).mean():.4f} "
            f"| spread/err {r['spread_error_ratio']:.2f} | cov95 {r['coverage_95']:.3f} "
            f"| {wall:.1f}s, NFE {nfe}, backward {nbw}, {peak:.0f} MiB"
        )

    # post-assimilation forecast: frames strictly beyond the last observation
    fc = np.arange(int(obs_frames.max()) + 1, L)
    if fc.size:
        for tag in ("posterior", "prior"):
            pass
        out["forecast_frames"] = fc
        out["forecast_taus"] = fc * DT
    return out, obs_frames, unobs_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", type=Path, default=Path("model_outputs_ks/sda/win56/best_model.pth")
    )
    ap.add_argument("--test", type=Path, default=Path("data/ks/da_test.nc"))
    ap.add_argument("--out-dir", type=Path, default=Path("da_results_v2"))
    ap.add_argument("--n-problems", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--sample-steps", type=int, default=256)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument(
        "--sigma-y-clean",
        type=float,
        default=0.05,
        help="likelihood sigma for noise-free observations "
        "(validation-selected regulariser)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    dev = torch.device(args.device)
    data = KSData(args.test, dev)
    sda, L = build(args.ckpt, 56, dev, args.sample_steps)
    sda.guidance = args.guidance
    logger.info(
        f"score model window L={L} ({(L - 1) * DT:.1f} time units), "
        f"{args.sample_steps} DDIM steps, guidance {args.guidance}, "
        f"{args.n_samples} posterior samples per problem"
    )

    rows: List[Dict] = []
    raw: Dict[str, np.ndarray] = {}
    summary: Dict = {
        "ckpt": str(args.ckpt),
        "window": L,
        "sample_steps": args.sample_steps,
        "guidance": args.guidance,
        "n_samples": args.n_samples,
        "test_file": str(args.test),
    }

    for tag, frac, noise in [("clean", 1.0, 0.0), ("sparse_noisy", 0.25, 0.05)]:
        # rebuilt with the campaign's seed so the observations are identical
        prob = build_problem(
            data,
            name=f"canonical_{tag}",
            n_problems=args.n_problems,
            taus=CANONICAL_TAUS,
            obs_frac=frac,
            noise_std=noise,
            seed=args.seed,
            forecast_taus=FORECAST_TAUS,
        )
        logger.info(
            f"=== SDA on canonical_{tag} "
            f"(obs_frac={frac}, sigma={noise}, N={len(prob.sim)}) ==="
        )
        # For noise-free observations the Gaussian likelihood is degenerate and the
        # guidance diverges; sigma_y then acts as a regulariser and was selected on
        # validation (da_results_v2/sda_sigma_tuning.json), not on the test set.
        sigma_y = noise if noise > 0 else args.sigma_y_clean
        res, obs_f, unobs_f = evaluate_setting(
            sda, data, prob, L, args.n_samples, sigma_y, args.chunk, args.seed
        )
        for kind in ("posterior", "prior"):
            r = res[kind]
            row = {
                "stage": "sda",
                "condition": "setting",
                "cond_value": tag,
                "method": "SDA" if kind == "posterior" else "SDA-prior",
                "n_problems": len(prob.sim),
                "n_obs": len(prob.taus),
                "obs_frac": frac,
                "noise_std": noise,
                "window": L,
                "n_samples": args.n_samples,
                "sample_steps": args.sample_steps,
            }
            for k in (
                "analysis_rel_l2",
                "obs_rel_l2",
                "spacetime_rel_l2",
                "obs_frame_rel_l2",
                "unobs_frame_rel_l2",
                "analysis_rel_l2_per_sample",
                "forecast_rel_l2",
            ):
                if k not in r:
                    continue
                st = summarize(r[k])
                row.update(
                    {
                        f"{k}_{m}": v
                        for m, v in st.items()
                        if m in ("mean", "std", "median", "sem")
                    }
                )
                raw[f"{tag}__{kind}__{k}"] = r[k]
            for k in (
                "wall_s",
                "peak_mem_MiB",
                "nfe",
                "n_backward",
                "time_per_sample_s",
                "mean_spread",
                "mean_abs_error",
                "spread_error_ratio",
                "coverage_95",
                "forecast_tau_lo",
                "forecast_tau_hi",
            ):
                if k not in r:
                    continue
                row[k] = r[k]
            rows.append(row)
        # check 9 as a first-class result
        d_obs = (
            res["prior"]["obs_frame_rel_l2"].mean()
            - res["posterior"]["obs_frame_rel_l2"].mean()
        )
        d_un = (
            res["prior"]["unobs_frame_rel_l2"].mean()
            - res["posterior"]["unobs_frame_rel_l2"].mean()
        )
        summary.setdefault("prior_vs_posterior", {})[tag] = {
            "obs_frames_prior": float(res["prior"]["obs_frame_rel_l2"].mean()),
            "obs_frames_posterior": float(res["posterior"]["obs_frame_rel_l2"].mean()),
            "obs_frames_improvement": float(d_obs),
            "unobs_frames_prior": float(res["prior"]["unobs_frame_rel_l2"].mean()),
            "unobs_frames_posterior": float(
                res["posterior"]["unobs_frame_rel_l2"].mean()
            ),
            "unobs_frames_improvement": float(d_un),
            "n_observed_frames": int(len(obs_f)),
            "n_unobserved_frames": int(len(unobs_f)),
            "genuine_trajectory_inference": bool(d_un > 0.01),
        }
        logger.info(
            f"  CHECK 9 | observed frames {res['prior']['obs_frame_rel_l2'].mean():.4f}"
            f" -> {res['posterior']['obs_frame_rel_l2'].mean():.4f} ({d_obs:+.4f}); "
            f"never-observed frames "
            f"{res['prior']['unobs_frame_rel_l2'].mean():.4f} -> "
            f"{res['posterior']['unobs_frame_rel_l2'].mean():.4f} ({d_un:+.4f})"
        )
        raw[f"{tag}__obs_frames"] = obs_f
        raw[f"{tag}__unobs_frames"] = unobs_f
        raw[f"{tag}__sim"] = prob.sim
        raw[f"{tag}__t0"] = prob.t0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "raw").mkdir(exist_ok=True)
    np.savez_compressed(args.out_dir / "raw" / "sda.npz", **raw)
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.out_dir / "sda_final.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(args.out_dir / "sda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    logger.info(f"saved -> {args.out_dir}/sda_final.csv, sda_summary.json")


if __name__ == "__main__":
    main()
