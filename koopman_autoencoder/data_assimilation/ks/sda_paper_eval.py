# ruff: noqa: F841
"""Shared evaluation for paper-faithful SDA: posterior sampling + metrics.

Used by the validation tuner and by the final test campaign, so both score identically.
Everything is computed from posterior *samples*; the mean is one summary among several and
never replaces the ensemble.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from data_assimilation.ks.protocol import DT, KSData, Problem, rel_l2
from data_assimilation.ks.sda_paper import SDA, LinearObservation


@dataclass
class SamplerConfig:
    n_steps: int = 256
    corrections: int = 1
    tau: float = 0.5
    n_samples: int = 8
    sigma_y_clean: float = 0.05  # regulariser when observations are noise-free
    gamma_mode: str = "appendix_b"  # "appendix_b" | "scalar" | "none"
    gamma_scale: float = 1e-2  # used by "scalar"; the paper's practical fallback
    gamma_floor: float = 1e-2  # eigenvalue floor on the Appendix-B Gamma
    chunk: int = 8
    seed: int = 0


def configure(sda: SDA, cfg: SamplerConfig, Gamma: Optional[torch.Tensor]):
    """Point the sampler at the requested Gamma variant."""
    if cfg.gamma_mode == "appendix_b":
        sda.Gamma, sda.gamma_floor = Gamma, cfg.gamma_floor
    elif cfg.gamma_mode == "scalar":
        sda.Gamma, sda.gamma_scale = None, cfg.gamma_scale
    elif cfg.gamma_mode == "none":
        sda.Gamma, sda.gamma_scale = None, 0.0
    else:
        raise ValueError(cfg.gamma_mode)
    return sda


@torch.no_grad()
def truth_window(data: KSData, problem: Problem, L: int) -> torch.Tensor:
    sim = torch.as_tensor(problem.sim, device=data.device)
    t0 = torch.as_tensor(problem.t0, device=data.device)
    return torch.stack([data.frames(sim, t0 + k) for k in range(L)], dim=1)  # [B, L, X]


def draw(
    sda: SDA,
    data: KSData,
    problem: Problem,
    L: int,
    cfg: SamplerConfig,
    prior: bool = False,
) -> Dict:
    """Posterior (or prior) trajectory samples for every problem. -> dict with [S,B,L,X]."""
    dev = data.device
    B = len(problem.sim)
    X = data.X
    frames = torch.as_tensor(np.round(problem.taus / DT).astype(int), device=dev)
    assert int(frames.max()) < L, "observation outside the requested window"
    mask = torch.as_tensor(problem.mask, device=dev)
    sigma_y = problem.noise_std if problem.noise_std > 0 else cfg.sigma_y_clean
    obs = LinearObservation(frames, mask, sigma_y)
    y = torch.as_tensor(problem.y, device=dev).permute(1, 0, 2).contiguous()  # [B,m,X]

    sda.reset_counters()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    chunks = []
    for i in range(0, B, cfg.chunk):
        sl = slice(i, min(i + cfg.chunk, B))
        o = LinearObservation(frames, mask, sigma_y)
        s = sda.sample(
            L,
            X,
            sl.stop - sl.start,
            y=None if prior else y[sl],
            obs=None if prior else o,
            n_steps=cfg.n_steps,
            corrections=cfg.corrections,
            tau=cfg.tau,
            seed=cfg.seed + 1000 * i,
            n_samples=cfg.n_samples,
        )
        chunks.append(s)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t_start
    peak = (
        torch.cuda.max_memory_allocated() / 2**20
        if dev.type == "cuda"
        else float("nan")
    )
    samples = torch.cat(chunks, dim=1)  # [S, B, L, X]
    return {
        "samples": samples,
        "wall_s": wall,
        "peak_mem_MiB": peak,
        "nfe": sda.nfe,
        "n_segment_evals": sda.n_segment_evals,
        "n_backward": sda.n_backward,
        "frames": frames.cpu().numpy(),
        "sigma_y": sigma_y,
    }


@torch.no_grad()
def score(res: Dict, data: KSData, problem: Problem, L: int) -> Dict:
    """Point-estimate and posterior metrics from the sample ensemble."""
    dev = data.device
    samples = res["samples"]  # [S, B, L, X]
    S, B = samples.shape[0], samples.shape[1]
    truth = truth_window(data, problem, L)  # [B, L, X]
    obs_f = res["frames"]
    unobs_f = np.array([i for i in range(L) if i not in set(obs_f.tolist())])
    dn = data.denorm
    mean = samples.mean(0)
    spread = samples.std(0)

    def rl(a, b):
        return rel_l2(dn(a), dn(b))

    out = {
        "analysis_rel_l2": rl(mean[:, 0], truth[:, 0]).cpu().numpy(),
        "spacetime_rel_l2": rl(mean, truth).mean(1).cpu().numpy(),
        "obs_frame_rel_l2": rl(mean[:, obs_f], truth[:, obs_f]).mean(1).cpu().numpy(),
        "unobs_frame_rel_l2": rl(mean[:, unobs_f], truth[:, unobs_f])
        .mean(1)
        .cpu()
        .numpy(),
        "per_sample_analysis": torch.stack(
            [rl(samples[s][:, 0], truth[:, 0]) for s in range(S)]
        )
        .cpu()
        .numpy(),
        "n_samples": S,
        "n_problems": B,
        "obs_frames": obs_f,
        "unobs_frames": unobs_f,
    }
    # observation-space fit at the observed entries
    mask = torch.as_tensor(problem.mask, device=dev)
    y = torch.as_tensor(problem.y, device=dev)
    pred = dn(mean[:, torch.as_tensor(obs_f, device=dev)]).permute(1, 0, 2)
    d = (pred - dn(y)) * mask[:, None, :]
    num = torch.linalg.vector_norm(d, dim=-1)
    den = torch.linalg.vector_norm(dn(y) * mask[:, None, :], dim=-1).clamp_min(1e-12)
    out["obs_rel_l2"] = (num / den).mean(0).cpu().numpy()

    # posterior quality
    err = (dn(mean) - dn(truth)).abs()
    sp = data.std * spread
    out["mean_spread"] = float(sp.mean())
    out["mean_abs_error"] = float(err.mean())
    out["spread_error_ratio"] = float(sp.mean() / max(float(err.mean()), 1e-12))
    out["coverage_95"] = float((err <= 1.96 * sp.clamp_min(1e-12)).float().mean())
    out["coverage_50"] = float((err <= 0.674 * sp.clamp_min(1e-12)).float().mean())
    # credible-interval width, and spread split by observed / unobserved frames
    out["spread_obs_frames"] = float(sp[:, obs_f].mean())
    out["spread_unobs_frames"] = float(sp[:, unobs_f].mean())
    # trajectory-space variability between independent posterior draws
    if S > 1:
        pd = [
            float((samples[i] - samples[j]).pow(2).mean().sqrt())
            for i in range(S)
            for j in range(i + 1, S)
        ]
        out["mean_pairwise_sample_rmse"] = float(np.mean(pd))
    for k in (
        "wall_s",
        "peak_mem_MiB",
        "nfe",
        "n_segment_evals",
        "n_backward",
        "sigma_y",
    ):
        out[k] = res[k]
    return out
