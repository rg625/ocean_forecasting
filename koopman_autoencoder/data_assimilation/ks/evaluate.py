# mypy: disable-error-code="no-any-return"
"""Metrics for an assimilation solution.

Everything is computed on **denormalised** fields so numbers are comparable across
methods.  All errors are per-assimilation-problem relative L2, returned as arrays of
length B so that trajectory-level statistics (and paired tests) are possible.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from data_assimilation.ks.protocol import DT, KSData, Problem, masked_rel_l2, rel_l2


@torch.no_grad()
def evaluate_solution(
    method,
    control: torch.Tensor,
    problem: Problem,
    data: KSData,
    window_frames: Optional[int] = None,
    forecast_taus: Optional[Sequence[float]] = None,
) -> Dict:
    """Analysis, observation-fit, space-time and post-assimilation forecast errors."""
    dev = data.device
    sim = torch.as_tensor(problem.sim, device=dev)
    t0 = torch.as_tensor(problem.t0, device=dev)
    out: Dict = {}

    # 1. analysis error at the unobserved t0 --------------------------------
    x_hat0 = data.denorm(method.analysis_field(control))
    x_true0 = data.denorm(data.frames(sim, t0))
    out["init_rel_l2"] = rel_l2(x_hat0, x_true0).cpu().numpy()

    # 2. observation fit (at the observed entries only) ---------------------
    prep = method.prepare(problem.taus)
    pred_obs = method.predict_at(control, problem.taus, prep)  # [n_obs, B, X]
    y = torch.as_tensor(problem.y, device=dev)
    mask = torch.as_tensor(problem.mask, device=dev)
    obs_err = torch.stack(
        [
            masked_rel_l2(data.denorm(pred_obs[i]), data.denorm(y[i]), mask[i])
            for i in range(len(problem.taus))
        ]
    )
    out["obs_rel_l2"] = obs_err.mean(0).cpu().numpy()  # [B]
    out["obs_rel_l2_per_tau"] = obs_err.cpu().numpy()  # [n_obs, B]

    # 3. space-time reconstruction over the assimilation window -------------
    n_win = window_frames or int(round(problem.taus.max() / DT)) + 1
    taus_win = np.arange(n_win) * DT
    pred_win = method.forecast(control, taus_win)  # [n_win, B, X]
    true_win = torch.stack([data.frames(sim, t0 + k) for k in range(n_win)])
    st = rel_l2(data.denorm(pred_win), data.denorm(true_win))  # [n_win, B]
    out["spacetime_rel_l2"] = st.mean(0).cpu().numpy()
    out["window_rel_l2_curve"] = st.cpu().numpy()
    out["window_taus"] = taus_win

    # 4. post-assimilation forecast (strictly beyond the last observation) --
    ft = np.asarray(
        forecast_taus if forecast_taus is not None else problem.forecast_taus,
        dtype=float,
    )
    if ft.size:
        pred_f = method.forecast(control, ft)
        n_f = np.round(ft / DT).astype(int)
        true_f = torch.stack([data.frames(sim, t0 + int(k)) for k in n_f])
        fe = rel_l2(data.denorm(pred_f), data.denorm(true_f))  # [n_f, B]
        out["forecast_rel_l2_curve"] = fe.cpu().numpy()
        out["forecast_taus"] = ft
        out["forecast_rel_l2"] = fe.mean(0).cpu().numpy()
    return out


@torch.no_grad()
def ae_reconstruction_floor(model, data: KSData, problem: Problem) -> np.ndarray:
    """Best analysis error the KAE could attain: decode(encode(u(t0))) vs u(t0)."""
    from tensordict import TensorDict

    dev = data.device
    sim = torch.as_tensor(problem.sim, device=dev)
    t0 = torch.as_tensor(problem.t0, device=dev)
    f = data.frames(sim, t0)  # [B, X]
    x = TensorDict({"u": f.unsqueeze(1).unsqueeze(-1)}, batch_size=[f.shape[0], 1])
    z = model.present_encoding(x, cond_input=None)
    rec = model.decode(z)["u"].squeeze(-1)
    return rel_l2(data.denorm(rec), data.denorm(f)).cpu().numpy()


def summarize(arr: np.ndarray) -> Dict[str, float]:
    a = np.asarray(arr, dtype=float).ravel()
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
        "sem": float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0,
    }
