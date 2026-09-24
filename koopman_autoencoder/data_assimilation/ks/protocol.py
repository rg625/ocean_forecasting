"""Shared observation protocol for the KS data-assimilation campaign.

Every assimilation method in this campaign consumes observations produced *only*
by this module, so that the comparison cannot accidentally differ in what each
method is allowed to see.  A :class:`Problem` fully determines an assimilation
task: which trajectories, which analysis times, which future observation times,
which spatial sensors, and which noise realisation.

Conventions
-----------
* Fields are stored **normalised** with the deterministic KS constants used to train
  every model (``KS_MEAN``/``KS_STD``); metrics are reported on **denormalised**
  fields so that they are comparable across methods and to physical units.
* ``t0`` is the *unobserved* analysis frame.  Every observation lies strictly in its
  future: ``tau_i > 0``.  Nothing about ``u(t0)`` is ever given to a method.
* Observation times are stored as real-valued ``tau`` (physical time).  For the
  canonical protocol these are exact multiples of ``dt``; the off-grid protocol
  deliberately places them between frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import xarray as xr

DT = 0.1  # physical time between stored KS frames


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def ks_stats() -> Tuple[float, float]:
    from models.dataloader import KS_MEAN, KS_STD

    def _f(v):
        return float(v.item()) if torch.is_tensor(v) else float(v)

    return _f(KS_MEAN["u"]), _f(KS_STD["u"])


class KSData:
    """Normalised KS fields plus the machinery to slice assimilation windows."""

    def __init__(self, path: Path, device: torch.device):
        ds = xr.open_dataset(path)
        self.u_raw = torch.from_numpy(ds["u"].values).float().squeeze(-1).to(device)
        self.x = np.asarray(ds["x"].values).squeeze()
        self.attrs = dict(ds.attrs)
        ds.close()
        self.path = str(path)
        self.device = device
        self.n_sim, self.n_t, self.X = self.u_raw.shape
        self.mean, self.std = ks_stats()
        self.u = (self.u_raw - self.mean) / (self.std + 1e-8)  # [sim, t, X]

    def frames(self, sim: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Gather normalised frames at (sim_i, t_i) -> [B, X]."""
        return self.u[sim, t]

    def denorm(self, u_norm: torch.Tensor) -> torch.Tensor:
        return u_norm * self.std + self.mean


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
@dataclass
class Problem:
    """One fully specified assimilation task, shared verbatim by all methods."""

    name: str
    data_path: str
    sim: np.ndarray  # [B] trajectory index (one per assimilation problem)
    t0: np.ndarray  # [B] analysis frame (unobserved)
    taus: np.ndarray  # [n_obs] observation lead times, physical units
    obs_frac: float  # fraction of spatial points observed
    noise_std: float  # observation noise std, normalised units
    noise_dist: str  # "gaussian" | "laplace"; both at the SAME std
    seed: int
    forecast_taus: np.ndarray = field(default_factory=lambda: np.array([]))
    on_grid: bool = True  # are all taus exact multiples of dt?

    # populated by build_problem
    mask: Optional[np.ndarray] = None  # [n_obs, X] 1 = observed
    y: Optional[np.ndarray] = None  # [n_obs, B, X] noisy normalised observations

    def meta(self) -> Dict:
        d = asdict(self)
        for k in ("sim", "t0", "taus", "forecast_taus"):
            d[k] = np.asarray(d[k]).tolist()
        d["mask_sum_per_obs"] = (
            None if self.mask is None else np.asarray(self.mask).sum(axis=1).tolist()
        )
        d.pop("y", None)
        d.pop("mask", None)
        return d


def sample_analysis_points(
    data: KSData, n_problems: int, max_tau: float, seed: int, t0_min_frame: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """One analysis point per *distinct* trajectory where possible.

    Trajectories are consumed in order and re-used only if ``n_problems`` exceeds the
    number of available simulations; the ``t0`` of a re-used trajectory is drawn from a
    disjoint part of the record.  ``t0_min_frame`` discards the initial transient.
    """
    rng = np.random.default_rng(seed)
    max_off = int(np.ceil(max_tau / DT)) + 2
    t0_hi = data.n_t - max_off - 1
    assert t0_hi > t0_min_frame, "requested horizon does not fit in the record"

    sims, t0s = [], []
    for i in range(n_problems):
        s = i % data.n_sim
        rep = i // data.n_sim  # which pass over the trajectory pool
        lo = t0_min_frame + rep * (t0_hi - t0_min_frame) // max(
            1, (n_problems - 1) // data.n_sim + 1
        )
        hi = t0_min_frame + (rep + 1) * (t0_hi - t0_min_frame) // max(
            1, (n_problems - 1) // data.n_sim + 1
        )
        sims.append(s)
        t0s.append(int(rng.integers(lo, max(lo + 1, min(hi, t0_hi)))))
    return np.array(sims), np.array(t0s)


def build_problem(
    data: KSData,
    *,
    name: str,
    n_problems: int,
    taus: np.ndarray,
    obs_frac: float = 1.0,
    noise_std: float = 0.0,
    noise_dist: str = "gaussian",
    seed: int = 0,
    forecast_taus: Optional[np.ndarray] = None,
    shared_mask: bool = True,
    t0_min_frame: int = 100,
) -> Problem:
    """Instantiate observations once; every method then sees exactly these arrays.

    ``obs_frac < 1`` draws a random spatial sensor mask.  With ``shared_mask`` the same
    sensors are used at every observation time (a fixed sensor network); otherwise each
    observation time gets its own draw.
    """
    taus = np.asarray(taus, dtype=float)
    assert (taus > 0).all(), "observations must lie strictly in the future of t0"
    forecast_taus = (
        np.array([])
        if forecast_taus is None
        else np.asarray(forecast_taus, dtype=float)
    )
    max_tau = float(max(taus.max(), forecast_taus.max() if forecast_taus.size else 0.0))

    sim, t0 = sample_analysis_points(data, n_problems, max_tau, seed, t0_min_frame)
    on_grid = bool(np.allclose(taus / DT, np.round(taus / DT), atol=1e-9))

    rng = np.random.default_rng(seed + 999)
    X = data.X
    if obs_frac >= 1.0:
        mask = np.ones((len(taus), X), dtype=np.float32)
    elif shared_mask:
        n_keep = max(1, int(round(obs_frac * X)))
        idx = rng.choice(X, size=n_keep, replace=False)
        m = np.zeros(X, dtype=np.float32)
        m[idx] = 1.0
        mask = np.repeat(m[None], len(taus), axis=0)
    else:
        mask = np.zeros((len(taus), X), dtype=np.float32)
        n_keep = max(1, int(round(obs_frac * X)))
        for i in range(len(taus)):
            mask[i, rng.choice(X, size=n_keep, replace=False)] = 1.0

    # true fields at the observation times.  Off-grid taus are read from the *true*
    # trajectory by linear interpolation between the bracketing stored frames -- this
    # is a property of the data, identical for every method, not a model workaround.
    sim_t = torch.as_tensor(sim, device=data.device)
    y = np.zeros((len(taus), len(sim), X), dtype=np.float32)
    for i, tau in enumerate(taus):
        f = tau / DT
        lo, hi = int(np.floor(f)), int(np.ceil(f))
        w = float(f - lo)
        u_lo = data.frames(sim_t, torch.as_tensor(t0 + lo, device=data.device))
        u_hi = data.frames(sim_t, torch.as_tensor(t0 + hi, device=data.device))
        y[i] = ((1 - w) * u_lo + w * u_hi).cpu().numpy()

    if noise_std > 0:
        # Both distributions are drawn at the SAME standard deviation, so the sweep
        # isolates the SHAPE of the error law (Gaussian tails vs the heavier, spikier
        # Laplace tails) rather than its magnitude. A Laplace(0, b) has std b*sqrt(2),
        # hence b = noise_std / sqrt(2).
        if noise_dist == "gaussian":
            eta = rng.standard_normal(y.shape)
        elif noise_dist == "laplace":
            eta = rng.laplace(0.0, 1.0 / np.sqrt(2.0), y.shape)
        else:
            raise ValueError(f"unknown noise_dist {noise_dist!r}")
        y = y + noise_std * eta.astype(np.float32)

    return Problem(
        name=name,
        data_path=data.path,
        sim=sim,
        t0=t0,
        taus=taus,
        obs_frac=float(obs_frac),
        noise_std=float(noise_std),
        noise_dist=str(noise_dist),
        seed=seed,
        forecast_taus=forecast_taus,
        on_grid=on_grid,
        mask=mask,
        y=y,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rel_l2(pred: torch.Tensor, true: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Per-sample relative L2 error along `dim`."""
    num = torch.linalg.vector_norm(pred - true, dim=dim)
    den = torch.linalg.vector_norm(true, dim=dim).clamp_min(1e-12)
    return num / den


def masked_rel_l2(
    pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Relative L2 restricted to observed entries. mask broadcasts over the batch."""
    m = mask.expand_as(pred)
    num = torch.linalg.vector_norm((pred - true) * m, dim=-1)
    den = torch.linalg.vector_norm(true * m, dim=-1).clamp_min(1e-12)
    return num / den
