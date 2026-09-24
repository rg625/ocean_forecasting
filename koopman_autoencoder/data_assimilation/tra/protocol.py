# mypy: disable-error-code="assignment"
"""The assimilation problem, defined once so every method sees exactly the same one.

A problem is: an analysis time ``t0`` that is NEVER observed, and ``N`` observations at
strictly future lead times ``t0 + tau_i``, each a noisy, possibly spatially sparse view of
the physical state.  Every method receives the identical ``Problem`` arrays; nothing about
the geometry, the noise realisation or the mask is regenerated per method.

Conventions carried over from the KS campaign, so the two are readable side by side:
  delta_f = tau_1        lead to the FIRST observation
  delta_l = tau_N        the recovery horizon
  N                      the number of observation times
These are varied independently; the canonical schedule fixes all three at once and so
cannot attribute an outcome to any of them.

Observations live in PHYSICAL units.  The obstacle interior is never observed and never
scored: it carries no physics and each model fills it differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Problem:
    name: str
    sim: np.ndarray  # [B] trajectory index
    t0: np.ndarray  # [B] analysis frame
    offsets: np.ndarray  # [N] observation lead times, in FRAMES, all > 0
    y: np.ndarray  # [N, B, C, H, W] noisy physical observations
    mask: np.ndarray  # [N, B, 1, H, W] 1 where observed
    noise_std: float
    obs_frac: float
    regime: str
    data_file: str
    seed: int
    extras: dict = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return len(self.offsets)

    def subset(self, idx) -> "Problem":
        """A Problem holding only the selected problems, for memory-bounded solving.

        4D-Var over N problems builds one rollout graph for all of them, so its memory
        grows linearly in N -- the U-Net at N = 24 with a 25-step rollout needs ~18 GB and
        OOMs on a shared card. Each problem is independent (its own control, and the cost
        is a mean over problems), so solving in batches and concatenating is equivalent.
        """
        idx = np.asarray(idx, dtype=int)
        return Problem(
            name=f"{self.name}[{idx[0]}:{idx[-1] + 1}]",
            sim=self.sim[idx],
            t0=self.t0[idx],
            offsets=self.offsets,
            y=self.y[:, idx],
            mask=self.mask[:, idx],
            noise_std=self.noise_std,
            obs_frac=self.obs_frac,
            regime=self.regime,
            data_file=self.data_file,
            seed=self.seed,
            extras=dict(self.extras),
        )

    @property
    def offsets_eval(self) -> np.ndarray:
        """Lead times a method may evaluate at exactly: real if there are any, else grid."""
        r = self.extras.get("offsets_real")
        return self.offsets.astype(float) if r is None else np.asarray(r, dtype=float)

    def meta(self) -> dict:
        return {
            "name": self.name,
            "offsets": self.offsets.tolist(),
            "offsets_real": (
                None
                if "offsets_real" not in self.extras
                else [float(x) for x in self.extras["offsets_real"]]
            ),
            "delta_f_frames": int(self.offsets.min()),
            "delta_l_frames": int(self.offsets.max()),
            "N": self.n_obs,
            "noise_std": self.noise_std,
            "noise_dist": self.extras.get("noise_dist", "gaussian"),
            "obs_frac": self.obs_frac,
            "n_problems": len(self.sim),
            "n_trajectories": int(len(set(self.sim.tolist()))),
            "regime": self.regime,
            "data_file": self.data_file,
            "seed": self.seed,
        }


def build_problem(
    data,
    *,
    name: str,
    n_problems: int,
    seed: int,
    offsets=None,
    noise_std: float = 0.0,
    obs_frac: float = 1.0,
    t0_min: int = 2,
    margin: int = 0,
    offsets_real=None,
    noise_dist: str = "gaussian",
    stratify: bool = False,
) -> Problem:
    """Draw problems and generate their observations once.

    ``t0_min`` leaves room for the conditioning window that the autoregressive baselines
    need BEFORE t0 (they are conditioned on frames ending at t0), so every method can be
    posed on the same analysis time.

    ``noise_dist`` selects the observation-error distribution.  Every method here assumes
    a Gaussian one -- 4D-Var through its least-squares cost, the samplers through
    ``Sigma_y`` -- so drawing the noise from a heavy-tailed Laplace instead, at MATCHED
    VARIANCE, misspecifies all of them equally and asks which degrades.  The Laplace scale
    is ``noise_std / sqrt(2)``, so the two distributions differ in shape and not in power.

    ``offsets_real`` puts the observations at REAL-valued lead times, between stored
    frames.  The record only exists on the integer grid, so the observed state is linearly
    interpolated between the two bracketing frames -- an approximation, and its size is
    recorded in ``extras['interp_gap_rel']`` so it can be read off rather than assumed.
    ``offsets`` then holds the SNAPPED integer times, which is all an autoregressive model
    can reach; a method able to evaluate at a real lead time reads ``offsets_real``.
    """
    assert (offsets is None) != (
        offsets_real is None
    ), "give exactly one of offsets (grid) or offsets_real (off-grid)"
    if offsets_real is not None:
        offsets_real = np.asarray(offsets_real, dtype=float)
        assert (offsets_real > 0).all(), "observations must be strictly future of t0"
        offsets = np.floor(offsets_real + 0.5).astype(int)  # nearest, ties away from 0
    offsets = np.asarray(offsets, dtype=int)
    assert (offsets > 0).all(), "observations must be strictly in the future of t0"
    rng = np.random.default_rng(seed)
    span = int(
        max(offsets.max(), 0 if offsets_real is None else np.ceil(offsets_real.max()))
    )
    hi = data.n_t - span - 1 - margin
    assert (
        hi > t0_min
    ), f"record has {data.n_t} frames; delta_l={span} leaves no room for t0"
    if stratify:
        # Spread the problems evenly over the trajectories, and spread the analysis times
        # within each. The trajectory is the unit of replication for a confidence interval
        # (see data_assimilation.tra.stats), so an unbalanced draw wastes the little replication there
        # is, and closely spaced analysis times share most of their observation window.
        reps = int(np.ceil(n_problems / data.n_sim))
        sim = np.tile(np.arange(data.n_sim), reps)[:n_problems]
        t0 = np.empty(n_problems, dtype=int)
        for u in np.unique(sim):
            m = sim == u
            k = int(m.sum())
            span = max(1, (hi - t0_min) // (4 * max(k, 1)))
            base = np.linspace(t0_min, hi - 1, k + 2)[1:-1]
            t0[m] = np.clip(
                np.round(base).astype(int) + rng.integers(-span, span + 1, k),
                t0_min,
                hi - 1,
            )
    else:
        sim = rng.integers(0, data.n_sim, n_problems)
        t0 = rng.integers(t0_min, hi, n_problems)

    s = torch.as_tensor(sim)
    extras: dict = {}
    if offsets_real is None:
        truth = torch.stack(
            [data.frames(s, torch.as_tensor(t0 + int(o))) for o in offsets]
        )  # [N,B,C,H,W]
    else:
        lo_i = np.floor(offsets_real).astype(int)
        w = (offsets_real - lo_i).astype(np.float32)  # [N]
        lo_f = torch.stack([data.frames(s, torch.as_tensor(t0 + int(o))) for o in lo_i])
        hi_f = torch.stack(
            [data.frames(s, torch.as_tensor(t0 + int(o) + 1)) for o in lo_i]
        )
        wt = torch.as_tensor(w, device=lo_f.device).view(-1, 1, 1, 1, 1)
        truth = (1 - wt) * lo_f + wt * hi_f
        gap = (hi_f - lo_f).pow(2).mean((-1, -2, -3)).sqrt() / lo_f.pow(2).mean(
            (-1, -2, -3)
        ).sqrt().clamp_min(1e-12)
        extras["offsets_real"] = offsets_real
        extras["interp_gap_rel"] = gap.mean(1).cpu().numpy()
    truth = truth.cpu().numpy()

    # spatial mask: the SAME sensor layout at every observation time, so temporal and
    # spatial sparsity stay separable
    H, W = truth.shape[-2], truth.shape[-1]
    if obs_frac >= 1.0:
        m = np.ones((1, 1, 1, H, W), dtype=np.float32)
    else:
        keep = rng.random((1, len(sim), 1, H, W)) < obs_frac
        m = keep.astype(np.float32)
    mask = np.broadcast_to(m, (len(offsets), len(sim), 1, H, W)).copy()

    # the obstacle interior is never observed
    om = data.mask_for(s)
    if om is not None:
        mask = mask * om.cpu().numpy()[None, :, None]

    if noise_std <= 0:
        eta = 0.0
    elif noise_dist == "laplace":
        # matched variance: Var[Laplace(b)] = 2 b^2, so b = sigma / sqrt(2)
        eta = rng.laplace(0.0, noise_std / np.sqrt(2.0), truth.shape).astype(np.float32)
    elif noise_dist == "gaussian":
        eta = noise_std * rng.standard_normal(truth.shape).astype(np.float32)
    else:
        raise ValueError(f"unknown noise_dist {noise_dist!r}")
    y = truth + eta
    extras["noise_dist"] = noise_dist
    if noise_std > 0:
        extras["noise_realised_std"] = float(np.std(eta))
    return Problem(
        name=name,
        sim=sim,
        t0=t0,
        offsets=offsets,
        y=y.astype(np.float32),
        mask=mask.astype(np.float32),
        noise_std=float(noise_std),
        obs_frac=float(obs_frac),
        regime=data.regime.name,
        data_file=str(data.path),
        seed=seed,
        extras=extras,
    )


def canonical(n_frames_max: int = 25) -> np.ndarray:
    """The reference schedule: irregular, geometric, strictly future."""
    return np.array([1, 3, 7, 15, 25], dtype=int)
