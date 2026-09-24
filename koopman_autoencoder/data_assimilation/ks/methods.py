"""Assimilation methods: continuous-KAE latent 4D-Var and U-Net physical-space 4D-Var.

Both solve the *same* variational problem

    J(c) = (1/m) sum_i || M_i o [ G(c, tau_i) - y_i ] ||^2

over a control variable ``c`` that is the only quantity optimised; every learned weight
is frozen.  The methods differ solely in the control variable and the propagator:

    Continuous KAE : c = z0 in R^D,  G = decoder( expm(K tau_i) z0 )
    U-Net 4D-Var   : c = x0 in R^X,  G = F_theta^{n_i}(x0),  n_i = tau_i / dt

Both implementations roll out **once** to the furthest observation time and tap the
intermediate states, which is the efficient (and therefore fair) way to evaluate a
multi-time 4D-Var cost for an autoregressive model.  The KAE's exact propagator needs
no rollout at all: one precomputed matrix exponential per observation time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from data_assimilation.ks.protocol import DT, KSData, Problem, masked_rel_l2, rel_l2


# ---------------------------------------------------------------------------
@dataclass
class SolveConfig:
    iters: int = 1000
    lr: float = 1e-2
    optimizer: str = "adam"
    init: str = "random"  # "random" | "zero" | "given" (uses init_value)
    init_value: Optional[torch.Tensor] = None  # starting control for init="given"
    reg: float = 0.0  # weight of the min-norm prior on the control
    track_every: int = 0  # 0 disables per-iteration diagnostics
    seed: int = 0


class Assimilator:
    """Common driver: identical optimiser, budget, bookkeeping and metrics."""

    name = "base"
    control_kind = "?"

    # -- to be provided by subclasses ---------------------------------------
    def control_shape(self, B: int) -> Sequence[int]:
        raise NotImplementedError

    def init_scale(self) -> float:
        raise NotImplementedError

    def prepare(self, taus: np.ndarray) -> Dict:
        """Any per-problem precomputation (timed and reported separately)."""
        return {}

    def predict_at(
        self, control: torch.Tensor, taus: np.ndarray, prep: Dict
    ) -> torch.Tensor:
        """Differentiable normalised fields at every tau -> [n_tau, B, X]."""
        raise NotImplementedError

    def eval_counts(self, taus: np.ndarray) -> Dict[str, float]:
        """Per-optimisation-iteration cost accounting."""
        raise NotImplementedError

    def analysis_field(self, control: torch.Tensor) -> torch.Tensor:
        """The recovered state at t0 (tau = 0), normalised -> [B, X]."""
        raise NotImplementedError

    # -- shared -------------------------------------------------------------
    def solve(self, problem: Problem, data: KSData, cfg: SolveConfig) -> Dict:
        dev = data.device
        B = len(problem.sim)
        y = torch.as_tensor(problem.y, device=dev)  # [n_obs, B, X]
        mask = torch.as_tensor(problem.mask, device=dev)  # [n_obs, X]
        taus = problem.taus

        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t_setup0 = time.perf_counter()
        prep = self.prepare(taus)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        setup_s = time.perf_counter() - t_setup0

        g = torch.Generator(device=dev).manual_seed(cfg.seed)
        shape = tuple(self.control_shape(B))
        if cfg.init == "given":
            assert cfg.init_value is not None, "init='given' needs init_value"
            c = cfg.init_value.detach().clone().to(dev)
            assert tuple(c.shape) == shape, (tuple(c.shape), shape)
        elif cfg.init == "zero":
            c = torch.zeros(shape, device=dev)
        elif cfg.init == "random":
            c = torch.randn(shape, device=dev, generator=g) * self.init_scale()
        else:
            raise ValueError(
                f"unknown init {cfg.init!r}; expected random, zero or given"
            )
        c = c.requires_grad_(True)
        c0 = c.detach().clone()

        opt = (torch.optim.Adam if cfg.optimizer == "adam" else torch.optim.SGD)(
            [c], lr=cfg.lr
        )

        hist: Dict[str, List[float]] = {
            "iter": [],
            "loss": [],
            "init_rel_l2": [],
            "obs_rel_l2": [],
            "grad_norm": [],
            "grad_norm_per_dim": [],
        }
        true_t0 = data.frames(
            torch.as_tensor(problem.sim, device=dev),
            torch.as_tensor(problem.t0, device=dev),
        )  # [B, X]
        n_ctrl = int(np.prod(shape[1:]))  # control dimensions per problem
        grad_first = grad_last = float("nan")

        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_opt0 = time.perf_counter()
        for it in range(cfg.iters):
            opt.zero_grad(set_to_none=True)
            pred = self.predict_at(c, taus, prep)  # [n_obs, B, X]
            diff = (pred - y) * mask[:, None, :]
            loss = (diff**2).sum() / mask[:, None, :].expand_as(pred).sum().clamp_min(
                1.0
            )
            if cfg.reg > 0:
                loss = loss + cfg.reg * (c**2).mean()
            loss.backward()

            track = cfg.track_every and (
                it % cfg.track_every == 0 or it == cfg.iters - 1
            )
            if track or it == 0 or it == cfg.iters - 1:
                # per-problem gradient norm of the objective w.r.t. the control variable,
                # averaged over problems; the per-dimension version makes the 128-d latent
                # and the 64-d physical control comparable.
                gn = float(
                    torch.linalg.vector_norm(c.grad.detach().flatten(1), dim=1).mean()
                )
                if it == 0:
                    grad_first = gn
                grad_last = gn
            opt.step()

            if track:
                with torch.no_grad():
                    r = rel_l2(
                        data.denorm(self.analysis_field(c.detach())),
                        data.denorm(true_t0),
                    )
                    obs = torch.stack(
                        [
                            masked_rel_l2(
                                data.denorm(pred[i].detach()),
                                data.denorm(y[i]),
                                mask[i],
                            )
                            for i in range(len(taus))
                        ]
                    ).mean(0)
                hist["iter"].append(it)
                hist["loss"].append(float(loss))
                hist["init_rel_l2"].append(float(r.mean()))
                hist["obs_rel_l2"].append(float(obs.mean()))
                hist["grad_norm"].append(gn)
                hist["grad_norm_per_dim"].append(gn / np.sqrt(n_ctrl))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        opt_s = time.perf_counter() - t_opt0

        peak_mem = (
            torch.cuda.max_memory_allocated() / 2**20
            if dev.type == "cuda"
            else float("nan")
        )

        counts = self.eval_counts(taus)
        out = {
            "method": self.name,
            "control_kind": self.control_kind,
            "control": c.detach(),
            "control_init": c0,
            "prep": prep,
            "setup_s": setup_s,
            "opt_s": opt_s,
            "total_s": setup_s + opt_s,
            "ms_per_iter": opt_s / cfg.iters * 1e3,
            "peak_mem_MiB": peak_mem,
            "hist": hist,
            "grad_norm_first": grad_first,
            "grad_norm_last": grad_last,
            "n_control_dims": n_ctrl,
            "iters": cfg.iters,
            "lr": cfg.lr,
            "init": cfg.init,
            "seed": cfg.seed,
            **{f"evals_{k}": v * cfg.iters for k, v in counts.items()},
            **{f"evals_per_iter_{k}": v for k, v in counts.items()},
        }
        return out


# ---------------------------------------------------------------------------
class KAE4DVar(Assimilator):
    """Latent 4D-Var over ``z0`` with the continuous Koopman generator.

    ``propagator='expm'``  -> z(tau) = expm(K tau) z0, one precomputed matrix per tau.
    ``propagator='rk4'``   -> the same learned generator integrated with the model's own
                              RK4 step at the training cadence dt (used for the
                              accuracy-parity / cost experiment; it is *not* a different
                              model, only a different way of evaluating it).
    """

    control_kind = "latent z0"

    def __init__(
        self,
        model,
        K: torch.Tensor,
        latent_dim: int,
        init_scale: float,
        propagator: str = "expm",
        dt: float = DT,
    ):
        self.model = model
        self.K = K
        self.D = latent_dim
        self._init_scale = init_scale
        self.propagator = propagator
        self.dt = dt
        self.name = f"KAE-{propagator}"

    def control_shape(self, B):
        return (B, self.D)

    def init_scale(self):
        return self._init_scale

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z)["u"].squeeze(-1)  # [B, H=X, W=1] -> [B, X]

    def prepare(self, taus: np.ndarray) -> Dict:
        if self.propagator == "expm":
            return {"phis": [torch.matrix_exp(self.K * float(t)) for t in taus]}
        # RK4: integer step counts at the training cadence
        n = np.round(np.asarray(taus) / self.dt).astype(int)
        return {"n_steps": n, "tau_realised": n * self.dt}

    def predict_at(self, z0, taus, prep):
        if self.propagator == "expm":
            return torch.stack([self._decode(z0 @ phi.T) for phi in prep["phis"]])
        # one rollout to the furthest time, tapping the intermediate states
        n = prep["n_steps"]
        order = np.argsort(n)
        outs: List[Optional[torch.Tensor]] = [None] * len(n)
        z, done = z0, 0
        for j in order:
            for _ in range(int(n[j]) - done):
                z = self.model.koopman_operator(z, cond=None, dt=self.dt)
            done = int(n[j])
            outs[j] = self._decode(z)
        return torch.stack(outs)

    def eval_counts(self, taus):
        n_obs = len(taus)
        if self.propagator == "expm":
            return {"decoder": n_obs, "propagation_steps": 0.0, "network": n_obs}
        n_max = int(np.round(np.asarray(taus) / self.dt).astype(int).max())
        return {"decoder": n_obs, "propagation_steps": float(n_max), "network": n_obs}

    def analysis_field(self, z0):
        return self._decode(z0)

    # convenience for forecasting past the assimilation window
    @torch.no_grad()
    def forecast(self, z0: torch.Tensor, taus: Sequence[float]) -> torch.Tensor:
        if self.propagator == "expm":
            return torch.stack(
                [self._decode(z0 @ torch.matrix_exp(self.K * float(t)).T) for t in taus]
            )
        n = np.round(np.asarray(taus) / self.dt).astype(int)
        order = np.argsort(n)
        outs: List[Optional[torch.Tensor]] = [None] * len(n)
        z, done = z0, 0
        for j in order:
            for _ in range(int(n[j]) - done):
                z = self.model.koopman_operator(z, cond=None, dt=self.dt)
            done = int(n[j])
            outs[j] = self._decode(z)
        return torch.stack(outs)


# ---------------------------------------------------------------------------
class UNet4DVar(Assimilator):
    """Physical-space 4D-Var over ``x0`` with a frozen autoregressive U-Net.

    The U-Net is a map at the fixed training cadence ``dt``, so an observation at
    ``tau`` is reached by ``n = tau/dt`` applications.  When ``tau`` is not a multiple of
    ``dt`` the model cannot land on it; ``time_handling`` selects the workaround and the
    induced time mismatch is recorded in ``tau_realised`` so it can be reported.
    """

    name = "UNet-4DVar"
    control_kind = "physical x0"

    def __init__(
        self,
        model,
        init_scale: float = 1.0,
        dt: float = DT,
        time_handling: str = "nearest",
        checkpoint_every: int = 0,
    ):
        self.model = model
        self._init_scale = init_scale
        self.dt = dt
        self.time_handling = time_handling
        # Backpropagating through a long autoregressive rollout stores every intermediate
        # activation, so memory grows linearly with the assimilation horizon.  Segmented
        # gradient checkpointing keeps long windows feasible at the cost of recomputing
        # each segment's forward pass once; ``0`` disables it, ``-1`` picks sqrt(n).
        self.checkpoint_every = checkpoint_every

    def control_shape(self, B):
        return (B, 64)

    def init_scale(self):
        return self._init_scale

    def prepare(self, taus: np.ndarray) -> Dict:
        f = np.asarray(taus) / self.dt
        if self.time_handling == "nearest":
            n = np.round(f).astype(int)
        elif self.time_handling == "floor":
            n = np.floor(f).astype(int)
        else:
            raise ValueError(self.time_handling)
        n = np.maximum(n, 1)
        return {
            "n_steps": n,
            "tau_realised": n * self.dt,
            "tau_error": np.abs(n * self.dt - np.asarray(taus)),
        }

    def _step(self, x):
        return self.model(x.unsqueeze(1)).squeeze(1)  # [B, X]

    CKPT_THRESHOLD = 32  # rollouts shorter than this fit comfortably in memory

    def _segment_size(self, n_max: int) -> int:
        if self.checkpoint_every == 0:
            return 0
        if self.checkpoint_every > 0:
            return self.checkpoint_every
        if n_max <= self.CKPT_THRESHOLD:  # "auto": only when needed
            return 0
        return max(1, int(round(np.sqrt(max(1, n_max)))))

    def _advance(self, x: torch.Tensor, k: int, seg: int) -> torch.Tensor:
        """Apply the map ``k`` times, optionally in checkpointed segments."""
        if k <= 0:
            return x
        if seg <= 0 or not torch.is_grad_enabled():
            for _ in range(k):
                x = self._step(x)
            return x
        from torch.utils.checkpoint import checkpoint

        done = 0
        while done < k:
            m = min(seg, k - done)

            def run(inp, m=m):
                for _ in range(m):
                    inp = self._step(inp)
                return inp

            x = checkpoint(run, x, use_reentrant=False)
            done += m
        return x

    def predict_at(self, x0, taus, prep):
        n = prep["n_steps"]
        seg = self._segment_size(int(n.max()))
        order = np.argsort(n)
        outs: List[Optional[torch.Tensor]] = [None] * len(n)
        x, done = x0, 0
        for j in order:
            x = self._advance(x, int(n[j]) - done, seg)
            done = int(n[j])
            outs[j] = x
        return torch.stack(outs)

    def eval_counts(self, taus):
        n = self.prepare(taus)["n_steps"]
        n_max = int(n.max())
        # checkpointing recomputes each segment's forward pass once during the backward
        recompute = 1.0 if self._segment_size(n_max) > 0 else 0.0
        return {
            "decoder": 0.0,
            "propagation_steps": float(n_max),
            "network": float(n_max) * (1.0 + recompute),
        }

    def analysis_field(self, x0):
        return x0

    @torch.no_grad()
    def forecast(self, x0: torch.Tensor, taus: Sequence[float]) -> torch.Tensor:
        n = np.maximum(np.round(np.asarray(taus) / self.dt).astype(int), 0)
        order = np.argsort(n)
        outs: List[Optional[torch.Tensor]] = [None] * len(n)
        x, done = x0, 0
        for j in order:
            for _ in range(int(n[j]) - done):
                x = self._step(x)
            done = int(n[j])
            outs[j] = x
        return torch.stack(outs)
