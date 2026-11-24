# ruff: noqa: E731
"""Uniform interface over models that live in two codebases and share nothing.

Every adapter exposes the same three things, and NOTHING inside either codebase is
modified -- the models are loaded by their own loaders and called by their own forward
methods:

    n_control_frames   how many consecutive frames the model needs to start a rollout
    encode / decode    physical  <->  whatever the model's control variable actually is
    rollout(c, n)      advance the control n steps, returning PHYSICAL frames

The control is the quantity 4D-Var optimises.  For the turbpred baselines it is the
conditioning window in that model's own normalised space; for the KAE it is the latent
z0.  Both are differentiable functions of the control, which is what makes a like-for-like
4D-Var possible at all.

Why the turbpred rollout is reproduced here rather than called directly: ``forwardDirect``
takes a full sequence tensor and reads the *true* simulation-parameter channels at every
step (it overwrites the model's own prediction of them).  A DA driver has no true future
frames -- it only knows the parameter.  The loop below is a transcription of
``PredictionModel.forwardDirect``/``forwardDiffusionDirect`` with the parameter channel
supplied from the known constant instead of from unavailable future data.  It is verified
against the original in ``verify_adapters.py``, which requires exact agreement when the
true sequence IS supplied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from data_assimilation.tra.bridge import Regime, turbpred_norm


# ---------------------------------------------------------------------------
def prev_steps_of(arch: str) -> int:
    """Conditioning-window length, exactly as ``forwardDirect`` derives it."""
    if "+3Prev" in arch:
        return 4
    if "+2Prev" in arch:
        return 3
    if "+Prev" in arch:
        return 2
    return 1


class TurbpredAdapter:
    """Wraps any ``PredictionModel`` checkpoint (U-Net, FNO, ResNet, ACDM, ACDM-ncn)."""

    def __init__(
        self,
        ckpt: str | Path,
        regime: Regime,
        device="cuda",
        diffusion_opts: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        from turbpred.model import PredictionModel
        from turbpred.model_diffusion import DiffusionModel

        self.regime = regime
        self.dev = torch.device(device)
        self.path = Path(ckpt)
        self.name = name or self.path.parent.name
        self.model = PredictionModel.load(str(ckpt), useGPU=(self.dev.type == "cuda"))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)  # frozen, like every DA baseline

        self.arch = self.model.p_md.arch
        self.n_control_frames = prev_steps_of(self.arch)
        self.is_diffusion = isinstance(self.model.modelDecoder, DiffusionModel)
        if self.is_diffusion and diffusion_opts:
            for k, v in diffusion_opts.items():
                setattr(self.model.modelDecoder, f"inference{k[0].upper() + k[1:]}", v)

        fm, fs, pm, ps = turbpred_norm(regime, self.model.p_d)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.dev)
        self.fmean, self.fstd = t(fm).view(1, -1, 1, 1), t(fs).view(1, -1, 1, 1)
        self.pmean, self.pstd = float(pm), float(ps)
        self.n_fields = len(regime.turbpred_fields)
        self.n_params = len(self.model.p_d.simParams or [])
        self.perm = regime.nc_to_turbpred
        self.inv_perm = regime.turbpred_to_nc

    # -- conversions ---------------------------------------------------------
    def to_model(self, x_phys: torch.Tensor) -> torch.Tensor:
        """Physical [..., C_nc, H, W] -> normalised, turbpred channel order AND layout.

        The .nc files store (x=64, y=128); turbpred's checkpoints are trained on
        ``dataSize=[128, 64]``, i.e. the TRANSPOSE.  Convolutions accept either happily and
        return plausible-looking output, so the mistake is silent: it cost a 58x inflation
        of the one-step error (0.1456 -> 0.0025) and made every baseline look worse than
        persistence.  ``verify_adapters`` could not catch it because it compares this
        adapter with turbpred's own forward on the SAME input -- both were consistently
        wrong.  The gate that catches it is `verify_forward_skill`: a forward model must
        beat persistence.
        """
        x = x_phys[..., self.perm, :, :].transpose(-1, -2)
        sh = x.shape
        return (
            (x.reshape(-1, self.n_fields, *sh[-2:]) - self.fmean) / self.fstd
        ).reshape(sh)

    def to_physical(self, z: torch.Tensor) -> torch.Tensor:
        """Normalised turbpred layout -> physical, .nc channel order and layout."""
        sh = z.shape
        x = (z.reshape(-1, self.n_fields, *sh[-2:]) * self.fstd + self.fmean).reshape(
            sh
        )
        return x.transpose(-1, -2)[..., self.inv_perm, :, :]

    def param_channel(self, param_phys: torch.Tensor, shape) -> torch.Tensor:
        """The simulation parameter as its normalised constant-valued channel(s)."""
        v = (param_phys.to(self.dev).float() - self.pmean) / self.pstd
        return v.view(-1, 1, 1, 1).expand(-1, self.n_params, *shape)

    # -- the rollout ---------------------------------------------------------
    def rollout(
        self,
        control: torch.Tensor,
        n_steps: int,
        param_phys: torch.Tensor,
        checkpoint_every: int = -1,
    ) -> torch.Tensor:
        """Advance ``control`` (normalised, [B, k, C, H, W]) by ``n_steps``.

        Returns PHYSICAL frames ``[B, k + n_steps, C_nc, H, W]``: the control window
        followed by the predicted frames, so index ``k-1`` is the analysis state.
        Differentiable with respect to ``control``.

        ``checkpoint_every`` controls segmented gradient checkpointing, which is what makes
        4D-Var over a long rollout fit in memory at all: storing every activation of a
        25-step rollout at this resolution exceeds 22 GB, while recomputing each segment
        once during the backward pass keeps it bounded.  ``-1`` picks sqrt(n_steps)
        automatically, ``0`` disables it.  The forward VALUES are unchanged either way --
        `verify_adapters.py` still has to pass.
        """
        B, k, C, H, W = control.shape
        assert (
            k == self.n_control_frames
        ), f"{self.name} needs {self.n_control_frames} conditioning frames, got {k}"
        pch = self.param_channel(param_phys, (H, W)) if self.n_params else None

        def with_param(f):
            return torch.cat([f, pch], dim=1) if pch is not None else f

        def advance(*frames):
            """One step from the last k frames -> the next frame."""
            win = list(frames)
            if self.is_diffusion:
                nxt = self._diffusion_step(win, win[-1])
            else:
                dec = self.model.modelDecoder
                u_in = torch.cat(win, dim=1)
                nxt = dec(u_in) if _is_fno(dec) else dec(u_in, None)
            if self.n_params:
                # turbpred overwrites the predicted parameter channels with the true
                # constant at every step; the same is done here, from the known parameter
                nxt = torch.cat([nxt[:, : self.n_fields], pch], dim=1)
            return nxt

        seg = (
            max(1, int(round(np.sqrt(max(1, n_steps)))))
            if checkpoint_every < 0
            else int(checkpoint_every)
        )
        use_ckpt = (
            seg > 0
            and n_steps > seg
            and torch.is_grad_enabled()
            and control.requires_grad
            and not self.is_diffusion
        )

        seq = [with_param(control[:, i]) for i in range(k)]
        if not use_ckpt:
            for _ in range(n_steps):
                seq.append(advance(*seq[-k:]))
        else:
            from torch.utils.checkpoint import checkpoint

            def run_segment(m, *frames):
                win = list(frames)
                outs = []
                for _ in range(m):
                    nxt = advance(*win[-k:])
                    win.append(nxt)
                    outs.append(nxt)
                return tuple(outs)

            done = 0
            while done < n_steps:
                m = min(seg, n_steps - done)
                outs = checkpoint(run_segment, m, *seq[-k:], use_reentrant=False)
                seq.extend(outs if isinstance(outs, tuple) else [outs])
                done += m
        out = torch.stack(seq, dim=1)[:, :, : self.n_fields]
        return self.to_physical(out)

    def _diffusion_step(self, window, prev) -> torch.Tensor:
        """One ACDM frame, matching ``forwardDiffusionDirect`` exactly.

        turbpred builds the conditioning by concatenating the last ``prevSteps`` frames
        along the CHANNEL dimension of a ``[B, 1, k*C, H, W]`` tensor, and passes the most
        recent frame as ``data`` (the sampler uses it for shape and device only).
        """
        cond = torch.cat([f.unsqueeze(1) for f in window], dim=2)
        return self.model.modelDecoder(
            conditioning=cond, data=prev.unsqueeze(1)
        ).squeeze(1)

    def __repr__(self):
        return (
            f"TurbpredAdapter({self.name}, arch={self.arch}, "
            f"k={self.n_control_frames}, diffusion={self.is_diffusion})"
        )


def _is_fno(dec) -> bool:
    from neuralop.models import FNO  # the class turbpred itself imports

    return isinstance(dec, FNO)
