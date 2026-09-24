# ruff: noqa: E731
"""KAE adapter, built exactly the way ``evaluate.py`` builds it.

Nothing about the architecture or the checkpoint loading is re-implemented: the model is
constructed from its own experiment config with ``KoopmanAutoencoder(...)`` and the weights
are restored with the project's ``load_checkpoint``.  The adapter only adds the two things
a DA driver needs -- a control variable and a differentiable map from it to physical
frames at requested lead times.

The KAE and the turbpred baselines were trained against the SAME normalisation constants
(``models.dataloader.TRA_MEAN/STD`` equal ``Transforms.normMean/normStd`` entry for entry),
so physical units are a common ground with no rescaling error.  The channel ORDER still
differs and is handled in ``bridge.Regime``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from tensordict import TensorDict

from data_assimilation.tra.bridge import Regime


class KAEAdapter:
    """Control is the latent ``z0``; ``e^{K tau}`` reaches any lead time in one product."""

    name = "KAE"
    is_diffusion = False

    def __init__(
        self,
        run_dir: str | Path,
        regime: Regime,
        device="cuda",
        ckpt: Optional[str | Path] = None,
        input_frames: int = 2,
        config_path: Optional[str | Path] = None,
    ):
        """``run_dir`` is a training run directory.

        The config is taken from the YAML the run SAVED NEXT TO ITS OWN CHECKPOINTS, not
        from ``configs/experiment/...``.  The tracked configs have been edited since these
        checkpoints were trained (transformer feed-forward 512 -> 256, decoder head
        8192 -> 32768, hypernet 1024 -> 4096), so building from them raises a shape
        mismatch.  Pairing each checkpoint with its own recorded config is the only way to
        rebuild exactly the architecture that was trained, and it changes nothing on disk.
        """
        from models.autoencoder import KoopmanAutoencoder
        from models.utils import load_config, load_checkpoint
        from omegaconf import OmegaConf

        self.regime = regime
        self.dev = torch.device(device)
        run_dir = Path(run_dir)
        if config_path is not None:
            # explicit config: needed for runs that did not save one next to their
            # checkpoints (model_outputs_full). Accepts either a path to a YAML or a
            # configs/-relative name understood by the project's own load_config.
            cp = Path(config_path)
            if cp.is_file():
                self.config_path = cp
                self.cfg = OmegaConf.load(cp)
            else:
                self.config_path = Path(str(config_path))
                self.cfg = load_config(str(config_path))
        else:
            cfgs = [
                c
                for c in sorted(run_dir.glob("*.yaml"))
                if "training_history" not in c.name
            ]
            if not cfgs:
                raise FileNotFoundError(
                    f"no saved config in {run_dir}, and none given. The tracked configs/ "
                    "have drifted from some checkpoints, so pass config_path explicitly "
                    "and check the load is strict."
                )
            self.config_path = cfgs[0]
            self.cfg = OmegaConf.load(self.config_path)
        ckpt = Path(ckpt) if ckpt else run_dir / "checkpoints" / "best_model.pth"
        self.ckpt_path = ckpt
        # the run saves its MODEL config; the data side (which variables, how many input
        # frames, which control parameter) comes from the verified regime table so nothing
        # is read from the drifted tracked configs
        m = self.cfg.model if "model" in self.cfg else self.cfg
        self.n_control_frames = int(input_frames)
        data_variables = {v: 1 for v in regime.nc_fields}
        self.model = KoopmanAutoencoder(
            data_variables=data_variables,
            input_frames=self.n_control_frames,
            height=m.height,
            width=m.width,
            latent_dim=m.latent_dim,
            cond_embedding_dim=m.cond_embedding_dim,
            cond_type=m.cond_type,
            operator_mode=m.operator_mode,
            hidden_dims=m.hidden_dims,
            transformer_config=m.transformer,
            use_checkpoint=False,
            predict_cond=m.predict_cond,
            cond_grad_enabled=m.cond_grad_enabled,
            disturb_std=None,
            is_continuous=m.is_continuous,
            operator_type=m.operator_type,
            rank=m.rank,
            spectral=m.spectral,
            cond_expansion_type=regime.param_var,
            use_attention=m.use_attention,
            **m.conv_kwargs,
        ).to(self.dev)
        # load_checkpoint restores optimiser state unconditionally, so it needs one even
        # for inference; evaluate.py does the same. It is discarded immediately.
        _opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model, _, _, _ = load_checkpoint(
            str(self.ckpt_path), model=self.model, optimizer=_opt, strict=True
        )
        del _opt
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.latent_dim = int(m.latent_dim)
        self.is_continuous = bool(self.model.koopman_operator.is_continuous)
        self.vars = list(regime.nc_fields)
        self.param_name = regime.param_var
        self._norm = self._fit_normalizer()

    # -- normalisation, taken from the project's own constants ---------------
    def _fit_normalizer(self):
        from models import dataloader as dl

        table = {"tra": (dl.TRA_MEAN, dl.TRA_STD), "inc": (dl.INC_MEAN, dl.INC_STD)}
        mean, std = table[self.regime.name]
        t = lambda d, k: torch.as_tensor(float(d[k]), device=self.dev)
        fm = torch.stack([t(mean, v) for v in self.regime.nc_fields]).view(1, -1, 1, 1)
        fs = torch.stack([t(std, v) for v in self.regime.nc_fields]).view(1, -1, 1, 1)
        return fm, fs, t(mean, self.param_name), t(std, self.param_name)

    def to_model(self, x_phys):
        fm, fs, _, _ = self._norm
        sh = x_phys.shape
        return (
            (x_phys.reshape(-1, len(self.regime.nc_fields), *sh[-2:]) - fm) / fs
        ).reshape(sh)

    def to_physical(self, z):
        fm, fs, _, _ = self._norm
        sh = z.shape
        return (z.reshape(-1, len(self.regime.nc_fields), *sh[-2:]) * fs + fm).reshape(
            sh
        )

    def _cond(self, param_phys: torch.Tensor, n_t: int = 1) -> torch.Tensor:
        """The normalised control parameter as a plain [B, n_t] tensor.

        ``present_encoding`` slices ``cond_input[..., -1:]`` to take the present frame's
        condition, so this must be a Tensor (the trainer passes one too), never a
        TensorDict -- the conditioning encoder is an RBF layer that operates on tensors.
        """
        _, _, pm, ps = self._norm
        v = (param_phys.to(self.dev).float() - pm) / ps
        return v.view(-1, 1).expand(-1, n_t)

    def _td(self, x_norm: torch.Tensor) -> TensorDict:
        """[B, T, C, H, W] normalised -> the TensorDict the encoder expects."""
        B, T = x_norm.shape[:2]
        # each 2-D variable is [B, T, H, W]; present_encoding adds the channel axis
        # itself (it unsqueezes when ndim == 3 after taking the last frame)
        return TensorDict(
            {v: x_norm[:, :, i] for i, v in enumerate(self.vars)}, batch_size=[B, T]
        )

    # -- the DA interface ----------------------------------------------------
    def encode(
        self, window_phys: torch.Tensor, param_phys: torch.Tensor
    ) -> torch.Tensor:
        """Physical conditioning window [B, k, C, H, W] -> latent z0 [B, D]."""
        x = self._td(self.to_model(window_phys))
        return self.model.present_encoding(
            x, self._cond(param_phys, window_phys.shape[1])
        )

    def decode(
        self, z: torch.Tensor, obstacle_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Latent -> PHYSICAL field [B, C, H, W].

        ``KoopmanAutoencoder.decode`` takes an optional obstacle mask, not a condition:
        the conditioning enters through the encoder and the operator, not the decoder.
        """
        out = self.model.decode(z, obstacle_mask=obstacle_mask)
        f = torch.stack(
            [
                (
                    out[v].squeeze(1)
                    if out[v].dim() == 4 and out[v].shape[1] == 1
                    else out[v]
                )
                for v in self.regime.nc_fields
            ],
            dim=-3,
        )
        while f.dim() > 4:
            f = f.squeeze(1)
        return self.to_physical(f)

    @property
    def dt_train(self) -> float:
        """The operator's own training timestep -- one stored frame."""
        return float(self.model.koopman_operator.dt_train)

    def generator(self, param_phys: torch.Tensor) -> torch.Tensor:
        """The continuous generator K, built through the model's OWN cond encoding.

        These operators are conditioned (LoRA on Ma / Re), so K depends on the control
        parameter.  The encoding must go through ``dynamics._encode_cond``: there is no
        ``encode_cond`` on the operator, and passing ``None`` silently yields the
        UNCONDITIONED map, which forecasts badly and looks like a modelling failure rather
        than the wiring error it is.
        Returns [D, D], or [B, D, D] when the condition varies across the batch.
        """
        dyn = self.model.koopman_operator.dynamics
        return dyn._get_effective_linear_map(
            dyn._encode_cond(self._cond(param_phys, 1))
        )

    def step(
        self, z: torch.Tensor, param_phys: torch.Tensor, dt: Optional[float] = None
    ) -> torch.Tensor:
        """One step through the model's own operator (RK4 for linear/MLP mode).

        This is the propagation the checkpoint was trained with, so it is the reference;
        ``propagate`` below is the exact-exponential alternative.
        """
        return self.model.koopman_operator(
            z,
            cond=self._cond(param_phys, 1),
            dt=dt if dt is not None else self.dt_train,
        )

    def propagate(
        self,
        z0: torch.Tensor,
        tau: float,
        K: Optional[torch.Tensor] = None,
        param_phys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """z(tau) = exp(K tau) z0 -- one matrix product, whatever tau is."""
        if not self.is_continuous:
            raise RuntimeError("this checkpoint is not a continuous-time operator")
        if K is None:
            K = self.generator(param_phys)
        phi = torch.matrix_exp(K * float(tau))
        return (
            torch.bmm(phi, z0.unsqueeze(-1)).squeeze(-1)
            if phi.dim() == 3
            else z0 @ phi.T
        )

    def __repr__(self):
        return (
            f"KAEAdapter(latent={self.latent_dim}, k={self.n_control_frames}, "
            f"continuous={self.is_continuous}, param={self.param_name}, "
            f"cfg={self.config_path.name}, ckpt={self.ckpt_path.parent.parent.name})"
        )
