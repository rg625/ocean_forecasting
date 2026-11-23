"""The KS CT-KAE (models/ct_kae_ks.CTKAE1d) behind the legacy KoopmanAutoencoder interface,
so that the paper's own training pipeline (train.py: dataset, dataloader, windows,
KoopmanLoss, AdamW, CosineWarmup, Trainer, input noise and clamping) trains it unchanged.

Only the architecture differs from the paper's KS KAE:
  * encoder: 1-D conv ResNet on the delay window of all ``input_frames`` frames (the legacy
    model encodes (history + present) / 2 with an attention history encoder);
  * dynamics: block generator of damped oscillators, exact z(t + dt) = exp(dt A) z(t);
    rolling out with exp(0.1 A) step by step and propagating to the lead times directly are
    the same map, so the latter is used;
  * decoder: 1-D conv ResNet.
Interface details kept from the legacy forward: the reconstruction decodes z0 plus N(0,
0.01^2) latent noise in training (models/autoencoder.py), rollout steps are dt = 0.1, and
the operator exposes ``dt_train`` = 0.1, so KoopmanLoss's forward/backward linearity check
applies to it exactly as to the legacy operator.

The legacy trainer takes the latent target of every target frame from
``present_encoding`` of that frame alone. A delay-window encoder has no single-frame map;
its latent of the state at a target time is the encoding of the window ending there, built
from the input and target frames of the same batch (``window_latents``; MatchedTrainer in
the training driver calls it). ``back_term`` is the CT-KAE's backward latent consistency
(train_ctkae.step_backward, w_back), added on top of the paper's loss for back001 only.
"""

from typing import Dict, List, Optional

import torch
from tensordict import TensorDict
from torch import Tensor, nn

from models.ct_kae import CTKAEConfig
from models.ct_kae_ks import CTKAE1d

DT = 0.1  # the legacy forward's base_dt


class CTOperator(nn.Module):
    """legacy KoopmanOperator call signature over the CT-KAE's exact propagator"""

    def __init__(self, ct: CTKAE1d):
        super().__init__()
        self.ct = [ct]  # not registered twice as a submodule
        self.dt_train = DT
        self.is_continuous = True

    def forward(self, z: Tensor, cond: Optional[Tensor] = None, dt=None) -> Tensor:
        dt = self.dt_train if dt is None else dt
        if isinstance(dt, Tensor):
            taus = dt.reshape(-1, 1).to(z.dtype).expand(z.shape[0], 1)
        else:
            taus = z.new_full((z.shape[0], 1), float(dt))
        return self.ct[0].propagate(z, taus, None)[:, 0].to(z.dtype)


class CTKAEMatched(nn.Module):
    def __init__(
        self,
        data_variables: Dict[str, int],
        input_frames: int,
        latent_dim: int,
        hidden_dims: List[int],
        grid: int = 64,
        ct_overrides: Optional[dict] = None,
    ):
        super().__init__()
        assert list(data_variables.values()) == [1], "KS: one scalar variable"
        self.data_variables = dict(data_variables)
        self.var = next(iter(self.data_variables))
        self.input_frames = input_frames
        self.latent_dim = latent_dim
        kw = dict(
            in_channels=1,
            n_frames=input_frames,
            height=1,
            width=grid,
            latent_dim=latent_dim,
            widths=list(hidden_dims),
            blocks_per_stage=2,
            film_encoder=False,
            film_decoder=False,
            cond_generator=False,
            fixed_point="none",
            generator="block",
            padding_mode="circular",
        )
        kw.update(ct_overrides or {})
        self.ct = CTKAE1d(CTKAEConfig(**kw))
        self.koopman_operator = CTOperator(self.ct)
        self.register_buffer("mask", torch.ones(1, grid))
        self.timings: Dict = {}  # logged by the legacy Trainer
        self._win: Optional[Tensor] = None
        self._z0: Optional[Tensor] = None

    # --- layout: legacy [B, T, X, 1]  <->  CT-KAE [B, T, C=1, H=1, W=X] -------------------
    @staticmethod
    def _to_ct(u: Tensor) -> Tensor:
        return u.squeeze(-1)[:, :, None, None, :]

    def _from_ct(self, x: Tensor) -> Tensor:  # [..., 1, 1, X] -> [..., X, 1]
        return x[..., 0, 0, :, None]

    # --- legacy interface ------------------------------------------------------------------
    def encode(self, x: TensorDict, cond_input: Optional[Tensor] = None) -> Tensor:
        u = x[self.var][:, -self.input_frames :]
        return self.ct.encode(self._to_ct(u), self.mask, None)

    def present_encoding(
        self, x: TensorDict, cond_input: Optional[Tensor] = None
    ) -> Tensor:
        """single frame: the window is the frame repeated (used only outside training)"""
        u = x[self.var][:, -1:]
        return self.ct.encode(
            self._to_ct(u).expand(-1, self.input_frames, -1, -1, -1), self.mask, None
        )

    def decode(self, z: Tensor, obstacle_mask: Optional[Tensor] = None) -> TensorDict:
        out = self._from_ct(self.ct.decode(z, self.mask, None))
        return TensorDict({self.var: out}, batch_size=list(z.shape[:-1]))

    def forward(self, x: TensorDict, seq_length, cond_future=None, drop_prob=0.0):
        from models.autoencoder import KoopmanOutput

        T = (
            int(seq_length.view(-1)[0].item())
            if isinstance(seq_length, Tensor)
            else int(seq_length)
        )
        win = self._to_ct(x[self.var][:, -self.input_frames :])
        z0 = self.ct.encode(win, self.mask, None)
        B = z0.shape[0]
        taus = DT * torch.arange(1, T + 1, device=z0.device, dtype=torch.float32)
        zs = self.ct.propagate(z0, taus.expand(B, T), None).to(z0.dtype)
        self._win, self._z0 = win, z0
        z_rec = z0 + torch.randn_like(z0) * 0.01 if self.training else z0
        x_recon = self.decode(z_rec)
        x_preds = self.decode(zs)
        return KoopmanOutput(
            x_recon=x_recon,
            x_preds=x_preds,
            z_preds=zs,
            reynolds=None,
            disturbed_latents=None,
            kept_indices=torch.arange(T, device=z0.device),
        )

    # --- training helpers ------------------------------------------------------------------
    def window_latents(self, target_td: TensorDict) -> Tensor:
        """latent of each target frame = encoding of the n-frame window ending at it"""
        assert self._win is not None, "forward first"
        tgt = self._to_ct(target_td[self.var])
        full = torch.cat([self._win.to(tgt.dtype), tgt], 1)
        n, T = self.input_frames, tgt.shape[1]
        wins = torch.stack([full[:, k + 1 : k + 1 + n] for k in range(T)], 1)
        z = self.ct.encode(wins.flatten(0, 1), self.mask, None)
        return z.view(tgt.shape[0], T, -1)

    def back_term(self, true_latents: Tensor) -> Tensor:
        """train_ctkae w_back: the last target window's latent propagated back to every
        earlier target latent and to z0 (stop-gradient references), / E|z|^2"""
        assert self._z0 is not None
        zt = true_latents.float()
        B, T, _ = zt.shape
        z_end = zt[:, -1]
        t = DT * torch.arange(1, T + 1, device=zt.device, dtype=torch.float32)
        back = torch.cat([t - t[-1], -t[-1:]]).expand(B, T + 1)
        zb = self.ct.propagate(z_end, back, None)
        ref = torch.cat([zt, self._z0.float()[:, None]], 1).detach()
        return ((zb - ref) ** 2).mean() / (zt.detach().pow(2).mean() + 1e-6)
