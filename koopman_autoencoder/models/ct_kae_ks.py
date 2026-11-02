"""The CT-KAE (models/ct_kae.py) on 1-D Kuramoto-Sivashinsky.

Same latent dynamics (block generator, exact propagation, mixture readout) and the same
training objective (train_ctkae.step_backward); only what is tied to the 2-D TRA grid
changes:
  * encoder / decoder: 1-D convolutions with circular padding (KS is periodic), no obstacle
    mask and no coordinate channels (KS is translation invariant);
  * no physical parameter: the condition is a dummy zero (FiLM, conditional generator and
    fixed point are off in the KS configs).
Fields keep the TRA layout [..., C, H, W] with C = H = 1 and W = 64 grid points, so the
training loop, its loss terms and the readouts apply unchanged.
"""

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch import Tensor, nn

from models.ct_kae import CTKAE, BlockGenerator, CTKAEConfig, MatrixGenerator, _groups
from models.dataloader import KS_MEAN, KS_STD

DT_KS = 0.1  # time between stored KS frames
MEAN = float(KS_MEAN["u"])
STD = float(KS_STD["u"])


class ResBlock1d(nn.Module):
    def __init__(self, cin, cout, padding_mode):
        super().__init__()
        self.n1 = nn.GroupNorm(_groups(cin), cin)
        self.c1 = nn.Conv1d(cin, cout, 3, padding=1, padding_mode=padding_mode)
        self.n2 = nn.GroupNorm(_groups(cout), cout)
        self.c2 = nn.Conv1d(cout, cout, 3, padding=1, padding_mode=padding_mode)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Encoder1d(nn.Module):
    def __init__(self, cfg: CTKAEConfig):
        super().__init__()
        w, pm = cfg.widths, cfg.padding_mode
        self.stem = nn.Conv1d(
            cfg.in_channels * cfg.n_frames, w[0], 3, padding=1, padding_mode=pm
        )
        self.stages = nn.ModuleList()
        self.down = nn.ModuleList()
        c = w[0]
        for wo in w:
            self.stages.append(
                nn.Sequential(
                    *[
                        ResBlock1d(c if j == 0 else wo, wo, pm)
                        for j in range(cfg.blocks_per_stage)
                    ]
                )
            )
            self.down.append(nn.Conv1d(wo, wo, 3, stride=2, padding=1, padding_mode=pm))
            c = wo
        self.norm = nn.GroupNorm(_groups(c), c)
        self.to_z = nn.Linear(c * (cfg.width // 2 ** len(w)), cfg.latent_dim)

    def forward(self, x):
        h = self.stem(x)
        for blocks, down in zip(self.stages, self.down):
            h = down(blocks(h))
        return self.to_z(F.silu(self.norm(h)).flatten(1))


class Decoder1d(nn.Module):
    def __init__(self, cfg: CTKAEConfig):
        super().__init__()
        w, pm = list(reversed(cfg.widths)), cfg.padding_mode
        self.w0, self.c0 = cfg.width // 2 ** len(w), w[0]
        self.from_z = nn.Linear(cfg.latent_dim, w[0] * self.w0)
        self.up = nn.ModuleList()
        self.stages = nn.ModuleList()
        c = w[0]
        for wo in w:
            self.up.append(nn.Conv1d(c, wo, 3, padding=1, padding_mode=pm))
            self.stages.append(
                nn.Sequential(
                    *[ResBlock1d(wo, wo, pm) for _ in range(cfg.blocks_per_stage)]
                )
            )
            c = wo
        self.out = nn.Sequential(
            nn.GroupNorm(_groups(c), c),
            nn.SiLU(),
            nn.Conv1d(c, cfg.in_channels, 3, padding=1, padding_mode=pm),
        )

    def forward(self, z):
        h = self.from_z(z).view(-1, self.c0, self.w0)
        for up, blocks in zip(self.up, self.stages):
            h = blocks(up(F.interpolate(h, scale_factor=2, mode="nearest")))
        return self.out(h)


class CTKAE1d(CTKAE):
    """CTKAE with 1-D encoder / decoder; generator, propagation and readout inherited."""

    def __init__(self, cfg: CTKAEConfig):
        nn.Module.__init__(self)
        if cfg.height != 1 or cfg.detail_channels or cfg.readout_damping:
            raise ValueError("CTKAE1d: height 1, no detail latent, no readout damping")
        self.cfg = cfg
        self.register_buffer("c_ref", torch.tensor(cfg.cond_ref, dtype=torch.float32))
        self.register_buffer(
            "c_scale", torch.tensor(cfg.cond_scale, dtype=torch.float32)
        )
        e = cfg.cond_embed
        self.embed = nn.Sequential(
            nn.Linear(cfg.cond_dim, e), nn.SiLU(), nn.Linear(e, e), nn.SiLU()
        )
        self.encoder = Encoder1d(cfg)
        self.decoder = Decoder1d(cfg)
        self.detail = None
        self.slow_dim = self.state_dim = cfg.latent_dim
        self.gen: nn.Module = (
            BlockGenerator(cfg)
            if cfg.generator == "block"
            else MatrixGenerator(cfg, cfg.generator)
        )
        self.fixed_point_mode = cfg.fixed_point
        if cfg.fixed_point != "none":
            raise ValueError("CTKAE1d: fixed_point 'none' only (as the candidates)")
        self.register_buffer("mu0", torch.zeros(cfg.latent_dim))
        self.mu_head = None

    def encode(self, window: Tensor, mask: Tensor, c: Optional[Tensor]) -> Tensor:
        """window [B, n, C, 1, W] (oldest first) -> z [B, d]"""
        return self.encoder(window.flatten(1, 3))

    def decode(self, z: Tensor, mask: Tensor, c: Optional[Tensor]) -> Tensor:
        """z [..., d] -> [..., C, 1, W]"""
        lead = z.shape[:-1]
        x = self.decoder(z.reshape(-1, z.shape[-1]))
        return x.view(*lead, x.shape[1], 1, x.shape[-1])


def build_ks_model(mcfg: dict) -> CTKAE1d:
    return CTKAE1d(CTKAEConfig(**mcfg))


def normalise(u):
    return (u - MEAN) / (STD + 1e-8)


def denormalise(x):
    return x * (STD + 1e-8) + MEAN


class KSWindows:
    """All frames of a KS split, [S, T, 1, 1, X] normalised, on ``device``; the TraWindows
    interface (sample, fixed_windows) with a dummy zero condition."""

    def __init__(self, path: str, device: torch.device, dtype=torch.float32, sims=None):
        ds = xr.open_dataset(path)
        u = ds["u"].values.astype(np.float32)[..., 0]  # [S, T, X]
        ds.close()
        if sims is not None:
            u = u[sims]
        self.x = torch.from_numpy(normalise(u))[:, :, None, None].to(
            device=device, dtype=dtype
        )
        self.S, self.T = self.x.shape[:2]
        self.X = self.x.shape[-1]
        self.ma = torch.zeros(self.S, device=device)
        self.mask = torch.ones(1, self.X, device=device)
        self.device = device

    def sample(
        self,
        batch: int,
        n_frames: int,
        n_leads: int,
        strides: Sequence[int],
        generator: Optional[torch.Generator] = None,
        fine: bool = False,
    ):
        if fine:
            raise ValueError("fine targets are not used on KS")
        s = int(strides[torch.randint(len(strides), (1,), generator=generator).item()])
        sim = torch.randint(self.S, (batch,), generator=generator)
        t0 = torch.randint(
            (n_frames - 1) * s, self.T - n_leads * s, (batch,), generator=generator
        )
        offs_in = torch.arange(-(n_frames - 1), 1) * s
        offs_out = torch.arange(1, n_leads + 1) * s
        sd, td = sim.to(self.device), t0.to(self.device)
        win = self.x[sd[:, None], td[:, None] + offs_in.to(self.device)]
        tgt = self.x[sd[:, None], td[:, None] + offs_out.to(self.device)]
        taus = (offs_out.float() * DT_KS).to(self.device).expand(batch, -1)
        return win.float(), tgt.float(), taus, self.ma[sd], s

    def fixed_windows(
        self, n_frames: int, n_leads: int, stride: int, starts: Sequence[int]
    ):
        out = []
        for sidx in range(self.S):
            for t0 in starts:
                if t0 - (n_frames - 1) * stride < 0 or t0 + n_leads * stride >= self.T:
                    continue
                win = self.x[sidx, t0 - (n_frames - 1) * stride : t0 + 1 : stride]
                tgt = self.x[sidx, t0 + stride : t0 + n_leads * stride + 1 : stride]
                out.append((win.float(), tgt.float()))
        win = torch.stack([o[0] for o in out])
        tgt = torch.stack([o[1] for o in out])
        taus = (
            torch.arange(1, n_leads + 1, device=self.device).float() * stride * DT_KS
        ).expand(len(out), -1)
        return win, tgt, taus, torch.zeros(len(out), device=self.device)
