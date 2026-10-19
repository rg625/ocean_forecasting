"""Conditional continuous-time Koopman autoencoder (CT-KAE), structured redesign.

    z(t)        = E_phi(x_{t-n+1..t}, c)                    encoder on an n-frame window
    dz/dt       = A(c) (z - mu(c))                          linear generator about a
                                                            parameter-dependent fixed point
    z(t + tau)  = mu(c) + exp(tau A(c)) (z(t) - mu(c))      EXACT propagation, any tau
    x_hat(t)    = D_theta(z(t), c)

c is the physical parameter (Mach number for TRA), embedded after nondimensionalising
as (c - c_ref) / c_scale.

Generators (``generator=``):
  "block"  : A(c) = blockdiag_j [[-g_j(c), -w_j(c)], [w_j(c), -g_j(c)]],
             g_j = softplus(.) >= 0 (never growing, can be ~neutral), w_j(c) free.
             exp(tau A) is a decay x rotation per block: O(d) and exact.
  "hurwitz": A(c) = J(c) - R(c),  J skew, R = L L^T + eps I  (contractive), matrix_exp.
  "dense"  : A(c) = skew(W) + sym(D) + low-rank(c), unconstrained (legacy form), matrix_exp.

Everything lives in this file so the legacy KoopmanAutoencoder stays the reference.
"""

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class CTKAEConfig:
    in_channels: int = 4
    n_frames: int = 2  # input window length (delay embedding)
    height: int = 64
    width: int = 128
    latent_dim: int = 256  # must be even for "block"
    widths: List[int] = field(default_factory=lambda: [64, 128, 256])
    blocks_per_stage: int = 2
    cond_dim: int = 1  # number of physical parameters
    cond_ref: List[float] = field(default_factory=lambda: [0.7])
    cond_scale: List[float] = field(default_factory=lambda: [0.1])
    cond_embed: int = 64
    film_encoder: bool = True
    film_decoder: bool = True
    cond_generator: bool = True  # A depends on c
    cond_fixed_point: bool = True  # mu depends on c
    generator: str = "block"  # block | hurwitz | dense
    omega_max: float = 12.0  # init range of block frequencies (rad / time unit)
    gamma_init: Tuple[float, float] = (0.01, 0.3)
    dense_rank: int = 8  # low-rank conditional correction for hurwitz/dense
    padding_mode: str = "zeros"


# ----------------------------------------------------------------------------- utils
def _groups(c: int) -> int:
    for g in (32, 16, 8, 4, 2):
        if c % g == 0:
            return g
    return 1


class FiLM(nn.Module):
    """Feature-wise affine modulation from the condition embedding (zero-init: identity)."""

    def __init__(self, cond_embed: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_embed, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: Tensor, e: Optional[Tensor]) -> Tensor:
        if e is None:
            return h
        s, b = self.proj(e).chunk(2, dim=-1)
        return h * (1 + s[..., None, None]) + b[..., None, None]


class ResBlock(nn.Module):
    def __init__(self, cin, cout, cond_embed, film, padding_mode):
        super().__init__()
        self.n1 = nn.GroupNorm(_groups(cin), cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, padding_mode=padding_mode)
        self.n2 = nn.GroupNorm(_groups(cout), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, padding_mode=padding_mode)
        self.film = FiLM(cond_embed, cout) if film else None
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        h = self.n2(h)
        if self.film is not None:
            h = self.film(h, e)
        h = self.c2(F.silu(h))
        return h + self.skip(x)


# --------------------------------------------------------------------- encoder/decoder
class Encoder(nn.Module):
    def __init__(self, cfg: CTKAEConfig):
        super().__init__()
        cin = cfg.in_channels * cfg.n_frames + 1 + 2  # fields x frames, mask, coords
        w = cfg.widths
        self.stem = nn.Conv2d(cin, w[0], 3, padding=1, padding_mode=cfg.padding_mode)
        self.stages = nn.ModuleList()
        self.down = nn.ModuleList()
        c = w[0]
        for i, wo in enumerate(w):
            blocks = nn.ModuleList(
                [
                    ResBlock(
                        c if j == 0 else wo,
                        wo,
                        cfg.cond_embed,
                        cfg.film_encoder,
                        cfg.padding_mode,
                    )
                    for j in range(cfg.blocks_per_stage)
                ]
            )
            self.stages.append(blocks)
            self.down.append(nn.Conv2d(wo, wo, 3, stride=2, padding=1))
            c = wo
        h, wd = cfg.height // 2 ** len(w), cfg.width // 2 ** len(w)
        self.norm = nn.GroupNorm(_groups(c), c)
        self.to_z = nn.Linear(c * h * wd, cfg.latent_dim)  # flatten: keeps "where"

    def forward(self, x, e):
        h = self.stem(x)
        for blocks, down in zip(self.stages, self.down):
            for b in blocks:
                h = b(h, e)
            h = down(h)
        return self.to_z(F.silu(self.norm(h)).flatten(1))


class Decoder(nn.Module):
    def __init__(self, cfg: CTKAEConfig):
        super().__init__()
        w = list(reversed(cfg.widths))
        self.h0, self.w0 = cfg.height // 2 ** len(w), cfg.width // 2 ** len(w)
        self.c0 = w[0]
        self.from_z = nn.Linear(cfg.latent_dim, w[0] * self.h0 * self.w0)
        self.stages = nn.ModuleList()
        self.up = nn.ModuleList()
        c = w[0]
        for wo in w:
            self.up.append(
                nn.Conv2d(c, wo, 3, padding=1, padding_mode=cfg.padding_mode)
            )
            self.stages.append(
                nn.ModuleList(
                    [
                        ResBlock(
                            wo, wo, cfg.cond_embed, cfg.film_decoder, cfg.padding_mode
                        )
                        for _ in range(cfg.blocks_per_stage)
                    ]
                )
            )
            c = wo
        self.out = nn.Sequential(
            nn.GroupNorm(_groups(c), c),
            nn.SiLU(),
            nn.Conv2d(c, cfg.in_channels, 3, padding=1, padding_mode=cfg.padding_mode),
        )

    def forward(self, z, e):
        h = self.from_z(z).view(-1, self.c0, self.h0, self.w0)
        for up, blocks in zip(self.up, self.stages):
            h = up(F.interpolate(h, scale_factor=2, mode="nearest"))
            for b in blocks:
                h = b(h, e)
        return self.out(h)


# ----------------------------------------------------------------------- generators
def _inv_softplus(y: Tensor) -> Tensor:
    return y + torch.log(-torch.expm1(-y))


class BlockGenerator(nn.Module):
    """A(c) = blockdiag([[-g, -w], [w, -g]]); exact exp(tau A) = e^{-g tau} R(w tau)."""

    def __init__(self, cfg: CTKAEConfig):
        super().__init__()
        assert cfg.latent_dim % 2 == 0
        m = cfg.latent_dim // 2
        self.m = m
        g0 = torch.exp(
            torch.empty(m).uniform_(
                math.log(cfg.gamma_init[0]), math.log(cfg.gamma_init[1])
            )
        )
        self.g_raw = nn.Parameter(_inv_softplus(g0))
        # frequencies log-spaced over (0.05, omega_max) with random sign-free jitter
        w0 = torch.exp(torch.linspace(math.log(0.05), math.log(cfg.omega_max), m))
        self.w = nn.Parameter(w0[torch.randperm(m)])
        self.head = None
        if cfg.cond_generator:
            self.head = nn.Sequential(
                nn.Linear(cfg.cond_embed, cfg.cond_embed),
                nn.SiLU(),
                nn.Linear(cfg.cond_embed, 2 * m),
            )
            nn.init.zeros_(
                self.head[-1].weight
            )  # starts at the unconditional generator
            nn.init.zeros_(self.head[-1].bias)

    def params(self, e: Optional[Tensor], batch: int) -> Tuple[Tensor, Tensor]:
        g_raw, w = self.g_raw.expand(batch, -1), self.w.expand(batch, -1)
        if self.head is not None and e is not None:
            dg, dw = self.head(e).chunk(2, dim=-1)
            g_raw, w = g_raw + dg, w + dw
        return F.softplus(g_raw), w

    def propagate(self, y: Tensor, taus: Tensor, e: Optional[Tensor]) -> Tensor:
        """y: [B, d] (deviation from fixed point); taus: [B, K] -> [B, K, d]."""
        B = y.shape[0]
        g, w = self.params(e, B)  # [B, m]
        a = torch.exp(-g[:, None] * taus[..., None])  # [B, K, m]
        th = w[:, None] * taus[..., None]
        c, s = torch.cos(th) * a, torch.sin(th) * a
        y1, y2 = y[:, 0::2][:, None], y[:, 1::2][:, None]  # [B, 1, m]
        out = torch.stack([c * y1 - s * y2, s * y1 + c * y2], dim=-1)  # [B, K, m, 2]
        return out.flatten(-2)

    def matrix(self, e: Optional[Tensor], batch: int = 1) -> Tensor:
        g, w = self.params(e, batch)
        A = torch.zeros(batch, 2 * self.m, 2 * self.m, device=g.device, dtype=g.dtype)
        i = torch.arange(self.m, device=g.device)
        A[:, 2 * i, 2 * i] = -g
        A[:, 2 * i + 1, 2 * i + 1] = -g
        A[:, 2 * i, 2 * i + 1] = -w
        A[:, 2 * i + 1, 2 * i] = w
        return A


class MatrixGenerator(nn.Module):
    """hurwitz: A = J - (L L^T + eps I); dense: A = skew(W) + sym(D). Low-rank c-correction."""

    def __init__(self, cfg: CTKAEConfig, kind: str):
        super().__init__()
        d = cfg.latent_dim
        self.kind, self.d, self.r = kind, d, cfg.dense_rank
        self.W = nn.Parameter(torch.randn(d, d) * (1.0 / math.sqrt(d)))
        if kind == "hurwitz":
            self.L = nn.Parameter(torch.randn(d, d) * (0.1 / math.sqrt(d)))
        else:
            self.D = nn.Parameter(torch.zeros(d, d))
        self.head = None
        if cfg.cond_generator and self.r > 0:
            self.head = nn.Sequential(
                nn.Linear(cfg.cond_embed, cfg.cond_embed),
                nn.SiLU(),
                nn.Linear(cfg.cond_embed, 2 * self.r * d),
            )
            # LoRA-style: U random, V zero -> zero update, non-zero gradient (cf. BUG-02)
            nn.init.zeros_(self.head[-1].bias)
            with torch.no_grad():
                wv = self.head[-1].weight.view(2, self.r * d, -1)
                wv[0].normal_(0, cfg.cond_embed**-0.5)
                wv[1].zero_()

    def matrix(self, e: Optional[Tensor], batch: int = 1) -> Tensor:
        Wb = self.W.expand(batch, -1, -1)
        if self.head is not None and e is not None:
            U, V = self.head(e).view(batch, 2, self.r, self.d).unbind(1)
            Wb = Wb + torch.einsum("brd,bre->bde", U, V) / self.r
        J = 0.5 * (Wb - Wb.transpose(1, 2))
        if self.kind == "hurwitz":
            R = self.L @ self.L.T + 1e-4 * torch.eye(self.d, device=self.L.device)
            return J - R
        S = 0.5 * (Wb + Wb.transpose(1, 2)) + 0.5 * (self.D + self.D.T)
        return J + S

    def propagate(self, y: Tensor, taus: Tensor, e: Optional[Tensor]) -> Tensor:
        B, K = taus.shape
        A = self.matrix(e, B)  # [B, d, d]
        grid = taus[:, :1] * torch.arange(1, K + 1, device=taus.device)
        if torch.allclose(taus, grid, rtol=1e-6, atol=1e-7):
            # uniform grid: one expm, then repeated products (exact up to round-off)
            P = torch.linalg.matrix_exp(A * taus[:, :1, None])
            outs, cur = [], y
            for _ in range(K):
                cur = torch.einsum("bij,bj->bi", P, cur)
                outs.append(cur)
            return torch.stack(outs, 1)
        E = torch.linalg.matrix_exp(A[:, None] * taus[..., None, None])  # [B, K, d, d]
        return torch.einsum("bkij,bj->bki", E, y)


# ------------------------------------------------------------------------- the model
class CTKAE(nn.Module):
    def __init__(self, cfg: CTKAEConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("c_ref", torch.tensor(cfg.cond_ref, dtype=torch.float32))
        self.register_buffer(
            "c_scale", torch.tensor(cfg.cond_scale, dtype=torch.float32)
        )
        e = cfg.cond_embed
        self.embed = nn.Sequential(
            nn.Linear(cfg.cond_dim, e), nn.SiLU(), nn.Linear(e, e), nn.SiLU()
        )
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        if cfg.generator == "block":
            self.gen: nn.Module = BlockGenerator(cfg)
        elif cfg.generator in ("hurwitz", "dense"):
            self.gen = MatrixGenerator(cfg, cfg.generator)
        else:
            raise ValueError(f"unknown generator {cfg.generator}")
        self.mu0 = nn.Parameter(torch.zeros(cfg.latent_dim))
        self.mu_head = None
        if cfg.cond_fixed_point:
            self.mu_head = nn.Sequential(
                nn.Linear(e, e), nn.SiLU(), nn.Linear(e, cfg.latent_dim)
            )
            nn.init.zeros_(self.mu_head[-1].weight)
            nn.init.zeros_(self.mu_head[-1].bias)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, cfg.height),
            torch.linspace(-1, 1, cfg.width),
            indexing="ij",
        )
        self.register_buffer("coords", torch.stack([xx, yy])[None])

    # --- conditioning ----------------------------------------------------------------
    def embed_cond(self, c: Optional[Tensor]) -> Optional[Tensor]:
        if c is None:
            return None
        c = c.view(c.shape[0], -1).to(self.c_ref.dtype)
        return self.embed((c - self.c_ref) / self.c_scale)

    def fixed_point(self, e: Optional[Tensor], batch: int) -> Tensor:
        mu = self.mu0.expand(batch, -1)
        if self.mu_head is not None and e is not None:
            mu = mu + self.mu_head(e)
        return mu

    # --- maps ----------------------------------------------------------------------
    def encode(self, window: Tensor, mask: Tensor, c: Optional[Tensor]) -> Tensor:
        """window: [B, n, C, H, W] (oldest first); mask: [H, W] or [B, H, W]."""
        B = window.shape[0]
        e = self.embed_cond(c)
        m = mask.expand(B, *mask.shape[-2:])[:, None].to(window.dtype)
        x = torch.cat(
            [
                window.flatten(1, 2),
                m,
                self.coords.expand(B, -1, -1, -1).to(window.dtype),
            ],
            1,
        )
        return self.encoder(x, e)

    def decode(self, z: Tensor, mask: Tensor, c: Optional[Tensor]) -> Tensor:
        """z: [..., d] -> [..., C, H, W] (obstacle cells zeroed)."""
        lead = z.shape[:-1]
        zf = z.reshape(-1, z.shape[-1])
        e = self.embed_cond(c)
        if e is not None and e.shape[0] != zf.shape[0]:
            e = e.repeat_interleave(zf.shape[0] // e.shape[0], dim=0)
        x = self.decoder(zf, e)
        x = x * mask.to(x.dtype)  # [H, W] static obstacle mask, broadcast
        return x.view(*lead, *x.shape[1:])

    def propagate(self, z0: Tensor, taus: Tensor, c: Optional[Tensor]) -> Tensor:
        """Exact latent states at lead times taus [B, K] (time units) -> [B, K, d]."""
        e = self.embed_cond(c)
        mu = self.fixed_point(e, z0.shape[0])
        # latent dynamics always run in >= fp32, outside autocast
        ty = torch.promote_types(torch.float32, self.mu0.dtype)
        with torch.autocast(device_type=z0.device.type, enabled=False):
            y = self.gen.propagate(
                (z0 - mu).to(ty), taus.to(ty), None if e is None else e.to(ty)
            )
        return mu[:, None] + y

    def forward(self, window, mask, c, taus):
        z0 = self.encode(window, mask, c)
        zs = self.propagate(z0, taus, c)
        return z0, zs

    def generator_matrix(self, c: Optional[Tensor]) -> Tensor:
        e = self.embed_cond(c)
        return self.gen.matrix(e, 1 if c is None else c.shape[0])

    def config_dict(self):
        return asdict(self.cfg)
