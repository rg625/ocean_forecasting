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
    cond_fixed_point: bool = True  # mu depends on c (legacy switch; see fixed_point)
    # "conditional": mu(c) = mu0 + MLP(c); "global": mu = mu0; "none": mu = 0, i.e. the
    # latent dynamics are dz/dt = A(c) z with no offset. With an offset the model is still
    # exactly linear in [z; 1] (see augmented_generator).
    fixed_point: str = "conditional"
    generator: str = "block"  # block | hurwitz | dense
    omega_max: float = 12.0  # init range of block frequencies (rad / time unit)
    gamma_init: Tuple[float, float] = (0.01, 0.3)
    dense_rank: int = 8  # low-rank conditional correction for hurwitz/dense
    padding_mode: str = "zeros"
    # multiscale decoder: channels z writes directly at each decoder stage's resolution
    # (coarse to fine, one entry per stage, 0 = none). Empty = single 1/8-resolution entry.
    z_inject: List[int] = field(default_factory=list)
    # Two-timescale latent: besides the d oscillators, a spatial detail latent of
    # detail_channels maps on the grid of encoder stage detail_level (resolution
    # H / 2^level), zero-padded by detail_pad = [py, px] cells; the padding is part of
    # the state, so wrap-around of the periodic propagator never breaks the semigroup.
    # Its generator is diagonal in Fourier space: lambda = -gamma(|k|, c) + i 2 pi k.u(c),
    # i.e. linear advection plus decay ("advect"), or decay only ("decay"). 0 = off.
    detail_channels: int = 0
    detail_mode: str = "advect"
    detail_level: int = 1
    detail_pad: List[int] = field(default_factory=lambda: [8, 16])
    # floor on the detail decay rate (per time unit; 2.5 = e-folding in 4 frames of 0.1),
    # so the detail latent cannot carry the dynamics beyond the near term
    detail_gamma_min: float = 0.0


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

    def forward(self, x, e, tap: Optional[int] = None):
        """z [B, d]; with ``tap``, also the stage-``tap`` features before downsampling."""
        h = self.stem(x)
        feat = None
        for i, (blocks, down) in enumerate(zip(self.stages, self.down)):
            for b in blocks:
                h = b(h, e)
            if i == tap:
                feat = h
            h = down(h)
        z = self.to_z(F.silu(self.norm(h)).flatten(1))
        return z if tap is None else (z, feat)


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
        # Multiscale: z also writes k channels straight onto each stage's grid (a linear,
        # POD-like path to the fine scales), added through a zero-initialised 1x1 conv so
        # training starts from the plain decoder. Still a function of (z, Ma) only, so the
        # latent dynamics are untouched.
        if cfg.z_inject and len(cfg.z_inject) != len(w):
            raise ValueError(f"z_inject needs one entry per decoder stage ({len(w)})")
        self.inject = nn.ModuleDict()
        for s, (k, wo) in enumerate(zip(cfg.z_inject, w)):
            if k <= 0:
                continue
            hs, ws = self.h0 * 2 ** (s + 1), self.w0 * 2 ** (s + 1)
            proj = nn.Conv2d(k, wo, 1)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.inject[str(s)] = nn.ModuleDict(
                {"lin": nn.Linear(cfg.latent_dim, k * hs * ws), "proj": proj}
            )

    def forward(self, z, e, extra: Optional[dict] = None):
        """``extra``: {stage: features} added after that stage's upsampling."""
        h = self.from_z(z).view(-1, self.c0, self.h0, self.w0)
        for s, (up, blocks) in enumerate(zip(self.up, self.stages)):
            h = up(F.interpolate(h, scale_factor=2, mode="nearest"))
            if str(s) in self.inject:
                inj = self.inject[str(s)]
                k = inj["proj"].in_channels
                h = h + inj["proj"](inj["lin"](z).view(-1, k, *h.shape[-2:]))
            if extra is not None and s in extra:
                h = h + extra[s]
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
class DetailLatent(nn.Module):
    """Spatial detail latent with a linear, Fourier-diagonal continuous-time generator.

    State: [B, c, Hp, Wp] (the stage grid plus its zero padding), flattened into z after
    the d oscillator coordinates. Per channel c and wavenumber k (cycles per cell),

        lambda(k) = -(gamma_min + softplus(a_c) + softplus(b_c) |k|^2 / k_max^2) + i 2 pi k . u_c

    with a, b, u shifted by a zero-initialised head of the condition embedding. Then
    exp(tau L) = F^-1 diag(e^{tau lambda}) F exactly. lambda(-k) = conj(lambda(k)), and the
    phase is zeroed on the self-conjugate Nyquist bins, so the map is real, linear, a
    semigroup and never grows (Re lambda <= 0)."""

    def __init__(self, cfg: CTKAEConfig):
        super().__init__()
        n = len(cfg.widths)
        if not 0 <= cfg.detail_level < n:
            raise ValueError(f"detail_level must be in [0, {n})")
        if cfg.detail_mode not in ("advect", "decay"):
            raise ValueError(f"unknown detail_mode {cfg.detail_mode}")
        c, lvl = cfg.detail_channels, cfg.detail_level
        self.c, self.level = c, lvl
        self.dec_stage = n - 1 - lvl
        self.h, self.w = cfg.height // 2**lvl, cfg.width // 2**lvl
        self.py, self.px = cfg.detail_pad
        self.hp, self.wp = self.h + 2 * self.py, self.w + 2 * self.px
        self.dim = c * self.hp * self.wp
        self.advect = cfg.detail_mode == "advect"
        self.gamma_min = float(cfg.detail_gamma_min)
        cin = cfg.widths[lvl]
        self.enc = nn.Sequential(
            nn.GroupNorm(_groups(cin), cin), nn.SiLU(), nn.Conv2d(cin, c, 1)
        )
        cdec = list(reversed(cfg.widths))[self.dec_stage]
        self.dec = nn.Conv2d(c, cdec, 1)
        nn.init.zeros_(self.dec.weight)  # training starts from the plain decoder
        nn.init.zeros_(self.dec.bias)
        one = float(_inv_softplus(torch.tensor(1.0)))
        self.a = nn.Parameter(torch.full((c,), one))  # e-fold time 1 at k = 0
        self.b = nn.Parameter(torch.full((c,), one))
        # advection speeds in cells per time unit, spread over channels
        self.u = nn.Parameter(
            torch.stack([torch.linspace(0.0, 10.0, c), torch.zeros(c)], -1)
        )
        self.head = None
        if cfg.cond_generator:
            self.head = nn.Sequential(
                nn.Linear(cfg.cond_embed, cfg.cond_embed),
                nn.SiLU(),
                nn.Linear(cfg.cond_embed, 4 * c),
            )
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        ky = torch.fft.fftfreq(self.hp)
        kx = torch.fft.rfftfreq(self.wp)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        nyq = (KY.abs() == 0.5) | (KX.abs() == 0.5)  # self-conjugate bins
        # deterministic from the grid, so kept out of checkpoints
        for name, v in (
            ("ky", KY),
            ("kx", KX),
            ("k2", (KY**2 + KX**2) / 0.5),  # |k|^2 / k_max^2
            ("phase_ok", (~nyq).float()),
        ):
            self.register_buffer(name, v.contiguous().clone(), persistent=False)

    def params(self, e: Optional[Tensor], batch: int):
        a, b, u = self.a.expand(batch, -1), self.b.expand(batch, -1), self.u
        u = u.expand(batch, -1, -1)
        if self.head is not None and e is not None:
            d = self.head(e).view(batch, self.c, 4)
            a, b, u = a + d[..., 0], b + d[..., 1], u + d[..., 2:]
        if not self.advect:
            u = torch.zeros_like(u)
        return F.softplus(a), F.softplus(b), u

    def rates(self, e: Optional[Tensor], batch: int) -> Tensor:
        """lambda [B, c, Hp, Wp//2+1] (complex)."""
        a, b, u = self.params(e, batch)
        gam = self.gamma_min + a[..., None, None] + b[..., None, None] * self.k2
        ph = (
            2
            * math.pi
            * (u[..., 0, None, None] * self.kx + u[..., 1, None, None] * self.ky)
        )
        return torch.complex(-gam, ph * self.phase_ok)

    def to_state(self, feat: Tensor) -> Tensor:
        """Encoder stage features -> padded detail state, flattened [B, dim]."""
        g = self.enc(feat)
        return F.pad(g, (self.px, self.px, self.py, self.py)).flatten(1)

    def to_decoder(self, y: Tensor) -> Tensor:
        """Flattened detail state [N, dim] -> decoder stage features (inner region)."""
        g = y.view(-1, self.c, self.hp, self.wp)
        g = g[..., self.py : self.py + self.h, self.px : self.px + self.w]
        return self.dec(g)

    def propagate(self, y: Tensor, taus: Tensor, e: Optional[Tensor]) -> Tensor:
        """y: [B, dim]; taus: [B, K] -> [B, K, dim], exactly e^{tau L} y."""
        B = y.shape[0]
        lam = self.rates(e, B)  # [B, c, Hp, Wf]
        Y = torch.fft.rfft2(y.view(B, self.c, self.hp, self.wp))
        m = torch.exp(lam[:, None] * taus[:, :, None, None, None])  # [B, K, c, Hp, Wf]
        out = torch.fft.irfft2(Y[:, None] * m, s=(self.hp, self.wp))
        return out.flatten(2)

    def dense_generator(self, e: Optional[Tensor] = None) -> Tensor:
        """The generator as an explicit [dim, dim] matrix (tests on tiny grids only)."""
        lam = self.rates(e, 1)[0]
        eye = torch.eye(self.dim, device=lam.device, dtype=lam.real.dtype)
        eye = eye.view(-1, self.c, self.hp, self.wp)
        cols = torch.fft.irfft2(torch.fft.rfft2(eye) * lam, s=(self.hp, self.wp))
        return cols.flatten(1).T


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
        self.detail = DetailLatent(cfg) if cfg.detail_channels > 0 else None
        self.slow_dim = cfg.latent_dim
        self.state_dim = cfg.latent_dim + (
            0 if self.detail is None else self.detail.dim
        )
        if cfg.generator == "block":
            self.gen: nn.Module = BlockGenerator(cfg)
        elif cfg.generator in ("hurwitz", "dense"):
            self.gen = MatrixGenerator(cfg, cfg.generator)
        else:
            raise ValueError(f"unknown generator {cfg.generator}")
        mode = cfg.fixed_point
        if mode == "conditional" and not cfg.cond_fixed_point:
            mode = "global"  # back-compat with configs that only set cond_fixed_point
        if mode not in ("conditional", "global", "none"):
            raise ValueError(f"unknown fixed_point {cfg.fixed_point}")
        self.fixed_point_mode = mode
        if mode == "none":
            self.register_buffer("mu0", torch.zeros(cfg.latent_dim))
        else:
            self.mu0 = nn.Parameter(torch.zeros(cfg.latent_dim))
        self.mu_head = None
        if mode == "conditional":
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
        """[B, state_dim]; the detail latent (if any) has its fixed point at 0."""
        mu = self.mu0.expand(batch, -1)
        if self.mu_head is not None and e is not None:
            mu = mu + self.mu_head(e)
        if self.detail is not None:
            mu = F.pad(mu, (0, self.detail.dim))
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
        if self.detail is None:
            return self.encoder(x, e)
        z, feat = self.encoder(x, e, tap=self.detail.level)
        return torch.cat([z, self.detail.to_state(feat).to(z.dtype)], -1)

    def decode(self, z: Tensor, mask: Tensor, c: Optional[Tensor]) -> Tensor:
        """z: [..., d] -> [..., C, H, W] (obstacle cells zeroed)."""
        lead = z.shape[:-1]
        zf = z.reshape(-1, z.shape[-1])
        e = self.embed_cond(c)
        if e is not None and e.shape[0] != zf.shape[0]:
            e = e.repeat_interleave(zf.shape[0] // e.shape[0], dim=0)
        if self.detail is None:
            x = self.decoder(zf, e)
        else:
            zs, zd = zf[:, : self.slow_dim], zf[:, self.slow_dim :]
            x = self.decoder(zs, e, {self.detail.dec_stage: self.detail.to_decoder(zd)})
        x = x * mask.to(x.dtype)  # [H, W] static obstacle mask, broadcast
        return x.view(*lead, *x.shape[1:])

    def propagate(self, z0: Tensor, taus: Tensor, c: Optional[Tensor]) -> Tensor:
        """Exact latent states at lead times taus [B, K] (time units) -> [B, K, d]."""
        e = self.embed_cond(c)
        mu = self.fixed_point(e, z0.shape[0])
        # latent dynamics always run in >= fp32, outside autocast
        ty = torch.promote_types(torch.float32, self.mu0.dtype)
        with torch.autocast(device_type=z0.device.type, enabled=False):
            y0 = (z0 - mu).to(ty)
            ee = None if e is None else e.to(ty)
            y = self.gen.propagate(y0[:, : self.slow_dim], taus.to(ty), ee)
            if self.detail is not None:
                yd = self.detail.propagate(y0[:, self.slow_dim :], taus.to(ty), ee)
                y = torch.cat([y, yd], -1)
        return mu[:, None] + y

    def forward(self, window, mask, c, taus):
        z0 = self.encode(window, mask, c)
        zs = self.propagate(z0, taus, c)
        return z0, zs

    def generator_matrix(self, c: Optional[Tensor]) -> Tensor:
        """Dense generator of the oscillator latent. With a detail latent the full
        generator is blockdiag(this, detail) and is never formed: use propagate()."""
        if self.detail is not None:
            raise NotImplementedError(
                "the detail latent's generator is not formed densely; use propagate()"
            )
        e = self.embed_cond(c)
        return self.gen.matrix(e, 1 if c is None else c.shape[0])

    def augmented_generator(self, c: Optional[Tensor]) -> Tensor:
        """Purely linear form of the latent dynamics on the state [z; 1]:

            d/dt [z; 1] = [[A, -A mu], [0, 0]] [z; 1]

        so exp(tau * A_aug) [z0; 1] = [mu + exp(tau A)(z0 - mu); 1], identical to propagate().
        The constant observable 1 is a Koopman eigenfunction with eigenvalue 0."""
        if self.detail is not None:
            raise NotImplementedError(
                "augmented form is defined for the oscillators only"
            )
        B = 1 if c is None else c.shape[0]
        e = self.embed_cond(c)
        A = self.gen.matrix(e, B)
        mu = self.fixed_point(e, B)
        d = A.shape[-1]
        Aug = torch.zeros(B, d + 1, d + 1, dtype=A.dtype, device=A.device)
        Aug[:, :d, :d] = A
        Aug[:, :d, d] = -(A @ mu.to(A.dtype)[..., None])[..., 0]
        return Aug

    def config_dict(self):
        return asdict(self.cfg)
