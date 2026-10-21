"""CT-KAE (models/ct_kae.py) behind the same interface as ``KAEAdapter``.

The DA code optimises a latent control ``z0`` and propagates it with ``exp(K tau)``. The
CT-KAE dynamics are dz/dt = A(Ma)(z - mu(Ma)), so the adapter works in the SHIFTED
coordinates y = z - mu(Ma), where the dynamics are exactly linear:

    encode(window)  -> y0 = E(window, Ma) - mu(Ma)
    generator(Ma)   -> A(Ma)                    (dy/dt = A y)
    decode(y)       -> D(mu(Ma) + y, Ma)        (physical units)

so ``solve_kae``, ``informed_latent`` and every exp(K tau) path run unchanged, and the
uninformed start y0 = 0 is the Mach-specific mean state. The CT-KAE decoder is
Mach-conditioned but the DA code calls ``decode(z)`` without a parameter; the adapter
therefore remembers the parameter from the most recent ``encode`` / ``generator`` call
(``solve_kae`` and ``informed_latent`` always make one first) and refuses to decode a
batch whose size does not match it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from data_assimilation.tra.bridge import Regime


class StructuredGenerator:
    """e^{tau K(Ma)} for a CT-KAE whose generator is too large to form densely (detail
    latent). ``apply`` acts on the adapter's shifted latent y = z - mu."""

    def __init__(self, adapter: "CTKAEAdapter", par: torch.Tensor):
        self.ad, self.par = adapter, par

    def apply(self, y: torch.Tensor, tau: float) -> torch.Tensor:
        par = self.par
        if par.shape[0] != y.shape[0]:
            par = par.expand(y.shape[0])
        mu = self.ad._mu(par)
        taus = torch.full((y.shape[0], 1), float(tau), device=y.device)
        return self.ad.model.propagate(mu + y, taus, par)[:, 0] - mu


class CTKAEAdapter:
    name = "KAE"
    is_diffusion = False

    def __init__(
        self, run_dir: str | Path, regime: Regime, device="cuda", ckpt: str = "best.pth"
    ):
        import xarray as xr

        from models.ct_data import _mask
        from models.ct_kae import CTKAE, CTKAEConfig

        self.regime = regime
        self.dev = torch.device(device)
        path = Path(run_dir) / ckpt
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model = CTKAE(CTKAEConfig(**ck["model_config"]))
        self.model.load_state_dict(ck["model"], strict=True)
        self.model = self.model.to(self.dev).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.ckpt = str(path)
        self.n_control_frames = int(self.model.cfg.n_frames)
        self.latent_dim = int(self.model.state_dim)
        self.is_continuous = True
        self.vars = list(regime.nc_fields)
        self.param_name = regime.param_var
        repo = Path(__file__).resolve().parents[2]
        self.mask = torch.from_numpy(
            _mask(xr.open_dataset(repo / "data/acdm/128_tra/train.nc"))
        ).to(self.dev)
        self._norm = self._fit_normalizer()
        self._par: Optional[torch.Tensor] = None

    # -- normalisation (identical constants to KAEAdapter / the training pipeline) ---
    def _fit_normalizer(self):
        from models import dataloader as dl

        mean, std = dl.TRA_MEAN, dl.TRA_STD
        t = lambda d, k: torch.as_tensor(float(d[k]), device=self.dev)  # noqa: E731
        fm = torch.stack([t(mean, v) for v in self.regime.nc_fields]).view(1, -1, 1, 1)
        fs = torch.stack([t(std, v) for v in self.regime.nc_fields]).view(1, -1, 1, 1)
        return fm, fs

    def to_model(self, x_phys):
        fm, fs = self._norm
        sh = x_phys.shape
        return ((x_phys.reshape(-1, len(self.vars), *sh[-2:]) - fm) / fs).reshape(sh)

    def to_physical(self, x_norm):
        fm, fs = self._norm
        sh = x_norm.shape
        return (x_norm.reshape(-1, len(self.vars), *sh[-2:]) * fs + fm).reshape(sh)

    # -- conditioning ------------------------------------------------------------
    def _remember(self, param_phys: torch.Tensor) -> torch.Tensor:
        par = param_phys.to(self.dev).float().view(-1)
        self._par = par
        return par

    def _mu(self, par: torch.Tensor) -> torch.Tensor:
        return self.model.fixed_point(self.model.embed_cond(par), par.shape[0])

    # -- the DA interface ------------------------------------------------------------
    @property
    def dt_train(self) -> float:
        return 0.1  # one benchmark frame; the generator is defined for any tau

    def encode(
        self, window_phys: torch.Tensor, param_phys: torch.Tensor
    ) -> torch.Tensor:
        """Physical window [B, k, C, H, W] -> shifted latent y0 = E - mu [B, D]."""
        par = self._remember(param_phys)
        x = self.to_model(window_phys) * self.mask
        return self.model.encode(x, self.mask, par) - self._mu(par)

    def generator(self, param_phys: torch.Tensor):
        """A(Ma) [B, D, D]: the exactly linear generator of the shifted latent. With a
        detail latent the generator is never formed densely; a StructuredGenerator that
        applies e^{tau K} through the model is returned instead (same mathematics)."""
        par = self._remember(param_phys)
        if self.model.detail is not None:
            return StructuredGenerator(self, par)
        return self.model.generator_matrix(par)

    def propagate(self, z0, tau: float, K=None, param_phys=None) -> torch.Tensor:
        if K is None:
            K = self.generator(param_phys)
        if isinstance(K, StructuredGenerator):
            return K.apply(z0, tau)
        phi = torch.matrix_exp(K * float(tau))
        return (
            torch.bmm(phi, z0.unsqueeze(-1)).squeeze(-1)
            if phi.dim() == 3
            else z0 @ phi.T
        )

    def decode(
        self, y: torch.Tensor, obstacle_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Shifted latent [B, D] -> PHYSICAL field [B, C, H, W]."""
        if self._par is None:
            raise RuntimeError(
                "CTKAEAdapter.decode needs a preceding encode()/generator() call"
            )
        par = self._par
        if par.shape[0] != y.shape[0]:
            if par.shape[0] == 1:
                par = par.expand(y.shape[0])
            else:
                raise RuntimeError(
                    f"decode batch {y.shape[0]} does not match the remembered parameter "
                    f"batch {par.shape[0]}"
                )
        x = self.model.decode(self._mu(par) + y, self.mask, par)
        return self.to_physical(x)

    def __repr__(self):
        return (
            f"CTKAEAdapter(latent={self.latent_dim}, k={self.n_control_frames}, "
            f"generator={self.model.cfg.generator}, fixed_point={self.model.fixed_point_mode}, "
            f"ckpt={self.ckpt})"
        )
