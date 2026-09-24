"""Bridge between the two codebases, in PHYSICAL units.

Neither codebase is modified.  The KAE lives in ``koopman_autoencoder`` with its own
dataloader normalisation; the autoregressive/diffusion baselines live in
``autoreg_pde_diffusion`` with hardcoded per-dataset normalisation in
``turbpred.data_transformations.Transforms``.  The ONLY common ground is the physical
state, so everything here converts to and from physical units and every reported metric is
computed there.

Two traps this module exists to handle, both verified against source rather than assumed:

1. **Channel order differs for `tra`.**  ``TurbulenceDataset`` builds its field list as
   ``["velocity"] + ["density"] + ["pressure"]``, so turbpred orders the transonic channels
   ``[v_x, v_y, rho, p]``.  The ``.nc`` files store ``[v_x, v_y, p, rho]``.  Density and
   pressure are SWAPPED.  For `inc` (``simFields=["pres"]``) the orders agree.

2. **Simulation parameters are channels, not metadata.**  turbpred appends ``mach`` (tra)
   or ``rey`` (inc) as an extra normalised channel and overwrites it with the true value at
   every autoregressive step.  The ``.nc`` files keep them as separate variables.

The normalisation constants below are READ from turbpred at import time, never copied, so
they cannot silently drift from the values the checkpoints were trained with.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import xarray as xr

REPO = Path(__file__).resolve().parents[3]
TURBPRED_SRC = REPO / "autoreg_pde_diffusion" / "src"
if str(TURBPRED_SRC) not in sys.path:
    sys.path.insert(0, str(TURBPRED_SRC))


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Regime:
    """How one dataset maps between the .nc files and turbpred's channel layout."""

    name: str
    nc_fields: List[str]  # field order as stored in the .nc
    turbpred_fields: List[str]  # SAME fields, in turbpred's channel order
    param_var: str  # .nc variable holding the simulation parameter
    param_index: int  # its index in turbpred's 7-slot norm arrays
    field_indices: List[int]  # indices of the fields in the 7-slot norm arrays
    normalize_mode: str  # the p_d.normalizeMode string that selects the constants
    data_dir: str

    @property
    def nc_to_turbpred(self) -> List[int]:
        """Permutation taking .nc field order -> turbpred field order."""
        return [self.nc_fields.index(f) for f in self.turbpred_fields]

    @property
    def turbpred_to_nc(self) -> List[int]:
        return [self.turbpred_fields.index(f) for f in self.nc_fields]


REGIMES: Dict[str, Regime] = {
    # transonic: turbpred = [velocity, density, pressure] -> rho BEFORE p
    "tra": Regime(
        name="tra",
        nc_fields=["v_x", "v_y", "p", "rho"],
        turbpred_fields=["v_x", "v_y", "rho", "p"],
        param_var="Ma",
        param_index=5,
        field_indices=[0, 1, 2, 3],
        normalize_mode="machMixed",  # confirmed against the transonic checkpoints
        data_dir="data/acdm/128_tra",
    ),
    # incompressible: turbpred = [velocity, pressure]; orders agree
    "inc": Regime(
        name="inc",
        nc_fields=["v_x", "v_y", "p"],
        turbpred_fields=["v_x", "v_y", "p"],
        param_var="Re",
        param_index=4,
        field_indices=[0, 1, 3],
        normalize_mode="karmanMixed",  # what the incompressible checkpoints actually use
        data_dir="data/acdm/128_inc",
    ),
}


# ---------------------------------------------------------------------------
def turbpred_norm(regime: Regime, p_d=None):
    """Read (mean, std) straight out of turbpred, never copied here.

    ``p_d`` should be the DataParams of the LOADED CHECKPOINT whenever one is available:
    the constants are selected by ``p_d.normalizeMode``, and the checkpoints do not all use
    the string this module would guess (the incompressible models are trained with
    ``karmanMixed``, not ``incMixed``).  Reading it from the checkpoint is what
    ``sample_models_*.py`` does, and it removes the guess entirely.  The regime's own mode
    is used only when no checkpoint is in hand, e.g. for verification.

    Returns field (mean, std) in turbpred channel order, plus the parameter (mean, std).
    """
    from turbpred.data_transformations import Transforms
    from turbpred.params import DataParams

    if p_d is None:
        p = DataParams()
        p.normalizeMode = regime.normalize_mode
        p.augmentations = ["normalize"]
    else:
        import copy

        p = copy.deepcopy(p_d)
        p.augmentations = ["normalize"]
    t = Transforms(p)
    if not hasattr(t, "normMean"):
        raise RuntimeError(
            f"turbpred did not recognise normalizeMode={p.normalizeMode!r}; "
            "its Transforms sets normMean only for known modes"
        )
    fm = t.normMean[regime.field_indices].astype(np.float64)
    fs = t.normStd[regime.field_indices].astype(np.float64)
    pm = float(t.normMean[regime.param_index])
    ps = float(t.normStd[regime.param_index])
    return fm, fs, pm, ps


# ---------------------------------------------------------------------------
class PhysicalData:
    """Ground truth in physical units, loaded once from a .nc file.

    Shapes follow ``[sim, t, C, H, W]`` with ``C`` in **.nc field order**; conversion to
    turbpred order happens only inside the adapters.
    """

    def __init__(self, nc_path: str | Path, regime: str, device="cuda"):
        self.regime = REGIMES[regime]
        self.path = Path(nc_path)
        self.dev = torch.device(device)
        with xr.open_dataset(self.path) as ds:
            miss = [v for v in self.regime.nc_fields if v not in ds.data_vars]
            if miss:
                raise KeyError(
                    f"{self.path} is missing {miss}; has {list(ds.data_vars)}"
                )
            arr = np.stack([ds[v].values for v in self.regime.nc_fields], axis=2)
            self.u = torch.as_tensor(arr, dtype=torch.float32)  # [S,T,C,H,W]
            p = ds[self.regime.param_var].values
            self.param = torch.as_tensor(np.asarray(p, dtype=np.float32)).reshape(-1)
            self.mask = (
                torch.as_tensor(ds["obstacle_mask"].values, dtype=torch.float32)
                if "obstacle_mask" in ds.data_vars
                else None
            )
        self.n_sim, self.n_t, self.C, self.H, self.W = self.u.shape
        if self.param.numel() == self.n_sim * self.n_t:
            self.param = self.param.reshape(self.n_sim, self.n_t)[:, 0]
        assert self.param.numel() == self.n_sim, (
            f"{self.regime.param_var} has {self.param.numel()} values for "
            f"{self.n_sim} sims"
        )

    def frames(self, sim: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Physical state at [sim_i, t_i] -> [B, C, H, W]."""
        return self.u[sim.cpu(), t.cpu()].to(self.dev)

    def window(self, sim: torch.Tensor, t0: torch.Tensor, n: int) -> torch.Tensor:
        """`n` consecutive physical frames starting at t0 -> [B, n, C, H, W]."""
        idx = t0.cpu().unsqueeze(1) + torch.arange(n).unsqueeze(0)
        return self.u[sim.cpu().unsqueeze(1), idx].to(self.dev)

    def params_for(self, sim: torch.Tensor) -> torch.Tensor:
        return self.param[sim.cpu()].to(self.dev)

    def mask_for(self, sim: torch.Tensor) -> Optional[torch.Tensor]:
        return None if self.mask is None else self.mask[sim.cpu()].to(self.dev)

    def __repr__(self):
        return (
            f"PhysicalData({self.path.name}, {self.regime.name}: {self.n_sim} sims x "
            f"{self.n_t} frames x {self.C}ch x {self.H}x{self.W})"
        )


# ---------------------------------------------------------------------------
def rel_l2(
    pred: torch.Tensor, true: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Relative L2 per leading index, over channels and space, in physical units.

    ``mask`` is the obstacle mask (1 = fluid): the region inside the obstacle carries no
    physics and every model treats it differently, so it is excluded from the metric rather
    than allowed to flatter whichever model happens to fill it in most plausibly.
    """
    d = pred - true
    if mask is not None:
        while mask.dim() < d.dim():
            mask = (
                mask.unsqueeze(-3) if mask.dim() == d.dim() - 1 else mask.unsqueeze(0)
            )
        d = d * mask
        true = true * mask
    dims = tuple(range(d.dim() - 3, d.dim()))
    return d.pow(2).sum(dims).sqrt() / true.pow(2).sum(dims).sqrt().clamp_min(1e-12)
