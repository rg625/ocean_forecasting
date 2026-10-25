"""GPU-resident TRA data for the CT-KAE: window sampling at arbitrary temporal stride and
evaluation sets in the ACDM ground-truth layout.

Raw TRA frames are DT_RAW = 0.05 apart. Training windows use frames spaced by an integer
stride s (s = 2 reproduces the benchmark's dt = 0.1); sampling several strides trains the
generator at several physical time steps, which is what makes it genuinely continuous-time.

Fields are z-scored with the same constants as the legacy pipeline (TRA_MEAN / TRA_STD) and
the obstacle cells are zeroed. Channel order is VARS = [v_x, v_y, p, rho]; the ACDM ground
truth uses [v_x, v_y, rho, p], handled by ``to_acdm_layout``.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import xarray as xr

from .dataloader import TRA_MEAN, TRA_STD

VARS = ["v_x", "v_y", "p", "rho"]
DT_RAW = 0.05
ACDM_ORDER = [0, 1, 3, 2]  # VARS index for [v_x, v_y, rho, p]


def _mask(ds) -> np.ndarray:
    m = ds.obstacle_mask.values
    while m.ndim > 2:
        m = m[0]
    out: np.ndarray = (m > 0.5).astype(np.float32)
    return out


def normalise(ds, sims=None, t=slice(None)) -> np.ndarray:
    """[S, T, C, H, W] float32, z-scored and obstacle-masked."""
    sel = dict(t=t) if sims is None else dict(sim=sims, t=t)
    m = _mask(ds)
    return np.stack(
        [
            ((ds[v].isel(**sel).values - TRA_MEAN[v]) / (TRA_STD[v] + 1e-8)) * m
            for v in VARS
        ],
        axis=2,
    ).astype(np.float32)


def denormalise(x: torch.Tensor) -> torch.Tensor:
    """[..., C, H, W] normalised (VARS order) -> physical units."""
    mean = torch.tensor([TRA_MEAN[v] for v in VARS], device=x.device, dtype=x.dtype)
    std = torch.tensor(
        [TRA_STD[v] + 1e-8 for v in VARS], device=x.device, dtype=x.dtype
    )
    return x * std[:, None, None] + mean[:, None, None]


def channel_weights(mode: str) -> torch.Tensor:
    """Loss weights per channel. 'phys' weights by std^2 so the normalised-space MSE is
    proportional to the physical-unit MSE the benchmark reports; 'uniform' weights equally.
    """
    if mode == "uniform":
        return torch.ones(len(VARS))
    s2 = torch.tensor([TRA_STD[v] ** 2 for v in VARS])
    return s2 / s2.mean()


class TraWindows:
    """All frames of the selected simulations, resident on ``device`` in ``dtype``."""

    def __init__(
        self,
        path: str,
        ma_intervals: Optional[Sequence[Tuple[float, float]]],
        device: torch.device,
        dtype=torch.float16,
        extra_raw_sims: Sequence[str] = (),
    ):
        ds = xr.open_dataset(path)
        ma = ds.Ma.values.astype(np.float32)
        keep = (
            np.ones_like(ma, dtype=bool)
            if ma_intervals is None
            else np.zeros_like(ma, dtype=bool)
        )
        for lo, hi in ma_intervals or []:
            keep |= (ma >= lo) & (ma <= hi)
        idx = np.nonzero(keep)[0]
        self.ma = torch.from_numpy(ma[idx]).to(device)
        self.mask = torch.from_numpy(_mask(ds)).to(device)
        x = normalise(ds, sims=idx)
        if extra_raw_sims:  # e.g. ACDM sim 40 (Ma 0.90), absent from train.nc
            T = x.shape[1]
            raw = [load_raw_sim(d, T, _mask(ds)) for d in extra_raw_sims]
            x = np.concatenate([x] + [r[0][None] for r in raw], axis=0)
            self.ma = torch.cat(
                [self.ma, torch.tensor([r[1] for r in raw], device=device)]
            )
        self.x = torch.from_numpy(x).to(device=device, dtype=dtype)
        self.S, self.T = self.x.shape[:2]
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
        """Random windows. Returns window [B, n, C, H, W] (oldest first, frames t-(n-1)s..t),
        targets [B, K, C, H, W] (frames t+s..t+Ks), taus [B, K] in time units, ma [B], s.

        fine=True keeps the window and the horizon Ks but draws the K target frames on the
        raw grid (spacing DT_RAW): K-1 distinct offsets from 1..Ks-1 plus Ks, shared by the
        batch. Their encoder windows are then not part of [window, targets], so the n-1
        history frames of each target (offsets o-(n-1)s..o-s) are returned as a sixth
        value hist [B, K, n-1, C, H, W].
        """
        s = int(strides[torch.randint(len(strides), (1,), generator=generator).item()])
        span = (n_frames - 1 + n_leads) * s
        sim = torch.randint(self.S, (batch,), generator=generator)
        t0 = torch.randint(
            (n_frames - 1) * s, self.T - n_leads * s, (batch,), generator=generator
        )
        offs_in = torch.arange(-(n_frames - 1), 1) * s
        offs_out = torch.arange(1, n_leads + 1) * s
        if fine:
            pick = torch.randperm(n_leads * s - 1, generator=generator)[: n_leads - 1]
            offs_out = torch.cat([torch.sort(pick + 1).values, offs_out[-1:]])
        assert (
            int(t0.max()) + n_leads * s < self.T
            and int(t0.min()) - (n_frames - 1) * s >= 0
        ), span
        sim_d, t0_d = sim.to(self.device), t0.to(self.device)
        win = self.x[sim_d[:, None], t0_d[:, None] + offs_in.to(self.device)]
        tgt = self.x[sim_d[:, None], t0_d[:, None] + offs_out.to(self.device)]
        taus = (offs_out.float() * DT_RAW).to(self.device).expand(batch, -1)
        if fine:
            offs_h = offs_out[:, None] + torch.arange(-(n_frames - 1), 0)[None] * s
            hist = self.x[
                sim_d[:, None, None], t0_d[:, None, None] + offs_h.to(self.device)
            ]
            return win.float(), tgt.float(), taus, self.ma[sim_d], s, hist.float()
        return win.float(), tgt.float(), taus, self.ma[sim_d], s

    def fixed_windows(
        self, n_frames: int, n_leads: int, stride: int, starts: Sequence[int]
    ):
        """Deterministic validation windows at the given raw-frame start indices."""
        out = []
        for sidx in range(self.S):
            for t0 in starts:
                if t0 - (n_frames - 1) * stride < 0 or t0 + n_leads * stride >= self.T:
                    continue
                win = self.x[sidx, t0 - (n_frames - 1) * stride : t0 + 1 : stride]
                tgt = self.x[sidx, t0 + stride : t0 + n_leads * stride + 1 : stride]
                out.append((win.float(), tgt.float(), self.ma[sidx]))
        win = torch.stack([o[0] for o in out])
        tgt = torch.stack([o[1] for o in out])
        ma = torch.stack([o[2] for o in out])
        taus = (
            torch.arange(1, n_leads + 1, device=self.device).float() * stride * DT_RAW
        ).expand(len(out), -1)
        return win, tgt, taus, ma


def load_raw_sim(sim_dir: str, n_frames: int, mask: np.ndarray):
    """One simulation from the official ACDM export (sim_XXXXXX/{velocity,density,pressure}_
    NNNNNN.npz, arrays [c, 128, 64]) -> ([T, C, 64, 128] normalised + masked, Mach).
    Frame t of the export equals frame t of train.nc after a transpose (verified on sim 39
    to 3e-8)."""
    import json

    d = Path(sim_dir)
    ma = float(json.loads((d / "src" / "description.json").read_text())["Mach Number"])
    frames = []
    for t in range(n_frames):
        vel = np.load(d / f"velocity_{t:06d}.npz")["arr_0"]
        rho = np.load(d / f"density_{t:06d}.npz")["arr_0"][0]
        p = np.load(d / f"pressure_{t:06d}.npz")["arr_0"][0]
        fields = {"v_x": vel[0].T, "v_y": vel[1].T, "p": p.T, "rho": rho.T}
        frames.append(
            np.stack(
                [((fields[v] - TRA_MEAN[v]) / (TRA_STD[v] + 1e-8)) * mask for v in VARS]
            )
        )
    return np.stack(frames).astype(np.float32), ma


def load_eval_case(path: str, device: torch.device, stride: int = 1):
    """gt_<case>.nc -> (x [S, T, C, H, W] normalised, ma [S], mask). gt_interp/extrap/longer
    store frames 0.1 apart (benchmark resolution); *_subsample_1 / *_1000 files are 0.05
    apart despite their t coordinate (audit DATA-03), so use stride=2 for those."""
    ds = xr.open_dataset(path)
    x = normalise(ds, t=slice(None, None, stride))
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(ds.Ma.values.astype(np.float32)).to(device),
        torch.from_numpy(_mask(ds)).to(device),
    )


def to_acdm_layout(x_norm: torch.Tensor, ma: torch.Tensor) -> np.ndarray:
    """[S, T, C, H, W] normalised (VARS order, H=64, W=128) -> ACDM ground-truth layout
    [S, T, 5, 128, 64] physical units, fields [v_x, v_y, rho, p, Ma]."""
    phys = denormalise(x_norm)[:, :, ACDM_ORDER]
    phys = phys.transpose(-1, -2)
    S, T = phys.shape[:2]
    m = ma.view(S, 1, 1, 1, 1).expand(S, T, 1, *phys.shape[-2:]).to(phys.dtype)
    out: np.ndarray = torch.cat([phys, m], 2).cpu().numpy()
    return out


def train_val_intervals(
    val_ma: Sequence[float], lo: float = 0.525, hi: float = 0.905, half: float = 0.005
):
    """Ma intervals for a training set that excludes the validation Mach numbers."""
    edges = sorted(val_ma)
    train, cur = [], lo
    for v in edges:
        train.append((cur, v - half))
        cur = v + half
    train.append((cur, hi))
    val = [(v - half, v + half) for v in edges]
    return train, val


def describe(ds: TraWindows) -> Dict:
    return {"sims": ds.S, "frames": ds.T, "ma": [round(float(m), 3) for m in ds.ma]}


__all__: List[str] = [
    "TraWindows",
    "load_eval_case",
    "to_acdm_layout",
    "channel_weights",
    "denormalise",
    "train_val_intervals",
    "VARS",
    "DT_RAW",
    "describe",
]
