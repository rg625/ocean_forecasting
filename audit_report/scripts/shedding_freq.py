"""Vortex-shedding frequency per TRAINING simulation (train.nc), for initialising the
harmonic-lattice generator. Leading temporal POD coefficient of v_y (frames 200-999, raw
dt 0.05, space subsampled by 4), FFT peak with 8x zero padding and parabolic interpolation.
Also: share of the coefficient spectrum's energy within +-5% of the harmonics n*f0
(n = 1..4), a periodicity index. Test files are not read.

    python shedding_freq.py [--json out.json]
"""

from _paths import REPO_ROOT
import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

ap = argparse.ArgumentParser()
ap.add_argument("--json", default=None)
args = ap.parse_args()
ds = xr.open_dataset(str(REPO_ROOT / "koopman_autoencoder/data/acdm/128_tra/train.nc"))
DT = 0.05
rows = []
for i in range(ds.sizes["sim"]):
    v = ds.v_y.isel(
        sim=i, t=slice(200, None), x=slice(None, None, 4), y=slice(None, None, 4)
    ).values
    X = v.reshape(v.shape[0], -1)
    X = X - X.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    for k in range(2):
        a = U[:, k] * S[k]
        n = len(a) * 8
        P = np.abs(np.fft.rfft(a * np.hanning(len(a)), n)) ** 2
        f = np.fft.rfftfreq(n, DT)
        j = int(P[1:].argmax()) + 1
        if 0 < j < len(P) - 1:  # parabolic peak interpolation
            y0, y1, y2 = np.log(P[j - 1 : j + 2] + 1e-30)
            j = j + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        f0 = float(np.interp(j, np.arange(len(f)), f))
        on = sum(
            P[(f > n_ * f0 * 0.95) & (f < n_ * f0 * 1.05)].sum() for n_ in range(1, 5)
        )
        if k == 0:
            row = {"sim": int(ds.sim.values[i]), "Ma": float(ds.Ma.values[i])}
        row[f"omega_pod{k + 1}"] = 2 * np.pi * f0
        row[f"lattice_share_pod{k + 1}"] = float(on / P[1:].sum())
    rows.append(row)
    print(row)
ma = np.array([r["Ma"] for r in rows])
w = np.array([r["omega_pod1"] for r in rows])
m = (ma - 0.7) / 0.1
c = np.polyfit(m, w, 2)[::-1]  # c0 + c1 m + c2 m^2
res = w - (c[0] + c[1] * m + c[2] * m**2)
print(
    "quadratic fit omega_1(Ma) = c0 + c1 m + c2 m^2, m = (Ma-0.7)/0.1:",
    np.round(c, 4),
    "rms resid",
    round(float(np.sqrt((res**2).mean())), 4),
)
if args.json:
    Path(args.json).write_text(
        json.dumps(
            {
                "rows": rows,
                "quad_coef": c.tolist(),
                "rms_resid": float(np.sqrt((res**2).mean())),
            },
            indent=1,
        )
    )
