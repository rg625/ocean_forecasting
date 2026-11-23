"""Build data/acdm/128_tra/val_heldout.nc: validation trajectories in the gt_<case>.nc format
from the TRAINING simulations held out in the loss sweep (Ma 0.60, 0.75), for validation-only
model selection (forecasting and DA). Four 60-frame segments per simulation at the
benchmark spacing (every 2nd raw frame, 0.1 apart), starting at raw frames 200/400/600/800.
No test file is read.

    python make_val_nc.py
"""

from _paths import REPO_ROOT
import numpy as np
import xarray as xr

ROOT = str(REPO_ROOT / "koopman_autoencoder/data/acdm/128_tra")
ds = xr.open_dataset(f"{ROOT}/train.nc")
VAL_MA = (0.60, 0.75)
segs = []
for ma in VAL_MA:
    i = int(np.argmin(np.abs(ds.Ma.values - ma)))
    assert abs(float(ds.Ma.values[i]) - ma) < 1e-3, ds.Ma.values[i]
    for t0 in (200, 400, 600, 800):
        segs.append(ds.isel(sim=i, t=slice(t0, t0 + 120, 2)))
out = xr.concat(
    [
        s.drop_vars(["t", "obstacle_mask"]).assign_coords(t=np.arange(60) * 0.1)
        for s in segs
    ],
    dim="sim",
)
out = out.assign_coords(sim=np.arange(len(segs)))
out["obstacle_mask"] = (
    ("sim", "x", "y"),
    np.stack([ds.obstacle_mask.values] * len(segs)),
)
out = out[["v_x", "v_y", "p", "rho", "obstacle_mask"]]
out.to_netcdf(f"{ROOT}/val_heldout.nc")
print(out)
