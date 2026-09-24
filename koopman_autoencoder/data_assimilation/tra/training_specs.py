"""The reproducibility table, read from the artefacts the campaign actually loaded.

Nothing here is typed in by hand.  Every field comes out of the checkpoint, the YAML that
checkpoint's own run saved, or the .nc file itself, so the table cannot drift away from the
models that produced the numbers.  Ported from ``data_assimilation/ks/training_specs.py`` in the KS campaign.

Two things it is specifically there to make checkable:

  * **Test discipline.** The DA test regime is ``gt_interp.nc`` (Mach 0.66-0.68) and the
    validation regime used for every hyper-parameter is ``gt_extrap.nc`` (Mach 0.50-0.52).
    The Mach ranges are read from the files and reported, so the claim that the two do not
    overlap is a measurement.
  * **What each baseline was trained to do.** The turbpred checkpoints are trained on a
    2-frame sequence with ``predMSE`` only; the KAE on rollouts with a Koopman generator.
    A reader comparing an inverse-problem result across them should be able to see that.

``python -m data_assimilation.tra.training_specs`` writes ``da_results_tra/training_specs.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml

from data_assimilation.tra.bridge import REPO
from data_assimilation.tra.verify_adapters import MODELS

TURB = REPO / "autoreg_pde_diffusion" / "pretrained_models" / "models_tra"
KAE_RUN = Path("model_outputs_tra/continous_linear_128/run-20260821_031121")
OUT = Path("da_results_tra")
DATA = {
    "test (DA)": "data/acdm/128_tra/gt_interp.nc",
    "validation (tuning only)": "data/acdm/128_tra/gt_extrap.nc",
    "long (post-DA forecast)": "data/acdm/128_tra/gt_longer.nc",
    "turbpred training set": "data/acdm/128_tra/train.nc",
}


def sha(p: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()[:n]


def n_params(sd) -> int:
    return int(sum(v.numel() for v in sd.values() if torch.is_tensor(v)))


def dataset_row(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": path, "present": False}
    with xr.open_dataset(p) as ds:
        # the Mach number is a COORDINATE ("Ma"), not a data variable -- the same name
        # data_assimilation.tra.bridge.Regime.param_var reads, so the two cannot disagree
        mach, mach_values = None, None
        for k in ("Ma", "mach", "Mach"):
            if k in ds.coords or k in ds:
                v = np.asarray(ds[k].values, dtype=float).ravel()
                mach = [float(v.min()), float(v.max())]
                mach_values = sorted({round(float(x), 6) for x in v})
                break
        return {
            "path": path,
            "present": True,
            "sizes": {k: int(v) for k, v in ds.sizes.items()},
            "variables": sorted(map(str, ds.data_vars)),
            "mach_range": mach,
            "mach_values": mach_values,
            "bytes": p.stat().st_size,
            # the training set is 2.7 GB; hashing it every run is not worth the I/O
            "sha256_12": (
                sha(p) if p.stat().st_size < 2 << 28 else "not hashed (>512 MB)"
            ),
        }


def turbpred_row(name: str) -> dict:
    ck = torch.load(
        TURB / MODELS[name] / "Model.pth", map_location="cpu", weights_only=False
    )
    dp, tp, lp = ck["dataParams"], ck["trainingParams"], ck["lossParams"]
    return {
        "checkpoint": str((TURB / MODELS[name] / "Model.pth").relative_to(REPO)),
        "arch": ck["modelParamsDecoder"].get("arch"),
        "params": n_params(ck["stateDictDecoder"]),
        "data_size": dp.get("dataSize"),
        "sim_fields": dp.get("simFields"),
        "sim_params": dp.get("simParams"),
        "sequence_length": dp.get("sequenceLength"),
        "normalize_mode": dp.get("normalizeMode"),
        "epochs": tp.get("epochs"),
        "lr": tp.get("lr"),
        "batch": dp.get("batch"),
        "losses": {k: v for k, v in lp.items() if v},
        "conditioning_integration": ck["modelParamsDecoder"].get(
            "conditioningIntegration", "n/a (deterministic)"
        ),
        "sha256_12": sha(TURB / MODELS[name] / "Model.pth"),
    }


def kae_row() -> dict:
    ckp = KAE_RUN / "checkpoints" / "best_model.pth"
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    ymls = sorted(KAE_RUN.glob("*.yaml"))
    cfg = yaml.safe_load(ymls[0].read_text()) if ymls else {}
    hist = ck.get("history", {})
    return {
        "checkpoint": str(ckp),
        # the checkpoint is paired with the YAML ITS OWN RUN saved; configs/experiment/
        # has drifted since and rebuilding from it raises shape mismatches
        "config": str(ymls[0]) if ymls else None,
        "params": n_params(ck["model_state_dict"]),
        "epoch": ck.get("epoch"),
        "best_val_loss": ck.get("best_val_loss"),
        "latent_dim": cfg.get("latent_dim"),
        "hidden_dims": cfg.get("hidden_dims"),
        "operator_type": cfg.get("operator_type"),
        "operator_mode": cfg.get("operator_mode"),
        "cond_type": cfg.get("cond_type"),
        "rank": cfg.get("rank"),
        "is_continuous": cfg.get("is_continuous"),
        "spectral": cfg.get("spectral"),
        "grid": [cfg.get("height"), cfg.get("width")],
        "history_keys": sorted(hist)[:12],
        "sha256_12": sha(ckp),
    }


def frame_overlap(a_path: str, b_path: str, var: str = "rho") -> dict:
    """Do two .nc files literally share trajectories?

    Comparing Mach ranges is not enough.  ``gt_interp.nc`` -- the DA test set -- turns out
    to be a bit-identical excerpt of ``val.nc``, the split the KAE's checkpoint selection
    ran on, which a Mach-range check would never have shown.  Each first frame of ``a`` is
    matched against every frame of ``b``; a max|diff| of exactly zero is the same data.
    """
    pa, pb = Path(a_path), Path(b_path)
    if not (pa.exists() and pb.exists()):
        return {"a": a_path, "b": b_path, "present": False}
    with xr.open_dataset(pa) as da_, xr.open_dataset(pb) as db_:
        A = np.asarray(da_[var].values)[:, 0]  # first frame of each sim
        B = np.asarray(db_[var].values)
        best = []
        for i in range(A.shape[0]):
            d = min(
                (float(np.abs(B[j] - A[i]).max(axis=(1, 2)).min()), j)
                for j in range(B.shape[0])
            )
            best.append({"a_sim": i, "b_sim": d[1], "max_abs_diff": d[0]})
    n_ident = sum(1 for r in best if r["max_abs_diff"] == 0.0)
    return {
        "a": a_path,
        "b": b_path,
        "present": True,
        "per_sim": best,
        "n_identical": n_ident,
        "n_sims": len(best),
        "shares_trajectories": n_ident > 0,
    }


def main() -> int:
    specs = {
        "kae": kae_row(),
        "turbpred": {m: turbpred_row(m) for m in MODELS},
        "data": {k: dataset_row(v) for k, v in DATA.items()},
    }
    te = specs["data"]["test (DA)"].get("mach_range")
    va = specs["data"]["validation (tuning only)"].get("mach_range")
    overlap = None
    if te and va:
        overlap = not (te[1] < va[0] or va[1] < te[0])
    specs["test_discipline"] = {
        "test_regime": "gt_interp.nc",
        "validation_regime": "gt_extrap.nc",
        "test_mach_range": te,
        "validation_mach_range": va,
        "mach_ranges_overlap": overlap,
        "note": (
            "Every learning rate and every guidance strength is chosen on the "
            "validation regime and frozen before the test regime is touched. The two "
            "regimes are different Mach ranges of the same solver; if "
            "`mach_ranges_overlap` is ever true, that separation has been lost."
        ),
    }

    # --- provenance: which SPLIT each DA regime was cut from -----------------
    # A Mach-range comparison is not sufficient. Every DA file is checked frame-by-frame
    # against the three training splits, because a regime that is a slice of a model's own
    # validation split is a selection leak no matter what its Mach number is.
    tr, va, te = (
        "data/acdm/128_tra/train.nc",
        "data/acdm/128_tra/val.nc",
        "data/acdm/128_tra/test.nc",
    )
    prov = {}
    for da_file in ("gt_interp.nc", "gt_extrap.nc", "gt_longer.nc"):
        f = f"data/acdm/128_tra/{da_file}"
        prov[da_file] = {
            "vs_train": frame_overlap(f, tr),
            "vs_val": frame_overlap(f, va),
            "vs_test": frame_overlap(f, te),
        }
    specs["provenance"] = prov
    leaks = {
        k: [s_ for s_, v in d.items() if v.get("shares_trajectories")]
        for k, d in prov.items()
    }
    specs["test_discipline"]["da_regime_provenance"] = leaks
    specs["test_discipline"]["training_mach_values"] = specs["data"][
        "turbpred training set"
    ].get("mach_values")
    specs["test_discipline"]["known_asymmetry"] = (
        "gt_interp.nc -- the DA TEST regime -- is a bit-identical excerpt of val.nc, which "
        "is the split the KAE's own checkpoint selection (best_val_loss) ran on. No weight "
        "is fitted to it and it is outside every model's TRAINING Mach range "
        "(0.53-0.63 and 0.69-0.89), but the KAE's choice of epoch saw these trajectories "
        "and the turbpred checkpoints' selection did not. gt_longer.nc (Mach 0.64-0.65) is "
        "cut from test.nc instead and is disjoint from both train.nc and val.nc, so it is "
        "clean for every model; the `leakage_control` section re-runs the headline there."
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "training_specs.json").write_text(json.dumps(specs, indent=2, default=str))
    print(json.dumps(specs, indent=2, default=str)[:4000])
    print(f"\nsaved -> {OUT / 'training_specs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
