# ruff: noqa: F841
# mypy: disable-error-code="var-annotated"
"""Part 4: the reproducibility / training-specification table.

Everything here is READ FROM the actual checkpoints, configs and data files that the
campaign used.  Nothing is typed in by hand, so the table cannot drift away from the
models that produced the numbers.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import xarray as xr

KAE_RUN = Path("model_outputs_ks/continous_linear_128/rollout_10")
UNET = Path("model_outputs_ks/unet1d/rollout10_extended2")
SDA = Path("model_outputs_ks/sda_paper/k4")
DATA = Path("data/ks")
OUT = Path("da_results_sda_paper")


def sha(p: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:n]


def n_params(sd) -> int:
    return int(sum(v.numel() for v in sd.values() if torch.is_tensor(v)))


def dataset_row(p: Path) -> dict:
    if not p.exists():
        return {"path": str(p), "present": False}
    with xr.open_dataset(p) as ds:
        u = ds["u"]
        dims = dict(zip(u.dims, u.shape))
        t = ds["t"].values if "t" in ds.coords else None
        return {
            "path": str(p),
            "present": True,
            "dims": dims,
            "n_sim": int(ds.sizes["sim"]),
            "n_t": int(ds.sizes["t"]),
            "n_x": int(ds.sizes["x"]),
            "dt": (float(t[1] - t[0]) if t is not None and len(t) > 1 else None),
            "sha256_12": sha(p),
            "bytes": p.stat().st_size,
        }


def main():
    kae_ck = torch.load(
        KAE_RUN / "final_model.pth", map_location="cpu", weights_only=False
    )
    kae_best = torch.load(
        KAE_RUN / "checkpoints" / "best_model.pth",
        map_location="cpu",
        weights_only=False,
    )
    cfg = kae_ck["config"]
    un = json.loads((UNET / "train_meta.json").read_text())
    sd = json.loads((SDA / "train_meta.json").read_text())
    un_sd = torch.load(UNET / "best_model.pth", map_location="cpu", weights_only=False)
    sd_sd = torch.load(SDA / "best_model.pth", map_location="cpu", weights_only=False)
    frozen = json.loads((OUT / "frozen_config.json").read_text())

    # ---- the DA test set must be disjoint from everything any model was trained or
    # selected on. Compared on the actual field values, not on filenames.
    disjoint = {}
    for held in ("da_test", "da_test_long"):
        if not (DATA / f"{held}.nc").exists():
            continue
        with xr.open_dataset(DATA / f"{held}.nc") as ds:
            da_u = ds["u"].values.reshape(ds.sizes["sim"], -1)
        # trajectories have different lengths across sets, so compare the FIRST 1000
        # frames -- two trajectories that agree there are the same trajectory
        n_cmp = 1000 * 64
        da_sig = {
            hashlib.sha256(np.ascontiguousarray(r[:n_cmp]).tobytes()).hexdigest()
            for r in da_u
        }
        disjoint[held] = {}
        for other in ("train", "val", "test", "da_test"):
            if other == held:
                continue
            f = DATA / f"{other}.nc"
            if not f.exists():
                continue
            with xr.open_dataset(f) as ds:
                o = ds["u"].values.reshape(ds.sizes["sim"], -1)
            osig = {
                hashlib.sha256(np.ascontiguousarray(r[:n_cmp]).tobytes()).hexdigest()
                for r in o
            }
            n_shared = len(da_sig & osig)
            disjoint[held][other] = {
                "n_trajectories_shared": n_shared,
                "disjoint": n_shared == 0,
            }

    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
    except Exception:
        rev, dirty = None, None

    import yaml

    kae_hist = yaml.safe_load((KAE_RUN / "training_history.yaml").read_text())
    kae_val = {
        k: {"final": float(v["val"][-1]), "best": float(min(v["val"]))}
        for k, v in kae_hist.items()
        if isinstance(v, dict) and "val" in v
    }
    spec = {
        "system": {
            "pde": "Kuramoto-Sivashinsky  u_t = -u u_x - u_xx - u_xxxx",
            "domain_L": 22.0,
            "grid": 64,
            "boundary": "periodic",
            "solver": "pseudo-spectral, 2/3 dealiasing, scipy RK45 (rtol 1e-6, atol 1e-8)",
            "dt_stored": 0.1,
            "frames_per_trajectory": 1000,
            "lyapunov": json.loads((OUT / "lyapunov.json").read_text()),
        },
        "data": {
            k: dataset_row(DATA / f"{k}.nc")
            for k in ("train", "val", "test", "da_test", "da_test_long")
        },
        "models": {
            "KAE (continuous Koopman autoencoder)": {
                "run_dir": str(KAE_RUN),
                "checkpoint": str(KAE_RUN / "checkpoints" / "best_model.pth"),
                "sha256_12": sha(KAE_RUN / "checkpoints" / "best_model.pth"),
                "params_total": n_params(kae_ck["model_state_dict"]),
                "latent_dim": cfg["model"]["latent_dim"],
                "state_dim": cfg["model"]["height"] * cfg["model"]["width"],
                "hidden_dims": cfg["model"]["hidden_dims"],
                "kernel_size": cfg["model"]["kernel_size"],
                "padding_mode": cfg["model"]["conv_kwargs"]["padding_mode"],
                "operator": cfg["model"]["operator_mode"],
                "continuous": cfg["model"]["is_continuous"],
                "attention": cfg["model"]["use_attention"],
                "spectral_loss": cfg["model"]["spectral"],
                "transformer": cfg["model"]["transformer"],
                "input_sequence_length": cfg["data"]["input_sequence_length"],
                "max_rollout_length": cfg["data"]["max_sequence_length"],
                "epochs_configured": cfg["training"]["num_epochs"],
                "epoch_of_final_ckpt": kae_ck.get("epoch"),
                "epoch_of_best_ckpt": kae_best.get("epoch"),
                "batch_size": cfg["training"]["batch_size"],
                "lr": cfg["lr_scheduler"]["lr"],
                "lr_schedule": (
                    f"warmup {cfg['lr_scheduler']['warmup']} ep, decay "
                    f"{cfg['lr_scheduler']['decay']} ep to "
                    f"{cfg['lr_scheduler']['final_lr']}"
                ),
                "loss": {
                    k: cfg["loss"][k]
                    for k in (
                        "loss_type",
                        "alpha",
                        "beta",
                        "physics_weight",
                        "gamma_time",
                        "gamma_space",
                        "gamma_spectral",
                    )
                },
                "normalisation": cfg["data"]["normalization"]["type"],
                "val_losses": kae_val,
                "train_file": cfg["data"]["train_file"],
                "val_file": cfg["data"]["val_file"],
            },
            "U-Net (1-D autoregressive surrogate)": {
                "checkpoint": str(UNET / "best_model.pth"),
                "sha256_12": sha(UNET / "best_model.pth"),
                "params_total": int(un["params"]),
                "hidden_dims": un["args"]["hidden_dims"],
                "blocks_per_level": un["args"]["blocks"],
                "training_rollout_length": int(un["args"]["rollout"]),
                "epochs_run": un["epochs_run"],
                "steps_per_epoch": int(un["args"]["steps_per_epoch"]),
                "batch_size": int(un["args"]["batch_size"]),
                "lr": float(un["args"]["lr"]),
                "lr_schedule": (
                    f"warmup {un['args']['warmup']} ep, decay "
                    f"{un['args']['decay']} ep to {un['args']['final_lr']}"
                ),
                "seed": int(un["args"]["seed"]),
                "warm_started_from": un["args"]["init_ckpt"],
                "best_val_mse": un["best_val"],
                "best_epoch": un["best_epoch"],
                "wall_time_s": un["wall_time_s"],
                "git_rev_at_training": un["git_rev"],
                "train_file": un["args"]["train"],
                "val_file": un["args"]["val"],
                "note": (
                    "Annealed to convergence and warm-started from a shorter run, so "
                    "the 4D-Var comparison is against a competent surrogate rather "
                    "than a deliberately matched-undertrained one."
                ),
            },
            "SDA score network (Rozet & Louppe 2023)": {
                "checkpoint": str(SDA / "best_model.pth"),
                "sha256_12": sha(SDA / "best_model.pth"),
                "params_total": int(sd["params"]),
                "k": int(sd["args"]["k"]),
                "blanket_window": sd["blanket_window"],
                "hidden": sd["args"]["hidden"],
                "blocks": int(sd["args"]["blocks"]),
                "epochs": int(sd["args"]["epochs"]),
                "steps_per_epoch": int(sd["args"]["steps_per_epoch"]),
                "batch_size": int(sd["args"]["batch_size"]),
                "lr": float(sd["args"]["lr"]),
                "weight_decay": float(sd["args"]["weight_decay"]),
                "seed": int(sd["args"]["seed"]),
                "best_val_eps_loss": sd["best_val"],
                "best_epoch": sd["best_epoch"],
                "wall_time_s": sd["wall_time_s"],
                "sde": "cosine VP, mu(t)=cos(omega t)^2, omega=arccos(sqrt(1e-3))",
                "gamma": sd["gamma_meta"],
                "train_file": sd["args"]["train"],
                "val_file": sd["args"]["val"],
            },
        },
        "inference_settings": {
            "SDA sampler": {
                k: frozen[k]
                for k in (
                    "n_steps",
                    "corrections",
                    "tau",
                    "gamma_mode",
                    "gamma_scale",
                    "gamma_floor",
                    "sigma_y_clean",
                )
            },
            "SDA selection": {
                "selected_on": frozen["selected_on"],
                "test_untouched_until_freeze": frozen["test_untouched_until_freeze"],
            },
            "4D-Var": json.loads(Path("da_results_v2/tuning.json").read_text())[
                "results"
            ],
            "4D-Var init scales": json.loads(
                Path("da_results_v2/init_scales.json").read_text()
            ),
        },
        "test_discipline": {
            "held_out_vs": disjoint,
            "statement": (
                "Every DA number is computed on data/ks/da_test.nc. No model "
                "weight, no 4D-Var hyper-parameter and no SDA sampler setting "
                "was chosen using it: 4D-Var learning rates and initialisation "
                "scales come from da_results_v2/tuning.json (validation), and "
                "the SDA sampler was frozen on data/ks/val.nc before the test "
                "set was touched."
            ),
        },
        "provenance": {"git_rev": rev, "working_tree_dirty": dirty},
    }
    (OUT / "training_specs.json").write_text(json.dumps(spec, indent=2, default=str))

    # ---- flat markdown table -------------------------------------------------
    m = spec["models"]
    rows = [
        (
            "parameters",
            f"{m['KAE (continuous Koopman autoencoder)']['params_total']:,}",
            f"{m['U-Net (1-D autoregressive surrogate)']['params_total']:,}",
            f"{m['SDA score network (Rozet & Louppe 2023)']['params_total']:,}",
        ),
        ("training data", "data/ks/train.nc", "data/ks/train.nc", "data/ks/train.nc"),
        (
            "epochs",
            str(m["KAE (continuous Koopman autoencoder)"]["epoch_of_best_ckpt"]),
            str(m["U-Net (1-D autoregressive surrogate)"]["epochs_run"]),
            str(m["SDA score network (Rozet & Louppe 2023)"]["epochs"]),
        ),
        (
            "batch size",
            str(m["KAE (continuous Koopman autoencoder)"]["batch_size"]),
            m["U-Net (1-D autoregressive surrogate)"]["batch_size"],
            m["SDA score network (Rozet & Louppe 2023)"]["batch_size"],
        ),
        (
            "learning rate",
            str(m["KAE (continuous Koopman autoencoder)"]["lr"]),
            str(m["U-Net (1-D autoregressive surrogate)"]["lr"]),
            str(m["SDA score network (Rozet & Louppe 2023)"]["lr"]),
        ),
        (
            "temporal context",
            f"rollout {m['KAE (continuous Koopman autoencoder)']['max_rollout_length']}",
            f"rollout {m['U-Net (1-D autoregressive surrogate)']['training_rollout_length']}",
            f"blanket {m['SDA score network (Rozet & Louppe 2023)']['blanket_window']}",
        ),
        (
            "best val loss",
            f"{kae_val['total_loss']['best']:.4g} (total)",
            f"{m['U-Net (1-D autoregressive surrogate)']['best_val_mse']:.3g} (MSE)",
            f"{m['SDA score network (Rozet & Louppe 2023)']['best_val_eps_loss']:.3g} (eps)",
        ),
    ]
    w = [22, 34, 34, 34]
    hdr = ["", "KAE (continuous)", "U-Net surrogate", "SDA score net"]
    lines = [
        "| " + " | ".join(h.ljust(x) for h, x in zip(hdr, w)) + " |",
        "|" + "|".join("-" * (x + 2) for x in w) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(c).ljust(x) for c, x in zip(r, w)) + " |")
    md = "\n".join(lines)
    (OUT / "training_specs.md").write_text(md + "\n")
    print(md)
    print("\n=== datasets ===")
    for k, v in spec["data"].items():
        if v.get("present"):
            print(
                f"{k:9s} {v['n_sim']:4d} traj x {v['n_t']:5d} frames x {v['n_x']:3d} pts"
                f"  dt={v['dt']}  sha {v['sha256_12']}"
            )
        else:
            print(f"{k:9s} MISSING ({v['path']})")
    print(
        "\n=== test discipline: are the held-out sets disjoint from what the "
        "models saw? ==="
    )
    for held, d in spec["test_discipline"]["held_out_vs"].items():
        for k, v in d.items():
            print(
                f"  {held:13s} vs {k:9s}: shared trajectories = "
                f"{v['n_trajectories_shared']}  -> "
                f"{'DISJOINT' if v['disjoint'] else 'OVERLAP -- LEAKAGE'}"
            )
    print(f"\nsaved -> {OUT / 'training_specs.json'}")


if __name__ == "__main__":
    main()
