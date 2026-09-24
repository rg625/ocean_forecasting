"""Train the 1-D U-Net autoregressive forecaster on KS.

Protocol deliberately mirrors the Koopman autoencoder's KS run so that the DA
comparison is not confounded by training budget:

  * same data      : ``data/ks/train.nc`` (80 sims), validated on ``data/ks/val.nc``
  * same cadence   : one network application == one data frame == dt = 0.1
  * same objective : L2 on the normalised field
  * same curriculum: rollout of up to ``--rollout`` frames (KAE used ``rollout_10``)
  * same optimiser : Adam, lr 1e-3, linear warmup(20)/decay(180) over 200 epochs
  * same batch     : 1024 sequences

Only the *initial frame* is fed to the network; the rest of the window is produced
autoregressively and every predicted frame is supervised, which is the standard
pushforward-style curriculum for neural PDE surrogates.

Usage:
    python data_assimilation/ks/train_unet_ks.py --out-dir model_outputs_ks/unet1d/rollout_10
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from data_assimilation.ks.unet_ks import UNet1d, UNet1dConfig


def _git_rev() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def load_norm(path: Path, device) -> torch.Tensor:
    """[sim, t, X] normalised with the deterministic KS constants used by the KAE."""
    from models.dataloader import KS_MEAN, KS_STD

    mean = (
        float(KS_MEAN["u"])
        if not torch.is_tensor(KS_MEAN["u"])
        else float(KS_MEAN["u"].item())
    )
    std = (
        float(KS_STD["u"])
        if not torch.is_tensor(KS_STD["u"])
        else float(KS_STD["u"].item())
    )
    ds = xr.open_dataset(path)
    u = torch.from_numpy(ds["u"].values).float().squeeze(-1)  # [sim, t, X]
    ds.close()
    return ((u - mean) / (std + 1e-8)).to(device)


def make_windows(u: torch.Tensor, length: int) -> torch.Tensor:
    """All contiguous windows of `length` frames -> [N, length, X]."""
    n_sim, n_t, X = u.shape
    idx = (
        torch.arange(n_t - length + 1, device=u.device)[:, None]
        + torch.arange(length, device=u.device)[None, :]
    )
    return u[:, idx].reshape(-1, length, X)


def lr_at(epoch: int, lr: float, warmup: int, decay: int, final_lr: float) -> float:
    if epoch < warmup:
        return lr * (epoch + 1) / warmup
    frac = min(1.0, (epoch - warmup) / max(1, decay))
    return lr + (final_lr - lr) * frac


@torch.no_grad()
def evaluate(model, u_val: torch.Tensor, rollout: int, bs: int = 2048) -> float:
    """Mean L2 rollout loss on the validation split."""
    win = make_windows(u_val, rollout + 1)
    tot, n = 0.0, 0
    for i in range(0, win.shape[0], bs):
        w = win[i : i + bs]
        x = w[:, 0].unsqueeze(1)
        loss = 0.0
        for k in range(1, rollout + 1):
            x = model(x)
            loss = loss + F.mse_loss(x.squeeze(1), w[:, k])
        tot += float(loss) / rollout * w.shape[0]
        n += w.shape[0]
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/ks/train.nc"))
    ap.add_argument("--val", type=Path, default=Path("data/ks/val.nc"))
    ap.add_argument(
        "--out-dir", type=Path, default=Path("model_outputs_ks/unet1d/rollout_10")
    )
    ap.add_argument("--hidden-dims", type=str, default="64,128,256")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--rollout", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--decay", type=int, default=180)
    ap.add_argument("--final-lr", type=float, default=2e-5)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument(
        "--patience",
        type=int,
        default=0,
        help="stop after this many epochs without validation improvement "
        "(0 disables early stopping)",
    )
    ap.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="relative improvement required to reset the patience counter",
    )
    ap.add_argument(
        "--init-ckpt",
        type=Path,
        default=None,
        help="warm-start from this checkpoint (used for the "
        "convergence-extension control)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    u_tr = load_norm(args.train, dev)
    u_va = load_norm(args.val, dev)
    print(f"train {tuple(u_tr.shape)}  val {tuple(u_va.shape)}", flush=True)

    win_tr = make_windows(u_tr, args.rollout + 1)  # [N, R+1, X]
    print(f"training windows: {win_tr.shape[0]}", flush=True)

    dims = tuple(int(d) for d in args.hidden_dims.split(","))
    model = UNet1d(UNet1dConfig(hidden_dims=dims, blocks_per_level=args.blocks)).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(
        f"UNet1d hidden_dims={dims} blocks={args.blocks} params={n_par:,}", flush=True
    )

    if args.init_ckpt is not None:
        st = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model_state_dict"], strict=True)
        print(
            f"warm-started from {args.init_ckpt} "
            f"(epoch {st.get('epoch')}, val {st.get('val_loss'):.5e})",
            flush=True,
        )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator(device=dev).manual_seed(args.seed)

    best, best_ep, hist = float("inf"), -1, []
    t_start = time.time()
    for ep in range(args.epochs):
        lr = lr_at(ep, args.lr, args.warmup, args.decay, args.final_lr)
        for gparam in opt.param_groups:
            gparam["lr"] = lr
        model.train()
        run = 0.0
        for _ in range(args.steps_per_epoch):
            idx = torch.randint(
                0, win_tr.shape[0], (args.batch_size,), device=dev, generator=g
            )
            w = win_tr[idx]
            x = w[:, 0].unsqueeze(1)
            loss = x.new_zeros(())
            for k in range(1, args.rollout + 1):
                x = model(x)
                loss = loss + F.mse_loss(x.squeeze(1), w[:, k])
            loss = loss / args.rollout
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss)
        run /= args.steps_per_epoch

        model.eval()
        val = evaluate(model, u_va, args.rollout)
        hist.append({"epoch": ep, "lr": lr, "train": run, "val": val})
        if val < best * (1.0 - args.min_delta):
            best, best_ep = val, ep
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "hidden_dims": list(dims),
                        "blocks": args.blocks,
                        "residual": True,
                        "dt": 0.1,
                    },
                    "epoch": ep,
                    "val_loss": val,
                },
                args.out_dir / "best_model.pth",
            )
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(
                f"[{ep:03d}] lr={lr:.2e} train={run:.5e} val={val:.5e} "
                f"best={best:.5e}@{best_ep} ({time.time() - t_start:.0f}s)",
                flush=True,
            )
        if args.patience and (ep - best_ep) >= args.patience:
            print(
                f"early stop at epoch {ep}: no validation improvement for "
                f"{args.patience} epochs (best {best:.5e} @ {best_ep})",
                flush=True,
            )
            break

    meta = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "params": n_par,
        "best_val": best,
        "best_epoch": best_ep,
        "epochs_run": ep + 1,
        "git_rev": _git_rev(),
        "wall_time_s": time.time() - t_start,
        "history": hist,
    }
    with open(args.out_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done: best val {best:.5e} @ epoch {best_ep} -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
