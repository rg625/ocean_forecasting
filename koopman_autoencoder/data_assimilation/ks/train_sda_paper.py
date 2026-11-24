# ruff: noqa: F841
"""Train the local trajectory-score network for paper-faithful SDA (Alg. 1).

The network sees only short segments ``x_{i-k:i+k}`` drawn from the TRAINING split, and
never sees an observation: the observation model is introduced at sampling time only. The
Appendix-B matrix Gamma is estimated here, also from training data alone, and stored beside
the checkpoint so that assimilation never recomputes it.

    python data_assimilation/ks/train_sda_paper.py --k 4 --out-dir model_outputs_ks/sda_paper/k4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from data_assimilation.ks.sda_paper import (
    SDA,
    CosineVPSchedule,
    LocalScoreUNet,
    build_gamma_circulant,
)
from data_assimilation.ks.train_unet_ks import load_norm, make_windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/ks/train.nc"))
    ap.add_argument("--val", type=Path, default=Path("data/ks/val.nc"))
    ap.add_argument("--k", type=int, default=4, help="temporal blanket half-width")
    ap.add_argument("--hidden", type=str, default="96,192,384")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=256)
    ap.add_argument("--steps-per-epoch", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = args.out_dir or Path(f"model_outputs_ks/sda_paper/k{args.k}")
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    W = 2 * args.k + 1

    u_tr = load_norm(args.train, dev)  # [sim, t, X]
    u_va = load_norm(args.val, dev)
    seg_tr = make_windows(u_tr, W)  # [N, W, X]
    seg_va = make_windows(u_va, W)
    print(
        f"k={args.k} (blanket {W} frames) | train segments {tuple(seg_tr.shape)} "
        f"| val {tuple(seg_va.shape)}",
        flush=True,
    )

    # ---- Gamma (App. B) from TRAINING states only --------------------------
    flat_states = u_tr.reshape(-1, u_tr.shape[-1])
    Gamma, gmeta = build_gamma_circulant(flat_states)
    torch.save({"Gamma": Gamma.cpu(), "meta": gmeta}, out / "gamma.pt")
    print(
        f"Gamma: eig in [{gmeta['gamma_eig_min']:.4f}, {gmeta['gamma_eig_max']:.4f}] "
        f"from {gmeta['n_states']} training states",
        flush=True,
    )

    dims = tuple(int(d) for d in args.hidden.split(","))
    net = LocalScoreUNet(k=args.k, hidden=dims, blocks=args.blocks).to(dev)
    sched = CosineVPSchedule()
    sda = SDA(net, sched, dev, gamma=Gamma.to(dev))
    n_par = sum(p.numel() for p in net.parameters())
    print(f"local score net: {n_par:,} params", flush=True)

    opt = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sch = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.01, total_iters=args.epochs)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    gv = torch.Generator(device=dev).manual_seed(1234)

    best, best_ep, hist = float("inf"), -1, []
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        run = 0.0
        for _ in range(args.steps_per_epoch):
            idx = torch.randint(
                0, seg_tr.shape[0], (args.batch_size,), device=dev, generator=g
            )
            loss = sda.loss(seg_tr[idx], gen=g)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run += float(loss)
        run /= args.steps_per_epoch
        sch.step()

        net.eval()
        with torch.no_grad():
            gv2 = torch.Generator(device=dev).manual_seed(1234)
            val = float(
                np.mean(
                    [
                        float(
                            sda.loss(
                                seg_va[
                                    torch.randint(
                                        0,
                                        seg_va.shape[0],
                                        (args.batch_size,),
                                        device=dev,
                                        generator=gv2,
                                    )
                                ],
                                gen=gv2,
                            )
                        )
                        for _ in range(24)
                    ]
                )
            )
        hist.append(
            {"epoch": ep, "train": run, "val": val, "lr": opt.param_groups[0]["lr"]}
        )
        if val < best:
            best, best_ep = val, ep
            torch.save(
                {
                    "model_state_dict": net.state_dict(),
                    "k": args.k,
                    "hidden": list(dims),
                    "blocks": args.blocks,
                    "schedule": "cosine_vp",
                    "epoch": ep,
                    "val_loss": val,
                },
                out / "best_model.pth",
            )
        if ep % 16 == 0 or ep == args.epochs - 1:
            print(
                f"[{ep:03d}] train={run:.5f} val={val:.5f} best={best:.5f}@{best_ep} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    with open(out / "train_meta.json", "w") as f:
        json.dump(
            {
                "args": {k: str(v) for k, v in vars(args).items()},
                "params": n_par,
                "blanket_window": W,
                "best_val": best,
                "best_epoch": best_ep,
                "gamma_meta": gmeta,
                "wall_time_s": time.time() - t0,
                "history": hist,
                "note": (
                    "Trained on short segments only (Alg. 1); the observation "
                    "model is never seen during training."
                ),
            },
            f,
            indent=2,
        )
    print(f"done: best val {best:.5f} @ {best_ep} -> {out}", flush=True)


if __name__ == "__main__":
    main()
