"""Train the KS trajectory score model that backs the score-based DA baseline.

The model is a *prior* over windows of ``--window`` consecutive frames drawn from the
training split.  It sees no observations at any point: conditioning happens only at
sampling time through the observation likelihood in :mod:`data_assimilation.ks.sda`.

    python data_assimilation/ks/train_sda.py --window 26 --out-dir model_outputs_ks/sda/win26
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data_assimilation.ks.sda import ScoreBasedDA, ScoreUNet2d, VPSchedule
from data_assimilation.ks.train_unet_ks import load_norm, make_windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/ks/train.nc"))
    ap.add_argument("--val", type=Path, default=Path("data/ks/val.nc"))
    ap.add_argument("--window", type=int, default=26)
    ap.add_argument("--out-dir", type=Path, default=Path("model_outputs_ks/sda/win26"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps-per-epoch", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    u_tr = load_norm(args.train, dev)
    u_va = load_norm(args.val, dev)
    w_tr = make_windows(u_tr, args.window).unsqueeze(1)  # [N, 1, L, X]
    w_va = make_windows(u_va, args.window).unsqueeze(1)
    print(f"train windows {tuple(w_tr.shape)}  val {tuple(w_va.shape)}", flush=True)

    net = ScoreUNet2d().to(dev)
    sched = VPSchedule()
    sda = ScoreBasedDA(net, sched, args.window, dev)
    print(f"score net params {sum(p.numel() for p in net.parameters()):,}", flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    best, best_ep, hist = float("inf"), -1, []
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        run = 0.0
        for _ in range(args.steps_per_epoch):
            idx = torch.randint(
                0, w_tr.shape[0], (args.batch_size,), device=dev, generator=g
            )
            loss = sda.loss(w_tr[idx], gen=g)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run += float(loss)
        run /= args.steps_per_epoch

        net.eval()
        with torch.no_grad():
            gv = torch.Generator(device=dev).manual_seed(1234)
            val = float(
                np.mean(
                    [
                        float(
                            sda.loss(
                                w_va[
                                    torch.randint(
                                        0,
                                        w_va.shape[0],
                                        (args.batch_size,),
                                        device=dev,
                                        generator=gv,
                                    )
                                ],
                                gen=gv,
                            )
                        )
                        for _ in range(20)
                    ]
                )
            )
        hist.append({"epoch": ep, "train": run, "val": val})
        if val < best:
            best, best_ep = val, ep
            torch.save(
                {
                    "model_state_dict": net.state_dict(),
                    "window": args.window,
                    "schedule": vars(sched),
                    "epoch": ep,
                    "val_loss": val,
                },
                args.out_dir / "best_model.pth",
            )
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(
                f"[{ep:03d}] train={run:.5f} val={val:.5f} best={best:.5f}@{best_ep} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    with open(args.out_dir / "train_meta.json", "w") as f:
        json.dump(
            {
                "args": {k: str(v) for k, v in vars(args).items()},
                "params": sum(p.numel() for p in net.parameters()),
                "best_val": best,
                "best_epoch": best_ep,
                "wall_time_s": time.time() - t0,
                "history": hist,
            },
            f,
            indent=2,
        )
    print(f"done: best val {best:.5f} @ {best_ep}", flush=True)


if __name__ == "__main__":
    main()
