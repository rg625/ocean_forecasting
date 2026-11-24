"""Train the CT-KAE candidates on 1-D Kuramoto-Sivashinsky (models/ct_kae_ks.py).

    python train_ctkae_ks.py --config=configs/ctkae_ks/ref.yaml [key=value ...]

The objective, curriculum, optimiser and schedule are train_ctkae.py's (step_backward,
horizon_at); data: data/ks/train.nc (80 simulations, dt 0.1), one dummy zero condition.
As for the TRA candidates the final checkpoint is used (last.pth); val.nc rollouts of 50 and
400 frames are logged, never used for selection. Test files are never read here.
"""

import json
import math
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from models.ct_kae_ks import KSWindows, build_ks_model  # noqa: E402
from models.utils import git_state  # noqa: E402
from train_ctkae import DEFAULTS, horizon_at, step_backward  # noqa: E402

KS_DEFAULTS = OmegaConf.merge(
    DEFAULTS,
    {
        "run_name": "ks_ctkae",
        "out_root": "model_outputs_ctkae_ks",
        "data": {"train_file": "data/ks/train.nc", "val_file": "data/ks/val.nc"},
        "train": {"strides": [1], "channel_weighting": "uniform", "select": "last"},
    },
)


@torch.no_grad()
def validate(model, val, n_frames):
    model.eval()
    out = {}
    for K, starts in ((50, [100, 300, 500, 700, 900]), (400, [100, 500])):
        win, tgt, taus, ma = val.fixed_windows(n_frames, K, 1, starts)
        zs = model.forecast(model.encode(win, val.mask, ma), taus, ma)
        pred = model.decode(zs, val.mask, ma).float()
        e = ((pred - tgt) ** 2).mean((-1, -2, -3))  # [N, K], normalised units
        out[f"val_mse{K}"] = float(e.mean())
        out[f"val_rel{K}"] = float(
            (((pred - tgt) ** 2).sum((-1, -2, -3)) / (tgt**2).sum((-1, -2, -3)))
            .sqrt()
            .mean()
        )
    model.train()
    return out


def main():
    cli = OmegaConf.from_cli([a for a in sys.argv[1:] if not a.startswith("--config")])
    cfg_path = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--config=")]
    cfg = OmegaConf.merge(
        KS_DEFAULTS, OmegaConf.load(cfg_path[0]) if cfg_path else {}, cli
    )
    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda")
    tc = cfg.train
    run = (
        Path(cfg.out_root)
        / f"{cfg.run_name}_s{cfg.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run.mkdir(parents=True, exist_ok=False)
    train = KSWindows(cfg.data.train_file, dev)
    val = KSWindows(cfg.data.val_file, dev)
    model = build_ks_model(OmegaConf.to_container(cfg.model, resolve=True)).to(dev)
    n_frames = model.cfg.n_frames
    meta = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "git": git_state(Path(__file__).parent),
        "train": {"sims": train.S, "frames": train.T},
        "val": {"sims": val.S, "frames": val.T},
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    (run / "run_metadata.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("train", "val", "n_params")}), flush=True)
    decay: list = []
    no_decay: list = []
    for n, p in model.named_parameters():
        (
            no_decay if (p.ndim < 2 or n.startswith("gen.") or "mu" in n) else decay
        ).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": tc.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=tc.lr,
        betas=(0.9, 0.99),
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / tc.warmup)
        * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / tc.steps)))),
    )
    w_ch = torch.ones(1, device=dev)
    g = torch.Generator().manual_seed(cfg.seed)
    log = open(run / "log.jsonl", "a")
    t_start = time.time()
    acc: dict = {}
    for step in range(1, tc.steps + 1):
        K = horizon_at(step, tc)
        win, tgt, taus, ma, _ = train.sample(tc.batch, n_frames, K, list(tc.strides), g)
        n_dec = min(tc.n_dec, K)
        rest = (
            torch.randperm(K - 2, generator=g)[: max(0, n_dec - 2)] + 1
            if K > 2
            else torch.tensor([], dtype=torch.long)
        )
        idx = torch.unique(torch.cat([torch.tensor([0, K - 1]), rest])).to(dev)
        opt.zero_grad(set_to_none=True)
        parts = step_backward(
            model, win, tgt, taus, ma, train.mask, idx, K, n_frames, tc, w_ch, tc.accum
        )
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        sched.step()
        for k, v in (*parts.items(), ("gn", gn)):
            acc[k] = acc.get(k, 0.0) + float(v)
        if step % tc.log_every == 0:
            rec = {
                "step": step,
                "K": K,
                "lr": sched.get_last_lr()[0],
                "sec": round(time.time() - t_start, 1),
                **{k: v / tc.log_every for k, v in acc.items()},
            }
            acc = {}
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % tc.val_every == 0 or step == tc.steps:
            v = validate(model, val, n_frames)
            v.update(step=step, sec=round(time.time() - t_start, 1))
            log.write(json.dumps({"val": v}) + "\n")
            log.flush()
            print(json.dumps(v), flush=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": model.config_dict(),
                    "step": step,
                    "val": v,
                    "run_metadata": meta,
                },
                run / "last.pth",
            )
    (run / "DONE").write_text(json.dumps({"seconds": time.time() - t_start}))
    print("run dir:", run)


if __name__ == "__main__":
    main()
