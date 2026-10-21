"""Train the conditional continuous-time KAE (models/ct_kae.py) on TRA.

    python train_ctkae.py --config configs/ctkae/base.yaml [key=value ...]

Objective (all distances channel-weighted, see ct_data.channel_weights):
    L = w_rec  * |D(E(x_t)) - x_t|^2
      + w_pred * mean_{k in S} |D(z_k) - x_{t+k}|^2          z_k exact: mu + e^{tau_k A}(z_0 - mu)
      + w_lat  * mean_{k in S} |z_k - sg(E(x_{t+k-n+1..t+k}))|^2 / E|sg(E)|^2
      + w_amp  * mean_{k in S} (|y_k|_block - |sg(y_k^enc)|_block)^2   (block generator only)
With a detail latent (models/ct_kae.DetailLatent) the w_lat term is computed for the
oscillators and the detail state separately, each normalised by its own E|sg(E)|^2, and
summed; w_amp applies to the oscillators only.
S: random subset of leads (always containing 1 and K); K grows from k_start to k_max.
Strides are drawn from ``strides`` (raw frames 0.05 apart), so the generator is trained
at several physical time steps.

Validation: held-out Mach numbers (default 0.60, 0.75; removed from training), stride 2,
rollouts of 60 and 240 steps; MSE in physical units on the predicted frames. Selection on
val_mse60 + val_mse240 / 4 (configurable). Test files are never read here.
"""

import json
import math
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from models.ct_data import (
    TraWindows,
    channel_weights,
    denormalise,
    describe,
    train_val_intervals,
)  # noqa: E402
from models.ct_kae import CTKAE, CTKAEConfig  # noqa: E402
from models.utils import git_state  # noqa: E402

DEFAULTS = OmegaConf.create(
    {
        "run_name": "ctkae",
        "out_root": "model_outputs_ctkae",
        "seed": 0,
        # val_ma: [] trains on every simulation with no validation split and keeps the
        # final checkpoint (the ACDM protocol); extra_raw_sims appends simulations read
        # from the official ACDM export (e.g. sim 40, Ma 0.90, absent from train.nc).
        "data": {
            "train_file": "data/acdm/128_tra/train.nc",
            "val_ma": [0.60, 0.75],
            "extra_raw_sims": [],
        },
        "model": {},
        "train": {
            "steps": 20000,
            "batch": 16,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "warmup": 500,
            "grad_clip": 1.0,
            "k_start": 10,
            "k_max": 60,
            "curriculum_frac": 0.5,
            "k_hold_frac": 0.0,
            "n_dec": 8,
            "strides": [2],
            "amp": True,
            "channel_weighting": "phys",
            "w_rec": 1.0,
            "w_pred": 1.0,
            "w_lat": 0.1,
            "w_amp": 0.0,
            "val_every": 1000,
            "log_every": 100,
            "select": "val_mse60+0.25*val_mse240",
        },
    }
)


def build_model(mcfg) -> CTKAE:
    return CTKAE(CTKAEConfig(**OmegaConf.to_container(mcfg, resolve=True)))


def horizon_at(step: int, tc) -> int:
    """Training horizon K at a given step: held at k_start for the first k_hold_frac of
    training (e.g. one-step pre-training of a sharp autoencoder), then ramped linearly to
    k_max over the next curriculum_frac of training."""
    hold = tc.get("k_hold_frac", 0.0) * tc.steps
    frac = min(1.0, max(0.0, step - hold) / max(1.0, tc.curriculum_frac * tc.steps))
    return int(round(tc.k_start + (tc.k_max - tc.k_start) * frac))


def weighted_mse(a, b, w):
    return (((a - b) ** 2).mean(dim=(-1, -2)) * w).mean()


def amp_loss(y_pred, y_tgt):
    rp = y_pred.view(*y_pred.shape[:-1], -1, 2).norm(dim=-1)
    rt = y_tgt.view(*y_tgt.shape[:-1], -1, 2).norm(dim=-1)
    return ((rp - rt) ** 2).mean() / (rt.pow(2).mean() + 1e-6)


@torch.no_grad()
def validate(model, val, n_frames):
    model.eval()
    out = {}
    for K, starts in ((60, [100, 300, 500, 700, 860]), (240, [100, 300, 500])):
        win, tgt, taus, ma = val.fixed_windows(n_frames, K, 2, starts)
        z0 = model.encode(win, val.mask, ma)
        zs = model.propagate(z0, taus, ma)
        err, fe_p, fe_t = [], [], []
        for b in range(win.shape[0]):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                pred = model.decode(zs[b], val.mask, ma[b : b + 1].expand(K)).float()
            pp, tt = denormalise(pred), denormalise(tgt[b])
            err.append(((pp - tt) ** 2).mean(dim=(-1, -2)).mean(-1))  # [K]
            om = tgt[b].mean(0, keepdim=True)
            fe_p.append(((pred - om) ** 2).mean())
            fe_t.append(((tgt[b] - om) ** 2).mean())
        e = torch.stack(err)  # [N, K]
        out[f"val_mse{K}"] = float(e.mean() * 1e3)
        out[f"val_mse{K}_first10"] = float(e[:, :10].mean() * 1e3)
        out[f"val_energy_ratio{K}"] = float(
            torch.stack(fe_p).sum() / torch.stack(fe_t).sum()
        )
    model.train()
    return out


def main():
    cli = OmegaConf.from_cli([a for a in sys.argv[1:] if not a.startswith("--config")])
    cfg_path = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--config=")]
    cfg = OmegaConf.merge(
        DEFAULTS, OmegaConf.load(cfg_path[0]) if cfg_path else {}, cli
    )
    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda")
    tc = cfg.train
    run = (
        Path(cfg.out_root)
        / f"{cfg.run_name}_s{cfg.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run.mkdir(parents=True, exist_ok=False)

    extra = list(cfg.data.get("extra_raw_sims", []) or [])
    if cfg.data.val_ma:
        tr_int, va_int = train_val_intervals(cfg.data.val_ma)
        train = TraWindows(cfg.data.train_file, tr_int, dev, extra_raw_sims=extra)
        val = TraWindows(cfg.data.train_file, va_int, dev)
    else:
        train = TraWindows(cfg.data.train_file, None, dev, extra_raw_sims=extra)
        val = None
    model = build_model(cfg.model).to(dev)
    n_frames = model.cfg.n_frames
    meta = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "git": git_state(Path(__file__).parent),
        "train": describe(train),
        "val": describe(val) if val is not None else None,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    (run / "run_metadata.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("train", "val", "n_params")}))

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
    w_ch = channel_weights(tc.channel_weighting).to(dev)
    g = torch.Generator().manual_seed(cfg.seed)
    log = open(run / "log.jsonl", "a")
    best, t_start = float("inf"), time.time()
    acc: dict = {}
    for step in range(1, tc.steps + 1):
        K = horizon_at(step, tc)
        win, tgt, taus, ma, s = train.sample(tc.batch, n_frames, K, list(tc.strides), g)
        n_dec = min(tc.n_dec, K)
        rest = (
            torch.randperm(K - 2, generator=g)[: max(0, n_dec - 2)] + 1
            if K > 2
            else torch.tensor([], dtype=torch.long)
        )
        idx = torch.unique(torch.cat([torch.tensor([0, K - 1]), rest])).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tc.amp):
            z0 = model.encode(win, train.mask, ma)
            zs = model.propagate(z0.float(), taus, ma)[:, idx]  # [B, n, d]
            B, nd = zs.shape[:2]
            ma_rep = ma.repeat_interleave(nd)
            x_hat = model.decode(zs, train.mask, ma_rep).float()
            x_rec = model.decode(z0, train.mask, ma).float()
            l_pred = weighted_mse(x_hat, tgt[:, idx], w_ch)
            l_rec = weighted_mse(x_rec, win[:, -1], w_ch)
            l_lat = torch.zeros((), device=dev)
            l_amp = torch.zeros((), device=dev)
            if tc.w_lat > 0 or tc.w_amp > 0:
                full = torch.cat([win, tgt], 1)
                wins = torch.stack(
                    [full[:, int(k) + 1 : int(k) + 1 + n_frames] for k in idx], 1
                )
                with torch.no_grad():
                    zt = (
                        model.encode(wins.flatten(0, 1), train.mask, ma_rep)
                        .float()
                        .view(B, nd, -1)
                    )
                # normalise by the second moment E|z|^2 (scale-invariant); the variance
                # across samples can be tiny when latents share a common component, which
                # made this term explode (~1e3) and dominate the clipped gradient
                # oscillators and (if present) the detail latent are normalised
                # separately, so the much larger detail state cannot swamp the term
                sd = model.slow_dim
                l_lat = ((zs[..., :sd] - zt[..., :sd]) ** 2).mean() / (
                    zt[..., :sd].pow(2).mean() + 1e-6
                )
                if model.detail is not None:
                    l_lat = l_lat + ((zs[..., sd:] - zt[..., sd:]) ** 2).mean() / (
                        zt[..., sd:].pow(2).mean() + 1e-6
                    )
                if tc.w_amp > 0 and model.cfg.generator == "block":
                    mu = model.fixed_point(model.embed_cond(ma), B)[:, None]
                    l_amp = amp_loss((zs - mu)[..., :sd], (zt - mu)[..., :sd].detach())
            loss = (
                tc.w_rec * l_rec
                + tc.w_pred * l_pred
                + tc.w_lat * l_lat
                + tc.w_amp * l_amp
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        sched.step()
        for k, v in (
            ("loss", loss),
            ("rec", l_rec),
            ("pred", l_pred),
            ("lat", l_lat),
            ("amp", l_amp),
            ("gn", gn),
        ):
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
            v = validate(model, val, n_frames) if val is not None else {}
            # without a validation split the final checkpoint is used (best == last)
            score = eval(tc.select, {}, v) if val is not None else -float(step)
            v.update(
                step=step,
                score=score,
                sec=round(time.time() - t_start, 1),
                peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
            )
            log.write(json.dumps({"val": v}) + "\n")
            log.flush()
            print(json.dumps(v))
            state = {
                "model": model.state_dict(),
                "model_config": model.config_dict(),
                "step": step,
                "val": v,
                "run_metadata": meta,
            }
            torch.save(state, run / "last.pth")
            if score < best:
                best = score
                torch.save(state, run / "best.pth")
    (run / "DONE").write_text(
        json.dumps({"best_score": best, "seconds": time.time() - t_start})
    )
    print("run dir:", run)


if __name__ == "__main__":
    main()
