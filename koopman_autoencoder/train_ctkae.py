"""Train the conditional continuous-time KAE (models/ct_kae.py) on TRA.

    python train_ctkae.py --config configs/ctkae/base.yaml [key=value ...]

Objective (all distances channel-weighted, see ct_data.channel_weights):
    L = w_rec  * |D(E(x_t)) - x_t|^2
      + w_pred * mean_{k in S} |D(z_k) - x_{t+k}|^2          z_k exact: mu + e^{tau_k A}(z_0 - mu)
      + w_lat  * mean_{k in S} |z_k - sg(E(x_{t+k-n+1..t+k}))|^2 / E|sg(E)|^2
      + w_amp  * mean_{k in S} (|y_k|_block - |sg(y_k^enc)|_block)^2   (block generator only)
      (w_pred term optionally weighted linearly over leads, 1 -> w_lead_ramp, mean 1)
      + w_tc   * multistart_consistency (windows starting at tc_starts inside the clip; off by default)
      + optional terms of the original KAE loss, off by default (see step_backward):
        w_back (backward latent consistency), w_cos (directional cosine), w_energy (latent
        norm smoothness), w_sob_t / w_sob_x (temporal / spatial Sobolev), w_spec (spectral)
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
import torch.nn.functional as F
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
            "w_tc": 0.0,
            "w_lead_ramp": 1.0,
            "pred_weighting": "uniform",  # uniform | cosine (original KAE, see lead_weights)
            # optional terms of the original KAE formulation, off by default (ablations)
            "w_back": 0.0,
            "w_cos": 0.0,
            "w_energy": 0.0,
            "w_sob_t": 0.0,
            "w_sob_x": 0.0,
            "w_spec": 0.0,
            "accum": 1,
            # regularisers (ablations): fine_targets draws the K targets at random raw-grid
            # times (spacing 0.05) within the same horizon (TraWindows.sample); ma_noise adds
            # N(0, s^2) to the training Mach numbers (all conditioning paths); input_noise
            # adds N(0, s^2) to the encoder input window (normalised units; targets clean);
            # latent_noise injects N(0, (s rms z)^2) before every propagation step, as the
            # original KAE's disturb_std (absolute there; relative here, z is unnormalised)
            "fine_targets": False,
            "ma_noise": 0.0,
            "input_noise": 0.0,
            "latent_noise": 0.0,
            "tc_starts": [0, 2, 4],
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


def weighted_mse(a, b, w, lead_w=None):
    """Channel-weighted MSE; ``lead_w`` [n] (mean 1) weights the lead axis of [B, n, C, H, W]."""
    e = ((a - b) ** 2).mean(dim=(-1, -2)) * w
    if lead_w is not None:
        e = e * lead_w[:, None]
    return e.mean()


def lead_weights(idx, K: int, ramp: float, mode: str = "uniform"):
    """Weights of the decoded leads ``idx`` (0-based) in the prediction loss, mean 1; None
    when uniform. ``ramp``: linear from 1 (lead 1) to ``ramp`` (lead K). ``mode="cosine"``:
    the original KAE's weighting 0.5 (1 + cos(pi k / (K - 1))) (models/loss.py), largest at
    lead 1 and zero at lead K."""
    if mode == "cosine":
        k = idx.float()
        lw = (
            0.5 * (1 + torch.cos(math.pi * k / max(K - 1, 1)))
            if K > 1
            else torch.ones_like(k)
        )
        if ramp != 1.0:
            lw = lw * (1.0 + (ramp - 1.0) * k / max(K - 1, 1))
        return lw / lw.mean().clamp_min(1e-8)
    if mode != "uniform":
        raise ValueError(f"unknown pred_weighting {mode}")
    if ramp == 1.0:
        return None
    lw = 1.0 + (ramp - 1.0) * idx.float() / max(K - 1, 1)
    return lw / lw.mean()


def amp_loss(y_pred, y_tgt, den=None):
    rp = y_pred.reshape(*y_pred.shape[:-1], -1, 2).norm(dim=-1)
    rt = y_tgt.reshape(*y_tgt.shape[:-1], -1, 2).norm(dim=-1)
    return ((rp - rt) ** 2).mean() / (rt.pow(2).mean() + 1e-6 if den is None else den)


def multistart_consistency(model, full, z0, taus, ma, mask, n_frames, starts):
    """tcKAE-style temporal consistency (Nayak et al. 2024) inside the training clip.

    Windows starting at frames ``starts`` of ``full`` [B, n + K, ...] are encoded (start 0
    reuses ``z0``) and propagated to every frame after the latest window; pairwise
    disagreement of these latents and their (detached) mean square are returned; the loss is
    their ratio. Both sides carry gradients. Only frames inside the supervised clip are used.
    """
    dtf = taus[:, :1]  # constant frame spacing within a batch
    first = max(starts) + n_frames  # first frame after every window
    common = torch.arange(first, full.shape[1], device=full.device, dtype=dtf.dtype)
    sd = model.slow_dim
    zs = []
    for st in starts:
        z = z0 if st == 0 else model.encode(full[:, st : st + n_frames], mask, ma)
        lead = (common - (st + n_frames - 1))[None] * dtf  # [B, C]
        zs.append(model.propagate(z.float(), lead, ma)[..., :sd])
    num = [
        ((zs[i] - zs[j]) ** 2).mean()
        for i in range(len(zs))
        for j in range(i + 1, len(zs))
    ]
    return torch.stack(num).mean(), torch.stack(zs).pow(2).mean().detach() + 1e-6


def sobolev_time(x_hat_seq, x_seq, w_ch, ok):
    """Temporal Sobolev term: channel-weighted MSE between consecutive-frame differences of
    the forecast and of the truth, over the frame pairs flagged in ``ok`` [n-1]."""
    dh = x_hat_seq[:, 1:] - x_hat_seq[:, :-1]
    dt = x_seq[:, 1:] - x_seq[:, :-1]
    return weighted_mse(dh[:, ok], dt[:, ok], w_ch)


def sobolev_space(e, w_ch):
    """Spatial Sobolev term on the error field e [..., C, H, W]: channel-weighted mean of
    its squared finite differences along both grid directions."""
    gx = (e[..., :, 1:] - e[..., :, :-1]).pow(2).mean(dim=(-1, -2))
    gy = (e[..., 1:, :] - e[..., :-1, :]).pow(2).mean(dim=(-1, -2))
    return ((gx + gy) * w_ch).mean()


def spectral_l1(x_hat, x, w_ch, den=None):
    """Spectral term: channel-weighted L1 distance of the 2-D Fourier amplitudes, divided by
    the channel-weighted mean amplitude of the truth (``den``, computed on the full batch
    by the caller)."""
    fh = torch.fft.rfft2(x_hat.float()).abs()
    ft = torch.fft.rfft2(x.float()).abs()
    num = ((fh - ft).abs().mean(dim=(-1, -2)) * w_ch).mean()
    if den is None:
        den = (ft.mean(dim=(-1, -2)) * w_ch).mean() + 1e-6
    return num / den


LOSS_TERMS = (
    "loss",
    "rec",
    "pred",
    "lat",
    "amp",
    "tc",
    "back",
    "cos",
    "energy",
    "sob_t",
    "sob_x",
    "spec",
)


def step_backward(
    model,
    win,
    tgt,
    taus,
    ma,
    mask,
    idx,
    K,
    n_frames,
    tc,
    w_ch,
    accum=1,
    tgt_hist=None,
    win_enc=None,
):
    """Forward and backward of one training step on a batch split into ``accum`` equal
    chunks, whose gradients add up to the full-batch gradient (less GPU memory, same
    training). Terms that divide by batch statistics (latent consistency, amplitude,
    multi-start, backward consistency, energy, spectral) take those statistics from
    detached quantities computed once on the full batch, so every term is a chunk mean
    with a full-batch denominator and the result does not depend on ``accum``.

    Terms (weights in ``tc``; all but rec / pred / lat / amp are off by default):
      rec    |D(E(window)) - x_t|^2                     pred   |D(z_k) - x_{t+k}|^2
      lat    |z_k - sg E(x_{t+k})|^2 / E|z|^2           amp    oscillator amplitudes
      tc     multi-start consistency                    back   |e^{-tau A} E(last window) - sg E(.)|^2
      cos    1 - cos(z_k, sg E(x_{t+k}))                energy (|z_{k+1}| - |z_k|)^2 / E|z|^2
      sob_t  temporal Sobolev (frame differences)       sob_x  spatial Sobolev (error gradients)
      spec   spectral L1 of Fourier amplitudes
    ``tgt_hist`` [B, K, n-1, ...] gives the encoder history of each target when the targets
    are not equally spaced continuations of the window (fine_targets); ``win_enc`` is a
    perturbed copy of the window fed to the encoder for z0 (input_noise; the reconstruction
    target stays ``win``). ``tc.latent_noise`` > 0 replaces the one-shot propagation by a
    sequential one with noise injected before every step (not chunk-invariant).
    Returns the full-batch loss terms."""
    B = win.shape[0]
    if B % accum:
        raise ValueError(f"batch {B} not divisible by accum {accum}")
    b, nd, sd = B // accum, len(idx), model.slow_dim
    lw = lead_weights(idx, K, tc.w_lead_ramp, getattr(tc, "pred_weighting", "uniform"))
    amp_on = tc.w_amp > 0 and model.cfg.generator == "block"
    w_back = getattr(tc, "w_back", 0.0)
    w_cos = getattr(tc, "w_cos", 0.0)
    w_energy = getattr(tc, "w_energy", 0.0)
    w_sob_t = getattr(tc, "w_sob_t", 0.0)
    w_sob_x = getattr(tc, "w_sob_x", 0.0)
    w_spec = getattr(tc, "w_spec", 0.0)
    lat_noise = getattr(tc, "latent_noise", 0.0)
    if tgt_hist is not None and (
        tc.w_tc > 0 or w_back > 0 or w_energy > 0 or w_sob_t > 0
    ):
        raise ValueError("tc / back / energy / sob_t assume equally spaced targets")
    need_zt = tc.w_lat > 0 or tc.w_amp > 0 or w_back > 0 or w_cos > 0 or w_energy > 0
    zt = den_lat = den_det = den_amp = den_tc = den_en = den_spec = None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=tc.amp):
        if need_zt:
            if tgt_hist is not None:
                wins = torch.cat([tgt_hist[:, idx], tgt[:, idx, None]], 2)
            else:
                full = torch.cat([win, tgt], 1)
                wins = torch.stack(
                    [full[:, int(k) + 1 : int(k) + 1 + n_frames] for k in idx], 1
                )
            # normalise by the second moment E|z|^2 (scale-invariant); the variance across
            # samples can be tiny when latents share a common component, which made this
            # term explode (~1e3) and dominate the clipped gradient. Oscillators and (if
            # present) the detail latent are normalised separately.
            zt = (
                model.encode(wins.flatten(0, 1), mask, ma.repeat_interleave(nd))
                .float()
                .view(B, nd, -1)
            )
            den_lat = zt[..., :sd].pow(2).mean() + 1e-6
            den_en = zt[..., :sd].pow(2).sum(-1).mean() + 1e-6
            if model.detail is not None:
                den_det = zt[..., sd:].pow(2).mean() + 1e-6
            if amp_on:
                mu = model.fixed_point(model.embed_cond(ma), B)[:, None]
                rt = (zt - mu)[..., :sd]
                rt = rt.reshape(*rt.shape[:-1], -1, 2).norm(dim=-1)
                den_amp = rt.pow(2).mean() + 1e-6
        if tc.w_tc > 0:
            _, den_tc = multistart_consistency(
                model,
                torch.cat([win, tgt], 1),
                model.encode(win, mask, ma),
                taus,
                ma,
                mask,
                n_frames,
                list(tc.tc_starts),
            )
    if w_spec > 0:
        ft = torch.fft.rfft2(tgt[:, idx].float()).abs()
        den_spec = (ft.mean(dim=(-1, -2)) * w_ch).mean() + 1e-6
    # frame pairs (window frame t, lead idx[0]+1, ...) that are consecutive in time
    seq_idx = torch.cat([idx.new_tensor([-1]), idx])
    ok = (seq_idx[1:] - seq_idx[:-1]) == 1
    tot = dict.fromkeys(LOSS_TERMS, 0.0)
    for c in range(accum):
        sl = slice(c * b, (c + 1) * b)
        w_, t_, ta_, m_ = win[sl], tgt[sl], taus[sl], ma[sl]
        zero = torch.zeros((), device=win.device)
        l_lat = l_amp = l_tc = l_back = l_cos = l_en = l_st = l_sx = l_sp = zero
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tc.amp):
            z0 = model.encode(w_ if win_enc is None else win_enc[sl], mask, m_)
            if lat_noise > 0:
                rms = (
                    den_lat if den_lat is not None else z0.float().pow(2).mean()
                ).sqrt()
                zc, prev, steps = z0.float(), torch.zeros_like(ta_[:, :1]), []
                for j in range(ta_.shape[1]):
                    zc = zc + lat_noise * rms.detach() * torch.randn_like(zc)
                    zc = model.propagate(zc, ta_[:, j : j + 1] - prev, m_)[:, 0]
                    prev = ta_[:, j : j + 1]
                    steps.append(zc)
                zs = torch.stack(steps, 1)[:, idx]
            else:
                zs = model.propagate(z0.float(), ta_, m_)[:, idx]  # [b, n, d]
            x_hat = model.decode(zs, mask, m_.repeat_interleave(nd)).float()
            x_rec = model.decode(z0, mask, m_).float()
            l_pred = weighted_mse(x_hat, t_[:, idx], w_ch, lw)
            l_rec = weighted_mse(x_rec, w_[:, -1], w_ch)
            if zt is not None:
                ztc = zt[sl]
                l_lat = ((zs[..., :sd] - ztc[..., :sd]) ** 2).mean() / den_lat
                if den_det is not None:
                    l_lat = (
                        l_lat + ((zs[..., sd:] - ztc[..., sd:]) ** 2).mean() / den_det
                    )
                if amp_on:
                    mu = model.fixed_point(model.embed_cond(m_), b)[:, None]
                    l_amp = amp_loss(
                        (zs - mu)[..., :sd], (ztc - mu)[..., :sd].detach(), den_amp
                    )
                if w_cos > 0:
                    l_cos = (
                        1 - F.cosine_similarity(zs[..., :sd], ztc[..., :sd], dim=-1)
                    ).mean()
                if w_energy > 0:
                    nrm = torch.cat(
                        [
                            z0.float()[:, None, :sd].norm(dim=-1),
                            zs[..., :sd].norm(dim=-1),
                        ],
                        1,
                    )
                    d = nrm[:, 1:] - nrm[:, :-1]
                    l_en = d[:, ok].pow(2).mean() / den_en
                if w_back > 0:
                    # backward consistency: encode the last window of the clip and run
                    # the same generator backwards to every earlier latent (and to t)
                    full_ = torch.cat([w_, t_], 1)
                    z_end = model.encode(full_[:, -n_frames:], mask, m_).float()
                    t_end = ta_[:, -1:]
                    back = torch.cat([ta_[:, idx] - t_end, -t_end], 1)
                    zb = model.propagate(z_end, back, m_)[..., :sd]
                    ref = torch.cat([ztc, z0.float()[:, None]], 1)[..., :sd].detach()
                    l_back = ((zb - ref) ** 2).mean() / den_lat
            if den_tc is not None:
                num, _ = multistart_consistency(
                    model,
                    torch.cat([w_, t_], 1),
                    z0,
                    ta_,
                    m_,
                    mask,
                    n_frames,
                    list(tc.tc_starts),
                )
                l_tc = num / den_tc
            if w_sob_t > 0:
                l_st = sobolev_time(
                    torch.cat([x_rec[:, None], x_hat], 1),
                    torch.cat([w_[:, -1:], t_[:, idx]], 1),
                    w_ch,
                    ok,
                )
            if w_sob_x > 0:
                l_sx = sobolev_space(x_hat - t_[:, idx], w_ch)
            if w_spec > 0:
                l_sp = spectral_l1(x_hat, t_[:, idx], w_ch, den_spec)
            loss = (
                tc.w_rec * l_rec
                + tc.w_pred * l_pred
                + tc.w_lat * l_lat
                + tc.w_amp * l_amp
                + tc.w_tc * l_tc
                + w_back * l_back
                + w_cos * l_cos
                + w_energy * l_en
                + w_sob_t * l_st
                + w_sob_x * l_sx
                + w_spec * l_sp
            )
        (loss / accum).backward()
        for k, v in zip(
            LOSS_TERMS,
            (
                loss,
                l_rec,
                l_pred,
                l_lat,
                l_amp,
                l_tc,
                l_back,
                l_cos,
                l_en,
                l_st,
                l_sx,
                l_sp,
            ),
        ):
            tot[k] += float(v) / accum
    return tot


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
        smp = train.sample(
            tc.batch, n_frames, K, list(tc.strides), g, fine=tc.fine_targets
        )
        win, tgt, taus, ma, s = smp[:5]
        tgt_hist = smp[5] if tc.fine_targets else None
        win_enc = None
        if tc.ma_noise > 0:
            ma = ma + tc.ma_noise * torch.randn(ma.shape, generator=g).to(ma)
        if tc.input_noise > 0:
            win_enc = win + tc.input_noise * torch.randn(win.shape, generator=g).to(win)
        n_dec = min(tc.n_dec, K)
        rest = (
            torch.randperm(K - 2, generator=g)[: max(0, n_dec - 2)] + 1
            if K > 2
            else torch.tensor([], dtype=torch.long)
        )
        idx = torch.unique(torch.cat([torch.tensor([0, K - 1]), rest])).to(dev)
        opt.zero_grad(set_to_none=True)
        parts = step_backward(
            model,
            win,
            tgt,
            taus,
            ma,
            train.mask,
            idx,
            K,
            n_frames,
            tc,
            w_ch,
            tc.accum,
            tgt_hist,
            win_enc,
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
            # without a validation split best.pth would be a copy of last.pth (score =
            # -step), so it is only written when there is a validation set
            if score < best:
                best = score
                if val is not None:
                    torch.save(state, run / "best.pth")
    (run / "DONE").write_text(
        json.dumps({"best_score": best, "seconds": time.time() - t_start})
    )
    print("run dir:", run)


if __name__ == "__main__":
    main()
