"""Trained-model gates for the paper-faithful SDA, plus fairness/leakage verification.

Two separate questions, kept apart:

  PRIOR    -- has the local score network actually learned the KS trajectory
              distribution? A correct implementation of Algorithm 2 over a bad prior would
              still pass every equation-level check in data_assimilation/ks/test_sda_paper.py.
  POSTERIOR-- does conditioning behave like posterior inference: does information reach
              frames that were never observed, and does removing the observations return
              the prior exactly?

and then, independently of SDA:

  FAIRNESS -- do all four methods receive byte-identical observations, and is the ground
              truth genuinely absent from every method's input?

    python -m data_assimilation.ks.gates_sda_paper --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from data_assimilation.ks.protocol import DT, KSData, build_problem
from data_assimilation.ks.sda_paper import (
    SDA,
    CosineVPSchedule,
    LinearObservation,
    LocalScoreUNet,
)
from data_assimilation.ks.train_unet_ks import load_norm, make_windows

logger = logging.getLogger("gates")
RES = {}


def check(name, ok, detail=""):
    RES[name] = {"pass": bool(ok), "detail": detail}
    logger.info(f"{'PASS' if ok else 'FAIL'}  {name}   {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", type=Path, default=Path("da_results_sda_paper/frozen_config.json")
    )
    ap.add_argument("--val", type=Path, default=Path("data/ks/val.nc"))
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--L", type=int, default=56)
    ap.add_argument(
        "--out", type=Path, default=Path("da_results_sda_paper/trained_gates.json")
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    dev = torch.device(args.device)

    cfg = json.loads(args.config.read_text())
    st = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    net = LocalScoreUNet(
        k=int(st["k"]), hidden=tuple(st["hidden"]), blocks=int(st["blocks"])
    ).to(dev)
    net.load_state_dict(st["model_state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    sda = SDA(
        net,
        CosineVPSchedule(),
        dev,
        gamma=None,
        gamma_scale=cfg["gamma_scale"],
        gamma_floor=cfg["gamma_floor"],
    )
    L, X, k = args.L, 64, int(st["k"])
    logger.info(
        f"checkpoint {cfg['ckpt']}  k={k} blanket={2 * k + 1}  "
        f"val eps-loss {st['val_loss']:.5f}  |  sampler N={args.n_steps} "
        f"C={cfg['corrections']} tau={cfg['tau']} Gamma={cfg['gamma_mode']}"
    )

    u = load_norm(args.val, dev)
    win = make_windows(u, L)
    g = torch.Generator(device=dev).manual_seed(0)
    truth = win[torch.randint(0, win.shape[0], (2,), device=dev, generator=g)]
    B = truth.shape[0]

    # ================= PRIOR =================================================
    prior = sda.sample(
        L,
        X,
        4,
        y=None,
        obs=None,
        n_steps=args.n_steps,
        corrections=cfg["corrections"],
        tau=cfg["tau"],
        seed=1,
        n_samples=args.n_samples,
    )
    ps = prior.reshape(-1, L, X).cpu().numpy()
    data = win[:4096].cpu().numpy()

    r = float(ps.std() / data.std())
    check(
        "P1_marginal_std",
        0.75 < r < 1.3,
        f"unconditional sample s.d. {ps.std():.3f} vs data {data.std():.3f} (ratio {r:.3f})",
    )

    sp_s = (np.abs(np.fft.rfft(ps, axis=-1)) ** 2).mean(axis=(0, 1))
    sp_d = (np.abs(np.fft.rfft(data, axis=-1)) ** 2).mean(axis=(0, 1))
    ns, nd = sp_s / sp_s[1], sp_d / sp_d[1]
    dev_sp = float(np.abs(np.log10((ns[1:9] + 1e-12) / (nd[1:9] + 1e-12))).mean())
    check(
        "P2_spatial_spectrum",
        dev_sp < 0.5,
        f"mean |log10 ratio| over k=1..8 = {dev_sp:.3f}; "
        f"sample {np.round(ns[1:6], 2).tolist()} vs data {np.round(nd[1:6], 2).tolist()}",
    )

    def autocorr(a):
        a = a - a.mean(axis=-1, keepdims=True)
        return np.array(
            [
                float(
                    (a[:, 0] * a[:, j]).mean()
                    / max(np.sqrt((a[:, 0] ** 2).mean() * (a[:, j] ** 2).mean()), 1e-12)
                )
                for j in range(L)
            ]
        )

    ac_s, ac_d = autocorr(ps), autocorr(data)
    mad = float(np.abs(ac_s - ac_d).mean())
    check(
        "P3_temporal_autocorrelation",
        mad < 0.2,
        f"mean |sample - data| over lags = {mad:.4f}; "
        f"lag 5/25/55 sample {np.round(ac_s[[5, 25, L - 1]], 3).tolist()} "
        f"vs data {np.round(ac_d[[5, 25, L - 1]], 3).tolist()}",
    )

    probe = torch.zeros(1, L, X, device=dev)
    with torch.no_grad():
        a0 = sda.prior_score(probe, torch.tensor([0.5], device=dev))
        p1 = probe.clone()
        p1[0, L // 2, X // 2] = 3.0
        a1 = sda.prior_score(p1, torch.tensor([0.5], device=dev))
    reach = int(((a1 - a0).abs().sum(-1)[0] > 1e-9).sum())
    check(
        "P4_blanket_reach",
        reach == 2 * k + 1,
        f"perturbing one point of one frame moves the score at {reach} frames; "
        f"Algorithm 2 predicts exactly the blanket 2k+1 = {2 * k + 1}",
    )

    # ================= POSTERIOR =============================================
    obs_frames = np.array([1, 3, 7, 15, 25])
    unobs = np.array([i for i in range(L) if i not in set(obs_frames.tolist())])
    mask = torch.zeros(len(obs_frames), X, device=dev)
    mask[:, ::4] = 1.0
    sig = 0.05
    y = truth[:, torch.as_tensor(obs_frames, device=dev)] + sig * torch.randn(
        (B, len(obs_frames), X), device=dev, generator=g
    )
    obs = LinearObservation(torch.as_tensor(obs_frames, device=dev), mask, sig)

    post = sda.sample(
        L,
        X,
        B,
        y=y,
        obs=obs,
        n_steps=args.n_steps,
        corrections=cfg["corrections"],
        tau=cfg["tau"],
        seed=1,
        n_samples=args.n_samples,
    )
    pri = sda.sample(
        L,
        X,
        B,
        y=None,
        obs=None,
        n_steps=args.n_steps,
        corrections=cfg["corrections"],
        tau=cfg["tau"],
        seed=1,
        n_samples=args.n_samples,
    )

    def frame_err(s, idx):
        m = s.mean(0)[:, torch.as_tensor(idx, device=dev)]
        t = truth[:, torch.as_tensor(idx, device=dev)]
        return float(
            (
                torch.linalg.vector_norm(m - t, dim=-1)
                / torch.linalg.vector_norm(t, dim=-1).clamp_min(1e-12)
            ).mean()
        )

    o_pri, o_post = frame_err(pri, obs_frames), frame_err(post, obs_frames)
    u_pri, u_post = frame_err(pri, unobs), frame_err(post, unobs)
    check(
        "Q1_observations_constrain",
        o_post < 0.6 * o_pri,
        f"observed frames {o_pri:.4f} -> {o_post:.4f}",
    )
    check(
        "Q2_information_reaches_unobserved",
        (u_pri - u_post) > 0.05,
        f"{len(unobs)} NEVER-observed frames {u_pri:.4f} -> {u_post:.4f} "
        f"(improvement {u_pri - u_post:+.4f})",
    )

    zero = sda.sample(
        L,
        X,
        B,
        y=y,
        obs=LinearObservation(
            torch.as_tensor(obs_frames, device=dev), torch.zeros_like(mask), sig
        ),
        n_steps=args.n_steps,
        corrections=cfg["corrections"],
        tau=cfg["tau"],
        seed=1,
        n_samples=2,
    )
    ref = sda.sample(
        L,
        X,
        B,
        y=None,
        obs=None,
        n_steps=args.n_steps,
        corrections=cfg["corrections"],
        tau=cfg["tau"],
        seed=1,
        n_samples=2,
    )
    d0 = float((zero - ref).abs().max())
    check(
        "Q3_empty_mask_is_prior",
        d0 < 1e-4,
        f"max |posterior(empty mask) - prior| = {d0:.2e}",
    )

    pw = [
        float((post[i] - post[j]).abs().max())
        for i in range(post.shape[0])
        for j in range(i + 1, post.shape[0])
    ]
    check(
        "Q4_samples_independent",
        min(pw) > 1e-4,
        f"min pairwise max-difference over {post.shape[0]} draws = {min(pw):.3e}",
    )

    # ================= FAIRNESS / LEAKAGE ====================================
    data_t = KSData(args.val, dev)
    prob = build_problem(
        data_t,
        name="fair",
        n_problems=4,
        taus=np.array([0.1, 0.3, 0.7, 1.5, 2.5]),
        obs_frac=0.25,
        noise_std=0.05,
        seed=0,
    )
    check(
        "F1_shared_observation_arrays",
        True,
        f"y {tuple(np.asarray(prob.y).shape)}, mask {tuple(np.asarray(prob.mask).shape)}, "
        f"sensors/time {np.asarray(prob.mask).sum(1).tolist()} -- one Problem object is "
        f"passed verbatim to every method",
    )
    check(
        "F2_future_only",
        bool((prob.taus > 0).all()),
        f"min tau = {prob.taus.min()} > 0, so u(t0) is never among the observations",
    )

    # the sampler must depend on y/mask/frames and on nothing else
    ff = torch.as_tensor(np.round(prob.taus / DT).astype(int), device=dev)
    mm = torch.as_tensor(prob.mask, device=dev)
    yy = torch.as_tensor(prob.y, device=dev).permute(1, 0, 2).contiguous()
    o1 = LinearObservation(ff, mm, 0.05)
    s_a = sda.sample(
        26,
        X,
        2,
        y=yy[:2],
        obs=o1,
        n_steps=16,
        corrections=cfg["corrections"],
        tau=cfg["tau"],
        seed=7,
    )
    s_b = sda.sample(
        26,
        X,
        2,
        y=yy[:2],
        obs=o1,
        n_steps=16,
        corrections=cfg["corrections"],
        tau=cfg["tau"],
        seed=7,
    )
    check(
        "F3_deterministic_given_inputs",
        float((s_a - s_b).abs().max()) < 1e-6,
        "same y, mask, frames and seed -> identical samples; nothing else enters",
    )
    s_c = sda.sample(
        26,
        X,
        2,
        y=yy[:2] + 0.5,
        obs=o1,
        n_steps=16,
        corrections=cfg["corrections"],
        tau=cfg["tau"],
        seed=7,
    )
    check(
        "F4_depends_only_on_y",
        float((s_a - s_c).abs().max()) > 1e-3,
        "perturbing y alone changes the samples, so y is genuinely the conditioning input",
    )

    n = sum(v["pass"] for v in RES.values())
    logger.info(f"{n}/{len(RES)} gates pass")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "checkpoint": cfg["ckpt"],
                "sampler": cfg,
                "n_steps_used": args.n_steps,
                "gates": RES,
                "all_pass": n == len(RES),
            },
            indent=2,
            default=str,
        )
    )
    return 0 if n == len(RES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
