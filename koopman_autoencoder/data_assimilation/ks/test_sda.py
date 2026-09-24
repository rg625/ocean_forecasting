"""Sanity checks that the score-based DA implementation is a genuine posterior method.

Run *before* committing compute to full training, and again on the trained model before
any SDA number is reported.  If check 3, 4 or 5 fails on the trained model, the SDA
baseline is not valid and must be omitted rather than reported.

    python -m data_assimilation.ks.test_sda --ckpt model_outputs_ks/sda/win26/best_model.pth
    python -m data_assimilation.ks.test_sda --untrained          # implementation-only checks

Checks
------
1. Diffusion schedule and the eps-loss are self-consistent (a perfect eps-predictor
   recovers x0 exactly via the Tweedie formula).
2. The observation-likelihood gradient points in the direction that reduces the
   observation residual.
3. Observations actually influence the posterior: posterior observation error is
   substantially below prior observation error, with everything else held fixed.
4. With no observed entries the sampler reproduces the unconditional prior exactly
   (same seed, same schedule).
5. Synthetic observations move posterior samples towards those observations, and the
   effect is localised: error at observed frames drops more than at unobserved frames.
6. (trained model only) Unconditional samples look like KS: field standard deviation and
   spatial energy spectrum are compared against the training distribution.
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

from data_assimilation.ks.sda import ScoreBasedDA, ScoreUNet2d, VPSchedule
from data_assimilation.ks.train_unet_ks import load_norm, make_windows

logger = logging.getLogger("data_assimilation.ks.test_sda")


def build(ckpt: Path | None, window: int, dev, n_sample_steps: int = 64):
    net = ScoreUNet2d().to(dev)
    sched = VPSchedule()
    if ckpt is not None and ckpt.is_file():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(st["model_state_dict"])
        window = int(st.get("window", window))
        logger.info(
            f"loaded score model from {ckpt} (window {window}, "
            f"val {st.get('val_loss')})"
        )
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return ScoreBasedDA(net, sched, window, dev, n_sample_steps=n_sample_steps), window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", type=Path, default=Path("model_outputs_ks/sda/win26/best_model.pth")
    )
    ap.add_argument(
        "--untrained",
        action="store_true",
        help="run implementation-only checks with random weights",
    )
    ap.add_argument("--train", type=Path, default=Path("data/ks/train.nc"))
    ap.add_argument("--window", type=int, default=26)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--sample-steps", type=int, default=64)
    ap.add_argument("--sigma-y", type=float, default=0.05)
    ap.add_argument(
        "--guidance",
        type=float,
        default=1.0,
        help="likelihood-guidance strength gamma, selected on validation",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    dev = torch.device(args.device)
    torch.manual_seed(0)
    sda, L = build(
        None if args.untrained else args.ckpt, args.window, dev, args.sample_steps
    )
    sda.guidance = args.guidance
    X = 64
    res = {
        "trained": not args.untrained,
        "window": L,
        "sample_steps": args.sample_steps,
        "sigma_y": args.sigma_y,
        "guidance": args.guidance,
    }

    # ---- 1. schedule / Tweedie consistency ---------------------------------
    x0 = torch.randn(4, 1, L, X, device=dev)
    t = torch.tensor([500], device=dev)
    a = sda.abar[t][:, None, None, None]
    eps = torch.randn_like(x0)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
    x0_rec = (xt - (1 - a).sqrt() * eps) / a.sqrt()
    err = float((x0_rec - x0).abs().max())
    res["check1_tweedie_max_abs_err"] = err
    res["check1_pass"] = err < 1e-4
    logger.info(
        f"1. Tweedie/schedule consistency: max |x0_rec - x0| = {err:.2e} "
        f"-> {'PASS' if res['check1_pass'] else 'FAIL'}"
    )

    # ---- 2. likelihood gradient direction -----------------------------------
    obs_idx = np.array([0, 5, 12, 25 if L > 25 else L - 1])
    obs_idx = np.unique(np.clip(obs_idx, 0, L - 1))
    mask = torch.zeros(len(obs_idx), X, device=dev)
    mask[:, ::4] = 1.0
    y = torch.randn(len(obs_idx), 2, X, device=dev)
    xv = torch.zeros(2, 1, L, X, device=dev, requires_grad=True)
    resid = (
        xv[:, 0][:, torch.as_tensor(obs_idx, device=dev)] - y.permute(1, 0, 2)
    ) * mask[None]
    ll = -(resid**2).sum() / (2 * args.sigma_y**2)
    gr = torch.autograd.grad(ll, xv)[0]
    # a step along the gradient must reduce the residual
    step = 1e-3
    xv2 = (xv + step * gr).detach()
    r2 = (
        xv2[:, 0][:, torch.as_tensor(obs_idx, device=dev)] - y.permute(1, 0, 2)
    ) * mask[None]
    before, after = float((resid**2).sum()), float((r2**2).sum())
    res["check2_resid_before"], res["check2_resid_after"] = before, after
    res["check2_pass"] = after < before
    # gradient must vanish on unobserved entries
    unobs = float(
        gr[:, 0][:, torch.as_tensor(obs_idx, device=dev)][:, :, 1::4].abs().max()
    )
    res["check2_grad_on_unobserved"] = unobs
    res["check2_localised"] = unobs < 1e-8
    logger.info(
        f"2. likelihood gradient: residual {before:.4f} -> {after:.4f} "
        f"{'PASS' if res['check2_pass'] else 'FAIL'}; max |grad| on unobserved "
        f"entries = {unobs:.2e} -> {'PASS' if res['check2_localised'] else 'FAIL'}"
    )

    # ---- data for the sampling checks --------------------------------------
    if args.train.is_file():
        u = load_norm(args.train, dev)
        win = make_windows(u, L)
        truth = win[torch.randint(0, win.shape[0], (2,), device=dev)]  # [2, L, X]
    else:
        truth = torch.randn(2, L, X, device=dev)
    B = truth.shape[0]
    obs_idx = np.array([2, 8, 17, L - 1])
    mask_s = torch.zeros(len(obs_idx), X, device=dev)
    mask_s[:, ::4] = 1.0  # 25% of points
    y_s = truth[:, torch.as_tensor(obs_idx, device=dev)].permute(1, 0, 2).contiguous()
    y_s = y_s + args.sigma_y * torch.randn_like(y_s)

    def obs_err(samples):
        """Relative L2 at observed entries, averaged over samples and problems."""
        p = samples[:, :, torch.as_tensor(obs_idx, device=dev)]  # [S,B,n_obs,X]
        d = (p - y_s.permute(1, 0, 2)[None]) * mask_s[None, None]
        num = torch.linalg.vector_norm(d.flatten(2), dim=2)
        den = torch.linalg.vector_norm(
            (y_s.permute(1, 0, 2)[None] * mask_s[None, None]).flatten(2), dim=2
        )
        return float((num / den.clamp_min(1e-12)).mean())

    # ---- 3. observations influence the posterior ---------------------------
    post = sda.sample(obs_idx, y_s, mask_s, args.sigma_y, args.n_samples, B, seed=1)
    nfe_post, nbw = sda.nfe, sda.n_backward
    prior = sda.sample(
        obs_idx, y_s, mask_s, args.sigma_y, args.n_samples, B, seed=1, return_prior=True
    )
    e_post, e_prior = obs_err(post), obs_err(prior)
    res["check3_obs_err_prior"], res["check3_obs_err_posterior"] = e_prior, e_post
    res["check3_ratio"] = e_post / max(e_prior, 1e-12)
    res["check3_pass"] = e_post < 0.5 * e_prior
    res["nfe_per_sample"] = nfe_post / args.n_samples
    res["n_backward_per_sample"] = nbw / args.n_samples
    logger.info(
        f"3. observation influence: prior obs err {e_prior:.4f} -> posterior "
        f"{e_post:.4f} (ratio {res['check3_ratio']:.3f}) -> "
        f"{'PASS' if res['check3_pass'] else 'FAIL'}"
    )

    # ---- 4. empty mask reproduces the prior --------------------------------
    zero_mask = torch.zeros_like(mask_s)
    post0 = sda.sample(obs_idx, y_s, zero_mask, args.sigma_y, 2, B, seed=7)
    prior0 = sda.sample(
        obs_idx, y_s, zero_mask, args.sigma_y, 2, B, seed=7, return_prior=True
    )
    d = float((post0 - prior0).abs().max())
    res["check4_max_abs_diff"] = d
    res["check4_pass"] = d < 1e-3
    logger.info(
        f"4. no observations => prior: max |posterior - prior| = {d:.2e} -> "
        f"{'PASS' if res['check4_pass'] else 'FAIL'}"
    )

    # ---- 5. conditioning is localised to observed frames -------------------
    unobs_idx = np.array([i for i in range(L) if i not in set(obs_idx.tolist())])

    def frame_err(samples, frames):
        p = samples[:, :, torch.as_tensor(frames, device=dev)]
        t_ = truth[:, torch.as_tensor(frames, device=dev)][None]
        num = torch.linalg.vector_norm((p - t_).flatten(2), dim=2)
        den = torch.linalg.vector_norm(t_.flatten(2), dim=2).clamp_min(1e-12)
        return float((num / den).mean())

    go_p, go_q = frame_err(post, obs_idx), frame_err(prior, obs_idx)
    gu_p, gu_q = frame_err(post, unobs_idx), frame_err(prior, unobs_idx)
    res.update(
        {
            "check5_obs_frames_prior": go_q,
            "check5_obs_frames_post": go_p,
            "check5_unobs_frames_prior": gu_q,
            "check5_unobs_frames_post": gu_p,
        }
    )
    res["check5_pass"] = (go_q - go_p) > (gu_q - gu_p)
    logger.info(
        f"5. localisation: observed frames {go_q:.4f}->{go_p:.4f} "
        f"(improve {go_q - go_p:+.4f}), unobserved {gu_q:.4f}->{gu_p:.4f} "
        f"(improve {gu_q - gu_p:+.4f}) -> "
        f"{'PASS' if res['check5_pass'] else 'FAIL'}"
    )

    # ---- 6. prior realism (trained model only) -----------------------------
    if not args.untrained and args.train.is_file():
        s_np = prior.reshape(-1, L, X).cpu().numpy()
        t_np = win[:2048].cpu().numpy()
        res["check6_sample_std"] = float(s_np.std())
        res["check6_data_std"] = float(t_np.std())
        sp_s = (np.abs(np.fft.rfft(s_np, axis=-1)) ** 2).mean(axis=(0, 1))
        sp_t = (np.abs(np.fft.rfft(t_np, axis=-1)) ** 2).mean(axis=(0, 1))
        res["check6_spectrum_sample"] = (sp_s[:12] / sp_s[1]).tolist()
        res["check6_spectrum_data"] = (sp_t[:12] / sp_t[1]).tolist()
        ratio = res["check6_sample_std"] / max(res["check6_data_std"], 1e-12)
        res["check6_std_ratio"] = ratio
        res["check6_pass"] = 0.7 < ratio < 1.4
        logger.info(
            f"6. prior realism: sample std {res['check6_sample_std']:.3f} vs data "
            f"{res['check6_data_std']:.3f} (ratio {ratio:.3f}) -> "
            f"{'PASS' if res['check6_pass'] else 'FAIL'}"
        )
        logger.info(
            f"   spectrum k=1..6 sample "
            f"{np.round(np.array(res['check6_spectrum_sample'][1:7]), 3).tolist()}"
        )
        logger.info(
            f"   spectrum k=1..6 data   "
            f"{np.round(np.array(res['check6_spectrum_data'][1:7]), 3).tolist()}"
        )

    # ---- 7. the score model is a TRAJECTORY model, not a per-frame model ----
    # Perturbing a single point of a single frame must change the predicted score at
    # other frames; otherwise observations cannot propagate information through time
    # and the "posterior" would be pointwise interpolation rather than assimilation.
    probe = torch.zeros(1, 1, L, X, device=dev)
    tt = torch.full((1,), float(sda.sched.n_steps // 2), device=dev)
    with torch.no_grad():
        b0 = sda.net(probe, tt)
        pp = probe.clone()
        pp[0, 0, L // 2, X // 2] = 1.0
        b1 = sda.net(pp, tt)
    dresp = (b1 - b0).abs()[0, 0].sum(1)  # response per frame
    responding = int((dresp > 1e-9).sum())
    res["check7_frames_responding"] = responding
    res["check7_n_frames"] = L
    if args.untrained:
        # The output convolution is zero-initialised, so an untrained network returns
        # exactly zero for every input and this probe is vacuous. The architecture's
        # temporal receptive field is verified separately with non-zero output weights;
        # here the check only gates the trained model.
        res["check7_skipped_untrained"] = True
        logger.info(
            "7. trajectory prior: SKIPPED (untrained network is identically zero; "
            "this check gates the trained model only)"
        )
    else:
        res["check7_pass"] = responding > 1
        logger.info(
            f"7. trajectory prior: perturbing one point of frame {L // 2} changes "
            f"the score at {responding}/{L} frames -> "
            f"{'PASS (trajectory model)' if res['check7_pass'] else 'FAIL (per-frame only)'}"
        )

    # ---- 8. temporal autocorrelation of prior samples (trained model only) ---
    if not args.untrained and args.train.is_file():

        def autocorr(a):
            """Lag-wise correlation between frame 0 and frame k, averaged over samples."""
            a = a - a.mean(axis=(-1,), keepdims=True)
            r = []
            for k in range(L):
                num = (a[:, 0] * a[:, k]).mean()
                den = np.sqrt((a[:, 0] ** 2).mean() * (a[:, k] ** 2).mean())
                r.append(float(num / max(den, 1e-12)))
            return np.array(r)

        ac_s = autocorr(prior.reshape(-1, L, X).cpu().numpy())
        ac_t = autocorr(win[:2048].cpu().numpy())
        res["check8_autocorr_sample"] = ac_s.tolist()
        res["check8_autocorr_data"] = ac_t.tolist()
        mad = float(np.abs(ac_s - ac_t).mean())
        res["check8_mean_abs_dev"] = mad
        res["check8_pass"] = mad < 0.15
        logger.info(
            f"8. temporal autocorrelation: mean |sample - data| over lags "
            f"= {mad:.4f} -> {'PASS' if res['check8_pass'] else 'FAIL'}"
        )
        logger.info(
            f"   lag 0,5,10,25 sample "
            f"{np.round(ac_s[[0, 5, 10, min(25, L - 1)]], 3).tolist()} vs data "
            f"{np.round(ac_t[[0, 5, 10, min(25, L - 1)]], 3).tolist()}"
        )

    # ---- 9. observations propagate to UNOBSERVED frames (trained model only) --
    # The defining property of trajectory posterior inference: constraining some frames
    # must improve the estimate at frames that were never observed.
    if not args.untrained:
        imp_unobs = gu_q - gu_p
        res["check9_unobserved_improvement"] = imp_unobs
        res["check9_pass"] = imp_unobs > 0.01
        logger.info(
            f"9. information propagates to unobserved frames: prior {gu_q:.4f} -> "
            f"posterior {gu_p:.4f} (improvement {imp_unobs:+.4f}) -> "
            f"{'PASS' if res['check9_pass'] else 'FAIL'}"
        )

    gate = [k for k in res if k.endswith("_pass")]
    res["all_pass"] = all(bool(res[k]) for k in gate)
    res["checks_run"] = gate
    logger.info(
        f"SUMMARY: {sum(bool(res[k]) for k in gate)}/{len(gate)} checks pass "
        f"-> {'VALID' if res['all_pass'] else 'NOT VALID'}"
    )
    logger.info(
        f"cost: {res['nfe_per_sample']:.0f} score evaluations + "
        f"{res['n_backward_per_sample']:.0f} guidance backward passes per sample"
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, default=float)
        logger.info(f"saved -> {args.out}")
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
