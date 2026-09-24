# mypy: disable-error-code="arg-type, assignment, index, no-any-return, operator, var-annotated"
"""Data-assimilation campaign on the transonic (tra) flow.

Methods, and why each gets the treatment it does:

    KAE          4D-Var on the latent z0.  e^{K tau} reaches any lead in one product.
    U-Net, FNO   4D-Var on the conditioning window.  Gradients traverse the rollout.
    ACDM         score-based DA over its own 3-frame window (see sda_blanket).
    ACDM-ncn     the SAME path as ACDM, by request.  It is trained with clean
                 conditioning, so a noised window is out of distribution for it; the
                 Tweedie diagnostic in STATUS.md quantifies that, and it is reported
                 beside the DA numbers so any gap is attributable.

Every method sees the identical `Problem`: same trajectories, same analysis times, same
observation geometry, same noise realisation, same obstacle mask.  All weights are frozen.
Learning rates are chosen on a VALIDATION regime (`gt_extrap.nc`, Mach 0.50-0.52) and then
frozen before the test regime (`gt_interp.nc`, Mach 0.66-0.68) is touched, so no
hyper-parameter is ever fitted on the numbers being reported.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from data_assimilation.tra.bridge import REGIMES, PhysicalData, REPO, rel_l2
from data_assimilation.tra.adapters import TurbpredAdapter
from data_assimilation.tra.kae_adapter import KAEAdapter
from data_assimilation.tra.protocol import build_problem, canonical
from data_assimilation.tra.fourdvar import solve_kae, solve_autoregressive
from data_assimilation.tra.verify_adapters import MODELS, DIFF_OPTS

logger = logging.getLogger("tra_da")
FOURDVAR = ["KAE", "UNet", "FNO"]
DIFFUSION = ["ACDM", "ACDM-ncn"]
DATA = {
    "test": "data/acdm/128_tra/gt_interp.nc",
    "val": "data/acdm/128_tra/gt_extrap.nc",
    "long": "data/acdm/128_tra/gt_longer.nc",
}


class Bench:
    """Loads each model once, on demand, and keeps only one on the GPU at a time."""

    def __init__(self, a):
        self.a = a
        self.dev = torch.device(a.device)
        self.regime = REGIMES["tra"]
        # set from tuning.json once it is loaded; every section then inherits the frozen
        # first guess without having to pass it down, as Bench.cfg does on KS
        self.tuning: Dict = {}
        self.base = REPO / "autoreg_pde_diffusion" / "pretrained_models" / "models_tra"
        self._cache = {}

    def data(self, split: str) -> PhysicalData:
        key = f"data:{split}"
        if key not in self._cache:
            self._cache[key] = PhysicalData(DATA[split], "tra", self.dev)
        return self._cache[key]

    def model(self, name: str):
        if name == "KAE":
            # the March ablation checkpoints predate the current architecture; they need
            # their own model config and, for no_history, a single conditioning frame
            kw = {}
            if getattr(self.a, "kae_config", None):
                kw["config_path"] = self.a.kae_config
            if getattr(self.a, "kae_input_frames", 0):
                kw["input_frames"] = self.a.kae_input_frames
            return KAEAdapter(self.a.kae_run, self.regime, self.dev, **kw)
        return TurbpredAdapter(
            self.base / MODELS[name] / "Model.pth",
            self.regime,
            self.dev,
            DIFF_OPTS.get(name),
            name=name,
        )

    def solve(
        self,
        name: str,
        ad,
        data,
        prob,
        *,
        iters: int,
        lr: float,
        seed: int,
        track_every: int = 0,
        init: Optional[str] = None,
    ) -> Dict:
        if init is None:
            init = self.tuning.get(name, {}).get("init", "climatology")
        fn = solve_kae if name == "KAE" else solve_autoregressive
        return fn(
            ad,
            data,
            prob,
            iters=iters,
            lr=lr,
            seed=seed,
            track_every=track_every,
            init=init,
        )


# ---------------------------------------------------------------------------
def tune(b: Bench, out: Path) -> Dict:
    """Pick each 4D-Var method's learning rate on the VALIDATION regime only."""
    logger.info("=== TUNING (validation regime: gt_extrap, Mach 0.50-0.52) ===")
    data = b.data("val")
    prob = build_problem(
        data,
        name="tune",
        n_problems=b.a.n_tune,
        offsets=canonical(),
        seed=1234,
        noise_std=b.a.noise,
    )
    best = {}
    for name in FOURDVAR:
        ad = b.model(name)
        rows = []
        for lr in b.a.lrs:
            r = b.solve(name, ad, data, prob, iters=b.a.tune_iters, lr=lr, seed=1)
            rows.append(
                {"lr": lr, "rel": float(np.mean(r["rel"])), "wall_s": r["wall_s"]}
            )
            logger.info(
                f"  {name:9s} lr={lr:<8g} rel-L2 {rows[-1]['rel']:.4f} "
                f"[{r['wall_s']:.0f}s]"
            )
        pick = min(rows, key=lambda x: x["rel"])
        best[name] = {
            "lr": pick["lr"],
            "sweep": rows,
            "chosen_on": "gt_extrap.nc (validation regime)",
            "iters_used_for_tuning": b.a.tune_iters,
        }
        logger.info(f"  -> {name}: lr = {pick['lr']}")
        del ad
        torch.cuda.empty_cache()
    (out / "tuning.json").write_text(json.dumps(best, indent=2))
    return best


# ---------------------------------------------------------------------------
def headline(b: Bench, out: Path, tuning: Dict) -> None:
    """The canonical problem, on the test regime, for every method."""
    logger.info(f"=== HEADLINE (regime: {DATA[b.a.regime]}) ===")
    data = b.data(b.a.regime)
    prob = build_problem(
        data,
        name="A",
        n_problems=b.a.n_problems,
        offsets=canonical(),
        seed=100,
        noise_std=b.a.noise,
        obs_frac=b.a.obs_frac,
    )
    logger.info(f"  {json.dumps(prob.meta())}")
    store, summary = {"problem": prob.meta()}, {}
    for name in FOURDVAR:
        ad = b.model(name)
        lr = tuning[name]["lr"]
        r = b.solve(
            name,
            ad,
            data,
            prob,
            iters=b.a.iters,
            lr=lr,
            seed=100,
            track_every=max(1, b.a.iters // 40),
        )
        v = np.asarray(r["rel"])
        summary[name] = {
            "mean": float(v.mean()),
            "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
            "n": int(v.size),
            "lr": lr,
            "iters": b.a.iters,
            "wall_s": r["wall_s"],
        }
        store[f"{name}__rel"] = v
        store[f"{name}__analysis"] = r["analysis"]
        for k2, v2 in r["hist"].items():
            store[f"{name}__hist_{k2}"] = np.asarray(v2)
        logger.info(
            f"  {name:9s} rel-L2 {v.mean():.4f} ± {summary[name]['sem']:.4f} "
            f"(n={v.size})  [{r['wall_s']:.0f}s, lr={lr}]"
        )
        del ad
        torch.cuda.empty_cache()
    np.savez_compressed(out / "A_headline.npz", **store)
    (out / "A_headline.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"saved -> {out / 'A_headline'}.npz/.json")


def headline_diffusion(b: Bench, out: Path, tuning: Dict) -> None:
    """ACDM and ACDM-ncn on the SAME problem, through the blanket score.

    Chunked over problems: the guidance term differentiates through every blanket segment,
    so a full batch would hold ``n_problems x (L - 2)`` U-Net graphs at once.
    """
    from data_assimilation.tra.sda_blanket import ACDMBlanketScore, BlanketSDA

    logger.info(f"=== HEADLINE, diffusion methods (regime: {DATA[b.a.regime]}) ===")
    data = b.data(b.a.regime)
    prob = build_problem(
        data,
        name="A",
        n_problems=b.a.n_problems,
        offsets=canonical(),
        seed=100,
        noise_std=b.a.noise,
        obs_frac=b.a.obs_frac,
    )
    L = int(prob.offsets.max()) + 1  # frames t0 .. t0 + delta_l
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    truth_t0 = data.frames(sim, t0)
    om = data.mask_for(sim)
    store, summary = {}, {}

    for name in DIFFUSION:
        ad = b.model(name)
        C = ad.n_fields + ad.n_params
        sc = ACDMBlanketScore(ad, C)
        td = (
            json.loads((out / "tuning_diffusion.json").read_text())[name]
            if (out / "tuning_diffusion.json").is_file()
            else None
        )
        gam = td["gamma"] if td else b.a.gamma
        corr = td["corrections"] if td else b.a.corrections
        sda = BlanketSDA(
            sc,
            sigma_y=max(prob.noise_std, b.a.sigma_y_floor),
            gamma=gam,
            corrections=corr,
            tau=b.a.tau,
        )
        obs_idx = torch.as_tensor(prob.offsets, device=b.dev)
        rels, t_start = [], time.perf_counter()
        for lo in range(0, len(prob.sim), b.a.chunk):
            hi = min(lo + b.a.chunk, len(prob.sim))
            sl = slice(lo, hi)
            y = ad.to_model(torch.as_tensor(prob.y[:, sl], device=b.dev)).transpose(
                0, 1
            )  # [b, N, F, H, W]
            # y goes through to_model, which puts it in turbpred's (128, 64) layout;
            # the mask comes straight from the .nc in (64, 128) and must follow it
            msk = (
                torch.as_tensor(prob.mask[:, sl], device=b.dev)
                .transpose(0, 1)
                .transpose(-1, -2)
            )
            par = data.params_for(sim[sl])
            H, W = y.shape[-2], y.shape[-1]
            pch = ad.param_channel(par, (H, W))
            # observations cover the field channels only; the parameter is known exactly
            y_full = torch.cat(
                [y, pch.unsqueeze(1).expand(-1, len(prob.offsets), -1, -1, -1)], dim=2
            )
            m_full = torch.cat(
                [
                    msk.expand(-1, -1, ad.n_fields, -1, -1),
                    torch.ones_like(pch)
                    .unsqueeze(1)
                    .expand(-1, len(prob.offsets), -1, -1, -1),
                ],
                dim=2,
            )
            smp = sda.sample(
                (hi - lo, L, C, H, W),
                obs_idx,
                y_full,
                m_full,
                seed=100,
                n_samples=b.a.n_samples,
            )
            ana = ad.to_physical(smp[:, :, 0, : ad.n_fields])  # [S, b, C_nc, H, W]
            tt = truth_t0[sl]
            omc = None if om is None else om[sl]
            per_draw = torch.stack(
                [rel_l2(ana[i], tt, omc) for i in range(ana.shape[0])]
            )
            rels.append(per_draw.mean(0).cpu().numpy())  # mean over draws
        v = np.concatenate(rels)
        summary[name] = {
            "mean": float(v.mean()),
            "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
            "n": int(v.size),
            "n_samples": b.a.n_samples,
            "blanket_window": sc.window,
            "k": sc.k,
            "timesteps": sc.timesteps,
            "gamma": gam,
            "corrections": corr,
            "wall_s": time.perf_counter() - t_start,
            "note": (
                (
                    "ACDM-ncn is trained with CLEAN conditioning, so a "
                    "noised window is out of distribution for it; it is run "
                    "on the identical path by request and its Tweedie "
                    "diagnostic is reported alongside."
                )
                if name == "ACDM-ncn"
                else ""
            ),
        }
        store[f"{name}__rel"] = v
        logger.info(
            f"  {name:9s} rel-L2 {v.mean():.4f} ± {summary[name]['sem']:.4f} "
            f"(n={v.size}, {b.a.n_samples} draws) "
            f"[{summary[name]['wall_s']:.0f}s]"
        )
        del ad, sc, sda
        torch.cuda.empty_cache()
    np.savez_compressed(out / "A_headline_diffusion.npz", **store)
    (out / "A_headline_diffusion.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"saved -> {out / 'A_headline_diffusion'}.npz/.json")


def tune_diffusion(b: Bench, out: Path) -> Dict:
    """Pick the guidance strength for the score-based methods, on VALIDATION only.

    ``gamma`` sets the covariance in the likelihood term, hence how hard the observations
    pull.  It is not transferable from the KS campaign: at 2-D resolution the KS value
    (1e-2) makes the guidance strong enough that the DDPM predictor amplifies instead of
    contracting, and the sampler runs away to NaN.  Sweeping it here, on the extrapolation
    regime, keeps it out of the reported numbers.  A configuration that DIVERGES is
    recorded as such rather than dropped.
    """
    from data_assimilation.tra.sda_blanket import ACDMBlanketScore, BlanketSDA

    logger.info("=== TUNING diffusion guidance (validation: gt_extrap) ===")
    data = b.data("val")
    prob = build_problem(
        data,
        name="tuneD",
        n_problems=b.a.n_tune_diff,
        offsets=canonical(),
        seed=4321,
        noise_std=b.a.noise,
    )
    L = int(prob.offsets.max()) + 1
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    tt, om = data.frames(sim, t0), data.mask_for(sim)
    best = {}
    for name in DIFFUSION:
        ad = b.model(name)
        C = ad.n_fields + ad.n_params
        sc = ACDMBlanketScore(ad, C)
        y = ad.to_model(torch.as_tensor(prob.y, device=b.dev)).transpose(0, 1)
        # same layout correction as in _solve_diffusion
        msk = torch.as_tensor(prob.mask, device=b.dev).transpose(0, 1).transpose(-1, -2)
        par = data.params_for(sim)
        H, W = y.shape[-2], y.shape[-1]
        N = len(prob.offsets)
        pch = ad.param_channel(par, (H, W))
        yf = torch.cat([y, pch.unsqueeze(1).expand(-1, N, -1, -1, -1)], dim=2)
        mf = torch.cat(
            [
                msk.expand(-1, -1, ad.n_fields, -1, -1),
                torch.ones_like(pch).unsqueeze(1).expand(-1, N, -1, -1, -1),
            ],
            dim=2,
        )
        obs = torch.as_tensor(prob.offsets, device=b.dev)
        rows = []
        for gam in b.a.gammas:
            for corr in b.a.corrections_grid:
                sda = BlanketSDA(
                    sc,
                    sigma_y=b.a.sigma_y_floor,
                    gamma=gam,
                    corrections=corr,
                    tau=b.a.tau,
                )
                smp = sda.sample(
                    (len(prob.sim), L, C, H, W), obs, yf, mf, seed=7, n_samples=1
                )
                fin = bool(torch.isfinite(smp).all())
                rel = (
                    float(
                        rel_l2(
                            ad.to_physical(smp[0, :, 0, : ad.n_fields]), tt, om
                        ).mean()
                    )
                    if fin
                    else float("inf")
                )
                rows.append(
                    {
                        "gamma": gam,
                        "corrections": corr,
                        "diverged": not fin,
                        "rel": rel if fin else None,
                    }
                )
                logger.info(
                    f"  {name:9s} gamma={gam:<7g} corr={corr} "
                    + ("DIVERGED" if not fin else f"rel-L2 {rel:.4f}")
                )
        ok = [r for r in rows if not r["diverged"]]
        if not ok:
            raise SystemExit(f"{name}: every guidance setting diverged")
        pick = min(ok, key=lambda r: r["rel"])
        best[name] = {
            "gamma": pick["gamma"],
            "corrections": pick["corrections"],
            "sweep": rows,
            "chosen_on": "gt_extrap.nc (validation regime)",
            "sigma_y": b.a.sigma_y_floor,
        }
        logger.info(
            f"  -> {name}: gamma={pick['gamma']}, corrections={pick['corrections']}"
        )
        del ad, sc
        torch.cuda.empty_cache()
    f = out / "tuning_diffusion.json"
    f.write_text(json.dumps(best, indent=2))
    return best


# ---------------------------------------------------------------------------
def _solve_diffusion(
    b: Bench,
    ad,
    data,
    prob,
    name: str,
    out: Path,
    *,
    back: int = 0,
    keep_traj: bool = False,
    seed: int = 100,
):
    """One score-based DA solve on a prepared Problem -> per-problem rel-L2.

    Chunked over problems because the guidance term differentiates through every blanket
    segment at once.  Guidance strength comes from the frozen validation tuning.

    ``back`` extends the sampled trajectory ``back`` frames BEFORE t_0.  The observations
    keep their absolute times (their indices shift with the trajectory) and the analysis
    is still the frame at t_0, now at trajectory index ``back``.  Two things need this:
    rolling the analysis forward under ACDM's own dynamics requires a conditioning window
    of ``n_control_frames`` frames ENDING at t_0, and asking for a state before t_0 at all
    requires the trajectory to cover it.

    Returns a dict: per-problem rel-L2 at t_0, the posterior-mean analysis for the first
    chunk, the posterior-mean conditioning window in MODEL space for every problem (so it
    can be handed to ``adapter.rollout`` exactly as a 4D-Var control would be), and
    optionally the posterior-mean trajectory in physical units.
    """
    from data_assimilation.tra.sda_blanket import ACDMBlanketScore, BlanketSDA

    C = ad.n_fields + ad.n_params
    sc = ACDMBlanketScore(ad, C)
    tdp = out / "tuning_diffusion.json"
    td = json.loads(tdp.read_text())[name] if tdp.is_file() else None
    gam = td["gamma"] if td else b.a.gamma
    corr = td["corrections"] if td else b.a.corrections
    sda = BlanketSDA(
        sc,
        sigma_y=max(prob.noise_std, b.a.sigma_y_floor),
        gamma=gam,
        corrections=corr,
        tau=b.a.tau,
    )
    # the sampler needs at least one full blanket window. With a single observation at
    # tau = 1 the natural length is 2 frames, shorter than ACDM's 3-frame window, so the
    # trajectory is padded. The extra frames carry no observation and the analysis is
    # still frame 0, so nothing about the problem changes.
    L = max(int(prob.offsets.max()) + 1 + int(back), sc.window)
    k = ad.n_control_frames
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    truth_t0, om = data.frames(sim, t0), data.mask_for(sim)
    obs_idx = torch.as_tensor(prob.offsets + int(back), device=b.dev)
    rels, ana_keep, wins, trajs = [], [], [], []
    for lo in range(0, len(prob.sim), b.a.chunk):
        hi = min(lo + b.a.chunk, len(prob.sim))
        sl = slice(lo, hi)
        y = ad.to_model(torch.as_tensor(prob.y[:, sl], device=b.dev)).transpose(0, 1)
        # same layout correction: y is in turbpred's (128, 64) after to_model
        msk = (
            torch.as_tensor(prob.mask[:, sl], device=b.dev)
            .transpose(0, 1)
            .transpose(-1, -2)
        )
        par = data.params_for(sim[sl])
        H, W = y.shape[-2], y.shape[-1]
        N = len(prob.offsets)
        pch = ad.param_channel(par, (H, W))
        y_full = torch.cat([y, pch.unsqueeze(1).expand(-1, N, -1, -1, -1)], dim=2)
        m_full = torch.cat(
            [
                msk.expand(-1, -1, ad.n_fields, -1, -1),
                torch.ones_like(pch).unsqueeze(1).expand(-1, N, -1, -1, -1),
            ],
            dim=2,
        )
        smp = sda.sample(
            (hi - lo, L, C, H, W),
            obs_idx,
            y_full,
            m_full,
            seed=int(seed) + lo,
            n_samples=b.a.n_samples,
        )
        ana = ad.to_physical(smp[:, :, int(back), : ad.n_fields])
        per = torch.stack(
            [
                rel_l2(ana[i], truth_t0[sl], None if om is None else om[sl])
                for i in range(ana.shape[0])
            ]
        )
        rels.append(per.mean(0).cpu().numpy())
        ana_keep.append(ana.mean(0).cpu().numpy())
        # the posterior-mean window ending at t_0, kept in MODEL space so that
        # adapter.rollout can advance it exactly as it advances a 4D-Var control
        wins.append(
            smp[:, :, max(0, int(back) - k + 1) : int(back) + 1, : ad.n_fields]
            .mean(0)
            .detach()
        )
        if keep_traj:
            trajs.append(ad.to_physical(smp[:, :, :, : ad.n_fields].mean(0)).cpu())
        del smp
        torch.cuda.empty_cache()
    res = {
        "rel": np.concatenate(rels),
        "analysis": (np.concatenate(ana_keep, 0) if ana_keep else None),
        "window": torch.cat(wins, 0) if wins else None,
        "cfg": {
            "gamma": gam,
            "corrections": corr,
            "n_samples": b.a.n_samples,
            "traj_back": int(back),
            "traj_len": int(L),
        },
    }
    if keep_traj:
        res["traj"] = torch.cat(trajs, 0).numpy()
    return res


def _sweep(
    b: Bench,
    out: Path,
    tuning: Dict,
    *,
    name: str,
    points: List[Dict],
    varies: str,
    methods=None,
) -> None:
    """Run every method over a list of problem variations, flushing after each point.

    ``points`` carry the kwargs that differ; everything else is held fixed, so whatever
    moves in the result is attributable to the swept quantity alone.  Results are written
    after every point because these sweeps run for hours on a shared GPU.
    """
    data = b.data(b.a.regime)
    rows, fields = [], {}
    for i, pt in enumerate(points):
        kw = dict(
            offsets=pinned_schedule(b.a), noise_std=b.a.noise, obs_frac=b.a.obs_frac
        )
        kw.update(pt["kw"])
        prob = build_problem(
            data, name=f"{name}{i}", n_problems=b.a.n_problems, seed=200 + i, **kw
        )
        row = {
            "tag": pt["tag"],
            "value": pt["value"],
            **pt.get("extra", {}),
            "problem": prob.meta(),
        }
        logger.info(f"  [{i + 1}/{len(points)}] {pt['tag']}: {pt['value']}")
        for m in methods or FOURDVAR:
            if m in DIFFUSION:
                ad = b.model(m)
                t0_ = time.perf_counter()
                _r = _solve_diffusion(b, ad, data, prob, m, out)
                v, ana, cfg = _r["rel"], _r["analysis"], _r["cfg"]
                # a diverged sampler comes back as NaN and would otherwise average to NaN
                # and sit in the results file looking like a measurement. G6_single_obs
                # spent a whole campaign that way, run with the untuned gamma = 1e-2 that
                # tune_diffusion had already recorded as divergent.
                n_bad = int((~np.isfinite(v)).sum())
                if n_bad:
                    logger.error(
                        f"        {m:9s} DIVERGED on {n_bad}/{v.size} problems "
                        f"(gamma={cfg['gamma']}, corrections={cfg['corrections']})"
                        " -- this point is NOT a measurement"
                    )
                row[m] = {
                    "mean": float(np.nanmean(v)) if n_bad < v.size else None,
                    "sem": (
                        float(np.nanstd(v, ddof=1) / np.sqrt(v.size - n_bad))
                        if v.size - n_bad > 1
                        else None
                    ),
                    "n": int(v.size),
                    "n_diverged": n_bad,
                    "wall_s": time.perf_counter() - t0_,
                    **cfg,
                }
                fields[f"{pt['tag']}__{m}__analysis"] = ana[: b.a.n_fields_saved]
                logger.info(
                    f"        {m:9s} rel-L2 {v.mean():.4f} ± "
                    f"{row[m]['sem']:.4f} [{row[m]['wall_s']:.0f}s]"
                )
                del ad
                torch.cuda.empty_cache()
                continue
            ad = b.model(m)
            r = b.solve(
                m, ad, data, prob, iters=b.a.iters, lr=tuning[m]["lr"], seed=200 + i
            )
            v = np.asarray(r["rel"])
            row[m] = {
                "mean": float(v.mean()),
                "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
                "n": int(v.size),
                "wall_s": r["wall_s"],
            }
            # keep the analysis FIELDS for the first few problems, not just the error:
            # a sweep of scalars says how much is lost, never what is lost
            nf = min(b.a.n_fields_saved, v.size)
            fields[f"{pt['tag']}__{m}__analysis"] = r["analysis"][:nf]
            logger.info(
                f"        {m:9s} rel-L2 {v.mean():.4f} ± {row[m]['sem']:.4f} "
                f"[{r['wall_s']:.0f}s]"
            )
            del ad
            torch.cuda.empty_cache()
        sim0 = torch.as_tensor(prob.sim[: b.a.n_fields_saved], device=b.dev)
        t00 = torch.as_tensor(prob.t0[: b.a.n_fields_saved], device=b.dev)
        fields[f"{pt['tag']}__truth"] = data.frames(sim0, t00).cpu().numpy()
        om = data.mask_for(sim0)
        if om is not None:
            fields[f"{pt['tag']}__mask"] = om.cpu().numpy()
        np.savez_compressed(out / f"{name}_fields.npz", **fields)
        rows.append(row)
        (out / f"{name}.json").write_text(
            json.dumps(
                {
                    "meta": {
                        "varies": varies,
                        "iters": b.a.iters,
                        "n_problems": b.a.n_problems,
                        "partial": i < len(points) - 1,
                        "pinned_delta_f": int(b.a.pin_delta_f),
                        "default_frames": pinned_schedule(b.a).tolist(),
                    },
                    "rows": rows,
                },
                indent=2,
            )
        )
    logger.info(f"saved -> {out / name}.json")


def exp_delta_f(b, out, tuning):
    """Lead to the FIRST observation, with delta_l and N pinned."""
    logger.info("=== G1. delta_f (delta_l = 25 frames, N = 5 pinned) ===")
    pts = []
    for df in b.a.delta_f:
        g = np.unique(np.round(np.geomspace(df, 25, 5)).astype(int))
        g = np.unique(np.concatenate([[df], g, [25]]))
        pts.append(
            {
                "tag": f"df{df}",
                "value": int(df),
                "kw": {"offsets": g},
                "extra": {"frames": g.tolist()},
            }
        )
    _sweep(
        b,
        out,
        tuning,
        name="G1_delta_f",
        points=pts,
        varies="delta_f",
        methods=b.a.methods,
    )


def exp_noise(b, out, tuning):
    """Observation noise, everything else fixed."""
    logger.info("=== G2. observation noise ===")
    pts = [
        {"tag": f"noise{s:g}", "value": float(s), "kw": {"noise_std": float(s)}}
        for s in b.a.noise_levels
    ]
    _sweep(
        b,
        out,
        tuning,
        name="G2_noise",
        points=pts,
        varies="noise_std",
        methods=b.a.methods,
    )


def exp_sparsity(b, out, tuning):
    """Spatial sparsity: the fraction of grid points observed."""
    logger.info("=== G3. spatial sparsity ===")
    pts = [
        {"tag": f"frac{f:g}", "value": float(f), "kw": {"obs_frac": float(f)}}
        for f in b.a.obs_fracs
    ]
    _sweep(
        b,
        out,
        tuning,
        name="G3_sparsity",
        points=pts,
        varies="obs_frac",
        methods=b.a.methods,
    )


def _geom(df, dl, n):
    """N observations geometrically spaced in [df, dl], both endpoints pinned exactly."""
    g = np.unique(np.round(np.geomspace(df, dl, n)).astype(int))
    return np.unique(np.concatenate([[df], g, [dl]]))


def pinned_schedule(a) -> np.ndarray:
    """The schedule held fixed by every sweep that does not vary delta_f.

    N = 5 from ``--pin-delta-f`` to 25, by the same geometric rule G1 uses, so the pinned
    point is also a point of the delta_f sweep.  delta_f = 1 returns ``canonical()`` itself
    so the original campaign reproduces bit for bit.
    """
    df = int(a.pin_delta_f)
    return canonical() if df == 1 else _geom(df, 25, 5)


def exp_delta_l(b, out, tuning):
    """Recovery horizon: how far the LAST observation sits, with delta_f and N pinned."""
    df = int(b.a.pin_delta_f)
    logger.info(f"=== G4. delta_l (delta_f = {df}, N = 5 pinned) ===")
    pts = []
    for dl in b.a.delta_l:
        if int(dl) <= df:
            logger.warning(f"  delta_l = {dl} is not beyond delta_f = {df}; skipped")
            continue
        g = _geom(df, int(dl), 5)
        pts.append(
            {
                "tag": f"dl{dl}",
                "value": int(dl),
                "kw": {"offsets": g},
                "extra": {"frames": g.tolist(), "N": int(len(g))},
            }
        )
    _sweep(
        b,
        out,
        tuning,
        name="G4_delta_l",
        points=pts,
        varies="delta_l",
        methods=b.a.methods,
    )


def exp_n_obs(b, out, tuning):
    """Number of observation times, with the window [delta_f, delta_l] pinned."""
    df = int(b.a.pin_delta_f)
    logger.info(f"=== G5. N (delta_f = {df}, delta_l = 25 pinned) ===")
    pts, seen = [], set()
    for n in b.a.n_obs:
        g = _geom(df, 25, int(n))
        # rounding to the frame grid merges requested times, so the REALISED N is the
        # value; a request that realises an N already run would only repeat that point
        if len(g) in seen:
            continue
        seen.add(len(g))
        pts.append(
            {
                "tag": f"n{len(g)}",
                "value": int(len(g)),
                "kw": {"offsets": g},
                "extra": {"frames": g.tolist(), "n_requested": int(n)},
            }
        )
    _sweep(b, out, tuning, name="G5_n_obs", points=pts, varies="N", methods=b.a.methods)


def exp_single_obs(b, out, tuning):
    """ONE observation, swept in distance from t_0.

    With N = 1 the three geometry parameters collapse (delta_f = delta_l = tau), so this
    is the cleanest measure of how far a lone observation can sit and still constrain the
    unobserved state.
    """
    logger.info("=== G6. a SINGLE observation, swept in distance ===")
    pts = [
        {
            "tag": f"tau{t}",
            "value": int(t),
            "kw": {"offsets": np.array([int(t)])},
            "extra": {"frames": [int(t)]},
        }
        for t in b.a.single_taus
    ]
    _sweep(
        b,
        out,
        tuning,
        name="G6_single_obs",
        points=pts,
        varies="tau, with a single observation",
        methods=b.a.methods,
    )


def exp_forecast(b, out, tuning):
    """Free-running forecast skill from the EXACT state -- model error alone.

    No assimilation and no optimisation: this is the floor that no analysis can beat, and
    it separates 'the model cannot predict this' from 'the assimilation cannot find it'.
    """
    logger.info("=== G7. free-running forecast from the EXACT state ===")
    data = b.data(b.a.regime)
    n_max = int(b.a.forecast_frames)
    rng = np.random.default_rng(9)
    nP = min(b.a.n_problems, data.n_sim)
    sim = torch.arange(nP, device=b.dev)
    t0 = torch.full((nP,), 2, device=b.dev, dtype=torch.long)
    par = data.params_for(sim)
    om = data.mask_for(sim)
    frames = np.unique(np.round(np.geomspace(1, n_max, 14)).astype(int))
    truth = torch.stack([data.frames(sim, t0 + int(f)) for f in frames])
    store = {"frames": frames, "n_problems": np.array(nP)}
    x0 = data.frames(sim, t0)
    store["persistence"] = (
        torch.stack([rel_l2(x0, truth[i], om) for i in range(len(frames))])
        .mean(1)
        .cpu()
        .numpy()
    )
    store["climatology"] = (
        torch.stack(
            [
                rel_l2(torch.zeros_like(truth[i]), truth[i], om)
                for i in range(len(frames))
            ]
        )
        .mean(1)
        .cpu()
        .numpy()
    )
    oth = data.frames(
        torch.as_tensor((sim.cpu() + nP // 2) % data.n_sim, device=b.dev),
        torch.as_tensor(rng.integers(2, data.n_t - 1, nP), device=b.dev),
    )
    store["saturation"] = (
        torch.stack([rel_l2(oth, truth[i], om) for i in range(len(frames))])
        .mean(1)
        .cpu()
        .numpy()
    )

    for m in FOURDVAR + DIFFUSION:  # every model rolls forward from the truth
        ad = b.model(m)
        with torch.no_grad():
            if m == "KAE":
                w = data.window(
                    sim, t0 - (ad.n_control_frames - 1), ad.n_control_frames
                )
                z = ad.encode(w, par)
                K = ad.generator(par)
                pred = torch.stack(
                    [
                        ad.decode(ad.propagate(z, int(f) * ad.dt_train, K=K))
                        for f in frames
                    ]
                )
            else:
                w = data.window(
                    sim, t0 - (ad.n_control_frames - 1), ad.n_control_frames
                )
                traj = ad.rollout(ad.to_model(w), n_max, par, checkpoint_every=0)
                k = ad.n_control_frames
                pred = torch.stack([traj[:, k - 1 + int(f)] for f in frames])
            e = torch.stack([rel_l2(pred[i], truth[i], om) for i in range(len(frames))])
        store[f"{m}__rel_mean"] = e.mean(1).cpu().numpy()
        store[f"{m}__rel_sem"] = (e.std(1) / np.sqrt(nP)).cpu().numpy()
        sat = store["saturation"]
        bad = np.where(store[f"{m}__rel_mean"] >= 0.9 * sat)[0]
        store[f"{m}__skill_horizon_frames"] = np.array(
            frames[bad[0]] if len(bad) else np.inf, dtype=float
        )
        logger.info(
            f"  {m:5s} @1 fr {store[f'{m}__rel_mean'][0]:.5f} | "
            f"@{n_max} fr {store[f'{m}__rel_mean'][-1]:.5f} | skill lost at "
            f"{store[f'{m}__skill_horizon_frames']} frames"
        )
        del ad
        torch.cuda.empty_cache()
    np.savez_compressed(out / "G7_forecast.npz", **store)
    logger.info(f"saved -> {out / 'G7_forecast'}.npz")


def exp_cost(b, out, tuning):
    """Per-iteration 4D-Var cost against the assimilation horizon, ONE METHOD AT A TIME.

    Interleaving the methods would share a roughly constant per-step contention and so
    inflate the cheap method far more than the expensive one.
    """
    logger.info("=== G8. cost vs horizon (isolated timing) ===")
    data = b.data(b.a.regime)
    rows = []
    for H in b.a.cost_horizons:
        prob = build_problem(
            data,
            name=f"C{H}",
            n_problems=b.a.cost_batch,
            offsets=np.array([int(H)]),
            seed=7,
        )
        row = {"horizon_frames": int(H)}
        for m in FOURDVAR:
            ad = b.model(m)
            fn = solve_kae if m == "KAE" else solve_autoregressive
            fn(ad, data, prob, iters=8, lr=tuning[m]["lr"], seed=7)  # warm-up
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn(ad, data, prob, iters=b.a.cost_iters, lr=tuning[m]["lr"], seed=7)
            torch.cuda.synchronize()
            row[m] = (time.perf_counter() - t) / b.a.cost_iters * 1e3
            del ad
            torch.cuda.empty_cache()
        # The samplers have no iteration to price, so a "ms per iteration" column would be
        # empty for them and the slide would silently be a 3-way one again.  What they do
        # have is a cost that grows with the horizon for the same structural reason: the
        # blanket score is evaluated once per window and the number of windows is L - 2.
        # Both are therefore also reported as SECONDS PER COMPLETE SOLVE, which is the
        # quantity a user actually pays and the only one that is defined for all five.
        for m in DIFFUSION:
            ad = b.model(m)
            saved = b.a.n_samples
            b.a.n_samples = 1  # one draw, so the axis is the horizon
            _solve_diffusion(b, ad, data, prob, m, out, seed=7)  # warm-up
            torch.cuda.synchronize()
            t = time.perf_counter()
            _solve_diffusion(b, ad, data, prob, m, out, seed=7)
            torch.cuda.synchronize()
            row[m] = None  # no per-iteration cost exists
            row[f"{m}__solve_s"] = time.perf_counter() - t
            b.a.n_samples = saved
            del ad
            torch.cuda.empty_cache()
        for m in FOURDVAR:  # same quantity, for comparability
            row[f"{m}__solve_s"] = row[m] * 1e-3 * b.a.iters
        logger.info(
            f"  {H:4d} frames | "
            + " | ".join(f"{m}={row[m]:8.2f} ms" for m in FOURDVAR)
            + " || "
            + " | ".join(
                f"{m}={row[f'{m}__solve_s']:6.1f} s/solve" for m in FOURDVAR + DIFFUSION
            )
        )
        rows.append(row)
    (out / "G8_cost.json").write_text(
        json.dumps(
            {
                "meta": {
                    "iters": b.a.cost_iters,
                    "batch": b.a.cost_batch,
                    "solve_iters": b.a.iters,
                    "n_samples_for_solve_s": 1,
                    "method": "one method at a time, warm-up + cuda sync",
                    "note": (
                        "`<m>` is milliseconds per 4D-Var iteration and is undefined "
                        "for the samplers (null). `<m>__solve_s` is seconds for one "
                        "COMPLETE solve and is defined for all five: for the 4D-Var "
                        "methods it is the per-iteration cost times `solve_iters`, "
                        "for the samplers it is measured directly at one draw."
                    ),
                },
                "rows": rows,
            },
            indent=2,
        )
    )
    logger.info(f"saved -> {out / 'G8_cost'}.json")


def exp_spacetime(b, out, tuning):
    """The whole recovered trajectory, not just the analysis frame.

    Each method's analysis at t_0 is rolled forward under its own dynamics and compared
    with the truth over the window, so an error at t_0 can be seen either persisting or
    being erased by the flow.
    """
    logger.info("=== G9. space-time recovery ===")
    data = b.data(b.a.regime)
    T = int(b.a.spacetime_frames)
    prob = build_problem(
        data, name="ST", n_problems=b.a.n_spacetime, offsets=canonical(), seed=100
    )
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    par, om = data.params_for(sim), data.mask_for(sim)
    truth = torch.stack([data.frames(sim, t0 + k) for k in range(T)], dim=1)
    store = {"truth": truth.cpu().numpy(), "obs_frames": prob.offsets, "T": np.array(T)}
    if om is not None:
        store["mask"] = om.cpu().numpy()
    for m in FOURDVAR + DIFFUSION:
        ad = b.model(m)
        if m in DIFFUSION:
            # the sampler is a SMOOTHER: left alone it would keep fitting the
            # observations at every frame of the window, which is not the question.
            # It is asked only for a conditioning window ending at t_0 -- the same
            # object 4D-Var optimises for U-Net/FNO -- and that window is then rolled
            # FREELY forward under ACDM's own dynamics, like every other method here.
            rr = _solve_diffusion(
                b, ad, data, prob, m, out, back=ad.n_control_frames - 1, seed=100
            )
            r = None
        else:
            r = b.solve(
                m, ad, data, prob, iters=b.a.iters, lr=tuning[m]["lr"], seed=100
            )
        with torch.no_grad():
            if m in DIFFUSION:
                k0 = ad.n_control_frames
                roll = ad.rollout(rr["window"], T - 1, par, checkpoint_every=0)[
                    :, k0 - 1 : k0 - 1 + T
                ]
            else:
                if m == "KAE":
                    z = r["control"]
                    K = ad.generator(par)
                    roll = torch.stack(
                        [
                            ad.decode(ad.propagate(z, k * ad.dt_train, K=K))
                            for k in range(T)
                        ],
                        dim=1,
                    )
                else:
                    k0 = ad.n_control_frames
                    roll = ad.rollout(r["control"], T - 1, par, checkpoint_every=0)[
                        :, k0 - 1 : k0 - 1 + T
                    ]
        store[f"{m}__roll"] = roll.cpu().numpy()
        e = torch.stack([rel_l2(roll[:, k], truth[:, k], om) for k in range(T)])
        store[f"{m}__per_frame"] = e.mean(1).cpu().numpy()
        logger.info(
            f"  {m:9s} rel-L2 t0 {float(e[0].mean()):.4f} -> "
            f"frame {T-1} {float(e[-1].mean()):.4f}"
        )
        del ad
        torch.cuda.empty_cache()
    np.savez_compressed(out / "G9_spacetime.npz", **store)
    logger.info(f"saved -> {out / 'G9_spacetime'}.npz")


def exp_post_da(b, out, tuning):
    """Forecast skill AFTER assimilation -- the operational question.

    G7 starts from the exact state and so measures the propagator alone. This starts from
    each method's OWN analysis, which is what a real system has.
    """
    logger.info("=== G10. forecast after assimilation ===")
    # A 25-frame assimilation window plus a 50-frame forecast needs 76+ frames, and the
    # interp/extrap records hold 60. Rather than quietly substituting gt_longer's answer
    # under another regime's name, the forecast horizon is cut to whatever this regime can
    # actually hold, and the cut is recorded in the output.
    data = b.data(b.a.regime)
    # build_problem needs hi = n_t - delta_l - 1 - margin to exceed t0_min, and enough
    # room above it to draw distinct analysis times. Solving that for the margin, with a
    # floor of MIN_T0 choices, is what actually fits -- an earlier version cut the horizon
    # to 32 on a 60-frame record and still left zero valid t0.
    MIN_T0 = 8
    room = data.n_t - int(canonical().max()) - 1 - 2 - MIN_T0
    n_max = int(min(b.a.forecast_frames, max(1, room)))
    if n_max < b.a.forecast_frames:
        logger.warning(
            f"  {b.a.regime}: record has {data.n_t} frames, so the post-DA "
            f"forecast horizon is cut {b.a.forecast_frames} -> {n_max}"
        )
    prob = build_problem(
        data,
        name="PF",
        n_problems=b.a.n_problems,
        offsets=canonical(),
        seed=71,
        margin=n_max,
    )
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    par, om = data.params_for(sim), data.mask_for(sim)
    frames = np.unique(np.round(np.geomspace(1, n_max, 12)).astype(int))
    truth = torch.stack([data.frames(sim, t0 + int(f)) for f in frames])
    store = {
        "frames": frames,
        "regime": np.array(DATA[b.a.regime]),
        "forecast_frames_requested": np.array(int(b.a.forecast_frames)),
        "forecast_frames_used": np.array(int(n_max)),
    }
    for m in FOURDVAR + DIFFUSION:
        ad = b.model(m)
        if m in DIFFUSION:
            rr = _solve_diffusion(
                b, ad, data, prob, m, out, back=ad.n_control_frames - 1, seed=71
            )
            ana_rel, r = np.asarray(rr["rel"]), None
        else:
            r = b.solve(m, ad, data, prob, iters=b.a.iters, lr=tuning[m]["lr"], seed=71)
            ana_rel = np.asarray(r["rel"])
        with torch.no_grad():
            if m in DIFFUSION:
                k0 = ad.n_control_frames
                tr = ad.rollout(rr["window"], n_max, par, checkpoint_every=0)
                pred = torch.stack([tr[:, k0 - 1 + int(f)] for f in frames])
            else:
                if m == "KAE":
                    K = ad.generator(par)
                    pred = torch.stack(
                        [
                            ad.decode(
                                ad.propagate(r["control"], int(f) * ad.dt_train, K=K)
                            )
                            for f in frames
                        ]
                    )
                else:
                    k0 = ad.n_control_frames
                    tr = ad.rollout(r["control"], n_max, par, checkpoint_every=0)
                    pred = torch.stack([tr[:, k0 - 1 + int(f)] for f in frames])
            e = torch.stack([rel_l2(pred[i], truth[i], om) for i in range(len(frames))])
        store[f"{m}__rel_mean"] = e.mean(1).cpu().numpy()
        store[f"{m}__rel_sem"] = (
            (e.std(1, correction=1) / np.sqrt(e.shape[1])).cpu().numpy()
        )
        store[f"{m}__analysis_rel"] = np.array(float(ana_rel.mean()))
        logger.info(
            f"  {m:9s} analysis {float(ana_rel.mean()):.4f} | "
            f"@{frames[-1]} fr {float(e[-1].mean()):.4f}"
        )
        del ad
        torch.cuda.empty_cache()
    np.savez_compressed(out / "G10_post_da.npz", **store)
    logger.info(f"saved -> {out / 'G10_post_da'}.npz")


def exp_calibration(b, out, tuning):
    """Is ACDM's posterior calibrated, or just accurate on average?

    ACDM is a SAMPLER: reporting only its mean error hides whether its spread is
    meaningful.  A calibrated posterior puts the truth inside its own credible interval at
    the stated rate.  Here many draws are taken per problem and the truth's rank among
    them is histogrammed (a rank histogram): flat means calibrated, U-shaped means
    over-confident, dome-shaped means under-confident.
    """
    from data_assimilation.tra.sda_blanket import ACDMBlanketScore, BlanketSDA

    logger.info("=== G11. posterior calibration of the score-based methods ===")
    data = b.data(b.a.regime)
    prob = build_problem(
        data,
        name="CAL",
        n_problems=b.a.n_cal,
        offsets=canonical(),
        seed=300,
        noise_std=b.a.noise,
    )
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    truth, om = data.frames(sim, t0), data.mask_for(sim)
    td = json.loads((out / "tuning_diffusion.json").read_text())
    store = {}
    for name in DIFFUSION:
        ad = b.model(name)
        C = ad.n_fields + ad.n_params
        sc = ACDMBlanketScore(ad, C)
        sda = BlanketSDA(
            sc,
            sigma_y=b.a.sigma_y_floor,
            gamma=td[name]["gamma"],
            corrections=td[name]["corrections"],
            tau=b.a.tau,
        )
        L = max(int(prob.offsets.max()) + 1, sc.window)
        draws = []
        for lo in range(0, len(prob.sim), b.a.chunk):
            hi = min(lo + b.a.chunk, len(prob.sim))
            sl = slice(lo, hi)
            y = ad.to_model(torch.as_tensor(prob.y[:, sl], device=b.dev)).transpose(
                0, 1
            )
            msk = (
                torch.as_tensor(prob.mask[:, sl], device=b.dev)
                .transpose(0, 1)
                .transpose(-1, -2)
            )
            par = data.params_for(sim[sl])
            H, W = y.shape[-2], y.shape[-1]
            N = len(prob.offsets)
            pch = ad.param_channel(par, (H, W))
            yf = torch.cat([y, pch.unsqueeze(1).expand(-1, N, -1, -1, -1)], dim=2)
            mf = torch.cat(
                [
                    msk.expand(-1, -1, ad.n_fields, -1, -1),
                    torch.ones_like(pch).unsqueeze(1).expand(-1, N, -1, -1, -1),
                ],
                dim=2,
            )
            smp = sda.sample(
                (hi - lo, L, C, H, W),
                torch.as_tensor(prob.offsets, device=b.dev),
                yf,
                mf,
                seed=300 + lo,
                n_samples=b.a.n_cal_draws,
            )
            draws.append(ad.to_physical(smp[:, :, 0, : ad.n_fields]).cpu())
        ens = torch.cat(draws, dim=1)  # [S, B, C, H, W]
        store[f"{name}__ensemble_std"] = ens.std(0).mean().numpy()
        # rank of the truth among the draws, per observed grid point
        t_cpu = truth.cpu()
        m_cpu = None if om is None else om.cpu()
        below = (ens < t_cpu.unsqueeze(0)).sum(0).float()  # [B,C,H,W] in 0..S
        if m_cpu is not None:
            below = below[m_cpu.unsqueeze(1).expand_as(below) > 0]
        store[f"{name}__ranks"] = below.flatten().numpy().astype(np.int16)
        per = torch.stack(
            [rel_l2(ens[i].to(b.dev), truth, om) for i in range(ens.shape[0])]
        )
        store[f"{name}__rel_per_draw"] = per.cpu().numpy()
        store[f"{name}__rel_ens_mean"] = (
            rel_l2(ens.mean(0).to(b.dev), truth, om).cpu().numpy()
        )
        logger.info(
            f"  {name:9s} per-draw {float(per.mean()):.4f} | ensemble-mean "
            f"{float(store[f'{name}__rel_ens_mean'].mean()):.4f} | spread "
            f"{float(store[f'{name}__ensemble_std']):.4f}"
        )
        del ad, sc, sda
        torch.cuda.empty_cache()
    store["n_draws"] = np.array(b.a.n_cal_draws)
    np.savez_compressed(out / "G11_calibration.npz", **store)
    logger.info(f"saved -> {out / 'G11_calibration'}.npz")


def exp_offgrid(b, out, tuning):
    """Observations at times BETWEEN stored frames.

    The KAE evaluates e^{K tau} at any real tau; every other method here can only land on
    integer multiples of its training step and must snap the observation to the nearest
    one, assimilating it as though it had been taken up to half a frame away.

    A note on what is and is not measured.  The record exists only on the integer grid, so
    the state at t_0 + tau + shift is LINEARLY INTERPOLATED between the two bracketing
    frames.  That is an approximation to the true sub-frame state, and it is a generous one
    for everybody: the interpolant is closer to both neighbours than the truth is.  The
    size of the sub-frame motion is written out as ``interp_gap_rel`` so the reader can see
    how much room there was for snapping to hurt at all.  An earlier version of this
    experiment applied no shift whatsoever and reported three identical rows.
    """
    logger.info("=== G12. off-grid observation times ===")
    data = b.data(b.a.regime)
    rows = []
    for shift in b.a.offgrid_shifts:
        offs_real = canonical().astype(float) + float(shift)
        prob = build_problem(
            data,
            name=f"OG{shift:g}",
            n_problems=b.a.n_problems,
            offsets_real=offs_real,
            seed=400,
        )
        snapped = prob.offsets.astype(float)
        row = {
            "shift": float(shift),
            "offsets_real": [float(x) for x in offs_real],
            "offsets_snapped": prob.offsets.tolist(),
            "snap_error_frames": [
                abs(float(a - c)) for a, c in zip(offs_real, snapped)
            ],
            "interp_gap_rel": [float(x) for x in prob.extras.get("interp_gap_rel", [])],
        }
        logger.info(
            f"  shift = {shift:+.2f} frames  (real {offs_real.tolist()} -> "
            f"snapped {prob.offsets.tolist()})"
        )
        for m in FOURDVAR + DIFFUSION:
            ad = b.model(m)
            exact = m == "KAE"  # only the KAE reads prob.offsets_eval
            if m in DIFFUSION:
                v = _solve_diffusion(b, ad, data, prob, m, out, seed=400)["rel"]
                wall = None
            else:
                r = b.solve(
                    m, ad, data, prob, iters=b.a.iters, lr=tuning[m]["lr"], seed=400
                )
                v = np.asarray(r["rel"])
                wall = r["wall_s"]
            row[m] = {
                "mean": float(v.mean()),
                "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
                "n": int(v.size),
                "wall_s": wall,
                "evaluates_exactly": bool(exact),
                "time_error_frames": (
                    0.0 if exact else float(np.abs(offs_real - snapped).max())
                ),
            }
            logger.info(
                f"        {m:9s} rel-L2 {v.mean():.4f}"
                f"{'' if exact else f'  (snapped, |dt| <= {float(np.abs(offs_real - snapped).max()):.2f} frames)'}"
            )
            del ad
            torch.cuda.empty_cache()
        rows.append(row)
        (out / "G12_offgrid.json").write_text(
            json.dumps(
                {
                    "meta": {
                        "note": (
                            "Observations are placed at real lead times tau + shift and "
                            "generated by LINEAR INTERPOLATION between the bracketing "
                            "frames. The KAE evaluates e^{K tau} there exactly; U-Net, "
                            "FNO and the samplers snap to the nearest stored frame. "
                            "`interp_gap_rel` is the relative change between the two "
                            "bracketing frames, i.e. how much sub-frame motion there "
                            "is for the snapping to get wrong."
                        ),
                        "iters": b.a.iters,
                        "n_problems": b.a.n_problems,
                        "partial": len(rows) < len(b.a.offgrid_shifts),
                    },
                    "rows": rows,
                },
                indent=2,
            )
        )
    logger.info(f"saved -> {out / 'G12_offgrid'}.json")


def exp_backward(b, out, tuning):
    """Recover a WINDOW of states ending at t_0, not just t_0 itself.

    All observations still lie strictly after t_0, so every frame at t_0 - j must be
    inferred backwards through the dynamics.  The methods differ structurally in whether
    they can do this at all:

      KAE       one latent z_0 generates the whole window through e^{-K j dt}: the
                backward direction costs nothing extra and is exact.
      ACDM      as run here it is a SMOOTHER, not an autoregressive model: the control is
                the whole sampled trajectory, so extending it to t_0-j costs only the
                extra blanket windows.  Its reach is therefore unbounded too, for a
                different reason than the KAE's.  (Used as an autoregressive propagator it
                would be limited to its `prevSteps` conditioning frames, which is what an
                earlier version of this slide claimed; that is not how it is run.)
      U-Net/FNO their control is a single frame (prevSteps = 1), so a state before t_0 is
                not representable at all -- it is not that they recover it badly, they
                cannot express it.

    Reporting error at t_0 alone hides this entirely.
    """
    logger.info("=== G13. recovering a WINDOW of states ending at t_0 ===")
    data = b.data(b.a.regime)
    W = int(b.a.back_frames)
    prob = build_problem(
        data,
        name="BK",
        n_problems=b.a.n_problems,
        offsets=canonical(),
        seed=500,
        t0_min=W + 2,
    )
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    par, om = data.params_for(sim), data.mask_for(sim)
    # truth at t_0 - j for j = 0 .. W-1
    truth = torch.stack([data.frames(sim, t0 - j) for j in range(W)])
    store = {"offsets_back": np.arange(W), "obs_frames": prob.offsets}
    rows = []
    for m in FOURDVAR + DIFFUSION:
        ad = b.model(m)
        k = ad.n_control_frames
        if m in DIFFUSION:
            # ONE solve, with the trajectory extended to cover t_0-(W-1); the sampled
            # trajectory IS the control, so every past frame comes out of the same
            # assimilation rather than a re-solve (that is exp_reach's question).
            rr = _solve_diffusion(
                b, ad, data, prob, m, out, back=W - 1, keep_traj=True, seed=500
            )
            r = None
        else:
            r = b.solve(
                m, ad, data, prob, iters=b.a.iters, lr=tuning[m]["lr"], seed=500
            )
        with torch.no_grad():
            if m in DIFFUSION:
                tj = torch.as_tensor(rr["traj"], device=b.dev)  # [B, L, C, H, W]
                back = torch.stack([tj[:, (W - 1) - j] for j in range(W)])
                repr_ = W  # the whole window is in the control
            elif m == "KAE":
                K = ad.generator(par)
                back = torch.stack(
                    [
                        ad.decode(ad.propagate(r["control"], -j * ad.dt_train, K=K))
                        for j in range(W)
                    ]
                )
                repr_ = W  # every past frame is representable
            else:
                phys = ad.to_physical(r["control"])  # [B, k, C, H, W]
                repr_ = min(k, W)
                back = torch.stack(
                    [phys[:, k - 1 - j] for j in range(repr_)]
                    + [
                        torch.full_like(phys[:, 0], float("nan"))
                        for _ in range(W - repr_)
                    ]
                )
            e = torch.stack([rel_l2(back[j], truth[j], om) for j in range(W)])
        store[f"{m}__back"] = back.cpu().numpy()
        store[f"{m}__err"] = e.mean(1).cpu().numpy()
        store[f"{m}__n_representable"] = np.array(repr_)
        rows.append(
            {
                "method": m,
                "n_representable": int(repr_),
                "err": [float(x) for x in e.mean(1).cpu().numpy()],
            }
        )
        logger.info(
            f"  {m:9s} representable {repr_}/{W} frames | "
            + " ".join(f"t0-{j}:{float(e[j].mean()):.4f}" for j in range(min(repr_, 4)))
        )
        del ad
        torch.cuda.empty_cache()
    store["truth"] = truth.cpu().numpy()
    if om is not None:
        store["mask"] = om.cpu().numpy()
    np.savez_compressed(out / "G13_backward.npz", **store)
    (out / "G13_backward.json").write_text(
        json.dumps(
            {
                "meta": {
                    "window": W,
                    "iters": b.a.iters,
                    "n_problems": b.a.n_problems,
                    "note": (
                        "All observations are strictly after t_0. A frame at t_0-j is "
                        "representable only if the method's control spans it. The KAE "
                        "generates the whole window from one latent via e^{-K j dt}. "
                        "The score-based methods are run as SMOOTHERS -- the control "
                        "is the sampled trajectory -- so extending it backwards costs "
                        "only extra blanket windows and their reach is unbounded too. "
                        "U-Net and FNO optimise a single frame AT t_0 and cannot "
                        "express anything earlier at all."
                    ),
                },
                "rows": rows,
            },
            indent=2,
        )
    )
    logger.info(f"saved -> {out / 'G13_backward'}.npz/.json")


def exp_reach(b, out, tuning):
    """How far back can each method recover, given a FAIR chance?

    ``exp_backward`` asks whether one assimilation determines a whole window, which
    favours the KAE by construction (its single latent generates any t_0-j, while the
    U-Net's control IS the frame at t_0 and cannot express anything earlier).

    This is the fair counterpart: for each j the analysis time is MOVED to t_0-j and every
    method is re-solved from scratch, with the observations left where they are.  Now
    every method can attempt every target, and the comparison is of accuracy rather than
    of representational reach.
    """
    logger.info("=== G18. reach, with every method re-solved at each target ===")
    data = b.data(b.a.regime)
    rows = []
    for j in b.a.reach_back:
        offs = canonical() + int(j)  # observations fixed; analysis time moves back
        prob = build_problem(
            data,
            name=f"RH{j}",
            n_problems=b.a.n_problems,
            offsets=offs,
            seed=700,
            t0_min=int(j) + 3,
        )
        row = {"frames_back": int(j), "offsets": offs.tolist()}
        logger.info(f"  target t0-{j} (observations at {offs.tolist()})")
        for m in b.a.methods or FOURDVAR + DIFFUSION:
            ad = b.model(m)
            if m in DIFFUSION:
                v = _solve_diffusion(b, ad, data, prob, m, out, seed=700)["rel"]
            else:
                r = b.solve(
                    m, ad, data, prob, iters=b.a.iters, lr=tuning[m]["lr"], seed=700
                )
                v = np.asarray(r["rel"])
            row[m] = {
                "mean": float(np.nanmean(v)),
                "sem": float(np.nanstd(v, ddof=1) / np.sqrt(v.size)),
            }
            logger.info(f"        {m:9s} rel-L2 {row[m]['mean']:.4f}")
            del ad
            torch.cuda.empty_cache()
        rows.append(row)
    (out / "G18_reach.json").write_text(
        json.dumps(
            {
                "meta": {
                    "iters": b.a.iters,
                    "n_problems": b.a.n_problems,
                    "note": (
                        "The analysis time is moved to t_0-j and every method is "
                        "re-solved; observations stay put, so the lead to the first "
                        "observation grows with j. Unlike G13 this asks about "
                        "ACCURACY, not representational reach."
                    ),
                },
                "rows": rows,
            },
            indent=2,
        )
    )
    logger.info(f"saved -> {out / 'G18_reach'}.json")


def exp_window(b, out, tuning):
    """Sweep how many frames the KAE control is asked to reconstruct backwards.

    Only the KAE can vary this: its single latent generates an arbitrarily long window.
    The question is whether accuracy degrades as the window is extended backwards, i.e.
    how far back one latent can faithfully carry the state.
    """
    logger.info("=== G14. how far back can one latent reach? ===")
    data = b.data(b.a.regime)
    Wmax = int(max(b.a.window_sweep))
    prob = build_problem(
        data,
        name="WS",
        n_problems=b.a.n_problems,
        offsets=canonical(),
        seed=501,
        t0_min=Wmax + 2,
    )
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    par, om = data.params_for(sim), data.mask_for(sim)
    ad = b.model("KAE")
    r = b.solve(
        "KAE", ad, data, prob, iters=b.a.iters, lr=tuning["KAE"]["lr"], seed=501
    )
    K = ad.generator(par)
    rows = []
    for j in b.a.window_sweep:
        with torch.no_grad():
            pred = ad.decode(ad.propagate(r["control"], -int(j) * ad.dt_train, K=K))
            e = rel_l2(pred, data.frames(sim, t0 - int(j)), om)
        rows.append(
            {
                "frames_back": int(j),
                "mean": float(e.mean()),
                "sem": float(e.std(correction=1) / np.sqrt(e.numel())),
            }
        )
        logger.info(f"  t0 - {j:3d} frames: rel-L2 {float(e.mean()):.4f}")
    (out / "G14_window.json").write_text(
        json.dumps(
            {
                "meta": {
                    "method": "KAE",
                    "iters": b.a.iters,
                    "note": "one latent propagated backwards; no re-optimisation",
                },
                "rows": rows,
            },
            indent=2,
        )
    )
    logger.info(f"saved -> {out / 'G14_window'}.json")


def exp_budget(b, out, tuning):
    """Accuracy against a WALL-CLOCK budget, which is the comparison a user actually faces.

    The headline gives each method a fixed iteration count, which is not a fair basis for
    choosing between them: KAE takes 43 s and ACDM 138 s.  Here every method is run at a
    range of budgets and plotted against the time it actually spent, so the question
    "which is best if I have N seconds?" has an answer.
    """
    logger.info("=== G15. accuracy vs wall-clock budget ===")
    data = b.data(b.a.regime)
    prob = build_problem(
        data, name="BG", n_problems=b.a.n_problems, offsets=canonical(), seed=600
    )
    rows = []
    for it in b.a.budget_iters:
        for m in FOURDVAR:
            ad = b.model(m)
            r = b.solve(m, ad, data, prob, iters=int(it), lr=tuning[m]["lr"], seed=600)
            v = np.asarray(r["rel"])
            rows.append(
                {
                    "method": m,
                    "iters": int(it),
                    "wall_s": r["wall_s"],
                    "mean": float(v.mean()),
                    "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
                }
            )
            logger.info(
                f"  {m:5s} {it:5d} iters  {r['wall_s']:7.1f}s  "
                f"rel-L2 {v.mean():.4f}"
            )
            del ad
            torch.cuda.empty_cache()
    # the samplers' knob is the number of draws, not iterations
    for n_draw in b.a.budget_draws:
        for m in DIFFUSION:
            ad = b.model(m)
            t_start = time.perf_counter()
            saved = b.a.n_samples
            b.a.n_samples = int(n_draw)
            v = _solve_diffusion(b, ad, data, prob, m, out)["rel"]
            b.a.n_samples = saved
            rows.append(
                {
                    "method": m,
                    "draws": int(n_draw),
                    "wall_s": time.perf_counter() - t_start,
                    "mean": float(np.nanmean(v)),
                    "sem": float(np.nanstd(v, ddof=1) / np.sqrt(v.size)),
                }
            )
            logger.info(
                f"  {m:9s} {n_draw:3d} draws  {rows[-1]['wall_s']:7.1f}s  "
                f"rel-L2 {rows[-1]['mean']:.4f}"
            )
            del ad
            torch.cuda.empty_cache()
    (out / "G15_budget.json").write_text(
        json.dumps(
            {
                "meta": {
                    "n_problems": b.a.n_problems,
                    "note": (
                        "4D-Var methods are swept over iterations, samplers over the "
                        "number of draws; both are plotted against measured wall "
                        "clock so they can be compared at equal cost."
                    ),
                },
                "rows": rows,
            },
            indent=2,
        )
    )
    logger.info(f"saved -> {out / 'G15_budget'}.json")


def exp_background(b, out, tuning):
    """Give the physical-space methods the background term real 4D-Var always has.

        J(c) = observation term + w * || c - c_b ||^2

    The U-Net and FNO analyses come out as speckle because nothing forbids it: the
    dynamics damp grid-scale error before the first observation, so the cost cannot see
    it.  Operational 4D-Var adds a background term for exactly this reason.  Without it
    the comparison flatters the KAE and ACDM, whose parameterisations regularise for
    free.  ``c_b`` is climatology (zero in normalised units), so no truth is leaked.
    """
    logger.info(
        "=== G16. does a background term rescue the physical-space methods? ==="
    )
    data = b.data(b.a.regime)
    prob = build_problem(
        data, name="BG2", n_problems=b.a.n_problems, offsets=canonical(), seed=601
    )
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    par, om = data.params_for(sim), data.mask_for(sim)
    truth = data.frames(sim, t0)
    y = torch.as_tensor(prob.y, device=b.dev)
    mask = torch.as_tensor(prob.mask, device=b.dev)
    rows = []
    for m in ["UNet", "FNO"]:
        ad = b.model(m)
        k = ad.n_control_frames
        n_steps = int(prob.offsets.max())
        idx = torch.as_tensor(prob.offsets + (k - 1), device=b.dev)
        for w in b.a.background_weights:
            torch.manual_seed(601)
            c = torch.zeros_like(ad.to_model(data.window(sim, t0 - (k - 1), k)))
            c.requires_grad_(True)
            opt = torch.optim.Adam([c], lr=tuning[m]["lr"])
            t_start = time.perf_counter()
            for _ in range(b.a.iters):
                opt.zero_grad(set_to_none=True)
                traj = ad.rollout(c, n_steps, par)
                pred = traj[:, idx].transpose(0, 1)
                obs_term = ((pred - y) * mask).pow(2).mean()
                # background: penalise departure from climatology (c_b = 0 normalised)
                bg = c.pow(2).mean()
                (obs_term + float(w) * bg).backward()
                opt.step()
            with torch.no_grad():
                a = ad.to_physical(c)[:, -1]
                e = rel_l2(a, truth, om)
                # how rough is the analysis? high-wavenumber energy fraction
                F = torch.fft.rfft2(a.float())
                hi = F[..., F.shape[-2] // 4 :, F.shape[-1] // 4 :].abs().pow(2).sum()
                rough = float(hi / F.abs().pow(2).sum())
            rows.append(
                {
                    "method": m,
                    "weight": float(w),
                    "mean": float(e.mean()),
                    "sem": float(e.std(correction=1) / np.sqrt(e.numel())),
                    "high_k_fraction": rough,
                    "wall_s": time.perf_counter() - t_start,
                }
            )
            logger.info(
                f"  {m:5s} w={w:<8g} rel-L2 {float(e.mean()):.4f}  "
                f"high-k energy {rough:.4f}"
            )
        del ad
        torch.cuda.empty_cache()
    (out / "G16_background.json").write_text(
        json.dumps(
            {
                "meta": {
                    "iters": b.a.iters,
                    "n_problems": b.a.n_problems,
                    "background": "climatology (zero in normalised units); no truth used",
                    "note": (
                        "high_k_fraction is the share of spectral energy above a "
                        "quarter Nyquist -- a direct measure of how much speckle the "
                        "analysis carries."
                    ),
                },
                "rows": rows,
            },
            indent=2,
        )
    )
    logger.info(f"saved -> {out / 'G16_background'}.json")


def exp_corrector(b, out, tuning):
    """Is Algorithm 4's Langevin corrector genuinely harmful here, or just mis-scaled?

    Validation chose 0 correctors because 1 corrector made ACDM 4.6-7.6x worse.  The
    corrector's step is delta = tau * dim(s) / ||s||^2, so a large guided score collapses
    delta and leaves only the injected noise.  Sweeping tau tests that explanation.
    """
    from data_assimilation.tra.sda_blanket import ACDMBlanketScore, BlanketSDA

    logger.info("=== G17. Langevin corrector: harmful, or mis-scaled? ===")
    data = b.data("val")
    prob = build_problem(
        data, name="CR", n_problems=b.a.n_tune_diff, offsets=canonical(), seed=602
    )
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    truth, om = data.frames(sim, t0), data.mask_for(sim)
    td = json.loads((out / "tuning_diffusion.json").read_text())
    rows = []
    ad = b.model("ACDM")
    C = ad.n_fields + ad.n_params
    sc = ACDMBlanketScore(ad, C)
    L = max(int(prob.offsets.max()) + 1, sc.window)
    y = ad.to_model(torch.as_tensor(prob.y, device=b.dev)).transpose(0, 1)
    msk = torch.as_tensor(prob.mask, device=b.dev).transpose(0, 1).transpose(-1, -2)
    par = data.params_for(sim)
    H, W = y.shape[-2], y.shape[-1]
    N = len(prob.offsets)
    pch = ad.param_channel(par, (H, W))
    yf = torch.cat([y, pch.unsqueeze(1).expand(-1, N, -1, -1, -1)], dim=2)
    mf = torch.cat(
        [
            msk.expand(-1, -1, ad.n_fields, -1, -1),
            torch.ones_like(pch).unsqueeze(1).expand(-1, N, -1, -1, -1),
        ],
        dim=2,
    )
    obs = torch.as_tensor(prob.offsets, device=b.dev)
    for corr in (0, 1, 2):
        for tau in b.a.corrector_taus:
            if corr == 0 and tau != b.a.corrector_taus[0]:
                continue  # tau is irrelevant with no corrector
            sda = BlanketSDA(
                sc,
                sigma_y=b.a.sigma_y_floor,
                gamma=td["ACDM"]["gamma"],
                corrections=corr,
                tau=float(tau),
            )
            smp = sda.sample(
                (len(prob.sim), L, C, H, W), obs, yf, mf, seed=602, n_samples=2
            )
            fin = bool(torch.isfinite(smp).all())
            e = (
                float(
                    rel_l2(
                        ad.to_physical(smp[0, :, 0, : ad.n_fields]), truth, om
                    ).mean()
                )
                if fin
                else float("nan")
            )
            rows.append(
                {"corrections": corr, "tau": float(tau), "rel": e, "diverged": not fin}
            )
            logger.info(
                f"  corrections={corr} tau={tau:<5g} "
                + ("DIVERGED" if not fin else f"rel-L2 {e:.4f}")
            )
    (out / "G17_corrector.json").write_text(
        json.dumps(
            {
                "meta": {
                    "chosen_in_campaign": td["ACDM"],
                    "note": (
                        "Algorithm 4 specifies a predictor-corrector. Validation "
                        "selected 0 correctors; this sweep asks whether the corrector "
                        "is harmful in principle or simply mis-scaled at tau=0.5."
                    ),
                },
                "rows": rows,
            },
            indent=2,
        )
    )
    logger.info(f"saved -> {out / 'G17_corrector'}.json")


def _solve_all(b, out, tuning, data, prob, seed):
    """Every method on one prepared Problem -> {method: per-problem rel-L2 array}.

    Only the per-problem errors are kept. The solve dictionaries carry the control and the
    analysis as GPU tensors, and at the larger problem counts this section uses, holding
    one of those across the next method's allocation is enough to exhaust the card -- the
    first attempt at `regimes` died with 23.4 GB in use, in ACDM's group-norm. Each result
    is reduced to a numpy array and dropped, and the allocator is emptied, before the next
    model is built.
    """
    import gc

    res = {}
    nB = int(b.a.solve_batch) or len(prob.sim)
    for m in FOURDVAR + DIFFUSION:
        ad = b.model(m)
        t0_ = time.perf_counter()
        if m in DIFFUSION:
            # the sampler already chunks internally, via --chunk
            v = np.asarray(
                _solve_diffusion(b, ad, data, prob, m, out, seed=seed)["rel"],
                dtype=float,
            ).copy()
        else:
            # 4D-Var builds ONE rollout graph for the whole batch, so its memory grows
            # linearly in the problem count: the U-Net at 24 problems x a 25-step rollout
            # needed 18 GB and OOMed beside another user's job. Each problem is
            # independent -- its own control, and a cost that is a mean over problems --
            # so batching and concatenating gives the same answer at bounded memory.
            parts = []
            for lo in range(0, len(prob.sim), nB):
                sub = prob.subset(np.arange(lo, min(lo + nB, len(prob.sim))))
                r = b.solve(
                    m, ad, data, sub, iters=b.a.iters, lr=tuning[m]["lr"], seed=seed
                )
                parts.append(np.asarray(r["rel"], dtype=float).copy())
                del r
                gc.collect()
                torch.cuda.empty_cache()
            v = np.concatenate(parts)
        res[m] = (v, time.perf_counter() - t0_)
        del ad
        gc.collect()
        torch.cuda.empty_cache()
    return res


def exp_confounded(b, out, tuning):
    """The same question asked the CONFOUNDED way, to show what separating it buys.

    The natural way to ask "how much do observations help" is to sweep one knob -- observe
    every k-th frame -- and read off the curve.  That knob moves delta_f, delta_l and N
    **all at once**: at k = 1 the first observation is at frame 1 and there are 25 of them;
    at k = 12 the first is at frame 12 and there are 2.  Any curve it produces is a mixture,
    and attributing it to "more observations" is a guess.

    This runs that confounded sweep deliberately, so it can be laid beside the separated
    sweeps (G1 delta_f, G4 delta_l, G5 N) that hold the other two fixed. The comparison is
    the point: if the confounded curve tracks the delta_f sweep and not the N sweep, then
    what looked like an observation-count effect was a lead-time effect all along.
    """
    logger.info(
        "=== G23. the confounded sweep (delta_f, delta_l and N move together) ==="
    )
    pts = []
    for k in b.a.confound_every:
        offs = np.arange(int(k), 26, int(k), dtype=int)
        if offs.size == 0:
            continue
        pts.append(
            {
                "tag": f"every{k}",
                "value": int(k),
                "kw": {"offsets": offs},
                "extra": {
                    "observe_every": int(k),
                    "frames": offs.tolist(),
                    "delta_f": int(offs.min()),
                    "delta_l": int(offs.max()),
                    "N": int(offs.size),
                },
            }
        )
    _sweep(
        b,
        out,
        tuning,
        name="G23_confounded",
        points=pts,
        varies=(
            "observe every k-th frame -- delta_f, delta_l and N all move together, "
            "which is exactly what G1/G4/G5 separate"
        ),
        methods=b.a.methods,
    )


def exp_convergence(b, out, tuning):
    """Is the 4D-Var baselines' poor analysis an unconverged optimiser?

    The whole claim rests on U-Net's 0.2146 being a property of the cost surface rather
    than of the iteration budget.  `budget` gives indirect evidence -- U-Net is *worse* at
    2000 iterations than at 500 -- but indirect is not the same as watching it.  This
    tracks the objective and the analysis error against iteration for every 4D-Var method
    on the canonical problem, so a reader can see whether the curve has flattened.
    """
    logger.info("=== G24. convergence of the 4D-Var methods ===")
    data = b.data(b.a.regime)
    prob = build_problem(
        data, name="CV", n_problems=b.a.n_problems, offsets=canonical(), seed=100
    )
    store = {"iters": np.array(b.a.iters), "regime": np.array(DATA[b.a.regime])}
    for m in FOURDVAR:
        ad = b.model(m)
        r = b.solve(
            m,
            ad,
            data,
            prob,
            iters=b.a.iters,
            lr=tuning[m]["lr"],
            seed=100,
            track_every=max(1, b.a.iters // 60),
        )
        h = r["hist"]
        store[f"{m}__iter"] = np.asarray(h["iter"])
        store[f"{m}__loss"] = np.asarray(h["loss"])
        store[f"{m}__rel_t0"] = np.asarray(h["rel_t0"])
        rel = np.asarray(h["rel_t0"])
        # "converged" here means the last fifth of the run moved the analysis error by
        # less than 2% -- a weak criterion on purpose, so failing it is meaningful
        tail = rel[max(0, len(rel) - len(rel) // 5) :]
        drift = (
            float(abs(tail[-1] - tail[0]) / max(tail[0], 1e-12))
            if len(tail) > 1
            else 0.0
        )
        store[f"{m}__tail_drift"] = np.array(drift)
        store[f"{m}__best_rel"] = np.array(float(rel.min()))
        store[f"{m}__final_rel"] = np.array(float(rel[-1]))
        logger.info(
            f"  {m:5s} rel-L2 {rel[0]:.4f} -> {rel[-1]:.4f} "
            f"(best {rel.min():.4f} at iter {int(h['iter'][int(rel.argmin())])}), "
            f"last-fifth drift {100 * drift:.1f}%"
        )
        del ad
        torch.cuda.empty_cache()
    np.savez_compressed(out / "G24_convergence.npz", **store)
    logger.info(f"saved -> {out / 'G24_convergence'}.npz")


def exp_gap(b, out, tuning):
    """Recovery against the GAP to the nearest observation, with IRREGULAR RANDOM schedules.

    Every other experiment here uses `canonical()` = [1, 3, 7, 15, 25] or a deterministic
    geometric variant of it. That schedule is irregular in spacing but it is FIXED: the
    same five lead times in every problem, every sweep, every regime. Two things follow
    that this experiment exists to remove.

    First, no result so far separates "ACDM is good" from "ACDM is good on THIS schedule".
    Here the observation times are drawn at random for each schedule replicate, so the
    schedule becomes a nuisance variable that is averaged over instead of a constant.

    Second, the win/loss record across the campaign separates on one quantity -- the lead
    to the FIRST observation, i.e. the gap between the target and the nearest thing that
    constrains it. Every ACDM win in the campaign has that gap equal to 1. That is a
    hypothesis generated post hoc from a fixed schedule, so it needs its own experiment
    with the gap as the swept variable and the rest of the schedule randomised.

    Observations are strictly after the target because U-Net and FNO cannot represent a
    state before their conditioning frame at all (see exp_backward). That is a constraint
    of the baselines, not a choice: an interior target would be a different, easier
    problem that only three of the five methods could attempt.
    """
    logger.info(
        "=== G22. gap to the nearest observation, random irregular schedules ==="
    )
    from data_assimilation.tra.stats import summarise, fmt

    data = b.data(b.a.regime)
    rows = []
    room = data.n_t - 4
    for gap in b.a.gaps:
        gap = int(gap)
        per_sched = {m: [] for m in FOURDVAR + DIFFUSION}
        scheds = []
        for si in range(int(b.a.n_schedules)):
            rng = np.random.default_rng(3000 + 97 * gap + si)
            # a random irregular set: the nearest observation sits exactly at `gap`, the
            # rest are drawn without replacement from the remaining reachable frames
            hi = min(int(b.a.gap_max_lead), room - gap)
            pool = np.arange(gap + 1, gap + 1 + max(1, hi))
            k = min(int(b.a.n_obs_gap) - 1, len(pool))
            offs = np.sort(
                np.concatenate([[gap], rng.choice(pool, size=k, replace=False)])
            )
            scheds.append(offs.tolist())
            prob = build_problem(
                data,
                name=f"GAP{gap}_{si}",
                n_problems=b.a.n_problems,
                offsets=offs,
                seed=3000 + si,
                t0_min=2,
                stratify=True,
            )
            logger.info(
                f"  gap={gap:3d} schedule {si + 1}/{b.a.n_schedules}: "
                f"{offs.tolist()}"
            )
            res = _solve_all(b, out, tuning, data, prob, seed=3000 + si)
            for m, (v, _) in res.items():
                per_sched[m].append(v)
        row = {
            "gap": gap,
            "schedules": scheds,
            "n_problems": int(b.a.n_problems),
            "n_schedules": int(b.a.n_schedules),
        }
        for m, vs in per_sched.items():
            allv = np.concatenate(vs)
            # the schedule is the replicate here: report the spread ACROSS schedules so a
            # result that depends on one lucky draw is visible as a wide interval
            per_mean = np.array([float(np.nanmean(v)) for v in vs])
            row[m] = {
                **summarise(allv),
                "per_schedule_means": [float(x) for x in per_mean],
                "schedule_spread": (
                    float(per_mean.max() / max(per_mean.min(), 1e-12))
                    if len(per_mean) > 1
                    else 1.0
                ),
            }
            logger.info(
                f"        {m:9s} {fmt(row[m])}  "
                f"(across schedules x{row[m]['schedule_spread']:.2f})"
            )
        rows.append(row)
        (out / "G22_gap.json").write_text(
            json.dumps(
                {
                    "meta": {
                        "iters": b.a.iters,
                        "regime": DATA[b.a.regime],
                        "n_obs": int(b.a.n_obs_gap),
                        "note": (
                            "Observation times are drawn at RANDOM for each schedule "
                            "replicate, with the nearest one pinned at `gap`. Every "
                            "other experiment uses the fixed canonical schedule, so "
                            "this is the only one that separates a method's quality "
                            "from the schedule it was measured on."
                        ),
                        "targets_are_exterior_because": (
                            "U-Net and FNO optimise a single frame AT the target and cannot "
                            "represent anything earlier, so an interior target would be a "
                            "different problem only 3 of 5 methods could attempt"
                        ),
                        "partial": len(rows) < len(b.a.gaps),
                    },
                    "rows": rows,
                },
                indent=2,
            )
        )
    logger.info(f"saved -> {out / 'G22_gap'}.json")


def exp_regimes(b, out, tuning):
    """The canonical problem on EVERY tra regime, with cluster-robust intervals.

    Three regimes, three different senses of "held out", and the deck should not pretend
    they are interchangeable:

      gt_interp  Mach 0.66-0.68, inside the training Mach gap -- but a bit-identical
                 excerpt of val.nc, the split the KAE's checkpoint selection ran on.
      gt_longer  Mach 0.64-0.65, inside the gap and cut from test.nc: disjoint from both
                 train.nc and val.nc, so clean for every model.
      gt_extrap  Mach 0.50-0.52, OUTSIDE the training range entirely -- genuine
                 extrapolation -- but it is also the regime every learning rate and every
                 guidance strength was tuned on. It is reported as a stress test, never as
                 a clean test set.

    Problems are drawn stratified over trajectories and the interval is computed with the
    TRAJECTORY as the unit of replication (data_assimilation.tra.stats), because 48 problems drawn from
    6 trajectories are not 48 independent cases.
    """
    logger.info("=== R. every regime, with cluster-robust 95% CIs ===")
    from data_assimilation.tra.stats import summarise, fmt

    rows = {}
    for split in b.a.regimes:
        data = b.data(split)
        prob = build_problem(
            data,
            name=f"R_{split}",
            n_problems=b.a.n_regime,
            offsets=canonical(),
            seed=900,
            noise_std=b.a.noise,
            obs_frac=b.a.obs_frac,
            stratify=True,
        )
        logger.info(
            f"  --- {split}: {prob.meta()['n_problems']} problems over "
            f"{prob.meta()['n_trajectories']} trajectories ---"
        )
        res = _solve_all(b, out, tuning, data, prob, seed=900)
        row = {"problem": prob.meta(), "data_file": DATA[split]}
        for m, (v, wall) in res.items():
            st = summarise(v, clusters=prob.sim)
            st["wall_s"] = wall
            row[m] = st
            logger.info(
                f"        {m:9s} {fmt(st)}  (naive CI would be "
                f"±{(st.get('ci95_naive_halfwidth') or float('nan')):.4f}, "
                f"clustered ±{(st.get('ci95_cluster_halfwidth') or float('nan')):.4f})"
            )
        rows[split] = row
        (out / "R_regimes.json").write_text(
            json.dumps(
                {
                    "meta": {
                        "iters": b.a.iters,
                        "n_problems": b.a.n_regime,
                        "ci": (
                            "95% t-interval over PER-TRAJECTORY means; the naive "
                            "all-problems interval is reported alongside and is too "
                            "narrow because problems from one trajectory overlap"
                        ),
                        "regime_roles": {
                            "test": "gt_interp, Mach 0.66-0.68, excerpt of the KAE's val split",
                            "long": "gt_longer, Mach 0.64-0.65, disjoint from train and val",
                            "val": "gt_extrap, Mach 0.50-0.52, OUTSIDE training range AND the "
                            "tuning regime -- a stress test, not a clean test set",
                        },
                        "partial": len(rows) < len(b.a.regimes),
                    },
                    "rows": rows,
                },
                indent=2,
            )
        )
    logger.info(f"saved -> {out / 'R_regimes'}.json")


def exp_window_scaling(b, out, tuning):
    """How far back can each method reach, as a function of the window asked for?

    ``exp_backward`` fixes W = 8 and reports representability at that one width.  This
    sweeps W and turns the structural claim into a measured curve: the fraction of the
    requested window each method's control can express, and -- the part that is not
    free -- what asking for a wider window costs at t_0 itself.

    For the KAE and the 4D-Var baselines it costs nothing: the solve does not change with
    W.  For the score-based methods it does, because the sampled trajectory grows and the
    blanket has more segments to compose, so a wider window is paid for in accuracy at the
    analysis time.  That trade-off is the point of the experiment.

    Run on gt_longer: W = 32 needs t_0 >= 34 with observations out to t_0+25, which does
    not fit in a 60-frame record at all.
    """
    logger.info("=== G21. window-length scaling ===")
    from data_assimilation.tra.stats import summarise, fmt

    data = b.data("long")
    rows = []
    for W in b.a.window_scaling:
        W = int(W)
        prob = build_problem(
            data,
            name=f"WS{W}",
            n_problems=b.a.n_window,
            offsets=canonical(),
            seed=1100,
            t0_min=W + 2,
            stratify=True,
        )
        sim = torch.as_tensor(prob.sim, device=b.dev)
        t0 = torch.as_tensor(prob.t0, device=b.dev)
        par, om = data.params_for(sim), data.mask_for(sim)
        truth = torch.stack([data.frames(sim, t0 - j) for j in range(W)])
        row = {"W": W, "problem": prob.meta()}
        logger.info(
            f"  --- W = {W} ({prob.meta()['n_problems']} problems, "
            f"{prob.meta()['n_trajectories']} trajectories) ---"
        )
        for m in FOURDVAR + DIFFUSION:
            ad = b.model(m)
            k = ad.n_control_frames
            if m in DIFFUSION:
                rr = _solve_diffusion(
                    b, ad, data, prob, m, out, back=W - 1, keep_traj=True, seed=1100
                )
                r = None
            else:
                r = b.solve(
                    m, ad, data, prob, iters=b.a.iters, lr=tuning[m]["lr"], seed=1100
                )
            with torch.no_grad():
                if m in DIFFUSION:
                    tj = torch.as_tensor(rr["traj"], device=b.dev)
                    back = torch.stack([tj[:, (W - 1) - j] for j in range(W)])
                    repr_ = W
                elif m == "KAE":
                    K = ad.generator(par)
                    back = torch.stack(
                        [
                            ad.decode(ad.propagate(r["control"], -j * ad.dt_train, K=K))
                            for j in range(W)
                        ]
                    )
                    repr_ = W
                else:
                    phys = ad.to_physical(r["control"])
                    repr_ = min(k, W)
                    back = torch.stack(
                        [phys[:, k - 1 - j] for j in range(repr_)]
                        + [
                            torch.full_like(phys[:, 0], float("nan"))
                            for _ in range(W - repr_)
                        ]
                    )
                e = torch.stack([rel_l2(back[j], truth[j], om) for j in range(W)])
            per_j = []
            for j in range(W):
                st = summarise(e[j].cpu().numpy(), clusters=prob.sim)
                per_j.append(
                    {
                        "j": j,
                        **{
                            kk: st[kk]
                            for kk in ("mean", "ci95_cluster", "n", "n_clusters")
                        },
                    }
                )
            t0_stat = summarise(e[0].cpu().numpy(), clusters=prob.sim)
            row[m] = {
                "n_representable": int(repr_),
                "fraction_representable": float(repr_) / W,
                "analysis_t0": t0_stat,
                "per_j": per_j,
                "worst_representable": per_j[repr_ - 1] if repr_ >= 1 else None,
            }
            logger.info(
                f"        {m:9s} repr {repr_}/{W}  t0 {fmt(t0_stat)}"
                + (f"  t0-{repr_-1} {per_j[repr_-1]['mean']:.4f}" if repr_ > 1 else "")
            )
            del ad
            torch.cuda.empty_cache()
        rows.append(row)
        (out / "G21_window_scaling.json").write_text(
            json.dumps(
                {
                    "meta": {
                        "iters": b.a.iters,
                        "n_problems": b.a.n_window,
                        "regime": DATA["long"],
                        "note": (
                            "W is the number of past frames requested, t_0-0 .. "
                            "t_0-(W-1). `fraction_representable` is structural; "
                            "`analysis_t0` shows what asking for a wider window costs "
                            "at the analysis time itself, which is nonzero only for "
                            "the score-based methods."
                        ),
                        "why_gt_longer": (
                            "W=32 with observations to t_0+25 needs a record "
                            "longer than 60 frames; gt_interp and gt_extrap "
                            "cannot host it."
                        ),
                        "partial": len(rows) < len(b.a.window_scaling),
                    },
                    "rows": rows,
                },
                indent=2,
            )
        )
    logger.info(f"saved -> {out / 'G21_window_scaling'}.json")


def exp_misspec_noise(b, out, tuning):
    """Heavy-tailed observation noise, at MATCHED VARIANCE.

    Every method here assumes Gaussian observation error -- the 4D-Var methods through
    their least-squares cost, the samplers through ``Sigma_y`` in the Eq. 15 covariance.
    Drawing the noise from a Laplace distribution with the same variance misspecifies all
    five equally: the power is identical and only the tails differ, so whatever moves is
    attributable to the shape assumption and not to the noise level.

    Both distributions are run at each level so the comparison is paired, not inferred
    from a separate Gaussian sweep with different problem draws.
    """
    logger.info("=== G19. misspecified (Laplace) observation noise ===")
    pts = []
    for sd in b.a.misspec_levels:
        for dist in ("gaussian", "laplace"):
            pts.append(
                {
                    "tag": f"{dist}{sd:g}",
                    "value": float(sd),
                    "kw": {"noise_std": float(sd), "noise_dist": dist},
                    "extra": {"noise_dist": dist},
                }
            )
    _sweep(
        b,
        out,
        tuning,
        name="G19_misspec",
        points=pts,
        varies="noise_std x distribution (Gaussian vs Laplace, matched variance)",
        methods=b.a.methods,
    )


def exp_joint_sparsity(b, out, tuning):
    """A FIXED sensor budget, spent differently: densely once or sparsely often.

    Spatial and temporal sparsity are swept separately elsewhere, which cannot answer the
    question an experiment designer actually has.  Here the product ``N x obs_frac`` is
    held (approximately) constant, so every point costs the same number of scalar
    measurements and only their arrangement changes.  A method that reconstructs from a
    global latent should prefer many sparse looks; one that optimises a full field should
    prefer few dense ones.
    """
    logger.info("=== G20. joint spatial x temporal sparsity, fixed budget ===")
    budget = float(b.a.joint_budget)
    pts, seen = [], set()
    for n in b.a.joint_n_obs:
        offs = _geom(int(b.a.pin_delta_f), 25, int(n))
        # _geom pins both endpoints and deduplicates rounded frames, so the number of
        # observation times it returns is not the n that was asked for. The budget is
        # divided by what actually comes back, otherwise the "fixed budget" would vary by
        # a factor of nearly 3 across the sweep and the experiment would measure nothing.
        N = len(offs)
        if N in seen:
            continue
        seen.add(N)
        frac = min(1.0, budget / N)
        pts.append(
            {
                "tag": f"N{N}",
                "value": int(N),
                "kw": {"offsets": offs, "obs_frac": float(frac)},
                "extra": {
                    "obs_frac": float(frac),
                    "frames": offs.tolist(),
                    "n_times": N,
                    "measurements_per_problem_rel": float(N * frac),
                },
            }
        )
    _sweep(
        b,
        out,
        tuning,
        name="G20_joint_sparsity",
        points=pts,
        varies=f"N (with obs_frac = {budget:g}/N, so N x obs_frac is held fixed)",
        methods=b.a.methods,
    )


def exp_leakage_control(b, out, tuning):
    """The headline, re-run on a regime no model's SELECTION ever saw.

    ``data_assimilation.tra.training_specs`` checks each DA regime frame-by-frame against the three
    training splits and finds that gt_interp.nc -- the regime every headline number in this
    campaign is measured on -- is a bit-identical excerpt of val.nc.  No weight is fitted
    to it, and it is outside the training Mach range for all five models, but val.nc is the
    split the KAE's checkpoint selection (best_val_loss) ran on, and the turbpred
    checkpoints were not selected on it.  That asymmetry favours the KAE and a Mach-range
    comparison would never have shown it.

    gt_longer.nc (Mach 0.64-0.65) is cut from test.nc, is disjoint from train.nc and
    val.nc, and is not the tuning regime either, so it is clean for every model.  Running
    the identical canonical problem there is the control: if the ordering survives, the
    selection asymmetry is not what produces it.
    """
    logger.info(
        "=== L. leakage control: canonical problem on gt_longer (Mach 0.64-0.65) ==="
    )
    data = b.data("long")
    prob = build_problem(
        data,
        name="L",
        n_problems=b.a.n_problems,
        offsets=canonical(),
        seed=100,
        noise_std=b.a.noise,
        obs_frac=b.a.obs_frac,
    )
    summary = {}
    for m in FOURDVAR + DIFFUSION:
        ad = b.model(m)
        t_start = time.perf_counter()
        if m in DIFFUSION:
            v = _solve_diffusion(b, ad, data, prob, m, out, seed=100)["rel"]
        else:
            v = np.asarray(
                b.solve(
                    m, ad, data, prob, iters=b.a.iters, lr=tuning[m]["lr"], seed=100
                )["rel"]
            )
        summary[m] = {
            "mean": float(np.nanmean(v)),
            "sem": float(np.nanstd(v, ddof=1) / np.sqrt(v.size)),
            "n": int(v.size),
            "wall_s": time.perf_counter() - t_start,
        }
        logger.info(
            f"  {m:9s} rel-L2 {summary[m]['mean']:.4f} ± {summary[m]['sem']:.4f}"
        )
        del ad
        torch.cuda.empty_cache()
    (out / "L_leakage_control.json").write_text(
        json.dumps(
            {
                "meta": {
                    "regime": DATA["long"],
                    "mach": "0.64-0.65",
                    "iters": b.a.iters,
                    "n_problems": b.a.n_problems,
                    "why": (
                        "gt_interp.nc, where every headline number is measured, is an "
                        "excerpt of val.nc -- the split the KAE's checkpoint selection "
                        "used. gt_longer.nc is cut from test.nc and is disjoint from "
                        "both train.nc and val.nc, so no model's selection saw it. "
                        "Same canonical problem, same seed, same iteration budget."
                    ),
                    "headline_for_comparison": "A_headline.json + A_headline_diffusion.json",
                },
                "rows": summary,
            },
            indent=2,
        )
    )
    logger.info(f"saved -> {out / 'L_leakage_control'}.json")


SECTIONS = {
    "gap": exp_gap,
    "confounded": exp_confounded,
    "convergence": exp_convergence,
    "regimes": exp_regimes,
    "window_scaling": exp_window_scaling,
    "misspec_noise": exp_misspec_noise,
    "joint_sparsity": exp_joint_sparsity,
    "leakage_control": exp_leakage_control,
    "reach": exp_reach,
    "budget": exp_budget,
    "background": exp_background,
    "corrector": exp_corrector,
    "backward": exp_backward,
    "window": exp_window,
    "calibration": exp_calibration,
    "offgrid": exp_offgrid,
    "spacetime": exp_spacetime,
    "post_da": exp_post_da,
    "tune": tune,
    "headline": headline,
    "diffusion": headline_diffusion,
    "tune_diffusion": tune_diffusion,
    "delta_f": exp_delta_f,
    "noise": exp_noise,
    "sparsity": exp_sparsity,
    "delta_l": exp_delta_l,
    "n_obs": exp_n_obs,
    "single_obs": exp_single_obs,
    "forecast": exp_forecast,
    "cost": exp_cost,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=Path("da_results_tra"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--kae-run",
        default=("model_outputs_tra/continous_linear_128/run-20260821_031121"),
    )
    ap.add_argument("--sections", nargs="+", default=["tune", "headline"])
    ap.add_argument(
        "--regime",
        default="test",
        choices=["test", "long", "val"],
        help=(
            "which split every section treats as the regime under study. "
            "`tune`, `tune_diffusion` and `corrector` ignore it and stay on "
            "gt_extrap, so hyper-parameters are frozen and results remain "
            "comparable across regimes. NOTE: --regime val runs the campaign "
            "ON the tuning regime -- an out-of-training-range stress test, "
            "never a clean test set."
        ),
    )
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--tune-iters", type=int, default=400)
    ap.add_argument("--lrs", type=float, nargs="+", default=[3e-3, 1e-2, 3e-2, 1e-1])
    ap.add_argument("--n-problems", type=int, default=12)
    ap.add_argument("--n-tune", type=int, default=6)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--obs-frac", type=float, default=1.0)
    # diffusion DA
    ap.add_argument(
        "--kae-config",
        default=None,
        help="explicit model config, for checkpoints that did not save one",
    )
    ap.add_argument(
        "--kae-input-frames",
        type=int,
        default=0,
        help="override the conditioning-window length (0 = adapter default)",
    )
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--sigma-y-floor", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=1e-2)
    ap.add_argument("--corrections", type=int, default=1)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0])
    ap.add_argument("--corrections-grid", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-tune-diff", type=int, default=4)
    ap.add_argument("--delta-f", type=int, nargs="+", default=[1, 2, 4, 8, 12, 18])
    ap.add_argument(
        "--pin-delta-f",
        type=int,
        default=9,
        help=(
            "delta_f held fixed by every sweep that does not vary it (noise, "
            "sparsity, delta_l, n_obs, misspec_noise, joint_sparsity). 1 "
            "reproduces the original campaign, which pinned canonical()."
        ),
    )
    # The first five entries of each list are the canonical sweep and MUST stay in this
    # order: _sweep seeds problem draw i with 200+i, so reordering a list silently changes
    # which trajectories a point was measured on.  The tail extends into the regime the KS
    # campaign calls gap-filling, without disturbing the points the deck quotes.
    ap.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.03, 0.1, 0.3, 0.05, 0.15],
    )
    ap.add_argument(
        "--obs-fracs",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005],
    )
    # delta_l = 100 does not fit: gt_interp.nc holds 60 frames, so t0 + 100 does not exist.
    # 13 is the shortest horizon that still holds N = 5 distinct frames after delta_f = 9.
    ap.add_argument("--delta-l", type=int, nargs="+", default=[13, 18, 25, 35, 50])
    # 25 asks for more times than [9, 25] holds, so it realises every frame (N = 17)
    ap.add_argument("--n-obs", type=int, nargs="+", default=[2, 3, 5, 9, 13, 25])
    ap.add_argument("--single-taus", type=int, nargs="+", default=[1, 2, 4, 8, 16, 25])
    ap.add_argument(
        "--methods", nargs="+", default=["KAE", "UNet", "FNO", "ACDM", "ACDM-ncn"]
    )
    ap.add_argument("--forecast-frames", type=int, default=50)
    ap.add_argument("--spacetime-frames", type=int, default=30)
    ap.add_argument("--n-spacetime", type=int, default=3)
    ap.add_argument("--n-cal", type=int, default=6)
    ap.add_argument("--n-cal-draws", type=int, default=16)
    ap.add_argument("--offgrid-shifts", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    ap.add_argument("--back-frames", type=int, default=8)
    ap.add_argument(
        "--window-sweep", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 25]
    )
    ap.add_argument("--budget-iters", type=int, nargs="+", default=[50, 200, 500, 2000])
    ap.add_argument("--budget-draws", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument(
        "--background-weights",
        type=float,
        nargs="+",
        default=[0.0, 1e-4, 1e-3, 1e-2, 1e-1],
    )
    ap.add_argument(
        "--corrector-taus", type=float, nargs="+", default=[0.05, 0.1, 0.5, 2.0]
    )
    ap.add_argument("--reach-back", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    ap.add_argument("--misspec-levels", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    ap.add_argument(
        "--regimes",
        nargs="+",
        default=["test", "long", "val"],
        help="splits for --sections regimes; 'val' is the TUNING regime",
    )
    ap.add_argument(
        "--n-regime",
        type=int,
        default=48,
        help="problems per regime; note the trajectory pool is only 4-6",
    )
    ap.add_argument(
        "--window-scaling", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    ap.add_argument("--n-window", type=int, default=16)
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument(
        "--confound-every", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12]
    )
    ap.add_argument(
        "--n-schedules",
        type=int,
        default=3,
        help="random irregular observation sets drawn per gap",
    )
    ap.add_argument("--n-obs-gap", type=int, default=5)
    ap.add_argument("--gap-max-lead", type=int, default=25)
    ap.add_argument(
        "--solve-batch",
        type=int,
        default=8,
        help=(
            "problems per 4D-Var solve. Memory grows linearly in this, not "
            "in --n-problems, because the rollout graph is built once per "
            "batch. 0 means solve all at once."
        ),
    )
    ap.add_argument(
        "--joint-budget",
        type=float,
        default=1.0,
        help="N x obs_frac held at this value, so every point costs the same",
    )
    ap.add_argument("--joint-n-obs", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--cost-horizons", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    ap.add_argument("--cost-iters", type=int, default=30)
    ap.add_argument("--cost-batch", type=int, default=4)
    ap.add_argument(
        "--n-fields-saved",
        type=int,
        default=2,
        help="how many problems to keep ANALYSIS FIELDS for, per sweep point",
    )
    a = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    a.out_dir.mkdir(parents=True, exist_ok=True)
    b = Bench(a)
    logger.info(f"REGIME UNDER STUDY: {a.regime} -> {DATA[a.regime]}")
    if a.regime == "val":
        logger.warning(
            "--regime val is the TUNING regime (learning rates and the "
            "guidance strength were chosen here). Report it as an "
            "out-of-training-range stress test, not as a clean test set."
        )
    logger.info(f"frozen tuning stays on {DATA['val']}")

    tuning = None
    tp = a.out_dir / "tuning.json"
    if "tune" in a.sections:
        tuning = tune(b, a.out_dir)
    elif tp.is_file():
        tuning = json.loads(tp.read_text())
    if tuning:
        b.tuning = tuning
        picked = {
            m: (v.get("init", "climatology"), v.get("lr"))
            for m, v in tuning.items()
            if isinstance(v, dict)
        }
        logger.info(f"frozen 4D-Var settings (init, lr): {picked}")
    for s in a.sections:
        if s == "tune":
            continue
        if s == "tune_diffusion":
            tune_diffusion(b, a.out_dir)
            continue
        if tuning is None:
            raise SystemExit("no tuning.json; run --sections tune first")
        SECTIONS[s](b, a.out_dir, tuning)


if __name__ == "__main__":
    main()
