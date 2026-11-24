# mypy: disable-error-code="var-annotated"
"""da_ks_experiments_3way.py — the KS data-assimilation study of `da_ks_experiments.py`,
run for THREE methods instead of one.

Section for section this reproduces the experiments A–J of the original study
(`da_ks_experiments.py`, visualised by `visualize_da_ks.ipynb`), with the same observation
protocol, the same analysis metric and the same output layout, but every quantity is now
computed for:

    KAE-expm    Continuous Koopman latent 4D-Var, exact  z(tau) = exp(K tau) z0
    KAE-rk4     the same model and the same weights, propagated by the model's own RK4
    UNet        physical-space 4D-Var over x0 through a frozen autoregressive U-Net
    SDA         score-based data assimilation: a diffusion trajectory prior conditioned on
                the observations through an observation likelihood

All four see byte-identical observations, masks, noise realisations and seeds.  The two
4D-Var methods share one optimisation driver, budget and initialisation scale.

Sections
--------
A. HEADLINE       recover the unobserved state at t0 from a few irregular future obs.
B. COST vs HORIZON per-iteration cost against assimilation horizon.
C. #OBSERVATIONS  recovery error vs number of (irregular) observations.
D. NOISE          recovery error vs observation-noise level.
E. SPARSITY       recovery error vs fraction of spatially observed grid points.
F. STATISTICS     distribution of recovery error over trajectories; exact-vs-RK4 parity.
G. CONTINUOUS-T   propagation as a smooth function of real tau; irregular vs uniform.
H. GALLERY        recovered vs true initial state across many trajectories.
I. SPACE-TIME     assimilate from a few future obs, then reconstruct the whole window.
J. SPARSE SENSORS recover the full initial field from sparse spatial sensors.

SDA operates on a fixed-length trajectory window, so it takes part in a section only when
that section's furthest observation fits inside the window; where it does not, its entries
are NaN and the reason is recorded in `sda_coverage` rather than the point being dropped.

Run:
    python -m data_assimilation.ks.da_ks_experiments_3way --out-dir da_results_3way
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from data_assimilation.ks.evaluate import ae_reconstruction_floor
from data_assimilation.ks.init_utils import init_kwargs
from data_assimilation.ks.methods import KAE4DVar, SolveConfig, UNet4DVar
from data_assimilation.ks.models_io import load_kae, load_unet
from data_assimilation.ks.protocol import DT, KSData, Problem, build_problem, rel_l2
from data_assimilation.ks.sda_paper import (
    SDA as SDAPaper,
    CosineVPSchedule,
    LinearObservation,
    LocalScoreUNet,
)

logger = logging.getLogger("da3way")

FOURDVAR = ["KAE-expm", "KAE-rk4", "UNet"]
ALL = FOURDVAR + ["SDA"]


# ---------------------------------------------------------------------------
class Bench:
    """Holds the three frozen models and runs any of them on a shared Problem."""

    def __init__(self, args):
        self.args = args
        self.dev = torch.device(args.device)
        self.data = KSData(args.test, self.dev)
        self.kae, self.K, self.D = load_kae(args.kae_run, None, self.dev)
        self.unet = load_unet(args.unet_ckpt, self.dev)

        with open(args.init_scales) as f:
            sc = json.load(f)
        self.z_scale = float(sc["kae_latent_init_scale"])
        self.x_scale = float(sc["unet_state_init_scale"])
        with open(args.tuning) as f:
            tun = json.load(f)
        self.hp = {m: tun["results"][m]["best"] for m in tun["results"]}

        # Paper-faithful SDA (Rozet & Louppe 2023). The local-score / Algorithm-2
        # construction means one trained model scores a trajectory of ANY length, so --
        # unlike the previous fixed-window implementation -- SDA is no longer excluded
        # from sections whose window exceeds the training segment.
        with open(args.sda_config) as f:
            self.sda_cfg = json.load(f)
        st = torch.load(self.sda_cfg["ckpt"], map_location="cpu", weights_only=False)
        net = LocalScoreUNet(
            k=int(st["k"]), hidden=tuple(st["hidden"]), blocks=int(st["blocks"])
        ).to(self.dev)
        net.load_state_dict(st["model_state_dict"])
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        gpath = Path(self.sda_cfg["ckpt"]).parent / "gamma.pt"
        Gam = torch.load(gpath, map_location=self.dev, weights_only=False)["Gamma"].to(
            self.dev
        )
        self.sda = SDAPaper(
            net,
            CosineVPSchedule(),
            self.dev,
            gamma=Gam,
            gamma_scale=self.sda_cfg["gamma_scale"],
            gamma_floor=self.sda_cfg["gamma_floor"],
        )
        if self.sda_cfg["gamma_mode"] == "scalar":
            self.sda.Gamma = None
        elif self.sda_cfg["gamma_mode"] == "none":
            self.sda.Gamma, self.sda.gamma_scale = None, 0.0
        self.sda_k = int(st["k"])
        self.sda_L = None  # no window limit any more
        logger.info(
            f"KAE latent D={self.D} | U-Net {args.unet_ckpt.name} | "
            f"SDA (paper) k={self.sda_k} blanket={2 * self.sda_k + 1} "
            f"N={self.sda_cfg['n_steps']} C={self.sda_cfg['corrections']} "
            f"tau={self.sda_cfg['tau']} Gamma={self.sda_cfg['gamma_mode']} | "
            f"tuned lr {self.hp['KAE-expm']['lr']}/{self.hp['UNet-4DVar']['lr']}"
        )

    # -- method construction -------------------------------------------------
    def method(self, name: str):
        if name == "KAE-expm":
            return KAE4DVar(self.kae, self.K, self.D, self.z_scale, "expm")
        if name == "KAE-rk4":
            return KAE4DVar(self.kae, self.K, self.D, self.z_scale, "rk4")
        if name == "UNet":
            return UNet4DVar(
                self.unet,
                init_scale=self.x_scale,
                checkpoint_every=getattr(self.args, "unet_checkpoint_every", -1),
            )
        raise KeyError(name)

    def cfg(
        self, name: str, iters: int, seed: int, prob: Optional[Problem] = None
    ) -> SolveConfig:
        hp = self.hp["KAE-expm" if name.startswith("KAE") else "UNet-4DVar"]
        per = (
            getattr(self.args, "iters_kae", 0)
            if name.startswith("KAE")
            else getattr(self.args, "iters_unet", 0)
        )
        ov = per or getattr(self.args, "iters_override", 0)
        it = ov if ov else max(10, int(iters * getattr(self.args, "iters_scale", 1.0)))
        # "obs" is the informed first guess and needs the problem to build it, so it is
        # resolved here rather than in SolveConfig.  Only the KAE has a backward map.
        ik = {"init": hp["init"]}
        if hp["init"] == "obs":
            assert prob is not None, "init='obs' needs the problem"
            assert name.startswith("KAE"), f"init='obs' is KAE-only, not {name}"
            ik = init_kwargs("obs", self.kae, self.K, prob, self.dev)
        return SolveConfig(iters=it, lr=hp["lr"], seed=seed, **ik)

    def sda_fits(self, prob: Problem, extra_frames: int = 0) -> bool:
        """Always true: Algorithm 2 composes local scores to any trajectory length."""
        return True

    # -- one method on one problem -------------------------------------------
    def run(
        self,
        name: str,
        prob: Problem,
        *,
        iters: int,
        seed: int,
        track_every: int = 0,
        window_frames: Optional[int] = None,
    ) -> Dict:
        """Returns analysis field, observation fit, per-trajectory error and history."""
        dev = self.dev
        sim = torch.as_tensor(prob.sim, device=dev)
        t0 = torch.as_tensor(prob.t0, device=dev)
        truth0 = self.data.frames(sim, t0)
        out: Dict = {"method": name}

        if name in FOURDVAR:
            m = self.method(name)
            cfg = self.cfg(name, iters, seed, prob)
            cfg.track_every = track_every
            t_start = time.perf_counter()
            res = m.solve(prob, self.data, cfg)
            out["wall_s"] = time.perf_counter() - t_start
            c = res["control"]
            with torch.no_grad():
                ana = m.analysis_field(c)
                prep = m.prepare(prob.taus)
                obs_pred = m.predict_at(c, prob.taus, prep)
                if window_frames:
                    st = m.forecast(c, np.arange(window_frames) * DT)
            out["hist"] = res["hist"]
            out["ms_per_iter"] = res["ms_per_iter"]
            out["peak_mem_MiB"] = res["peak_mem_MiB"]
            out["nfe"] = float(res["evals_network"])
        else:
            obs_frames = np.round(prob.taus / DT).astype(int)
            # the trajectory spans whatever the section asks for, at least far enough to
            # contain the furthest observation and at least one blanket
            L = max(
                int(window_frames or 0), int(obs_frames.max()) + 1, 2 * self.sda_k + 1
            )
            y_all = torch.as_tensor(prob.y, device=dev).permute(1, 0, 2).contiguous()
            mask = torch.as_tensor(prob.mask, device=dev)
            sigma_y = (
                prob.noise_std if prob.noise_std > 0 else self.sda_cfg["sigma_y_clean"]
            )
            frames_t = torch.as_tensor(obs_frames, device=dev)
            B = len(prob.sim)
            self.sda.reset_counters()
            t_start = time.perf_counter()
            chunks = []
            for i in range(0, B, self.args.chunk):
                sl = slice(i, min(i + self.args.chunk, B))
                o = LinearObservation(frames_t, mask, sigma_y)
                chunks.append(
                    self.sda.sample(
                        L,
                        self.data.X,
                        sl.stop - sl.start,
                        y=y_all[sl],
                        obs=o,
                        n_steps=self.sda_cfg["n_steps"],
                        corrections=self.sda_cfg["corrections"],
                        tau=self.sda_cfg["tau"],
                        seed=seed + 1000 * i,
                        n_samples=self.args.n_samples,
                    )
                )
            out["wall_s"] = time.perf_counter() - t_start
            samples = torch.cat(chunks, dim=1)  # [S, B, L, X]
            mean = samples.mean(0)
            out["hist"] = {}
            out["nfe"] = float(self.sda.nfe)
            out["n_segment_evals"] = float(self.sda.n_segment_evals)
            out["n_backward"] = float(self.sda.n_backward)
            out["spread"] = float((samples.std(0) * self.data.std).mean())
            out["samples"] = samples
            # denormalised posterior spread and a few raw draws, so the notebook can show
            # SDA as a distribution rather than collapsing it to its mean
            out["spread_field"] = (
                (samples.std(0) * self.data.std).cpu().numpy()
            )  # [B,L,X]
            out["draws"] = (
                self.data.denorm(samples[: min(4, samples.shape[0])]).cpu().numpy()
            )  # [s,B,L,X]
            out["trajectory_L"] = L
            ana = mean[:, 0]
            obs_pred = mean[:, frames_t].permute(1, 0, 2)
            if window_frames:
                st = mean[:, :window_frames].permute(1, 0, 2)

        # shared metrics, denormalised.
        #
        # For SDA the headline `rel` is the error of an INDIVIDUAL POSTERIOR DRAW, averaged
        # over draws -- not the error of the posterior mean. The paper evaluates the
        # posterior as a distribution and displays sampled trajectories; it never uses a
        # posterior mean. The mean is also the optimal estimator under squared error, so
        # scoring it against two point estimators would flatter SDA by roughly 1.7x on this
        # problem. The posterior-mean error is still recorded, as `rel_posterior_mean`.
        out["rel"] = (
            rel_l2(self.data.denorm(ana), self.data.denorm(truth0)).cpu().numpy()
        )
        if "samples" in out:
            smp = out["samples"]  # [S, B, L, X]
            per = torch.stack(
                [
                    rel_l2(self.data.denorm(smp[i][:, 0]), self.data.denorm(truth0))
                    for i in range(smp.shape[0])
                ]
            )  # [S, B]
            out["rel_posterior_mean"] = out["rel"]
            out["rel_per_draw"] = per.cpu().numpy()
            out["rel"] = per.mean(0).cpu().numpy()  # headline = a draw
        out["analysis"] = self.data.denorm(ana).cpu().numpy()
        out["obs_pred"] = self.data.denorm(obs_pred).cpu().numpy()
        if window_frames:
            out["spacetime"] = self.data.denorm(st).cpu().numpy()  # [T, B, X]
        return out


# ===========================================================================
# Sections — same experiments, same parameters, as da_ks_experiments.py
# ===========================================================================
CANON = [1, 3, 7, 15, 25]  # canonical irregular offsets, in frames


def _save(out: Path, name: str, **kw):
    np.savez_compressed(out / f"{name}.npz", **kw)


def exp_A_headline(b: Bench, out: Path) -> Dict:
    logger.info("=== A. HEADLINE recovery (irregular future observations) ===")
    taus = np.array(CANON) * DT
    prob = build_problem(b.data, name="A", n_problems=1, taus=taus, seed=100)
    floor = float(ae_reconstruction_floor(b.kae, b.data, prob).mean())
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    store = {
        "x": b.data.x,
        "offsets": np.array(CANON),
        "dt": DT,
        "ae_floor": floor,
        "u_t0_true": b.data.denorm(b.data.frames(sim, t0)).cpu().numpy()[0],
        "obs_true": np.stack(
            [b.data.denorm(b.data.frames(sim, t0 + o)).cpu().numpy()[0] for o in CANON]
        ),
    }
    summ = {"headline_ae_floor": floor}
    for m in ALL:
        r = b.run(m, prob, iters=2000, seed=100, track_every=25)
        if r.get("skipped"):
            logger.info(f"   {m:9s} skipped: {r['reason']}")
            continue
        store[f"{m}__u_t0_recon"] = r["analysis"][0]
        store[f"{m}__obs_pred"] = r["obs_pred"][:, 0]
        if "spread_field" in r:
            store[f"{m}__spread_t0"] = r["spread_field"][0, 0]
            store[f"{m}__draws_t0"] = r["draws"][:, 0, 0]
        store[f"{m}__rel_final"] = float(r["rel"][0])
        if "rel_per_draw" in r:
            store[f"{m}__rel_per_draw"] = r["rel_per_draw"][:, 0]
            store[f"{m}__rel_posterior_mean"] = float(r["rel_posterior_mean"][0])
        for k, v in r["hist"].items():
            store[f"{m}__hist_{k}"] = np.asarray(v)
        summ[f"headline_rel_L2_{m}"] = float(r["rel"][0])
        logger.info(
            f"   {m:9s} field rel-L2 = {r['rel'][0]:.4f}   (AE floor {floor:.4f})"
        )
    _save(out, "A_headline", **store)
    return summ


def exp_B_cost(b: Bench, out: Path) -> Dict:
    logger.info("=== B. COST vs HORIZON ===")
    cuda = b.dev.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize()

    horizons = [0.5, 1, 2, 5, 10, 20, 40]
    B = 4
    rows = {m: [] for m in FOURDVAR}
    for Ht in horizons:
        o = int(round(Ht / DT))
        prob = build_problem(
            b.data, name=f"B{Ht}", n_problems=B, taus=np.array([o * DT]), seed=7
        )
        y = torch.as_tensor(prob.y, device=b.dev)
        mask = torch.as_tensor(prob.mask, device=b.dev)
        steppers = {}
        for m in FOURDVAR:
            mm = b.method(m)
            prep = mm.prepare(prob.taus)
            c = torch.zeros(
                tuple(mm.control_shape(B)), device=b.dev, requires_grad=True
            )
            opt = torch.optim.Adam([c], lr=1e-2)

            def make(mm=mm, c=c, opt=opt, prep=prep):
                def step():
                    opt.zero_grad(set_to_none=True)
                    p = mm.predict_at(c, prob.taus, prep)
                    ((p - y) * mask[:, None, :]).pow(2).mean().backward()
                    opt.step()

                return step

            steppers[m] = make()
        for _ in range(5):
            for m in FOURDVAR:
                steppers[m]()
        sync()
        samp = {m: [] for m in FOURDVAR}
        for _ in range(25):  # interleaved so contention is shared
            for m in FOURDVAR:
                sync()
                t = time.perf_counter()
                steppers[m]()
                sync()
                samp[m].append((time.perf_counter() - t) * 1e3)
        for m in FOURDVAR:
            rows[m].append(float(np.median(samp[m])))
        logger.info(
            f"   tau={Ht:5.1f} (steps={o:4d}) | "
            + " | ".join(f"{m}={rows[m][-1]:8.3f} ms" for m in FOURDVAR)
            + f" | speedup vs UNet x{rows['UNet'][-1] / rows['KAE-expm'][-1]:.1f}"
            f" vs RK4 x{rows['KAE-rk4'][-1] / rows['KAE-expm'][-1]:.1f}"
        )
    # ---- score-based DA on the same axis -----------------------------------
    # SDA is not an iterative optimiser, so "ms per optimisation step" does not exist for
    # it. What does exist, and what its user actually pays, is the cost of drawing one
    # posterior sample: n_sample_steps x (one score-network forward + one likelihood-
    # guidance backward) over a trajectory window long enough to contain the observation.
    # Reaching horizon tau therefore requires a window of L = tau/dt + 1 frames, so we
    # time the sampler at exactly those window lengths.
    #
    # Two caveats are recorded with the numbers rather than hidden:
    #  * the score network is fully convolutional in time, so it *runs* at any L, but it
    #    was trained at one fixed L. These are cost measurements only; accuracy at other
    #    window lengths would require retraining.
    #  * that retraining is itself a cost the two 4D-Var methods never pay, because their
    #    propagators are horizon-agnostic by construction.
    sda_ms, sda_L_used = [], []
    for Ht in horizons:
        Lw = int(round(Ht / DT)) + 1
        Lw = max(Lw, 2 * b.sda_k + 1)
        sda_L_used.append(Lw)
        try:
            frames_t = torch.tensor([Lw - 1], device=b.dev)
            mk = torch.ones(1, b.data.X, device=b.dev)
            o = LinearObservation(frames_t, mk, 0.05)
            yv = torch.zeros(B, 1, b.data.X, device=b.dev)
            for _ in range(2):
                b.sda.sample(
                    Lw,
                    b.data.X,
                    B,
                    y=yv,
                    obs=o,
                    n_steps=b.sda_cfg["n_steps"],
                    corrections=b.sda_cfg["corrections"],
                    tau=b.sda_cfg["tau"],
                    seed=0,
                )
            sync()
            ts = []
            for _ in range(3):
                sync()
                t = time.perf_counter()
                b.sda.sample(
                    Lw,
                    b.data.X,
                    B,
                    y=yv,
                    obs=o,
                    n_steps=b.sda_cfg["n_steps"],
                    corrections=b.sda_cfg["corrections"],
                    tau=b.sda_cfg["tau"],
                    seed=0,
                )
                sync()
                ts.append((time.perf_counter() - t) * 1e3)
            sda_ms.append(float(np.median(ts)))
            logger.info(
                f"   tau={Ht:5.1f} SDA L={Lw:4d} | {sda_ms[-1]:9.1f} ms / "
                f"posterior sample ({b.sda_cfg['n_steps']} steps x "
                f"(1 + {b.sda_cfg['corrections']}) score evaluations, each a "
                f"batched pass over {Lw - 2 * b.sda_k} blanket segments)"
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            sda_ms.append(float("nan"))
            logger.info(
                f"   tau={Ht:5.1f} SDA L={Lw:4d} | unavailable ({type(e).__name__})"
            )
            if cuda:
                torch.cuda.empty_cache()

    h = np.array(horizons, dtype=float)
    ms_exact = np.array(rows["KAE-expm"])
    _save(
        out,
        "B_cost_vs_horizon",
        horizon_t=h,
        n_steps=np.round(h / DT).astype(int),
        ms_exact=ms_exact,
        ms_rollout=np.array(rows["KAE-rk4"]),
        ms_unet=np.array(rows["UNet"]),
        speedup=np.array(rows["KAE-rk4"]) / ms_exact,
        speedup_unet=np.array(rows["UNet"]) / ms_exact,
        # SDA has no optimisation step, so its ratio is only meaningful in the
        # like-for-like unit: total cost to one finished assimilation.
        speedup_sda_total=(np.array(sda_ms) * 8.0) / (ms_exact * 1000.0),
        ms_sda_per_sample=np.array(sda_ms),
        sda_window_L=np.array(sda_L_used),
        sda_blanket=2 * b.sda_k + 1,
        sda_sample_steps=b.sda_cfg["n_steps"],
        sda_corrections=b.sda_cfg["corrections"],
        batch=B,
    )
    return {
        "cost_max_speedup_rk4": float((np.array(rows["KAE-rk4"]) / ms_exact).max()),
        "cost_max_speedup_unet": float((np.array(rows["UNet"]) / ms_exact).max()),
        "sda_ms_per_sample_at_tau2.5": float(np.interp(2.5, h, np.array(sda_ms))),
    }


def _sweep(
    b: Bench,
    out: Path,
    name: str,
    values,
    make_kwargs,
    *,
    n_problems,
    iters,
    seed,
    log_fmt,
) -> None:
    res = {m: {"mean": [], "std": []} for m in ALL}
    floors = []
    for v in values:
        kw = make_kwargs(v)
        prob = build_problem(
            b.data, name=f"{name}{v}", n_problems=n_problems, seed=seed, **kw
        )
        floors.append(float(ae_reconstruction_floor(b.kae, b.data, prob).mean()))
        line = log_fmt.format(v=v)
        for m in ALL:
            r = b.run(m, prob, iters=iters, seed=seed)
            if r.get("skipped"):
                res[m]["mean"].append(np.nan)
                res[m]["std"].append(np.nan)
                continue
            res[m]["mean"].append(float(r["rel"].mean()))
            res[m]["std"].append(float(r["rel"].std()))
            if "rel_posterior_mean" in r:
                res[m].setdefault("pm", []).append(
                    float(r["rel_posterior_mean"].mean())
                )
            line += f" | {m} {res[m]['mean'][-1]:.4f}±{res[m]['std'][-1]:.4f}"
        logger.info(line)
    payload = {"values": np.asarray(values, dtype=float), "ae_floor": np.array(floors)}
    for m in ALL:
        payload[f"{m}__rel_mean"] = np.array(res[m]["mean"])
        payload[f"{m}__rel_std"] = np.array(res[m]["std"])
        if res[m].get("pm"):
            payload[f"{m}__rel_posterior_mean"] = np.array(res[m]["pm"])
    # keep the original single-method key names so the old plotting code still works
    payload["rel_mean"] = payload["KAE-expm__rel_mean"]
    payload["rel_std"] = payload["KAE-expm__rel_std"]
    _save(out, name, **payload)


def exp_C_nobs(b: Bench, out: Path):
    logger.info("=== C. RECOVERY vs NUMBER OF OBSERVATIONS ===")
    pool = [1, 3, 7, 15, 25, 40, 60, 90, 120, 160]
    _sweep(
        b,
        out,
        "C_nobs",
        [1, 2, 3, 5, 7, 10],
        lambda n: dict(taus=np.array(pool[:n]) * DT),
        n_problems=b.args.n_sweep,
        iters=1000,
        seed=11,
        log_fmt="   n_obs={v:3d}",
    )


def exp_D_noise(b: Bench, out: Path):
    logger.info("=== D. ROBUSTNESS to OBSERVATION NOISE ===")
    _sweep(
        b,
        out,
        "D_noise",
        [0.0, 0.01, 0.05, 0.1, 0.2, 0.4],
        lambda s: dict(taus=np.array(CANON) * DT, noise_std=s),
        n_problems=b.args.n_sweep,
        iters=1000,
        seed=21,
        log_fmt="   noise={v:.3f}",
    )


def exp_E_sparsity(b: Bench, out: Path):
    logger.info("=== E. ROBUSTNESS to SPATIAL SPARSITY ===")
    _sweep(
        b,
        out,
        "E_sparsity",
        [1.0, 0.5, 0.25, 0.1, 0.05],
        lambda f: dict(taus=np.array(CANON) * DT, obs_frac=f),
        n_problems=b.args.n_sweep,
        iters=1500,
        seed=31,
        log_fmt="   observed_frac={v:.2f}",
    )


def exp_F_statistics(b: Bench, out: Path) -> Dict:
    logger.info("=== F. STATISTICS over trajectories + exact/RK4 parity ===")
    prob = build_problem(
        b.data, name="F", n_problems=b.args.n_stats, taus=np.array(CANON) * DT, seed=123
    )
    floor = ae_reconstruction_floor(b.kae, b.data, prob)
    store = {"ae_floor": floor}
    summ = {}
    for m in ALL:
        r = b.run(m, prob, iters=1200, seed=123)
        if r.get("skipped"):
            continue
        store[f"{m}__rel"] = r["rel"]
        if "rel_per_draw" in r:
            store[f"{m}__rel_per_draw"] = r["rel_per_draw"]
            store[f"{m}__rel_posterior_mean"] = r["rel_posterior_mean"]
        store[f"{m}__wall_s"] = np.array(r.get("wall_s", np.nan))
        if "samples" in r:
            store[f"{m}__spread_mean"] = np.array(r["spread"])
        summ[f"stats_rel_mean_{m}"] = float(r["rel"].mean())
        summ[f"stats_rel_median_{m}"] = float(np.median(r["rel"]))
        summ[f"stats_rel_std_{m}"] = float(r["rel"].std())
        logger.info(
            f"   {m:9s} rel-L2 = {r['rel'].mean():.4f} ± {r['rel'].std():.4f} "
            f"(median {np.median(r['rel']):.4f})"
        )
    logger.info(f"   AE floor mean = {floor.mean():.4f}")
    # original key names: exact vs rollout parity
    store["rel_exact"] = store["KAE-expm__rel"]
    store["rel_rollout"] = store["KAE-rk4__rel"]
    _save(out, "F_statistics", **store)
    return summ


def exp_G_continuous(b: Bench, out: Path):
    logger.info("=== G. CONTINUOUS-TIME propagation + irregular vs uniform ===")
    from tensordict import TensorDict

    prob = build_problem(
        b.data, name="G", n_problems=1, taus=np.array([30 * DT]), seed=55
    )
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    x0 = b.data.frames(sim, t0)
    xi = b.data.X // 2
    z0 = b.kae.present_encoding(
        TensorDict({"u": x0.unsqueeze(1).unsqueeze(-1)}, batch_size=[1, 1]), None
    )
    taus = np.linspace(0, 3.0, 200)
    mk = b.method("KAE-expm")
    with torch.no_grad():
        cont = np.array(
            [float(b.data.denorm(mk.forecast(z0, [float(t)])[0])[0, xi]) for t in taus]
        )
        grid = list(range(0, 31))
        mu = b.method("UNet")
        disc_kae = np.array(
            [
                float(
                    b.data.denorm(b.method("KAE-rk4").forecast(z0, [n * DT])[0])[0, xi]
                )
                for n in grid
            ]
        )
        disc_unet = np.array(
            [float(b.data.denorm(mu.forecast(x0, [n * DT])[0])[0, xi]) for n in grid]
        )
    true_val = np.array(
        [float(b.data.denorm(b.data.frames(sim, t0 + n))[0, xi]) for n in grid]
    )
    store = {
        "cont_tau": taus,
        "cont_val": cont,
        "disc_tau": np.array(grid) * DT,
        "disc_val": disc_kae,
        "disc_val_unet": disc_unet,
        "true_tau": np.array(grid) * DT,
        "true_val": true_val,
        "probe_x": float(b.data.x[xi]),
    }

    irr, uni = [1, 3, 7, 15, 25], [5, 10, 15, 20, 25]
    for tag, offs in [("irregular", irr), ("uniform", uni)]:
        p = build_problem(
            b.data,
            name=f"G_{tag}",
            n_problems=b.args.n_sweep,
            taus=np.array(offs) * DT,
            seed=77,
        )
        for m in ALL:
            r = b.run(m, p, iters=1000, seed=77)
            if r.get("skipped"):
                continue
            store[f"{m}__rel_{tag}"] = r["rel"]
            logger.info(
                f"   {tag:9s} {offs} {m:9s} rel-L2 "
                f"{r['rel'].mean():.4f} ± {r['rel'].std():.4f}"
            )
    store["irr_offsets"] = np.array(irr)
    store["uni_offsets"] = np.array(uni)
    store["rel_irregular"] = store["KAE-expm__rel_irregular"]
    store["rel_uniform"] = store["KAE-expm__rel_uniform"]
    _save(out, "G_continuous", **store)


def exp_H_gallery(b: Bench, out: Path):
    logger.info("=== H. RECOVERY GALLERY ===")
    n = 12
    prob = build_problem(
        b.data, name="H", n_problems=n, taus=np.array(CANON) * DT, seed=202
    )
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    store = {
        "x": b.data.x,
        "offsets": np.array(CANON),
        "dt": DT,
        "true": b.data.denorm(b.data.frames(sim, t0)).cpu().numpy(),
    }
    for m in ALL:
        r = b.run(m, prob, iters=1500, seed=202)
        if r.get("skipped"):
            continue
        store[f"{m}__recon"] = r["analysis"]
        store[f"{m}__rel"] = r["rel"]
        if "rel_per_draw" in r:
            store[f"{m}__rel_per_draw"] = r["rel_per_draw"]
            store[f"{m}__rel_posterior_mean"] = r["rel_posterior_mean"]
        if "spread_field" in r:
            store[f"{m}__spread"] = r["spread_field"][:, 0]
            store[f"{m}__draws"] = r["draws"][:, :, 0]
        logger.info(
            f"   {m:9s} {n} trajectories | rel-L2 mean {r['rel'].mean():.4f} "
            f"[min {r['rel'].min():.4f}, max {r['rel'].max():.4f}]"
        )
    store["recon"] = store["KAE-expm__recon"]
    store["rel"] = store["KAE-expm__rel"]
    _save(out, "H_gallery", **store)


def exp_I_spacetime(b: Bench, out: Path):
    logger.info("=== I. SPACE-TIME assimilation + forecast ===")
    T, n_ex = 80, 3
    obs_off = [2, 9, 22, 40, 63]
    prob = build_problem(
        b.data, name="I", n_problems=n_ex, taus=np.array(obs_off) * DT, seed=303
    )
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    true = np.stack(
        [b.data.denorm(b.data.frames(sim, t0 + k)).cpu().numpy() for k in range(T)],
        axis=1,
    )  # [n_ex, T, X]
    store = {
        "x": b.data.x,
        "true": true,
        "obs_off": np.array(obs_off),
        "dt": DT,
        "T": T,
    }
    for m in ALL:
        r = b.run(m, prob, iters=2000, seed=303, window_frames=T)
        if r.get("skipped"):
            logger.info(f"   {m:9s} skipped: {r['reason']}")
            continue
        pred = np.transpose(r["spacetime"], (1, 0, 2))  # [n_ex, T', X]
        cov = pred.shape[1]
        if cov < T:  # score model's window is shorter than the panel
            pad = np.full((pred.shape[0], T - cov, pred.shape[2]), np.nan)
            pred = np.concatenate([pred, pad], axis=1)
        store[f"{m}__pred"] = pred
        if "spread_field" in r:
            store[f"{m}__spread_st"] = np.transpose(r["spread_field"], (0, 1, 2))
        store[f"{m}__covered_frames"] = cov
        store[f"{m}__rel_t0"] = r["rel"]
        if "rel_posterior_mean" in r:
            store[f"{m}__rel_t0_posterior_mean"] = r["rel_posterior_mean"]
        pf = np.linalg.norm(pred[:, :cov] - true[:, :cov], axis=2) / (
            np.linalg.norm(true[:, :cov], axis=2) + 1e-12
        )
        if cov < T:
            pf = np.concatenate([pf, np.full((pf.shape[0], T - cov), np.nan)], axis=1)
        store[f"{m}__per_frame_rel"] = pf
        logger.info(
            f"   {m:9s} covers {cov}/{T} frames (tau<={(cov - 1) * DT:.1f}) | "
            f"rel@t0 {r['rel'].mean():.4f} | window mean "
            f"{np.nanmean(pf):.4f}"
        )
    store["pred"] = store["KAE-expm__pred"]
    store["rel_t0"] = store["KAE-expm__rel_t0"]
    store["per_frame_rel"] = store["KAE-expm__per_frame_rel"]
    _save(out, "I_spacetime", **store)


def exp_J_sparse(b: Bench, out: Path):
    logger.info("=== J. SPARSE-SENSOR gap-filling recovery ===")
    n, frac = 6, 0.2
    prob = build_problem(
        b.data,
        name="J",
        n_problems=n,
        taus=np.array(CANON) * DT,
        obs_frac=frac,
        seed=404,
    )
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)
    obs_x = np.where(prob.mask[0] > 0)[0]
    store = {
        "x": b.data.x,
        "obs_x": obs_x,
        "frac": frac,
        "offsets": np.array(CANON),
        "true": b.data.denorm(b.data.frames(sim, t0)).cpu().numpy(),
    }
    for m in ALL:
        r = b.run(m, prob, iters=2000, seed=404)
        if r.get("skipped"):
            continue
        store[f"{m}__recon"] = r["analysis"]
        store[f"{m}__rel"] = r["rel"]
        if "rel_per_draw" in r:
            store[f"{m}__rel_per_draw"] = r["rel_per_draw"]
            store[f"{m}__rel_posterior_mean"] = r["rel_posterior_mean"]
        if "spread_field" in r:
            store[f"{m}__spread"] = r["spread_field"][:, 0]
            store[f"{m}__draws"] = r["draws"][:, :, 0]
        logger.info(
            f"   {m:9s} {len(obs_x)}/{b.data.X} sensors ({frac:.0%}) | "
            f"full-field rel-L2 mean {r['rel'].mean():.4f}"
        )
    store["recon"] = store["KAE-expm__recon"]
    store["rel"] = store["KAE-expm__rel"]
    _save(out, "J_sparse_recovery", **store)


# ===========================================================================
def add_common_args(ap):
    """Model, data and budget flags shared by every experiment driver, so that a new
    driver cannot silently pick up a different checkpoint or a different test set."""
    ap.add_argument("--test", type=Path, default=Path("data/ks/da_test.nc"))
    ap.add_argument(
        "--kae-run",
        type=Path,
        default=Path("model_outputs_ks/continous_linear_128/rollout_10"),
    )
    ap.add_argument(
        "--unet-ckpt",
        type=Path,
        default=Path("model_outputs_ks/unet1d/rollout10_extended2/best_model.pth"),
    )
    ap.add_argument(
        "--sda-config",
        type=Path,
        default=Path("da_results_sda_paper/frozen_config.json"),
        help="frozen paper-faithful SDA settings",
    )
    ap.add_argument("--tuning", type=Path, default=Path("da_results_v2/tuning.json"))
    ap.add_argument(
        "--init-scales", type=Path, default=Path("da_results_v2/init_scales.json")
    )
    ap.add_argument(
        "--iters-kae",
        type=int,
        default=0,
        help="optimisation budget for the KAE rows in EVERY section. Set it "
        "from the measured convergence sweep: each 4D-Var method is run "
        "to ITS OWN plateau, so neither is starved, and the resulting "
        "difference in cost is reported rather than hidden by forcing a "
        "common iteration count.",
    )
    ap.add_argument(
        "--iters-unet",
        type=int,
        default=0,
        help="optimisation budget for the U-Net rows in every section.",
    )
    ap.add_argument(
        "--iters-override",
        type=int,
        default=0,
        help="single budget for both 4D-Var methods (equal-iteration mode).",
    )
    ap.add_argument(
        "--iters-scale",
        type=float,
        default=1.0,
        help="scale every optimisation budget; <1 gives a fast, "
        "NOT-for-reporting pass that exercises all sections",
    )
    ap.add_argument("--n-sweep", type=int, default=48)
    ap.add_argument("--n-stats", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=8, help="SDA posterior samples")
    ap.add_argument("--sample-steps", type=int, default=128)
    ap.add_argument("--guidance", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument(
        "--unet-checkpoint-every",
        type=int,
        default=-1,
        help="rollout segment length for the U-Net 4D-Var graph. -1 picks "
        "sqrt(rollout) automatically; a smaller value trades time for "
        "memory, which is what a long delta_l needs on a contended card",
    )
    ap.add_argument("--sigma-y-clean", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=Path("da_results_3way"))
    add_common_args(ap)
    ap.add_argument("--sections", nargs="+", default=list("ABCDEFGHIJ"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in (
        logging.StreamHandler(),
        logging.FileHandler(args.out_dir / "logs" / "run.log", mode="w"),
    ):
        h.setFormatter(fmt)
        root.addHandler(h)
    logging.getLogger("models").setLevel(logging.WARNING)

    torch.manual_seed(0)
    np.random.seed(0)
    b = Bench(args)

    summary: Dict = {
        "methods": ALL,
        "sda_implementation": "paper-faithful (Rozet & Louppe 2023)",
        "sda_k": b.sda_k,
        "sda_blanket": 2 * b.sda_k + 1,
        "sda_settings": b.sda_cfg,
        "sda_coverage": (
            "Algorithm-2 composition of local scores lets one "
            "trained model score any trajectory length, so SDA "
            "now takes part in every section -- the window "
            "exclusions of the previous implementation are "
            "gone."
        ),
        "n_sweep": args.n_sweep,
        "n_stats": args.n_stats,
        "sda_n_samples": args.n_samples,
        "unet_ckpt": str(args.unet_ckpt),
        "sda_config": str(args.sda_config),
        "test_file": str(args.test),
    }
    table = {
        "A": exp_A_headline,
        "B": exp_B_cost,
        "C": exp_C_nobs,
        "D": exp_D_noise,
        "E": exp_E_sparsity,
        "F": exp_F_statistics,
        "G": exp_G_continuous,
        "H": exp_H_gallery,
        "I": exp_I_spacetime,
        "J": exp_J_sparse,
    }
    t0 = time.time()
    for s in args.sections:
        r = table[s](b, args.out_dir)
        if r:
            summary.update(r)
    summary["wall_time_s"] = time.time() - t0

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(args.out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, v])
    logger.info(f"finished in {time.time() - t0:.0f}s -> {args.out_dir}/")


if __name__ == "__main__":
    main()
