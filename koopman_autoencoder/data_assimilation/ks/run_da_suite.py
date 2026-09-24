# ruff: noqa: E741
# mypy: disable-error-code="assignment, var-annotated"
"""Master entry point for the KS data-assimilation campaign (ICLR 2027 revision).

Reproduces the entire campaign end to end:

    python -m data_assimilation.ks.run_da_suite --stage all --out-dir da_results_v2

Stages
------
canonical   Main KAE vs U-Net comparison (clean and sparse+noisy settings) with
            post-assimilation forecasting.
horizon     Assimilation cost and accuracy as observations move into the future.
nobs        Recovery vs number of observations.
noise       Recovery vs observation-noise level.
sparsity    Recovery vs fraction of observed spatial points.
irregular   Regular vs irregular observation times, on-grid and off-grid.
propagator  Exact matrix exponential vs the model's own RK4 integration (accuracy parity
            and runtime), same KAE weights.

Every stage appends rows to ``results.csv`` and per-problem arrays to
``raw/<stage>.npz``; ``summary.json`` collects the headline numbers.  Figures and LaTeX
tables are generated from these files only -- no number is ever transcribed by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_assimilation.ks.evaluate import (
    ae_reconstruction_floor,
    evaluate_solution,
    summarize,
)
from data_assimilation.ks.methods import KAE4DVar, SolveConfig, UNet4DVar
from data_assimilation.ks.models_io import load_kae, load_unet
from data_assimilation.ks.protocol import DT, KSData, build_problem

logger = logging.getLogger("da_suite")

CANONICAL_TAUS = np.array([0.1, 0.3, 0.7, 1.5, 2.5])
FORECAST_TAUS = np.round(np.arange(3.0, 20.01, 0.5), 3)


# ---------------------------------------------------------------------------
def git_rev() -> str:
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


class Recorder:
    """Accumulates tidy rows plus raw per-problem arrays."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.raw_dir = out_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict] = []
        self.summary: Dict = {}

    def add(self, **row):
        self.rows.append(row)

    def raw(self, name: str, **arrays):
        np.savez_compressed(self.raw_dir / f"{name}.npz", **arrays)

    def flush(self):
        if self.rows:
            keys: List[str] = []
            for r in self.rows:
                for k in r:
                    if k not in keys:
                        keys.append(k)
            with open(self.out_dir / "results.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.rows)
        with open(self.out_dir / "summary.json", "w") as f:
            json.dump(self.summary, f, indent=2, default=float)


def record_run(
    rec: Recorder,
    *,
    stage: str,
    condition: str,
    cond_value,
    method: str,
    problem,
    res: Dict,
    ev: Dict,
    floor: Optional[np.ndarray] = None,
    extra: Optional[Dict] = None,
):
    """One tidy row + the underlying per-problem arrays."""
    row = {
        "stage": stage,
        "condition": condition,
        "cond_value": cond_value,
        "method": method,
        "problem": problem.name,
        "n_problems": len(problem.sim),
        "n_obs": len(problem.taus),
        "taus": ";".join(f"{t:g}" for t in problem.taus),
        "obs_frac": problem.obs_frac,
        "noise_std": problem.noise_std,
        "on_grid": problem.on_grid,
        "seed": res["seed"],
        "iters": res["iters"],
        "lr": res["lr"],
        "init": res["init"],
        "control_kind": res["control_kind"],
    }
    for key in ("init_rel_l2", "obs_rel_l2", "spacetime_rel_l2", "forecast_rel_l2"):
        if key in ev:
            s = summarize(ev[key])
            row.update(
                {
                    f"{key}_{k}": v
                    for k, v in s.items()
                    if k in ("mean", "std", "median", "q25", "q75", "sem")
                }
            )
    if floor is not None:
        row["ae_floor_mean"] = float(np.mean(floor))
        row["ae_floor_std"] = float(np.std(floor, ddof=1))
    row.update(
        {
            "total_s": res["total_s"],
            "setup_s": res["setup_s"],
            "opt_s": res["opt_s"],
            "ms_per_iter": res["ms_per_iter"],
            "peak_mem_MiB": res["peak_mem_MiB"],
            "evals_network": res["evals_network"],
            "evals_decoder": res["evals_decoder"],
            "evals_propagation_steps": res["evals_propagation_steps"],
            "evals_per_iter_network": res["evals_per_iter_network"],
            "evals_per_iter_propagation_steps": res["evals_per_iter_propagation_steps"],
            "grad_norm_first": res["grad_norm_first"],
            "grad_norm_last": res["grad_norm_last"],
            "n_control_dims": res["n_control_dims"],
        }
    )
    if extra:
        row.update(extra)
    rec.add(**row)
    return row


# ---------------------------------------------------------------------------
class Campaign:
    def __init__(self, args):
        self.args = args
        self.dev = torch.device(args.device)
        self.data = KSData(args.test, self.dev)
        self.kae, self.K, self.D = load_kae(args.kae_run, args.kae_ckpt, self.dev)
        # Initialisation scales were frozen from the TRAINING split by data_assimilation/ks/init_scales.py and
        # are only read here, so nothing about the held-out analysis states can influence
        # either method's initialisation.
        with open(args.init_scales) as f:
            self.scales = json.load(f)
        self.z_scale = float(self.scales["kae_latent_init_scale"])
        self.x_scale = float(self.scales["unet_state_init_scale"])
        logger.info(
            f"frozen init scales (from {self.scales['source']}): "
            f"KAE z0 {self.z_scale:.4f}, U-Net x0 {self.x_scale:.4f}"
        )
        self.unet = load_unet(args.unet_ckpt, self.dev)
        with open(args.tuning) as f:
            self.tuning = json.load(f)
        self.hp = {m: self.tuning["results"][m]["best"] for m in self.tuning["results"]}
        logger.info(
            f"tuned hyper-parameters (chosen on {self.tuning['val_file']}): "
            + json.dumps(
                {k: {"lr": v["lr"], "init": v["init"]} for k, v in self.hp.items()}
            )
        )

    # -- method factory ------------------------------------------------------
    def method(self, name: str):
        if name == "KAE-expm":
            return KAE4DVar(self.kae, self.K, self.D, self.z_scale, "expm")
        if name == "KAE-rk4":
            return KAE4DVar(self.kae, self.K, self.D, self.z_scale, "rk4")
        if name == "UNet-4DVar":
            return UNet4DVar(self.unet, init_scale=self.x_scale, checkpoint_every=-1)
        raise KeyError(name)

    def cfg(self, name: str, iters: Optional[int] = None, seed: int = 0) -> SolveConfig:
        """Hyper-parameters chosen on validation; KAE-rk4 inherits KAE-expm's."""
        key = "KAE-expm" if name.startswith("KAE") else name
        hp = self.hp[key]
        return SolveConfig(
            iters=iters or self.args.iters, lr=hp["lr"], init=hp["init"], seed=seed
        )

    # -- generic runner ------------------------------------------------------
    def run(
        self,
        rec: Recorder,
        stage: str,
        condition: str,
        cond_value,
        problem,
        method_names,
        iters=None,
        seed=0,
        forecast=False,
        keep_raw=True,
    ):
        raws: Dict[str, np.ndarray] = {}
        for mname in method_names:
            m = self.method(mname)
            res = m.solve(problem, self.data, self.cfg(mname, iters, seed))
            ev = evaluate_solution(
                m,
                res["control"],
                problem,
                self.data,
                forecast_taus=FORECAST_TAUS if forecast else None,
            )
            floor = (
                ae_reconstruction_floor(self.kae, self.data, problem)
                if mname.startswith("KAE")
                else None
            )
            extra = {}
            if not problem.on_grid:
                prep = m.prepare(problem.taus)
                if "tau_error" in prep:
                    extra["tau_error_mean"] = float(np.mean(prep["tau_error"]))
                    extra["tau_realised"] = ";".join(
                        f"{t:g}" for t in prep["tau_realised"]
                    )
                else:
                    extra["tau_error_mean"] = 0.0
                    extra["tau_realised"] = ";".join(f"{t:g}" for t in problem.taus)
            record_run(
                rec,
                stage=stage,
                condition=condition,
                cond_value=cond_value,
                method=mname,
                problem=problem,
                res=res,
                ev=ev,
                floor=floor,
                extra=extra,
            )
            logger.info(
                f"  {stage:10s} {condition}={cond_value!s:>8s} {mname:11s} | "
                f"analysis {ev['init_rel_l2'].mean():.4f}±{ev['init_rel_l2'].std(ddof=1):.4f} "
                f"| obs {ev['obs_rel_l2'].mean():.4f} "
                f"| {res['ms_per_iter']:.2f} ms/it | {res['total_s']:.1f}s"
            )
            if keep_raw:
                tag = f"{mname}"
                raws[f"{tag}__init_rel_l2"] = ev["init_rel_l2"]
                raws[f"{tag}__obs_rel_l2"] = ev["obs_rel_l2"]
                raws[f"{tag}__spacetime_rel_l2"] = ev["spacetime_rel_l2"]
                raws[f"{tag}__window_curve"] = ev["window_rel_l2_curve"]
                raws[f"{tag}__window_taus"] = ev["window_taus"]
                raws[f"{tag}__control"] = res["control"].cpu().numpy()
                with torch.no_grad():
                    raws[f"{tag}__analysis"] = (
                        self.data.denorm(m.analysis_field(res["control"])).cpu().numpy()
                    )
                raws[f"{tag}__ms_per_iter"] = np.array(res["ms_per_iter"])
                raws[f"{tag}__total_s"] = np.array(res["total_s"])
                if floor is not None:
                    raws[f"{tag}__ae_floor"] = floor
                if forecast:
                    raws[f"{tag}__forecast_curve"] = ev["forecast_rel_l2_curve"]
                    raws[f"{tag}__forecast_taus"] = ev["forecast_taus"]
        if keep_raw:
            sim_t = torch.as_tensor(problem.sim, device=self.dev)
            t0_t = torch.as_tensor(problem.t0, device=self.dev)
            raws["truth_t0"] = (
                self.data.denorm(self.data.frames(sim_t, t0_t)).cpu().numpy()
            )
            raws["sim"] = problem.sim
            raws["t0"] = problem.t0
            raws["taus"] = problem.taus
            raws["mask"] = problem.mask
            raws["y"] = problem.y
        return raws

    # ======================================================================
    # STAGES
    # ======================================================================
    def stage_canonical(self, rec: Recorder):
        """Main comparison: clean and sparse+noisy, both with irregular observation times."""
        logger.info("=== CANONICAL: KAE vs U-Net 4D-Var ===")
        settings = [("clean", 1.0, 0.0), ("sparse_noisy", 0.25, 0.05)]
        for tag, frac, noise in settings:
            prob = build_problem(
                self.data,
                name=f"canonical_{tag}",
                n_problems=self.args.n_problems,
                taus=CANONICAL_TAUS,
                obs_frac=frac,
                noise_std=noise,
                seed=self.args.seed,
                forecast_taus=FORECAST_TAUS,
            )
            raws = self.run(
                rec,
                "canonical",
                "setting",
                tag,
                prob,
                ["KAE-expm", "UNet-4DVar"],
                forecast=True,
            )
            rec.raw(f"canonical_{tag}", **raws)
            with open(rec.out_dir / f"problem_canonical_{tag}.json", "w") as f:
                json.dump(prob.meta(), f, indent=2)
            # paired trajectory-level statistics
            a, b = raws["KAE-expm__init_rel_l2"], raws["UNet-4DVar__init_rel_l2"]
            self._paired(rec, f"canonical_{tag}", "init_rel_l2", a, b)
            fa = raws["KAE-expm__forecast_curve"].mean(0)
            fb = raws["UNet-4DVar__forecast_curve"].mean(0)
            self._paired(rec, f"canonical_{tag}", "forecast_rel_l2", fa, fb)

    def _paired(
        self, rec: Recorder, tag: str, metric: str, kae: np.ndarray, unet: np.ndarray
    ):
        """Paired (per-trajectory) KAE - U-Net comparison with a Wilcoxon test."""
        from scipy import stats

        d = np.asarray(kae) - np.asarray(unet)
        res = {
            "n": int(d.size),
            "mean_diff": float(d.mean()),
            "median_diff": float(np.median(d)),
            "sd_diff": float(d.std(ddof=1)),
            "kae_wins": int((d < 0).sum()),
            "ci95_low": float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(d.size)),
            "ci95_high": float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(d.size)),
        }
        try:
            w = stats.wilcoxon(kae, unet)
            res["wilcoxon_stat"] = float(w.statistic)
            res["wilcoxon_p"] = float(w.pvalue)
            t = stats.ttest_rel(kae, unet)
            res["ttest_p"] = float(t.pvalue)
        except Exception as e:  # noqa: BLE001
            res["test_error"] = str(e)
        rec.summary.setdefault("paired", {})[f"{tag}__{metric}"] = res
        logger.info(
            f"  paired {tag} {metric}: KAE-UNet mean diff {res['mean_diff']:+.4f} "
            f"(KAE better in {res['kae_wins']}/{res['n']}, "
            f"p={res.get('wilcoxon_p', float('nan')):.2e})"
        )

    def stage_horizon(self, rec: Recorder):
        """Cost and accuracy as the observation window moves further into the future."""
        logger.info("=== HORIZON scaling ===")
        horizons = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        n_obs = 5
        for H in horizons:
            taus = np.round(np.linspace(H / n_obs, H, n_obs) / DT).astype(int) * DT
            prob = build_problem(
                self.data,
                name=f"horizon_{H:g}",
                n_problems=self.args.n_problems_sweep,
                taus=taus,
                obs_frac=1.0,
                noise_std=0.0,
                seed=self.args.seed + 7,
            )
            raws = self.run(
                rec,
                "horizon",
                "max_tau",
                f"{H:g}",
                prob,
                ["KAE-expm", "KAE-rk4", "UNet-4DVar"],
                iters=self.args.horizon_iters,
                keep_raw=True,
            )
            rec.raw(f"horizon_{H:g}", **raws)

    def stage_nobs(self, rec: Recorder):
        logger.info("=== NUMBER OF OBSERVATIONS ===")
        pool = np.array([0.1, 0.3, 0.7, 1.5, 2.5, 4.0, 6.0, 9.0, 12.0, 16.0])
        for n in [1, 2, 3, 5, 7, 10]:
            taus = np.sort(pool[:n])
            prob = build_problem(
                self.data,
                name=f"nobs_{n}",
                n_problems=self.args.n_problems_sweep,
                taus=taus,
                obs_frac=1.0,
                noise_std=0.0,
                seed=self.args.seed + 11,
            )
            raws = self.run(
                rec, "nobs", "n_obs", str(n), prob, ["KAE-expm", "UNet-4DVar"]
            )
            rec.raw(f"nobs_{n}", **raws)

    def stage_noise(self, rec: Recorder):
        logger.info("=== OBSERVATION NOISE ===")
        for s in [0.0, 0.01, 0.05, 0.1, 0.2, 0.4]:
            prob = build_problem(
                self.data,
                name=f"noise_{s:g}",
                n_problems=self.args.n_problems_sweep,
                taus=CANONICAL_TAUS,
                obs_frac=1.0,
                noise_std=s,
                seed=self.args.seed + 21,
            )
            raws = self.run(
                rec, "noise", "noise_std", f"{s:g}", prob, ["KAE-expm", "UNet-4DVar"]
            )
            rec.raw(f"noise_{s:g}", **raws)

    def stage_sparsity(self, rec: Recorder):
        logger.info("=== SPATIAL SENSOR SPARSITY ===")
        for frac in [1.0, 0.5, 0.25, 0.1, 0.05]:
            prob = build_problem(
                self.data,
                name=f"sparsity_{frac:g}",
                n_problems=self.args.n_problems_sweep,
                taus=CANONICAL_TAUS,
                obs_frac=frac,
                noise_std=0.0,
                seed=self.args.seed + 31,
            )
            raws = self.run(
                rec,
                "sparsity",
                "obs_frac",
                f"{frac:g}",
                prob,
                ["KAE-expm", "UNet-4DVar"],
            )
            rec.raw(f"sparsity_{frac:g}", **raws)

    def stage_irregular(self, rec: Recorder):
        """Regular vs irregular observation times, on and off the training grid.

        ``regular``/``irregular`` both land on multiples of dt, so both models can
        evaluate them exactly: this isolates the effect of the *sampling pattern*.
        ``*_offgrid`` shifts every time off the grid by a fraction of dt; the KAE
        evaluates expm(K tau) at the exact tau, the U-Net must snap to the nearest
        whole step and the resulting time mismatch is recorded.
        """
        logger.info("=== IRREGULAR TEMPORAL SAMPLING ===")
        span, n = 2.5, 5
        regular = np.round(np.linspace(span / n, span, n) / DT).astype(int) * DT
        irregular = CANONICAL_TAUS.copy()  # same n, same span
        offs = {
            "regular": regular,
            "irregular": irregular,
            "regular_offgrid": regular + 0.043,
            "irregular_offgrid": irregular
            + np.array([0.043, -0.037, 0.041, -0.045, 0.039]),
        }
        for tag, taus in offs.items():
            prob = build_problem(
                self.data,
                name=f"irregular_{tag}",
                n_problems=self.args.n_problems_sweep,
                taus=np.round(taus, 4),
                obs_frac=1.0,
                noise_std=0.0,
                seed=self.args.seed + 41,
            )
            raws = self.run(
                rec, "irregular", "pattern", tag, prob, ["KAE-expm", "UNet-4DVar"]
            )
            rec.raw(f"irregular_{tag}", **raws)

    def stage_convergence(self, rec: Recorder):
        """Optimisation behaviour of the two variational problems at several horizons.

        Tests the hypothesis -- it is not assumed -- that backpropagating through a long
        autoregressive rollout makes the U-Net's assimilation problem harder to optimise
        than the KAE's, whose latent propagation is exactly linear so that the only
        nonlinearity in the cost is the decoder.  Records the objective, the observation
        error, the analysis error and the gradient norm against iteration.
        """
        logger.info("=== CONVERGENCE / GRADIENT diagnostics ===")
        for H in [0.5, 2.5, 10.0, 20.0]:
            taus = np.round(np.linspace(H / 5, H, 5) / DT).astype(int) * DT
            prob = build_problem(
                self.data,
                name=f"conv_{H:g}",
                n_problems=self.args.n_problems_sweep,
                taus=taus,
                obs_frac=1.0,
                noise_std=0.0,
                seed=self.args.seed + 71,
            )
            store: Dict[str, np.ndarray] = {}
            for mname in ["KAE-expm", "UNet-4DVar"]:
                m = self.method(mname)
                cfg = self.cfg(mname, self.args.horizon_iters, self.args.seed)
                cfg.track_every = max(1, self.args.horizon_iters // 200)
                res = m.solve(prob, self.data, cfg)
                ev = evaluate_solution(m, res["control"], prob, self.data)
                for k, v in res["hist"].items():
                    store[f"{mname}__{k}"] = np.asarray(v)
                store[f"{mname}__n_control_dims"] = np.array(res["n_control_dims"])
                record_run(
                    rec,
                    stage="convergence",
                    condition="max_tau",
                    cond_value=f"{H:g}",
                    method=mname,
                    problem=prob,
                    res=res,
                    ev=ev,
                    floor=(
                        ae_reconstruction_floor(self.kae, self.data, prob)
                        if mname.startswith("KAE")
                        else None
                    ),
                )
                logger.info(
                    f"  tau_max={H:5.1f} {mname:11s} | J {res['hist']['loss'][0]:.3e} -> "
                    f"{res['hist']['loss'][-1]:.3e} | analysis "
                    f"{res['hist']['init_rel_l2'][0]:.3f} -> {res['hist']['init_rel_l2'][-1]:.4f} "
                    f"| grad {res['grad_norm_first']:.3e} -> {res['grad_norm_last']:.3e}"
                )
            rec.raw(f"convergence_{H:g}", **store)

    def stage_memory(self, rec: Recorder):
        """Peak GPU memory of one 4D-Var iteration versus assimilation horizon.

        ``torch.cuda.max_memory_allocated`` reports *this process's* allocations, so the
        unrelated job sharing the device does not enter the measurement.  Each point is
        measured in isolation after emptying the cache and resetting the counter, at a
        fixed batch size, so the horizon is the only thing that varies.
        """
        logger.info("=== PEAK MEMORY vs horizon (controlled) ===")
        B = self.args.timing_batch
        rows = []
        for H in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]:
            taus = np.round(np.linspace(H / 5, H, 5) / DT).astype(int) * DT
            prob = build_problem(
                self.data,
                name=f"mem_{H:g}",
                n_problems=B,
                taus=taus,
                seed=self.args.seed + 81,
            )
            y = torch.as_tensor(prob.y, device=self.dev)
            mask = torch.as_tensor(prob.mask, device=self.dev)
            entry = {"max_tau": H, "n_steps": int(round(H / DT))}
            for mname in ["KAE-expm", "KAE-rk4", "UNet-4DVar", "UNet-4DVar-nockpt"]:
                base = mname.replace("-nockpt", "")
                m = self.method(base)
                if mname.endswith("-nockpt"):
                    m = UNet4DVar(
                        self.unet, init_scale=self.x_scale, checkpoint_every=0
                    )
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                # resident weights before the iteration: subtracting this isolates the part
                # of the footprint that actually scales with the assimilation window
                base = torch.cuda.memory_allocated() / 2**20
                try:
                    prep = m.prepare(taus)
                    c = torch.zeros(
                        tuple(m.control_shape(B)), device=self.dev, requires_grad=True
                    )
                    opt = torch.optim.Adam([c], lr=1e-2)
                    for _ in range(2):
                        opt.zero_grad(set_to_none=True)
                        pred = m.predict_at(c, taus, prep)
                        d = (pred - y) * mask[:, None, :]
                        (d**2).mean().backward()
                        opt.step()
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_allocated() / 2**20
                except torch.cuda.OutOfMemoryError:
                    peak = float("nan")
                    logger.info(f"    {mname} OOM at tau_max={H:g}")
                entry[mname] = peak
                entry[f"{mname}__base"] = base
                rec.add(
                    stage="memory",
                    condition="max_tau",
                    cond_value=f"{H:g}",
                    method=mname,
                    problem=prob.name,
                    n_problems=B,
                    n_obs=5,
                    peak_mem_MiB=peak,
                    base_mem_MiB=base,
                    activation_mem_MiB=peak - base,
                    evals_per_iter_propagation_steps=int(round(H / DT)),
                )
                del c, opt
                torch.cuda.empty_cache()
            rows.append(entry)
            logger.info(
                "  tau_max=%5.1f | " % H
                + " | ".join(
                    f"{k}={entry[k]:8.1f}"
                    for k in entry
                    if k not in ("max_tau", "n_steps") and not k.endswith("__base")
                )
                + " MiB"
            )
        rec.raw(
            "memory", **{k: np.array([r.get(k, np.nan) for r in rows]) for k in rows[0]}
        )
        rec.summary["memory"] = rows

    def stage_propagator(self, rec: Recorder):
        """Exact matrix exponential vs the model's own RK4 integration (same weights)."""
        logger.info("=== PROPAGATOR: exact expm vs RK4 (identical KAE weights) ===")
        self._propagator_operator_agreement(rec)
        for H in [2.5, 5.0, 10.0, 20.0, 40.0]:
            taus = np.round(np.linspace(H / 5, H, 5) / DT).astype(int) * DT
            prob = build_problem(
                self.data,
                name=f"prop_{H:g}",
                n_problems=self.args.n_problems_sweep,
                taus=taus,
                obs_frac=1.0,
                noise_std=0.0,
                seed=self.args.seed + 51,
            )
            raws = self.run(
                rec, "propagator", "max_tau", f"{H:g}", prob, ["KAE-expm", "KAE-rk4"]
            )
            rec.raw(f"propagator_{H:g}", **raws)

    def _propagator_operator_agreement(self, rec: Recorder):
        """Operator-level check that RK4 and expm realise the same continuous generator.

        For a linear generator the model's RK4 step is exactly the 4th-order Taylor
        truncation of ``expm(K dt)``, so ``RK4^n`` and ``expm(K n dt)`` can be compared
        directly as matrices -- independently of any assimilation run.
        """
        K = self.K
        I = torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
        A = DT * K
        R1 = I + A + A @ A / 2 + A @ A @ A / 6 + A @ A @ A @ A / 24
        rows = []
        for tau in [0.1, 0.5, 2.5, 5.0, 10.0, 20.0, 40.0]:
            n = int(round(tau / DT))
            P = torch.matrix_exp(K * tau)
            R = torch.linalg.matrix_power(R1, n)
            rel = float(
                torch.linalg.matrix_norm(P - R, "fro")
                / torch.linalg.matrix_norm(P, "fro")
            )
            rows.append({"tau": tau, "n_steps": n, "rel_fro_diff": rel})
            rec.add(
                stage="propagator",
                condition="operator_agreement",
                cond_value=f"{tau:g}",
                method="expm_vs_rk4",
                problem="operator",
                n_obs=0,
                n_problems=0,
                rel_fro_diff=rel,
                evals_per_iter_propagation_steps=n,
            )
        ev = torch.linalg.eigvals(K.float())
        rec.summary["generator"] = {
            "latent_dim": int(K.shape[0]),
            "eig_real_min": float(ev.real.min()),
            "eig_real_max": float(ev.real.max()),
            "eig_abs_max": float(ev.abs().max()),
            "spectral_norm": float(torch.linalg.matrix_norm(K, 2)),
            "operator_agreement": rows,
        }
        logger.info(
            "  operator agreement ||expm-RK4^n||_F/||expm||_F: "
            + ", ".join(f"tau={r['tau']:g}:{r['rel_fro_diff']:.1e}" for r in rows)
        )

    def stage_timing(self, rec: Recorder):
        """Interleaved wall-clock benchmark of one full DA iteration per method.

        The GPU is shared with other jobs, so methods are timed **A/B/A/B interleaved**
        within the same loop and the *median* over repeats is reported; contention then
        perturbs both methods alike instead of whichever ran first.  Each timed unit is a
        complete optimisation iteration: propagation to every observation time, decoding,
        the loss, the backward pass and the optimiser step -- i.e. everything the DA loop
        actually pays for.  The matrix exponentials are additionally timed on their own.
        """
        logger.info("=== TIMING (interleaved) ===")
        # Steady-state cost of the propagator setup itself. The exact method pays this
        # once per observation time before optimisation starts; it is included in the
        # total DA time but not in the per-iteration cost, so it is measured separately.
        for _ in range(5):
            torch.matrix_exp(self.K * 1.0)
        torch.cuda.synchronize()
        ts = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.matrix_exp(self.K * 2.5)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        expm_ms = float(np.median(ts))
        rec.summary["matrix_exp_ms_per_call"] = expm_ms
        rec.summary["matrix_exp_note"] = (
            "Median wall-clock of one torch.matrix_exp on the %dx%d generator, after "
            "warm-up. One such call is required per observation time, once, before the "
            "optimisation loop begins." % (self.K.shape[0], self.K.shape[1])
        )
        logger.info(
            f"  matrix_exp on {tuple(self.K.shape)}: {expm_ms:.3f} ms per call "
            f"(paid once per observation time, before optimisation)"
        )
        horizons = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
        n_obs, B = 5, self.args.timing_batch
        rows = []
        for H in horizons:
            taus = np.round(np.linspace(H / n_obs, H, n_obs) / DT).astype(int) * DT
            prob = build_problem(
                self.data,
                name=f"timing_{H:g}",
                n_problems=B,
                taus=taus,
                seed=self.args.seed + 61,
            )
            y = torch.as_tensor(prob.y, device=self.dev)
            mask = torch.as_tensor(prob.mask, device=self.dev)

            steppers, preps, setup = {}, {}, {}
            for mname in ["KAE-expm", "KAE-rk4", "UNet-4DVar"]:
                m = self.method(mname)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                prep = m.prepare(taus)
                torch.cuda.synchronize()
                setup[mname] = time.perf_counter() - t0
                c = torch.zeros(
                    tuple(m.control_shape(B)), device=self.dev, requires_grad=True
                )
                opt = torch.optim.Adam([c], lr=1e-2)

                def make(m=m, c=c, opt=opt, prep=prep):
                    def step():
                        opt.zero_grad(set_to_none=True)
                        pred = m.predict_at(c, taus, prep)
                        d = (pred - y) * mask[:, None, :]
                        loss = (d**2).mean()
                        loss.backward()
                        opt.step()

                    return step

                steppers[mname] = make()
                preps[mname] = prep

            names = list(steppers)
            for _ in range(5):  # warm-up
                for nm in names:
                    steppers[nm]()
            torch.cuda.synchronize()
            samples = {nm: [] for nm in names}
            for _ in range(self.args.timing_repeats):  # interleaved A/B/A/B
                for nm in names:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    steppers[nm]()
                    torch.cuda.synchronize()
                    samples[nm].append((time.perf_counter() - t0) * 1e3)

            med = {nm: float(np.median(v)) for nm, v in samples.items()}
            for nm in names:
                m = self.method(nm)
                counts = m.eval_counts(taus)
                rows.append(
                    {
                        "max_tau": H,
                        "method": nm,
                        "ms_per_iter": med[nm],
                        "iqr_ms": float(
                            np.percentile(samples[nm], 75)
                            - np.percentile(samples[nm], 25)
                        ),
                        "setup_s": setup[nm],
                        "evals_per_iter_network": counts["network"],
                        "evals_per_iter_propagation_steps": counts["propagation_steps"],
                        "n_steps_max": int(round(H / DT)),
                        "batch": B,
                    }
                )
                rec.add(
                    stage="timing",
                    condition="max_tau",
                    cond_value=f"{H:g}",
                    method=nm,
                    problem=prob.name,
                    n_problems=B,
                    n_obs=n_obs,
                    ms_per_iter=med[nm],
                    setup_s=setup[nm],
                    evals_per_iter_network=counts["network"],
                    evals_per_iter_propagation_steps=counts["propagation_steps"],
                )
            logger.info(
                f"  tau_max={H:5.1f} (steps={int(round(H/DT)):4d}) | "
                + " | ".join(f"{nm}={med[nm]:8.3f}ms" for nm in names)
                + f" | speedup vs U-Net x{med['UNet-4DVar']/med['KAE-expm']:.1f}"
                f" vs RK4 x{med['KAE-rk4']/med['KAE-expm']:.1f}"
            )
        rec.raw(
            "timing",
            **{k: np.array([r[k] for r in rows]) for k in rows[0] if k != "method"},
            method=np.array([r["method"] for r in rows]),
        )
        rec.summary["timing"] = rows


STAGES = [
    "canonical",
    "horizon",
    "convergence",
    "memory",
    "nobs",
    "noise",
    "sparsity",
    "irregular",
    "propagator",
    "timing",
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", nargs="+", default=["all"], choices=STAGES + ["all"])
    ap.add_argument("--out-dir", type=Path, default=Path("da_results_v2"))
    ap.add_argument("--test", type=Path, default=Path("data/ks/da_test.nc"))
    ap.add_argument("--train", type=Path, default=Path("data/ks/train.nc"))
    ap.add_argument(
        "--kae-run",
        type=Path,
        default=Path("model_outputs_ks/continous_linear_128/rollout_10"),
    )
    ap.add_argument("--kae-ckpt", type=Path, default=None)
    ap.add_argument(
        "--unet-ckpt",
        type=Path,
        default=Path("model_outputs_ks/unet1d/rollout10_base/best_model.pth"),
    )
    ap.add_argument("--tuning", type=Path, default=Path("da_results_v2/tuning.json"))
    ap.add_argument(
        "--init-scales", type=Path, default=Path("da_results_v2/init_scales.json")
    )
    ap.add_argument("--n-problems", type=int, default=64)
    ap.add_argument("--n-problems-sweep", type=int, default=32)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--horizon-iters", type=int, default=1000)
    ap.add_argument("--timing-batch", type=int, default=16)
    ap.add_argument("--timing-repeats", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in (
        logging.StreamHandler(),
        logging.FileHandler(args.out_dir / "logs" / "suite.log", mode="a"),
    ):
        h.setFormatter(fmt)
        root.addHandler(h)
    logging.getLogger("models").setLevel(logging.WARNING)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stages = STAGES if "all" in args.stage else args.stage
    logger.info(f"stages: {stages}")

    camp = Campaign(args)
    rec = Recorder(args.out_dir)
    rec.summary["provenance"] = {
        "git_rev": git_rev(),
        "argv": " ".join(sys.argv),
        "args": {k: str(v) for k, v in vars(args).items()},
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "kae_ckpt": str(
            args.kae_ckpt or (args.kae_run / "checkpoints" / "best_model.pth")
        ),
        "unet_ckpt": str(args.unet_ckpt),
        "test_file": str(args.test),
        "test_attrs": camp.data.attrs,
        "init_scales": camp.scales,
        "tuned_hp": camp.hp,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": (
            "GPU is shared with an unrelated training job; timings are measured "
            "interleaved across methods and reported as medians."
        ),
    }

    t0 = time.time()
    for st in stages:
        getattr(camp, f"stage_{st}")(rec)
        rec.flush()
    rec.summary["wall_time_s"] = time.time() - t0
    rec.flush()
    logger.info(f"campaign finished in {time.time() - t0:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
