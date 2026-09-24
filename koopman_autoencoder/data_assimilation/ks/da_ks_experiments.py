# mypy: disable-error-code="arg-type, assignment, call-overload, index, var-annotated"
"""
da_ks_experiments.py — Comprehensive, reviewer-proof data-assimilation study for KS.

Runs the full battery and dumps everything (npz + CSV + JSON + log) into --out-dir,
to be visualised by `visualize_da_ks.ipynb`.

Sections
--------
A. HEADLINE      : recover the unobserved state at t0 from a few irregular future obs.
B. COST vs HORIZON: exact matrix-exp propagator (flat cost) vs autoregressive-rollout
                    baseline (cost linear in horizon)  -- the paper's central claim.
C. #OBSERVATIONS : recovery error vs number of (irregular) observations. Statistics.
D. NOISE         : recovery error vs observation noise level. Statistics.
E. SPARSITY      : recovery error vs fraction of spatially observed grid points. Statistics.
F. STATISTICS    : distribution of recovery error over many trajectories; exact-vs-rollout
                   accuracy parity (both solve the same inverse problem).
G. CONTINUOUS-T  : the exact propagator is a smooth function of real tau (single expm per
                   time), so irregular / off-grid observation times need no interpolation;
                   the discrete baseline only lands on multiples of dt. Plus irregular-vs-
                   uniform sampling recovery.

All statistics use a BATCH of trajectories (z0 is [B, D]) so N-trajectory statistics cost
a single DA solve, not N solves.

Run (training env):
    python -m data_assimilation.ks.da_ks_experiments --out-dir da_results
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from data_assimilation.ks.da_ks import (
    load_arch,
    build_model,
    _ks_stats,
    get_generator,
    encode_frame,
    evolve,
    propagate_rollout,
    masked_mse,
)

logger = logging.getLogger("da_ks_exp")


# ============================================================================
# Setup / data
# ============================================================================
class DAContext:
    def __init__(
        self, run_dir: Path, ckpt: Optional[Path], val: Path, device: torch.device
    ):
        self.device = device
        arch = load_arch(run_dir)
        self.arch = arch
        self.D = arch["latent_dim"]
        ckpt = ckpt or (run_dir / "checkpoints" / "best_model.pth")
        self.model = build_model(arch, ckpt, device)
        self.K = get_generator(self.model)  # [D, D]

        ds = xr.open_dataset(val)
        self.u_raw = (
            torch.from_numpy(ds["u"].values).float().to(device)
        )  # [sim, t, H, W]
        self.x = np.asarray(ds["x"].values).squeeze()
        ds.close()
        self.n_sim, self.n_t, self.H, self.W = self.u_raw.shape
        self.mean, self.std = _ks_stats(device)
        self.u_norm = (self.u_raw - self.mean) / (self.std + 1e-8)
        logger.info(
            f"Data: sims={self.n_sim} t={self.n_t} grid={self.H}x{self.W} | "
            f"latent D={self.D} | K eig(real) in "
            f"[{torch.linalg.eigvals(self.K.float()).real.min():.3f}, "
            f"{torch.linalg.eigvals(self.K.float()).real.max():.3f}]"
        )

    # ---- sampling & fields -------------------------------------------------
    def sample_batch(
        self, B: int, max_h_frames: int, t0_min: int = 50, seed: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g = np.random.default_rng(seed)
        t0_max = self.n_t - max_h_frames - 1
        sim = torch.tensor(g.integers(0, self.n_sim, size=B), device=self.device)
        t0 = torch.tensor(g.integers(t0_min, t0_max, size=B), device=self.device)
        return sim, t0

    def frame(self, sim: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Gather [B, H, W] normalised frames at (sim_i, t_i)."""
        return self.u_norm[sim, t]

    def field_relL2(
        self, z0: torch.Tensor, frame_t0_norm: torch.Tensor
    ) -> torch.Tensor:
        """Per-trajectory denormalised relative L2 error of decode(z0) vs u(t0). -> [B]."""
        with torch.no_grad():
            rec = self.model.decode(z0)["u"] * self.std + self.mean  # [B, H, W]
        tru = frame_t0_norm * self.std + self.mean
        num = torch.linalg.vector_norm((rec - tru).flatten(1), dim=1)
        den = torch.linalg.vector_norm(tru.flatten(1), dim=1).clamp_min(1e-12)
        return num / den

    def ae_floor(self, frame_t0_norm: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = encode_frame(self.model, frame_t0_norm)
            rec = self.model.decode(z)["u"] * self.std + self.mean
        tru = frame_t0_norm * self.std + self.mean
        num = torch.linalg.vector_norm((rec - tru).flatten(1), dim=1)
        den = torch.linalg.vector_norm(tru.flatten(1), dim=1).clamp_min(1e-12)
        return num / den


# ============================================================================
# Core batched assimilation
# ============================================================================
def assimilate(
    ctx: DAContext,
    sim: torch.Tensor,
    t0: torch.Tensor,
    offsets: List[int],
    *,
    method: str = "exact",  # "exact" | "rollout"
    mask: Optional[torch.Tensor] = None,  # [H, W] observed entries (None => full field)
    noise_std: float = 0.0,
    iters: int = 1000,
    lr: float = 1e-2,
    reg: float = 0.0,
    dt: float = 0.1,
    seed: int = 0,
    init_scale: Optional[float] = None,
    track: bool = True,
) -> Dict:
    """Batched variational DA. Recovers z0 [B, D] for B trajectories at once."""
    dev, D = ctx.device, ctx.D
    B = sim.shape[0]
    frame_t0 = ctx.frame(sim, t0)  # [B, H, W] (reference only)

    # observations at future times (+ optional noise), on the data grid
    g = torch.Generator(device=dev).manual_seed(seed)
    targets = []
    for o in offsets:
        y = ctx.frame(sim, t0 + o).clone()
        if noise_std > 0:
            y = y + noise_std * torch.randn(y.shape, generator=g, device=dev)
        targets.append(y)

    # exact propagators precomputed ONCE (cost independent of tau)
    phis = (
        [torch.matrix_exp(ctx.K * (o * dt)) for o in offsets]
        if method == "exact"
        else None
    )

    # init
    if init_scale is None:
        with torch.no_grad():
            init_scale = float(encode_frame(ctx.model, ctx.frame(sim, t0)).std().item())
    z0 = (torch.randn(B, D, device=dev, generator=g) * init_scale).requires_grad_(True)
    opt = torch.optim.Adam([z0], lr=lr)

    hist = {"iter": [], "obs_loss": [], "field_relL2_mean": [], "field_relL2_std": []}
    for it in range(iters):
        opt.zero_grad()
        loss = z0.new_zeros(())
        for i, o in enumerate(offsets):
            z_t = (
                evolve(z0, phis[i])
                if method == "exact"
                else propagate_rollout(z0, ctx.model, o, dt)
            )
            pred = ctx.model.decode(z_t)["u"]
            loss = loss + masked_mse(pred, targets[i], mask)
        loss = loss / len(offsets) + reg * (z0**2).mean()
        loss.backward()
        opt.step()
        if track and (it % max(1, iters // 40) == 0 or it == iters - 1):
            rel = ctx.field_relL2(z0.detach(), frame_t0)
            hist["iter"].append(it)
            hist["obs_loss"].append(float(loss.item()))
            hist["field_relL2_mean"].append(float(rel.mean()))
            hist["field_relL2_std"].append(float(rel.std()))

    rel = ctx.field_relL2(z0.detach(), frame_t0)
    return {
        "z0": z0.detach(),
        "frame_t0": frame_t0,
        "rel": rel.cpu().numpy(),
        "hist": hist,
        "offsets": offsets,
        "method": method,
    }


# ============================================================================
# Timing helper
# ============================================================================
def time_iter(fn, n_warm=5, n_time=30) -> float:
    """Median ms per (forward+backward+step) call of `fn` (returns a scalar loss)."""
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_time):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


# ============================================================================
# Experiment sections
# ============================================================================
def exp_headline(ctx: DAContext, out: Path, dt: float):
    logger.info("=== A. HEADLINE recovery (irregular future observations) ===")
    offsets = [1, 3, 7, 15, 25]
    sim, t0 = ctx.sample_batch(1, max_h_frames=max(offsets), seed=100)
    r = assimilate(ctx, sim, t0, offsets, method="exact", iters=2000, lr=1e-2, seed=100)
    floor = float(ctx.ae_floor(r["frame_t0"]).item())
    # fields (denormalised, 1-D along x)
    with torch.no_grad():
        u_t0_true = (r["frame_t0"] * ctx.std + ctx.mean).cpu().numpy()[0].squeeze()
        u_t0_rec = (
            (ctx.model.decode(r["z0"])["u"] * ctx.std + ctx.mean)
            .cpu()
            .numpy()[0]
            .squeeze()
        )
        obs_true = np.stack(
            [
                (ctx.frame(sim, t0 + o) * ctx.std + ctx.mean).cpu().numpy()[0].squeeze()
                for o in offsets
            ]
        )
        obs_pred = np.stack(
            [
                (
                    ctx.model.decode(
                        evolve(r["z0"], torch.matrix_exp(ctx.K * (o * dt)))
                    )["u"]
                    * ctx.std
                    + ctx.mean
                )
                .cpu()
                .numpy()[0]
                .squeeze()
                for o in offsets
            ]
        )
    np.savez(
        out / "A_headline.npz",
        x=ctx.x,
        u_t0_true=u_t0_true,
        u_t0_recon=u_t0_rec,
        obs_true=obs_true,
        obs_pred=obs_pred,
        offsets=np.array(offsets),
        dt=dt,
        ae_floor=floor,
        rel_final=float(r["rel"][0]),
        **{f"hist_{k}": np.array(v) for k, v in r["hist"].items()},
    )
    logger.info(f"   final field rel-L2={r['rel'][0]:.4f}  (AE floor {floor:.4f})")
    return {"headline_rel_L2": float(r["rel"][0]), "headline_ae_floor": floor}


def exp_cost_vs_horizon(ctx: DAContext, out: Path, dt: float):
    logger.info("=== B. COST vs HORIZON (exact vs autoregressive-rollout) ===")
    horizons_t = [0.5, 1, 2, 5, 10, 20, 40]  # physical time units
    B = 4
    sim, t0 = ctx.sample_batch(B, max_h_frames=int(max(horizons_t) / dt) + 1, seed=7)
    rows = []
    for Ht in horizons_t:
        o = int(round(Ht / dt))
        y = ctx.frame(sim, t0 + o)
        phi = torch.matrix_exp(ctx.K * (o * dt))
        z0 = (torch.randn(B, ctx.D, device=ctx.device) * 1.0).requires_grad_(True)
        opt = torch.optim.Adam([z0], lr=1e-2)

        def step_exact():
            opt.zero_grad()
            loss = F.mse_loss(ctx.model.decode(evolve(z0, phi))["u"], y)
            loss.backward()
            opt.step()
            return loss

        def step_rollout():
            opt.zero_grad()
            loss = F.mse_loss(
                ctx.model.decode(propagate_rollout(z0, ctx.model, o, dt))["u"], y
            )
            loss.backward()
            opt.step()
            return loss

        ms_exact = time_iter(step_exact)
        ms_roll = time_iter(step_rollout)
        rows.append((Ht, o, ms_exact, ms_roll, ms_roll / ms_exact))
        logger.info(
            f"   tau={Ht:5.1f} (steps={o:4d}) | exact={ms_exact:7.3f} ms | "
            f"rollout={ms_roll:8.3f} ms | speedup x{ms_roll/ms_exact:6.1f}"
        )
    rows = np.array(rows)
    np.savez(
        out / "B_cost_vs_horizon.npz",
        horizon_t=rows[:, 0],
        n_steps=rows[:, 1],
        ms_exact=rows[:, 2],
        ms_rollout=rows[:, 3],
        speedup=rows[:, 4],
    )
    return {"cost_max_speedup": float(rows[:, 4].max())}


def exp_statistics(ctx, out, dt):
    logger.info("=== F. STATISTICS over trajectories + exact/rollout parity ===")
    B, offsets = 128, [1, 3, 7, 15, 25]
    sim, t0 = ctx.sample_batch(B, max_h_frames=max(offsets), seed=123)
    r_ex = assimilate(
        ctx, sim, t0, offsets, method="exact", iters=1200, seed=123, track=False
    )
    r_ro = assimilate(
        ctx, sim, t0, offsets, method="rollout", iters=1200, seed=123, track=False
    )
    floor = ctx.ae_floor(r_ex["frame_t0"]).cpu().numpy()
    logger.info(
        f"   exact  : rel-L2 = {r_ex['rel'].mean():.4f} ± {r_ex['rel'].std():.4f} "
        f"(median {np.median(r_ex['rel']):.4f})"
    )
    logger.info(
        f"   rollout: rel-L2 = {r_ro['rel'].mean():.4f} ± {r_ro['rel'].std():.4f}"
    )
    logger.info(f"   AE floor mean = {floor.mean():.4f}")
    np.savez(
        out / "F_statistics.npz",
        rel_exact=r_ex["rel"],
        rel_rollout=r_ro["rel"],
        ae_floor=floor,
    )
    return {
        "stats_rel_mean": float(r_ex["rel"].mean()),
        "stats_rel_median": float(np.median(r_ex["rel"])),
        "stats_rel_std": float(r_ex["rel"].std()),
    }


def exp_continuous(ctx, out, dt):
    logger.info(
        "=== G. CONTINUOUS-TIME propagation + irregular vs uniform sampling ==="
    )
    sim, t0 = ctx.sample_batch(1, max_h_frames=60, seed=55)
    z0t = encode_frame(ctx.model, ctx.frame(sim, t0))
    xi = ctx.H // 2  # probe a fixed grid point
    # continuous curve from the EXACT propagator at arbitrary real tau
    taus = np.linspace(0, 3.0, 200)
    cont = []
    with torch.no_grad():
        for tau in taus:
            z = evolve(z0t, torch.matrix_exp(ctx.K * float(tau)))
            cont.append(
                float((ctx.model.decode(z)["u"] * ctx.std + ctx.mean)[0, xi, 0])
            )
    # discrete baseline: only lands on multiples of dt
    grid_frames = list(range(0, 31))
    disc_tau, disc_val, true_tau, true_val = [], [], [], []
    with torch.no_grad():
        for n in grid_frames:
            z = propagate_rollout(z0t, ctx.model, n, dt)
            disc_tau.append(n * dt)
            disc_val.append(
                float((ctx.model.decode(z)["u"] * ctx.std + ctx.mean)[0, xi, 0])
            )
            true_tau.append(n * dt)
            true_val.append(float((ctx.u_raw[sim[0], t0[0] + n])[xi, 0]))

    # irregular vs uniform sampling: same #obs, matched horizon span
    B = 48
    irr = [1, 3, 7, 15, 25]
    uni = [5, 10, 15, 20, 25]
    s_i, t_i = ctx.sample_batch(B, max_h_frames=25, seed=77)
    r_irr = assimilate(
        ctx, s_i, t_i, irr, method="exact", iters=1000, seed=77, track=False
    )
    r_uni = assimilate(
        ctx, s_i, t_i, uni, method="exact", iters=1000, seed=77, track=False
    )
    logger.info(
        f"   irregular {irr}: rel-L2 {r_irr['rel'].mean():.4f} ± {r_irr['rel'].std():.4f}"
    )
    logger.info(
        f"   uniform   {uni}: rel-L2 {r_uni['rel'].mean():.4f} ± {r_uni['rel'].std():.4f}"
    )
    np.savez(
        out / "G_continuous.npz",
        cont_tau=taus,
        cont_val=np.array(cont),
        disc_tau=np.array(disc_tau),
        disc_val=np.array(disc_val),
        true_tau=np.array(true_tau),
        true_val=np.array(true_val),
        probe_x=float(ctx.x[xi]),
        irr_offsets=np.array(irr),
        uni_offsets=np.array(uni),
        rel_irregular=r_irr["rel"],
        rel_uniform=r_uni["rel"],
    )


def exp_gallery(ctx, out, dt):
    """H. Gallery: true vs recovered initial state across many trajectories."""
    logger.info("=== H. RECOVERY GALLERY (many trajectories) ===")
    offsets = [1, 3, 7, 15, 25]
    B = 12
    sim, t0 = ctx.sample_batch(B, max_h_frames=max(offsets), seed=202)
    r = assimilate(
        ctx, sim, t0, offsets, method="exact", iters=1500, seed=202, track=False
    )
    with torch.no_grad():
        true = (r["frame_t0"] * ctx.std + ctx.mean).cpu().numpy().reshape(B, -1)
        rec = (
            (ctx.model.decode(r["z0"])["u"] * ctx.std + ctx.mean)
            .cpu()
            .numpy()
            .reshape(B, -1)
        )
    logger.info(
        f"   {B} trajectories | rel-L2 mean {r['rel'].mean():.4f} "
        f"[min {r['rel'].min():.4f}, max {r['rel'].max():.4f}]"
    )
    np.savez(
        out / "H_gallery.npz",
        x=ctx.x,
        true=true,
        recon=rec,
        rel=r["rel"],
        offsets=np.array(offsets),
        dt=dt,
    )


def exp_spacetime(ctx, out, dt):
    """I. Space-time: assimilate from a few future obs, then forecast the whole window."""
    logger.info("=== I. SPACE-TIME assimilation + forecast ===")
    T = 80
    n_ex = 3
    obs_off = [2, 9, 22, 40, 63]  # irregular observation times within window
    sim, t0 = ctx.sample_batch(n_ex, max_h_frames=T, seed=303)
    r = assimilate(
        ctx, sim, t0, obs_off, method="exact", iters=2000, seed=303, track=False
    )
    z0 = r["z0"]  # [n_ex, D]
    with torch.no_grad():
        pred = np.stack(
            [  # frame k=0 is t0 (expm(0)=I)
                (
                    ctx.model.decode(evolve(z0, torch.matrix_exp(ctx.K * (k * dt))))[
                        "u"
                    ]
                    * ctx.std
                    + ctx.mean
                )
                .cpu()
                .numpy()
                .reshape(n_ex, -1)
                for k in range(T)
            ],
            axis=1,
        )  # [n_ex, T, H]
    true = np.stack(
        [ctx.u_raw[sim, t0 + k].cpu().numpy().reshape(n_ex, -1) for k in range(T)],
        axis=1,
    )  # [n_ex, T, H]
    per_frame_rel = np.linalg.norm(pred - true, axis=2) / (
        np.linalg.norm(true, axis=2) + 1e-12
    )  # [n_ex, T]
    logger.info(
        f"   {n_ex} examples, window T={T} frames (τ<= {T*dt:.1f}), "
        f"obs at frames {obs_off} | final-window rel-L2 mean {per_frame_rel.mean():.4f}"
    )
    np.savez(
        out / "I_spacetime.npz",
        x=ctx.x,
        true=true,
        pred=pred,
        obs_off=np.array(obs_off),
        per_frame_rel=per_frame_rel,
        rel_t0=r["rel"],
        dt=dt,
        T=T,
    )


def exp_sparse_recovery(ctx, out, dt):
    """J. Gap-filling: recover the FULL initial field from sparse spatial sensors."""
    logger.info("=== J. SPARSE-SENSOR gap-filling recovery ===")
    offsets = [1, 3, 7, 15, 25]
    B = 6
    frac = 0.2
    g = torch.Generator(device=ctx.device).manual_seed(9)
    mask = (torch.rand(ctx.H, ctx.W, generator=g, device=ctx.device) < frac).float()
    sim, t0 = ctx.sample_batch(B, max_h_frames=max(offsets), seed=404)
    r = assimilate(
        ctx,
        sim,
        t0,
        offsets,
        method="exact",
        mask=mask,
        iters=2000,
        seed=404,
        track=False,
    )
    with torch.no_grad():
        true = (r["frame_t0"] * ctx.std + ctx.mean).cpu().numpy().reshape(B, -1)
        rec = (
            (ctx.model.decode(r["z0"])["u"] * ctx.std + ctx.mean)
            .cpu()
            .numpy()
            .reshape(B, -1)
        )
    obs_x = np.where(mask.cpu().numpy().reshape(-1) > 0)[0]
    logger.info(
        f"   observed {len(obs_x)}/{ctx.H} sensors ({frac:.0%}) | "
        f"full-field rel-L2 mean {r['rel'].mean():.4f}"
    )
    np.savez(
        out / "J_sparse_recovery.npz",
        x=ctx.x,
        true=true,
        recon=rec,
        obs_x=obs_x,
        rel=r["rel"],
        frac=frac,
        offsets=np.array(offsets),
    )


# ---- wrappers that inject per-call trajectory sampling into the sweeps -----
def _patched_sweep(
    ctx, out, name, values, base_kwargs_fn, B, iters, seed, max_h, log_fmt
):
    means, stds, floors = [], [], []
    for v in values:
        sim, t0 = ctx.sample_batch(B, max_h_frames=max_h, seed=seed)
        kw = base_kwargs_fn(v)
        kw.pop("sim", None)
        kw.pop("t0", None)
        r = assimilate(ctx, sim, t0, iters=iters, seed=seed, track=False, **kw)
        means.append(float(r["rel"].mean()))
        stds.append(float(r["rel"].std()))
        floors.append(float(ctx.ae_floor(r["frame_t0"]).mean()))
        logger.info(log_fmt.format(v=v, m=means[-1], s=stds[-1]))
    np.savez(
        out / name,
        values=np.array(values, dtype=float),
        rel_mean=np.array(means),
        rel_std=np.array(stds),
        ae_floor=np.array(floors),
    )
    return means, stds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path("model_outputs_ks/continous_linear_128/rollout_10"),
    )
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--val", type=Path, default=Path("data/ks/val.nc"))
    ap.add_argument("--out-dir", type=Path, default=Path("da_results"))
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.out_dir / "logs" / "run.log", mode="w"),
        ],
    )
    for noisy in ["models", "matplotlib"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    ctx = DAContext(args.run_dir, args.ckpt, args.val, dev)

    summary: Dict = {}
    summary.update(exp_headline(ctx, args.out_dir, args.dt))
    summary.update(exp_cost_vs_horizon(ctx, args.out_dir, args.dt))

    _patched_sweep(
        ctx,
        args.out_dir,
        "C_nobs.npz",
        [1, 2, 3, 5, 7, 10],
        lambda n: dict(
            offsets=[1, 3, 7, 15, 25, 40, 60, 90, 120, 160][:n], method="exact"
        ),
        48,
        1000,
        11,
        160,
        "   n_obs={v:3d} | field rel-L2 = {m:.4f} ± {s:.4f}",
    )
    _patched_sweep(
        ctx,
        args.out_dir,
        "D_noise.npz",
        [0.0, 0.01, 0.05, 0.1, 0.2, 0.4],
        lambda s: dict(offsets=[1, 3, 7, 15, 25], method="exact", noise_std=s),
        48,
        1000,
        21,
        25,
        "   noise_std={v:.3f} | field rel-L2 = {m:.4f} ± {s:.4f}",
    )
    # sparsity needs a per-value mask generator
    g_mask = torch.Generator(device=dev).manual_seed(5)

    def sparsity_kwargs(frac):
        m = (torch.rand(ctx.H, ctx.W, generator=g_mask, device=dev) < frac).float()
        if m.sum() == 0:
            m[0, 0] = 1.0
        return dict(offsets=[1, 3, 7, 15, 25], method="exact", mask=m)

    logger.info("=== E. ROBUSTNESS to SPATIAL SPARSITY ===")
    _patched_sweep(
        ctx,
        args.out_dir,
        "E_sparsity.npz",
        [1.0, 0.5, 0.25, 0.1, 0.05],
        sparsity_kwargs,
        48,
        1500,
        31,
        25,
        "   observed_frac={v:.2f} | field rel-L2 = {m:.4f} ± {s:.4f}",
    )

    summary.update(exp_statistics(ctx, args.out_dir, args.dt))
    exp_continuous(ctx, args.out_dir, args.dt)
    exp_gallery(ctx, args.out_dir, args.dt)
    exp_spacetime(ctx, args.out_dir, args.dt)
    exp_sparse_recovery(ctx, args.out_dir, args.dt)

    # summary CSV + JSON
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(args.out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, v])
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")
    logger.info(f"All artefacts saved under {args.out_dir}/")


if __name__ == "__main__":
    main()
