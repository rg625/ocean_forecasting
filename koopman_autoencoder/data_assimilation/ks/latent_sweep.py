"""Latent-dimension sweep: assimilation accuracy, forecast skill and cost vs N_z.

Trains nothing. Reads the checkpoints produced by ``run_latent_sweep.sh`` (identical KS
recipe, 200 epochs, rollout 10, hidden [64,128,256], circular padding; only
``latent_dim`` differs) and, for each, measures on the SAME held-out problems:

  * analysis relative L2 at the canonical schedule (delta_f=0.1, delta_l=2.5, N=5),
  * the autoencoder round-trip floor, which bounds the analysis error,
  * free-running forecast error at a short and a moderate lead from the true state,
  * cost of one 4D-Var iteration and of a single ``matrix_exp`` on the generator,
  * peak activation memory of one iteration,
  * the generator's largest eigenvalue real part.

Every method sees the same observations, masks, noise realisation, initialisation rule,
learning rate, seed and iteration budget; only ``latent_dim`` varies.

    python -m data_assimilation.ks.latent_sweep --out da_results_latentdim/sweep.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_assimilation.ks.methods import KAE4DVar, SolveConfig
from data_assimilation.ks.models_io import kae_latent_scale, load_kae
from data_assimilation.ks.protocol import DT, KSData, build_problem

logger = logging.getLogger("data_assimilation.ks.latent_sweep")

CANONICAL_FRAMES = np.array([1, 3, 7, 15, 25])  # delta_f=0.1, delta_l=2.5, N=5
FORECAST_LEADS = np.array([0.5, 2.5])


def median_ms(step, reps: int, warmup: int, cuda: bool) -> float:
    for _ in range(warmup):
        step()
    ts = []
    for _ in range(reps):
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        step()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


@torch.no_grad()
def roundtrip_floor(model, data: KSData, problem) -> float:
    """Encode/decode the true analysis frame: the representational bound on the analysis."""
    from tensordict import TensorDict

    dev = data.device
    true = data.frames(
        torch.as_tensor(problem.sim, device=dev),
        torch.as_tensor(problem.t0, device=dev),
    )  # [B, X]
    f = true.unsqueeze(-1).unsqueeze(1)
    td = TensorDict({"u": f.squeeze(-1).unsqueeze(-1)}, batch_size=[f.shape[0], 1])
    z = model.present_encoding(td, cond_input=None)
    rec = model.decode(z)["u"].squeeze(-1).squeeze(-1)
    return float(
        (
            torch.linalg.vector_norm(rec - true, dim=-1)
            / torch.linalg.vector_norm(true, dim=-1)
        ).mean()
    )


@torch.no_grad()
def forecast_skill(kae: KAE4DVar, data: KSData, problem, leads) -> dict:
    """Free-running error from the TRUE state (no assimilation)."""
    from tensordict import TensorDict

    dev = data.device
    true0 = data.frames(
        torch.as_tensor(problem.sim, device=dev),
        torch.as_tensor(problem.t0, device=dev),
    )
    f = true0.unsqueeze(-1).unsqueeze(1)
    td = TensorDict({"u": f.squeeze(-1).unsqueeze(-1)}, batch_size=[f.shape[0], 1])
    z0 = kae.model.present_encoding(td, cond_input=None)
    pred = kae.forecast(z0, leads)  # [n_lead, B, X]
    out = {}
    for j, tau in enumerate(leads):
        n = int(round(float(tau) / DT))
        tgt = data.frames(
            torch.as_tensor(problem.sim, device=dev),
            torch.as_tensor(problem.t0 + n, device=dev),
        )
        out[f"forecast_tau{tau:g}"] = float(
            (
                torch.linalg.vector_norm(pred[j] - tgt, dim=-1)
                / torch.linalg.vector_norm(tgt, dim=-1)
            ).mean()
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-root", type=Path, default=Path("model_outputs_ks_latentdim")
    )
    ap.add_argument("--dims", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    ap.add_argument("--train", type=Path, default=Path("data/ks/train.nc"))
    ap.add_argument("--test", type=Path, default=Path("data/ks/da_test.nc"))
    ap.add_argument("--out", type=Path, default=Path("da_results_latentdim/sweep.json"))
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--n-problems", type=int, default=48)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.getLogger("models").setLevel(logging.WARNING)

    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    test = KSData(a.test, dev)
    train = KSData(a.train, dev)
    taus = CANONICAL_FRAMES * DT
    problem = build_problem(
        test, name="latentdim", n_problems=a.n_problems, taus=taus, seed=0
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for dz in a.dims:
        cands = sorted((a.runs_root / f"dz_{dz}").glob("run-*"))
        if not cands:
            logger.warning(f"  N_z={dz}: no run directory found, skipping")
            continue
        run = cands[-1]
        model, K, latent = load_kae(run, run / "checkpoints" / "best_model.pth", dev)
        assert (
            latent == dz
        ), f"checkpoint at {run} has latent_dim {latent}, expected {dz}"
        # initialisation scale from TRAINING data only, exactly as the main campaign does
        scale = kae_latent_scale(model, train)
        kae = KAE4DVar(model, K, latent, init_scale=scale, propagator="expm")

        res = kae.solve(
            problem, test, SolveConfig(iters=a.iters, lr=a.lr, init="zero", seed=0)
        )
        # analysis error, per problem, from the optimised control
        with torch.no_grad():
            analysis = kae.analysis_field(res["control"])  # [B, X]
            truth = test.frames(
                torch.as_tensor(problem.sim, device=dev),
                torch.as_tensor(problem.t0, device=dev),
            )
            per = torch.linalg.vector_norm(
                analysis - truth, dim=-1
            ) / torch.linalg.vector_norm(truth, dim=-1)
        a_mean = float(per.mean())
        a_sem = float(per.std(unbiased=True) / np.sqrt(per.numel()))
        floor = roundtrip_floor(model, test, problem)
        fc = forecast_skill(kae, test, problem, FORECAST_LEADS)

        # cost of one iteration, and of one matrix_exp on this generator
        y = torch.as_tensor(problem.y, device=dev)
        mask = torch.as_tensor(problem.mask, device=dev)
        prep = kae.prepare(taus)
        c = torch.zeros((a.n_problems, dz), device=dev, requires_grad=True)
        opt = torch.optim.Adam([c], lr=a.lr)

        def it_step():
            opt.zero_grad(set_to_none=True)  # noqa: F821
            p = kae.predict_at(c, taus, prep)  # noqa: F821
            ((p - y) * mask[:, None, :]).pow(2).mean().backward()
            opt.step()  # noqa: F821

        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        ms_iter = median_ms(it_step, a.reps, min(10, a.reps), cuda)
        peak = torch.cuda.max_memory_allocated(dev) / 1024**2 if cuda else float("nan")
        ms_expm = median_ms(
            lambda: torch.matrix_exp(K * 0.7),  # noqa: F821
            a.reps,
            min(10, a.reps),
            cuda,
        )

        ev = torch.linalg.eigvals(K.detach().cpu().to(torch.float32))
        row = {
            "latent_dim": dz,
            "run": str(run),
            "analysis_rel_l2": a_mean,
            "analysis_sem": a_sem,
            "obs_rel_l2": (
                res["hist"]["obs_rel_l2"][-1] if res["hist"]["obs_rel_l2"] else None
            ),
            "ae_floor": floor,
            **fc,
            "ms_per_iter": ms_iter,
            "ms_matrix_exp": ms_expm,
            "peak_mem_MB": float(peak),
            "setup_s": res.get("setup_s"),
            "total_s": res.get("total_s"),
            "params": int(sum(p.numel() for p in model.parameters())),
            "max_re_eig": float(ev.real.max()),
            "n_unstable": int((ev.real > 0).sum()),
            "K_norm": float(
                torch.linalg.matrix_norm(K.detach().cpu().to(torch.float32), 2)
            ),
        }
        rows.append(row)
        logger.info(
            f"  N_z={dz:4d}  analysis {row['analysis_rel_l2']:.4f}  floor {floor:.4f}  "
            f"fc(2.5) {row['forecast_tau2.5']:.4f}  {ms_iter:6.2f} ms/it  "
            f"expm {ms_expm:6.3f} ms  {peak:7.1f} MB  maxRe {row['max_re_eig']:+.5f}"
        )
        del model, K, kae, c, opt, prep
        if cuda:
            torch.cuda.empty_cache()

    a.out.write_text(
        json.dumps(
            {
                "protocol": {
                    "taus": taus.tolist(),
                    "frames": CANONICAL_FRAMES.tolist(),
                    "delta_f": float(taus.min()),
                    "delta_l": float(taus.max()),
                    "N": len(taus),
                    "iters": a.iters,
                    "lr": a.lr,
                    "init": "zero",
                    "n_problems": a.n_problems,
                    "test": str(a.test),
                    "seed": 0,
                    "timing": f"median of {a.reps} reps after warm-up, batch {a.n_problems}",
                    "device": torch.cuda.get_device_name(0) if cuda else "cpu",
                    "note": (
                        "all models share the KS training recipe; only latent_dim differs. "
                        "Every row solves the identical Problem object."
                    ),
                },
                "rows": rows,
            },
            indent=1,
        )
    )
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
