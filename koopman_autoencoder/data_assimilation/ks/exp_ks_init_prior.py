# mypy: disable-error-code="var-annotated"
"""Two ways to constrain the directions the observations do not see.

init   z0 starts from the nearest observation propagated back, e^{-K delta_f} enc(y_1),
       instead of zero. Uses observed data only -- never the truth at t_0.
prior  a background term is added to the 4D-Var cost. `ridge` is lambda ||z||^2, the crude
       isotropic version already supported by the solver; `gauss` is the proper background
       term (z - mu)^T Sigma^-1 (z - mu) with mu and Sigma estimated from encoded TRAINING
       states, which is the standard B matrix of variational assimilation.

Both target the diagnosis: at large delta_f the observations leave part of the latent
unconstrained, and the cost then prefers a latent that decodes far from the truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from data_assimilation.ks.exp_geometry import schedule
from data_assimilation.ks.methods import KAE4DVar, SolveConfig
from data_assimilation.ks.models_io import load_kae
from data_assimilation.ks.protocol import DT, KSData, build_problem, rel_l2

SDA = {1: 0.0094, 4: 0.0102, 9: 0.0192, 16: 0.1739}


def encode(kae, u):
    return kae.present_encoding(
        TensorDict({"u": u.unsqueeze(1).unsqueeze(-1)}, batch_size=[u.shape[0], 1]),
        cond_input=None,
    )


def latent_stats(kae, data, path, n_states=4096, seed=0):
    """mu and Sigma of the latent over TRAINING states (never the test set)."""
    tr = KSData(path, data.device)
    rng = np.random.default_rng(seed)
    sim = torch.as_tensor(rng.integers(0, tr.n_sim, n_states), device=tr.device)
    t = torch.as_tensor(rng.integers(0, tr.n_t, n_states), device=tr.device)
    with torch.no_grad():
        z = torch.cat(
            [
                encode(kae, tr.frames(sim[i : i + 256], t[i : i + 256]))
                for i in range(0, n_states, 256)
            ]
        )
    mu = z.mean(0)
    zc = z - mu
    cov = (zc.T @ zc) / (len(z) - 1)
    d = cov.shape[0]
    cov = cov + 1e-6 * torch.eye(d, device=cov.device) * torch.diagonal(cov).mean()
    return mu, torch.linalg.inv(cov), float(torch.diagonal(cov).mean())


def run(m, prob, data, u0, **kw):
    res = m.solve(prob, data, SolveConfig(**kw))
    with torch.no_grad():
        v = rel_l2(data.denorm(m.analysis_field(res["control"])), data.denorm(u0))
    return float(v.mean().cpu()), float((v.std(unbiased=True) / np.sqrt(len(v))).cpu())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--kae-run", default="model_outputs_ks_latentdim/dz_512/run-20260906_140222"
    )
    ap.add_argument("--test", type=Path, default=Path("data/ks/da_test.nc"))
    ap.add_argument("--train", type=Path, default=Path("data/ks/train.nc"))
    ap.add_argument(
        "--json", type=Path, default=Path("da_results_geometry_df9/init_prior.json")
    )
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 9, 16])
    ap.add_argument(
        "--lambdas", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    )
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--n-problems", type=int, default=16)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    data = KSData(a.test, dev)
    kae, K, D = load_kae(Path(a.kae_run), None, dev)
    rows = []

    def flush():
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(
            json.dumps(
                {
                    "meta": {
                        "d_z": int(D),
                        "iters": a.iters,
                        "lr": a.lr,
                        "n_problems": a.n_problems,
                        "seed": a.seed,
                        "sda": SDA,
                    },
                    "rows": rows,
                },
                indent=2,
            )
        )

    for df in a.deltas:
        fr = schedule(df, 25, 5)
        prob = build_problem(
            data, name=f"IP{df}", n_problems=a.n_problems, taus=fr * DT, seed=a.seed
        )
        sim = torch.as_tensor(prob.sim, device=dev)
        t0 = torch.as_tensor(prob.t0, device=dev)
        u0 = data.frames(sim, t0)
        m = KAE4DVar(kae, K, D, 1.0, "expm")
        base = dict(iters=a.iters, lr=a.lr, seed=a.seed)

        mean, sem = run(m, prob, data, u0, init="zero", **base)
        rows.append(
            {"delta_f": df, "setting": "baseline (zero init)", "mean": mean, "sem": sem}
        )
        print(f"df={df:2d} {'baseline':28s} {mean:.4f}", flush=True)
        flush()

        # informed start: the nearest observation, encoded and propagated back
        with torch.no_grad():
            y1 = torch.as_tensor(prob.y[0], device=dev)  # nearest observation
            z_obs = encode(kae, y1)
            z_init = z_obs @ torch.matrix_exp(-K * float(fr[0] * DT)).T
        mean, sem = run(m, prob, data, u0, init="given", init_value=z_init, **base)
        rows.append(
            {"delta_f": df, "setting": "informed init", "mean": mean, "sem": sem}
        )
        print(f"df={df:2d} {'informed init':28s} {mean:.4f}", flush=True)
        flush()

        for lam in a.lambdas:
            mean, sem = run(m, prob, data, u0, init="zero", reg=lam, **base)
            rows.append(
                {
                    "delta_f": df,
                    "setting": f"ridge lambda={lam:g}",
                    "mean": mean,
                    "sem": sem,
                }
            )
            print(f"df={df:2d} ridge lambda={lam:<8g}       {mean:.4f}", flush=True)
            flush()
        print(
            f"df={df:2d} SDA reference                {SDA.get(df, float('nan')):.4f}",
            flush=True,
        )
    print("saved ->", a.json)


if __name__ == "__main__":
    main()
