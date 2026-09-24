# ruff: noqa: E731
"""Analysis error on CHAOTIC problems, against a random draw from the same pool.

The random draw that every KS number in the campaign uses contains a lot of quiet
stretches: over the 2.5 t.u. assimilation window the median trajectory barely turns. This
picks the problems whose truth actually turns (path length in state space over net
displacement, measured on a long window) and re-runs the same methods on them, so the
ordering can be checked where the dynamics are hardest.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from data_assimilation.ks.protocol import DT, build_problem
from data_assimilation.ks.da_ks_experiments_3way import ALL, Bench, add_common_args

SCHEDULES = {1: [1, 3, 7, 15, 25], 9: [9, 12, 15, 19, 25]}


def subsets(b, frames, n_cand, n_keep, window, seed):
    prob = build_problem(
        b.data, name="cand", n_problems=n_cand, taus=np.array(frames) * DT, seed=seed
    )
    ok = np.where(prob.t0 + window - 1 < b.data.n_t)[0]
    sim = torch.as_tensor(prob.sim[ok], device=b.dev)
    t0 = torch.as_tensor(prob.t0[ok], device=b.dev)
    u = np.stack(
        [
            b.data.denorm(b.data.frames(sim, t0 + j)).cpu().numpy()
            for j in range(window)
        ],
        axis=1,
    )
    nrm = lambda a: np.linalg.norm(a, axis=-1)
    drift = nrm(u[:, -1] - u[:, 0]) / nrm(u[:, 0])
    wiggle = nrm(np.diff(u, axis=1)).sum(1) / nrm(u[:, 0]) / np.maximum(drift, 1e-6)
    chaotic = ok[np.argsort(wiggle)[::-1][:n_keep]]
    random = ok[:n_keep]
    cut = lambda idx: dataclasses.replace(
        prob, sim=prob.sim[idx], t0=prob.t0[idx], y=prob.y[:, idx]
    )
    return (
        {"chaotic": cut(chaotic), "random": cut(random)},
        {
            "chaotic": wiggle[np.argsort(wiggle)[::-1][:n_keep]],
            "random": wiggle[:n_keep],
        },
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument(
        "--out", type=Path, default=Path("da_results_geometry_df9/chaotic.json")
    )
    ap.add_argument("--n-keep", type=int, default=16)
    ap.add_argument("--n-candidates", type=int, default=200)
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=43)
    a = ap.parse_args()
    b = Bench(a)
    rows = []
    for df, frames in SCHEDULES.items():
        subs, wig = subsets(b, frames, a.n_candidates, a.n_keep, a.window, a.seed)
        for tag, prob in subs.items():
            row = {
                "delta_f": df,
                "subset": tag,
                "n": int(len(prob.sim)),
                "wiggle_mean": float(np.mean(wig[tag])),
                "wiggle_min": float(np.min(wig[tag])),
            }
            for m in ALL:
                r = b.run(m, prob, iters=a.iters, seed=a.seed)
                v = np.asarray(r["rel"], dtype=float)
                row[m] = {
                    "mean": float(np.nanmean(v)),
                    "sem": float(np.nanstd(v, ddof=1) / np.sqrt(v.size)),
                    "n_diverged": int((~np.isfinite(v)).sum()),
                }
                print(
                    f"df={df} {tag:8s} {m:9s} {row[m]['mean']:.4f} "
                    f"+-{row[m]['sem']:.4f}",
                    flush=True,
                )
            rows.append(row)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(
                json.dumps(
                    {
                        "meta": {
                            "n_candidates": a.n_candidates,
                            "window": a.window,
                            "iters": a.iters,
                            "seed": a.seed,
                            "metric": "path length / net displacement over the window",
                        },
                        "rows": rows,
                    },
                    indent=2,
                )
            )
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
