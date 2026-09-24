"""Recover the analysis FIELDS for a geometry sweep that only stored errors.

The sweeps record rel-L2 per problem, which says how much is lost but never *what* is
lost.  This re-solves each sweep point for a couple of problems and keeps the recovered
state, so the sweep can be shown as trajectories rather than as a curve of scalars.
Cheap relative to the sweep itself: the cost is ``n_fields`` problems instead of 48.

Nothing about the protocol changes -- the schedules, seeds and budgets are read back from
the sweep's own JSON so the fields correspond to the points that were reported.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import torch

from data_assimilation.ks.protocol import DT, build_problem
from data_assimilation.ks.da_ks_experiments_3way import ALL, Bench, add_common_args

logger = logging.getLogger("sweep_fields")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sweep", default="C1_delta_f")
    ap.add_argument("--geo-dir", type=Path, default=Path("da_results_geometry"))
    add_common_args(ap)
    ap.add_argument("--n-fields", type=int, default=2)
    ap.add_argument(
        "--problem-index",
        type=int,
        default=None,
        help="solve ONE specific problem from the sweep's own draw. Index 0 "
        "is not neutral: in C1 it happens to be the U-Net's best case "
        "of 48 (0.0073 against a mean of 0.0987), so plotting it would "
        "misrepresent the sweep. Pick a representative index instead.",
    )
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument(
        "--rollout",
        type=int,
        default=30,
        help="frames to roll the analysis forward, for space-time views",
    )
    a = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    meta = json.loads((a.geo_dir / f"{a.sweep}.json").read_text())
    rows = meta["rows"]
    b = Bench(a)
    seed = {
        "C1_delta_f": 41,
        "C2_delta_l": 42,
        "C3_n_obs": 43,
        "noise_law": 44,
        "joint_sparsity": 45,
        "S1_single_obs": 51,
    }.get(a.sweep, 41)
    store = {
        "x": b.data.x,
        "dt": DT,
        "sweep": a.sweep,
        "tags": np.array([r["tag"] for r in rows]),
        "values": np.array([r.get("value", np.nan) for r in rows], dtype=float),
    }

    for r in rows:
        frames = np.asarray(r["frames"], dtype=int)
        kw = {}
        if "noise" in r:
            kw["noise_std"] = float(r["noise"])
        if "obs_frac" in r:
            kw["obs_frac"] = float(r["obs_frac"])
        if a.problem_index is None:
            prob = build_problem(
                b.data,
                name=r["tag"],
                n_problems=a.n_fields,
                taus=frames * DT,
                seed=seed,
                **kw,
            )
        else:
            # --n-sweep comes from add_common_args and MUST equal what the original
            # sweep used, or the seed draws different trajectories and the index means
            # something else.
            # draw the FULL set the sweep drew, then keep one problem, so the selected
            # trajectory is exactly the one the sweep reported at that index
            full = build_problem(
                b.data,
                name=r["tag"],
                n_problems=a.n_sweep,
                taus=frames * DT,
                seed=seed,
                **kw,
            )
            i = a.problem_index
            prob = dataclasses.replace(
                full,
                sim=full.sim[i : i + 1],
                t0=full.t0[i : i + 1],
                y=full.y[:, i : i + 1],
                mask=full.mask[:, i : i + 1],
            )
        sim = torch.as_tensor(prob.sim, device=b.dev)
        t0 = torch.as_tensor(prob.t0, device=b.dev)
        store[f"{r['tag']}__frames"] = frames
        store[f"{r['tag']}__truth_t0"] = (
            b.data.denorm(b.data.frames(sim, t0)).cpu().numpy()
        )
        if a.rollout > 0:  # rollout is optional: 0 = t_0 field only
            store[f"{r['tag']}__truth_roll"] = np.stack(
                [
                    b.data.denorm(b.data.frames(sim, t0 + k)).cpu().numpy()
                    for k in range(a.rollout)
                ],
                axis=1,
            )
        logger.info(f"{r['tag']}: frames {frames.tolist()}")
        for m in ALL:
            res = b.run(
                m, prob, iters=a.iters, seed=seed, window_frames=(a.rollout or None)
            )
            if res.get("skipped"):
                continue
            store[f"{r['tag']}__{m}__analysis"] = np.asarray(res["analysis"])
            if "spacetime" in res:
                # [T, B, X] -> [B, T, X], to match truth_roll
                store[f"{r['tag']}__{m}__roll"] = np.transpose(
                    np.asarray(res["spacetime"])[: a.rollout], (1, 0, 2)
                )
            logger.info(f"    {m:9s} rel-L2 {np.mean(res['rel']):.4f}")
        np.savez_compressed(a.geo_dir / f"{a.sweep}_fields.npz", **store)
    logger.info(f"saved -> {a.geo_dir / (a.sweep + '_fields.npz')}")


if __name__ == "__main__":
    main()
