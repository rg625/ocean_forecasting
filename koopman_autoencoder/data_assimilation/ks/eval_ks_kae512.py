# ruff: noqa: E741
# mypy: disable-error-code="index"
"""Re-run every KS KAE number with a different checkpoint, KAE only.

The KS tables in the report were measured with the d_z = 128 KAE. This reproduces the
same problems -- same seeds, schedules, budgets and problem counts as the campaign -- and
solves them with another checkpoint, so the KAE column can be swapped without touching
U-Net or SDA, which do not depend on the KAE at all.

The headline section also re-runs the ORIGINAL checkpoint at each candidate iteration
count. The campaign's iteration count for that table is not recorded in its output, so the
control identifies it: whichever budget reproduces the published 0.0049 is the one to use
for the replacement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_assimilation.ks.evaluate import ae_reconstruction_floor
from data_assimilation.ks.exp_geometry import schedule
from data_assimilation.ks.init_utils import init_kwargs
from data_assimilation.ks.methods import KAE4DVar, SolveConfig
from data_assimilation.ks.models_io import load_kae
from data_assimilation.ks.protocol import DT, KSData, build_problem, rel_l2

CANON = [1, 3, 7, 15, 25]
NOISE = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4]
C2_DL = {
    1: [5, 10, 25, 50, 100, 200, 400, 700],
    9: [13, 25, 50, 100, 200, 400, 700],
    16: [20, 25, 50, 100, 200, 400, 700],
}
C3_N = {1: [2, 3, 5, 9, 17, 25], 9: [2, 3, 5, 9, 13, 25], 16: [2, 3, 5, 9, 13, 25]}
JS_FRAC, JS_N = [1.0, 0.5, 0.25, 0.1], [2, 3, 5, 9, 17]


def solve(kae, K, D, data, prob, iters, mode="expm", lr=0.01, seed=0, init="zero"):
    m = KAE4DVar(kae, K, D, 1.0, mode)
    ik = init_kwargs(init, kae, K, prob, data.device)
    r = m.solve(prob, data, SolveConfig(iters=iters, lr=lr, seed=seed, **ik))
    with torch.no_grad():
        u0 = data.frames(
            torch.as_tensor(prob.sim, device=data.device),
            torch.as_tensor(prob.t0, device=data.device),
        )
        v = (
            rel_l2(data.denorm(m.analysis_field(r["control"])), data.denorm(u0))
            .cpu()
            .numpy()
        )
    return {
        "mean": float(v.mean()),
        "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
        "median": float(np.median(v)),
        "std": float(v.std(ddof=1)),
        "n": int(v.size),
        "wall_s": float(r["total_s"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--kae-run", default="model_outputs_ks_latentdim/dz_512/run-20260906_140222"
    )
    ap.add_argument(
        "--control-run",
        default="model_outputs_ks/continous_linear_128/rollout_10",
        help="the checkpoint the report's KS tables were measured with",
    )
    ap.add_argument("--test", type=Path, default=Path("data/ks/da_test.nc"))
    ap.add_argument(
        "--out", type=Path, default=Path("da_results_geometry_df9/kae512.json")
    )
    ap.add_argument(
        "--tuning",
        type=Path,
        default=None,
        help="kae_tuning.json from tune_ks_kae_init; its frozen 'best' learning "
        "rate and initialisation are applied to every point",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--delta-f",
        type=int,
        nargs="+",
        default=[1, 9],
        help="which pinned delta_f to replay the sweeps at",
    )
    ap.add_argument(
        "--skip-headline",
        action="store_true",
        help="the headline is on the canonical schedule and does not depend on "
        "the pinned delta_f, so it need not be repeated per delta_f",
    )
    a = ap.parse_args()
    dev = torch.device(a.device)
    data = KSData(a.test, dev)
    hp = {"lr": 0.01, "init": "zero"}
    if a.tuning is not None:
        hp.update(json.loads(a.tuning.read_text())["best"])
    print(f"frozen hyper-parameters (chosen on validation): {hp}", flush=True)
    for df in a.delta_f:
        assert (
            df in C2_DL and df in C3_N
        ), f"no delta_l / N grid defined for delta_f={df}"
    new = load_kae(Path(a.kae_run), None, dev)
    ctl = load_kae(Path(a.control_run), None, dev)
    out = {
        "meta": {
            "kae_run": a.kae_run,
            "control_run": a.control_run,
            "hp": hp,
            "note": "KAE only; U-Net and SDA are unaffected by the KAE checkpoint",
        },
        "headline": [],
        "sweeps": [],
    }

    def flush():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2))

    # --- headline: 64 trajectories, canonical schedule -----------------------
    prob = build_problem(
        data, name="F", n_problems=64, taus=np.array(CANON) * DT, seed=123
    )
    for tag, (kae, K, D) in (
        () if a.skip_headline else (("control_dz128", ctl), ("new", new))
    ):
        for iters in (1200, 8000):
            for mode in ("expm", "rk4"):
                r = solve(kae, K, D, data, prob, iters, mode=mode, seed=123, **hp)
                r.update(ckpt=tag, iters=iters, mode=mode)
                out["headline"].append(r)
                print(
                    f"headline {tag:13s} {mode:4s} iters={iters:5d} "
                    f"rel {r['mean']:.4f} median {r['median']:.4f} "
                    f"[{r['wall_s']:.0f}s]",
                    flush=True,
                )
                flush()
    if not a.skip_headline:
        out["meta"]["ae_floor_new"] = float(
            ae_reconstruction_floor(new[0], data, prob).mean()
        )

    # --- sweeps, both delta_f, new checkpoint only ---------------------------
    def run_points(section, df, points, iters, n_problems, seed):
        for i, (label, frames, kw) in enumerate(points):
            p = build_problem(
                data,
                name=f"{section}{i}",
                n_problems=n_problems,
                taus=np.asarray(frames) * DT,
                seed=seed,
                **kw,
            )
            r = solve(new[0], new[1], new[2], data, p, iters, seed=seed, **hp)
            r.update(
                section=section,
                delta_f=df,
                label=label,
                frames=list(map(int, frames)),
                iters=iters,
                **{k: v for k, v in kw.items()},
            )
            out["sweeps"].append(r)
            print(
                f"{section:14s} df={df} {label:18s} rel {r['mean']:.4f} "
                f"[{r['wall_s']:.0f}s]",
                flush=True,
            )
            flush()

    out["meta"]["delta_f"] = list(a.delta_f)
    for df in a.delta_f:
        run_points(
            "C2_delta_l",
            df,
            [(f"dl={l}", schedule(df, l, 5), {}) for l in C2_DL[df]],
            iters=1000,
            n_problems=24,
            seed=42,
        )
        run_points(
            "C3_n_obs",
            df,
            [(f"N={n}", schedule(df, 25, n), {}) for n in C3_N[df]],
            iters=8000,
            n_problems=48,
            seed=43,
        )
        fr = CANON if df == 1 else schedule(df, 25, 5)
        run_points(
            "noise_law",
            df,
            [
                (f"{d}_{s}", fr, {"noise_std": s, "noise_dist": d})
                for d in ("gaussian", "laplace")
                for s in NOISE
            ],
            iters=8000,
            n_problems=48,
            seed=44,
        )
        run_points(
            "joint_sparsity",
            df,
            [
                (f"f{f}_n{n}", schedule(df, 25, n), {"obs_frac": f})
                for f in JS_FRAC
                for n in JS_N
            ],
            iters=8000,
            n_problems=48,
            seed=45,
        )
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
