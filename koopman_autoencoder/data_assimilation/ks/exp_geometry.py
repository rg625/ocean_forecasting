# ruff: noqa: E741
# mypy: disable-error-code="no-any-return"
"""Temporal geometry, noise-law and joint-sparsity experiments (parts C1-C3, 6, 7).

Every section here separates the three quantities that the canonical setup conflates:

    delta_f = t1 - t0     lead time to the FIRST observation
    delta_l = tN - t0     lead time to the LAST observation  (the recovery horizon)
    N                     number of observation times

The canonical schedule CANON = [1, 3, 7, 15, 25] frames has delta_f = 0.1, delta_l = 2.5
and N = 5 simultaneously, so a single number from it cannot be attributed to any one of
them.  C1/C2/C3 each vary exactly one and pin the other two.  Observations remain
IRREGULARLY sampled: delta_l is NOT (N - 1) * delta_f.

Horizons are additionally reported in Lyapunov times using the measured lambda_1 from
`data_assimilation/ks/lyapunov.py`, because "2.5" in the supervisor's brief refers to time units, not
Lyapunov times -- on this system T_L is about 22.6 time units, so the canonical horizon
is roughly 0.11 T_L.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from data_assimilation.ks.protocol import DT, build_problem
from data_assimilation.ks.da_ks_experiments_3way import (
    ALL,
    Bench,
    ae_reconstruction_floor,
    _save,
    add_common_args,
)

logger = logging.getLogger("geometry")


# ---------------------------------------------------------------------------
# observation schedules
# ---------------------------------------------------------------------------
def schedule(delta_f: int, delta_l: int, n: int) -> np.ndarray:
    """N observation frames spanning [delta_f, delta_l] inclusive.

    Endpoints are pinned exactly so that varying N never changes delta_f or delta_l.
    Interior points are geometrically spaced, which keeps the schedule irregular and
    puts more of the budget near t0 where the information is.  Rounding to the frame
    grid can collide, so duplicates are dropped and the realised N is reported.
    """
    if n == 1:
        return np.array([delta_f], dtype=int)
    g = np.geomspace(delta_f, delta_l, n)
    f = np.unique(np.round(g).astype(int))
    f = f[(f >= delta_f) & (f <= delta_l)]
    return np.unique(np.concatenate([[delta_f], f, [delta_l]]))


def describe(frames: np.ndarray) -> str:
    return (
        f"frames={list(map(int, frames))} "
        f"t={[round(float(x) * DT, 2) for x in frames]} "
        f"delta_f={frames[0] * DT:.2f} delta_l={frames[-1] * DT:.2f} N={len(frames)}"
    )


# ---------------------------------------------------------------------------
# generic runner
# ---------------------------------------------------------------------------
def _run_points(
    b: Bench,
    points: List[Dict],
    *,
    n_problems: int,
    iters: int,
    seed: int,
    build_kw=None,
    flush=None,
    methods=None,
) -> Dict:
    """Run every method on every point.  `points` carry the schedule and any extras.

    Results are flushed to disk after every point.  These sweeps take hours on a shared
    GPU and have been killed mid-run before; without an incremental write a kill loses the
    whole section rather than the point in flight.
    """
    build_kw = build_kw or {}
    rows, store = [], {}
    for pi, pt in enumerate(points):
        fr = pt["frames"]
        kw = dict(build_kw)
        kw.update(pt.get("build", {}))
        prob = build_problem(
            b.data,
            name=f"{pt['tag']}",
            n_problems=n_problems,
            taus=fr * DT,
            seed=seed,
            **kw,
        )
        it = int(pt.get("iters", iters))
        row = {
            "value": pt["value"],
            "tag": pt["tag"],
            "iters": it,
            "frames": list(map(int, fr)),
            "taus": (fr * DT).tolist(),
            "delta_f": float(fr[0] * DT),
            "delta_l": float(fr[-1] * DT),
            "N": int(len(fr)),
            "ae_floor": float(ae_reconstruction_floor(b.kae, b.data, prob).mean()),
        }
        logger.info(
            f"  [{pi + 1}/{len(points)}] {pt['tag']}: {describe(fr)}  iters={it}"
        )
        for m in methods or ALL:
            r = b.run(m, prob, iters=it, seed=seed)
            if r.get("skipped"):
                row[m] = dict(mean=np.nan, sem=np.nan, n=0)
                continue
            v = np.asarray(r["rel"], dtype=float)
            row[m] = dict(
                mean=float(v.mean()),
                sem=float(v.std(ddof=1) / np.sqrt(len(v))),
                std=float(v.std(ddof=1)),
                n=int(len(v)),
                wall_s=float(r["wall_s"]),
            )
            if "rel_posterior_mean" in r:
                row[m]["posterior_mean"] = float(np.mean(r["rel_posterior_mean"]))
            store[f"{pt['tag']}__{m}__rel"] = v
            logger.info(
                f"        {m:9s} rel-L2 {v.mean():.4f} ± "
                f"{v.std(ddof=1) / np.sqrt(len(v)):.4f} (SEM, n={len(v)})"
                f"  [{r['wall_s']:.0f}s]"
            )
        rows.append(row)
        if flush is not None:
            flush({"rows": rows, "per_problem": store})
    return {"rows": rows, "per_problem": store}


def add_anchor(pts: List[Dict], a, tag_match) -> List[Dict]:
    """Repeat one point of a sweep at the main-campaign budget.

    Every sweep here runs at a constant, reduced iteration count so that the quantity
    being varied is not confounded with the optimisation budget, and so that the campaign
    fits in the available GPU time: at 8000 iterations these five sweeps would cost well
    over a hundred GPU-hours, because 4D-Var cost is linear in the iteration count and, for
    the U-Net, in the rollout length as well.  The convergence study showed both 4D-Var
    methods follow clean power laws in the iteration count with no plateau, so a change of
    budget shifts a curve almost multiplicatively rather than changing its shape -- which
    is what these sweeps are about.  The anchor MEASURES that shift instead of assuming it.
    """
    if not a.anchor_iters or a.anchor_iters == a.iters:
        return pts
    for pt in pts:
        if tag_match(pt):
            q = dict(pt)
            q["tag"] = pt["tag"] + "_anchor"
            q["iters"] = a.anchor_iters
            return pts + [q]
    raise ValueError(f"no anchor point matched in {[p['tag'] for p in pts]}")


def _dump(out: Path, name: str, payload: Dict, meta: Dict) -> None:
    js = {"meta": meta, "rows": payload["rows"]}
    (out / f"{name}.json").write_text(json.dumps(js, indent=2))
    arrs = {k: v for k, v in payload["per_problem"].items()}
    arrs["_rows_json"] = np.array(json.dumps(js))
    _save(out, name, **arrs)
    logger.info(f"saved -> {out / name}.json / .npz")


# ---------------------------------------------------------------------------
# C1 / C2 / C3
# ---------------------------------------------------------------------------
def exp_C1(b, out, a):
    """delta_f varies; delta_l and N pinned."""
    logger.info("=== C1. lead time to the FIRST observation (delta_l, N fixed) ===")
    dl, n = a.c1_delta_l, a.c1_n
    pts = [
        {"value": f * DT, "tag": f"C1_df{f}", "frames": schedule(f, dl, n)}
        for f in a.c1_delta_f
    ]
    pts = add_anchor(pts, a, lambda pt: pt["frames"][0] == 1)
    p = _run_points(
        b,
        pts,
        n_problems=a.n_sweep,
        iters=a.iters,
        seed=41,
        methods=a.methods,
        flush=lambda pp: _dump(
            out, "C1_delta_f", pp, {"partial": True, "varies": "delta_f"}
        ),
    )
    _dump(
        out,
        "C1_delta_f",
        p,
        {
            "varies": "delta_f",
            "fixed": {"delta_l": dl * DT, "N": n},
            "iters": a.iters,
            "n_problems": a.n_sweep,
        },
    )


def exp_C2(b, out, a):
    """delta_l varies; delta_f and N pinned.  This is the recovery-horizon sweep."""
    logger.info("=== C2. RECOVERY HORIZON delta_l (delta_f, N fixed) ===")
    df = a.c2_delta_f if a.c2_delta_f is not None else a.pin_delta_f
    n = a.c2_n
    # a horizon at or inside the first observation is not a window at all
    pts = [
        {"value": l * DT, "tag": f"C2_dl{l}", "frames": schedule(df, l, n)}
        for l in a.c2_delta_l
        if l > df
    ]
    # The 4D-Var cost per iteration grows with the rollout length, so running this sweep
    # at the campaign's 8000 iterations would cost about 72 GPU-hours for the U-Net rows
    # alone. The budget is therefore held CONSTANT at a lower value across the whole
    # sweep -- so the horizon effect is not confounded with the budget -- and one anchor
    # point is repeated at the campaign budget to MEASURE the offset between the two
    # rather than assume it.
    if a.c2_anchor in a.c2_delta_l and a.c2_anchor_iters != a.iters_long:
        pts.append(
            {
                "value": a.c2_anchor * DT,
                "tag": f"C2_dl{a.c2_anchor}_anchor",
                "frames": schedule(df, a.c2_anchor, n),
                "iters": a.c2_anchor_iters,
            }
        )
    p = _run_points(
        b,
        pts,
        n_problems=a.n_sweep_long,
        iters=a.iters_long,
        seed=42,
        methods=a.methods,
        flush=lambda pp: _dump(
            out, "C2_delta_l", pp, {"partial": True, "varies": "delta_l"}
        ),
    )
    _dump(
        out,
        "C2_delta_l",
        p,
        {
            "varies": "delta_l",
            "fixed": {"delta_f": df * DT, "N": n},
            "iters": a.iters_long,
            "n_problems": a.n_sweep_long,
            "anchor": {
                "delta_l_frames": a.c2_anchor,
                "iters": a.c2_anchor_iters,
                "why": (
                    "repeats one horizon at the main-campaign budget so the "
                    "offset between the two budgets is measured, not assumed"
                ),
            },
            "budget_note": (
                "iters is CONSTANT across the sweep; wall_s per point is "
                "recorded so the cost-vs-horizon scaling is visible."
            ),
        },
    )


def exp_C3(b, out, a):
    """N varies; delta_f and delta_l pinned exactly."""
    logger.info("=== C3. NUMBER of observations (delta_f, delta_l fixed) ===")
    df = a.c3_delta_f if a.c3_delta_f is not None else a.pin_delta_f
    dl = a.c3_delta_l
    pts = [
        {"value": n, "tag": f"C3_N{n}", "frames": schedule(df, dl, n)} for n in a.c3_n
    ]
    pts = add_anchor(pts, a, lambda pt: len(pt["frames"]) == 5)
    p = _run_points(
        b,
        pts,
        n_problems=a.n_sweep,
        iters=a.iters,
        seed=43,
        methods=a.methods,
        flush=lambda pp: _dump(out, "C3_n_obs", pp, {"partial": True, "varies": "N"}),
    )
    _dump(
        out,
        "C3_n_obs",
        p,
        {
            "varies": "N",
            "fixed": {"delta_f": df * DT, "delta_l": dl * DT},
            "iters": a.iters,
            "n_problems": a.n_sweep,
        },
    )


# ---------------------------------------------------------------------------
# noise law
# ---------------------------------------------------------------------------
def exp_noise_law(b, out, a):
    """Gaussian vs Laplace observation noise at MATCHED standard deviation.

    Poisson is deliberately not tested: the KS state u(x, t) is a signed, continuous,
    zero-mean field with no count interpretation and no non-negativity constraint, so a
    Poisson likelihood is not merely a worse fit but undefined on roughly half the domain.
    Laplace is the meaningful stress test -- same variance, heavier tails, so a small
    fraction of observations are badly corrupted while the rest are cleaner than Gaussian.
    None of the three methods uses a Laplace likelihood: the 4D-Var cost stays quadratic
    and SDA keeps its Gaussian likelihood approximation, so this measures ROBUSTNESS to
    misspecification, identically for all of them.
    """
    logger.info("=== NOISE LAW: Gaussian vs Laplace at matched std ===")
    # the canonical schedule when delta_f = 1, otherwise the same geometric rule as C1-C3
    fr = (
        np.array(a.canon)
        if a.pin_delta_f == 1
        else schedule(a.pin_delta_f, a.canon[-1], len(a.canon))
    )
    pts = []
    for dist in ("gaussian", "laplace"):
        for s in a.noise:
            pts.append(
                {
                    "value": s,
                    "tag": f"NL_{dist}_{s}",
                    "frames": fr,
                    "build": {"noise_std": s, "noise_dist": dist},
                }
            )
    pts = add_anchor(pts, a, lambda pt: pt["tag"] == "NL_gaussian_0.0")
    p = _run_points(
        b,
        pts,
        n_problems=a.n_sweep,
        iters=a.iters,
        seed=44,
        methods=a.methods,
        flush=lambda pp: _dump(
            out, "noise_law", pp, {"partial": True, "varies": "noise"}
        ),
    )
    for r in p["rows"]:
        r.setdefault("dist", r["tag"].split("_")[1])
    _dump(
        out,
        "noise_law",
        p,
        {
            "varies": "noise_std x distribution",
            "schedule": list(map(int, fr)),
            "fixed": {
                "delta_f": float(fr[0] * DT),
                "delta_l": float(fr[-1] * DT),
                "N": int(len(fr)),
            },
            "iters": a.iters,
            "n_problems": a.n_sweep,
            "note": "Laplace scale b = std/sqrt(2) so both laws have identical variance.",
        },
    )


# ---------------------------------------------------------------------------
# joint spatial x temporal sparsity
# ---------------------------------------------------------------------------
def exp_joint_sparsity(b, out, a):
    """2-D sweep: fraction of sensors x number of observation times.

    delta_f and delta_l are pinned across the whole grid, so moving along the temporal
    axis changes ONLY how densely the same interval is sampled.
    """
    logger.info("=== JOINT spatial x temporal sparsity ===")
    df = a.c3_delta_f if a.c3_delta_f is not None else a.pin_delta_f
    dl = a.c3_delta_l
    # N must be at least 2 here: schedule() pins BOTH endpoints, and with n = 1 it can
    # only honour delta_f, which would silently collapse delta_l to 0.1 in that column
    # and contradict the claim that the temporal axis changes only the sampling density.
    assert (
        min(a.js_n) >= 2
    ), "joint sparsity needs N >= 2 to pin both delta_f and delta_l"
    pts = []
    for fr_s in a.js_frac:
        for n in a.js_n:
            f = schedule(df, dl, n)
            assert f[0] == df and f[-1] == dl, (f, df, dl)
            pts.append(
                {
                    "value": [fr_s, n],
                    "tag": f"JS_f{fr_s}_n{n}",
                    "frames": f,
                    "build": {"obs_frac": fr_s},
                }
            )
    pts = add_anchor(
        pts, a, lambda pt: pt["build"]["obs_frac"] == 1.0 and len(pt["frames"]) == 5
    )
    p = _run_points(
        b,
        pts,
        n_problems=a.n_sweep,
        iters=a.iters_sparse,
        seed=45,
        methods=a.methods,
        flush=lambda pp: _dump(
            out,
            "joint_sparsity",
            pp,
            {"partial": True, "axes": {"obs_frac": a.js_frac, "n_times": a.js_n}},
        ),
    )
    for r in p["rows"]:
        r["obs_frac"], r["n_times"] = r["value"][0], r["value"][1]
        r["total_scalars"] = int(round(r["obs_frac"] * b.data.X)) * r["N"]
    _dump(
        out,
        "joint_sparsity",
        p,
        {
            "axes": {"obs_frac": a.js_frac, "n_times": a.js_n},
            "fixed": {"delta_f": df * DT, "delta_l": dl * DT},
            "iters": a.iters_sparse,
            "n_problems": a.n_sweep,
            "state_dim": int(b.data.X),
        },
    )


def exp_S1_single_obs(b: Bench, out: Path, a) -> None:
    """ONE observation, at increasing distance from t_0.

    Every other experiment gives the assimilation several observations, so the recovery is
    never attributable to any one of them.  Here the trajectory is observed EXACTLY ONCE,
    at t_0 + tau, and tau is swept.  With N = 1 the three geometry parameters collapse
    (delta_f = delta_l = tau, N = 1), which makes this the cleanest possible measurement of
    how far a single observation can be and still constrain the unobserved state.

    The reference to read it against is the no-skill level: once the recovery error reaches
    the error between two independent states, that observation carries no usable
    information about t_0 any more.
    """
    TL = json.loads(Path("da_results_sda_paper/lyapunov.json").read_text())[
        "lyapunov_time"
    ]
    logger.info("=== S1. A SINGLE observation, swept in distance from t_0 ===")
    pts = []
    for f in a.s1_frames:
        pts.append(
            {
                "tag": f"S1_tau{f}",
                "frames": np.array([int(f)]),
                "value": float(f) * DT,
                "extra": {
                    "tau_frames": int(f),
                    "tau_tu": float(f) * DT,
                    "tau_TL": float(f) * DT / TL,
                },
            }
        )
        logger.info(
            f"  [{len(pts)}/{len(a.s1_frames)}] tau = {f} frames "
            f"= {f * DT:.1f} t.u. = {f * DT / TL:.3f} T_L"
        )
    p = _run_points(
        b,
        pts,
        n_problems=a.n_sweep,
        iters=a.iters,
        seed=51,
        methods=a.methods,
        flush=lambda pp: _dump(
            out, "S1_single_obs", pp, {"partial": True, "varies": "tau (N=1)"}
        ),
    )
    for r, pt in zip(p["rows"], pts):
        r.update(pt["extra"])
    _dump(
        out,
        "S1_single_obs",
        p,
        {
            "varies": "distance to the ONE observation",
            "fixed": {"N": 1},
            "iters": a.iters,
            "n_problems": a.n_sweep,
            "T_L": TL,
            "note": (
                "With a single observation delta_f = delta_l = tau and N = 1, so the "
                "three temporal-geometry parameters collapse into one axis."
            ),
        },
    )
    logger.info(f"saved -> {out / 'S1_single_obs'}.json")


SECTIONS = {
    "S1": exp_S1_single_obs,
    "C1": exp_C1,
    "C2": exp_C2,
    "C3": exp_C3,
    "NL": exp_noise_law,
    "JS": exp_joint_sparsity,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(ap)
    # da_results_geometry holds the original delta_f = 1 campaign (and the C1 sweep the
    # notebook still reads); the default keeps a pinned re-run from overwriting it
    ap.add_argument("--out-dir", type=Path, default=Path("da_results_geometry_df9"))
    ap.add_argument("--sections", nargs="+", default=["C1", "C3", "NL", "JS", "C2"])
    ap.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=ALL,
        help="subset of methods to run; the default is all of them. Use this "
        "when another script already supplies a column, so the expensive "
        "sweep does not recompute it",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=8000,
        help="budget held CONSTANT across a sweep, so the swept quantity is "
        "not confounded with the optimisation budget",
    )
    ap.add_argument("--iters-sparse", type=int, default=8000)
    ap.add_argument(
        "--anchor-iters",
        type=int,
        default=0,
        help="repeat one canonical point of each sweep at the main-campaign "
        "budget, to MEASURE the offset between the two budgets",
    )
    ap.add_argument(
        "--iters-long",
        type=int,
        default=1000,
        help="budget for the long-horizon C2 sweep; kept CONSTANT across the "
        "sweep so the horizon effect is not confounded with budget",
    )
    ap.set_defaults(n_sweep=48)  # --n-sweep comes from add_common_args
    ap.add_argument("--n-sweep-long", type=int, default=24)
    ap.add_argument("--canon", type=int, nargs="+", default=[1, 3, 7, 15, 25])
    ap.add_argument(
        "--pin-delta-f",
        type=int,
        default=9,
        help="delta_f (frames) held fixed by C2, C3, NL and JS; 1 reproduces "
        "the original campaign, which used --canon",
    )
    # C1
    ap.add_argument("--c1-delta-f", type=int, nargs="+", default=[1, 2, 4, 8, 12, 18])
    ap.add_argument("--c1-delta-l", type=int, default=25)
    ap.add_argument("--c1-n", type=int, default=5)
    # C2 -- 13 is the shortest horizon that holds N = 5 distinct frames after delta_f = 9
    ap.add_argument(
        "--c2-delta-l", type=int, nargs="+", default=[13, 25, 50, 100, 200, 400, 700]
    )
    ap.add_argument(
        "--c2-delta-f", type=int, default=None, help="defaults to --pin-delta-f"
    )
    ap.add_argument("--c2-n", type=int, default=5)
    ap.add_argument(
        "--c2-anchor",
        type=int,
        default=25,
        help="horizon (frames) repeated at the main-campaign budget",
    )
    ap.add_argument("--c2-anchor-iters", type=int, default=8000)
    # C3
    # [9, 25] holds 17 frames, so 25 realises every one of them
    ap.add_argument("--c3-n", type=int, nargs="+", default=[2, 3, 5, 9, 13, 25])
    ap.add_argument(
        "--c3-delta-f",
        type=int,
        default=None,
        help="defaults to --pin-delta-f; also pins the joint-sparsity grid",
    )
    ap.add_argument("--c3-delta-l", type=int, default=25)
    # noise / sparsity
    ap.add_argument(
        "--noise", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.4]
    )
    ap.add_argument("--js-frac", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1])
    ap.add_argument("--js-n", type=int, nargs="+", default=[2, 3, 5, 9, 17])
    ap.add_argument(
        "--s1-frames",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233],
    )
    a = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    a.out_dir.mkdir(parents=True, exist_ok=True)
    b = Bench(a)
    lam = json.loads(Path("da_results_sda_paper/lyapunov.json").read_text())
    logger.info(
        f"lambda_1 = {lam['lambda_1_mean']:.5f}  T_L = {lam['lyapunov_time']:.2f} t.u."
        f" = {lam['lyapunov_time'] / DT:.0f} frames"
    )
    (a.out_dir / "lyapunov_ref.json").write_text(json.dumps(lam, indent=2))
    for s in a.sections:
        SECTIONS[s](b, a.out_dir, a)


if __name__ == "__main__":
    main()
