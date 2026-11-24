# mypy: disable-error-code="no-any-return"
"""Extremely long rollouts, and data assimilation from far-future observations.

Two separate questions, deliberately not mixed:

FC  free-running forecast skill.  Start every model from the EXACT true state and roll it
    forward for many Lyapunov times.  No assimilation, no optimisation -- this isolates
    *model* error, and says how far each propagator can be trusted at all.

LB  long-baseline assimilation.  Push the observations themselves far into the future and
    ask whether t_0 is still recoverable.  This is the extension of the headline panel
    beyond delta_l = 2.5 time units.

Both are extrapolation tests by construction: every model here was trained on rollouts of
10 stored frames = 1.0 time unit = 0.044 Lyapunov times, so a 400-time-unit rollout is
400x the training horizon.  Results must be read as extrapolation, not as in-distribution
performance.

Reference levels plotted with every curve:
  persistence   u_hat(t_0 + tau) = u(t_0)                     -- do nothing
  climatology   u_hat = 0, the field mean                     -- rel-L2 == 1 by definition
  saturation    error between two INDEPENDENT true states     -- the no-skill level
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from data_assimilation.ks.protocol import DT, build_problem, rel_l2
from data_assimilation.ks.da_ks_experiments_3way import (
    ALL,
    Bench,
    add_common_args,
    _save,
)

logger = logging.getLogger("longhorizon")


def lyap():
    d = json.loads(Path("da_results_sda_paper/lyapunov.json").read_text())
    return float(d["lyapunov_time"])


# ---------------------------------------------------------------------------
# FC: free-running forecast from a perfect initial condition
# ---------------------------------------------------------------------------
def exp_forecast(b: Bench, out: Path, a) -> None:
    logger.info("=== FC. FREE-RUNNING FORECAST from the EXACT true state ===")
    TL = lyap()
    dev, data = b.dev, b.data
    n_max = int(a.fc_frames)
    assert (
        data.n_t > n_max + a.t0_min + 1
    ), f"record has {data.n_t} frames; need > {n_max + a.t0_min + 1}"

    rng = np.random.default_rng(7)
    n_prob = min(a.fc_problems, data.n_sim)
    sim = np.arange(n_prob)
    t0 = np.full(n_prob, a.t0_min)
    sim_t = torch.as_tensor(sim, device=dev)
    t0_t = torch.as_tensor(t0, device=dev)

    # the frames we score at: dense early, logarithmic later
    # tau = 0 is deliberately excluded. UNet4DVar.prepare clamps n = max(n, 1) -- correct
    # for assimilation, where observations are strictly in the future -- so a tau of 0
    # would silently score the ONE-STEP prediction against frame 0 and reproduce
    # persistence at one frame rather than the identity. The KAE round-trip, which is the
    # meaningful tau = 0 reference for that method, is reported separately in Part B.
    frames = np.unique(
        np.concatenate(
            [np.arange(1, 51), np.round(np.geomspace(51, n_max, 160)).astype(int)]
        )
    ).astype(int)
    taus = frames * DT
    logger.info(
        f"  {n_prob} trajectories, {len(frames)} lead times, "
        f"max {n_max} frames = {n_max * DT:.1f} t.u. = {n_max * DT / TL:.1f} T_L"
    )

    truth = torch.stack([data.frames(sim_t, t0_t + int(f)) for f in frames])  # [F,B,X]
    truth_dn = data.denorm(truth)
    x0 = data.frames(sim_t, t0_t)

    store: Dict[str, np.ndarray] = {"frames": frames, "taus": taus, "T_L": np.array(TL)}

    # ---- reference levels ---------------------------------------------------
    pers = rel_l2(data.denorm(x0).unsqueeze(0).expand_as(truth_dn), truth_dn)
    store["persistence"] = pers.mean(1).cpu().numpy()
    store["climatology"] = (
        rel_l2(torch.zeros_like(truth_dn), truth_dn).mean(1).cpu().numpy()
    )
    # saturation: an independent true state drawn from a different trajectory and time
    oth_s = torch.as_tensor((sim + n_prob // 2) % data.n_sim, device=dev)
    oth_t = torch.as_tensor(rng.integers(a.t0_min, data.n_t - 1, n_prob), device=dev)
    oth = data.denorm(data.frames(oth_s, oth_t)).unsqueeze(0).expand_as(truth_dn)
    store["saturation"] = rel_l2(oth, truth_dn).mean(1).cpu().numpy()

    # ---- the models ---------------------------------------------------------
    from tensordict import TensorDict

    for name in ["KAE-expm", "KAE-rk4", "UNet"]:
        m = b.method(name)
        t_start = time.perf_counter()
        with torch.no_grad():
            if name.startswith("KAE"):
                c = b.kae.present_encoding(
                    TensorDict(
                        {"u": x0.unsqueeze(1).unsqueeze(-1)}, batch_size=[n_prob, 1]
                    ),
                    None,
                )
            else:
                c = x0
            prep = m.prepare(taus)
            pred = m.predict_at(c, taus, prep)  # [F,B,X]
            e = rel_l2(data.denorm(pred), truth_dn)  # [F,B]
        wall = time.perf_counter() - t_start
        store[f"{name}__rel_mean"] = e.mean(1).cpu().numpy()
        store[f"{name}__rel_sem"] = (e.std(1) / np.sqrt(n_prob)).cpu().numpy()
        store[f"{name}__wall_s"] = np.array(wall)
        # how far before the forecast is indistinguishable from no skill
        sat = store["saturation"]
        bad = np.where(e.mean(1).cpu().numpy() >= 0.9 * sat)[0]
        horizon = frames[bad[0]] * DT if len(bad) else np.inf
        store[f"{name}__skill_horizon_tu"] = np.array(horizon)
        logger.info(
            f"  {name:9s} rel-L2 @1 T_L {np.interp(TL, taus, e.mean(1).cpu()):.4f}"
            f" | @5 T_L {np.interp(5 * TL, taus, e.mean(1).cpu()):.4f}"
            f" | skill lost at {horizon:.1f} t.u. = {horizon / TL:.2f} T_L"
            f" | {wall:.1f}s"
        )

    # the KAE's encode/decode round trip is a genuine floor for KAE FORECASTING (unlike
    # KAE assimilation, where z0 is a free control and need not equal Encoder(x))
    from tensordict import TensorDict as _TD

    with torch.no_grad():
        z = b.kae.present_encoding(
            _TD({"u": x0.unsqueeze(1).unsqueeze(-1)}, batch_size=[n_prob, 1]), None
        )
        rt = rel_l2(data.denorm(b.kae.decode(z)["u"].squeeze(-1)), data.denorm(x0))
    store["kae_roundtrip"] = rt.mean().cpu().numpy()
    logger.info(
        f"  KAE encode/decode round trip on these states: {float(rt.mean()):.5f} "
        f"(a floor for KAE forecasting, not for KAE assimilation)"
    )
    store["n_problems"] = np.array(n_prob)
    store["train_rollout_tu"] = np.array(1.0)
    _save(out, "FC_forecast", **store)
    logger.info(f"saved -> {out / 'FC_forecast'}.npz")


# ---------------------------------------------------------------------------
# LB: assimilation from far-future observations
# ---------------------------------------------------------------------------
def _cap(m: str, a) -> int:
    """Affordable rollout length per method, in frames.

    The asymmetry is the point.  KAE-expm evaluates e^{K tau} as ONE matrix product, so its
    cost does not grow with the horizon at all.  Every other method here advances the state
    one stored frame at a time -- the U-Net and KAE-RK4 inside the 4D-Var loop, SDA through
    the length of the trajectory it scores -- so their cost is linear in the horizon and
    they run out of budget first.  Caps are affordability limits, not capability claims,
    and are reported with every table and figure.
    """
    return {"UNet": a.lb_cap_unet, "SDA": a.lb_cap_sda, "KAE-rk4": a.lb_cap_rk4}.get(
        m, a.lb_cap_kae
    )


def _sched(df: int, dl: int, n: int) -> np.ndarray:
    g = np.geomspace(df, dl, n)
    f = np.unique(np.round(g).astype(int))
    return np.unique(np.concatenate([[df], f, [dl]]))


def exp_long_baseline(b: Bench, out: Path, a) -> None:
    logger.info("=== LB. ASSIMILATION FROM FAR-FUTURE OBSERVATIONS ===")
    TL = lyap()
    rows, per = [], {}
    for dl in a.lb_delta_l:
        fr = _sched(a.lb_delta_f, int(dl), a.lb_n)
        prob = build_problem(
            b.data,
            name=f"LB{dl}",
            n_problems=a.lb_problems,
            taus=fr * DT,
            seed=61,
            t0_min_frame=a.t0_min,
        )
        row = {
            "delta_l_frames": int(dl),
            "delta_l_tu": float(dl * DT),
            "delta_l_TL": float(dl * DT / TL),
            "frames": list(map(int, fr)),
            "N": len(fr),
            "iters": a.lb_iters,
        }
        logger.info(
            f"  delta_l = {dl * DT:7.1f} t.u. = {dl * DT / TL:5.2f} T_L | "
            f"frames {list(map(int, fr))}"
        )
        for m in ALL:
            # a hard wall-clock guard: U-Net 4D-Var cost is linear in the rollout length,
            # and SDA's Algorithm-2 composition is linear in the trajectory length. Both
            # become unrunnable long before the KAE does. That limit is a RESULT, so it is
            # recorded rather than quietly skipped.
            cap = _cap(m, a)
            if dl > cap:
                row[m] = {
                    "skipped": True,
                    "reason": f"rollout {dl} frames exceeds the "
                    f"affordable cap of {cap} frames for this method",
                }
                logger.info(f"        {m:9s} SKIPPED (> {cap} frames)")
                continue
            r = b.run(m, prob, iters=a.lb_iters, seed=61)
            if r.get("skipped"):
                row[m] = {"skipped": True, "reason": "method reported skip"}
                continue
            v = np.asarray(r["rel"], dtype=float)
            row[m] = {
                "mean": float(v.mean()),
                "sem": float(v.std(ddof=1) / np.sqrt(len(v))),
                "n": int(len(v)),
                "wall_s": float(r["wall_s"]),
            }
            per[f"LB{dl}__{m}"] = v
            logger.info(
                f"        {m:9s} rel-L2 {v.mean():.4f} ± "
                f"{v.std(ddof=1) / np.sqrt(len(v)):.4f}  [{r['wall_s']:.0f}s]"
            )
        rows.append(row)
        # flush after every horizon: the long points take hours each
        (out / "LB_long_baseline.json").write_text(
            json.dumps(
                {
                    "meta": {
                        "partial": True,
                        "T_L": TL,
                        "fixed": {
                            "delta_f_frames": a.lb_delta_f,
                            "N_requested": a.lb_n,
                        },
                        "iters": a.lb_iters,
                        "n_problems": a.lb_problems,
                        "caps": {m: _cap(m, a) for m in ALL},
                        "note": "partial write",
                    },
                    "rows": rows,
                },
                indent=2,
            )
        )

    meta = {
        "varies": "delta_l, far beyond the canonical 2.5 time units",
        "fixed": {"delta_f_frames": a.lb_delta_f, "N_requested": a.lb_n},
        "iters": a.lb_iters,
        "n_problems": a.lb_problems,
        "T_L": TL,
        "data": str(a.test),
        "caps": {m: _cap(m, a) for m in ALL},
        "train_rollout_frames": 10,
        "note": (
            "Every model was trained on 10-frame rollouts, so all of these "
            "horizons are extrapolation. Methods are capped at different "
            "horizons purely by affordability; the cap is reported, never "
            "silently applied."
        ),
    }
    (out / "LB_long_baseline.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2)
    )
    _save(
        out,
        "LB_long_baseline",
        _rows_json=np.array(json.dumps({"meta": meta, "rows": rows})),
        **per,
    )
    logger.info(f"saved -> {out / 'LB_long_baseline'}.json")


# ---------------------------------------------------------------------------
# HL: the headline panel, with the observations pushed far into the future
# ---------------------------------------------------------------------------
def exp_headline_long(b: Bench, out: Path, a) -> None:
    """Same figure as the headline panel, but the observations are no longer bunched
    inside 2.5 time units -- they are spread across several Lyapunov times.

    Emits the same keys as ``exp_A_headline`` so the notebook renders it with identical
    plotting code and the two are directly comparable panel for panel.
    """
    from data_assimilation.ks.da_ks_experiments_3way import ae_reconstruction_floor

    TL = lyap()
    for tag, offs, iters in a.hl_sets:
        offs = np.asarray(offs, dtype=int)
        logger.info(f"=== HL[{tag}]. HEADLINE with far-spaced observations ===")
        logger.info(
            f"  tau = {list(offs * DT)} t.u. = "
            f"{[round(float(o) * DT / TL, 2) for o in offs]} T_L"
        )
        prob = build_problem(
            b.data,
            name=f"HL{tag}",
            n_problems=1,
            taus=offs * DT,
            seed=100,
            t0_min_frame=a.t0_min,
        )
        floor = float(ae_reconstruction_floor(b.kae, b.data, prob).mean())
        sim = torch.as_tensor(prob.sim, device=b.dev)
        t0 = torch.as_tensor(prob.t0, device=b.dev)
        store = {
            "x": b.data.x,
            "offsets": offs,
            "dt": DT,
            "ae_floor": floor,
            "T_L": TL,
            "u_t0_true": b.data.denorm(b.data.frames(sim, t0)).cpu().numpy()[0],
            "obs_true": np.stack(
                [
                    b.data.denorm(b.data.frames(sim, t0 + int(o))).cpu().numpy()[0]
                    for o in offs
                ]
            ),
        }
        for m in ALL:
            cap = _cap(m, a)
            if int(offs.max()) > cap:
                logger.info(f"   {m:9s} SKIPPED (rollout {offs.max()} > cap {cap})")
                store[f"{m}__skipped_reason"] = np.array(
                    f"rollout {int(offs.max())} frames exceeds the affordable cap "
                    f"of {cap} frames for this method"
                )
                continue
            r = b.run(m, prob, iters=iters, seed=100, track_every=25)
            if r.get("skipped"):
                continue
            store[f"{m}__u_t0_recon"] = r["analysis"][0]
            store[f"{m}__obs_pred"] = r["obs_pred"][:, 0]
            if "spread_field" in r:
                store[f"{m}__spread_t0"] = r["spread_field"][0, 0]
                store[f"{m}__draws_t0"] = r["draws"][:, 0, 0]
            store[f"{m}__rel_final"] = float(r["rel"][0])
            if "rel_per_draw" in r:
                store[f"{m}__rel_per_draw"] = r["rel_per_draw"][:, 0]
            for k, v in r["hist"].items():
                store[f"{m}__hist_{k}"] = np.asarray(v)
            logger.info(f"   {m:9s} field rel-L2 = {r['rel'][0]:.4f}")
        _save(out, f"HL_headline_{tag}", **store)
        logger.info(f"saved -> {out / ('HL_headline_' + tag)}.npz")


# ---------------------------------------------------------------------------
# PF: forecast skill AFTER assimilation -- the operationally meaningful version
# ---------------------------------------------------------------------------
def exp_post_da_forecast(b: Bench, out: Path, a) -> None:
    """Assimilate at t_0 from a short observation window, then forecast a long way.

    FC starts every method from the EXACT true state and so measures the propagator alone.
    This section starts from each method's OWN analysis, which is what a real system has.
    The two curves bracket the honest answer, and they are expected to disagree in an
    interesting way: the KAE earns a far better analysis but propagates it badly, while the
    U-Net starts from a much worse analysis and propagates it well.  Whether, and where,
    those cross is a measurement, not a prediction.
    """
    logger.info("=== PF. FORECAST AFTER ASSIMILATION ===")
    TL = lyap()
    data = b.data
    obs = np.asarray(a.pf_obs, dtype=int)
    n_max = int(a.pf_frames)
    prob = build_problem(
        data,
        name="PF",
        n_problems=a.pf_problems,
        taus=obs * DT,
        seed=71,
        t0_min_frame=a.t0_min,
        forecast_taus=np.array([n_max * DT]),
    )
    sim = torch.as_tensor(prob.sim, device=b.dev)
    t0 = torch.as_tensor(prob.t0, device=b.dev)

    frames = np.unique(
        np.concatenate(
            [np.arange(1, 51), np.round(np.geomspace(51, n_max, 150)).astype(int)]
        )
    ).astype(int)
    taus = frames * DT
    truth_dn = data.denorm(torch.stack([data.frames(sim, t0 + int(f)) for f in frames]))
    x0_true = data.frames(sim, t0)

    store: Dict[str, np.ndarray] = {
        "frames": frames,
        "taus": taus,
        "T_L": np.array(TL),
        "obs_frames": obs,
        "n_problems": np.array(a.pf_problems),
        "iters": np.array(a.pf_iters),
    }
    store["persistence"] = (
        rel_l2(data.denorm(x0_true).unsqueeze(0).expand_as(truth_dn), truth_dn)
        .mean(1)
        .cpu()
        .numpy()
    )
    rng = np.random.default_rng(11)
    oth = (
        data.denorm(
            data.frames(
                torch.as_tensor(
                    (prob.sim + a.pf_problems // 2) % data.n_sim, device=b.dev
                ),
                torch.as_tensor(
                    rng.integers(a.t0_min, data.n_t - 1, a.pf_problems), device=b.dev
                ),
            )
        )
        .unsqueeze(0)
        .expand_as(truth_dn)
    )
    store["saturation"] = rel_l2(oth, truth_dn).mean(1).cpu().numpy()

    for name in ["KAE-expm", "KAE-rk4", "UNet"]:
        cap = _cap(name, a)
        m = b.method(name)
        t_start = time.perf_counter()
        res = m.solve(prob, data, b.cfg(name, a.pf_iters, 71))
        c = res["control"]
        with torch.no_grad():
            ana = rel_l2(data.denorm(m.analysis_field(c)), data.denorm(x0_true))
            keep = frames[frames <= cap]
            pred = m.predict_at(c, keep * DT, m.prepare(keep * DT))
            e = rel_l2(data.denorm(pred), truth_dn[: len(keep)])
        wall = time.perf_counter() - t_start
        mu = np.full(len(frames), np.nan)
        mu[: len(keep)] = e.mean(1).cpu().numpy()
        se = np.full(len(frames), np.nan)
        se[: len(keep)] = (e.std(1) / np.sqrt(a.pf_problems)).cpu().numpy()
        store[f"{name}__rel_mean"], store[f"{name}__rel_sem"] = mu, se
        store[f"{name}__analysis_rel"] = ana.mean().cpu().numpy()
        store[f"{name}__cap_frames"] = np.array(cap)
        store[f"{name}__wall_s"] = np.array(wall)
        ok = np.isfinite(mu)
        bad = np.where(ok & (mu >= 0.9 * store["saturation"]))[0]
        h = frames[bad[0]] * DT if len(bad) else np.inf
        store[f"{name}__skill_horizon_tu"] = np.array(h)
        logger.info(
            f"  {name:9s} analysis {float(ana.mean()):.4f} | "
            f"@1 T_L {np.interp(TL, taus[ok], mu[ok]):.4f} | "
            f"@5 T_L {np.interp(5 * TL, taus[ok], mu[ok]):.4f} | "
            f"skill lost at {h:.1f} t.u. = {h / TL:.2f} T_L | {wall:.0f}s"
        )
    _save(out, "PF_post_da_forecast", **store)
    logger.info(f"saved -> {out / 'PF_post_da_forecast'}.npz")


SECTIONS = {
    "FC": exp_forecast,
    "LB": exp_long_baseline,
    "HL": exp_headline_long,
    "PF": exp_post_da_forecast,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=Path("da_results_long"))
    add_common_args(ap)
    ap.add_argument("--sections", nargs="+", default=["FC", "HL", "LB"])
    ap.add_argument("--t0-min", type=int, default=100)
    # FC
    ap.add_argument("--fc-frames", type=int, default=4800)
    ap.add_argument("--fc-problems", type=int, default=32)
    # LB
    ap.add_argument(
        "--lb-delta-l",
        type=int,
        nargs="+",
        default=[25, 100, 250, 500, 1000, 2000, 4000],
    )
    ap.add_argument("--lb-delta-f", type=int, default=1)
    ap.add_argument("--lb-n", type=int, default=5)
    ap.add_argument("--lb-iters", type=int, default=1000)
    ap.add_argument("--lb-problems", type=int, default=24)
    ap.add_argument(
        "--lb-cap-kae",
        type=int,
        default=10**9,
        help="KAE-expm: e^{K tau} is one matrix product, cost is horizon-free",
    )
    # PF
    ap.add_argument("--pf-obs", type=int, nargs="+", default=[1, 3, 7, 15, 25])
    ap.add_argument("--pf-frames", type=int, default=4800)
    ap.add_argument("--pf-problems", type=int, default=24)
    ap.add_argument("--pf-iters", type=int, default=2000)
    ap.add_argument(
        "--lb-cap-rk4",
        type=int,
        default=1000,
        help="KAE-RK4 integrates step by step, so it is capped like the U-Net",
    )
    ap.add_argument("--lb-cap-unet", type=int, default=250)
    ap.add_argument("--lb-cap-sda", type=int, default=1000)
    # HL: (tag, observation offsets in frames, iterations). The first reproduces the
    # canonical schedule on this data for reference; the rest push tau_N far out.
    ap.set_defaults(
        hl_sets=[
            ("canonical", [1, 3, 7, 15, 25], 2000),  # delta_l = 2.5 t.u.  = 0.11 T_L
            ("medium", [1, 5, 25, 100, 250], 2000),  # delta_l = 25 t.u.   = 1.11 T_L
            ("long", [1, 10, 60, 300, 1000], 2000),  # delta_l = 100 t.u.  = 4.43 T_L
            (
                "extreme",
                [1, 25, 200, 1200, 4000],
                2000,
            ),  # delta_l = 400 t.u.  = 17.7 T_L
        ]
    )
    a = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    a.out_dir.mkdir(parents=True, exist_ok=True)
    b = Bench(a)
    logger.info(
        f"data {a.test}: {b.data.n_sim} trajectories x {b.data.n_t} frames "
        f"= {b.data.n_t * DT:.1f} t.u. = {b.data.n_t * DT / lyap():.1f} T_L"
    )
    for s in a.sections:
        SECTIONS[s](b, a.out_dir, a)


if __name__ == "__main__":
    main()
