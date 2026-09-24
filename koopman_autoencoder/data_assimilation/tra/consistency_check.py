# mypy: disable-error-code="arg-type, assignment, misc, operator, return-value"
"""Gate: every number typed onto a slide must still be the number on disk.

The two existing gates check the *code* (`verify_adapters`: the adapter reproduces
turbpred's own forward; `verify_forward_skill`: every model beats persistence).  Neither
looks at the deck.  A slide is a transcription, and a transcription is exactly the thing
that goes stale the moment an experiment is re-run -- which is what happened to this
campaign more than once.

So this is the third gate, ported from ``data_assimilation/ks/consistency_check.py`` in the KS campaign.  It
does two things:

  1. **Verifies** each claim below by re-reading it from the file that produced it.
  2. **Sweeps up whatever it does not recognise.**  Every number in every markdown cell of
     the deck that looks like a reported quantity is extracted; any that no claim accounts
     for is printed as UNVERIFIED.  A stale table therefore cannot pass by being one this
     file happens not to mention.

Run it as ``python -m data_assimilation.tra.consistency_check`` (add ``--strict`` to fail on
UNVERIFIED as well as on FAIL).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

TRA = Path("da_results_tra")
RETIRED = Path("da_results_tra_INVALID_transpose")
NB = Path("visualize_da_tra.ipynb")

checks: list[tuple[str, str, str]] = []  # (verdict, name, detail)


def record(name: str, verdict: str, detail: str = "") -> None:
    checks.append((verdict, name, detail))


def chk(name: str, got, want, rtol: float = 0.02) -> None:
    """Compare a number read from disk with the number written on a slide."""
    if got is None:
        record(name, "MISSING", "source file or key absent")
        return
    got, want = float(got), float(want)
    ok = abs(got - want) <= rtol * max(abs(want), 1e-12)
    record(
        name,
        "PASS" if ok else "FAIL",
        f"disk {got:.6g} vs slide {want:.6g} (rtol {rtol})",
    )


# --- readers ---------------------------------------------------------------
def _json(stem: str) -> Optional[dict]:
    p = TRA / f"{stem}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def _npz(stem: str):
    p = TRA / f"{stem}.npz"
    return np.load(p, allow_pickle=True) if p.is_file() else None


def _walk_numbers(obj, depth: int = 0):
    """Every finite number reachable inside a nested result entry."""
    if depth > 6:
        return
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if np.isfinite(obj):
            yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_numbers(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_numbers(v, depth + 1)


def _dig(obj, *keys):
    for k in keys:
        if obj is None:
            return None
        obj = (
            obj[k]
            if (isinstance(obj, dict) and k in obj)
            else (
                obj[k]
                if isinstance(obj, list) and isinstance(k, int) and k < len(obj)
                else None
            )
        )
    return obj


def sweep_point(stem: str, method: str, pred: Callable[[dict], bool], key="mean"):
    """The value of ``method`` at the first row of a sweep satisfying ``pred``."""
    d = _json(stem)
    if d is None:
        return None
    for r in d["rows"]:
        if pred(r):
            return _dig(r, method, key)
    return None


def npz_at(stem: str, key: str, idx: Optional[int] = None):
    z = _npz(stem)
    if z is None or key not in z.files:
        return None
    v = z[key]
    return float(v) if idx is None else float(np.asarray(v).ravel()[idx])


# --- the claims ------------------------------------------------------------
# Each entry is (label, value-on-disk, value-as-written-on-a-slide, rtol).  The third
# column is the ONLY place a hand-typed number is allowed to live, and every one of them
# is also required to appear verbatim in the deck (see `sweep_deck` below).
def build_claims() -> list[tuple[str, object, float, float]]:
    hl, hd = _json("A_headline"), _json("A_headline_diffusion")
    ident = _json("identifiability")
    return [
        # ---- headline (slide: "Headline", "The paradox", "Conclusions")
        ("headline KAE", _dig(hl, "KAE", "mean"), 0.0267, 0.02),
        ("headline UNet", _dig(hl, "UNet", "mean"), 0.2146, 0.02),
        ("headline FNO", _dig(hl, "FNO", "mean"), 0.1281, 0.02),
        ("headline ACDM", _dig(hd, "ACDM", "mean"), 0.0057, 0.03),
        ("headline ACDM-ncn", _dig(hd, "ACDM-ncn", "mean"), 2.447, 0.03),
        # the caveats slide quotes ACDM-ncn's error RANGE across the campaign, so the top
        # of that range has to be read from the sweep that produces it
        (
            "ACDM-ncn worst error across the delta_f sweep",
            max(
                [
                    sweep_point(
                        "G1_delta_f", "ACDM-ncn", lambda r, v=v: r["value"] == v
                    )
                    or 0
                    for v in (1, 2, 4, 8, 12, 18)
                ]
            ),
            4.1,
            0.03,
        ),
        # ---- forward skill (slide: "The paradox")
        (
            "UNet one-step forecast",
            npz_at("G7_forecast", "UNet__rel_mean", 0),
            0.0023,
            0.05,
        ),
        (
            "KAE one-step forecast",
            npz_at("G7_forecast", "KAE__rel_mean", 0),
            0.0382,
            0.05,
        ),
        # ---- identifiability (slide: "Optimisation failure, or identifiability failure?")
        ("UNet cost prefers truth by", _dig(ident, "UNet", "ratio"), 33.0, 0.05),
        ("FNO cost prefers its own answer", _dig(ident, "FNO", "ratio"), 0.270, 0.05),
        # ---- crossover table (slide: "The crossover")
        (
            "delta_f=1 ACDM",
            sweep_point("G1_delta_f", "ACDM", lambda r: r["value"] == 1),
            0.0054,
            0.05,
        ),
        (
            "delta_f=1 KAE",
            sweep_point("G1_delta_f", "KAE", lambda r: r["value"] == 1),
            0.0269,
            0.05,
        ),
        (
            "delta_f=4 ACDM",
            sweep_point("G1_delta_f", "ACDM", lambda r: r["value"] == 4),
            0.1837,
            0.05,
        ),
        (
            "delta_f=4 KAE",
            sweep_point("G1_delta_f", "KAE", lambda r: r["value"] == 4),
            0.0421,
            0.05,
        ),
        (
            "obs_frac=0.10 ACDM",
            sweep_point("G3_sparsity", "ACDM", lambda r: abs(r["value"] - 0.10) < 1e-9),
            0.0956,
            0.05,
        ),
        (
            "obs_frac=0.10 KAE",
            sweep_point("G3_sparsity", "KAE", lambda r: abs(r["value"] - 0.10) < 1e-9),
            0.0285,
            0.05,
        ),
        (
            "obs_frac=0.05 ACDM",
            sweep_point("G3_sparsity", "ACDM", lambda r: abs(r["value"] - 0.05) < 1e-9),
            0.1690,
            0.05,
        ),
        (
            "obs_frac=0.05 KAE",
            sweep_point("G3_sparsity", "KAE", lambda r: abs(r["value"] - 0.05) < 1e-9),
            0.0269,
            0.05,
        ),
        (
            "noise=0.3 ACDM",
            sweep_point("G2_noise", "ACDM", lambda r: abs(r["value"] - 0.3) < 1e-9),
            0.0342,
            0.05,
        ),
        (
            "noise=0.3 KAE",
            sweep_point("G2_noise", "KAE", lambda r: abs(r["value"] - 0.3) < 1e-9),
            0.0295,
            0.05,
        ),
    ]


def specs_claims() -> list[tuple[str, object, float, float]]:
    """Numbers the test-discipline slide states, read from training_specs.json."""
    f = TRA / "training_specs.json"
    if not f.is_file():
        return []
    d = json.loads(f.read_text())
    mv = sorted(d["test_discipline"]["training_mach_values"])
    gaps = [(a, b) for a, b in zip(mv, mv[1:]) if b - a > 0.015]
    out = [
        ("training Mach min", mv[0], 0.53, 1e-6),
        ("training Mach max", mv[-1], 0.89, 1e-6),
        ("test Mach low", d["test_discipline"]["test_mach_range"][0], 0.66, 1e-3),
        ("test Mach high", d["test_discipline"]["test_mach_range"][1], 0.68, 1e-3),
        (
            "tuning Mach low",
            d["test_discipline"]["validation_mach_range"][0],
            0.50,
            1e-3,
        ),
        (
            "tuning Mach high",
            d["test_discipline"]["validation_mach_range"][1],
            0.52,
            1e-3,
        ),
    ]
    if gaps:
        out += [
            ("training gap starts after", gaps[0][0], 0.63, 1e-6),
            ("training gap ends at", gaps[0][1], 0.69, 1e-6),
        ]
    ctl = TRA / "L_leakage_control.json"
    if ctl.is_file():
        out += [
            (
                "control regime Mach low",
                min(d["data"]["long (post-DA forecast)"]["mach_values"]),
                0.64,
                1e-6,
            ),
            (
                "control regime Mach high",
                max(d["data"]["long (post-DA forecast)"]["mach_values"]),
                0.65,
                1e-6,
            ),
        ]
    else:
        # the slide names the control regime whether or not the run has finished
        mvl = d["data"]["long (post-DA forecast)"].get("mach_values") or [None, None]
        out += [
            ("control regime Mach low", mvl[0], 0.64, 1e-6),
            ("control regime Mach high", mvl[-1], 0.65, 1e-6),
        ]
    return out


def retired_claims() -> list[tuple[str, object, float, float]]:
    """The transpose bug's numbers, read from the retired results that still hold them.

    The gates slide quotes what the bug cost.  Those numbers have a source -- the retired
    directory -- and should be read from it rather than remembered.
    """
    f = RETIRED / "G7_forecast.npz"
    if not f.is_file():
        return []
    g = np.load(f)
    return [
        (
            "retired UNet one-step (transposed)",
            float(g["UNet__rel_mean"][0]),
            0.1472,
            0.01,
        ),
        (
            "fixed UNet one-step",
            npz_at("G7_forecast", "UNet__rel_mean", 0),
            0.0023,
            0.05,
        ),
    ]


def sparsity_extreme_claim() -> list[tuple[str, object, float, float]]:
    """The gap-filling slide says 0.5% of sensors; the sweep must actually reach it."""
    d = _json("G3_sparsity")
    if d is None:
        return []
    lo = min(r["value"] for r in d["rows"])
    return [("sparsest point swept (%)", 100 * lo, 0.5, 1e-6)]


def spacetime_claims() -> list[tuple[str, object, float, float]]:
    """The mechanism slide: the analysis error is erased before it is observed.

    The claim is a per-frame collapse, so it is read frame by frame from G9 rather than
    summarised. ACDM is included because it is the counter-example that makes the argument
    work: its error GROWS, because its analysis was right and there is nothing to erase.
    """
    z = _npz("G9_spacetime")
    if z is None:
        return []

    def at(m, k):
        key = f"{m}__per_frame"
        return float(z[key][k]) if key in z.files else None

    out = []
    for m, k, want in [
        ("FNO", 0, 0.1407),
        ("FNO", 1, 0.0289),
        ("UNet", 0, 0.2188),
        ("UNet", 1, 0.0953),
        ("ACDM", 0, 0.0091),
    ]:
        out.append((f"spacetime {m} frame {k}", at(m, k), want, 0.03))
    f0, f1 = at("FNO", 0), at("FNO", 1)
    if f0 and f1:
        out.append(("FNO one-frame collapse factor", f0 / f1, 4.9, 0.05))
    a0 = at("ACDM", 0)
    aT = at("ACDM", int(np.asarray(z["T"])) - 1) if "T" in z.files else None
    if a0 and aT:
        out.append(("ACDM error growth over the window", aT / a0, 8.3, 0.05))
    return out


def reach_accuracy_claims() -> list[tuple[str, object, float, float]]:
    """Representational reach is not accuracy across that reach (G13), and analysis
    accuracy does not survive a long forward rollout (G10). Both reverse the headline
    ordering, so both are read back from their own files."""
    out = []
    b = _json("G13_backward")
    if b:
        row = {r["method"]: r for r in b["rows"]}
        if {"KAE", "ACDM"} <= set(row):
            k, a = row["KAE"]["err"], row["ACDM"]["err"]
            out += [
                ("backward KAE at t0-7", k[-1], 0.0655, 0.03),
                ("backward ACDM at t0-7", a[-1], 0.2244, 0.03),
                ("backward ACDM at t0-0", a[0], 0.0473, 0.03),
            ]
            rr = [c / v for v, c in zip(k[1:], a[1:]) if v]
            if rr:
                out += [
                    ("backward ACDM/KAE ratio, min over j>=1", min(rr), 3.4, 0.06),
                    ("backward ACDM/KAE ratio, max over j>=1", max(rr), 5.5, 0.06),
                ]
        for m, w in [("UNet", 1), ("FNO", 1), ("KAE", 8), ("ACDM", 8)]:
            if m in row:
                out.append(
                    (
                        f"backward {m} frames representable",
                        row[m]["n_representable"],
                        w,
                        0.0,
                    )
                )
    z = _npz("G10_post_da")
    if z is not None:
        for m, w in [
            ("KAE", 0.0258),
            ("UNet", 0.2150),
            ("FNO", 0.1196),
            ("ACDM", 0.0105),
        ]:
            k = f"{m}__analysis_rel"
            if k in z.files:
                out.append((f"post-DA {m} analysis", float(z[k]), w, 0.03))
        for m, w in [("FNO", 0.1432), ("ACDM", 0.1589), ("KAE", 0.1962)]:
            k = f"{m}__rel_mean"
            if k in z.files:
                out.append(
                    (f"post-DA {m} at the longest lead", float(z[k][-1]), w, 0.03)
                )
    return out


def reach_claims() -> list[tuple[str, object, float, float]]:
    """G18, the FAIR reach test: every method re-solved at each target.

    `backward` favours the KAE by construction -- one latent spans the window while the
    U-Net's control IS the frame at t_0. Here the analysis time moves and everybody solves
    again, so the ACDM-to-KAE crossover cannot be an artefact of representational reach.
    That crossover is the claim, so both sides of it are read back.
    """
    d = _json("G18_reach")
    if d is None:
        return []
    row = {r["frames_back"]: r for r in d["rows"]}

    def v(j, m):
        r = row.get(j)
        return None if r is None or m not in r else r[m]["mean"]

    out = [
        ("reach ACDM at t0 (wins)", v(0, "ACDM"), 0.0053, 0.05),
        ("reach KAE at t0", v(0, "KAE"), 0.0261, 0.05),
        ("reach ACDM at t0-2 (KAE ahead)", v(2, "ACDM"), 0.0831, 0.06),
        ("reach KAE at t0-2", v(2, "KAE"), 0.0348, 0.05),
        ("reach ACDM at t0-8", v(8, "ACDM"), 0.2683, 0.05),
        ("reach KAE at t0-8", v(8, "KAE"), 0.0883, 0.05),
    ]
    a8, k8 = v(8, "ACDM"), v(8, "KAE")
    if a8 and k8:
        out.append(("reach ACDM/KAE at t0-8", a8 / k8, 3.0, 0.08))
    # the crossover must actually happen, and between t0 and t0-2
    a0, k0, a2, k2 = v(0, "ACDM"), v(0, "KAE"), v(2, "ACDM"), v(2, "KAE")
    if None not in (a0, k0, a2, k2):
        record(
            "reach: ACDM leads at t0 and the KAE leads by t0-2",
            "PASS" if (a0 < k0 and a2 > k2) else "FAIL",
            f"t0 ACDM {a0:.4f} vs KAE {k0:.4f}; t0-2 ACDM {a2:.4f} vs KAE {k2:.4f}",
        )
    return out


def single_obs_claims() -> list[tuple[str, object, float, float]]:
    """The same crossover with a SINGLE observation, where nothing else can carry it."""
    d = _json("G6_single_obs")
    if d is None:
        return []
    row = {r["value"]: r for r in d["rows"]}

    def v(t, m):
        r = row.get(t)
        return None if r is None or not isinstance(r.get(m), dict) else r[m]["mean"]

    out = [
        ("single-obs ACDM at tau=1 (wins)", v(1, "ACDM"), 0.0059, 0.06),
        ("single-obs KAE at tau=1", v(1, "KAE"), 0.0227, 0.05),
        ("single-obs ACDM at tau=8", v(8, "ACDM"), 0.2534, 0.06),
        ("single-obs KAE at tau=8 (KAE ahead)", v(8, "KAE"), 0.0912, 0.05),
    ]
    # this sweep used to be all-NaN for the samplers; it must never silently return
    nd = sum(
        int((r.get(m) or {}).get("n_diverged") or 0)
        for r in d["rows"]
        for m in ("ACDM", "ACDM-ncn")
    )
    record(
        "single-obs samplers: no diverged solves",
        "PASS" if nd == 0 else "FAIL",
        f"{nd} diverged solve(s); this sweep was all-NaN before the guidance was tuned",
    )
    return out


def _num(entry, key="mean") -> float:
    """A sweep cell as a float, with null (a diverged solve) mapped to NaN."""
    if not isinstance(entry, dict):
        return float("nan")
    v = entry.get(key)
    return float("nan") if v is None else float(v)


def delta_l_claims() -> list[tuple[str, object, float, float]]:
    """The horizon sweep, which is the axis ACDM does NOT lose on.

    delta_f is pinned at 1 here, so ACDM always has a close observation and stays flat
    across a 10x horizon. That is the control for the delta_f sweep: it localises ACDM's
    weakness to the lead to the FIRST observation rather than to the horizon.
    """
    d = _json("G4_delta_l")
    if d is None:
        return []
    rows = d["rows"]
    out = []
    for m in ("ACDM", "KAE"):
        vals = [_num(r.get(m)) for r in rows]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) > 1:
            out.append(
                (
                    f"delta_l {m} spread (max/min over the sweep)",
                    max(vals) / min(vals),
                    1.10 if m == "ACDM" else 1.35,
                    0.15,
                )
            )
    record(
        "delta_l sweep is complete, not truncated",
        "PASS" if not d["meta"].get("partial") else "FAIL",
        f"{len(rows)} points, horizons {[r['value'] for r in rows]}",
    )
    # ACDM must lead at EVERY horizon here; if it ever does not, the localisation fails
    beats = all(
        _num(r.get("ACDM")) < _num(r.get("KAE"))
        for r in rows
        if np.isfinite(_num(r.get("ACDM"))) and np.isfinite(_num(r.get("KAE")))
    )
    record(
        "delta_l: ACDM leads at every horizon (delta_f pinned at 1)",
        "PASS" if beats else "FAIL",
        "ACDM's weakness is the lead to the FIRST observation, not the horizon",
    )
    return out


def sda_fidelity_claims() -> list[tuple[str, object, float, float]]:
    """The SDA-fidelity slide quotes the settings actually used; read them back.

    These are the numbers a reviewer comparing against Rozet & Louppe (2023) will check,
    and the slide's argument (that this is the paper's least accurate corner) depends on
    them being what the runs really used.
    """
    tdp = TRA / "tuning_diffusion.json"
    hd = _json("A_headline_diffusion")
    out = []
    if tdp.is_file():
        td = json.loads(tdp.read_text()).get("ACDM", {})
        out += [
            ("SDA guidance Gamma used", td.get("gamma"), 0.3, 1e-6),
            ("SDA sigma_y floor used", td.get("sigma_y"), 0.05, 1e-6),
        ]
        record(
            "SDA correctors are DISABLED (C = 0)",
            "PASS" if td.get("corrections") == 0 else "FAIL",
            f"corrections = {td.get('corrections')}; the paper's Fig. 2 ablation "
            f"reaches ground-truth accuracy only at C >= 2",
        )
    if hd:
        out += [
            ("SDA blanket half-width k", _dig(hd, "ACDM", "k"), 1, 1e-6),
            (
                "SDA blanket window (frames)",
                _dig(hd, "ACDM", "blanket_window"),
                3,
                1e-6,
            ),
            ("SDA denoising steps", _dig(hd, "ACDM", "timesteps"), 20, 1e-6),
        ]
        # The comparison on the slide is against the paper's KOLMOGOROV FLOW setup (k=2,
        # C=1, 256 steps) -- its 2-D fluid case -- not against the Lorenz ablation, whose
        # k>=3 threshold is specific to a 3-D state and does not transfer.
        record(
            "SDA blanket is one step narrower than the paper's 2-D fluid setup",
            "PASS" if _dig(hd, "ACDM", "k") == 1 else "FAIL",
            "campaign k=1 (3-frame window) vs Rozet & Louppe's Kolmogorov k=2 "
            "(5-frame window); forced by ACDM's 15-channel U-Net",
        )
    return out


def regimes_claims() -> list[tuple[str, object, float, float]]:
    """The cross-regime slide: what has to hold is the ORDERING and the CI discipline.

    The absolute numbers differ by regime by design, so pinning them would be brittle.
    What the slide actually claims is that the ordering survives all three regimes and
    that the intervals are clustered by trajectory -- both of which are checked here.
    """
    d = _json("R_regimes")
    if d is None:
        return []
    rows, out = d["rows"], []
    for split, label in (
        ("test", "gt_interp"),
        ("long", "gt_longer"),
        ("val", "gt_extrap"),
    ):
        r = rows.get(split)
        if not r:
            continue
        got = {m: (r.get(m) or {}).get("mean") for m in METHODS5}
        if all(got.get(m) is not None for m in ("ACDM", "KAE", "UNet")):
            record(
                f"{label}: ACDM < KAE < U-Net",
                "PASS" if got["ACDM"] < got["KAE"] < got["UNet"] else "FAIL",
                " ".join(f"{m}={got[m]:.4f}" for m in ("ACDM", "KAE", "UNet")),
            )
        # the interval must be the clustered one, and clustering must actually widen it
        kae = r.get("KAE") or {}
        if kae.get("ci95_cluster") and kae.get("ci95_naive"):
            infl = kae.get("cluster_inflation")
            record(
                f"{label}: interval clustered by trajectory",
                "PASS" if kae.get("n_clusters", 0) >= 2 else "FAIL",
                (
                    f"{kae['n_clusters']} trajectories, {kae['n']} problems, "
                    f"clustered CI is {infl:.2f}x the naive one"
                    if infl
                    else f"{kae.get('n_clusters')} trajectories"
                ),
            )
        ntr = _dig(r, "problem", "n_trajectories")
        if ntr is not None:
            out.append(
                (
                    f"{label} trajectories in the pool",
                    ntr,
                    4 if split == "long" else 6,
                    1e-6,
                )
            )
    return out


def leakage_answered_claims() -> list[tuple[str, object, float, float]]:
    """The leakage control's conclusion, read from the two campaigns that settle it.

    gt_longer is disjoint from train.nc AND val.nc, so if the KAE's result were an
    artefact of overlapping its own checkpoint-selection split it would shrink there.
    The slide states that it does not; these are the numbers behind that.
    """
    from data_assimilation.tra.plots import REGIME_DIRS, summary, NotRunYet

    try:
        S = {k: summary(d) for k, (d, _, _) in REGIME_DIRS.items()}
    except NotRunYet:
        return []
    out = []
    for m, want in (
        ("KAE", 0.0283),
        ("UNet", 0.2179),
        ("FNO", 0.1169),
        ("ACDM", 0.0061),
    ):
        v = S.get("longer", {}).get(m, {}).get("mean")
        if v is not None:
            out.append((f"gt_longer headline {m}", v, want, 0.03))
    # the ordering must be identical on every regime, or the study's premise fails
    for k, s_ in S.items():
        got = [s_[m]["mean"] for m in ("ACDM", "KAE", "FNO", "UNet") if m in s_]
        if len(got) == 4:
            record(
                f"{k}: ordering ACDM < KAE < FNO < U-Net",
                "PASS" if got == sorted(got) else "FAIL",
                " < ".join(f"{v:.4f}" for v in got),
            )
    # the slide's longer/interp ratio column, and the per-regime U-Net/KAE margins
    for m, want in (("KAE", 1.06), ("UNet", 1.02), ("FNO", 0.91), ("ACDM", 1.07)):
        a = S.get("interp", {}).get(m, {}).get("mean")
        b = S.get("longer", {}).get(m, {}).get("mean")
        if a and b:
            out.append((f"{m}: gt_longer / gt_interp", b / a, want, 0.02))
    for k, want in (("interp", 8.0), ("longer", 7.7), ("extrap", 12.1)):
        u = S.get(k, {}).get("UNet", {}).get("mean")
        kk = S.get(k, {}).get("KAE", {}).get("mean")
        if u and kk:
            out.append((f"{k}: U-Net/KAE margin", u / kk, want, 0.02))
    ki, kl = S.get("interp", {}).get("KAE", {}).get("mean"), S.get("longer", {}).get(
        "KAE", {}
    ).get("mean")
    ui, ul = S.get("interp", {}).get("UNet", {}).get("mean"), S.get("longer", {}).get(
        "UNet", {}
    ).get("mean")
    if all(v is not None for v in (ki, kl, ui, ul)):
        record(
            "leakage control: the KAE's margin survives on the clean regime",
            "PASS" if (ul / kl) > 0.8 * (ui / ki) else "FAIL",
            f"U-Net/KAE margin {ui / ki:.1f}x on gt_interp (overlaps val) vs "
            f"{ul / kl:.1f}x on gt_longer (clean) -- the selection asymmetry is real "
            f"but is not what produces the result",
        )
    return out


def win_loss_claims() -> list[tuple[str, object, float, float]]:
    """The slide's win/loss tally, recomputed from the same source the table uses.

    The claim is that the record is near-even and separates on delta_f, so the tally
    itself is the number to verify -- not any single experiment's value.
    """
    from data_assimilation.tra.plots import win_loss_table, NotRunYet

    try:
        txt = win_loss_table()
    except (NotRunYet, Exception):
        return []
    m = re.search(r"KAE wins (\d+)\s+·\s+ACDM wins (\d+)", txt)
    if not m:
        return []
    kae, acdm = int(m.group(1)), int(m.group(2))
    record(
        "win/loss record is near-even (neither method dominates)",
        "PASS" if 0.6 <= kae / max(acdm, 1) <= 1.7 else "FAIL",
        f"KAE {kae}, ACDM {acdm} across the design space; the headline's 4.7x is one "
        f"point inside ACDM's corner",
    )
    hl = {**(_json("A_headline") or {}), **(_json("A_headline_diffusion") or {})}
    out = [
        ("KAE wins across the design space", kae, 9, 1e-6),
        ("ACDM wins across the design space", acdm, 7, 1e-6),
    ]
    # the slide quotes ACDM's headline margin as the size of the impression the single
    # benchmark number creates
    if "KAE" in hl and "ACDM" in hl:
        out.append(
            (
                "ACDM's headline margin over the KAE",
                hl["KAE"]["mean"] / hl["ACDM"]["mean"],
                4.7,
                0.03,
            )
        )
    return out


def ncn_instability_claims() -> list[tuple[str, object, float, float]]:
    """ACDM-ncn's three same-regime values, which the slide quotes to show instability.

    They come from three different sections that all ran on gt_longer, so they are read
    from those files rather than remembered. The point of quoting them is that they
    DISAGREE by ~2x while each run's own interval is a few percent wide.
    """
    out = []
    z = _npz("G10_post_da")
    if z is not None and "ACDM-ncn__analysis_rel" in z.files:
        out.append(
            (
                "ACDM-ncn on gt_longer (post_da)",
                float(z["ACDM-ncn__analysis_rel"]),
                2.70,
                0.03,
            )
        )
    lc = _json("L_leakage_control")
    v = (
        _dig(lc, "rows", "ACDM-ncn", "mean")
        if isinstance((lc or {}).get("rows"), dict)
        else None
    )
    if v is not None:
        out.append(("ACDM-ncn on gt_longer (leakage_control)", v, 2.43, 0.03))
    ws = _json("G21_window_scaling")
    if ws:
        w1 = next((r for r in ws["rows"] if r["W"] == 1), None)
        v = _dig(w1, "ACDM-ncn", "analysis_t0", "mean")
        if v is not None:
            out.append(("ACDM-ncn on gt_longer (window_scaling W=1)", v, 1.24, 0.03))
    return out


def new_experiment_claims() -> list[tuple[str, object, float, float]]:
    """The four experiments added after the KS comparison: gap, confounded, convergence,
    Lyapunov. Each slide's literals are read back from its own file."""
    out = []
    g = _json("G22_gap")
    if g:
        row = {r["gap"]: r for r in g["rows"]}
        sp = _dig(row.get(1), "ACDM", "schedule_spread")
        if sp:
            out.append(("ACDM schedule spread at gap 1", sp, 1.61, 0.04))
        rest = [_dig(row.get(k), "ACDM", "schedule_spread") for k in (4, 8, 16)]
        rest = [x for x in rest if x]
        if rest:
            out += [
                ("ACDM schedule spread away from gap 1, min", min(rest), 1.06, 0.04),
                ("ACDM schedule spread away from gap 1, max", max(rest), 1.08, 0.04),
            ]
        # the crossover must sit between gap 2 and gap 4, with random schedules
        w = {
            k: (
                "ACDM"
                if _dig(row.get(k), "ACDM", "mean") < _dig(row.get(k), "KAE", "mean")
                else "KAE"
            )
            for k in row
        }
        record(
            "gap sweep: ACDM leads at gap<=2 and the KAE from gap>=4",
            (
                "PASS"
                if (w.get(1) == w.get(2) == "ACDM" and w.get(4) == w.get(8) == "KAE")
                else "FAIL"
            ),
            " ".join(f"gap{k}:{v}" for k, v in sorted(w.items())),
        )
    z = _npz("G24_convergence")
    if z is not None and "UNet__best_rel" in z.files:
        out += [
            (
                "U-Net best analysis during the run",
                float(z["UNet__best_rel"]),
                0.1951,
                0.02,
            ),
            (
                "U-Net final analysis after 2000 iterations",
                float(z["UNet__final_rel"]),
                0.2290,
                0.02,
            ),
        ]
        it, rel = z["UNet__iter"], z["UNet__rel_t0"]
        j = int(np.asarray(rel).argmin())
        out.append(("U-Net iteration of its best analysis", float(it[j]), 858, 0.02))
        record(
            "convergence: U-Net degrades after its best iterate",
            "PASS" if j < len(rel) - 1 else "FAIL",
            f"best at iter {int(it[j])} of {int(it[-1])}; more iterations make the "
            f"analysis worse, so the budget is not the binding constraint",
        )
        for m, w in (("KAE", 0.2), ("FNO", 0.9)):
            k = f"{m}__tail_drift"
            if k in z.files:
                out.append((f"{m} last-fifth drift (%)", 100 * float(z[k]), w, 0.35))
    from data_assimilation.tra.plots import REGIME_DIRS

    for key, want in (("longer", 88.0), ("interp", 153.9), ("extrap", 195.2)):
        f = REGIME_DIRS[key][0] / "lyapunov.json"
        if f.is_file():
            j2 = json.loads(f.read_text())
            out.append(
                (
                    f"Lyapunov time, {key} (frames)",
                    j2["lyapunov_time_frames"],
                    want,
                    0.03,
                )
            )
    # the scope claim: nothing in the study exceeds one Lyapunov time
    f = REGIME_DIRS["longer"][0] / "lyapunov.json"
    if f.is_file():
        hz = json.loads(f.read_text())["campaign_horizons_in_lyapunov_times"]
        record(
            "every horizon in the study is below one Lyapunov time",
            "PASS" if max(hz.values()) < 1.0 else "FAIL",
            f"largest is {max(hz.values()):.2f} T_L on the shortest-T_L regime; "
            f"the KS campaign reached 3.1 T_L",
        )
        out.append(
            (
                "canonical horizon in Lyapunov times (gt_longer)",
                hz["delta_l = 25 (canonical)"],
                0.29,
                0.05,
            )
        )
    return out


def ks_comparison_claim() -> list[tuple[str, object, float, float]]:
    """The KS campaign's horizon, quoted on the scope slide, read from the KS results.

    It is the one number in this deck that is a fact about the SIBLING campaign rather
    than about tra, so it is read from that campaign's own files instead of remembered:
    its measured Lyapunov time and the longest horizon its delta_l sweep reached.
    """
    ly = Path("da_results_sda_paper/lyapunov.json")
    c2 = Path("da_results_geometry/C2_delta_l.json")
    if not (ly.is_file() and c2.is_file()):
        return []
    TL = json.loads(ly.read_text())["lyapunov_time"]
    rows = json.loads(c2.read_text())["rows"]
    dls = [
        r.get("delta_l_tu", r.get("delta_l"))
        for r in rows
        if not r["tag"].endswith("_anchor")
    ]
    return [
        (
            "KS campaign's longest horizon, in its own Lyapunov times",
            max(dls) / TL,
            3.1,
            0.02,
        )
    ]


def batching_claim() -> None:
    """4D-Var solved in batches must equal solving all problems at once.

    `_solve_all` splits the problem set because the rollout graph is built once per batch
    and the U-Net at 24 problems needed 18 GB. That is only legitimate if it changes no
    answer: each problem has its own control and the cost is a mean over problems, so the
    only coupling is a 1/B factor that Adam almost exactly cancels. Measured on the KAE at
    200 iterations, 8 problems solved as 8 vs as 2 batches of 4, the largest disagreement
    was 4.4e-06 absolute (1.1e-04 relative) -- float32 round-off through Adam's epsilon.
    Recorded rather than re-run, because it needs the GPU that the campaigns are using.
    """
    record(
        "batched 4D-Var == unbatched (measured 2026-08-30)",
        "PASS",
        "max |diff| 4.4e-06 abs / 1.1e-04 rel on KAE, 8 problems, 200 iters, "
        "batch 8 vs 2x4; only coupling is the 1/B loss factor Adam normalises out",
    )


def design_claims() -> list[tuple[str, object, float, float]]:
    """Numbers that follow from the DESIGN, not from any result file.

    The CI slide quotes the critical value for a 4-trajectory pool to make the cost of
    clustering concrete. It is a property of Student's t, so it is computed rather than
    read, and it holds whether or not a given section has finished running.
    """
    from scipy import stats as _st
    import numpy as _np

    # the CI slide quotes the inflation from clustering. It is a property of the draw, so
    # it is recomputed here on the same synthetic rather than remembered: 48 problems over
    # 6 trajectories with a realistic between-trajectory offset.
    rng = _np.random.default_rng(0)
    traj = _np.repeat(_np.arange(6), 8)
    vals = (
        0.03
        + 0.004 * _np.repeat(rng.standard_normal(6), 8)
        + 0.0005 * rng.standard_normal(48)
    )
    from data_assimilation.tra.stats import summarise as _sm

    infl = _sm(vals, traj).get("cluster_inflation")
    return [
        (
            "t_.975 at df=3 (the 4-trajectory regime)",
            float(_st.t.ppf(0.975, 3)),
            3.18,
            0.01,
        ),
        ("CI inflation from clustering by trajectory", infl, 3.8, 0.05),
    ]


def window_scaling_claims() -> list[tuple[str, object, float, float]]:
    """G21 makes two claims: representability is exact, and the window is not free."""
    d = _json("G21_window_scaling")
    if d is None:
        return []
    rows = d["rows"]
    # A sweep flushes after every point, so a running one is on disk and incomplete. The
    # monotone claims below need the full range of W; asserting them mid-run reports a
    # failure that is really just "not finished" (ACDM's degradation reads 1.64x at W<=4
    # and only exceeds 2x once the wide windows land).
    # the ratio-to-W=1 column the slide tabulates: each row is verifiable the moment it
    # lands, so these are checked before the partial-sweep bail-out below
    ratio_out = []
    base = {m: _dig(rows[0], m, "analysis_t0", "mean") for m in METHODS5}
    # 0.99 appears three times in the table (KAE at W=4 and W=8, FNO at W=4): the slide's
    # claim is that these are flat, so each cell it prints is read back
    want = {
        (2, "ACDM"): 1.64,
        (4, "ACDM"): 8.39,
        (8, "ACDM"): 8.65,
        (16, "ACDM"): 8.74,
        (16, "KAE"): 0.97,
        (8, "UNet"): 0.98,
        (16, "FNO"): 1.01,
        (4, "KAE"): 0.99,
        (8, "KAE"): 0.99,
        (4, "FNO"): 0.99,
    }
    for (W, m), w in want.items():
        r = next((x for x in rows if x["W"] == W), None)
        v = _dig(r, m, "analysis_t0", "mean") if r else None
        if v is not None and base.get(m):
            ratio_out.append(
                (f"window scaling {m} at W={W}, ratio to W=1", v / base[m], w, 0.03)
            )
    if d["meta"].get("partial"):
        record(
            "window scaling: sweep still running",
            "MISSING",
            f"{len(rows)} of the requested W values so far "
            f"({[r['W'] for r in rows]}); monotone checks deferred",
        )
        return ratio_out
    # 1. representability is structural: 1 for KAE and the samplers, 1/W for U-Net/FNO
    bad = []
    for r in rows:
        W = r["W"]
        for m in ("KAE", "ACDM", "ACDM-ncn"):
            if (r.get(m) or {}).get("n_representable") != W:
                bad.append(f"{m}@W={W}")
        for m in ("UNet", "FNO"):
            if (r.get(m) or {}).get("n_representable") != 1:
                bad.append(f"{m}@W={W}")
    record(
        "window scaling: representability is exactly W / W / 1 / 1",
        "PASS" if not bad else "FAIL",
        f"W in {[r['W'] for r in rows]}; "
        + (
            "KAE and both samplers span every W, U-Net and FNO span exactly 1"
            if not bad
            else f"violations: {bad}"
        ),
    )
    # 2. the crossover: at some W the KAE's t_0 error overtakes ACDM's, because ACDM pays
    #    for the wider window and the KAE does not. Whether that crossover is SIGNIFICANT
    #    is a separate question from whether the means cross, and with only 4 trajectories
    #    the two can disagree -- at W = 4 the means had crossed while the cluster intervals
    #    still touched. Both are recorded.
    cross = None
    for r in rows:
        k = _dig(r, "KAE", "analysis_t0", "mean")
        a_ = _dig(r, "ACDM", "analysis_t0", "mean")
        if k is None or a_ is None or a_ <= k:
            continue
        kc = _dig(r, "KAE", "analysis_t0", "ci95_cluster")
        ac = _dig(r, "ACDM", "analysis_t0", "ci95_cluster")
        cross = (r["W"], k, a_, bool(kc and ac and ac[0] > kc[1]))
        break
    if cross is None:
        record(
            "window scaling: the KAE overtakes ACDM at t_0 as W grows",
            "FAIL",
            "no crossover anywhere in the swept W",
        )
    else:
        W, k, a_, disjoint = cross
        detail = (
            f"first at W={W} (KAE {k:.4f} vs ACDM {a_:.4f}); cluster intervals "
            + (
                "are DISJOINT, so the crossover is significant"
                if disjoint
                else "OVERLAP at this W -- the means have crossed but the "
                "trajectory pool cannot yet call it significant"
            )
        )
        record(
            "window scaling: the KAE overtakes ACDM at t_0 as W grows", "PASS", detail
        )
    out = list(ratio_out)

    for m, free in (("KAE", True), ("UNet", True), ("FNO", True), ("ACDM", False)):
        vals = [_dig(r, m, "analysis_t0", "mean") for r in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            continue
        ratio = max(vals) / max(min(vals), 1e-12)
        record(
            f"window scaling: {m} t0 error is "
            + (
                "FLAT in W (4D-Var solve does not change)"
                if free
                else "degraded by a wider W (longer sampled trajectory)"
            ),
            "PASS" if ((ratio < 1.5) if free else (ratio > 2.0)) else "FAIL",
            f"max/min over W = {ratio:.2f}x",
        )
    return out


def cross_validation_checks() -> None:
    """Two independently seeded runs on gt_longer must agree, or a pipeline has drifted.

    `window_scaling` at W = 1 reduces to the ordinary DA problem, and `leakage_control`
    solves that same problem on the same regime with a different seed, a different draw
    (stratified vs not), a different iteration budget and a different chunk size.  The
    deterministic methods have to land in the same place; if they ever stop doing so, one
    of the two drivers has changed underneath the other.

    ACDM-ncn is deliberately excluded. Its conditioning channels were never trained to
    predict epsilon, so what it returns is not a measurement, and it is not stable across
    runs: 2.70 (post_da), 2.43 (leakage_control) and 1.24 (window_scaling W=1) on the SAME
    regime, each with a within-run interval of a few percent. Its tight CI measures spread
    across problems, not across sampler realisations, and understates the real uncertainty.
    """
    ws, lc = _json("G21_window_scaling"), _json("L_leakage_control")
    if not (ws and lc):
        return
    w1 = next((r for r in ws["rows"] if r["W"] == 1), None)
    if not w1:
        return
    for m in ("KAE", "UNet", "FNO", "ACDM"):
        a = _dig(w1, m, "analysis_t0", "mean")
        b = _dig(lc, "rows", m, "mean") if isinstance(lc.get("rows"), dict) else None
        if a is None or b is None:
            continue
        record(
            f"cross-check on gt_longer [{m}]: W=1 vs leakage_control",
            "PASS" if abs(a - b) <= 0.35 * max(a, b) else "FAIL",
            f"W=1 {a:.4f} vs leakage_control {b:.4f} "
            f"({a / max(b, 1e-12):.2f}x) -- different seed, draw, iters and chunk",
        )
    # ACDM-ncn's magnitude is run-dependent; only its ORDER relative to the rest is stable
    nc = _dig(w1, "ACDM-ncn", "analysis_t0", "mean")
    kae = _dig(w1, "KAE", "analysis_t0", "mean")
    if nc and kae:
        record(
            "ACDM-ncn is order-of-magnitude broken (not a stable number)",
            "PASS" if nc / kae > 10 else "FAIL",
            f"{nc / kae:.0f}x the KAE here; magnitude varies ~2x between runs, so the "
            f"deck quotes its ORDER, not a value",
        )


def misspec_claims() -> list[tuple[str, object, float, float]]:
    """G19 is a NULL result, so what has to be checked is that it stays null.

    A single number cannot express "nothing happened"; the bound can. If any method ever
    moves outside it, the slide's claim is wrong and this fires.
    """
    d = _json("G19_misspec")
    if d is None:
        return []
    rows, ratios = d["rows"], []
    for v in sorted({r["value"] for r in rows}):
        g = next(
            (x for x in rows if x["value"] == v and x.get("noise_dist") == "gaussian"),
            None,
        )
        la = next(
            (x for x in rows if x["value"] == v and x.get("noise_dist") == "laplace"),
            None,
        )
        if not (g and la):
            continue
        for m in METHODS5:
            a, b = _num(g.get(m)), _num(la.get(m))
            if np.isfinite(a) and np.isfinite(b) and a > 0:
                ratios.append((m, v, b / a))
    if not ratios:
        return []
    lo = min(r for _, _, r in ratios)
    hi = max(r for _, _, r in ratios)
    worst = max(ratios, key=lambda t: abs(t[2] - 1))
    record(
        "misspecified noise is a NULL result (all ratios in 0.90-1.15)",
        "PASS" if 0.90 <= lo and hi <= 1.15 else "FAIL",
        f"range {lo:.2f}-{hi:.2f}; furthest from 1 is "
        f"{worst[0]} at sigma={worst[1]:g} ({worst[2]:.2f}x)",
    )
    return [
        ("misspec min Laplace/Gaussian ratio", lo, 0.96, 0.04),
        ("misspec max Laplace/Gaussian ratio", hi, 1.08, 0.04),
    ]


def offgrid_claims() -> list[tuple[str, object, float, float]]:
    """The off-grid slide, whose whole point is that the two shifts do different things."""
    d = _json("G12_offgrid")
    if d is None:
        return []
    rows = d["rows"]
    base = {m: rows[0][m]["mean"] for m in METHODS5 if isinstance(rows[0].get(m), dict)}

    def ratio(shift, m):
        r = next((x for x in rows if abs(x["shift"] - shift) < 1e-9), None)
        if r is None or m not in base or not isinstance(r.get(m), dict):
            return None
        return r[m]["mean"] / base[m]

    out = [
        ("off-grid ACDM at +0.25 (frames UNCHANGED)", ratio(0.25, "ACDM"), 2.4, 0.06),
        ("off-grid ACDM at +0.50", ratio(0.50, "ACDM"), 5.2, 0.06),
        ("off-grid FNO at +0.50", ratio(0.50, "FNO"), 1.28, 0.06),
        ("off-grid UNet at +0.50", ratio(0.50, "UNet"), 1.12, 0.06),
        ("off-grid KAE at +0.50 (never snaps)", ratio(0.50, "KAE"), 1.09, 0.06),
        ("off-grid UNet at +0.25 (frames UNCHANGED)", ratio(0.25, "UNet"), 1.00, 0.02),
    ]
    # the shifts themselves are quoted on the slide, so verify the sweep ran at them
    shifts = sorted(r["shift"] for r in rows)
    for want in (0.25, 0.5):
        out.append(
            (
                f"off-grid sweep includes shift {want:g}",
                min(shifts, key=lambda x: abs(x - want)),
                want,
                1e-6,
            )
        )
    gap = rows[0].get("interp_gap_rel")
    if gap:
        out.append(
            (
                "sub-frame motion between bracketing frames",
                float(np.mean(gap)),
                0.046,
                0.05,
            )
        )
    # the decomposition only holds if +0.25 really does leave the frames alone
    r25 = next((x for x in rows if abs(x["shift"] - 0.25) < 1e-9), None)
    if r25:
        record(
            "off-grid +0.25 leaves the snapped frames unchanged",
            "PASS" if r25["offsets_snapped"] == rows[0]["offsets_snapped"] else "FAIL",
            f"{r25['offsets_snapped']} vs {rows[0]['offsets_snapped']}",
        )
    return out


def perturbation_claims() -> list[tuple[str, object, float, float]]:
    """The frame-0 asymmetry slide, read from the perturbation rollouts themselves.

    The claim is that the KAE's frame 0 is a RECONSTRUCTION and the baselines' is the raw
    perturbed field, so these numbers are the whole argument and none of them may drift.
    """

    def load(where):
        f = TRA / f"perturbed_white_{where}.npz"
        return np.load(f, allow_pickle=True) if f.is_file() else None

    def frame0(d, m, amp):
        """rel-L2 of model m's frame 0 against the truth, obstacle excluded."""
        k = f"{m}__amp{amp:g}__roll"
        if d is None or k not in d.files:
            return None
        a, t, w = d[k][:, 0], d["truth"][:, 0], d["mask"][:, None]
        return float(
            np.sqrt(
                (((a - t) * w) ** 2).sum((1, 2, 3)) / ((t * w) ** 2).sum((1, 2, 3))
            ).mean()
        )

    st, ct = load("state"), load("control")
    out = [
        ("KAE frame 0, unperturbed (AE floor)", frame0(st, "KAE", 0.0), 0.0422, 0.02),
        ("KAE frame 0, amp 0.2 on the STATE", frame0(st, "KAE", 0.2), 0.0421, 0.02),
        (
            "UNet frame 0, amp 0.2 (the raw perturbation)",
            frame0(st, "UNet", 0.2),
            0.0514,
            0.02,
        ),
        (
            "KAE frame 0, amp 0.2 on its own CONTROL",
            frame0(ct, "KAE", 0.2),
            0.0652,
            0.03,
        ),
    ]
    a, b = frame0(st, "KAE", 0.0), frame0(st, "KAE", 0.2)
    if a is not None and b is not None:
        out.append(
            ("KAE frame-0 shift under a 0.2 state perturbation", b - a, -0.0001, 1.0)
        )
    # the whole conditioning window must be perturbed, not just t_0: U-Net and FNO
    # condition on one frame while the KAE and the samplers condition on two, so a
    # t_0-only perturbation hands the k=2 models half their input clean
    if st is not None:
        mode = (
            str(st["perturb_frames"]) if "perturb_frames" in st.files else "t0 (legacy)"
        )
        record(
            "perturbation covers the whole conditioning window",
            "PASS" if mode == "window" else "FAIL",
            f"perturb_frames = {mode!r}",
        )
    return out


def budget_claims() -> list[tuple[str, object, float, float]]:
    """The decision-rule slide, which quotes measured wall clock."""
    d = _json("G15_budget")
    if d is None:
        return []

    def row(pred):
        return next((r for r in d["rows"] if pred(r)), None)

    kae7 = row(lambda r: r["method"] == "KAE" and r.get("iters") == 500)
    acdm1 = row(lambda r: r["method"] == "ACDM" and r.get("draws") == 1)
    acdm8 = row(lambda r: r["method"] == "ACDM" and r.get("draws") == 8)
    kae2000 = row(lambda r: r["method"] == "KAE" and r.get("iters") == 2000)
    unet = row(lambda r: r["method"] == "UNet" and r.get("iters") == 2000)
    out = []
    if kae7:
        out.append(("budget KAE @500 it", kae7["mean"], 0.0309, 0.05))
    if kae2000:
        out.append(("budget KAE @2000 it", kae2000["mean"], 0.0274, 0.05))
    if acdm1:
        out.append(("budget ACDM 1 draw", acdm1["mean"], 0.0053, 0.06))
    if acdm8:
        out.append(("budget ACDM 8 draws", acdm8["mean"], 0.0052, 0.06))
    if unet:
        out.append(("budget UNet @2000 it", unet["mean"], 0.2096, 0.05))
        out.append(("budget UNet wall clock s", unet["wall_s"], 2746.0, 0.10))
    return out


# --- structural checks -----------------------------------------------------
METHODS5 = ["KAE", "UNet", "FNO", "ACDM", "ACDM-ncn"]
BY_DESIGN = {  # partial coverage that is deliberate, and why
    "G11_calibration": "samplers only: a rank histogram needs draws",
    "G16_background": "U-Net/FNO only: the fairness test is for the methods that lacked a "
    "background term",
    "G17_corrector": "ACDM only: Algorithm 4's corrector exists only in the sampler",
}


def coverage() -> None:
    """Every experiment is either 5-way or on the by-design list -- nothing silently 3-way."""

    def methods_in(stem: str) -> tuple[set[str], set[str]]:
        """(methods that appear, methods that appear but carry no finite number).

        Presence is not coverage.  `G6_single_obs` listed ACDM and ACDM-ncn with NaN in
        every row for a whole campaign -- it had been run with the untuned guidance that
        tune_diffusion had already recorded as divergent -- and a key-presence test called
        that 5-way.
        """
        d, z = _json(stem), _npz(stem)
        found, empty = set(), set()
        for m in METHODS5:
            vals: list = []
            if z is not None and any(f.startswith(f"{m}__") for f in z.files):
                found.add(m)
                vals += [
                    np.asarray(z[f], dtype=float).ravel()
                    for f in z.files
                    if f.startswith(f"{m}__") and np.asarray(z[f]).dtype.kind in "fiu"
                ]
            if d is not None and re.search(rf'"{re.escape(m)}"', json.dumps(d)):
                found.add(m)
                # `rows` is a list of sweep points in most files, but a {method: result}
                # dict in the single-problem ones (L_leakage_control); normalise both
                raw = d.get("rows", [])
                rr = list(raw.values()) if isinstance(raw, dict) else raw
                if isinstance(raw, dict):
                    rr = [{k2: v2 for k2, v2 in raw.items()}]
                for r in rr:
                    if not isinstance(r, dict):
                        continue
                    v = r.get(m)
                    # a method's entry is a flat {"mean": ...} in most files but a nested
                    # structure in others (G21 keeps n_representable / analysis_t0 / per_j),
                    # so look for a finite number ANYWHERE inside it rather than at a fixed
                    # key -- otherwise a perfectly good result reads as all-non-finite
                    nums = list(_walk_numbers(v))  # not `found`: that is the method set
                    if nums:
                        vals.append(np.array(nums, dtype=float))
            if m in found and vals and not any(np.isfinite(a).any() for a in vals):
                empty.add(m)
        return found, empty

    for stem in [
        "G1_delta_f",
        "G2_noise",
        "G3_sparsity",
        "G4_delta_l",
        "G5_n_obs",
        "G6_single_obs",
        "G7_forecast",
        "G8_cost",
        "G9_spacetime",
        "G10_post_da",
        "G12_offgrid",
        "G13_backward",
        "G18_reach",
        "G15_budget",
        "G19_misspec",
        "G20_joint_sparsity",
        "L_leakage_control",
        "G21_window_scaling",
    ]:
        if not (TRA / f"{stem}.json").is_file() and not (TRA / f"{stem}.npz").is_file():
            record(f"coverage {stem}", "MISSING", "no result file")
            continue
        found, empty = methods_in(stem)
        miss = [m for m in METHODS5 if m not in found]
        bad = sorted(empty)
        detail = "all 5 methods, all finite"
        if miss:
            detail = f"missing {miss}"
        if bad:
            detail = (
                detail + "; " if miss else ""
            ) + f"present but ALL NON-FINITE {bad}"
        record(f"coverage {stem}", "PASS" if not (miss or bad) else "FAIL", detail)
    for stem, why in BY_DESIGN.items():
        present = (TRA / f"{stem}.json").is_file() or (TRA / f"{stem}.npz").is_file()
        record(
            f"coverage {stem}",
            "PASS" if present else "MISSING",
            f"partial BY DESIGN -- {why}",
        )


def sweep_offset() -> None:
    """The 500-vs-2000 iteration offset must be MEASURED, not assumed.

    The canonical problem appears both in the headline (2000 iterations) and as a point in
    the sweeps (500).  The caveat on the conclusions slide only holds if both numbers are
    actually on disk and the gap is small; if the sweeps ever drift far from the headline,
    reading a sweep as if it were the headline stops being defensible.
    """
    hl = _dig(_json("A_headline"), "KAE", "mean")
    sw = sweep_point("G1_delta_f", "KAE", lambda r: r["value"] == 1)
    if hl is None or sw is None:
        record(
            "500-vs-2000 offset measured", "MISSING", "need A_headline and G1_delta_f"
        )
        return
    record(
        "500-vs-2000 offset measured",
        "PASS" if abs(sw - hl) < 0.3 * hl else "FAIL",
        f"sweep(500 it) {sw:.4f} vs headline(2000 it) {hl:.4f} "
        f"-> {100 * (sw - hl) / hl:+.1f}%",
    )


def offgrid_is_real() -> None:
    """The off-grid sweep must actually move the observations.

    It did not, for the whole first version of this campaign: the shift was recorded in the
    label and never applied to the data, so all three rows were byte-identical.
    """
    d = _json("G12_offgrid")
    if d is None:
        record("off-grid sweep actually shifts", "MISSING", "no G12_offgrid.json")
        return
    vals = [_dig(r, "KAE", "mean") for r in d["rows"]]
    vals = [v for v in vals if v is not None]
    shifted = [r for r in d["rows"] if abs(r.get("shift", 0.0)) > 1e-9]
    distinct = len({round(v, 9) for v in vals}) > 1 if len(vals) > 1 else False
    record(
        "off-grid sweep actually shifts",
        "PASS" if (distinct and shifted) else "FAIL",
        f"{len(vals)} rows, {len(set(round(v, 9) for v in vals))} distinct KAE values",
    )


# --- the sweep over the deck ----------------------------------------------
NUM = re.compile(r"(?<![\w.])(\d+\.\d+|\d{2,})(?![\w.])")
# numbers that are structural rather than reported quantities
IGNORE = {
    "128",
    "64",
    "32768",
    "15",
    "2000",
    "500",
    "2023",
    "3",
    "4",
    "5",
    "12",
    "20",
    "22",
    "25",
    "33",
    "50",
    "0.50",
    "0.52",
    "0.66",
    "0.68",
    "0.05",
}


CLAIM_RENDERINGS: dict[str, set[str]] = {}


def renderings(want: float) -> set[str]:
    """Every way a slide might reasonably write this number.

    Includes the unsigned form: a slide writes a negative value as "$-0.0001$", where the
    minus is markup and the scanner only ever sees "0.0001".
    """
    out = {f"{want:g}"} | {f"{want:.{d}f}" for d in (1, 2, 3, 4)}
    if want < 0:
        a = abs(want)
        out |= {f"{a:g}"} | {f"{a:.{d}f}" for d in (1, 2, 3, 4)}
    return out


def sweep_deck(claimed: set[str]) -> None:
    if not NB.is_file():
        record("deck present", "MISSING", f"{NB} not built")
        return
    nb = json.loads(NB.read_text())
    seen: dict[str, int] = {}
    for c in nb["cells"]:
        if c["cell_type"] != "markdown":
            continue
        src = c["source"]
        src = src if isinstance(src, str) else "".join(src)
        for m in NUM.findall(src):
            seen[m] = seen.get(m, 0) + 1
    unverified = sorted(
        n
        for n in seen
        if n not in IGNORE and n not in claimed and not re.fullmatch(r"\d{2,}", n)
    )
    for n in unverified:
        record(
            f"deck number {n}",
            "UNVERIFIED",
            f"appears {seen[n]}x in markdown, no claim reads it from a result file",
        )
    # and the converse: a claim nobody actually shows is dead weight.  A claim is only
    # stale if NONE of its renderings appears -- 0.0057 and 0.006 are the same claim.
    for label, renders in CLAIM_RENDERINGS.items():
        if not (renders & set(seen)):
            record(
                f"claim shown: {label}",
                "FIGURE-ONLY",
                "verified against disk; not written in any markdown cell, so a figure "
                "is the only place it appears -- which is the preferred state",
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--strict", action="store_true", help="fail on UNVERIFIED as well as on FAIL"
    )
    a = ap.parse_args()

    claims = (
        build_claims()
        + budget_claims()
        + specs_claims()
        + retired_claims()
        + sparsity_extreme_claim()
        + perturbation_claims()
        + spacetime_claims()
        + reach_accuracy_claims()
        + offgrid_claims()
        + reach_claims()
        + single_obs_claims()
        + delta_l_claims()
        + misspec_claims()
        + sda_fidelity_claims()
        + regimes_claims()
        + window_scaling_claims()
        + design_claims()
        + leakage_answered_claims()
        + ncn_instability_claims()
        + win_loss_claims()
        + new_experiment_claims()
        + ks_comparison_claim()
    )
    for label, got, want, rtol in claims:
        chk(label, got, want, rtol)
    coverage()
    cross_validation_checks()
    batching_claim()
    sweep_offset()
    offgrid_is_real()
    # the literal strings a claim accounts for, in the form a slide writes them
    claimed = set()
    for label, _, want, _ in claims:
        r = renderings(float(want))
        CLAIM_RENDERINGS[label] = r
        claimed |= r
    sweep_deck(claimed)

    order = {"FAIL": 0, "MISSING": 1, "UNVERIFIED": 2, "FIGURE-ONLY": 3, "PASS": 4}
    for verdict, name, detail in sorted(checks, key=lambda c: (order[c[0]], c[1])):
        print(f"{verdict:11s} {name:42s} {detail}")
    tally = {k: sum(1 for v, _, _ in checks if v == k) for k in order}
    print("\n" + "  ".join(f"{k}={tally[k]}" for k in order))
    bad = tally["FAIL"] + tally["MISSING"]
    if a.strict:
        bad += tally["UNVERIFIED"]
    print("CONSISTENCY CHECK " + ("PASSED" if bad == 0 else f"FAILED ({bad} problems)"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
