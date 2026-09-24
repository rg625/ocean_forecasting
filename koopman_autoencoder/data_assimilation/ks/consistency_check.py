"""Final QC: verify that every headline number in the report matches the saved results.

A report is only as trustworthy as its weakest transcription.  This re-reads each claimed
number from the .npz/.json that produced it and fails loudly on any mismatch, so a stale
figure caption or a hand-copied table cannot survive into the manuscript.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

GEO = Path("da_results_geometry")
LONG = Path("da_results_long")
SDA = Path("da_results_sda_paper")
REPORT = Path("DA_REVISION_REPORT.md")

checks: list[tuple[str, bool, str]] = []


def chk(name, got, want, rtol=0.02):
    if want is None or got is None or (isinstance(got, float) and not np.isfinite(got)):
        ok = got is want
    else:
        ok = abs(got - want) <= rtol * max(abs(want), 1e-12)
    checks.append((name, ok, f"got {got!r}, report says {want!r}"))


def geo_row(tag_file, idx, method, key="mean"):
    d = json.loads((GEO / f"{tag_file}.json").read_text())
    r = d["rows"][idx]
    return r[method][key] if method in r and key in r[method] else None


def main():
    # ---- Lyapunov -----------------------------------------------------------
    ly = json.loads((SDA / "lyapunov.json").read_text())
    chk("lambda_1", ly["lambda_1_mean"], 0.04432)
    chk("T_L", ly["lyapunov_time"], 22.56)

    # ---- Koopman spectrum ---------------------------------------------------
    ks = json.loads((SDA / "koopman_spectrum.json").read_text())
    g = ks["growth_rate_comparison"]
    chk("max Re eig K", g["kae_max_growth_rate_max_Re_eig_K"], 0.00727)
    chk("growth ratio", g["ratio_true_over_kae"], 6.1, rtol=0.05)
    chk("n decaying modes", ks["eigenvalues"]["n_unstable_re_gt_0"], 7, rtol=0)

    # ---- U-Net audit --------------------------------------------------------
    au = json.loads((SDA / "unet_spacetime_audit.json").read_text())
    chk(
        "rollout check exact",
        au["independent_rollout_check"]["max_abs_difference"],
        0.0,
        rtol=0,
    )
    sp = au["spectral"]["t0"]["UNet"]
    chk("t0 share k5-12", sp["k5-12"]["share_of_total_sq_error"], 0.991, rtol=0.01)
    chk("t0 share k13-32", sp["k13-32"]["share_of_total_sq_error"], 0.009, rtol=0.15)
    ns = json.loads((SDA / "unet_nullspace.json").read_text())
    chk(
        "J ratio (cost prefers truth)",
        ns["cost_function_blindness"]["ratio"],
        14.2,
        rtol=0.02,
    )
    chk(
        "contraction along error",
        ns["perturbation"]["one_step_contraction_mean"],
        0.125,
        rtol=0.02,
    )
    chk(
        "contraction random",
        ns["perturbation"]["random_direction_contraction_mean"],
        0.778,
        rtol=0.02,
    )

    # ---- forecast -----------------------------------------------------------
    fc = np.load(LONG / "FC_forecast.npz")
    chk(
        "FC skill horizon KAE (T_L)",
        float(fc["KAE-expm__skill_horizon_tu"]) / float(fc["T_L"]),
        2.42,
        rtol=0.02,
    )
    chk(
        "FC skill horizon UNet (T_L)",
        float(fc["UNet__skill_horizon_tu"]) / float(fc["T_L"]),
        9.29,
        rtol=0.02,
    )
    pf = np.load(LONG / "PF_post_da_forecast.npz")
    chk("PF analysis KAE", float(pf["KAE-expm__analysis_rel"]), 0.0134, rtol=0.03)
    chk("PF analysis UNet", float(pf["UNet__analysis_rel"]), 0.1472, rtol=0.03)
    chk(
        "PF skill horizon KAE (T_L)",
        float(pf["KAE-expm__skill_horizon_tu"]) / float(pf["T_L"]),
        0.53,
        rtol=0.03,
    )
    chk(
        "PF skill horizon UNet (T_L)",
        float(pf["UNet__skill_horizon_tu"]) / float(pf["T_L"]),
        10.88,
        rtol=0.03,
    )

    # ---- cost ---------------------------------------------------------------
    ci = json.loads((LONG / "cost_isolated.json").read_text())
    chk(
        "cost ratio KAE-expm over 800x",
        ci["scaling"]["KAE-expm"]["cost_ratio"],
        0.77,
        rtol=0.03,
    )
    chk("cost ratio KAE-rk4", ci["scaling"]["KAE-rk4"]["cost_ratio"], 124.25, rtol=0.03)

    # ---- temporal geometry --------------------------------------------------
    chk("C1 KAE at df=0.1", geo_row("C1_delta_f", 0, "KAE-expm"), 0.0045, rtol=0.03)
    chk("C1 KAE at df=1.8", geo_row("C1_delta_f", 5, "KAE-expm"), 0.5288, rtol=0.03)
    chk("C1 UNet at df=1.8", geo_row("C1_delta_f", 5, "UNet"), 0.3375, rtol=0.03)
    chk("C3 KAE at N=2", geo_row("C3_n_obs", 0, "KAE-expm"), 0.0422, rtol=0.03)
    chk("C3 KAE at N=5", geo_row("C3_n_obs", 2, "KAE-expm"), 0.0044, rtol=0.03)
    c2 = json.loads((GEO / "C2_delta_l.json").read_text())["rows"]
    sweep = [r for r in c2 if not r["tag"].endswith("_anchor")]
    chk("C2 KAE at 0.02 T_L", sweep[0]["KAE-expm"]["mean"], 0.0158, rtol=0.03)
    chk("C2 KAE at 3.10 T_L", sweep[-1]["KAE-expm"]["mean"], 0.0456, rtol=0.03)
    chk("C2 UNet at 3.10 T_L", sweep[-1]["UNet"]["mean"], 0.9930, rtol=0.03)
    anc = [r for r in c2 if r["tag"].endswith("_anchor")]
    if anc:
        chk("C2 anchor KAE (8000 it)", anc[0]["KAE-expm"]["mean"], 0.0041, rtol=0.05)
        chk(
            "C2 anchor SDA == sweep SDA",
            anc[0]["SDA"]["mean"],
            sweep[2]["SDA"]["mean"],
            rtol=0.01,
        )

    # ---- SDA divergence -----------------------------------------------------
    dv = json.loads((SDA / "sda_divergence.json").read_text())["onset"]
    chk("SDA diverged at N=13", dv[4]["n_diverged"], 2, rtol=0)
    chk("SDA diverged at N=16", dv[5]["n_diverged"], 10, rtol=0)

    # ---- independent cross-validation: C2 and LB share a horizon -------------
    # C2 runs on data/ks/da_test.nc and LB on data/ks/da_test_long.nc, with different
    # seeds, different t0 draws and separately written drivers. At delta_l = 2.5 with the
    # same 1000-iteration budget they must agree to within sampling error. If they ever
    # diverge, one of the two pipelines has drifted.
    lb_f = LONG / "LB_long_baseline.json"
    if lb_f.is_file():
        lb = json.loads(lb_f.read_text())
        row = next((r for r in lb["rows"] if abs(r["delta_l_tu"] - 2.5) < 1e-6), None)
        if row and lb["meta"].get("iters") == 1000:
            c2_25 = next(r for r in sweep if abs(r["delta_l"] - 2.5) < 1e-6)
            for m in ("KAE-expm", "UNet", "SDA"):
                if m in row and "mean" in row[m] and m in c2_25:
                    a, b = row[m]["mean"], c2_25[m]["mean"]
                    checks.append(
                        (
                            f"C2 vs LB agree at delta_l=2.5 [{m}]",
                            abs(a - b) <= 0.12 * max(a, b),
                            f"LB {a:.4f} vs C2 {b:.4f} on different test sets",
                        )
                    )

    # ---- test discipline ----------------------------------------------------
    ts = json.loads((SDA / "training_specs.json").read_text())
    shared = sum(
        v["n_trajectories_shared"]
        for d in ts["test_discipline"]["held_out_vs"].values()
        for v in d.values()
    )
    chk("total shared trajectories (leakage)", shared, 0, rtol=0)

    # ---- report cross-check -------------------------------------------------
    txt = REPORT.read_text()
    for pat, label in [
        (r"9\.29", "FC U-Net skill horizon in report"),
        (r"2\.42", "FC KAE skill horizon in report"),
        (r"14\.2", "J ratio in report"),
        (r"0\.00727", "max Re eig K in report"),
        (r"117", "C1 degradation factor in report"),
        (r"20\.8", "SDA divergence rate in report"),
    ]:
        checks.append(
            (label, bool(re.search(pat, txt)), "not found in the report text")
        )

    n_bad = sum(1 for _, ok, _ in checks if not ok)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:44s} {'' if ok else detail}")
    print(f"\n{len(checks) - n_bad}/{len(checks)} consistency checks pass")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
