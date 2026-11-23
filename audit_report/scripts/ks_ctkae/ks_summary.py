"""Collect the KS results of the CT-KAE candidates next to the paper's models.

Paper rows (legacy KAE = the paper's KS KAE, U-Net 4D-Var, SDA) are read from the main
repository's stored results (da_results_geometry, da_results_long), which were produced on the
same seeded problems; the CT-KAE rows from results/ks_ctkae/<model>_s<seed>/. Writes
results/ks_ctkae/summary.json and prints the tables.
"""

import json
from pathlib import Path

import numpy as np

MAIN = Path("/home/rg625/mnt/ocean_forecasting/koopman_autoencoder")
OUT = Path(__file__).resolve().parents[2] / "results/ks_ctkae"
# (paper campaign, CT-KAE dir) per section. The paper cites the delta_f = 1 campaign
# (da_results_geometry); run_ks_eval.sh ran C2-JS at exp_geometry's current default delta_f = 9,
# which is the main repository's da_results_geometry_df9 campaign, and run_ks_df1.sh re-ran them
# at delta_f = 1. S1 and C1 do not depend on the pin.
SWEEPS = ["C2_delta_l", "C3_n_obs", "noise_law", "joint_sparsity"]
GEO = {
    "geometry": {
        "S1_single_obs": ("da_results_geometry", "geometry"),
        "C1_delta_f": ("da_results_geometry", "geometry"),
        **{k: ("da_results_geometry", "geometry_df1") for k in SWEEPS},
    },
    "geometry_df9": {k: ("da_results_geometry_df9", "geometry") for k in SWEEPS},
}
PAPER = {"legacy KAE": "KAE-expm", "U-Net": "UNet", "SDA": "SDA"}
CT = {"CT-KAE ref": "ref", "CT-KAE back001": "back001"}


def mean(row, m):
    v = row.get(m)
    return (
        None if not isinstance(v, dict) or v.get("mean") is None else float(v["mean"])
    )


def jl(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def npz(p):
    return dict(np.load(p, allow_pickle=True)) if Path(p).exists() else None


def main():
    s: dict = {
        "geometry": {},
        "geometry_df9": {},
        "long_baseline": [],
        "forecast": {},
        "post_da": {},
        "tuning": {},
    }
    for camp, secs in GEO.items():
        for sec, (pdir, cdir) in secs.items():
            s[camp][sec] = geo_rows(
                jl(MAIN / f"{pdir}/{sec}.json"),
                {k: jl(OUT / f"{t}_s0/{cdir}/{sec}.json") for k, t in CT.items()},
            )
    _rest(s)


def geo_rows(pap, cts):
    """paper rows with the CT-KAE rows of the same tag; the schedules must agree"""
    rows = []
    for r in pap["rows"]:
        row = {
            "tag": r["tag"],
            "value": r["value"],
            "taus": r.get("taus"),
            "ae_floor": r.get("ae_floor"),
        }
        row.update({k: mean(r, m) for k, m in PAPER.items()})
        for k, d in cts.items():
            hit = (
                None
                if d is None
                else next((x for x in d["rows"] if x["tag"] == r["tag"]), None)
            )
            if hit is not None and r.get("frames") is not None:
                assert list(hit["frames"]) == list(r["frames"]), (
                    r["tag"],
                    hit["frames"],
                    r["frames"],
                )
            row[k] = None if hit is None else mean(hit, "KAE-expm")
            if hit is not None:
                row[k + " floor"] = hit.get("ae_floor")
        rows.append(row)
    return rows


def _rest(s):
    # long-baseline assimilation (da_test_long)
    pap = jl(MAIN / "da_results_long/LB_long_baseline.json")
    cts = {k: jl(OUT / f"{t}_s0/long/LB_long_baseline.json") for k, t in CT.items()}
    for r in pap["rows"]:
        row = {"delta_l_tu": r["delta_l_tu"], "delta_l_TL": r["delta_l_TL"]}
        row.update({k: mean(r, m) for k, m in PAPER.items()})
        for k, d in cts.items():
            hit = (
                None
                if d is None
                else next(
                    (
                        x
                        for x in d["rows"]
                        if x["delta_l_frames"] == r["delta_l_frames"]
                    ),
                    None,
                )
            )
            row[k] = None if hit is None else mean(hit, "KAE-expm")
        s["long_baseline"].append(row)
    # free-running forecast from the true state (da_test_long, 32 trajectories)
    fc = npz(MAIN / "da_results_long/FC_forecast.npz")
    TL = float(fc["T_L"])
    pts = [0.1, 0.5, 1.0, 2.0, 5.0]  # Lyapunov times

    def at(f, key):
        return {str(p): float(np.interp(p * TL, f["taus"], f[key])) for p in pts}

    s["forecast"]["lyapunov_time_tu"] = TL
    s["forecast"]["legacy KAE"] = {
        "rel_at_TL": at(fc, "KAE-expm__rel_mean"),
        "skill_horizon_TL": float(fc["KAE-expm__skill_horizon_tu"]) / TL,
        "roundtrip": float(fc["kae_roundtrip"]),
    }
    s["forecast"]["U-Net"] = {
        "rel_at_TL": at(fc, "UNet__rel_mean"),
        "skill_horizon_TL": float(fc["UNet__skill_horizon_tu"]) / TL,
    }
    for k, t in CT.items():
        per = []
        for sd in range(3):
            f = npz(OUT / f"{t}_s{sd}/long/FC_forecast.npz")
            if f is None:
                continue
            per.append(
                {
                    "rel_at_TL": at(f, "KAE-expm__rel_mean"),
                    "skill_horizon_TL": float(f["KAE-expm__skill_horizon_tu"]) / TL,
                    "roundtrip": float(f["kae_roundtrip"]),
                }
            )
        s["forecast"][k] = {"seeds": per}
    # forecast from each method's own analysis
    pf = npz(MAIN / "da_results_long/PF_post_da_forecast.npz")

    def pfrow(f, m):
        return {
            "analysis_rel": float(f[f"{m}__analysis_rel"]),
            "rel_at_TL": at(f, f"{m}__rel_mean"),
            "skill_horizon_TL": float(f[f"{m}__skill_horizon_tu"]) / TL,
        }

    s["post_da"]["legacy KAE"] = pfrow(pf, "KAE-expm")
    s["post_da"]["U-Net"] = pfrow(pf, "UNet")
    for k, t in CT.items():
        f = npz(OUT / f"{t}_s0/long/PF_post_da_forecast.npz")
        if f is not None:
            s["post_da"][k] = pfrow(f, "KAE-expm")
    for k, t in CT.items():
        d = jl(OUT / f"{t}_s0/tuning.json")
        if d is not None:
            s["tuning"][k] = {
                "best": d["results"]["KAE-expm"]["best"],
                "grid": d["results"]["KAE-expm"]["grid"],
            }
    s["tuning"]["legacy KAE"] = {
        "best": jl(MAIN / "da_results_v2/tuning.json")["results"]["KAE-expm"]["best"]
    }
    (OUT / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    cols = list(PAPER) + list(CT)

    def f3(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "   --  "
        return f"{v:7.4f}"

    for camp in ["geometry", "geometry_df9"]:
        for sec, rows in s[camp].items():
            print(f"\n== {camp} {sec}   " + "  ".join(f"{c:>14s}" for c in cols))
            for r in rows:
                print(
                    f"  {str(r['value']):>12s} "
                    + "  ".join(f"{f3(r[c]):>14s}" for c in cols)
                )
    print("\n== long baseline (delta_l in T_L)")
    for r in s["long_baseline"]:
        print(
            f"  {r['delta_l_TL']:6.2f} " + "  ".join(f"{f3(r[c]):>14s}" for c in cols)
        )
    print("\n== forecast", json.dumps(s["forecast"], indent=0)[:2500])
    print("\n== post-DA forecast", json.dumps(s["post_da"], indent=0)[:1500])


if __name__ == "__main__":
    main()
