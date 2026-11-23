"""Score the proxy-selected post-processing on the test cases (full checkpoints, 3 seeds each).

Inputs (RES = round-1/2 scratch outputs): single decodes res/pp/<key>.npz (sigma index 0), the
mixture + mirror average res/pp2/apply_<key>.npz (enc:cal:mirror), the s = 1 mirror average
OUT/apply0_<key>.npz (enc:0:mirror; pp2_apply.py). Candidates as in pp3_proxy.py, per seed
(target power = mean of the seed's plain and mirrored single decode) and on the 3-seed average
(target = mean over the 6 single decodes). Reported per model: mean over seeds with the paired
change vs the single decode +- SE over seeds, and the 3-seed version.

    python pp3_test.py MODEL OUT_DIR [--cands NAME ...] [--no-lsim] [--tag pp3_test]
(lpm(mirror), sharp+lpm, lpm+sharp: round-4 winners, pp4_proxy.py)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RES = Path(
    "/tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/"
    "093163e5-e75d-47aa-b04b-9bb0dda37f82/scratchpad/res"
)
sys.path.insert(0, str(RES / "pp"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import pp_analyze as P1  # noqa: E402
from pp3_proxy import ampmean, power, qmap, regain, spec_metrics  # noqa: E402
from pp4_proxy import lpm  # noqa: E402

CASES = ["extrap", "interp", "longer"]
BASE = ["single", "mirror", "mix+mirror"]


def build(name, f, e_single, M, fl):
    """f: dict of base fields; name e.g. sharp(mirror,a1,c5)"""
    if name in f:
        return f[name]
    if name.startswith("sharp("):
        an, al, cap = name[6:-1].split(",")
        A = f[an]
        g = np.clip(
            (e_single / np.maximum(power(A, M), 1e-30)) ** (float(al[1:]) / 2),
            1,
            float(cap[1:]),
        )
        return regain(A, M, g) * fl
    if name == "lpm(mirror)":
        return lpm(f["mirror"], f["singles"], fl > 0)
    if name == "sharp+lpm":
        return lpm(
            build("sharp(mirror,a1,c5)", f, e_single, M, fl), f["singles"], fl > 0
        )
    if name == "lpm+sharp":
        lp = lpm(f["mix+mirror"], f["singles"], fl > 0)
        g = np.clip(np.sqrt(e_single / np.maximum(power(lp, M), 1e-30)), 1, 5)
        return regain(lp, M, g) * fl
    if name == "ampmean(mirror)":
        xs = f["singles"]
        return ampmean(xs, [1 / len(xs)] * len(xs), M) * fl
    if name == "qmap(mix+mirror)":
        return np.stack(
            [
                qmap(f["mix+mirror"][s], [x[s] for x in f["singles"]], fl > 0)
                for s in range(len(M))
            ]
        )
    raise ValueError(name)


def score(case, x, do_lsim):
    r = {"mse": float(P1.mse_all(case, x)), **_spec(case, x)}
    if do_lsim:
        from lsim_score import lsim

        r["lsim"] = float(lsim(P1.to_acdm(case, x)[None], case)[0])
    return r


def _spec(case, x):
    return spec_metrics(P1.gt_case(case)[2], P1.to_acdm(case, x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--cands", nargs="*", default=[])
    ap.add_argument("--no-lsim", action="store_true")
    ap.add_argument("--tag", default="pp3_test")
    a = ap.parse_args()
    out = Path(a.out_dir)
    names = BASE + a.cands
    res: dict = {}
    for case in CASES:
        per_seed: dict = {n: [] for n in names}
        fields: list = []
        for s in range(3):
            key = f"{a.model}_s{s}"
            d = np.load(RES / "pp" / f"{key}.npz")
            P = d[f"{case}_pred"][0].astype(np.float64)
            M = d[f"{case}_clim"][:, None].astype(np.float64)  # [S, 1, C, H, W]
            mir = np.load(out / f"apply0_{key}.npz")[f"{case}|enc:0:mirror"].astype(
                np.float64
            )
            mixm = np.load(RES / "pp2" / f"apply_{key}.npz")[
                f"{case}|enc:cal:mirror"
            ].astype(np.float64)
            fl = (np.abs(P).sum((0, 1, 2)) > 0).astype(np.float64)
            Pm = (2 * mir - P) * fl  # mirrored single decode
            f = {"single": P, "mirror": mir, "mix+mirror": mixm, "singles": [P, Pm]}
            e_single = 0.5 * (power(P, M) + power(Pm, M))
            for n in names:
                x = build(n, f, e_single, M, fl)
                per_seed[n].append(score(case, x.astype(np.float32), not a.no_lsim))
                print(
                    case,
                    key,
                    f"{n:28s}",
                    " ".join(f"{k}={v:.3f}" for k, v in per_seed[n][-1].items()),
                    flush=True,
                )
            fields.append((f, e_single))
        # 3-seed versions
        f3 = {k: np.mean([f[k] for f, _ in fields], 0) for k in BASE}
        f3["singles"] = [x for f, _ in fields for x in f["singles"]]
        e3 = np.mean([e for _, e in fields], 0)
        ens = {}
        for n in names:
            if n == "single":
                continue
            ens[n] = score(
                case, build(n, f3, e3, M, fl).astype(np.float32), not a.no_lsim
            )
            print(
                case,
                "3seed",
                f"{n:28s}",
                " ".join(f"{k}={v:.3f}" for k, v in ens[n].items()),
                flush=True,
            )
        res[case] = {"per_seed": per_seed, "ens3": ens}
        del fields, f3
        json.dump(res, open(out / f"{a.tag}_{a.model}.json", "w"), indent=1)
    # paired summary
    for case in CASES:
        print(f"\n== {a.model} {case}: seed mean [paired d vs single +- SE] | 3-seed")
        ps, en = res[case]["per_seed"], res[case]["ens3"]
        for n in names:
            row = f"  {n:28s}"
            for k in ["mse", "lsim", "log_mid", "log_high"]:
                if k not in ps[n][0]:
                    continue
                v = np.array([q[k] for q in ps[n]])
                dlt = v - np.array([q[k] for q in ps["single"]])
                e = en.get(n, {}).get(k, np.nan)
                row += f" | {k} {v.mean():.3f} [{dlt.mean():+.3f}+-{dlt.std(ddof=1) / np.sqrt(3):.3f}] {e:.3f}"
            print(row)


if __name__ == "__main__":
    main()
