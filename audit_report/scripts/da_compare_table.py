"""Side-by-side DA results (rel-L2 of the t0 analysis, mean +- sem over problems) for the
KAE slot filled by different models, on identical DA code, plus the other methods from the
original September campaign for context (those rows were not re-run).

    python da_compare_table.py LABEL=DIR ... [--ref DIR_ORIGINAL] [--json out.json]
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument(
    "runs", nargs="+", help="LABEL=RESULT_DIR (contains canonical/ and df9/)"
)
ap.add_argument(
    "--ref", nargs="*", default=[], help="SCHEDULE=DIR of the original campaign"
)
ap.add_argument("--json", default=None)
args = ap.parse_args()
runs = dict(r.split("=", 1) for r in args.runs)
refs = dict(r.split("=", 1) for r in args.ref)
SECTIONS = [
    "A_headline",
    "G1_delta_f",
    "G2_noise",
    "G3_sparsity",
    "G4_delta_l",
    "G5_n_obs",
    "G6_single_obs",
    "G19_misspec",
    "G20_joint_sparsity",
]
OTHERS = ["UNet", "FNO", "ACDM", "ACDM-ncn"]
out = {}
for sched in ["canonical", "df9"]:
    for sec in SECTIONS:
        table: dict = {}
        for lab, d in runs.items():
            p = Path(d) / sched / f"{sec}.json"
            if not p.exists():
                continue
            js = json.loads(p.read_text())
            if "rows" in js:
                for r in js["rows"]:
                    if "KAE" in r:
                        table.setdefault(r["tag"], {})[lab] = (
                            r["KAE"]["mean"],
                            r["KAE"]["sem"],
                        )
            elif "summary" in js and "KAE" in js["summary"]:
                table.setdefault("headline", {})[lab] = (
                    js["summary"]["KAE"]["mean"],
                    js["summary"]["KAE"]["sem"],
                )
        ref = refs.get(sched)
        if ref and (Path(ref) / f"{sec}.json").exists():
            js = json.loads((Path(ref) / f"{sec}.json").read_text())
            for r in js.get("rows", []):
                for m in OTHERS:
                    if m in r and r["tag"] in table:
                        table[r["tag"]][m] = (r[m]["mean"], r[m]["sem"])
        if not table:
            continue
        out[f"{sched}/{sec}"] = table
        labs = list(runs) + [m for m in OTHERS if any(m in v for v in table.values())]
        print(f"\n== {sched} / {sec}")
        print(f"{'point':10s} " + " ".join(f"{lab:>16s}" for lab in labs))
        for tag, v in table.items():
            print(
                f"{tag:10s} "
                + " ".join(
                    f"{v[lab][0]:9.4f}±{v[lab][1]:.3f}" if lab in v else f"{'-':>16s}"
                    for lab in labs
                )
            )
if args.json:
    Path(args.json).write_text(json.dumps(out, indent=1))
