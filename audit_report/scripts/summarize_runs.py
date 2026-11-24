"""Summarise CT-KAE runs: best validation entry per run (held-out Mach only; no test data).

python summarize_runs.py [RUN_ROOT] [--json out.json]
"""

from _paths import REPO_ROOT
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument(
    "root",
    nargs="?",
    default=str(REPO_ROOT / "koopman_autoencoder/model_outputs_ctkae"),
)
ap.add_argument("--json", default=None)
ap.add_argument(
    "--select", default="val_mse60", help="val key used to pick each run's best entry"
)
args = ap.parse_args()
rows = []
for log in sorted(Path(args.root).glob("*/log.jsonl")):
    vals = [
        json.loads(line)["val"]
        for line in log.read_text().splitlines()
        if '"val"' in line
    ]
    if not vals:
        continue
    best = min(vals, key=lambda v: v[args.select])
    last = vals[-1]
    done = (log.parent / "DONE").exists()
    rows.append(
        {
            "run": log.parent.name,
            "done": done,
            "best_step": best["step"],
            "last_step": last["step"],
            **{
                k: round(best[k], 3)
                for k in (
                    "score",
                    "val_mse60",
                    "val_mse60_first10",
                    "val_mse240",
                    "val_energy_ratio60",
                    "val_energy_ratio240",
                )
            },
            "sec": last["sec"],
            "peak_mem_gb": last.get("peak_mem_gb"),
        }
    )
hdr = [
    "run",
    "done",
    "best_step",
    "score",
    "val_mse60",
    "val_mse60_first10",
    "val_mse240",
    "val_energy_ratio60",
    "val_energy_ratio240",
    "sec",
]
print(" | ".join(hdr))
for r in rows:
    print(" | ".join(str(r[h]) for h in hdr))
if args.json:
    Path(args.json).write_text(json.dumps(rows, indent=1))
