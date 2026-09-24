"""The TRA delta_f sweep as a figure: all five methods against the lead to the first
observation, with two references.

Reads whatever ``exp_tra --sections delta_f`` has written, so it can be run while the
sweep is still going and redrawn when more points land.  The references are recomputed
here rather than read from the sweep, because the sweep does not record them:

  copy the nearest observation   the analysis you get for free, by taking y_1 as if it
                                 were the state at t_0.  Any method must beat it.
  climatology                    the dataset mean field, i.e. rel-L2 = 1 by construction
                                 in normalised units; on TRA it is measured, since the
                                 error is in physical units over the obstacle-free region.

    python -m data_assimilation.tra.fig_tra_deltaf_fine --json da_results_tra_fine/G1_delta_f.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_assimilation.tra.bridge import rel_l2
from data_assimilation.tra.exp_tra import Bench
from data_assimilation.tra.protocol import build_problem

logger = logging.getLogger("fig_tra_deltaf")

# ACDM-ncn is left out: trained with clean conditioning, its score is invalid for
# assimilation and it sits at 4.1 against climatology 1.0 at every delta_f, which
# compresses the axis without telling the reader anything a sentence cannot.
# --with-ncn puts it back.
STYLE = {
    "KAE": ("#1b7837", "KAE"),
    "UNet": ("#e41a1c", "U-Net 4D-Var"),
    "FNO": ("#ff7f00", "FNO 4D-Var"),
    "ACDM": ("#6a3d9a", "ACDM $+$ SDA"),
}
NCN = {"ACDM-ncn": ("#4393c3", "ACDM-ncn")}


def references(b, rows, n_problems, seed0):
    """copy-the-nearest-observation and climatology, on the sweep's own problems.

    Climatology is the dataset mean field, which is what a zero control decodes to, so it
    is taken from the adapter's own normalisation rather than averaged over the handful of
    problems in the batch.
    """
    data = b.data(b.a.regime)
    ad = b.model("KAE")  # only its normalisation is used, not the model
    copy, clim = [], []
    for i, r in enumerate(rows):
        fr = np.asarray(r["frames"], dtype=int)
        prob = build_problem(
            data, name=f"ref{i}", n_problems=n_problems, offsets=fr, seed=seed0 + i
        )
        sim = torch.as_tensor(prob.sim, device=b.dev)
        t0 = torch.as_tensor(prob.t0, device=b.dev)
        truth = data.frames(sim, t0)
        om = data.mask_for(sim)
        j = int(np.argmin(prob.offsets))
        y1 = torch.as_tensor(prob.y[j], device=b.dev).float()
        copy.append(float(rel_l2(y1, truth, om).mean()))
        mean_field = ad.to_physical(torch.zeros_like(ad.to_model(truth)))
        clim.append(float(rel_l2(mean_field, truth, om).mean()))
        logger.info(
            f"  df={r['value']:2d}  copy {copy[-1]:.4f}  climatology {clim[-1]:.4f}"
        )
    return np.array(copy), np.array(clim)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json", type=Path, default=Path("da_results_tra_fine/G1_delta_f.json")
    )
    ap.add_argument("--out", type=Path, default=Path("figs/tra_deltaf_fine.png"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--kae-run",
        default=("model_outputs_tra/continous_linear_128/run-20260821_031121"),
    )
    ap.add_argument("--regime", default="test")
    ap.add_argument("--n-problems", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=200)
    ap.add_argument(
        "--with-ncn",
        action="store_true",
        help="include ACDM-ncn, which is off-scale at every point",
    )
    ap.add_argument(
        "--no-refs",
        action="store_true",
        help="skip the two reference curves (no GPU needed)",
    )
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=2)
    a = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    a.kae_run = Path(a.kae_run)

    d = json.loads(a.json.read_text())
    rows = sorted(d["rows"], key=lambda r: r["value"])
    x = [r["value"] for r in rows]
    logger.info(f"{len(rows)} points, delta_f {x[0]}..{x[-1]}")

    style = {**STYLE, **NCN} if a.with_ncn else STYLE
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    for m, (col, lab) in style.items():
        pts = [r for r in rows if m in r and r[m].get("mean") is not None]
        if not pts:
            continue
        ax.errorbar(
            [r["value"] for r in pts],
            [r[m]["mean"] for r in pts],
            yerr=[r[m].get("sem") or 0.0 for r in pts],
            color=col,
            label=lab,
            lw=1.9,
            marker="o",
            ms=4,
            capsize=2.5,
        )
    if not a.no_refs:
        b = Bench(a)
        copy, clim = references(b, rows, a.n_problems, a.seed0)
        ax.plot(
            x, copy, color="0.45", ls="-.", lw=1.5, label="copy the nearest observation"
        )
        ax.plot(x, clim, color="crimson", ls=":", lw=1.5, label="climatology")
    ax.set(
        yscale="log",
        xlabel=r"$\delta_f$  (frames to the first observation)",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        xticks=x[::2],
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    logger.info(f"wrote {a.out}")


if __name__ == "__main__":
    main()
