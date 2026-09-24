"""TRA qualitative fields: one forecast row and two assimilation examples.

Row 1 -- FORECAST from the exact state.  No assimilation and no optimisation: every model
is handed u(t_0) and rolled forward, so this row says how good the surrogates are as
forward models.  Nothing below it can then be dismissed as a badly trained baseline.

Rows 2 and 3 -- ASSIMILATION.  u(t_0) is never given; it is recovered from five
observations that start delta_f frames later.  Both rows are drawn from the same saved
batch (``--fields``), so they differ only in which problem of that batch is shown.

Columns are the same in every row: the field the method was GIVEN, the TARGET it had to
produce, then the four methods in a fixed order.  Each panel carries its own rel-L2 above
it; the fields are the evidence, so the error maps that used to sit under every row have
been dropped rather than shown twice.

    python -m data_assimilation.tra.fig_tra_fields_qual --examples 0 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_assimilation.tra.bridge import PhysicalData, rel_l2
from data_assimilation.tra.protocol import build_problem

# ACDM-ncn is excluded: trained with clean conditioning, its score is invalid here
ORDER = [
    ("KAE", "KAE", "#1b7837"),
    ("UNet", "U-Net", "#e41a1c"),
    ("FNO", "FNO", "#ff7f00"),
    ("ACDM", "ACDM $+$ SDA", "#6a3d9a"),
]
INK = "#2b2b2b"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fields",
        type=Path,
        default=Path("da_results_tra_gallery10/G1_delta_f_fields.npz"),
    )
    ap.add_argument(
        "--rollout", type=Path, default=Path("da_results_tra/rollout_from_true_ic.npz")
    )
    ap.add_argument("--out", type=Path, default=Path("figs/tra_fields_qual.png"))
    ap.add_argument(
        "--examples",
        type=int,
        nargs="+",
        default=[0, 8],
        help="which saved problems to show, one assimilation row each",
    )
    ap.add_argument("--lead", type=int, default=20, help="forecast lead, in frames")
    ap.add_argument("--roll-example", type=int, default=0)
    ap.add_argument("--channel", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=200)
    ap.add_argument("--data", default="data/acdm/128_tra/gt_interp.nc")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    d = np.load(a.fields, allow_pickle=True)
    meta = json.load(open(str(a.fields).replace("_fields.npz", ".json")))
    row = meta["rows"][0]
    tag, df = row["tag"], int(row["value"])
    n_ex = min(
        d[f"{tag}__{k}__analysis"].shape[0]
        for k, _, _ in ORDER
        if f"{tag}__{k}__analysis" in d.files
    )
    for ex in a.examples:
        assert ex < n_ex, f"example {ex} not saved ({n_ex} problems in {a.fields})"

    # the first observation is not stored, but the seed is: rebuild the same problem and
    # read y_1 off it, so the reader sees what the methods were actually handed
    data = PhysicalData(a.data, "tra", a.device)
    pr = build_problem(
        data,
        name=tag,
        n_problems=row["problem"]["n_problems"],
        offsets=np.asarray(row["frames"], dtype=int),
        seed=a.seed0,
    )
    obs = pr.y[int(np.argmin(pr.offsets))]

    ncol = 2 + len(ORDER)
    nblock = 1 + len(a.examples)
    fig, axes = plt.subplots(
        nblock,
        ncol,
        squeeze=False,
        layout="constrained",
        figsize=(2.62 * ncol, 2.16 * nblock),
    )
    # the only gap that matters now is the one between rows, which has to clear the next
    # row's column titles
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.01, hspace=0.045, wspace=0.01)

    def draw(b, given, target, preds, errs_of, left, titles, gap):
        """One field row."""
        fr = axes[b]
        lo, hi = np.nanpercentile(target, [1, 99])
        fr[0].imshow(
            given, origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi, aspect="auto"
        )
        fr[1].imshow(
            target, origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi, aspect="auto"
        )
        fr[0].set_ylabel(left, fontsize=9, color=INK)
        if titles:
            fr[0].set_title(f"{titles[0]}   {gap:.2f}", fontsize=8.5, color="#555555")
            fr[1].set_title(titles[1], fontsize=8.5, color=INK)
        else:
            fr[0].set_title(f"{gap:.2f}", fontsize=8.5, color="#555555")
        for c, (k, name, col) in enumerate(ORDER, start=2):
            fr[c].imshow(
                preds[k], origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi, aspect="auto"
            )
            fr[c].set_title(
                (
                    f"{name}\n{errs_of[k + '__rel']:.2f}"
                    if titles
                    else f"{errs_of[k + '__rel']:.2f}"
                ),
                fontsize=8.5,
                color=col,
            )

    # ---- block 0: forecast from the exact state -------------------------------------
    roll = np.load(a.rollout, allow_pickle=True)
    i, rex = a.lead - 1, a.roll_example
    tru_f = roll["truth"][rex, i, a.channel]
    e_f = {}
    for k, _, _ in ORDER:
        e_f[k] = roll[f"{k}__roll"][rex, i, a.channel] - tru_f
        e_f[k + "__rel"] = float(
            rel_l2(
                torch.as_tensor(roll[f"{k}__roll"][rex, i])[None],
                torch.as_tensor(roll["truth"][rex, i])[None],
            )[0]
        )
    draw(
        0,
        roll["truth"][rex, 0, a.channel],
        tru_f,
        {k: roll[f"{k}__roll"][rex, i, a.channel] for k, _, _ in ORDER},
        e_f,
        f"forecast\nfrom exact $u(t_0)$\n$+{a.lead}$ frames",
        ("GIVEN\n$u(t_0)$, exact", f"TARGET\n$u(t_0{{+}}{a.lead})$"),
        float(
            rel_l2(
                torch.as_tensor(roll["truth"][rex, 0])[None],
                torch.as_tensor(roll["truth"][rex, i])[None],
            )[0]
        ),
    )

    # ---- blocks 1..: assimilation, one saved problem each ---------------------------
    for b, ex in enumerate(a.examples, start=1):
        t_full = d[f"{tag}__truth"][ex]
        m2 = d[f"{tag}__mask"][ex] if f"{tag}__mask" in d.files else None
        show = (
            (lambda z: np.where(m2 > 0, z, np.nan)) if m2 is not None else (lambda z: z)
        )
        mt = None if m2 is None else torch.as_tensor(m2)[None]
        e_a = {}
        for k, _, _ in ORDER:
            e_a[k] = show(d[f"{tag}__{k}__analysis"][ex, a.channel] - t_full[a.channel])
            e_a[k + "__rel"] = float(
                rel_l2(
                    torch.as_tensor(d[f"{tag}__{k}__analysis"][ex])[None],
                    torch.as_tensor(t_full)[None],
                    mt,
                )[0]
            )
        draw(
            b,
            show(obs[ex, a.channel]),
            show(t_full[a.channel]),
            {k: show(d[f"{tag}__{k}__analysis"][ex, a.channel]) for k, _, _ in ORDER},
            e_a,
            f"analysis at $t_0$\n$\\delta_f={df}$\nexample {ex}",
            ("GIVEN\n$u(t_0{+}%d)$" % df, "TARGET\n$u(t_0)$") if b == 1 else None,
            float(
                rel_l2(
                    torch.as_tensor(obs[ex])[None], torch.as_tensor(t_full)[None], mt
                )[0]
            ),
        )

    for r in axes:
        for ax in r:
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#dddddd")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
