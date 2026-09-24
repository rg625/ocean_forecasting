"""TRA in one panel: the surrogates forecast well, and that does not make them invertible.

Top row -- FORECAST from the exact state.  No assimilation, no optimisation: each model is
handed u(t_0) and rolled forward.  The U-Net is the most accurate of the four and the KAE
the least.  This row exists so that nothing below it can be dismissed as a badly trained
baseline.

Lower rows -- ASSIMILATION.  u(t_0) is never given; it is recovered from observations that
start delta_f frames later.  The ordering inverts, and it inverts further the longer the
lead.  Same models, same checkpoints, same column order as the row above.

The assimilation example is chosen to be FAVOURABLE TO THE BASELINES (the problem where
the U-Net and FNO do best of those saved), so the comparison cannot be read as a bad draw
for them; the script prints every candidate's errors and where the chosen one sits.

    python -m data_assimilation.tra.fig_tra_panel --deltas 1 8 18 --lead 20
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


def score(d, tag, ex):
    """rel-L2 of every method's analysis on one saved problem."""
    tru = torch.as_tensor(d[f"{tag}__truth"][ex])[None]
    msk = d.get(f"{tag}__mask")
    om = None if msk is None else torch.as_tensor(msk[ex])[None]
    out = {}
    for key, _, _ in ORDER:
        k = f"{tag}__{key}__analysis"
        if k in d.files and ex < d[k].shape[0]:
            out[key] = float(rel_l2(torch.as_tensor(d[k][ex])[None], tru, om)[0])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fields",
        type=Path,
        default=Path("da_results_tra_gallery/G1_delta_f_fields.npz"),
    )
    ap.add_argument(
        "--rollout", type=Path, default=Path("da_results_tra/rollout_from_true_ic.npz")
    )
    ap.add_argument("--out", type=Path, default=Path("figs/tra_panel.png"))
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 8, 18])
    ap.add_argument("--lead", type=int, default=20)
    ap.add_argument("--roll-example", type=int, default=0)
    ap.add_argument(
        "--example",
        default="baselines",
        help="'baselines' picks the saved problem kindest to U-Net and FNO; "
        "an integer forces one",
    )
    ap.add_argument("--channel", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=200)
    ap.add_argument("--data", default="data/acdm/128_tra/gt_interp.nc")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--no-forecast",
        action="store_true",
        help="drop the forecast row; it is its own figure, "
        "data_assimilation.tra.fig_tra_forecast_check",
    )
    ap.add_argument(
        "--gallery",
        action="store_true",
        help="one row per saved problem at a single delta_f, to choose from",
    )
    ap.add_argument(
        "--split-rows",
        type=Path,
        default=None,
        help="also write each row as its own file in this directory, so a "
        "single example can be dropped into the paper unchanged",
    )
    a = ap.parse_args()

    d = np.load(a.fields, allow_pickle=True)
    tags = [f"df{v}" for v in a.deltas]
    n_ex = min(
        d[f"{t}__{k}__analysis"].shape[0]
        for t in tags
        for k, _, _ in ORDER
        if f"{t}__{k}__analysis" in d.files
    )
    print(f"  {n_ex} problem(s) saved for every method")

    if a.gallery:
        assert len(tags) == 1, "--gallery takes one delta_f"
        rows = [(tags[0], a.deltas[0], k) for k in range(n_ex)]
    else:
        if a.example == "baselines":
            cost = []
            for ex in range(n_ex):
                v = [
                    score(d, t, ex).get(m, np.nan)
                    for t in tags
                    for m in ("UNet", "FNO")
                ]
                cost.append(float(np.exp(np.mean(np.log(v)))))
            pick = int(np.argmin(cost))
            for i, c in enumerate(cost):
                print(
                    f"    example {i}: baseline geo-mean {c:.4f}"
                    + ("  <- shown" if i == pick else "")
                )
        else:
            pick = int(a.example)
        rows = [(t, v, pick) for t, v in zip(tags, a.deltas)]

    # the first observation is not stored, but the seed is: rebuild the same problem and
    # read y_1 off it, so the reader sees what the methods were actually handed
    meta = {
        r["tag"]: r
        for r in json.load(open(str(a.fields).replace("_fields.npz", ".json")))["rows"]
    }
    data = PhysicalData(a.data, "tra", a.device)
    obs = {}
    for i, tag in enumerate(tags):
        pr = build_problem(
            data,
            name=tag,
            n_problems=meta[tag]["problem"]["n_problems"],
            offsets=np.asarray(meta[tag]["frames"], dtype=int),
            seed=a.seed0 + i,
        )
        obs[tag] = pr.y[int(np.argmin(pr.offsets))]

    forecast = not (a.no_forecast or a.gallery)
    ncol, nrow = 2 + len(ORDER), len(rows) + (1 if forecast else 0)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(2.62 * ncol, 1.82 * nrow), squeeze=False
    )

    if forecast:
        roll = np.load(a.rollout, allow_pickle=True)
        i = a.lead - 1
        tru = roll["truth"][a.roll_example, i, a.channel]
        lo, hi = np.nanpercentile(tru, [1, 99])
        axes[0, 0].imshow(
            roll["truth"][a.roll_example, 0, a.channel],
            origin="lower",
            cmap="RdBu_r",
            vmin=lo,
            vmax=hi,
            aspect="auto",
        )
        axes[0, 0].set_title("GIVEN\n$u(t_0)$, exact", fontsize=8.5, color="#555555")
        axes[0, 0].set_ylabel(f"forecast\n$+{a.lead}$ frames", fontsize=8.5, color=INK)
        axes[0, 1].imshow(
            tru, origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi, aspect="auto"
        )
        axes[0, 1].set_title(f"TARGET\n$u(t_0{{+}}{a.lead})$", fontsize=8.5, color=INK)
        for c, (key, name, col) in enumerate(ORDER, start=2):
            axes[0, c].imshow(
                roll[f"{key}__roll"][a.roll_example, i, a.channel],
                origin="lower",
                cmap="RdBu_r",
                vmin=lo,
                vmax=hi,
                aspect="auto",
            )
            axes[0, c].set_title(
                f"{name}\n{float(roll[f'{key}__err'][i]):.4f}", fontsize=8.5, color=col
            )

    for r, (tag, df, ex) in enumerate(rows, start=1 if forecast else 0):
        s_ = score(d, tag, ex)
        t = d[f"{tag}__truth"][ex, a.channel]
        msk = d.get(f"{tag}__mask")
        m2 = None if msk is None else msk[ex]
        show = (
            (lambda z: np.where(m2 > 0, z, np.nan)) if m2 is not None else (lambda z: z)
        )
        lo, hi = np.nanpercentile(show(t), [1, 99])
        gap = float(
            rel_l2(
                torch.as_tensor(obs[tag][ex])[None],
                torch.as_tensor(d[f"{tag}__truth"][ex])[None],
                None if m2 is None else torch.as_tensor(m2)[None],
            )[0]
        )
        axes[r, 0].imshow(
            show(obs[tag][ex, a.channel]),
            origin="lower",
            cmap="RdBu_r",
            vmin=lo,
            vmax=hi,
            aspect="auto",
        )
        axes[r, 1].imshow(
            show(t), origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi, aspect="auto"
        )
        axes[r, 0].set_ylabel(
            (f"example {ex}" if a.gallery else f"analysis\n$\\delta_f={df}$"),
            fontsize=8.5,
            color=INK,
        )
        if r == (1 if forecast else 0):
            axes[r, 0].set_title(
                f"GIVEN\n$u(t_0{{+}}{df})$   {gap:.4f}", fontsize=8.5, color="#555555"
            )
            axes[r, 1].set_title("TARGET\n$u(t_0)$", fontsize=8.5, color=INK)
        else:
            axes[r, 0].set_title(f"{gap:.4f}", fontsize=8.5, color="#555555")
        for c, (key, name, col) in enumerate(ORDER, start=2):
            k = f"{tag}__{key}__analysis"
            if k not in d.files or ex >= d[k].shape[0]:
                axes[r, c].axis("off")
                continue
            axes[r, c].imshow(
                show(d[k][ex, a.channel]),
                origin="lower",
                cmap="RdBu_r",
                vmin=lo,
                vmax=hi,
                aspect="auto",
            )
            ttl = (
                f"{name}\n{s_[key]:.4f}"
                if r == (1 if forecast else 0)
                else f"{s_[key]:.4f}"
            )
            axes[r, c].set_title(ttl, fontsize=8.5, color=col)

    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#dddddd")
    fig.tight_layout(h_pad=2.6 if not a.gallery else 1.4)
    if forecast:
        y0, y1 = axes[0, 0].get_position().y0, axes[1, 0].get_position().y1
        fig.add_artist(
            plt.Line2D(
                [0.04, 0.99],
                [(y0 + y1) / 2] * 2,
                color="#c9c9c9",
                lw=1.0,
                transform=fig.transFigure,
                zorder=0,
            )
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)

    if a.split_rows is not None:
        # each row again on its own, with the column headings it needs to stand alone
        a.split_rows.mkdir(parents=True, exist_ok=True)
        for tag, df, ex in rows:
            s_ = score(d, tag, ex)
            t = d[f"{tag}__truth"][ex, a.channel]
            msk = d.get(f"{tag}__mask")
            m2 = None if msk is None else msk[ex]
            sh = (
                (lambda z: np.where(m2 > 0, z, np.nan))
                if m2 is not None
                else (lambda z: z)
            )
            lo, hi = np.nanpercentile(sh(t), [1, 99])
            gap = float(
                rel_l2(
                    torch.as_tensor(obs[tag][ex])[None],
                    torch.as_tensor(d[f"{tag}__truth"][ex])[None],
                    None if m2 is None else torch.as_tensor(m2)[None],
                )[0]
            )
            f1, ax1 = plt.subplots(1, ncol, figsize=(2.62 * ncol, 2.15), squeeze=False)
            ax1 = ax1[0]
            ax1[0].imshow(
                sh(obs[tag][ex, a.channel]),
                origin="lower",
                cmap="RdBu_r",
                vmin=lo,
                vmax=hi,
                aspect="auto",
            )
            ax1[0].set_title(
                f"GIVEN\n$u(t_0{{+}}{df})$   {gap:.4f}", fontsize=8.5, color="#555555"
            )
            ax1[1].imshow(
                sh(t), origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi, aspect="auto"
            )
            ax1[1].set_title("TARGET\n$u(t_0)$", fontsize=8.5, color=INK)
            for c, (key, name, col) in enumerate(ORDER, start=2):
                k = f"{tag}__{key}__analysis"
                if k not in d.files or ex >= d[k].shape[0]:
                    ax1[c].axis("off")
                    continue
                ax1[c].imshow(
                    sh(d[k][ex, a.channel]),
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=lo,
                    vmax=hi,
                    aspect="auto",
                )
                ax1[c].set_title(f"{name}\n{s_[key]:.4f}", fontsize=8.5, color=col)
            for ax in ax1:
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#dddddd")
            f1.tight_layout()
            q = a.split_rows / f"{tag}_ex{ex}.png"
            f1.savefig(q, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(f1)
            print(f"  {q}   KAE {s_.get('KAE', float('nan')):.4f}")
    for tag, df, ex in rows:
        s_ = score(d, tag, ex)
        print(
            f"  delta_f={df:2d} ex{ex}: "
            + "  ".join(f"{n} {s_[k]:.4f}" for k, n, _ in ORDER if k in s_)
        )


if __name__ == "__main__":
    main()
