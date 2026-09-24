# ruff: noqa: E741
# mypy: disable-error-code="var-annotated"
"""The four KS conditional sweeps as curves, at both pinned leads.

The per-point tables carry the numbers; these panels carry the shape, which is what the
argument actually rests on -- where a method is flat, where it turns over, and where two
methods cross.  One panel per axis, delta_f = 1 drawn faint and delta_f = 9 solid, so the
effect of the lead is visible within each panel instead of across two tables.

Read from the same files as the tables, including the KAE replay, so the curves and the
tables cannot drift apart.

    python -m data_assimilation.ks.fig_ks_sweeps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = [
    ("KAE", "KAE", "#1b7837"),
    ("SDA", "SDA", "#6a3d9a"),
    ("U-Net", "U-Net", "#e41a1c"),
]
INK = "#2b2b2b"

# section -> (file stem, tag prefix, x label, how to read x, optional row filter)
SPEC = {
    "C2_delta_l": (
        "C2_delta_l",
        "C2_",
        r"recovery horizon $\delta_l$  (t.u.)",
        lambda r, l: r["delta_l"],
        None,
    ),
    "C3_n_obs": (
        "C3_n_obs",
        "C3_",
        r"observations $N$ (realised)",
        lambda r, l: r["N"],
        None,
    ),
    "noise_law": (
        "noise_law",
        "NL_",
        r"observation noise $\sigma$",
        lambda r, l: float(l.split("_")[1]),
        lambda l: l.startswith("gaussian"),
    ),
    "joint_sparsity": (
        "joint_sparsity",
        "JS_",
        r"sensor coverage",
        lambda r, l: float(l[1:].split("_")[0]),
        lambda l: l.endswith("_n5"),
    ),
}
TITLE = {
    "C2_delta_l": "recovery horizon",
    "C3_n_obs": "number of observations",
    "noise_law": "observation noise",
    "joint_sparsity": "sensor coverage",
}


def series(section, df):
    """{method: (x, mean, sem)} for one axis at one pinned lead."""
    stem, pre, _, xof, keep = SPEC[section]
    dirn = "da_results_geometry" if df == 1 else "da_results_geometry_df9"
    rows = json.load(open(f"{dirn}/{stem}.json"))["rows"]
    # the coverage axis was refined between 1.0 and 0.1, where KAE and SDA cross; those
    # points were run separately with the same protocol and are merged in here
    extra = Path(f"{dirn}_fine/{stem}.json")
    if df == 9 and extra.is_file():
        have = {r["tag"] for r in rows}
        rows = rows + [
            r for r in json.load(open(extra))["rows"] if r["tag"] not in have
        ]
    tuned = json.load(open("da_results_geometry_df9/kae512_tuned.json"))["sweeps"]
    kae = {
        r["label"].replace("=", ""): (r["mean"], r["sem"])
        for r in tuned
        if r["section"] == section and int(r["delta_f"]) == df
    }
    out = {}
    for r in rows:
        lab = r["tag"][len(pre) :]
        if "anchor" in lab or (keep and not keep(lab)):
            continue
        x = xof(r, lab)
        for src, dst in (("SDA", "SDA"), ("UNet", "U-Net")):
            v = r.get(src)
            if not isinstance(v, dict) or v.get("mean") is None:
                continue
            m = v["mean"]
            # a partly diverged sampler returns a finite mean with an error bar its own
            # size; that is not a measurement and is dropped, as in the tables
            if m != m or (v.get("sem") and m > 0 and v["sem"] / m > 0.5):
                continue
            out.setdefault(dst, []).append((x, m, v.get("sem") or 0.0))
        if lab in kae:
            out.setdefault("KAE", []).append((x, *kae[lab]))
        elif (
            isinstance(r.get("KAE-expm"), dict)
            and r["KAE-expm"].get("mean") is not None
        ):
            # the refinement points were run in one pass with the tuned settings already
            # applied, so their KAE score sits in the row rather than in the tuning file
            out.setdefault("KAE", []).append(
                (x, r["KAE-expm"]["mean"], r["KAE-expm"].get("sem") or 0.0)
            )
    return {k: tuple(np.array(z) for z in zip(*sorted(v))) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("figs/ks_sweeps.png"))
    ap.add_argument(
        "--only-df9", action="store_true", help="drop the faint delta_f = 1 curves"
    )
    a = ap.parse_args()

    secs = list(SPEC)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))
    for ax, sec in zip(axes.ravel(), secs):
        for df, alpha, lw, ls in ((1, 0.32, 1.4, (0, (3, 2))), (9, 1.0, 2.0, "-")):
            if a.only_df9 and df == 1:
                continue
            for key, name, col in STYLE:
                s = series(sec, df).get(key)
                if s is None:
                    continue
                x, m, e = s
                ax.errorbar(
                    x,
                    m,
                    yerr=e,
                    color=col,
                    lw=lw,
                    ls=ls,
                    alpha=alpha,
                    marker="o" if df == 9 else None,
                    ms=4,
                    capsize=2.5,
                    label=name if df == 9 else None,
                    zorder=3 if df == 9 else 2,
                )
        ax.set(yscale="log", xlabel=SPEC[sec][2], title=TITLE[sec])
        if sec in ("C2_delta_l", "C3_n_obs"):
            ax.set_xscale("log")
        if sec == "joint_sparsity":
            ax.invert_xaxis()
        ax.grid(alpha=0.2, which="both")
        ax.tick_params(labelsize=9, colors="#666666")
        ax.title.set(fontsize=10, color=INK)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#d0d0d0")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"analysis rel-$L_2$ at $t_0$", fontsize=10, color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    extra = (
        []
        if a.only_df9
        else [
            plt.Line2D(
                [],
                [],
                color="#888888",
                lw=1.4,
                ls=(0, (3, 2)),
                alpha=0.6,
                label=r"$\delta_f=1$ (faint)",
            )
        ]
    )
    fig.legend(
        h + extra,
        l + [e.get_label() for e in extra],
        ncol=4,
        fontsize=10,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.045),
    )
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)
    for sec in secs:
        s9 = series(sec, 9)
        print(
            f"  {sec:16s} df=9 "
            + "  ".join(f"{k} {v[1][0]:.4f}->{v[1][-1]:.4f}" for k, v in s9.items())
        )


if __name__ == "__main__":
    main()
