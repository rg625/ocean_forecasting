# ruff: noqa: E731, F841
# mypy: disable-error-code="index, no-any-return, operator, var-annotated"
"""Figures and tables for the transonic DA campaign."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TRA = Path("da_results_tra")

STYLE = {
    "KAE": dict(color="#1b7837", marker="o", label="Continuous KAE — 4D-Var on $z_0$"),
    "UNet": dict(color="#d6604d", marker="^", label="U-Net — 4D-Var on $x_0$"),
    "FNO": dict(color="#4393c3", marker="s", label="FNO — 4D-Var on $x_0$"),
    "ACDM": dict(color="#762a83", marker="D", label="ACDM — blanket score DA"),
    "ACDM-ncn": dict(color="#b8a2c8", marker="v", label="ACDM-ncn — same path (OOD)"),
}
ORDER = ["KAE", "UNet", "FNO", "ACDM", "ACDM-ncn"]


class NotRunYet(Exception):
    pass


def require(path, how: str):
    """Fail with the command that produces the missing file, not a bare IOError."""
    if not Path(path).is_file():
        raise NotRunYet(f"{path} is not on disk yet.\n    produce it with:  {how}")
    return Path(path)


def _load(out_dir, stem):
    j = Path(out_dir) / f"{stem}.json"
    if not j.is_file():
        raise NotRunYet(
            f"{j} not on disk yet.\n    produce it with: "
            f"python -m data_assimilation.tra.exp_tra --sections headline diffusion"
        )
    npz = Path(out_dir) / f"{stem}.npz"
    return json.loads(j.read_text()), (np.load(npz) if npz.is_file() else None)


def summary(out_dir=TRA) -> dict:
    """Merge the 4D-Var and diffusion halves of the headline into one dict."""
    out = {}
    for stem in ["A_headline", "A_headline_diffusion"]:
        try:
            j, _ = _load(out_dir, stem)
        except NotRunYet:
            continue
        out.update({k: v for k, v in j.items() if k != "problem"})
    if not out:
        raise NotRunYet("no headline results on disk yet")
    return out


def fig_headline(out_dir=TRA, ax=None):
    """Analysis error at $t_0$, every method, same problem."""
    s = summary(out_dir)
    ms = [m for m in ORDER if m in s]
    ax = ax or plt.subplots(figsize=(7.4, 4.2))[1]
    x = np.arange(len(ms))
    mu = [s[m]["mean"] for m in ms]
    se = [s[m]["sem"] for m in ms]
    ax.bar(x, mu, yerr=se, capsize=4, color=[STYLE[m]["color"] for m in ms], alpha=0.85)
    for i, m in enumerate(ms):
        ax.text(i, mu[i] + se[i], f"{mu[i]:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m for m in ms], rotation=12, fontsize=8)
    ax.set(
        yscale="log",
        ylabel=r"analysis rel-$L_2$ at $t_0$ (physical units)",
        title=(
            "Recovery of the unobserved state, transonic flow\n"
            "same problem for every method; obstacle interior excluded"
        ),
    )
    ax.grid(alpha=0.3, axis="y", which="both")
    if "ACDM-ncn" in ms:
        i = ms.index("ACDM-ncn")
        ax.text(
            i,
            ax.get_ylim()[0],
            " out of\n distribution",
            fontsize=6.5,
            color="crimson",
            ha="center",
            va="bottom",
        )
    return ax


def fig_convergence(out_dir=TRA, ax=None):
    """Do the 4D-Var methods actually converge?"""
    _, d = _load(out_dir, "A_headline")
    ax = ax or plt.subplots(figsize=(7.0, 4.2))[1]
    for m in ORDER:
        k = f"{m}__hist_rel_t0"
        if d is None or k not in d.files:
            continue
        ax.plot(
            d[f"{m}__hist_iter"],
            d[k],
            lw=1.8,
            color=STYLE[m]["color"],
            label=STYLE[m]["label"],
        )
    ax.set(
        xlabel="4D-Var iteration",
        yscale="log",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title="Convergence from an uninformed (climatology) start",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    return ax


def table(out_dir=TRA) -> str:
    s = summary(out_dir)
    j, _ = _load(out_dir, "A_headline")
    L = []
    if "problem" in j:
        L += [f"problem: {json.dumps(j['problem'])}", ""]
    L += [
        f"{'method':10s} {'rel-L2 at t0':>18s} {'n':>4s} {'setting':>28s} {'wall':>8s}",
        "-" * 74,
    ]
    for m in ORDER:
        if m not in s:
            continue
        v = s[m]
        setting = (
            f"lr={v['lr']}, {v['iters']} iters"
            if "lr" in v
            else f"{v.get('n_samples','?')} draws, blanket W={v.get('blanket_window','?')}"
        )
        L.append(
            f"{m:10s} {v['mean']:11.4f}±{v['sem']:.4f} {v['n']:4d} "
            f"{setting:>28s} {v['wall_s']:7.0f}s"
        )
    notes = [f"  {m}: {s[m]['note']}" for m in ORDER if m in s and s[m].get("note")]
    if notes:
        L += ["", "notes:"] + notes
    return "\n".join(L)


# ---------------------------------------------------------------------------
SWEEPS = {
    "G1_delta_f": (r"$\delta_f$  (frames to the first observation)", "log"),
    "G4_delta_l": (r"$\delta_l$  (recovery horizon, frames)", "log"),
    "G5_n_obs": (r"$N$  (number of observation times)", "log"),
    "G6_single_obs": (r"$\tau$  (a SINGLE observation, frames)", "log"),
    "G3_sparsity": ("fraction of grid points observed", "log"),
    "G2_noise": (r"observation noise $\sigma_y$", "symlog"),
}


def _num(entry, key="mean") -> float:
    """A sweep cell as a float, with null (a diverged solve) mapped to NaN."""
    if not isinstance(entry, dict):
        return float("nan")
    v = entry.get(key)
    return float("nan") if v is None else float(v)


def pinned_delta_f(meta) -> int:
    """delta_f a sweep held fixed; files written before it was recorded pinned 1."""
    return int(meta.get("pinned_delta_f", 1))


def load_sweep(out_dir, stem):
    f = Path(out_dir) / f"{stem}.json"
    if not f.is_file():
        sec = {
            "G1_delta_f": "delta_f",
            "G2_noise": "noise",
            "G3_sparsity": "sparsity",
            "G4_delta_l": "delta_l",
            "G5_n_obs": "n_obs",
            "G6_single_obs": "single_obs",
        }.get(stem, "?")
        raise NotRunYet(
            f"{f} not on disk.\n    produce it with: "
            f"python -m data_assimilation.tra.exp_tra --sections {sec}"
        )
    return json.loads(f.read_text())


def fig_forecast(out_dir=TRA, ax=None):
    """Free-running forecast from the EXACT state: model error with no assimilation.

    This is the floor no analysis can beat, and it separates 'the model cannot predict
    this' from 'the assimilation cannot find it'.
    """
    d = np.load(Path(out_dir) / "G7_forecast.npz")
    fr = d["frames"]
    ax = ax or plt.subplots(figsize=(6.6, 4.2))[1]
    for m in ORDER:
        k = f"{m}__rel_mean"
        if k not in d.files:
            continue
        st = {kk: v for kk, v in STYLE[m].items() if kk != "marker"}
        ax.plot(fr, d[k], lw=1.9, **st)
        se = d.get(f"{m}__rel_sem")
        if se is not None:
            ax.fill_between(
                fr, d[k] - se, d[k] + se, color=STYLE[m]["color"], alpha=0.15
            )
    ax.plot(fr, d["persistence"], color="0.45", ls="-.", lw=1.5, label="persistence")
    ax.plot(
        fr, d["saturation"], color="crimson", ls=":", lw=1.8, label="no-skill level"
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="forecast lead (frames)",
        ylabel=r"rel-$L_2$",
        title="Free-running forecast from the EXACT state\n(model error alone — no "
        "assimilation)",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    return ax


def fig_cost(out_dir=TRA, ax=None, per_iteration: bool = False):
    """Cost against the assimilation horizon, timed one method at a time.

    By default this plots SECONDS FOR ONE COMPLETE SOLVE, which is the only cost defined
    for all five methods and the one a user actually pays.  ``per_iteration=True`` plots
    milliseconds per 4D-Var iteration instead; the samplers have no iteration to price and
    are simply absent from that version.
    """
    d = json.loads(
        require(
            Path(out_dir) / "G8_cost.json",
            "python -m data_assimilation.tra.exp_tra --sections cost",
        ).read_text()
    )
    rows = d["rows"]
    xs = [r["horizon_frames"] for r in rows]
    key = (lambda m: m) if per_iteration else (lambda m: f"{m}__solve_s")
    ax = ax or plt.subplots(figsize=(6.4, 4.2))[1]
    for m in ORDER:
        k = key(m)
        if k not in rows[0] or any(r.get(k) is None for r in rows):
            continue
        st = {kk: v for kk, v in STYLE[m].items() if kk != "marker"}
        ax.plot(
            xs, [r[k] for r in rows], lw=1.9, marker=STYLE[m]["marker"], ms=4.5, **st
        )
    it = d["meta"].get("solve_iters")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="assimilation horizon (frames)",
        ylabel=(
            "ms per 4D-Var iteration"
            if per_iteration
            else "seconds for one complete assimilation"
        ),
        title=(
            "Cost against horizon, one method at a time\n"
            + (
                "(the samplers have no iteration to price)"
                if per_iteration
                else f"4D-Var at {it} iterations; samplers at 1 draw"
            )
        ),
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    return ax


def fig_sweep(out_dir=TRA, stem="G1_delta_f", ax=None):
    """Analysis error against the swept quantity, every method."""
    d = load_sweep(out_dir, stem)
    # points are stored in RUN order (extra levels get appended so the seeds of the
    # original points do not move), so sort before drawing or the line doubles back
    rows = sorted(d["rows"], key=lambda r: r["value"])
    xlabel, xscale = SWEEPS[stem]
    xs = np.array([r["value"] for r in rows], dtype=float)
    ax = ax or plt.subplots(figsize=(6.4, 4.2))[1]
    for m in ORDER:
        if not any(m in r for r in rows):
            continue
        # a diverged sampler is written as null, not NaN, so that it cannot be mistaken
        # for a measurement; it has to become NaN before matplotlib sees it
        mu = np.array([_num(r.get(m), "mean") for r in rows], dtype=float)
        se = np.array([_num(r.get(m), "sem") for r in rows], dtype=float)
        if not np.isfinite(mu).any():
            continue
        ax.errorbar(xs, mu, yerr=se, capsize=2.5, lw=1.8, ms=5, **STYLE[m])
        nd = sum(
            int(r[m].get("n_diverged") or 0) for r in rows if isinstance(r.get(m), dict)
        )
        if nd:
            ax.plot([], [], " ", label=f"({m}: {nd} diverged solves omitted)")
    if xscale == "symlog":
        ax.set_xscale("symlog", linthresh=1e-2)
    else:
        ax.set_xscale(xscale)
    ax.set(
        yscale="log",
        xlabel=xlabel,
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title=(
            f"{stem}  —  {d['meta']['varies']}\n"
            f"{d['meta']['iters']} iterations, $n$={d['meta']['n_problems']}, "
            + (
                ""
                if stem in ("G1_delta_f", "G6_single_obs")
                else f"$\\delta_f$={pinned_delta_f(d['meta'])}, "
            )
            + "everything else held fixed"
        ),
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    return ax


def fig_sweep_fields(
    out_dir=TRA,
    stem="G1_delta_f",
    channel: int = 3,
    example: int = 0,
    axes=None,
    tags=None,
    suptitle: bool = True,
):
    """The recovered FIELD at $t_0$ at each sweep point: truth, then each method.

    One row per sweep point, one column per method, with the truth first.  The obstacle
    interior is masked out because no model is scored there.

    ``channel`` defaults to density (index 3 in the .nc field order
    ``[v_x, v_y, p, rho]``), which shows the von Karman wake far more clearly than the
    streamwise velocity.  Fields are drawn in the .nc orientation (64 x 128, flow left to
    right); the transpose into turbpred's [128, 64] layout happens inside the adapter and
    is undone by ``to_physical`` before anything is stored or plotted.
    """
    npz = Path(out_dir) / f"{stem}_fields.npz"
    if not npz.is_file():
        raise NotRunYet(
            f"{npz} not on disk (rerun the sweep; it saves fields as it goes)"
        )
    d = np.load(npz, allow_pickle=True)
    rows = load_sweep(out_dir, stem)["rows"]
    have = [r["tag"] for r in rows if f"{r['tag']}__truth" in d.files]
    if tags is not None:
        missing = [t for t in tags if t not in have]
        assert not missing, f"no fields saved for {missing}; have {have}"
        rows = [r for r in rows if r["tag"] in tags]
    tags = have if tags is None else list(tags)
    ms = [m for m in ORDER if f"{tags[0]}__{m}__analysis" in d.files]
    ncol = 1 + len(ms)
    if axes is None:
        _, axes = plt.subplots(
            len(tags), ncol, figsize=(3.1 * ncol, 2.0 * len(tags)), squeeze=False
        )
    axes = np.atleast_2d(axes)

    for i, (tag, r) in enumerate(zip(tags, rows)):
        tru = d[f"{tag}__truth"][example, channel]
        msk = d.get(f"{tag}__mask")
        m2 = None if msk is None else msk[example]
        show = lambda a: np.where(m2 > 0, a, np.nan) if m2 is not None else a
        # density is not zero-centred, so a symmetric scale would wash the wake out
        lo, hi = np.nanpercentile(show(tru), [1, 99])
        axes[i, 0].imshow(
            show(tru), origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi, aspect="auto"
        )
        axes[i, 0].set_ylabel(f"{r['value']:g}", fontsize=8)
        if i == 0:
            axes[i, 0].set_title("TRUTH", fontsize=8.5)
        for c, m in enumerate(ms, start=1):
            a = d[f"{tag}__{m}__analysis"][example, channel]
            axes[i, c].imshow(
                show(a), origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi, aspect="auto"
            )
            if i == 0:
                axes[i, c].set_title(m, fontsize=8.5, color=STYLE[m]["color"])
            axes[i, c].set_title(
                (m if i == 0 else "") + f"\n{r[m]['mean']:.4f}" if m in r else m,
                fontsize=7.5,
                color=STYLE[m]["color"],
            )
        for a in axes[i]:
            a.set_xticks([])
            a.set_yticks([])
    if suptitle:
        axes[0, 0].figure.suptitle(
            f"{stem}: recovered $u(t_0)$, channel {channel} — rows are "
            f"{SWEEPS[stem][0]}",
            fontsize=9,
        )
    return axes


# ---------------------------------------------------------------------------
def fig_rollout_fields(
    out_dir=TRA,
    npz="rollout_from_true_ic.npz",
    channel: int = 3,
    example: int = 0,
    frames=(0, 6, 12, 18, 24),
    axes=None,
):
    """Free-running rollout from the TRUE initial condition, as fields.

    No assimilation anywhere: each model is handed the exact state and asked to predict.
    This separates "the model cannot represent this flow" from "the assimilation cannot
    find the state" -- and the two answers turn out to be opposite.
    """
    d = np.load(Path(out_dir) / npz, allow_pickle=True)
    tru = d["truth"][example, :, channel]
    T = tru.shape[0]
    cols = [c for c in frames if c < T]
    ms = [m for m in ORDER if f"{m}__roll" in d.files]
    msk = d["mask"][example] if "mask" in d.files else None
    show = (
        (lambda f: np.where(msk > 0, f, np.nan)) if msk is not None else (lambda f: f)
    )
    lo, hi = np.nanpercentile(show(tru), [1, 99])
    rows = ["TRUTH"] + ms
    if axes is None:
        _, axes = plt.subplots(
            len(rows),
            len(cols),
            figsize=(2.5 * len(cols), 1.9 * len(rows)),
            squeeze=False,
        )
    axes = np.atleast_2d(axes)
    for i, r in enumerate(rows):
        f = tru if r == "TRUTH" else d[f"{r}__roll"][example, :, channel]
        for j, c in enumerate(cols):
            axes[i, j].imshow(
                show(f[c]),
                origin="lower",
                cmap="RdBu_r",
                vmin=lo,
                vmax=hi,
                aspect="auto",
            )
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(f"frame {c}", fontsize=9)
        lab = r if r == "TRUTH" else f"{r}\n{float(d[f'{r}__err'][-1]):.4f}"
        axes[i, 0].set_ylabel(lab, fontsize=7.5)
    return axes


def _perturbed_file(out_dir, kind, where, frames="window"):
    """The perturbation file.

    ``frames="window"`` is the corrected run, in which every frame of the conditioning
    window is perturbed.  ``frames="t0"`` is the earlier one that perturbed only t_0 and so
    left the k=2 models (KAE, ACDM, ACDM-ncn) with half their input clean while U-Net and
    FNO, which condition on a single frame, had all of theirs perturbed.  It is kept only
    so the two can be compared; nothing in the deck should read it as a result.
    """
    tag = "" if frames == "window" else "_t0only"
    return Path(out_dir) / f"perturbed_{kind}_{where}{tag}.npz"


def fig_perturbed(
    out_dir=TRA,
    kind="white",
    where="state",
    channel: int = 3,
    example: int = 0,
    frames=(0, 8, 24),
    amps=(0.0, 0.05, 0.2),
    axes=None,
    perturb_frames="window",
):
    """The same rollout, from a PERTURBED initial condition.

    **Frame 0 is the initial condition each model was HANDED, never a model output.**
    That matters because the two are not the same object for every method.  For U-Net, FNO
    and the samplers the control IS the input frame, so the two coincide.  The KAE's own
    frame 0 would be ``decode(encode(x0+noise))``: its 128-d latent is a 256x compression
    of a 32,768-d field and cannot represent grid-scale white noise, so it projects the
    perturbation out before any dynamics run and its reconstruction looks clean.  Plotting
    that reconstruction in the same column as everybody else's raw input made the KAE look
    robust when it had simply never received the perturbation.  So the frame-0 column shows
    the input for every row, and what each model DID with it is read from frame 1 onward.

    The size of that projection effect is not lost -- it is on its own slide, and
    ``fig_perturbed_curves`` reports each method's frame-0 excess in its legend.

    ``where="control"`` loads the like-for-like variant, in which each method's OWN control
    is perturbed by the same relative amount.  There the KAE is handed a CLEAN field and
    its latent is perturbed instead, so its frame-0 panel shows the clean field it actually
    received and is labelled as such.
    """
    f_ = _perturbed_file(out_dir, kind, where, perturb_frames)
    d = np.load(
        require(
            f_,
            f"python -m data_assimilation.tra.perturb_demo --kind {kind} "
            f"--where {where} --frames 25",
        ),
        allow_pickle=True,
    )
    tru = d["truth"][example, :, channel]
    ms = [m for m in ORDER if any(f"{m}__amp" in k for k in d.files)]
    msk = d["mask"][example] if "mask" in d.files else None
    show = (
        (lambda f: np.where(msk > 0, f, np.nan)) if msk is not None else (lambda f: f)
    )
    lo, hi = np.nanpercentile(show(tru), [1, 99])
    mskb = d["mask"] if "mask" in d.files else None
    nB = d["truth"].shape[0]
    if not any(k.startswith("input__amp") for k in d.files):
        raise NotRunYet(
            f"{f_} predates the stored perturbed input, so frame 0 cannot be shown as "
            f"the input.\n    produce it with:  python -m data_assimilation.tra.perturb_demo "
            f"--kind {kind} --where {where} --frames 25"
        )

    def err(arr_b, k):
        """Batch-mean rel-L2 at frame k, over ALL channels, obstacle excluded.

        The panels show one example and one channel; the numbers are the campaign's
        rel_l2 over the whole batch and every channel, so they agree with the slides and
        with every other figure rather than quietly reporting a density-only value.
        """
        t = d["truth"][:, k]  # [B, C, H, W]
        w = np.ones_like(t) if mskb is None else mskb[:, None]
        return float(
            np.sqrt(
                (((arr_b - t) * w) ** 2).sum((1, 2, 3))
                / np.maximum(((t * w) ** 2).sum((1, 2, 3)), 1e-12)
            ).mean()
        )

    def handed(m, a):
        """The batch of frames the model `m` was actually given at t_0, at amplitude `a`.

        Everyone gets the perturbed field, except the KAE in the control variant, whose
        field goes in clean because the perturbation is applied to its latent instead.
        """
        if where == "control" and m == "KAE":
            return d["truth"][:, 0], True
        return d[f"input__amp{a:g}"], False

    rows = [("TRUTH", None)] + [(m, a) for m in ms for a in amps]
    if axes is None:
        _, axes = plt.subplots(
            len(rows),
            len(frames),
            figsize=(2.6 * len(frames), 1.75 * len(rows)),
            squeeze=False,
        )
    axes = np.atleast_2d(axes)
    clean_note = False
    for i, (m, a) in enumerate(rows):
        f = tru if m == "TRUTH" else d[f"{m}__amp{a:g}__roll"][example, :, channel]
        for j, c in enumerate(frames):
            ax = axes[i, j]
            fr_b = None
            if m == "TRUTH":
                fr = f[c]  # the truth needs no error against itself
            elif c == 0:
                # the INPUT, not the model's own frame 0
                fr_b, was_clean = handed(m, a)
                clean_note |= was_clean
                fr, kk = fr_b[example, channel], 0
            else:
                fr = f[c]
                fr_b, kk = d[f"{m}__amp{a:g}__roll"][:, c], c
            ax.imshow(
                show(fr), origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi, aspect="auto"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                # ax.set_title("frame 0\n(the INITIAL CONDITION,\nnot a model output)"
                ax.set_title("frame 0" if c == 0 else f"frame {c}", fontsize=8.5)
            elif fr_b is not None:
                ax.set_title(f"{err(fr_b, kk):.4f}", fontsize=7.5, pad=2, color="0.25")
        if m == "TRUTH":
            lab, col = "TRUTH\nat $t_0$", "black"
        else:
            note = (
                "\n(handed a CLEAN field;\nits LATENT is perturbed)"
                if where == "control" and m == "KAE"
                else ""
            )
            lab, col = f"{m}\namp={a:g}{note}", "black"
        axes[i, 0].set_ylabel(lab, fontsize=7, color=col)
    sub = (
        "frame 0 is the perturbed field every model was handed"
        if not clean_note
        else "frame 0 is the field each model was handed — perturbed for all but the KAE, "
        "whose latent carries the perturbation instead"
    )
    # axes[0, 0].figure.suptitle(
    #     f"Perturbed rollout — {kind} noise on the {where}\n{sub}\n"
    #     f"fields are example {example}; panel labels are rel-$L_2$ against the truth, "
    #     f"mean over n={nB}", fontsize=9.5, y=1.005)
    axes[0, 0].figure.suptitle("Perturbed rollout from t0", fontsize=9.5, y=1.005)
    return axes


def perturbation_table(
    out_dir=TRA, kind="white", where="state", perturb_frames="window"
) -> str:
    d = np.load(
        _perturbed_file(out_dir, kind, where, perturb_frames), allow_pickle=True
    )
    amps = d["amps"]
    ms = [m for m in ORDER if f"{m}__curves" in d.files]
    L = [
        f"final-frame rel-L2 against perturbation size ({kind} noise on x0)",
        "",
        f"{'model':10s} " + " ".join(f"{f'amp={a:g}':>11s}" for a in amps),
    ]
    for m in ms:
        c = d[f"{m}__curves"]
        L.append(f"{m:10s} " + " ".join(f"{c[i][-1]:11.4f}" for i in range(len(amps))))
    L += [
        "",
        "growth of the perturbation itself (error above the unperturbed run,",
        "relative to its value at frame 0) -- below 1 means the dynamics DESTROY it:",
        f"{'model':10s} " + " ".join(f"{'fr' + str(k):>8s}" for k in (1, 2, 5, 12)),
    ]
    # Normalising by the frame-0 excess is only meaningful if the model's state actually
    # received the perturbation. The KAE's encoder projects it out, leaving an excess at
    # the noise floor; with the clamp at 1e-12 every ratio then came out as exactly 1.000,
    # which reads as "the perturbation neither grows nor decays" when the truth is "there
    # was no perturbation to track".
    inj = max(float((d[f"{m}__curves"][-1] - d[f"{m}__curves"][0])[0]) for m in ms)
    for m in ms:
        c = d[f"{m}__curves"]
        g = c[-1] - c[0]
        if g[0] < 0.1 * inj:
            L.append(
                f"{m:10s}   no measurable perturbation at frame 0 "
                f"(excess {g[0]:+.4f} against {inj:.4f} injected)"
            )
            continue
        gg = g.clip(min=1e-12)
        L.append(f"{m:10s} " + " ".join(f"{gg[k] / gg[0]:8.3f}" for k in (1, 2, 5, 12)))
    return "\n".join(L)


def identifiability_table(out_dir=TRA) -> str:
    f = Path(out_dir) / "identifiability.json"
    if not f.is_file():
        raise NotRunYet("run: python -m data_assimilation.tra.identifiability")
    d = json.loads(f.read_text())
    L = [
        "Is a poor analysis an OPTIMISATION failure or an IDENTIFIABILITY failure?",
        "",
        f"{'method':8s} {'J(recovered)':>14s} {'J(true x0)':>12s} {'ratio':>8s}  verdict",
    ]
    for m, v in d.items():
        verdict = (
            "ILL-CONDITIONED - the cost prefers the truth, the optimiser did not "
            "get there"
            if v["ill_conditioned"]
            else "UNIDENTIFIABLE - the cost does NOT prefer the truth; no amount of "
            "optimisation helps"
        )
        L.append(
            f"{m:8s} {v['J_recovered']:14.4e} {v['J_true']:12.4e} "
            f"{v['ratio']:8.1f}x  {verdict}"
        )
    return "\n".join(L)


def fig_post_da(out_dir=TRA, ax=None):
    """Forecast skill AFTER assimilation, against the exact-state floor."""
    d = np.load(
        require(
            Path(out_dir) / "G10_post_da.npz",
            "python -m data_assimilation.tra.exp_tra --sections post_da",
        )
    )
    fr = d["frames"]
    ax = ax or plt.subplots(figsize=(6.6, 4.2))[1]
    for m in ORDER:
        if f"{m}__rel_mean" not in d.files:
            continue
        st = {k: v for k, v in STYLE[m].items() if k != "marker"}
        st["label"] = (
            f"{st['label']}  (analysis " f"{float(d[f'{m}__analysis_rel']):.4f})"
        )
        ax.plot(fr, d[f"{m}__rel_mean"], lw=1.9, **st)
    fc = Path(out_dir) / "G7_forecast.npz"
    if fc.is_file():
        g = np.load(fc)
        for m in ["KAE", "UNet"]:
            if f"{m}__rel_mean" in g.files:
                ax.plot(
                    g["frames"],
                    g[f"{m}__rel_mean"],
                    ls=":",
                    lw=1.3,
                    color=STYLE[m]["color"],
                    alpha=0.8,
                    label=f"{m} from the EXACT state" if m == "UNet" else None,
                )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="forecast lead after $t_0$ (frames)",
        ylabel=r"rel-$L_2$",
        title="Forecast AFTER assimilation (solid) vs from the EXACT state (dotted)",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    return ax


# ---------------------------------------------------------------------------
def fig_spacetime(
    out_dir=TRA,
    channel: int = 3,
    example: int = 0,
    frames=(0, 3, 10, 20, 29),
    axes=None,
):
    """The recovered trajectory: analysis at $t_0$ rolled forward under each model.

    The interesting column is frame 0 -- the analysis itself -- and what happens to it
    afterwards.  A noisy analysis is destroyed by the dynamics within a few frames, which
    is exactly why the 4D-Var cost never penalised it.
    """
    d = np.load(
        require(
            Path(out_dir) / "G9_spacetime.npz",
            "python -m data_assimilation.tra.exp_tra --sections spacetime",
        )
    )
    tru = d["truth"][example, :, channel]
    T = int(d["T"])
    cols = [c for c in frames if c < T]
    ms = [m for m in ORDER if f"{m}__roll" in d.files]
    msk = d["mask"][example] if "mask" in d.files else None
    show = (
        (lambda f: np.where(msk > 0, f, np.nan)) if msk is not None else (lambda f: f)
    )
    lo, hi = np.nanpercentile(show(tru), [1, 99])
    rows = ["TRUTH"] + ms
    if axes is None:
        _, axes = plt.subplots(
            len(rows),
            len(cols),
            figsize=(2.5 * len(cols), 1.9 * len(rows)),
            squeeze=False,
        )
    axes = np.atleast_2d(axes)
    for i, r in enumerate(rows):
        f = tru if r == "TRUTH" else d[f"{r}__roll"][example, :, channel]
        for j, c in enumerate(cols):
            axes[i, j].imshow(
                show(f[c]),
                origin="lower",
                cmap="RdBu_r",
                vmin=lo,
                vmax=hi,
                aspect="auto",
            )
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(
                    f"frame {c}" + ("  ($t_0$)" if c == 0 else ""), fontsize=9
                )
        lab = r
        if r != "TRUTH":
            pf = d[f"{r}__per_frame"]
            lab = f"{r}\n{pf[0]:.3f} → {pf[cols[-1]]:.3f}"
        axes[i, 0].set_ylabel(lab, fontsize=7.5)
    return axes


def spacetime_table(out_dir=TRA) -> str:
    d = np.load(Path(out_dir) / "G9_spacetime.npz")
    ms = [m for m in ORDER if f"{m}__per_frame" in d.files]
    T = int(d["T"])
    ks = [0, 1, 2, 5, 10, 20, T - 1]
    L = [
        "per-frame rel-L2 of the rolled-forward analysis",
        "",
        f"{'frame':>6s} " + " ".join(f"{m:>10s}" for m in ms),
    ]
    for k in ks:
        L.append(f"{k:6d} " + " ".join(f"{d[f'{m}__per_frame'][k]:10.4f}" for m in ms))
    L += [
        "",
        "ratio  frame(T-1) / frame 0  -- below 1 means the dynamics DESTROY the",
        "analysis error rather than propagate it:",
    ]
    for m in ms:
        pf = d[f"{m}__per_frame"]
        L.append(f"   {m:9s} {pf[-1] / max(pf[0], 1e-12):6.3f}")
    return "\n".join(L)


def fig_error_vs_cost(out_dir=TRA, ax=None):
    """What each method costs for the accuracy it delivers."""
    j = {
        **json.loads((Path(out_dir) / "A_headline.json").read_text()),
        **json.loads((Path(out_dir) / "A_headline_diffusion.json").read_text()),
    }
    ax = ax or plt.subplots(figsize=(6.4, 4.4))[1]
    for m, v in j.items():
        if m not in STYLE or not np.isfinite(v["mean"]):
            continue
        ax.errorbar(
            v["wall_s"],
            v["mean"],
            yerr=v["sem"],
            ms=10,
            capsize=3,
            marker=STYLE[m]["marker"],
            color=STYLE[m]["color"],
            lw=0,
            elinewidth=1.2,
            label=m,
        )
        ax.annotate(
            m,
            (v["wall_s"], v["mean"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
            color=STYLE[m]["color"],
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="wall clock for one complete assimilation (s)",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title="Accuracy against cost\n(down and left is better)",
    )
    ax.grid(alpha=0.3, which="both")
    return ax


def summary_table(out_dir=TRA) -> str:
    """Every headline and sweep endpoint in one place."""
    j = {
        **json.loads((Path(out_dir) / "A_headline.json").read_text()),
        **json.loads((Path(out_dir) / "A_headline_diffusion.json").read_text()),
    }
    ms = [m for m in ORDER if m in j]
    L = [
        "HEADLINE (canonical schedule, no noise, fully observed)",
        "",
        f"{'method':10s} {'rel-L2':>18s} {'wall':>9s}",
    ]
    for m in ms:
        v = j[m]
        L.append(f"{m:10s} {v['mean']:11.4f}±{v['sem']:.4f} {v['wall_s']:8.0f}s")
    L += ["", "SWEEP ENDPOINTS (easiest -> hardest setting)"]
    for stem, lab in [
        ("G1_delta_f", "delta_f"),
        ("G4_delta_l", "delta_l"),
        ("G5_n_obs", "N"),
        ("G6_single_obs", "tau (1 obs)"),
        ("G3_sparsity", "obs_frac"),
        ("G2_noise", "noise"),
    ]:
        f = Path(out_dir) / f"{stem}.json"
        if not f.is_file():
            continue
        rows = json.loads(f.read_text())["rows"]
        a, b = rows[0], rows[-1]
        L.append(f"\n  {lab}: {a['value']:g} -> {b['value']:g}")
        for m in ms:
            va, vb = _num(a.get(m)), _num(b.get(m))
            if np.isfinite(va) and np.isfinite(vb):
                L.append(
                    f"     {m:9s} {va:8.4f} -> {vb:8.4f}"
                    f"   ({vb / max(va, 1e-12):6.1f}x)"
                )
    return "\n".join(L)


# ---------------------------------------------------------------------------
def fig_calibration(out_dir=TRA, axes=None):
    """Rank histogram plus per-draw vs ensemble-mean error, for the samplers.

    A calibrated posterior places the truth uniformly among its own draws: a FLAT rank
    histogram.  U-shaped means over-confident (spread too small), dome-shaped means
    under-confident.  Reporting a sampler's mean error alone says nothing about this.
    """
    d = np.load(
        require(
            Path(out_dir) / "G11_calibration.npz",
            "python -m data_assimilation.tra.exp_tra --sections calibration",
        ),
        allow_pickle=True,
    )
    ms = [m for m in ORDER if f"{m}__ranks" in d.files]
    S = int(d["n_draws"])
    if axes is None:
        _, axes = plt.subplots(1, len(ms) + 1, figsize=(4.2 * (len(ms) + 1), 3.8))
    axes = np.asarray(axes).ravel()
    for i, m in enumerate(ms):
        r = np.asarray(d[f"{m}__ranks"])
        axes[i].hist(
            r,
            bins=np.arange(S + 2) - 0.5,
            density=True,
            color=STYLE[m]["color"],
            alpha=0.85,
        )
        axes[i].axhline(
            1.0 / (S + 1), color="k", ls="--", lw=1.4, label="calibrated (flat)"
        )
        axes[i].set(
            xlabel="rank of the truth among draws",
            ylabel="frequency",
            title=f"{m}: rank histogram",
        )
        axes[i].legend(fontsize=7.5)
    a = axes[len(ms)]
    for m in ms:
        pd_ = np.asarray(d[f"{m}__rel_per_draw"]).mean()
        em = np.asarray(d[f"{m}__rel_ens_mean"]).mean()
        a.scatter([pd_], [em], s=90, color=STYLE[m]["color"], label=m)
    lim = a.get_xlim()
    a.plot(lim, lim, "k--", lw=1.2, label="equal")
    a.set(
        xlabel="per-draw rel-$L_2$",
        ylabel="ensemble-mean rel-$L_2$",
        title="does averaging draws help?",
    )
    a.legend(fontsize=7.5)
    return axes


def calibration_table(out_dir=TRA) -> str:
    d = np.load(Path(out_dir) / "G11_calibration.npz", allow_pickle=True)
    ms = [m for m in ORDER if f"{m}__ranks" in d.files]
    S = int(d["n_draws"])
    L = [
        f"posterior calibration, {S} draws per problem",
        "",
        f"{'method':10s} {'per-draw':>10s} {'ens-mean':>10s} {'spread':>9s} "
        f"{'rank u-shape':>13s}",
    ]
    for m in ms:
        r = np.asarray(d[f"{m}__ranks"], dtype=float)
        # fraction of ranks at the extremes vs what a flat histogram would give
        edge = float(((r == 0) | (r == S)).mean()) / (2.0 / (S + 1))
        L.append(
            f"{m:10s} {float(np.asarray(d[f'{m}__rel_per_draw']).mean()):10.4f} "
            f"{float(np.asarray(d[f'{m}__rel_ens_mean']).mean()):10.4f} "
            f"{float(d[f'{m}__ensemble_std']):9.4f} {edge:12.2f}x"
        )
    L += [
        "",
        "rank u-shape = how much mass sits at the extreme ranks relative to a flat",
        "histogram. 1.0 is calibrated; >1 means the spread is too small (over-confident).",
    ]
    return "\n".join(L)


def offgrid_table(out_dir=TRA) -> str:
    d = json.loads(
        require(
            Path(out_dir) / "G12_offgrid.json",
            "python -m data_assimilation.tra.exp_tra --sections offgrid",
        ).read_text()
    )
    rows = d["rows"]
    ms = [m for m in ORDER if m in rows[0]]
    L = [
        "observations at times BETWEEN stored frames",
        "",
        f"{'shift':>7s} " + " ".join(f"{m:>20s}" for m in ms),
    ]
    for r in rows:
        line = f"{r['shift']:+7.2f} "
        for m in ms:
            v = r[m]
            tag = (
                "exact"
                if v["evaluates_exactly"]
                else f"snap {v['time_error_frames']:.2f}"
            )
            line += f"  {v['mean']:8.4f} ({tag:>9s})"
        L.append(line)
    L += ["", d["meta"]["note"]]
    return "\n".join(L)


# ---------------------------------------------------------------------------
def fig_backward(out_dir=TRA, axes=None):
    """Error at t_0 - j: how far back can each method reconstruct?"""
    d = np.load(
        require(
            Path(out_dir) / "G13_backward.npz",
            "python -m data_assimilation.tra.exp_tra --sections backward",
        ),
        allow_pickle=True,
    )
    ms = [m for m in ORDER if f"{m}__err" in d.files]
    j = np.asarray(d["offsets_back"])
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
    a0, a1 = np.asarray(axes).ravel()[:2]
    for m in ms:
        e = np.asarray(d[f"{m}__err"])
        n = int(d[f"{m}__n_representable"])
        st = {k: v for k, v in STYLE[m].items() if k != "marker"}
        a0.plot(j[:n], e[:n], lw=2.0, marker=STYLE[m]["marker"], ms=5, **st)
        if n < len(j):  # mark where the control simply cannot reach
            a0.plot(
                j[n - 1], e[n - 1], marker="x", ms=13, mew=2.4, color=STYLE[m]["color"]
            )
            a0.annotate(
                "control ends",
                (j[n - 1], e[n - 1]),
                textcoords="offset points",
                xytext=(6, 8),
                fontsize=7,
                color=STYLE[m]["color"],
            )
    a0.set(
        xlabel="frames BEFORE $t_0$",
        ylabel=r"rel-$L_2$",
        yscale="log",
        title=("Reconstructing the past\n" "all observations lie strictly AFTER $t_0$"),
    )
    a0.grid(alpha=0.3, which="both")
    a0.legend(fontsize=7.5)

    reach = [int(d[f"{m}__n_representable"]) for m in ms]
    a1.barh(range(len(ms)), reach, color=[STYLE[m]["color"] for m in ms], alpha=0.85)
    a1.set_yticks(range(len(ms)))
    a1.set_yticklabels(ms, fontsize=9)
    a1.invert_yaxis()  # read top-to-bottom in ORDER, like every legend
    a1.set(
        xlabel="frames before $t_0$ the control can represent",
        title="How far back the control reaches\n(a structural property, not accuracy)",
    )
    for i, v in enumerate(reach):
        a1.text(v, i, f" {v}", va="center", fontsize=9)
    a1.grid(alpha=0.3, axis="x")
    return axes


def fig_backward_fields(out_dir=TRA, channel: int = 3, example: int = 0, axes=None):
    """The reconstructed past states themselves."""
    d = np.load(Path(out_dir) / "G13_backward.npz", allow_pickle=True)
    ms = [m for m in ORDER if f"{m}__back" in d.files]
    tru = d["truth"]
    W = min(5, tru.shape[0])
    msk = d["mask"][example] if "mask" in d.files else None
    show = (
        (lambda f: np.where(msk > 0, f, np.nan)) if msk is not None else (lambda f: f)
    )
    lo, hi = np.nanpercentile(show(tru[0, example, channel]), [1, 99])
    rows = ["TRUTH"] + ms
    if axes is None:
        _, axes = plt.subplots(
            len(rows), W, figsize=(2.5 * W, 1.9 * len(rows)), squeeze=False
        )
    axes = np.atleast_2d(axes)
    for i, r in enumerate(rows):
        arr = tru if r == "TRUTH" else d[f"{r}__back"]
        for jj in range(W):
            f = arr[jj, example, channel]
            axes[i, jj].imshow(
                show(f), origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi, aspect="auto"
            )
            axes[i, jj].set_xticks([])
            axes[i, jj].set_yticks([])
            if i == 0:
                axes[i, jj].set_title(f"$t_0-{jj}$" if jj else "$t_0$", fontsize=9)
            if r != "TRUTH" and np.all(~np.isfinite(f)):
                axes[i, jj].text(
                    0.5,
                    0.5,
                    "not\nrepresentable",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="crimson",
                    transform=axes[i, jj].transAxes,
                )
        axes[i, 0].set_ylabel(r, fontsize=8)
    return axes


def fig_window(out_dir=TRA, ax=None):
    """How far back one KAE latent stays faithful."""
    d = json.loads(
        require(
            Path(out_dir) / "G14_window.json",
            "python -m data_assimilation.tra.exp_tra --sections window",
        ).read_text()
    )
    rows = d["rows"]
    x = [r["frames_back"] for r in rows]
    y = [r["mean"] for r in rows]
    e = [r["sem"] for r in rows]
    ax = ax or plt.subplots(figsize=(6.4, 4.2))[1]
    ax.errorbar(
        x,
        y,
        yerr=e,
        lw=2.0,
        marker="o",
        ms=5,
        color=STYLE["KAE"]["color"],
        capsize=3,
        label="KAE, one latent propagated backwards",
    )
    ax.set(
        xlabel="frames before $t_0$",
        ylabel=r"rel-$L_2$",
        yscale="log",
        title=(
            "How far back does ONE latent stay faithful?\n"
            "no re-optimisation — the same $z_0$ evaluated at $e^{-K j \\Delta t}$"
        ),
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------------------
def fig_budget(out_dir=TRA, ax=None):
    """Accuracy against MEASURED wall clock -- the comparison a user actually faces."""
    d = json.loads(
        require(
            Path(out_dir) / "G15_budget.json",
            "python -m data_assimilation.tra.exp_tra --sections budget",
        ).read_text()
    )
    rows = d["rows"]
    ax = ax or plt.subplots(figsize=(7.4, 4.6))[1]
    for m in ORDER:
        r = sorted([x for x in rows if x["method"] == m], key=lambda z: z["wall_s"])
        if not r or not np.isfinite(r[0]["mean"]):
            continue
        st = {k: v for k, v in STYLE[m].items() if k != "marker"}
        ax.plot(
            [x["wall_s"] for x in r],
            [x["mean"] for x in r],
            marker=STYLE[m]["marker"],
            ms=6,
            lw=1.9,
            **st,
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="wall clock for one complete assimilation (s)",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title=(
            "Accuracy at equal cost\n"
            "4D-Var swept over iterations; samplers over draws"
        ),
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    return ax


def budget_table(out_dir=TRA) -> str:
    rows = json.loads((Path(out_dir) / "G15_budget.json").read_text())["rows"]
    L = [f"{'method':10s} {'setting':>12s} {'wall (s)':>9s} {'rel-L2':>9s}"]
    for r in sorted(rows, key=lambda x: (x["method"], x["wall_s"])):
        setting = f"{r['iters']} it" if "iters" in r else f"{r['draws']} draws"
        L.append(
            f"{r['method']:10s} {setting:>12s} {r['wall_s']:9.1f} {r['mean']:9.4f}"
        )
    L += [
        "",
        "Decision rule implied by these numbers:",
        "  budget below ~20 s -> KAE (the samplers cannot run at all)",
        "  budget above ~25 s -> ACDM (one draw suffices; more draws buy nothing)",
    ]
    return "\n".join(L)


def fig_corrector(out_dir=TRA, ax=None):
    """Algorithm 4's Langevin corrector: harmful, or merely mis-scaled?"""
    d = json.loads(
        require(
            Path(out_dir) / "G17_corrector.json",
            "python -m data_assimilation.tra.exp_tra --sections corrector",
        ).read_text()
    )
    rows = d["rows"]
    ax = ax or plt.subplots(figsize=(6.6, 4.2))[1]
    base = [r for r in rows if r["corrections"] == 0 and not r["diverged"]]
    if base:
        ax.axhline(
            base[0]["rel"],
            color="k",
            ls="--",
            lw=1.6,
            label=f"no corrector ({base[0]['rel']:.4f})",
        )
    for c, col in ((1, "#d6604d"), (2, "#762a83")):
        r = sorted(
            [x for x in rows if x["corrections"] == c and not x["diverged"]],
            key=lambda z: z["tau"],
        )
        if r:
            ax.plot(
                [x["tau"] for x in r],
                [x["rel"] for x in r],
                marker="o",
                ms=5,
                lw=1.9,
                color=col,
                label=f"{c} corrector step(s)",
            )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"Langevin step scale $\tau$",
        ylabel=r"analysis rel-$L_2$",
        title=(
            "Does the corrector help?\n"
            r"$\delta=\tau\,\dim(s)/\|s\|^2$ collapses when the score is large"
        ),
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return ax


def fig_background(out_dir=TRA, axes=None):
    """Does a background term rescue the physical-space methods?"""
    d = json.loads(
        require(
            Path(out_dir) / "G16_background.json",
            "python -m data_assimilation.tra.exp_tra --sections background",
        ).read_text()
    )
    rows = d["rows"]
    ms = sorted({r["method"] for r in rows})
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
    a0, a1 = np.asarray(axes).ravel()[:2]
    for m in ms:
        r = sorted([x for x in rows if x["method"] == m], key=lambda z: z["weight"])
        w = [max(x["weight"], 1e-6) for x in r]
        a0.errorbar(
            w,
            [x["mean"] for x in r],
            yerr=[x["sem"] for x in r],
            marker=STYLE[m]["marker"],
            ms=5,
            lw=1.9,
            capsize=2.5,
            color=STYLE[m]["color"],
            label=m,
        )
        a1.plot(
            w,
            [x["high_k_fraction"] for x in r],
            marker=STYLE[m]["marker"],
            ms=5,
            lw=1.9,
            color=STYLE[m]["color"],
            label=m,
        )
    for a, yl, ti in (
        (a0, r"analysis rel-$L_2$", "Accuracy"),
        (a1, "high-$k$ share of spectral energy", "Roughness"),
    ):
        a.set(
            xscale="log",
            yscale="log",
            xlabel="background weight $w$",
            ylabel=yl,
            title=f"{ti} against the background term",
        )
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
    return axes


def background_table(out_dir=TRA) -> str:
    d = json.loads((Path(out_dir) / "G16_background.json").read_text())
    rows = d["rows"]
    L = [f"{'method':7s} {'weight':>9s} {'rel-L2':>10s} {'high-k share':>13s}"]
    for r in sorted(rows, key=lambda x: (x["method"], x["weight"])):
        L.append(
            f"{r['method']:7s} {r['weight']:9g} {r['mean']:10.4f} "
            f"{r['high_k_fraction']:13.4f}"
        )
    L += ["", d["meta"]["note"]]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Added after the audit: the experiments that had no figure, and the tables that
# a figure serves better.
# ---------------------------------------------------------------------------
def fig_reach(out_dir=TRA, ax=None):
    """G18: the analysis time is MOVED to t_0-j and every method is re-solved.

    The fair counterpart to `fig_backward`.  `fig_backward` asks whether ONE assimilation
    determines a whole window, which the KAE wins by construction -- its single latent
    generates any t_0-j while the U-Net's control IS the frame at t_0.  Here each target
    gets its own solve, so every method can attempt every j and the comparison is of
    accuracy rather than of representational reach.  Both belong in the deck; conflating
    them would overstate the KAE.
    """
    d = json.loads(
        require(
            Path(out_dir) / "G18_reach.json",
            "python -m data_assimilation.tra.exp_tra --sections reach",
        ).read_text()
    )
    rows = d["rows"]
    xs = [r["frames_back"] for r in rows]
    ax = ax or plt.subplots(figsize=(7.0, 4.4))[1]
    for m in ORDER:
        if not any(m in r for r in rows):
            continue
        mu = np.array([_num(r.get(m)) for r in rows], dtype=float)
        se = np.array([_num(r.get(m), "sem") for r in rows], dtype=float)
        ax.errorbar(xs, mu, yerr=se, capsize=2.5, lw=1.9, ms=5, **STYLE[m])
    ax.set(
        xlabel="analysis time moved back from $t_0$ (frames)",
        ylabel=r"analysis rel-$L_2$ at the target",
        yscale="log",
        title=(
            "Reach, with EVERY method re-solved at each target\n"
            f"observations stay put, so the lead to the first grows with $j$ "
            f"({d['meta']['iters']} iterations, $n$={d['meta']['n_problems']})"
        ),
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    return ax


def fig_offgrid(out_dir=TRA, axes=None):
    """G12 as a figure: what half a frame of snapping costs each method.

    Left: absolute error against the shift.  Right: the same, divided by each method's own
    error at shift 0, which is the quantity the claim is actually about -- the KAE is
    exempt from snapping, so what matters is how much the others degrade relative to
    themselves, not that they started out worse.
    """
    d = json.loads(
        require(
            Path(out_dir) / "G12_offgrid.json",
            "python -m data_assimilation.tra.exp_tra --sections offgrid",
        ).read_text()
    )
    rows = d["rows"]
    xs = np.array([r["shift"] for r in rows], dtype=float)
    ms = [m for m in ORDER if m in rows[0]]
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    a0, a1 = np.asarray(axes).ravel()[:2]
    for m in ms:
        mu = np.array([r[m]["mean"] for r in rows], dtype=float)
        se = np.array([r[m].get("sem", np.nan) for r in rows], dtype=float)
        exact = bool(rows[0][m].get("evaluates_exactly"))
        st = dict(STYLE[m])
        st["label"] = st["label"] + ("  — exact in $\\tau$" if exact else "  — snapped")
        a0.errorbar(
            xs, mu, yerr=se, capsize=2.5, lw=1.9, ms=5, ls="-" if exact else "--", **st
        )
        a1.plot(
            xs,
            mu / max(mu[0], 1e-12),
            lw=1.9,
            ms=5,
            ls="-" if exact else "--",
            marker=STYLE[m]["marker"],
            color=STYLE[m]["color"],
            label=m,
        )
    # The sweep separates two effects that a single curve would conflate. At +0.25 the
    # nearest stored frame is unchanged, so every snapping method assimilates the SAME
    # frames and only the observed VALUES move (they are interpolated, hence slightly off
    # the data manifold). At +0.50 the nearest frame actually changes and real snapping
    # begins. Shading the second regime makes the decomposition readable.
    snapped0 = rows[0]["offsets_snapped"]
    moved = [r["shift"] for r in rows if r["offsets_snapped"] != snapped0]
    for a in (a0, a1):
        if moved:
            a.axvspan(min(moved) - 1e-9, max(xs), color="0.85", alpha=0.45, zorder=0)
    a0.set(
        xlabel="observation offset from the frame grid (frames)",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        yscale="log",
        title="Off-grid observation times\nunshaded: same frames, interpolated values"
        "  |  shaded: the frames snap too",
    )
    a1.axhline(1.0, color="0.5", lw=1, ls=":")
    a1.set(
        xlabel="observation offset from the frame grid (frames)",
        ylabel="error relative to the same method at shift 0",
        title="Cost of snapping, per method\n(1.0 = the shift cost this method nothing)",
    )
    # the shading label goes on AFTER the scales are set: a text anchored to get_ylim()
    # on a still-linear axis can land at a non-positive y, which has no finite position
    # once the axis is log and makes bbox_inches="tight" blow the figure up
    for a in (a0, a1):
        if moved:
            a.text(
                min(moved),
                a.get_ylim()[1],
                " frames actually snap",
                fontsize=7,
                color="0.35",
                va="top",
                ha="left",
            )
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=7.5)
    gap = rows[0].get("interp_gap_rel")
    if gap:
        a0.text(
            0.02,
            0.02,
            f"sub-frame motion between bracketing frames: "
            f"{np.mean(gap):.3f} rel-$L_2$",
            transform=a0.transAxes,
            fontsize=7,
            color="0.35",
        )
    return axes


def fig_leakage(out_dir=TRA, ax=None):
    """The headline against its leakage control, side by side.

    `gt_interp.nc` -- where every headline number is measured -- is a bit-identical excerpt
    of `val.nc`, the split the KAE's checkpoint selection ran on.  `gt_longer.nc` (Mach
    0.64-0.65) is cut from `test.nc` and is disjoint from both `train.nc` and `val.nc`, so
    no model's selection saw it.  If the ordering is the same in both panels, the
    asymmetry is not what produces it.
    """
    ctl = json.loads(
        require(
            Path(out_dir) / "L_leakage_control.json",
            "python -m data_assimilation.tra.exp_tra --sections leakage_control",
        ).read_text()
    )["rows"]
    base = summary(out_dir)
    ms = [m for m in ORDER if m in ctl and m in base]
    x = np.arange(len(ms))
    ax = ax or plt.subplots(figsize=(8.0, 4.4))[1]
    w = 0.38
    ax.bar(
        x - w / 2,
        [base[m]["mean"] for m in ms],
        w,
        yerr=[base[m]["sem"] for m in ms],
        capsize=3,
        color=[STYLE[m]["color"] for m in ms],
        alpha=0.95,
        label="headline — gt_interp (Mach 0.66–0.68), an excerpt of the KAE's val split",
    )
    ax.bar(
        x + w / 2,
        [ctl[m]["mean"] for m in ms],
        w,
        yerr=[ctl[m]["sem"] for m in ms],
        capsize=3,
        color=[STYLE[m]["color"] for m in ms],
        alpha=0.45,
        hatch="//",
        label="control — gt_longer (Mach 0.64–0.65), disjoint from train AND val",
    )
    for i, m in enumerate(ms):
        r = ctl[m]["mean"] / max(base[m]["mean"], 1e-12)
        ax.annotate(
            f"×{r:.2f}",
            (i + w / 2, ctl[m]["mean"]),
            ha="center",
            textcoords="offset points",
            xytext=(0, 4),
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(ms, fontsize=9)
    ax.set(
        yscale="log",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title="Leakage control: the same canonical problem on a regime\n"
        "no model's checkpoint selection ever saw",
    )
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=7.5, loc="upper left")
    return ax


def fig_recovery_gallery(
    out_dir=TRA, stem="G3_sparsity", channel: int = 3, example: int = 0, axes=None
):
    """The recovery gallery from the KS deck: what each method's analysis LOOKS like.

    A sweep of scalars says how much is lost and never what.  Rows are sweep points, one
    column per method with the truth first, and each panel carries its own error, so a
    reader can connect the number to the picture.  Density (channel 3) is used because
    ``v_x`` is nearly uniform and hides the wake; colour limits are percentile, not
    symmetric, because density is not zero-centred.
    """
    return fig_sweep_fields(
        out_dir, stem=stem, channel=channel, example=example, axes=axes
    )


def observation_gap(out_dir, stem, tag) -> dict:
    """How far the OBSERVED states are from the target, for one sweep point.

    The problems are rebuilt from the seed and settings recorded in the sweep JSON, and the
    rebuilt truth is checked against the truth saved beside the analyses, so the numbers
    are guaranteed to be for the problems actually solved.  Distances are rel-L2 over all
    channels with the obstacle masked, the same metric as the analysis error, so
    "recovered - truth" and "observation - truth" can be compared directly: an analysis
    that is no closer to u(t0) than the nearest observation has recovered nothing.
    """
    import torch
    from data_assimilation.tra.bridge import PhysicalData, rel_l2
    from data_assimilation.tra.protocol import build_problem

    row = next(r for r in load_sweep(out_dir, stem)["rows"] if r["tag"] == tag)
    pm = row["problem"]
    if pm.get("offsets_real") or pm.get("noise_std", 0) or pm.get("obs_frac", 1) < 1:
        raise ValueError(f"{stem}/{tag} is not a clean, fully observed point")
    data = PhysicalData(pm["data_file"], "tra", "cpu")
    prob = build_problem(
        data,
        name="gap",
        n_problems=pm["n_problems"],
        seed=pm["seed"],
        offsets=np.asarray(pm["offsets"]),
    )
    sim, t0 = torch.as_tensor(prob.sim), torch.as_tensor(prob.t0)
    tru, om = data.frames(sim, t0), data.mask_for(sim)
    saved = np.load(Path(out_dir) / f"{stem}_fields.npz")[f"{tag}__truth"]
    if not np.allclose(tru[: len(saved)].numpy(), saved):
        raise RuntimeError(
            f"rebuilt problems for {stem}/{tag} do not match the saved truth"
        )
    obs = torch.stack(
        [data.frames(sim, t0 + int(o)) for o in prob.offsets]
    )  # [N,B,...]
    gap = torch.stack([rel_l2(obs[i], tru, om) for i in range(len(prob.offsets))])
    gm = gap.mean(1).numpy()
    # nearest in TIME is not nearest in STATE: in a periodic wake a later observation can
    # sit closer to u(t0) than the first one, and copying it is the stronger baseline
    i_near, i_close = int(np.argmin(prob.offsets)), int(np.argmin(gm))
    return {
        "offsets": prob.offsets.tolist(),
        "gap_mean": gm,
        "gap": gap.numpy(),
        "i_near": i_near,
        "i_close": i_close,
        "i_last": int(np.argmax(prob.offsets)),
        "obs": obs.numpy(),
    }


def fig_error_map(
    out_dir=TRA,
    stem="G1_delta_f",
    tag=None,
    channel: int = 3,
    example: int = 0,
    axes=None,
    show_obs: bool = False,
    lim=None,
):
    """WHERE each method's analysis is wrong, not just by how much.

    The analysis minus the truth, on a shared symmetric scale.  This is where the two
    failure modes separate visibly: the KAE's residual is smooth and concentrated in the
    wake, while the U-Net's and FNO's is grid-scale speckle spread over the whole domain --
    exactly the component the dynamics erase before the first observation, which is why the
    4D-Var cost never penalised it.

    ``show_obs`` adds the NEAREST OBSERVATION minus the truth as a panel on the same scale,
    and puts the lead of every observation, and its rel-L2 distance from u(t0), in the
    title -- the reference an analysis has to beat.  ``lim`` fixes the colour scale so two
    figures (e.g. two values of delta_f) can be read against each other.
    """
    npz = Path(out_dir) / f"{stem}_fields.npz"
    if not npz.is_file():
        raise NotRunYet(
            f"{npz} not on disk (rerun the sweep; it saves fields as it goes)"
        )
    d = np.load(npz, allow_pickle=True)
    rows = load_sweep(out_dir, stem)["rows"]
    tag = tag or rows[0]["tag"]
    row = next(r for r in rows if r["tag"] == tag)
    ms = [m for m in ORDER if f"{tag}__{m}__analysis" in d.files]
    tru = d[f"{tag}__truth"][example, channel]
    msk = d.get(f"{tag}__mask")
    m2 = None if msk is None else msk[example]
    show = (lambda a: np.where(m2 > 0, a, np.nan)) if m2 is not None else (lambda a: a)
    errs = [show(d[f"{tag}__{m}__analysis"][example, channel] - tru) for m in ms]
    gap = observation_gap(out_dir, stem, tag) if show_obs else None
    if gap is not None:
        oi = sorted({gap["i_near"], gap["i_close"], gap["i_last"]})
        ms_obs = [f"obs{i}" for i in oi] + ms
        errs = [show(gap["obs"][i, example, channel] - tru) for i in oi] + errs
        best = float(gap["gap_mean"][gap["i_close"]])
        il = gap["i_last"]
        last_gt = float(gap["gap_mean"][il])
        last_case = float(gap["gap"][il, example])
        # how far each ANALYSIS sits from the last observed state, relative to that state:
        # if a method were just returning the observation this would be ~0
        y_l = gap["obs"][il, example]
        mm = 1.0 if m2 is None else (m2 > 0)[None]
        from_last = {
            m: float(
                np.sqrt((((d[f"{tag}__{m}__analysis"][example] - y_l) * mm) ** 2).sum())
                / np.sqrt(((y_l * mm) ** 2).sum())
            )
            for m in ms
        }
    else:
        ms_obs = ms
    # ACDM-ncn is run outside its training regime by request and its residual is ~100x
    # everyone else's; on a shared scale it flattens all four other panels to grey. It
    # keeps its own scale, stated in its title, rather than being dropped or rescaled
    # silently.
    inr = [e for m, e in zip(ms_obs, errs) if m != "ACDM-ncn"] or errs
    if lim is None:
        lim = float(np.nanpercentile(np.abs(np.stack(inr)), 99))
    if axes is None:
        nc = len(ms_obs) + 1
        fig, axes = plt.subplots(
            1, nc, figsize=(2.9 * nc, 3.1 if show_obs else 2.7), squeeze=False
        )
        # room for the colourbar reserved up front: tight_layout cannot place one that is
        # attached to a subset of the axes, and it lands on top of the last panel
        fig.subplots_adjust(
            left=0.02,
            right=0.90,
            top=0.72 if show_obs else 0.80,
            bottom=0.04,
            wspace=0.06,
        )
    else:
        fig = np.atleast_2d(axes).ravel()[0].figure
    axes = np.atleast_2d(axes).ravel()
    lo, hi = np.nanpercentile(show(tru), [1, 99])
    axes[0].imshow(
        show(tru), origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi, aspect="auto"
    )
    axes[0].set_title("TRUTH (density)", fontsize=9)
    for a, m, e in zip(axes[1:], ms_obs, errs):
        own = float(np.nanpercentile(np.abs(e), 99)) if m == "ACDM-ncn" else lim
        im_ = a.imshow(
            e, origin="lower", cmap="coolwarm", vmin=-own, vmax=own, aspect="auto"
        )
        if m != "ACDM-ncn":
            im = im_
        if m.startswith("obs"):
            i = int(m[3:])
            kind = " / ".join(
                k
                for k, j in (
                    ("nearest in time", gap["i_near"]),
                    ("closest in state", gap["i_close"]),
                    ("last", gap["i_last"]),
                )
                if j == i
            )
            a.set_title(
                f"observation $t_0{{+}}{gap['offsets'][i]}$ − truth\n"
                f"rel-$L_2$ {gap['gap_mean'][i]:.4f}  ({kind})",
                fontsize=8.5,
                color="0.25",
            )
            continue
        mu = row[m]["mean"] if isinstance(row.get(m), dict) else float("nan")
        ttl = f"{m} − truth\nrel-$L_2$ {mu:.4f}"
        if gap is not None:
            ttl = (
                f"{m} − truth\nrel-$L_2$ {mu:.4f}  ({mu / best:.2f}× closest obs)"
                f"\nfrom last obs {from_last[m]:.3f}  (truth {last_case:.3f})"
            )
        a.set_title(
            ttl + (f"\n(own scale ±{own:.2g})" if m == "ACDM-ncn" else ""),
            fontsize=8.5,
            color=STYLE[m]["color"],
        )
    if gap is not None:
        offs = gap["offsets"]
        fig.suptitle(
            f"Observations at $t_0$ + {offs} frames   ($\\delta_f$ = {min(offs)}, "
            f"$\\delta_l$ = {max(offs)}),   distance from $u(t_0)$: "
            + ", ".join(f"{g:.3f}" for g in gap["gap_mean"])
            + f"\nrel-$L_2$ over {len(gap['gap'][0])} problems; maps show problem {example}",
            fontsize=9.5,
            y=0.99,
        )
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    cax = fig.add_axes([0.915, 0.10, 0.010, 0.60 if show_obs else 0.68])
    fig.colorbar(im, cax=cax).set_label("analysis − truth (density)", fontsize=7.5)
    return axes


def training_spec_table(out_dir=TRA) -> str:
    """The reproducibility table, straight out of training_specs.json."""
    f = require(
        Path(out_dir) / "training_specs.json",
        "python -m data_assimilation.tra.training_specs",
    )
    d = json.loads(f.read_text())
    k, t = d["kae"], d["turbpred"]
    L = [
        "TRAINING SPECIFICATION  (read from the checkpoints, not typed in)",
        "",
        f"{'model':10s} {'params':>10s} {'trained on':>18s} {'losses':>26s} "
        f"{'epochs':>7s}",
        f"{'KAE':10s} {k['params']:10,d} "
        f"{'rollout, ' + str(k['operator_type']):>18s} "
        f"{'Koopman generator, rank ' + str(k['rank']):>26s} "
        f"{str(k['epoch']):>7s}",
    ]
    for m, v in t.items():
        L.append(
            f"{m:10s} {v['params']:10,d} "
            f"{'seq ' + str(v['sequence_length']):>18s} "
            f"{', '.join(f'{a}={b:g}' for a, b in v['losses'].items()):>26s} "
            f"{v['epochs']:7d}"
        )
    td = d["test_discipline"]
    mv = sorted(td["training_mach_values"])
    gaps = [(a, b) for a, b in zip(mv, mv[1:]) if b - a > 0.015]
    gtxt = ", ".join(f"{a + 0.01:.2f}–{b - 0.01:.2f}" for a, b in gaps) or "none"
    L += [
        "",
        "TEST DISCIPLINE",
        "",
        f"  training Mach values   {mv[0]:.2f}–{mv[-1]:.2f}, " f"with a GAP at {gtxt}",
        f"  DA test regime         gt_interp.nc, Mach "
        f"{td['test_mach_range'][0]:.2f}–{td['test_mach_range'][1]:.2f}",
        f"  tuning regime          gt_extrap.nc, Mach "
        f"{td['validation_mach_range'][0]:.2f}–{td['validation_mach_range'][1]:.2f}",
        f"  Mach ranges overlap    {td['mach_ranges_overlap']}",
        "",
        "  provenance of each DA regime (frame-by-frame, not by Mach number):",
    ]
    for reg, splits in td["da_regime_provenance"].items():
        L.append(
            f"    {reg:16s} shares trajectories with: "
            f"{', '.join(s.replace('vs_', '') for s in splits) or 'nothing'}"
        )
    L += ["", "  " + td["known_asymmetry"].replace(". ", ".\n  ")]
    return "\n".join(L)


def consistency_report(out_dir=TRA) -> str:
    """Run the third gate and return its output, so the deck shows its own audit."""
    import subprocess

    r = subprocess.run(
        ["python", "-m", "data_assimilation.tra.consistency_check"],
        capture_output=True,
        text=True,
        cwd=".",
    )
    return r.stdout or r.stderr


def fig_consistency(out_dir=TRA, ax=None):
    """The consistency gate as a picture: how many claims are verified, and against what."""
    txt = consistency_report(out_dir)
    tally = {}
    for line in txt.splitlines():
        w = line.split(maxsplit=1)
        if w and w[0] in ("PASS", "FAIL", "MISSING", "UNVERIFIED", "FIGURE-ONLY"):
            tally[w[0]] = tally.get(w[0], 0) + 1
    order = ["PASS", "FIGURE-ONLY", "UNVERIFIED", "MISSING", "FAIL"]
    cols = {
        "PASS": "#1b7837",
        "FIGURE-ONLY": "#7fbf7b",
        "UNVERIFIED": "#f0a202",
        "MISSING": "#d6604d",
        "FAIL": "#67001f",
    }
    ks = [k for k in order if tally.get(k)]
    ax = ax or plt.subplots(figsize=(7.4, 3.0))[1]
    left = 0
    for k in ks:
        ax.barh([0], [tally[k]], left=left, color=cols[k], label=f"{k} ({tally[k]})")
        if tally[k] > 1:
            ax.text(
                left + tally[k] / 2,
                0,
                str(tally[k]),
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
        left += tally[k]
    ax.set_yticks([])
    ax.set_xlim(0, left)
    ax.set(
        xlabel="claims checked against the file that produced them",
        title="Consistency gate: every number on a slide, re-read from its source",
    )
    ax.legend(
        fontsize=8,
        ncol=len(ks),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        frameon=False,
    )
    ax.grid(False)
    return ax


def fig_paradox(out_dir=TRA, ax=None):
    """Forward skill against inverse skill -- the dissociation, in one plot.

    x: one-step free-running forecast error from the EXACT state (how good a propagator
    the model is).  y: the analysis error at t_0 (how good an inverse model it is).  If the
    two were the same thing the points would lie on a rising line.  They do not: U-Net is
    the best propagator and the worst inverse model, and ACDM-ncn -- excellent forward,
    unusable inverse -- is the extreme case, for a reason the deck states separately.
    """
    fc = np.load(
        require(
            Path(out_dir) / "G7_forecast.npz",
            "python -m data_assimilation.tra.exp_tra --sections forecast",
        )
    )
    s = summary(out_dir)
    ms = [m for m in ORDER if m in s and f"{m}__rel_mean" in fc.files]
    ax = ax or plt.subplots(figsize=(7.0, 4.8))[1]
    for m in ms:
        x, y = float(fc[f"{m}__rel_mean"][0]), s[m]["mean"]
        ax.errorbar(
            x,
            y,
            yerr=s[m]["sem"],
            marker=STYLE[m]["marker"],
            ms=11,
            lw=0,
            elinewidth=1.2,
            capsize=3,
            color=STYLE[m]["color"],
        )
        ax.annotate(
            m,
            (x, y),
            textcoords="offset points",
            xytext=(9, -3),
            fontsize=9,
            color=STYLE[m]["color"],
        )
    lo = (
        min(
            min(float(fc[f"{m}__rel_mean"][0]) for m in ms),
            min(s[m]["mean"] for m in ms),
        )
        * 0.5
    )
    hi = (
        max(
            max(float(fc[f"{m}__rel_mean"][0]) for m in ms),
            max(s[m]["mean"] for m in ms),
        )
        * 2
    )
    ax.plot(
        [lo, hi],
        [lo, hi],
        color="0.6",
        ls=":",
        lw=1.2,
        label="if forward skill were inverse skill",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(lo, hi),
        ylim=(lo, hi),
        xlabel=r"ONE-STEP forecast error from the exact state (rel-$L_2$)",
        ylabel=r"ANALYSIS error at $t_0$ (rel-$L_2$)",
        title="The best forward model is not the best inverse model",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    return ax


# delta_l is included because with delta_f pinned at 1 there was NO crossover on it: ACDM
# had a close observation at every horizon and stayed flat, which located its weakness in
# the LEAD TO THE FIRST OBSERVATION rather than the horizon. The label reads the pinned
# delta_f from the file, so a re-run at another value is labelled as what it is.
CROSSOVER = [
    ("G1_delta_f", r"$\delta_f$ (frames to first obs)"),
    ("G4_delta_l", r"$\delta_l$ (horizon; $\delta_f$ pinned)"),
    ("G3_sparsity", "fraction of grid points observed"),
    ("G2_noise", r"observation noise $\sigma_y$"),
    ("G6_single_obs", r"$\tau$ (a single observation)"),
]


def fig_crossover(out_dir=TRA, a="ACDM", b="KAE", axes=None, dirs=None, stems=None):
    """WHERE each of the two leading methods wins, on every axis at once.

    The ratio error(a)/error(b) against the swept quantity.  Above 1, ``b`` wins; below 1,
    ``a`` wins; the crossing is the point the decision actually turns on.  This replaces a
    hand-typed table of the same comparison -- the ratio is the claim, so the figure should
    plot the ratio.

    ``dirs`` maps a sweep stem to the directory to read it from instead of ``out_dir``, so
    sweeps re-run at a different pinned delta_f can sit beside the ones that were not.
    ``stems`` picks which axes to draw (default: every one in CROSSOVER).
    """
    src = lambda st: Path((dirs or {}).get(st, out_dir))
    labels = dict(CROSSOVER, G5_n_obs=r"$N$ (observation times; $\delta_f$ pinned)")
    stems = [
        (st, labels[st])
        for st in (stems or [c for c, _ in CROSSOVER])
        if (src(st) / f"{st}.json").is_file()
    ]
    if not stems:
        raise NotRunYet("no sweeps on disk yet")
    if axes is None:
        _, axes = plt.subplots(
            1, len(stems), figsize=(3.7 * len(stems), 3.9), squeeze=False
        )
    axes = np.atleast_2d(axes).ravel()
    for ax, (stem, lab) in zip(axes, stems):
        sw = load_sweep(src(stem), stem)
        rows = sorted(sw["rows"], key=lambda r: r["value"])
        lab = lab.replace("pinned)", f"pinned at {pinned_delta_f(sw['meta'])})")
        xs, rs = [], []
        for r in rows:
            va, vb = _num(r.get(a)), _num(r.get(b))
            if np.isfinite(va) and np.isfinite(vb) and va > 0 and vb > 0:
                xs.append(r["value"])
                rs.append(va / vb)
        if not xs:
            ax.text(
                0.5,
                0.5,
                f"{a} not on disk\nfor {stem}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="crimson",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        ax.plot(xs, rs, lw=2.0, marker="o", ms=5, color="#333333")
        ax.axhline(1.0, color="0.5", lw=1.2, ls="--")
        ax.fill_between(
            xs,
            1,
            rs,
            where=np.array(rs) > 1,
            alpha=0.18,
            color=STYLE[b]["color"],
            interpolate=True,
        )
        ax.fill_between(
            xs,
            1,
            rs,
            where=np.array(rs) <= 1,
            alpha=0.18,
            color=STYLE[a]["color"],
            interpolate=True,
        )
        ax.set(xscale="log", yscale="log", xlabel=lab)
        ax.grid(alpha=0.3, which="both")
        ax.set_title(stem, fontsize=9)
    axes[0].set_ylabel(f"error({a}) / error({b})", fontsize=9)
    axes[0].figure.suptitle(
        f"Above the line {b} wins; below it {a} wins  " f"(shaded by winner)",
        fontsize=10,
    )
    return axes


def fig_misspec(out_dir=TRA, ax=None):
    """G19: Gaussian vs Laplace observation noise at matched variance.

    Every method assumes a Gaussian observation error.  Same variance, heavier tails: the
    ratio Laplace/Gaussian is the cost of that assumption being wrong, per method, and it
    is the quantity plotted -- an absolute pair of curves would mostly show what the noise
    sweep already shows.
    """
    d = json.loads(
        require(
            Path(out_dir) / "G19_misspec.json",
            "python -m data_assimilation.tra.exp_tra --sections misspec_noise",
        ).read_text()
    )
    rows = d["rows"]
    lv = sorted({r["value"] for r in rows})
    ax = ax or plt.subplots(figsize=(7.2, 4.4))[1]
    for m in ORDER:
        xs, ys = [], []
        for v in lv:
            g = next(
                (
                    r
                    for r in rows
                    if r["value"] == v and r.get("noise_dist") == "gaussian"
                ),
                None,
            )
            la = next(
                (
                    r
                    for r in rows
                    if r["value"] == v and r.get("noise_dist") == "laplace"
                ),
                None,
            )
            if not (
                g and la and isinstance(g.get(m), dict) and isinstance(la.get(m), dict)
            ):
                continue
            a, b = _num(g.get(m)), _num(la.get(m))
            if np.isfinite(a) and np.isfinite(b) and a > 0:
                xs.append(v)
                ys.append(b / a)
        if xs:
            ax.plot(xs, ys, lw=1.9, ms=5, **STYLE[m])
    ax.axhline(1.0, color="0.5", ls="--", lw=1.2)
    ax.set(
        xscale="log",
        xlabel=r"observation noise $\sigma_y$ (both distributions)",
        ylabel="error under Laplace / error under Gaussian",
        title=(
            "Misspecified observation noise\n"
            "same variance, heavier tails — 1.0 means the shape did not matter"
        ),
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    return ax


def fig_joint_sparsity(out_dir=TRA, ax=None):
    """G20: one sensor budget, spent densely-and-rarely or sparsely-and-often.

    ``N x obs_frac`` is held fixed, so every point on the x axis costs the same number of
    scalar measurements.  What changes is only how they are arranged in space and time.
    """
    d = json.loads(
        require(
            Path(out_dir) / "G20_joint_sparsity.json",
            "python -m data_assimilation.tra.exp_tra --sections joint_sparsity",
        ).read_text()
    )
    rows = d["rows"]
    xs = [r["value"] for r in rows]
    fr = [r.get("obs_frac") for r in rows]
    ax = ax or plt.subplots(figsize=(7.4, 4.4))[1]
    for m in ORDER:
        if not any(isinstance(r.get(m), dict) for r in rows):
            continue
        mu = np.array([_num(r.get(m)) for r in rows], dtype=float)
        se = np.array([_num(r.get(m), "sem") for r in rows], dtype=float)
        ax.errorbar(xs, mu, yerr=se, capsize=2.5, lw=1.9, ms=5, **STYLE[m])
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="observation times $N$   (with the sensor fraction cut to match)",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title=("One fixed sensor budget, spent differently\n" f"{d['meta']['varies']}"),
    )
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(xs)
    sec.set_xticklabels([f"{f:.3g}" if f is not None else "" for f in fr], fontsize=7.5)
    sec.set_xlabel("fraction of grid points observed", fontsize=8.5)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    return ax


def fig_identifiability(out_dir=TRA, ax=None):
    """Is a poor analysis an OPTIMISATION failure or an IDENTIFIABILITY failure?

    For each method, the 4D-Var cost evaluated at its own recovered analysis against the
    cost at the TRUE x_0.  Which side of the dashed line each pair falls on is the whole
    diagnosis, and it is opposite for the two baselines:

      J(true) < J(recovered)   the cost DOES prefer the truth and the optimiser failed to
                               reach it -- ill-conditioned, more iterations might help.
      J(true) > J(recovered)   the cost prefers the method's own answer to the truth --
                               unidentifiable, and no amount of optimisation can help.
    """
    f = Path(out_dir) / "identifiability.json"
    if not f.is_file():
        raise NotRunYet("run: python -m data_assimilation.tra.identifiability")
    d = json.loads(f.read_text())
    ms = [m for m in ORDER if m in d]
    x = np.arange(len(ms))
    ax = ax or plt.subplots(figsize=(7.6, 4.6))[1]
    w = 0.36
    ax.bar(
        x - w / 2,
        [d[m]["J_recovered"] for m in ms],
        w,
        color=[STYLE[m]["color"] for m in ms],
        alpha=0.95,
        label=r"$J$ at the method's own analysis",
    )
    ax.bar(
        x + w / 2,
        [d[m]["J_true"] for m in ms],
        w,
        color=[STYLE[m]["color"] for m in ms],
        alpha=0.40,
        hatch="//",
        label=r"$J$ at the TRUE $x_0$",
    )
    for i, m in enumerate(ms):
        r = d[m]["ratio"]
        ill = d[m]["ill_conditioned"]
        ax.annotate(
            (
                f"$J$(true)/$J$(analysis) = ×{1 / r:.2f}"
                if not ill
                else f"$J$(analysis)/$J$(true) = ×{r:.1f}"
            ),
            (i, max(d[m]["J_recovered"], d[m]["J_true"])),
            ha="center",
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=8.5,
            color="0.15",
        )
        ax.annotate(
            (
                "ill-conditioned\n(the cost prefers the truth)"
                if ill
                else "UNIDENTIFIABLE\n(the cost prefers its own answer)"
            ),
            (i, max(d[m]["J_recovered"], d[m]["J_true"])),
            ha="center",
            textcoords="offset points",
            xytext=(0, 22),
            fontsize=8.5,
            color="0.15" if ill else "crimson",
            fontweight="normal" if ill else "bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(ms, fontsize=10)
    ax.set(
        yscale="log",
        ylabel=r"4D-Var cost $J$",
        title="Optimisation failure, or identifiability failure?",
    )
    # headroom for the per-method verdicts: on a log axis the default top sits right on
    # the tallest bar and the annotation lands in the title
    hi = max(max(d[m]["J_recovered"], d[m]["J_true"]) for m in ms)
    lo = min(min(d[m]["J_recovered"], d[m]["J_true"]) for m in ms)
    ax.set_ylim(lo / 2, hi * 12)
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=8.5, loc="upper right")
    return ax


def fig_perturbed_curves(
    out_dir=TRA, kind="white", where="state", amp=None, ax=None, perturb_frames="window"
):
    """Does the perturbation GROW or DECAY under each model's dynamics?

    The quantity is the error ABOVE that model's own unperturbed run, normalised by its
    value at frame 0, so every model starts at 1 and what is compared is the growth rate
    rather than the offset.  Below 1 means the dynamics destroy the perturbation — which
    is the whole reason a 4D-Var analysis at t_0 is free to be speckle: the observations
    all lie downstream and cannot see what has already been erased.

    Note the KAE's curve is measuring something different from the others' when
    ``where="state"``: its encoder has already removed most of the perturbation before
    frame 0, so its "frame 0 error above baseline" is a tiny number and the ratio is
    noisy. ``where="control"`` is the like-for-like comparison.
    """
    f_ = _perturbed_file(out_dir, kind, where, perturb_frames)
    d = np.load(
        require(
            f_,
            f"python -m data_assimilation.tra.perturb_demo --kind {kind} "
            f"--where {where} --frames 25",
        ),
        allow_pickle=True,
    )
    amps = list(np.asarray(d["amps"]).ravel())
    amp = amps[-1] if amp is None else amp
    ai = amps.index(amp)
    ax = ax or plt.subplots(figsize=(7.2, 4.4))[1]
    # Normalising by the frame-0 excess is only meaningful if there IS one. With the
    # perturbation on the STATE, the KAE's encoder removes it before frame 0: its excess
    # is 4e-04 against an injected 5e-02, so the ratio divides by what is essentially
    # projection residue and the curve is noise. Each legend entry carries its own frame-0
    # excess, and a curve resting on less than a tenth of the injected perturbation is
    # drawn dashed and called out, rather than presented as a growth rate.
    inj = max(
        (
            float(np.clip(d[f"{m}__curves"][ai] - d[f"{m}__curves"][0], 0, None)[0])
            for m in ORDER
            if f"{m}__curves" in d.files
        ),
        default=0.0,
    )
    weak = []
    for m in ORDER:
        k = f"{m}__curves"
        if k not in d.files:
            continue
        c, c0 = d[k][ai], d[k][0]
        g = np.clip(c - c0, 1e-12, None)
        thin = g[0] < 0.1 * inj
        if thin:
            weak.append(m)
        st = {kk: v for kk, v in STYLE[m].items() if kk != "marker"}
        st["label"] = f"{st['label']}   (frame-0 excess {g[0]:.4f})"
        ax.plot(np.arange(len(g)), g / g[0], lw=1.9, ls=":" if thin else "-", **st)
    ax.axhline(1.0, color="0.5", ls="--", lw=1.2)
    # the caveat goes under the axis, where it cannot land on a curve or the legend
    xlab = "frames after the perturbed initial condition"
    if weak:
        xlab += (
            f"\ndotted: {', '.join(weak)} never received the perturbation at "
            f"frame 0 (excess < 10% of the {inj:.4f} injected),\n"
            f"so its ratio divides by projection residue and is not a growth rate"
        )
    ax.set(
        yscale="log",
        xlabel=xlab,
        ylabel="error above the unperturbed run, relative to frame 0",
        title=(
            f"Does the perturbation grow or decay?  ({kind} noise on the {where}, "
            f"amp={amp:g})\nbelow 1 = the dynamics DESTROY it"
        ),
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    return ax


# ---------------------------------------------------------------------------
# Extensions: all tra regimes with cluster-robust CIs, and window-length scaling.
# ---------------------------------------------------------------------------
REGIME_LABEL = {
    "test": (
        "gt_interp\nMa 0.66–0.68",
        "in the training gap;\nexcerpt of the KAE's val split",
    ),
    "long": (
        "gt_longer\nMa 0.64–0.65",
        "in the gap, from test.nc;\nclean for every model",
    ),
    "val": (
        "gt_extrap\nMa 0.50–0.52",
        "OUTSIDE training range,\nAND the tuning regime",
    ),
}


def _ci(entry):
    """(mean, lo, hi) using the cluster interval, falling back to the naive one."""
    if not isinstance(entry, dict) or entry.get("mean") is None:
        return None
    ci = entry.get("ci95_cluster") or entry.get("ci95_naive")
    m = entry["mean"]
    return (m, m, m) if ci is None else (m, ci[0], ci[1])


def load_regimes(out_dir=TRA) -> dict:
    """The cross-regime CI results, however they were produced.

    A single `--sections regimes --regimes test long val` run writes all three rows into
    one file. Each per-regime campaign also runs the section for its own regime, so the
    three campaign directories hold one row each. Either is usable; the merged form is
    preferred when the combined run is absent, so the comparison does not need a separate
    ~3 h job that would only recompute what the campaigns already did.
    """
    f = Path(out_dir) / "R_regimes.json"
    if f.is_file():
        d = json.loads(f.read_text())
        if len(d.get("rows", {})) > 1:
            return d
    merged, meta = {}, None
    for key, (dd, _, _) in REGIME_DIRS.items():
        g = dd / "R_regimes.json"
        if not g.is_file():
            continue
        gd = json.loads(g.read_text())
        meta = meta or gd.get("meta")
        for split, row in gd.get("rows", {}).items():
            merged[split] = row
    if not merged:
        raise NotRunYet(
            "no regime CI results yet.\n    produce them with: "
            "python -m data_assimilation.tra.exp_tra --sections regimes "
            "--regimes test long val"
        )
    return {
        "meta": {**(meta or {}), "merged_from_per_regime_campaigns": True},
        "rows": merged,
    }


def fig_regimes(out_dir=TRA, ax=None, log=True):
    """The canonical problem on every tra regime, with cluster-robust 95% intervals.

    The error bars are t-intervals over PER-TRAJECTORY means, not over problems: 32
    problems drawn from 6 trajectories are not 32 independent cases, and the naive
    interval is roughly sqrt(n/n_traj) too narrow.  The naive one is drawn as a thin inner
    tick so the difference is visible rather than asserted.

    The three regimes are held out in three different senses, which the x labels state --
    `gt_extrap` in particular is the TUNING regime and is a stress test, not a clean test.
    """
    d = load_regimes(out_dir)
    rows = d["rows"]
    splits = [s for s in ("test", "long", "val") if s in rows]
    ms = [m for m in ORDER if any(isinstance(rows[s].get(m), dict) for s in splits)]
    ax = ax or plt.subplots(figsize=(9.6, 5.0))[1]
    w = 0.8 / max(len(ms), 1)
    for i, m in enumerate(ms):
        xs, mu, lo, hi, nlo, nhi = [], [], [], [], [], []
        for j, s in enumerate(splits):
            c = _ci(rows[s].get(m))
            if c is None:
                continue
            xs.append(j + (i - (len(ms) - 1) / 2) * w)
            mu.append(c[0])
            lo.append(c[0] - c[1])
            hi.append(c[2] - c[0])
            nv = rows[s][m].get("ci95_naive_halfwidth") or 0.0
            nlo.append(nv)
            nhi.append(nv)
        if not xs:
            continue
        ax.errorbar(
            xs,
            mu,
            yerr=[lo, hi],
            fmt=STYLE[m]["marker"],
            ms=7,
            capsize=4,
            elinewidth=1.8,
            color=STYLE[m]["color"],
            lw=0,
            label=STYLE[m]["label"],
        )
        ax.errorbar(
            xs,
            mu,
            yerr=[nlo, nhi],
            fmt="none",
            capsize=8,
            elinewidth=0,
            ecolor=STYLE[m]["color"],
            alpha=0.55,
        )
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels(
        [f"{REGIME_LABEL[s][0]}\n\n{REGIME_LABEL[s][1]}" for s in splits], fontsize=8
    )
    n = d["meta"]["n_problems"]
    ntr = [rows[s]["problem"]["n_trajectories"] for s in splits]
    if log:
        ax.set_yscale("log")
    ax.set(
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title=(
            f"Every tra regime, {n} problems each over {min(ntr)}–{max(ntr)} "
            f"trajectories\nthick bar: 95% CI clustered BY TRAJECTORY   ·   "
            f"thin cap: the naive per-problem CI"
        ),
    )
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=7.5, ncol=2)
    return ax


def regimes_table(out_dir=TRA) -> str:
    d = load_regimes(out_dir)
    rows = d["rows"]
    splits = [s for s in ("test", "long", "val") if s in rows]
    L = [
        "CANONICAL PROBLEM ON EVERY tra REGIME",
        "",
        "95% CI clustered by trajectory (the unit of replication); the naive",
        "per-problem interval is shown after it, and is too narrow.",
        "",
    ]
    for s in splits:
        p = rows[s]["problem"]
        L += [
            f"  {REGIME_LABEL[s][0].replace(chr(10), '  ')}"
            f"   [{REGIME_LABEL[s][1].replace(chr(10), ' ')}]",
            f"    {p['n_problems']} problems over {p['n_trajectories']} trajectories",
            f"    {'method':10s} {'mean':>8s}  {'95% CI (clustered)':>22s}  "
            f"{'naive ±':>9s}  {'inflation':>9s}",
        ]
        for m in ORDER:
            v = rows[s].get(m)
            if not isinstance(v, dict) or v.get("mean") is None:
                continue
            ci = v.get("ci95_cluster")
            cis = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "n/a"
            nai = v.get("ci95_naive_halfwidth")
            inf = v.get("cluster_inflation")
            L.append(
                f"    {m:10s} {v['mean']:8.4f}  {cis:>22s}  "
                f"{(f'{nai:.4f}' if nai else 'n/a'):>9s}  "
                f"{(f'{inf:.2f}x' if inf else 'n/a'):>9s}"
            )
        L.append("")
    L += [
        "gt_extrap is the regime every learning rate and guidance strength was tuned on.",
        "It is reported as an out-of-training-range STRESS TEST, never as a clean test set.",
    ]
    return "\n".join(L)


def fig_window_scaling(out_dir=TRA, axes=None):
    """W-scaling: what each control can represent, and what asking for it costs.

    Left: the fraction of a W-frame window each method's control can express.  Flat at 1
    for the KAE and the samplers, and 1/W for U-Net and FNO, whose control is the single
    frame at t_0 -- the Q1 claim, now measured across W rather than asserted at W = 8.

    Right: the analysis error at t_0 itself against W.  This is the part that is not free.
    A 4D-Var solve does not change when a wider window is requested, so the KAE, U-Net and
    FNO are flat by construction.  The score-based methods sample a LONGER TRAJECTORY, with
    more blanket segments to compose, so their accuracy at t_0 degrades as W grows: the
    window is bought with accuracy at the analysis time.
    """
    d = json.loads(
        require(
            Path(out_dir) / "G21_window_scaling.json",
            "python -m data_assimilation.tra.exp_tra --sections window_scaling",
        ).read_text()
    )
    rows = d["rows"]
    Ws = [r["W"] for r in rows]
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    a0, a1 = np.asarray(axes).ravel()[:2]
    for m in ORDER:
        fr = [(r.get(m) or {}).get("fraction_representable") for r in rows]
        if not any(f is not None for f in fr):
            continue
        st = {k: v for k, v in STYLE[m].items() if k != "marker"}
        a0.plot(Ws, fr, lw=1.9, marker=STYLE[m]["marker"], ms=5.5, **st)
        c = [_ci((r.get(m) or {}).get("analysis_t0")) for r in rows]
        xs = [w for w, cc in zip(Ws, c) if cc]
        mu = [cc[0] for cc in c if cc]
        lo = [cc[0] - cc[1] for cc in c if cc]
        hi = [cc[2] - cc[0] for cc in c if cc]
        if xs:
            a1.errorbar(
                xs,
                mu,
                yerr=[lo, hi],
                lw=1.9,
                ms=5.5,
                capsize=3,
                marker=STYLE[m]["marker"],
                color=STYLE[m]["color"],
            )
    a0.axhline(1.0, color="0.5", ls=":", lw=1)
    a0.set(
        xscale="log",
        xlabel="window requested, $W$ frames",
        ylabel="fraction of the window the control can represent",
        title="Representability against $W$\n(structural: 1 for KAE and the samplers, "
        "$1/W$ for U-Net and FNO)",
    )
    a0.set_xticks(Ws)
    a0.set_xticklabels([str(w) for w in Ws])
    a1.set(
        xscale="log",
        yscale="log",
        xlabel="window requested, $W$ frames",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title="What the window COSTS at $t_0$\n4D-Var flat by construction; the "
        "samplers pay for it",
    )
    a1.set_xticks(Ws)
    a1.set_xticklabels([str(w) for w in Ws])
    for a in (a0, a1):
        a.grid(alpha=0.3, which="both")
    a0.legend(fontsize=7.5)
    return axes


# ---------------------------------------------------------------------------
# The campaign run separately on each tra regime.
# ---------------------------------------------------------------------------
REGIME_DIRS = {
    "interp": (
        TRA,
        "gt_interp · Ma 0.66–0.68",
        "in the training Mach gap, but an excerpt of the KAE's val split",
    ),
    "longer": (
        Path("da_results_tra_longer"),
        "gt_longer · Ma 0.64–0.65",
        "in the gap, cut from test.nc — clean for every model",
    ),
    "extrap": (
        Path("da_results_tra_extrap"),
        "gt_extrap · Ma 0.50–0.52",
        "OUTSIDE the training range — and the TUNING regime; a stress test",
    ),
}


def available_regimes(require_stem: str = "A_headline") -> list[str]:
    """Which per-regime campaigns have produced a given result yet."""
    return [
        k
        for k, (d, _, _) in REGIME_DIRS.items()
        if (d / f"{require_stem}.json").is_file()
    ]


def fig_across_regimes(
    fn, *, stem="A_headline", regimes=None, figsize_each=(6.2, 4.3), **kw
):
    """Draw the same figure once per regime, side by side, on a shared row.

    ``fn`` is any plots function taking ``out_dir`` and ``ax``.  Regimes whose campaign has
    not produced ``stem`` yet are skipped rather than raising, so this is usable while the
    later campaigns are still running.
    """
    regs = regimes or available_regimes(stem)
    if not regs:
        raise NotRunYet(
            f"no regime has {stem} yet.\n    produce it with: "
            f"./run_tra_campaign.sh <regime> <outdir>"
        )
    _, axes = plt.subplots(
        1,
        len(regs),
        squeeze=False,
        figsize=(figsize_each[0] * len(regs), figsize_each[1]),
    )
    axes = np.atleast_2d(axes).ravel()
    for ax, r in zip(axes, regs):
        d, label, role = REGIME_DIRS[r]
        try:
            fn(d, ax=ax, **kw)
        except NotRunYet:
            ax.text(
                0.5,
                0.5,
                f"{label}\nnot on disk yet",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="crimson",
            )
            ax.set_axis_off()
            continue
        ax.set_title(f"{label}\n{role}", fontsize=9)
    return axes


def fig_headline_regimes(ax=None, regimes=None):
    """The headline on every regime the campaign has been run on, as grouped bars.

    This is the whole point of running the campaign more than once: an ordering that only
    holds on the regime it was discovered on is not a finding.  `gt_extrap` is drawn
    hatched because it is the TUNING regime -- an out-of-training-range stress test, never
    a clean test set.
    """
    regs = regimes or available_regimes("A_headline")
    if not regs:
        raise NotRunYet("no per-regime campaign has finished its headline yet")
    ax = ax or plt.subplots(figsize=(9.8, 5.0))[1]
    ms, data = [], {}
    for r in regs:
        d = REGIME_DIRS[r][0]
        try:
            data[r] = summary(d)
        except NotRunYet:
            continue
        ms = [m for m in ORDER if m in data[r]] or ms
    regs = [r for r in regs if r in data]
    x = np.arange(len(regs))
    w = 0.8 / max(len(ms), 1)
    for i, m in enumerate(ms):
        xs = x + (i - (len(ms) - 1) / 2) * w
        mu = [data[r].get(m, {}).get("mean", np.nan) for r in regs]
        se = [data[r].get(m, {}).get("sem", 0.0) or 0.0 for r in regs]
        # the tuning regime is hatched; matplotlib takes one hatch per bar() call, so the
        # groups are drawn separately rather than with a per-bar list
        for j, (xi, r) in enumerate(zip(xs, regs)):
            ax.bar(
                [xi],
                [mu[j]],
                w,
                yerr=[se[j]],
                capsize=3,
                color=STYLE[m]["color"],
                hatch="//" if r == "extrap" else "",
                edgecolor="0.25" if r == "extrap" else "none",
                label=STYLE[m]["label"] if j == 0 else None,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{REGIME_DIRS[r][1]}\n{REGIME_DIRS[r][2]}" for r in regs], fontsize=8
    )
    ax.set(
        yscale="log",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title="The headline, campaign re-run per regime\n"
        "hatched = the tuning regime (stress test, not a clean test set)",
    )
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=7.5, ncol=2)
    return ax


def regime_campaign_table() -> str:
    """Which sections each per-regime campaign has produced, and the headline ordering."""
    stems = [
        "A_headline",
        "A_headline_diffusion",
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
        "G19_misspec",
        "G20_joint_sparsity",
    ]
    L = ["PER-REGIME CAMPAIGNS", ""]
    for r, (d, label, role) in REGIME_DIRS.items():
        have = sum(
            1
            for s in stems
            if (d / f"{s}.json").is_file() or (d / f"{s}.npz").is_file()
        )
        L.append(f"  {label:28s} {have:2d}/{len(stems)} sections   [{role}]")
        L.append(f"      dir: {d}")
        try:
            s = summary(d)
            order = sorted(
                (v["mean"], m)
                for m, v in s.items()
                if isinstance(v, dict) and v.get("mean") is not None
            )
            L.append(
                "      headline ordering: "
                + "  <  ".join(f"{m} {v:.4f}" for v, m in order)
            )
        except NotRunYet:
            L.append("      headline: not on disk yet")
        L.append("")
    L += [
        "Hyper-parameters are FROZEN across all three: tuning.json and",
        "tuning_diffusion.json are copied in, never recomputed, so the regimes stay",
        "comparable. gt_extrap is where they were chosen -- report it as a stress test.",
    ]
    return "\n".join(L)


# ---------------------------------------------------------------------------
def win_loss_table(out_dir=TRA, a="KAE", b="ACDM") -> str:
    """Who actually wins, experiment by experiment.

    The headline is ONE point of the design space, and it happens to sit inside the
    corner where the sampler wins. Reading only the headline gives the impression that
    ACDM dominates; reading only the sweeps gives the impression that the KAE does.
    Neither is true, and the deciding variable is visible only when the whole record is
    laid out: `delta_f`, the lead to the FIRST observation.
    """

    def J(s):
        f = Path(out_dir) / f"{s}.json"
        return json.loads(f.read_text()) if f.is_file() else None

    def Z(s):
        f = Path(out_dir) / f"{s}.npz"
        return np.load(f, allow_pickle=True) if f.is_file() else None

    rows = []
    hl = {**(J("A_headline") or {}), **(J("A_headline_diffusion") or {})}
    if a in hl and b in hl:
        rows.append(("headline, canonical schedule", hl[a]["mean"], hl[b]["mean"], 1))
    picks = [
        ("G1_delta_f", "delta_f = 1", lambda r: r["value"] == 1, 1),
        ("G1_delta_f", "delta_f = 4", lambda r: r["value"] == 4, 4),
        ("G1_delta_f", "delta_f = 18", lambda r: r["value"] == 18, 18),
        ("G3_sparsity", "2% of sensors", lambda r: abs(r["value"] - 0.02) < 1e-9, 1),
        ("G3_sparsity", "0.5% of sensors", lambda r: abs(r["value"] - 0.005) < 1e-9, 1),
        ("G2_noise", "noise sigma = 0.3", lambda r: abs(r["value"] - 0.3) < 1e-9, 1),
        ("G6_single_obs", "one obs at tau = 1", lambda r: r["value"] == 1, 1),
        ("G6_single_obs", "one obs at tau = 25", lambda r: r["value"] == 25, 25),
        ("G4_delta_l", "horizon 50", lambda r: r["value"] == 50, 1),
    ]
    for stem, lab, pred, df in picks:
        d = J(stem)
        if not d:
            continue
        r = next((x for x in d["rows"] if pred(x)), None)
        if r and isinstance(r.get(a), dict) and isinstance(r.get(b), dict):
            rows.append((lab, _num(r[a]), _num(r[b]), df))
    bk = J("G13_backward")
    if bk:
        m = {x["method"]: x for x in bk["rows"]}
        if a in m and b in m:
            rows.append(
                ("recover t0-7, one solve", m[a]["err"][-1], m[b]["err"][-1], 1)
            )
    rc = J("G18_reach")
    if rc:
        r = next((x for x in rc["rows"] if x["frames_back"] == 8), None)
        if r:
            rows.append(("recover t0-8, re-solved", _num(r[a]), _num(r[b]), 1))
    ws = J("G21_window_scaling")
    if ws:
        for W in (1, 16):
            r = next((x for x in ws["rows"] if x["W"] == W), None)
            if r:
                rows.append(
                    (
                        f"analysis at t0, window W={W}",
                        _num(r[a].get("analysis_t0")),
                        _num(r[b].get("analysis_t0")),
                        1,
                    )
                )
    pd_ = Z("G10_post_da")
    if pd_ is not None and f"{a}__rel_mean" in pd_.files:
        rows.append(
            (
                "forecast 50 frames after DA",
                float(pd_[f"{a}__rel_mean"][-1]),
                float(pd_[f"{b}__rel_mean"][-1]),
                1,
            )
        )
    og = J("G12_offgrid")
    if og:
        r = next((x for x in og["rows"] if abs(x["shift"] - 0.5) < 1e-9), None)
        if r:
            rows.append(("obs half a frame off-grid", _num(r[a]), _num(r[b]), 1))

    L = [
        f"WHO WINS, EXPERIMENT BY EXPERIMENT   ({a} vs {b})",
        "",
        f"{'experiment':32s} {'df':>3s} {a:>9s} {b:>9s}   {'winner':>6s}  margin",
    ]
    wa = wb = 0
    for lab, va, vb, df in rows:
        if not (np.isfinite(va) and np.isfinite(vb)):
            continue
        w = a if va < vb else b
        wa += va < vb
        wb += vb < va
        L.append(
            f"{lab:32s} {df:3d} {va:9.4f} {vb:9.4f}   {w:>6s}  "
            f"{max(va, vb) / max(min(va, vb), 1e-12):4.1f}x"
        )
    L += [
        "",
        f"  {a} wins {wa}   ·   {b} wins {wb}",
        "",
        f"  Sort by the `df` column ({b}'s deciding variable) and the record separates",
        f"  cleanly: every {b} win has its first observation at frame 1.",
        "  The canonical schedule is [1, 3, 7, 15, 25] -- delta_f = 1 -- so the",
        f"  HEADLINE PROBLEM SITS INSIDE {b}'s WINNING CORNER.",
    ]
    return "\n".join(L)


# ---------------------------------------------------------------------------
def fig_gap(out_dir=TRA, ax=None):
    """Error against the GAP to the nearest observation, with random irregular schedules.

    Every other experiment fixes the observation times at `canonical()`, so none of them
    can separate a method's quality from the schedule it was measured on.  Here the times
    are redrawn at random per replicate with only the nearest one pinned, so the schedule
    is averaged over rather than held constant.  The shaded band is the spread ACROSS
    schedules: a method whose result depends on which observations it happened to get has
    a wide band, and ACDM's is widest exactly where it wins.
    """
    d = json.loads(
        require(
            Path(out_dir) / "G22_gap.json",
            "python -m data_assimilation.tra.exp_tra --sections gap",
        ).read_text()
    )
    rows = d["rows"]
    xs = [r["gap"] for r in rows]
    ax = ax or plt.subplots(figsize=(7.4, 4.6))[1]
    for m in ORDER:
        if not any(isinstance(r.get(m), dict) for r in rows):
            continue
        mu = np.array([_num(r.get(m)) for r in rows], dtype=float)
        if not np.isfinite(mu).any():
            continue
        per = [
            np.asarray((r.get(m) or {}).get("per_schedule_means", []), dtype=float)
            for r in rows
        ]
        lo = np.array([p.min() if p.size else np.nan for p in per])
        hi = np.array([p.max() if p.size else np.nan for p in per])
        ax.plot(xs, mu, lw=1.9, ms=5.5, **STYLE[m])
        ax.fill_between(xs, lo, hi, color=STYLE[m]["color"], alpha=0.16)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="gap from the target to the NEAREST observation (frames)",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title=(
            "Recovery against the gap, with random irregular schedules\n"
            f"band = spread over {rows[0].get('n_schedules', '?')} independent "
            "observation sets"
        ),
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    return ax


def fig_confounded(out_dir=TRA, ax=None):
    """The confounded sweep laid beside the separated ones it is a mixture of.

    Observing every k-th frame moves delta_f, delta_l and N together.  The resulting curve
    looks like an observation-count effect.  Plotted against the SEPARATED sweeps -- G1,
    which moves delta_f alone, and G5, which moves N alone -- it is visibly the delta_f
    curve: ACDM is flat in N and steep in delta_f, and the confounded curve is steep.
    """
    d = json.loads(
        require(
            Path(out_dir) / "G23_confounded.json",
            "python -m data_assimilation.tra.exp_tra --sections confounded",
        ).read_text()
    )
    rows = d["rows"]
    ax = ax or plt.subplots(figsize=(7.6, 4.6))[1]
    for m in ("ACDM", "KAE"):
        df = [r.get("delta_f", r["value"]) for r in rows]
        mu = [_num(r.get(m)) for r in rows]
        ax.plot(
            df,
            mu,
            lw=2.2,
            marker="o",
            ms=6,
            color=STYLE[m]["color"],
            label=f"{m} — CONFOUNDED (every k-th frame)",
        )
        try:
            g1 = load_sweep(out_dir, "G1_delta_f")["rows"]
            ax.plot(
                [r["value"] for r in g1],
                [_num(r.get(m)) for r in g1],
                lw=1.6,
                ls="--",
                marker="s",
                ms=4,
                color=STYLE[m]["color"],
                alpha=0.75,
                label=f"{m} — $\\delta_f$ alone (G1)",
            )
        except NotRunYet:
            pass
        try:
            g5d = load_sweep(out_dir, "G5_n_obs")
            g5, p5 = g5d["rows"], pinned_delta_f(g5d["meta"])
            # G5 varies N with delta_f pinned; draw it at its own delta_f for contrast
            ax.plot(
                [p5] * len(g5),
                [_num(r.get(m)) for r in g5],
                ls="none",
                marker="x",
                ms=8,
                mew=1.8,
                color=STYLE[m]["color"],
                alpha=0.9,
                label=f"{m} — $N$ alone (G5), all at $\\delta_f={p5}$",
            )
        except NotRunYet:
            pass
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"$\delta_f$ — lead to the first observation (frames)",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title=(
            "The confounded sweep is the $\\delta_f$ sweep in disguise\n"
            "crosses: varying $N$ alone moves ACDM almost not at all"
        ),
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    return ax


def fig_convergence_check(out_dir=TRA, ax=None):
    """Analysis error against iteration: is the baselines' answer an unconverged optimiser?

    The claim that U-Net's analysis is a property of the COST rather than of the budget
    needs the optimiser watched, not inferred.  The diagnostic quantity is not whether the
    curve flattens but whether it turns UP: a cost that keeps falling while the analysis
    gets worse is the signature of an objective whose minimum is in the wrong place.
    """
    z = np.load(
        require(
            Path(out_dir) / "G24_convergence.npz",
            "python -m data_assimilation.tra.exp_tra --sections convergence",
        ),
        allow_pickle=True,
    )
    ax = ax or plt.subplots(figsize=(7.4, 4.6))[1]
    for m in ORDER:
        k = f"{m}__rel_t0"
        if k not in z.files:
            continue
        it, rel = z[f"{m}__iter"], z[k]
        st = {kk: v for kk, v in STYLE[m].items() if kk != "marker"}
        ax.plot(it, rel, lw=1.9, **st)
        j = int(np.asarray(rel).argmin())
        if j < len(rel) - 1:  # the minimum is not at the end
            ax.plot(it[j], rel[j], marker="v", ms=10, color=STYLE[m]["color"])
            ax.annotate(
                f"best {rel[j]:.4f} at {int(it[j])},\nthen DEGRADES",
                (it[j], rel[j]),
                textcoords="offset points",
                xytext=(10, -22),
                fontsize=7.5,
                color=STYLE[m]["color"],
            )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="4D-Var iteration",
        ylabel=r"analysis rel-$L_2$ at $t_0$",
        title=(
            "Convergence of the 4D-Var methods\n"
            "a curve that turns UP is an objective minimised in the wrong place"
        ),
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    return ax


def lyapunov_table(out_dir=None) -> str:
    """The predictability timescale per regime, and the campaign's horizons in its units."""
    L = [
        "PREDICTABILITY TIMESCALE  (Rosenstein on the stored trajectories)",
        "",
        "The transonic solver is not in this repository, so the twin experiment the KS",
        "campaign used is impossible; running it on a surrogate would measure the",
        "surrogate. A data-driven exponent is a LOWER bound at coarse sampling.",
        "",
        f"{'regime':28s} {'lambda_1/frame':>15s} {'T_L (frames)':>13s} {'R^2':>6s} {'pairs':>6s}",
    ]
    hz = None
    for key, (d, label, _role) in REGIME_DIRS.items():
        f = d / "lyapunov.json"
        if not f.is_file():
            continue
        j = json.loads(f.read_text())
        hz = hz or j["campaign_horizons_in_lyapunov_times"]
        L.append(
            f"{label.replace(chr(10), ' '):28s} {j['lambda_1_per_frame']:15.4f} "
            f"{j['lyapunov_time_frames']:13.1f} {j['fit_r2']:6.3f} {j['n_pairs']:6d}"
        )
    if hz:
        L += [
            "",
            "the campaign's horizons in Lyapunov times (gt_longer, the shortest T_L):",
        ]
        for k, v in hz.items():
            L.append(f"    {k:30s} {v:6.2f} T_L")
        L += [
            "",
            "EVERY horizon in this study is below one Lyapunov time. The KS campaign",
            "reached 3.1 T_L. This is a short-horizon regime, where chaos has not yet",
            "destroyed the information the observations carry.",
        ]
    return "\n".join(L)
