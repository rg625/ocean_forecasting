# ruff: noqa: E741, F821
# mypy: disable-error-code="name-defined"
I = L("I_spacetime")
x = np.asarray(I["x"])
true = I["true"]
dt = float(I["dt"])
obs = np.atleast_1d(I["obs_off"])
T = int(I["T"])
ms = avail(I, "pred")
nex = true.shape[0]
ext = [0, T * dt, float(x.min()), float(x.max())]

for e in range(nex):
    tr = true[e]
    vmin, vmax = tr.min(), tr.max()
    # a shared error scale keeps the panels comparable, but a single outlier would
    # flatten every other map to black, so the ceiling is a high percentile rather than
    # the maximum. Values above it are clipped and the colourbar says so.
    _allerr = np.concatenate([np.abs(I[f"{m}__pred"][e] - tr).ravel() for m in ms])
    emax = float(np.nanpercentile(_allerr, 99.0))
    emax = max(emax, 1e-6)
    ncol = 1 + len(ms)
    fig, ax = plt.subplots(
        2, ncol, figsize=(3.5 * ncol, 5.6), gridspec_kw={"height_ratios": [1, 1]}
    )
    im = ax[0, 0].imshow(
        tr.T,
        origin="lower",
        aspect="auto",
        extent=ext,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax[0, 0].set_title("TRUTH", fontsize=10)
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.02)
    ax[1, 0].axis("off")
    for c, m in enumerate(ms, start=1):
        pr = I[f"{m}__pred"][e]
        im = ax[0, c].imshow(
            pr.T,
            origin="lower",
            aspect="auto",
            extent=ext,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax[0, c].set_title(STYLE[m]["short"], fontsize=10, color=STYLE[m]["color"])
        fig.colorbar(im, ax=ax[0, c], fraction=0.046, pad=0.02)
        im2 = ax[1, c].imshow(
            np.abs(pr - tr).T,
            origin="lower",
            aspect="auto",
            extent=ext,
            cmap="magma",
            vmin=0,
            vmax=emax,
        )
        ax[1, c].set_title(
            f"|error|   rel@$t_0$={I[f'{m}__rel_t0'][e]:.3f}", fontsize=9
        )
        cb = fig.colorbar(im2, ax=ax[1, c], fraction=0.046, pad=0.02, extend="max")
        cb.ax.tick_params(labelsize=7)
    for a in ax.ravel():
        if a.has_data():
            for o in obs:
                a.axvline(float(o) * dt, color="w", ls=":", lw=1.1)
            a.set_xlabel("time")
    ax[0, 0].set_ylabel("space $x$")
    ax[1, 1].set_ylabel("space $x$")
    fig.suptitle(
        f"Example {e + 1} — white dotted lines are the ONLY times observed; "
        "everything else is inferred",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
for m in ms:
    pf = I[f"{m}__per_frame_rel"]
    mu = np.nanmean(pf, axis=0)
    s = STYLE[m]
    ax[0].semilogy(
        np.arange(T) * dt, mu, ls=s["ls"], color=s["color"], lw=2, label=s["label"]
    )
    if pf.shape[0] > 1:
        ax[0].fill_between(
            np.arange(T) * dt,
            np.nanmin(pf, 0),
            np.nanmax(pf, 0),
            color=s["color"],
            alpha=0.13,
            lw=0,
        )
for o in obs:
    ax[0].axvline(float(o) * dt, color="0.55", ls=":", lw=1.1)
ax[0].set_xlabel("time within the window")
ax[0].set_ylabel("rel-$L_2$")
ax[0].set_title("(a) error across the window (dotted = observation times)")
ax[0].legend(fontsize=8)

obs_set = set(int(o) for o in obs)
unobs = np.array([i for i in range(T) if i not in obs_set])
xs = np.arange(len(ms))
w = 0.36
for kk, (lab, sel, col) in enumerate(
    [
        ("observed frames", np.array(sorted(obs_set)), "#4393c3"),
        ("never-observed frames", unobs, "#b2182b"),
    ]
):
    v = [np.nanmean(I[f"{m}__per_frame_rel"][:, sel]) for m in ms]
    ax[1].bar(
        xs + (kk - 0.5) * w, v, w, color=col, alpha=0.9, edgecolor="0.3", label=lab
    )
ax[1].set_xticks(xs)
ax[1].set_xticklabels([STYLE[m]["short"] for m in ms])
ax[1].set_yscale("log")
ax[1].set_ylabel("rel-$L_2$")
ax[1].set_title(
    f"(b) observed ({len(obs_set)}) vs never-observed ({len(unobs)}) frames"
)
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

display(
    pd.DataFrame(
        {
            "observed frames": [
                np.nanmean(I[f"{m}__per_frame_rel"][:, sorted(obs_set)]) for m in ms
            ],
            "never-observed frames": [
                np.nanmean(I[f"{m}__per_frame_rel"][:, unobs]) for m in ms
            ],
            "whole window": [np.nanmean(I[f"{m}__per_frame_rel"]) for m in ms],
        },
        index=[STYLE[m]["short"] for m in ms],
    )
    .style.format("{:.4f}")
    .background_gradient(cmap="RdYlGn_r", axis=0)
)
print(
    "The never-observed column is the one that matters: it asks whether conditioning on"
)
print("a handful of frames improves the states the method was never shown.")
