# ruff: noqa: F821
# mypy: disable-error-code="name-defined"
def gallery(npz, title, sensors=False):
    d = L(npz)
    x = np.asarray(d["x"])
    true = d["true"]
    ms = avail(d, "recon")
    ref = "KAE-expm" if "KAE-expm" in ms else ms[0]
    order = np.argsort(d[f"{ref}__rel"])
    n = true.shape[0]
    cols = 3 if sensors else 4
    rows_ = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows_, cols, figsize=(4.6 * cols, 2.9 * rows_), sharex=True)
    ax = np.atleast_1d(ax).ravel()
    for j, idx in enumerate(order):
        # wide translucent underlay, so methods that match the truth stay visible
        ax[j].plot(
            x,
            true[idx],
            lw=6.0,
            color="k",
            alpha=0.22,
            solid_capstyle="round",
            zorder=1,
        )
        for zi, m in enumerate(ms):
            s = STYLE[m]
            ax[j].plot(
                x,
                d[f"{m}__recon"][idx],
                ls=s["ls"],
                lw=1.5,
                color=s["color"],
                zorder=3 + zi,
            )
            if present(d, m, "spread"):
                sd = d[f"{m}__spread"][idx]
                ax[j].fill_between(
                    x,
                    d[f"{m}__recon"][idx] - 1.96 * sd,
                    d[f"{m}__recon"][idx] + 1.96 * sd,
                    color=s["color"],
                    alpha=0.15,
                    lw=0,
                )
        if sensors:
            ax[j].scatter(
                x[d["obs_x"]], true[idx][d["obs_x"]], color="k", s=16, zorder=9
            )
        lim = truth_ylim(true[idx])
        ax[j].set_ylim(lim)
        note_clipped(
            ax[j], [(STYLE[m]["short"], d[f"{m}__recon"][idx]) for m in ms], lim
        )
        ax[j].set_title("  ".join(f"{d[f'{m}__rel'][idx]:.3f}" for m in ms), fontsize=8)
    for j in range(n, len(ax)):
        ax[j].axis("off")
    handles = [plt.Line2D([], [], color="k", lw=5, alpha=0.3, label="true $u(t_0)$")]
    handles += [
        plt.Line2D(
            [],
            [],
            color=STYLE[m]["color"],
            ls=STYLE[m]["ls"],
            lw=1.6,
            label=STYLE[m]["short"],
        )
        for m in ms
    ]
    if sensors:
        handles += [
            plt.Line2D(
                [], [], color="k", marker="o", ls="", ms=5, label="sensor locations"
            )
        ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    means = "   ".join(f"{STYLE[m]['short']} {d[f'{m}__rel'].mean():.3f}" for m in ms)
    fig.suptitle(
        f"{title}\npanel titles are rel-$L_2$ in method order;   means:  {means}",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
    return d


_ = gallery("H_gallery", "Recovered vs true initial state — independent trajectories")
