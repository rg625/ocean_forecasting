# ruff: noqa: F821
# mypy: disable-error-code="name-defined"
def sweep_plot(npz, xlabel, title, logx=False):
    d = L(npz)
    ms = avail(d, "rel_mean")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for m in ms:
        mu, sd, s = d[f"{m}__rel_mean"], d[f"{m}__rel_std"], STYLE[m]
        ok = ~np.isnan(mu)
        ax.errorbar(
            np.asarray(d["values"])[ok],
            mu[ok],
            yerr=sd[ok],
            marker=s["marker"],
            ls=s["ls"],
            lw=2,
            capsize=4,
            ms=6,
            color=s["color"],
            label=s["label"] + (" — one posterior draw" if m == "SDA" else ""),
        )
    # NOT a lower bound: it is the encode->decode round-trip error, and optimising z0
    # directly can and does beat it, because the encoder is not the decoder's optimal
    # inverse. Shown as a reference scale only.
    ax.plot(
        d["values"],
        d["ae_floor"],
        ":",
        color="0.35",
        lw=1.5,
        label="autoencoder round-trip error (reference, not a bound)",
    )
    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("field rel-$L_2$ @ $t_0$  (lower is better)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    # the same numbers as a table, so nothing has to be read off a log axis
    tab = pd.DataFrame(
        {STYLE[m]["short"]: d[f"{m}__rel_mean"] for m in ms},
        index=np.round(np.asarray(d["values"]), 4),
    )
    tab.index.name = xlabel
    display(tab.style.format("{:.4f}").background_gradient(cmap="RdYlGn_r", axis=1))
    return d


_ = sweep_plot(
    "C_nobs", "number of observations", "C. Recovery vs number of observations"
)
