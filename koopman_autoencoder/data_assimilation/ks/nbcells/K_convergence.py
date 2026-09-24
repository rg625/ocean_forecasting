# ruff: noqa: F821
# mypy: disable-error-code="name-defined"
bc = (
    json.loads(
        (
            RESULTS.parent
            / "koopman_autoencoder"
            / "da_results_sda_paper"
            / "budget_convergence.json"
        ).read_text()
    )
    if False
    else json.loads(Path("da_results_sda_paper/budget_convergence.json").read_text())
)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
for m, key in [("KAE-expm", "KAE-expm"), ("UNet", "UNet")]:
    d = bc[key]
    s = STYLE[m]
    it = np.array([r["iters"] for r in d])
    v = np.array([r["rel"] for r in d])
    ax[0].loglog(
        it,
        v,
        marker=s["marker"],
        ls=s["ls"],
        color=s["color"],
        lw=2,
        ms=6,
        label=f"{s['short']}   rel-$L_2 \\propto$ iters$^{{{bc[key+'_fit']['loglog_slope']:.2f}}}$",
    )
    fit = bc[key + "_fit"]["loglog_slope"]
    ax[0].loglog(it, v[0] * (it / it[0]) ** fit, ":", color=s["color"], lw=1, alpha=0.6)
    g = v[:-1] / v[1:]
    ax[1].semilogx(
        it[1:],
        g,
        marker=s["marker"],
        ls=s["ls"],
        color=s["color"],
        lw=2,
        ms=6,
        label=s["short"],
    )
for a, xl, yl, t in [
    (
        ax[0],
        "optimisation iterations",
        "analysis rel-$L_2$",
        "(a) no plateau: a power law over two decades",
    ),
    (
        ax[1],
        "optimisation iterations",
        "gain per doubling",
        "(b) gain per doubling stays far above 1.05",
    ),
]:
    a.set_xlabel(xl)
    a.set_ylabel(yl)
    a.set_title(t)
    a.legend(fontsize=8)
ax[1].axhline(1.05, color="0.35", ls="--", lw=1.3, label="1.05$\\times$ = converged")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
