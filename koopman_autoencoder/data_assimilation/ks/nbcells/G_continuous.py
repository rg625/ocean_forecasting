# ruff: noqa: F821
# mypy: disable-error-code="name-defined"
G = L("G_continuous")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))

ax[0].plot(
    G["cont_tau"],
    G["cont_val"],
    color=STYLE["KAE-expm"]["color"],
    lw=2.2,
    label="exact $e^{K\\tau}$ — continuous in $\\tau$",
)
ax[0].plot(
    G["disc_tau"],
    G["disc_val"],
    "s",
    color=STYLE["KAE-rk4"]["color"],
    ms=5,
    label="KAE / RK4 — grid $\\Delta t$ only",
)
if "disc_val_unet" in G.files:
    ax[0].plot(
        G["disc_tau"],
        G["disc_val_unet"],
        "^",
        color=STYLE["UNet"]["color"],
        ms=5,
        label="U-Net — grid $\\Delta t$ only",
    )
ax[0].plot(G["true_tau"], G["true_val"], "x", color="k", ms=6, label="ground truth")
ax[0].set_xlabel("elapsed time $\\tau$")
ax[0].set_ylabel(f"$u$ at $x$={float(G['probe_x']):.2f}")
ax[0].set_title("(a) propagation is continuous in $\\tau$")
ax[0].legend(fontsize=8)

ms = avail(G, "rel_irregular")
w = 0.36
xs = np.arange(len(ms))
for kk, (tag, off, col) in enumerate(
    [("irregular", -w / 2, "#4393c3"), ("uniform", w / 2, "#f4a582")]
):
    mu = [G[f"{m}__rel_{tag}"].mean() for m in ms]
    sd = [G[f"{m}__rel_{tag}"].std() for m in ms]
    ax[1].bar(
        xs + off,
        mu,
        w,
        yerr=sd,
        capsize=4,
        alpha=0.9,
        color=col,
        edgecolor="0.3",
        label=f"{tag}  {list(G[('irr' if tag == 'irregular' else 'uni') + '_offsets'])}",
    )
ax[1].set_xticks(xs)
ax[1].set_xticklabels([STYLE[m]["short"] for m in ms])
ax[1].set_yscale("log")
ax[1].set_ylabel("field rel-$L_2$ @ $t_0$")
ax[1].set_title("(b) irregular vs uniform sampling, same budget")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

tab = pd.DataFrame(
    {
        tag: [G[f"{m}__rel_{tag}"].mean() for m in ms]
        for tag in ["irregular", "uniform"]
    },
    index=[STYLE[m]["short"] for m in ms],
)
tab["irregular / uniform"] = tab["irregular"] / tab["uniform"]
display(tab.style.format("{:.4f}"))
print(
    "Irregular sampling helps every method: it is an information effect (short and long"
)
print("lags in one budget), not an advantage specific to any one model.")
