# ruff: noqa: F821
# mypy: disable-error-code="name-defined"
F = L("F_statistics")
fl = F["ae_floor"]
ms = avail(F, "rel")

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4))

# (a) the distribution over trajectories, one violin per method
data = [F[f"{m}__rel"] for m in ms]
parts = ax[0].violinplot(data, showmedians=True, widths=0.82)
for pc, m in zip(parts["bodies"], ms):
    pc.set_facecolor(STYLE[m]["color"])
    pc.set_alpha(0.55)
    pc.set_edgecolor("0.3")
for key in ("cmedians", "cbars", "cmins", "cmaxes"):
    if key in parts:
        parts[key].set_color("0.3")
for i, m in enumerate(ms):
    ax[0].scatter(
        np.full(len(F[f"{m}__rel"]), i + 1)
        + np.random.uniform(-0.06, 0.06, len(F[f"{m}__rel"])),
        F[f"{m}__rel"],
        s=5,
        color=STYLE[m]["color"],
        alpha=0.45,
        zorder=3,
    )
ax[0].axhline(
    fl.mean(),
    color="0.35",
    ls=":",
    lw=1.5,
    label=f"AE round-trip {fl.mean():.3f} (not a bound)",
)
ax[0].set_xticks(range(1, len(ms) + 1))
ax[0].set_xticklabels([STYLE[m]["short"] for m in ms])
ax[0].set_yscale("log")
ax[0].set_ylabel("field rel-$L_2$ @ $t_0$")
ax[0].set_title(f"(a) distribution over {len(data[0])} independent trajectories")
ax[0].legend(fontsize=8)

# (b) exact vs RK4 parity -- the closed form costs nothing in accuracy
re_ex, re_ro = F["rel_exact"], F["rel_rollout"]
lim = [min(re_ex.min(), re_ro.min()) * 0.9, max(re_ex.max(), re_ro.max()) * 1.1]
ax[1].plot(lim, lim, "k:", alpha=0.7, label="$y=x$ (identical)")
ax[1].scatter(
    re_ex,
    re_ro,
    s=18,
    alpha=0.7,
    color=STYLE["KAE-expm"]["color"],
    edgecolor="w",
    linewidth=0.4,
)
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_xlim(lim)
ax[1].set_ylim(lim)
ax[1].set_xlabel("exact $e^{K\\tau}$ rel-$L_2$")
ax[1].set_ylabel("RK4 rollout rel-$L_2$")
ax[1].set_title(
    "(b) exact vs RK4: the speed-up is free\n"
    f"mean |difference| = {np.abs(re_ex - re_ro).mean():.2e}"
)
ax[1].legend(fontsize=8)

# (c) paired comparison against the reference method
ref = "KAE-expm"
others = [m for m in ms if m != ref]
xs = np.arange(len(others))
diff = [np.mean(F[f"{m}__rel"] - F[f"{ref}__rel"]) for m in others]
err = [
    np.std(F[f"{m}__rel"] - F[f"{ref}__rel"], ddof=1) / np.sqrt(len(F[f"{m}__rel"]))
    for m in others
]
ax[2].bar(
    xs,
    diff,
    0.55,
    yerr=err,
    capsize=5,
    color=[STYLE[m]["color"] for m in others],
    alpha=0.85,
    edgecolor="0.3",
)
ax[2].axhline(0, color="k", lw=1)
ax[2].set_xticks(xs)
ax[2].set_xticklabels([STYLE[m]["short"] for m in others])
ax[2].set_ylabel(f"mean(method $-$ {STYLE[ref]['short']})")
_sc = max(abs(np.array(diff)).min(), 1e-3)
ax[2].set_yscale("symlog", linthresh=_sc)
ax[2].set_title("(c) paired per-trajectory difference\n(below zero = better than KAE)")
plt.tight_layout()
plt.show()

rows = []
for m in ms:
    r = F[f"{m}__rel"]
    rows.append(
        {
            "method": STYLE[m]["short"],
            "mean": r.mean(),
            "median": np.median(r),
            "std": r.std(ddof=1),
            "min": r.min(),
            "max": r.max(),
            "better than KAE": f"{int((F[f'{m}__rel'] < F[f'{ref}__rel']).sum())}/{len(r)}",
        }
    )
display(
    pd.DataFrame(rows)
    .set_index("method")
    .style.format({c: "{:.4f}" for c in ["mean", "median", "std", "min", "max"]})
)
