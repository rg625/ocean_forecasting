# ruff: noqa: F821
# mypy: disable-error-code="name-defined"
B = L("B_cost_vs_horizon")
h = B["horizon_t"]
has_sda = "ms_sda_per_sample" in B.files
N_ITERS, N_SAMPLES = 1000, int(summary.get("sda_n_samples", 8))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))

# (a) cost per unit of work -- units differ by family, which is why (c) exists
for key, m in [
    ("ms_exact", "KAE-expm"),
    ("ms_rollout", "KAE-rk4"),
    ("ms_unet", "UNet"),
]:
    s = STYLE[m]
    ax[0].plot(
        h,
        B[key],
        marker=s["marker"],
        ls=s["ls"],
        color=s["color"],
        lw=2,
        ms=6,
        label=s["label"] + "  — per optimisation step",
    )
if has_sda:
    sda = B["ms_sda_per_sample"]
    ok = ~np.isnan(sda)
    s = STYLE["SDA"]
    ax[0].plot(
        h[ok],
        sda[ok],
        marker=s["marker"],
        ls=":",
        color=s["color"],
        lw=2,
        ms=6,
        label=s["label"] + "  — per posterior sample",
    )
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel(r"assimilation horizon $\tau$")
ax[0].set_ylabel("ms per unit of work")
ax[0].set_title("(a) cost per unit of work\n(units are not interchangeable — see (c))")
ax[0].legend(fontsize=7)

# (b) the two speed-ups, deliberately kept apart
ax[1].plot(
    h,
    B["speedup"],
    marker="s",
    color=STYLE["KAE-rk4"]["color"],
    lw=2,
    ms=6,
    label="RK4 / exact — internal to the KAE",
)
ax[1].plot(
    h,
    B["speedup_unet"],
    marker="^",
    color=STYLE["UNet"]["color"],
    lw=2,
    ms=6,
    label="U-Net / exact — across methods",
)
# SDA has no optimisation step, so a per-step ratio does not exist for it. Its ratio is
# shown in the only unit where the comparison is meaningful: total cost to one finished
# assimilation (8 posterior samples against 1000 optimisation iterations).
if "speedup_sda_total" in B.files:
    ax[1].plot(
        h,
        B["speedup_sda_total"],
        marker="D",
        ls=":",
        color=STYLE["SDA"]["color"],
        lw=2,
        ms=6,
        label="SDA / exact — total cost to solution",
    )
ax[1].axhline(1.0, color="0.5", lw=1, ls="-", zorder=0)
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_xlabel(r"assimilation horizon $\tau$")
ax[1].set_ylabel(r"cost relative to exact $e^{K\tau}$")
ax[1].set_title("(b) relative cost — note the differing units")
ax[1].legend(fontsize=8)
for xi, yi, ns in zip(h, B["speedup_unet"], B["n_steps"]):
    ax[1].annotate(
        f"{int(ns)}",
        (xi, yi),
        fontsize=6.5,
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
    )

# (c) total cost to one finished assimilation -- the only like-for-like comparison
tot = {
    "KAE-expm": B["ms_exact"] * N_ITERS / 1e3,
    "KAE-rk4": B["ms_rollout"] * N_ITERS / 1e3,
    "UNet": B["ms_unet"] * N_ITERS / 1e3,
}
if has_sda:
    tot["SDA"] = B["ms_sda_per_sample"] * N_SAMPLES / 1e3
for m, v in tot.items():
    s = STYLE[m]
    ok = ~np.isnan(v)
    ax[2].plot(
        h[ok],
        v[ok],
        marker=s["marker"],
        ls=s["ls"],
        color=s["color"],
        lw=2,
        ms=6,
        label=s["short"],
    )
ax[2].set_xscale("log")
ax[2].set_yscale("log")
ax[2].set_xlabel(r"assimilation horizon $\tau$")
ax[2].set_ylabel("seconds to one finished assimilation")
ax[2].set_title(
    f"(c) total cost to solution\n{N_ITERS} iterations vs {N_SAMPLES} posterior samples"
)
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.show()

cols = {
    "tau": h,
    "KAE (s)": tot["KAE-expm"],
    "KAE/RK4 (s)": tot["KAE-rk4"],
    "U-Net (s)": tot["UNet"],
}
if has_sda:
    cols["SDA (s)"] = tot["SDA"]
    cols["SDA window L"] = B["sda_window_L"]
tab = pd.DataFrame(cols).set_index("tau")
print("Total wall-clock for ONE complete assimilation:")
display(
    tab.style.format("{:.1f}").background_gradient(
        cmap="RdYlGn_r", axis=1, subset=[c for c in tab.columns if c.endswith("(s)")]
    )
)
if has_sda:
    print(
        f"Score-based DA: {int(B['sda_sample_steps'])} predictor steps x "
        f"(1 + {int(B['sda_corrections'])} Langevin corrections) score evaluations per"
    )
    print(
        "sample, each a network forward AND a guidance backward pass, batched over the"
    )
    print(
        f"L-{int(B['sda_blanket']) - 1} blanket segments of the trajectory. The blanket is "
        f"{int(B['sda_blanket'])} frames and"
    )
    print(
        "Algorithm 2 composes it over any length, so the SAME trained model covers every"
    )
    print("row above -- no retraining, and no window to fall outside of.")
