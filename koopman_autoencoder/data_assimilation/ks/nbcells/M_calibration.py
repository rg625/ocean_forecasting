# ruff: noqa: F821
# mypy: disable-error-code="name-defined"
cal = ca["calibration"]
q = cal["nominal_vs_empirical"]
nom = np.array(sorted(float(k) for k in q))
emp = np.array([100 * q[str(int(n))] for n in nom])

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))

ax[0].plot([0, 100], [0, 100], "k--", lw=1.4, label="perfect calibration")
ax[0].plot(
    nom,
    emp,
    marker="D",
    color=STYLE["SDA"]["color"],
    lw=2.2,
    ms=8,
    label=f"SDA, {cal['n_samples']} posterior draws",
)
for n_, e_ in zip(nom, emp):
    ax[0].annotate(
        f"{e_:.1f}%", (n_, e_), fontsize=8, xytext=(6, -10), textcoords="offset points"
    )
ax[0].set_xlim(40, 100)
ax[0].set_ylim(40, 100)
ax[0].set_xlabel("nominal credible level (%)")
ax[0].set_ylabel("empirical coverage (%)")
ax[0].set_title("(a) calibration of the trajectory posterior")
ax[0].legend(fontsize=9)

dev = emp - nom
ax[1].bar(
    [f"{int(n)}%" for n in nom],
    dev,
    color=["#4393c3" if d >= 0 else "#d6604d" for d in dev],
    edgecolor="0.3",
    alpha=0.9,
)
ax[1].axhline(0, color="k", lw=1)
ax[1].set_ylabel("empirical $-$ nominal (percentage points)")
ax[1].set_title("(b) over- (positive) or under-confidence (negative)")
for i, d in enumerate(dev):
    ax[1].text(i, d + (0.6 if d >= 0 else -1.4), f"{d:+.1f}", ha="center", fontsize=9)
plt.tight_layout()
plt.show()

print(
    f"Estimated from {cal['n_samples']} posterior draws on {cal['n_problems']} "
    f"held-out trajectories ({cal['wall_s']:.0f} s)."
)
