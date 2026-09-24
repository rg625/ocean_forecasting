"""Draw the delta_f break figure from whatever points are on disk (results flush per point)."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "KAE-expm": ("#1b7837", "KAE"),
    "KAE-rk4": ("#7fbc41", "KAE / RK4"),
    "UNet": ("#d6604d", "U-Net 4D-Var"),
    "SDA": ("#762a83", "SDA"),
}
d = json.loads(Path("da_results_geometry_df9/deltaf_break.json").read_text())
rows, meta = d["rows"], d["meta"]
x = [r["delta_f"] for r in rows]
fig, ax = plt.subplots(figsize=(7.6, 4.9))
for m, (col, lab) in STYLE.items():
    ax.errorbar(
        x,
        [r[m]["mean"] for r in rows],
        yerr=[r[m]["sem"] for r in rows],
        color=col,
        label=lab,
        lw=1.9,
        marker="o",
        ms=4.5,
        capsize=3,
    )
ax.plot(
    x,
    [r["copy_nearest"] for r in rows],
    color="0.45",
    ls="-.",
    lw=1.6,
    label="copy the nearest observation",
)
ax.axhline(1.0, color="crimson", ls=":", lw=1.6, label="climatology")
ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"$\delta_f$  (frames to the first observation)",
    ylabel=r"analysis rel-$L_2$ at $t_0$",
    title=f"Pushing $\\delta_f$ until assimilation breaks\nwindow after the first "
    f"observation fixed at {meta['span']} frames, $N=5$, {meta['iters']} iterations, "
    f"$n={meta['n_problems']}$ ({len(rows)} points on disk)",
)
ax.set_xticks(x)
ax.set_xticklabels([str(v) for v in x])
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figs/ks_deltaf_break.png", dpi=150, bbox_inches="tight")
print("wrote figs/ks_deltaf_break.png from", len(rows), "points:", x)
