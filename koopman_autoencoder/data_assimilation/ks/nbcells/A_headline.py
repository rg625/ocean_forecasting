# ruff: noqa: F821
# mypy: disable-error-code="name-defined"
A = L("A_headline")
x = np.asarray(A["x"]).squeeze()
offs = np.atleast_1d(A["offsets"])
dt = float(A["dt"])
floor = float(A["ae_floor"])
ms = avail(A, "rel_final")

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4))

# (a) how each method converges on the hidden state
for m in ms:
    s = STYLE[m]
    key = f"{m}__hist_init_rel_l2"
    if key in A.files and len(A[key]):
        ax[0].semilogy(
            A[f"{m}__hist_iter"],
            A[key],
            color=s["color"],
            ls=s["ls"],
            lw=2,
            label=s["label"],
        )
    else:
        ax[0].axhline(
            float(A[f"{m}__rel_final"]),
            color=s["color"],
            ls=":",
            lw=2.2,
            label=s["short"] + " — one posterior draw (sampler, not an optimiser)",
        )
ax[0].axhline(
    floor,
    color="0.35",
    ls=":",
    lw=1.5,
    label=f"AE round-trip ({floor:.3f}, not a bound)",
)
ax[0].set_xlabel("optimisation iteration")
ax[0].set_ylabel("field rel-$L_2$ @ $t_0$")
ax[0].set_title("(a) recovery of the unobserved state")


ax[1].plot(
    x,
    np.asarray(A["u_t0_true"]).reshape(-1),
    lw=6.5,
    color="k",
    alpha=0.22,
    solid_capstyle="round",
    label="true $u(t_0)$  (never observed)",
    zorder=1,
)
for m in ms:
    s = STYLE[m]
    rec = np.asarray(A[f"{m}__u_t0_recon"]).reshape(-1)
    ax[1].plot(
        x,
        rec,
        ls=s["ls"],
        lw=1.7,
        color=s["color"],
        zorder=3 + ms.index(m),
        label=f"{s['short']}  ({float(A[f'{m}__rel_final']):.3f})",
    )
    if present(A, m, "spread_t0"):
        sd = np.asarray(A[f"{m}__spread_t0"]).reshape(-1)
        ax[1].fill_between(
            x,
            rec - 1.96 * sd,
            rec + 1.96 * sd,
            color=s["color"],
            alpha=0.18,
            lw=0,
            label=f"{s['short']} 95% credible band",
        )
ax[1].set_xlabel("space $x$")
ax[1].set_ylabel("$u$")
ax[1].set_title("(b) recovered initial state  (rel-$L_2$ in legend)")
_tru = np.asarray(A["u_t0_true"]).reshape(-1)
_lim = truth_ylim(_tru)
ax[1].set_ylim(_lim)
note_clipped(
    ax[1],
    [(STYLE[m]["short"], np.asarray(A[f"{m}__u_t0_recon"]).reshape(-1)) for m in ms],
    _lim,
)

# (c) fit to the observations each method was actually given
for i in range(np.asarray(A["obs_true"]).shape[0]):
    ax[2].plot(
        x,
        np.asarray(A["obs_true"])[i],
        color="k",
        lw=4.5,
        alpha=0.18,
        solid_capstyle="round",
        zorder=1,
        label="observed truth" if i == 0 else None,
    )
for m in ms:
    op = np.asarray(A[f"{m}__obs_pred"])
    s = STYLE[m]
    for i in range(op.shape[0]):
        ax[2].plot(
            x,
            op[i],
            ls=s["ls"],
            color=s["color"],
            lw=1.0,
            alpha=0.85,
            label=s["short"] if i == 0 else None,
        )
ax[2].set_xlabel("space $x$")
ax[2].set_ylabel("$u$")
_limo = truth_ylim(np.asarray(A["obs_true"]))
ax[2].set_ylim(_limo)
note_clipped(
    ax[2], [(STYLE[m]["short"], np.asarray(A[f"{m}__obs_pred"])) for m in ms], _limo
)
ax[2].set_title(
    "(c) fit to the five future observations\n$\\tau$ = "
    + str([round(float(o) * dt, 1) for o in offs])
)
for a in ax:
    a.legend(fontsize=7.5)
plt.tight_layout()
plt.show()

# # SDA returns a distribution, so show the draws themselves rather than any summary of them.
# if any(present(A, m, "draws_t0") for m in ms):
#     fig, ax = plt.subplots(figsize=(9, 4.0))
#     tru = np.asarray(A["u_t0_true"]).reshape(-1)
#     ax.plot(x, tru, lw=6.5, color="k", alpha=.22, solid_capstyle="round",
#             label="true $u(t_0)$  (never observed)", zorder=1)
#     for m in ms:
#         if not present(A, m, "draws_t0"):
#             continue
#         dr = np.asarray(A[f"{m}__draws_t0"])
#         for j in range(dr.shape[0]):
#             ax.plot(x, dr[j], color=STYLE[m]["color"], lw=1.2, alpha=.75, zorder=3,
#                     label=f"{STYLE[m]['short']} posterior draws ({dr.shape[0]} shown)"
#                           if j == 0 else None)
#         if present(A, m, "rel_per_draw"):
#             pdw = np.asarray(A[f"{m}__rel_per_draw"])
#             ax.text(0.01, 0.02,
#                     "per-draw rel-$L_2$: " + ", ".join(f"{v:.4f}" for v in np.atleast_1d(pdw)),
#                     transform=ax.transAxes, fontsize=8, va="bottom",
#                     bbox=dict(fc="w", ec="0.8", alpha=.9, pad=2))
#     ax.set_ylim(truth_ylim(tru, pad=2.4))
#     ax.set_xlabel("space $x$"); ax.set_ylabel("$u$")
#     ax.set_title("Score-based DA returns a distribution: every line is one posterior draw")
#     ax.legend(fontsize=8, loc="upper right")
#     plt.tight_layout(); plt.show()
