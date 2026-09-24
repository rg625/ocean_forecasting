# ruff: noqa: F821
# mypy: disable-error-code="assignment, name-defined"
# Everything above, collected. Read straight from the saved .npz files.
rows = []
spec = [
    ("A  headline (1 traj)", "A_headline", "rel_final"),
    ("F  statistics (mean)", "F_statistics", "rel"),
    ("H  gallery (mean)", "H_gallery", "rel"),
    ("J  sparse sensors 20% (mean)", "J_sparse_recovery", "rel"),
]
for name, npz, key in spec:
    try:
        d = L(npz)
    except FileNotFoundError:
        continue
    row = {"experiment": name}
    for m in METHODS:
        row[STYLE[m]["short"]] = (
            float(np.mean(d[f"{m}__{key}"])) if present(d, m, key) else np.nan
        )
    rows.append(row)
try:
    d = L("I_spacetime")
    ms_ = avail(d, "per_frame_rel")
    obs_set = set(int(o) for o in np.atleast_1d(d["obs_off"]))
    unobs = np.array([i for i in range(int(d["T"])) if i not in obs_set])
    for lbl, sel in [
        ("I  space-time, observed frames", np.array(sorted(obs_set))),
        ("I  space-time, NEVER-observed", unobs),
    ]:
        row = {"experiment": lbl}
        for m in METHODS:
            row[STYLE[m]["short"]] = (
                float(np.nanmean(d[f"{m}__per_frame_rel"][:, sel]))
                if m in ms_
                else np.nan
            )
        rows.append(row)
except FileNotFoundError:
    pass

if rows:
    df = pd.DataFrame(rows).set_index("experiment")
    print("Analysis / trajectory error, relative L2 — lower is better\n")
    display(
        df.style.format("{:.4f}", na_rep="—").background_gradient(
            cmap="RdYlGn_r", axis=1
        )
    )

# what each method costs, and what it returns
try:
    B_ = L("B_cost_vs_horizon")
    i25 = int(np.argmin(np.abs(B_["horizon_t"] - 2.5)))
    N_ITERS = 1000
    N_SAMPLES = int(summary.get("sda_n_samples", 8))
    cost = {
        "KAE": B_["ms_exact"][i25] * N_ITERS / 1e3,
        "KAE/RK4": B_["ms_rollout"][i25] * N_ITERS / 1e3,
        "U-Net": B_["ms_unet"][i25] * N_ITERS / 1e3,
    }
    if "ms_sda_per_sample" in B_.files:
        cost["SDA"] = B_["ms_sda_per_sample"][i25] * N_SAMPLES / 1e3
    prof = pd.DataFrame(
        {
            "returns": {
                "KAE": "point estimate (latent $z_0$)",
                "KAE/RK4": "point estimate (latent $z_0$)",
                "U-Net": "point estimate (physical $x_0$)",
                "SDA": "posterior over whole trajectories",
            },
            "propagation": {
                "KAE": "one $e^{K\\tau_i}$ per obs. time",
                "KAE/RK4": "RK4 steps of $\\dot z = Kz$",
                "U-Net": "autoregressive $F_\\theta^n$",
                "SDA": "none — no physical model is simulated",
            },
            "arbitrary $\\tau$": {
                "KAE": "yes",
                "KAE/RK4": "no (grid $\\Delta t$)",
                "U-Net": "no (grid $\\Delta t$)",
                "SDA": "no (grid $\\Delta t$)",
            },
            "uncertainty": {
                "KAE": "none",
                "KAE/RK4": "none",
                "U-Net": "none",
                "SDA": "posterior spread + coverage",
            },
            "cost at $\\tau$=2.5 (s)": {k: f"{v:.1f}" for k, v in cost.items()},
        }
    )
    print("\nWhat each method is, and what it costs for one complete assimilation:\n")
    display(prof)
except FileNotFoundError:
    pass
