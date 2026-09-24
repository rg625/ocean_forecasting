#!/usr/bin/env python
# ruff: noqa: E741
"""Assemble visualize_da_ks_3way.ipynb in the A-G structure asked for in the revision.

The notebook is generated rather than hand-edited so that its structure is reproducible
and so that locating a cell by a substring can never overwrite the wrong one again.
Existing, working cell bodies live in data_assimilation/ks/nbcells/*.py and are reused verbatim; only the
narrative and the new sections are written here.
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS = Path("data_assimilation/ks/nbcells")
NB = Path("visualize_da_ks_3way.ipynb")


def md(*lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in lines],
    }


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in src.rstrip("\n").split("\n")],
    }


def reuse(name):
    return code(CELLS.joinpath(f"{name}.py").read_text())


cells = []

# =====================================================================  TITLE
cells += [
    md(
        "# Data assimilation on Kuramoto–Sivashinsky",
        "## Continuous Koopman autoencoder · U-Net 4D-Var · score-based DA",
        "",
        "**The task.** We observe a KS trajectory at a few *future* times and must reconstruct the",
        "full state at an earlier time $t_0$ that was **never observed**.",
        "",
        "### Notation used throughout (this matters — the canonical setup conflates three things)",
        "",
        "| symbol | meaning |",
        "|---|---|",
        "| $t_0$ | the **analysis time**: the unobserved state we are asked to recover |",
        "| $T_\\mathrm{obs}=\\{t_1,\\dots,t_N\\}$ | the observation times, all $> t_0$ |",
        "| $\\delta_f = t_1 - t_0$ | lead time to the **first** observation |",
        "| $\\delta_l = t_N - t_0$ | lead time to the **last** observation — the *recovery horizon* |",
        "| $N$ | the **number** of observation times |",
        "",
        "Observations are **irregularly sampled**, so $\\delta_l \\neq (N-1)\\,\\delta_f$ in general.",
        "The canonical schedule used in Parts A, D and E is $[1,3,7,15,25]$ frames, i.e.",
        "$\\delta_f=0.1$, $\\delta_l=2.5$, $N=5$ **simultaneously** — one number from it cannot be",
        "attributed to any one of the three. Part C varies each in isolation.",
        "",
        "### The four methods",
        "",
        "| | control variable | propagation | nature |",
        "|---|---|---|---|",
        "| **Continuous KAE** | latent $z_0\\in\\mathbb{R}^{128}$ | $z(\\tau)=e^{K\\tau}z_0$, one matrix product | deterministic, continuous-time |",
        "| **KAE / RK4** | latent $z_0$ | the *same* generator integrated by the model's own RK4 | deterministic (numerical control) |",
        "| **U-Net 4D-Var** | physical $x_0\\in\\mathbb{R}^{64}$ | $x(n\\Delta t)=F_\\theta^{\\,n}(x_0)$, autoregressive | deterministic, discrete-time |",
        "| **Score-based DA** | the whole trajectory | reverse diffusion + observation likelihood | stochastic posterior |",
        "",
        "The first three minimise $\;\\min_c\\sum_i\\lVert M_i\\odot(\\mathcal G(c,\\tau_i)-y_i)\\rVert^2$",
        "with **all learned weights frozen**. The fourth samples $p(x_{0:T}\\mid y)$ following",
        "Rozet & Louppe (NeurIPS 2023).",
        "",
        "> **SDA is scored as a single posterior draw, not as the posterior mean.** The paper",
        "> treats the posterior as a distribution and never forms a mean; the mean is also the",
        "> optimal estimator under squared error, so scoring it against two point estimators",
        "> would flatter SDA by roughly $1.7\\times$ here. `rel_posterior_mean` is still recorded.",
        "",
        "**Fairness.** Every method in a given cell reads *the same* `Problem` object — the same",
        "trajectories, the same $t_0$, the same observation times, the same sensor mask and the",
        "same noise realisation, byte for byte.",
        "",
        "---",
        "### Contents",
        "**A.** Main DA comparison  ·  **B.** Models, training and reproducibility  ·",
        "**C.** Temporal geometry ($\\delta_f$, $\\delta_l$, $N$)  ·  **D.** Observation noise  ·",
        "**E.** Sparsity  ·  **F.** Space–time and spectral analysis  ·  **G.** Conclusions",
    ),
    reuse("setup"),
]

# =====================================================================  A
cells += [
    md(
        "---",
        "# A. Main data-assimilation comparison",
        "",
        "Canonical schedule $[1,3,7,15,25]$ frames ($\\delta_f=0.1$, $\\delta_l=2.5$, $N=5$),",
        "clean and fully observed. Both 4D-Var methods run at **8000 iterations**; Part A.4 shows",
        "what that budget does and does not buy.",
    ),
    md("## A.1 Headline recovery"),
    reuse("A_headline"),
    md(
        "## A.2 The same panel, with the observations pushed far into the future",
        "",
        "Panel (c) of A.1 shows the fit to five observations that all lie within",
        "$\\delta_l = 2.5$ time units — **0.11 Lyapunov times**. That is a very short baseline,",
        "and it is the only regime the headline number describes.",
        "",
        "Below, the same figure is redrawn at four baselines spanning nearly three orders of",
        "magnitude, out to $\\delta_l = 400$ t.u. $= 17.7\\,T_L$:",
        "",
        "| set | $\\tau$ (time units) | $\\tau/T_L$ |",
        "|---|---|---|",
        "| canonical | 0.1, 0.3, 0.7, 1.5, 2.5 | 0.00 – 0.11 |",
        "| medium | 0.1, 0.5, 2.5, 10, 25 | 0.00 – 1.11 |",
        "| long | 0.1, 1, 6, 30, 100 | 0.00 – 4.43 |",
        "| extreme | 0.1, 2.5, 20, 120, 400 | 0.00 – 17.7 |",
        "",
        "> Every model was trained on **10-frame (1.0 time unit) rollouts**, so even the",
        "> canonical set is already $2.5\\times$ the training horizon and the extreme set is",
        "> $400\\times$. These are extrapolation tests and are labelled as such.",
        "",
        "Methods that cannot be run at a given baseline are **left out and named**, never",
        "silently dropped: U-Net 4D-Var cost is linear in the rollout length and SDA's",
        "Algorithm-2 composition is linear in the trajectory length, so both hit a cost wall",
        "long before the KAE, whose $e^{K\\tau}$ is one matrix product regardless of $\\tau$.",
    ),
    code(
        """from data_assimilation.ks import plot_long as pl
LONG = Path("da_results_long")
axes = pl.fig_headline_long(LONG)
plt.tight_layout(); plt.show()"""
    ),
    md(
        "## A.3 Statistics over 64 held-out trajectories",
        "",
        "Mean $\\pm$ standard error, and the exact-vs-RK4 parity check that isolates the",
        "continuous-time propagator from the rest of the KAE.",
    ),
    reuse("F_statistics"),
    md("## A.4 Recovery gallery — does it work every time?"),
    reuse("H_gallery"),
    md(
        "## A.5 Are the 4D-Var baselines converged?",
        "",
        "**A negative result we are keeping.** Neither 4D-Var method plateaus: both follow a",
        "power law in the iteration count out to $1.3\\times10^5$ iterations. A '<1.05× per",
        "doubling' criterion would need $\\sim\\!10^6$ iterations. Every 4D-Var number in this",
        "notebook is therefore a number *at a stated budget*, not a converged optimum.",
    ),
    reuse("K_convergence"),
    md(
        "## A.6 Error against wall-clock time",
        "",
        "The honest form of the cost comparison: iterations and posterior draws are not",
        "commensurable, so both are placed on a common axis of seconds.",
    ),
    reuse("L_wallclock"),
    md(
        "## A.7 Is the posterior calibrated?",
        "",
        "SDA is the only method here that produces a distribution, so it is the only one that",
        "can be checked for calibration. Empirical coverage of central credible intervals.",
    ),
    reuse("M_calibration"),
]

# =====================================================================  B
cells += [
    md(
        "---",
        "# B. Models, training and reproducibility",
        "",
        "Everything in this part is **read from the actual checkpoints and configuration files**",
        "that produced the numbers above — nothing is transcribed by hand.",
    ),
    md("## B.1 Training specifications"),
    code(
        """spec = json.loads(Path("da_results_sda_paper/training_specs.json").read_text())

print("SYSTEM")
s = spec["system"]
for k in ("pde", "domain_L", "grid", "boundary", "solver", "dt_stored",
          "frames_per_trajectory"):
    print(f"   {k:24s} {s[k]}")
ly = s["lyapunov"]
print(f"   {'lambda_1':24s} {ly['lambda_1_mean']:.5f} +/- {ly['lambda_1_sem']:.5f}"
      f"  (T_L = {ly['lyapunov_time']:.2f} t.u.)")

print("\\nDATA")
for k, v in spec["data"].items():
    if v.get("present"):
        print(f"   {k:9s} {v['n_sim']:3d} traj x {v['n_t']} frames x {v['n_x']} pts"
              f"   dt={v['dt']:.3f}   sha256[:12]={v['sha256_12']}")

print("\\nTEST DISCIPLINE  (compared on field VALUES, not filenames)")
for held, d in spec["test_discipline"]["held_out_vs"].items():
    for k, v in d.items():
        print(f"   {held:13s} vs {k:9s}: {v['n_trajectories_shared']} shared"
              f"  ->  {'DISJOINT' if v['disjoint'] else 'OVERLAP -- LEAKAGE'}")
print("  ", spec["test_discipline"]["statement"])

rows = []
for name, m in spec["models"].items():
    rows.append({
        "model": name.split(" (")[0],
        "params": f"{m['params_total']:,}",
        "epochs": m.get("epoch_of_best_ckpt") or m.get("epochs_run") or m.get("epochs"),
        "batch": m.get("batch_size"),
        "lr": m.get("lr"),
        "temporal context": (f"rollout {m['max_rollout_length']}" if "max_rollout_length" in m
                             else f"rollout {m['training_rollout_length']}" if "training_rollout_length" in m
                             else f"blanket {m['blanket_window']}"),
        "sha256[:12]": m["sha256_12"],
    })
display(pd.DataFrame(rows).set_index("model"))
print("\\n4D-Var inference settings (tuned on VALIDATION, frozen before test):")
for k, v in spec["inference_settings"]["4D-Var"].items():
    print(f"   {k:12s} {v['best']}")
sd = spec["inference_settings"]["SDA sampler"]
print(f"   SDA          {sd}")
print(f"   selected on  {spec['inference_settings']['SDA selection']['selected_on']}")"""
    ),
    md(
        "## B.2 The KAE encoder and decoder, as actually built",
        "",
        "The revision asks for the *actual* architecture rather than a conceptual sketch, so",
        "the layer tables below are enumerated from the instantiated, weight-loaded model.",
        "",
        "The single most important number here is the **compression ratio**.",
    ),
    code(
        """ae = json.loads(Path("da_results_sda_paper/ae_roundtrip.json").read_text())
io_, cfg_ = ae["io"], ae["config"]
print(f"checkpoint : {ae['checkpoint']}")
print(f"state      : {io_['input_field'][0]} x {io_['input_field'][1]} = "
      f"{io_['input_field'][0] * io_['input_field'][1]} values")
print(f"latent     : {io_['latent_dim']}")
print(f"ratio      : {io_['compression_ratio']:.2f}   -> the representation is "
      f"{'OVER-COMPLETE' if io_['compression_ratio'] < 1 else 'compressive'}")
print(f"operator   : {cfg_['operator_mode']}, continuous={cfg_['is_continuous']}, "
      f"attention={cfg_['use_attention']}, spectral-loss={cfg_['spectral']}")
print(f"conv       : kernel {cfg_['kernel_size']}, padding_mode "
      f"{cfg_['conv_kwargs']['padding_mode']}, hidden {cfg_['hidden_dims']}")
print(f"transformer: {cfg_['transformer']}")
pc = ae["param_counts"]
print(f"\\nparameters : encoder {pc['encoder']:,} | decoder {pc['decoder']:,} | "
      f"Koopman generator K {pc['koopman_generator']:,}  "
      f"(= {int(io_['latent_dim'])}^2 + skew/sym parametrisation)")


def layer_table(layers, title):
    r = []
    for L in layers:
        d = {"path": L["path"], "type": L["type"], "params": L["params"]}
        if "in_channels" in L:
            d["shape"] = (f"{L['in_channels']}->{L['out_channels']} k{L['kernel_size']} "
                          f"s{L['stride']} p{L['padding']}")
        elif "in_features" in L:
            d["shape"] = f"{L['in_features']}->{L['out_features']}"
        elif "num_channels" in L:
            d["shape"] = f"{L['num_groups']} groups / {L['num_channels']} ch"
        else:
            d["shape"] = ""
        r.append(d)
    df = pd.DataFrame(r)
    print(f"\\n=== {title}  ({len(df)} modules, {df['params'].sum():,} parameters) ===")
    display(df.style.hide(axis="index").format({"params": "{:,}"}))


layer_table(ae["encoder_layers"], "ENCODER")
layer_table(ae["decoder_layers"], "DECODER")"""
    ),
    md(
        "### B.3 Encoder–decoder round trip",
        "",
        "How much of the field survives $x \\to \\mathrm{Encoder} \\to \\mathrm{Decoder} \\to \\hat x$?",
        "",
        "**This is not a floor.** Because the latent is over-complete, KAE data assimilation",
        "optimises $z_0$ freely and is never required to land on $\\mathrm{Encoder}(x)$ — and it",
        "does in fact beat the round trip. The round trip is a *reference scale* for what the",
        "learned representation captures, and is plotted as such.",
    ),
    code(
        """rt = ae["round_trip"]
print(f"round-trip rel-L2 over {rt['n_states']} held-out states from {rt['source']}")
print(f"   mean   {rt['mean']:.4f} +/- {rt['sem']:.4f} (SEM)   std {rt['std']:.4f}")
print(f"   median {rt['median']:.4f}   5-95%  {rt['p05']:.4f} - {rt['p95']:.4f}")
print(f"   latent std {rt['latent_std']:.4f}, |z|max {rt['latent_absmax']:.2f}")
print("\\nwhere the round-trip error lives:")
print(f"   {'band':8s} {'share of total sq err':>22s} {'rel err in band':>17s}")
for b, v in rt["spectral"].items():
    print(f"   {b:8s} {100 * v['share_of_total_sq_error']:21.1f}% "
          f"{v['rel_err_in_band']:17.4f}")
for k, v in ae["key_findings"].items():
    print(f"\\n[{k}]\\n   {v}")
print(f"\\n[interpretation]\\n   {ae['interpretation']}")"""
    ),
]

# =====================================================================  C
cells += [
    md(
        "---",
        "# C. Temporal geometry: $\\delta_f$, $\\delta_l$ and $N$ separated",
        "",
        "The canonical schedule fixes all three at once. Parts C.2–C.4 vary exactly one and pin",
        "the other two, so each curve answers a single question.",
    ),
    md(
        "## C.1 How long is a Lyapunov time here?",
        "",
        "**A units correction.** The brief asks to extend 'beyond the current 2.5 Lyapunov-time",
        "limit'. The canonical $\\delta_l = 2.5$ is in **time units**. Measured on held-out",
        "trajectories by a twin experiment with renormalisation, one Lyapunov time on this",
        "system is $\\approx\\!22.6$ time units — so the canonical horizon is about",
        "$0.11\\,T_L$, not $2.5\\,T_L$. Part C.3 extends to $3.1\\,T_L$.",
    ),
    code(
        """from IPython.display import Image, display as _disp
lyr = json.loads(Path("da_results_sda_paper/lyapunov_report.json").read_text())
print(f"lambda_1 = {lyr['lambda_1']:.5f} +/- {lyr['lambda_1_sem']:.5f}  "
      f"(n={lyr['n_trajectories']} held-out trajectories)")
print(f"T_L      = {lyr['lyapunov_time_t_units']:.2f} time units "
      f"= {lyr['lyapunov_time_frames']:.0f} frames\\n")
print(f"{'horizon':42s} {'t.u.':>7s} {'frames':>7s} {'T_L':>7s}")
for r in lyr["horizons"]:
    print(f"{r['name']:42s} {r['t_units']:7.1f} {r['frames']:7d} {r['lyapunov_times']:7.2f}")
print(f"\\n{'target':>8s} {'t.u.':>8s} {'frames':>8s}   fits a 1000-frame record?")
for t in lyr["targets"]:
    print(f"{t['lyapunov_times']:8.1f} {t['t_units']:8.1f} {t['frames']:8d}   "
          f"{'yes' if t['fits_in_1000_frame_record'] else 'NO'}")
print(f"\\n{lyr['note']}")
_disp(Image("da_results_sda_paper/lyapunov_report.png"))"""
    ),
    md(
        "## C.2 $\\delta_f$ — lead time to the *first* observation",
        "",
        "$\\delta_l$ and $N$ are pinned. Moving $\\delta_f$ out means every observation sits",
        "further from $t_0$, so the question is how quickly information about $t_0$ decays with",
        "the gap to the nearest measurement.",
    ),
    code(
        """from data_assimilation.ks import plot_geometry as pg
GEO = Path("da_results_geometry")
TL = lyr["lyapunov_time_t_units"]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
pg.fig_C1(GEO, ax=ax); plt.show()
print(pg.table_geometry(GEO, "C1_delta_f"))"""
    ),
    md(
        "## C.3 $\\delta_l$ — the recovery horizon, in Lyapunov times",
        "",
        "$\\delta_f$ and $N$ are pinned; only the span of the observation window grows, out to",
        "$3.1\\,T_L$.",
        "",
        "> **Budget note.** 4D-Var cost per iteration grows with the rollout length, so running",
        "> this sweep at the campaign's 8000 iterations would cost $\\approx\\!72$ GPU-hours for",
        "> the U-Net rows alone. The budget is held **constant across the sweep** at a lower",
        "> value — so the horizon effect is not confounded with the budget — and one anchor",
        "> horizon is repeated at 8000 iterations so the offset between the two budgets is",
        "> *measured* rather than assumed. Wall-clock per point is recorded alongside.",
    ),
    code(
        """fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.3))
pg.fig_C2(GEO, TL, ax=ax[0], xunits="lyapunov")
pg.fig_C2(GEO, TL, ax=ax[1], xunits="time")
plt.tight_layout(); plt.show()
print(pg.table_geometry(GEO, "C2_delta_l", TL=TL))

c2 = pg.load(GEO, "C2_delta_l")
print("\\nBUDGET:", c2["meta"]["budget_note"])
print("ANCHOR:", c2["meta"]["anchor"])
print("\\ncost of one solve, seconds  (this is the compute-scaling result):")
hdr = f"{'delta_l':>8s} {'frames':>7s} {'T_L':>6s} " + " ".join(f"{m:>12s}" for m in pg.ORDER)
print(hdr)
for r in c2["rows"]:
    print(f"{r['delta_l']:8.1f} {r['frames'][-1]:7d} {r['delta_l'] / TL:6.2f} " +
          " ".join(f"{r[m]['wall_s']:12.1f}" if m in r and 'wall_s' in r[m]
                   else f"{'--':>12s}" for m in pg.ORDER))"""
    ),
    md(
        "## C.4 $N$ — the number of observation times",
        "",
        "$\\delta_f$ and $\\delta_l$ are pinned **exactly**, so the observation window neither",
        "grows nor moves; only its interior fills in. Rounding to the frame grid can merge",
        "requested times, so the *realised* $N$ is what is plotted and tabulated.",
        "",
        "> **SDA diverges at high $N$, and the table says so.** Above $N\\approx8$ the sampler",
        "> blows up on a growing fraction of problems: 0% up to $N=8$, 4.2% at $N=13$, 20.8%",
        "> at $N=16$. An instrumented run shows this is *not* a singular-covariance NaN — the",
        "> score reaches $9.6\\times10^{18}$ and the field $9.2\\times10^{6}$ while staying",
        "> finite — it is a runaway of the guided reverse diffusion.",
        ">",
        "> **Mechanism.** The likelihood covariance is block diagonal, one $64\\times64$ solve",
        "> per observation time, so per-block conditioning does *not* degrade with $N$. But the",
        "> Mahalanobis term is a **sum** over blocks, so the guidance gradient grows with the",
        "> number of observation times while the prior score stays $O(1)$. Eq. 15 defines the",
        "> likelihood over all observations jointly with no normalisation, so the sum is",
        "> faithful to the paper — guidance simply overwhelms the prior as observations",
        "> accumulate.",
        ">",
        "> The sampler was **not** re-tuned to remove this. It is reported at the frozen,",
        "> validation-selected configuration, and the failure count is printed next to any",
        "> statistic computed over the survivors rather than replacing it.",
        ">",
        "> Note the trade-off this creates: on the runs that converge, SDA keeps *improving*",
        "> with $N$ (0.0181 at $N=2$ to 0.0061 at $N=16$) while becoming steadily less",
        "> reliable.",
    ),
    code(
        """fig, ax = plt.subplots(figsize=(6.4, 4.2))
pg.fig_C3(GEO, ax=ax); plt.show()
print(pg.table_geometry(GEO, "C3_n_obs"))
print("\\nrealised observation schedules (frames):")
for r in pg.load(GEO, "C3_n_obs")["rows"]:
    print(f"   requested N={str(r['value']):>3s} -> realised N={r['N']:2d}  {r['frames']}")"""
    ),
    md(
        "### C.4a The same question asked the *confounded* way",
        "",
        "This is the original 'recovery vs number of observations' sweep, kept deliberately.",
        "It takes nested prefixes of a fixed pool, so adding an observation also **pushes",
        "$\\delta_l$ further out** — $N$ and $\\delta_l$ move together. Comparing it with C.4,",
        "where $\\delta_l$ is pinned, shows how much of the apparent '$N$ effect' was really a",
        "horizon effect.",
    ),
    reuse("C_nobs_old"),
    md(
        "## C.5 Extremely long rollouts — free-running forecast skill",
        "",
        "This asks a *model* question, not an assimilation question: started from the **exact**",
        "true state, how far can each propagator be rolled before it is no better than guessing?",
        "No optimisation and no observations are involved.",
        "",
        "Three reference levels are drawn with the curves:",
        "**persistence** (hold $u(t_0)$ fixed), **climatology** ($\\hat u = 0$, which gives",
        "rel-$L_2 = 1$ by definition), and **saturation** — the error between two independent",
        "true states, which is the level a forecast with no remaining skill must reach.",
    ),
    code(
        """fig, ax = plt.subplots(figsize=(7.6, 4.8))
pl.fig_forecast(LONG, ax=ax); plt.show()
print(pl.table_forecast(LONG))"""
    ),
    md(
        "## C.6 Forecasting *after* assimilation — the operational question",
        "",
        "C.5 starts every method from the **exact** true state, so it measures the propagator",
        "alone. A real system never has that. This section starts each method from **its own",
        "analysis**, obtained from a short canonical observation window, and then forecasts a",
        "long way. The two curves bracket the honest answer and are drawn together (solid =",
        "after assimilation, dotted = from the exact state).",
        "",
        "The two effects pull in opposite directions, which is what makes this worth measuring",
        "rather than predicting: the KAE earns a far better analysis but propagates it badly,",
        "while the U-Net starts from a much worse analysis and propagates it well.",
    ),
    code(
        """fig, ax = plt.subplots(figsize=(8.0, 5.0))
pl.fig_post_da(LONG, ax=ax); plt.show()
print(pl.table_post_da(LONG))"""
    ),
    md(
        "## C.7 Assimilation from far-future observations",
        "",
        "The assimilation counterpart of C.5, and the sweep behind A.2: hold $\\delta_f$ and $N$",
        "fixed and push $\\delta_l$ from $0.11\\,T_L$ out to $17.7\\,T_L$.",
        "",
        "The question is whether $t_0$ is still identifiable at all once the observations are",
        "many Lyapunov times away. An $\\times$ marks where a method hits its cost wall; the",
        "caps are affordability limits, printed with the table, not statements of capability.",
    ),
    code(
        """fig, ax = plt.subplots(figsize=(7.6, 4.8))
pl.fig_long_baseline(LONG, ax=ax); plt.show()
print(pl.table_long_baseline(LONG))"""
    ),
    md(
        "## C.8 Continuous-time propagation and irregular sampling",
        "",
        "The KAE evaluates $e^{K\\tau}$ at arbitrary real $\\tau$; the U-Net can only take whole",
        "steps. This is where that distinction is measured rather than asserted.",
    ),
    reuse("G_continuous"),
    md("## C.9 Cost against horizon"),
    reuse("B_cost"),
]

# =====================================================================  D
cells += [
    md(
        "---",
        "# D. Observation noise",
        "",
        "The observation operator is held fixed across this entire part: canonical schedule,",
        "fully observed in space. Only the noise changes.",
    ),
    md("## D.1 Gaussian noise"),
    reuse("D_noise"),
    md(
        "## D.2 A misspecified error law: Gaussian vs Laplace at matched variance",
        "",
        "**Why not Poisson.** The KS state $u(x,t)$ is a signed, continuous, zero-mean field",
        "with no count interpretation and no non-negativity constraint, so a Poisson likelihood",
        "is not merely a poor fit — it is undefined on roughly half the domain. Laplace is the",
        "meaningful stress test: identical variance, heavier tails, so a few observations are",
        "badly corrupted while the rest are cleaner than Gaussian.",
        "",
        "**None of the three methods is told.** The 4D-Var cost stays quadratic and SDA keeps",
        "its Gaussian likelihood approximation, so this measures *robustness to",
        "misspecification*, identically for all of them.",
        "",
        "> **Two things to read carefully in the table.**",
        ">",
        "> 1. At $\\sigma_y = 0$ the Gaussian and Laplace rows are identical to four decimals,",
        ">    as they must be — no noise is added in either case. That is a free consistency",
        ">    gate on the sweep, and it passes.",
        "> 2. SDA is **not** monotone between $\\sigma_y = 0$ and $\\sigma_y = 0.01$. This is a",
        ">    convention, not an error: a Gaussian likelihood cannot represent a zero-noise",
        ">    hard constraint, so for clean observations SDA is given an assumed noise floor",
        ">    $\\sigma_y = 0.05$ (selected on validation and frozen). At $\\sigma_y = 0.01$ it is",
        ">    given the true value and so weights the observations correctly. The $\\sigma_y=0$",
        ">    point therefore understates SDA relative to its own $\\sigma_y=0.01$ point.",
    ),
    code(
        """fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.3))
pg.fig_noise_law(GEO, axes=ax); plt.tight_layout(); plt.show()
print(pg.table_geometry(GEO, "noise_law"))
print("\\n" + pg.load(GEO, "noise_law")["meta"]["note"])"""
    ),
]

# =====================================================================  E
cells += [
    md(
        "---",
        "# E. Sparse observations",
        "",
        "Two independent ways of removing information — fewer sensors in space, fewer times —",
        "and then both at once.",
    ),
    md("## E.1 Spatial sparsity"),
    reuse("E_sparsity"),
    md("## E.2 Gap filling — recovering the full field from a handful of sensors"),
    reuse("J_sparse"),
    md(
        "## E.3 Joint spatial $\\times$ temporal sparsity",
        "",
        "$\\delta_f$ and $\\delta_l$ are pinned across the whole grid, so the temporal axis",
        "changes only how densely the *same* interval is sampled. The two axes are not",
        "interchangeable: the total number of observed scalars can be equal in very different",
        "cells with very different outcomes.",
        "",
        "**The binding constraint is sensor coverage, not observation frequency.** At 6 sensors",
        "the KAE is pinned near 0.58 however many times are observed — going from $N=2$ to",
        "$N=13$ buys 3%. The equal-information rows below make the asymmetry explicit: the same",
        "48 observed scalars give 0.0745 as 16 sensors $\\times$ 3 times, but 0.5784 as 6",
        "sensors $\\times$ 8 times — **7.8$\\times$ worse for identical information**.",
        "",
        "**A crossover worth stating.** The learned trajectory prior earns its cost exactly",
        "where the observations stop constraining the state:",
        "",
        "| sensors | KAE | SDA | winner |",
        "|---|---|---|---|",
        "| 64 (100%) | **0.0042** | 0.0094 | KAE by 2.2$\\times$ |",
        "| 32 (50%) | **0.0123** | 0.0147 | KAE by 1.2$\\times$ |",
        "| 16 (25%) | 0.0704 | **0.0257** | SDA by 2.7$\\times$ |",
        "| 6 (10%) | 0.5867 | **0.2213** | SDA by 2.7$\\times$ |",
        "",
        "With dense observations the 4D-Var point estimate wins; below about half the sensors",
        "the diffusion prior wins, because it supplies structure the observations no longer",
        "constrain. The U-Net has no skill left at 6 sensors at all (1.015–1.029, i.e. above",
        "climatology).",
    ),
    code(
        """fig, ax = plt.subplots(1, 3, figsize=(14.4, 4.2))
pg.fig_joint_sparsity(GEO, axes=ax); plt.tight_layout(); plt.show()

js = pg.load(GEO, "joint_sparsity")
print(f"state dimension {js['meta']['state_dim']}, "
      f"delta_f={js['meta']['fixed']['delta_f']}, delta_l={js['meta']['fixed']['delta_l']}")
print(f"\\n{'frac':>6s} {'N':>3s} {'scalars':>8s} " +
      " ".join(f"{m:>16s}" for m in pg.ORDER))
for r in sorted(js["rows"], key=lambda r: (-r["obs_frac"], r["n_times"])):
    print(f"{r['obs_frac']:6.2f} {r['N']:3d} {r['total_scalars']:8d} " +
          " ".join(f"{r[m]['mean']:9.4f}+-{r[m]['sem']:.4f}" if m in r
                   else f"{'--':>16s}" for m in pg.ORDER))"""
    ),
]

# =====================================================================  F
cells += [
    md(
        "---",
        "# F. Space–time recovery and spectral analysis",
        "",
        "Part A scores the analysis state at $t_0$. This part asks a different question: what",
        "does each method's *whole reconstructed trajectory* look like, and where in wavenumber",
        "does the error actually live?",
        "",
        "> These are two different claims and they do not agree. Nothing below should be read as",
        "> 'method X is better' without saying **which** of the two is meant.",
    ),
    md("## F.1 Space–time recovery"),
    reuse("I_spacetime"),
    md(
        "## F.2 Audit: is the U-Net space–time result real?",
        "",
        "The U-Net's trajectory error is orders of magnitude *below* the other methods at every",
        "frame except $t_0$, where it is by far the worst. That pattern demands an audit, not an",
        "explanation, so:",
        "",
        "1. the plotted trajectory is re-rolled **independently** from the optimised $x_0$ and",
        "   compared against what the pipeline plotted;",
        "2. the error is decomposed by wavenumber band, reporting each band's **share of the",
        "   total squared error** — not just its band-relative error, which flatters bands that",
        "   carry no energy;",
        "3. the same decomposition is repeated at $t_0$, at the first *unobserved* step after",
        "   $t_0$, at the first observation, and at a later observed frame.",
    ),
    code(
        """from data_assimilation.ks import plot_unet_audit as ua
fig, ax = plt.subplots(2, 2, figsize=(12.6, 7.4))
ua.figure(axes=ax); plt.tight_layout(); plt.show()
print(ua.summary())"""
    ),
    md(
        "### F.2a The first ten steps — the actual fields",
        "",
        "Each method's **assimilated** state at $t_0$, rolled forward under its own dynamics",
        "for ten steps, against the truth. Nothing is re-observed along the way, so these",
        "panels show how an analysis error at $t_0$ actually propagates.",
    ),
    code(
        """fig, _ = ua.fig_spacetime_zoom(10, traj=0)
fig.tight_layout(); plt.show()"""
    ),
    md(
        "The bottom-left panel is the whole story: the U-Net's error is a bright vertical",
        "stripe confined to the first one or two columns, with visible fine-scale banding in",
        "$x$, and it **ends before the first observation** at $t=0.2$. KAE and SDA instead",
        "show faint, smooth error spread across the window.",
        "",
        "The two failure modes are therefore qualitatively different, not merely different in",
        "size: the U-Net makes a **large, short-lived, high-wavenumber** error that the",
        "dynamics erase before any observation sees it; the others make **small, long-lived,",
        "smooth** errors that persist. The same trajectories as line plots:",
    ),
    code(
        """ua.fig_fields_first_steps(10, traj=0)
plt.tight_layout(); plt.show()"""
    ),
    md(
        "**Frame 0 is the only panel where anything is visibly wrong.** The U-Net's analysis",
        "carries a high-frequency ripple on an otherwise correct large-scale shape — the",
        "$k\\approx8$ spike. By frame 1 it is gone; from frame 2 the three curves lie on the",
        "truth. The 4D-Var cost only ever evaluates frames $\\ge 2$, where the ripple has",
        "already been destroyed, which is exactly why it is barely penalised.",
        "",
        "The same thing as an error curve, on log and linear axes:",
    ),
    code(
        """fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.2))
ua.fig_zoom(10, axes=ax); plt.tight_layout(); plt.show()
print(ua.zoom_table(10))"""
    ),
    md(
        "The error falls **8$\\times$ in a single step** (0.1555 $\\to$ 0.0195) and",
        "**477$\\times$ by frame 3**, crossing the other two methods between frames 1 and 2,",
        "then flattening at $\\sim\\!10^{-4}$.",
        "",
        "That 8$\\times$ is not a coincidence: it matches the contraction the U-Net's own",
        "one-step map applies along this error direction ($0.125$, measured in F.3) and the",
        "linear-KS prediction at $k=8$ ($e^{(q^2-q^4)\\Delta t}=0.111$). The analysis error is",
        "not gradually corrected by the dynamics — it is **annihilated** by them, because it",
        "lives almost entirely in the strongly damped band.",
    ),
    md(
        "### F.3 Why 4D-Var stalls at $t_0$ — and what it is *not*",
        "",
        "Panel (d) above localises the U-Net's analysis error to a single narrow band around",
        "$k\\approx 8$, with an amplitude far above the truth's own spectrum there, which is gone",
        "one step later. Two hypotheses explain that, and they have different consequences:",
        "",
        "> **H1 (identifiability).** $x_0^{\\rm rec}$ and $x_0^{\\rm true}$ are *equally good* to the",
        "> cost — the error lies in a nullspace of the observation operator composed with the",
        "> dynamics, so no amount of extra optimisation would help.",
        ">",
        "> **H2 (conditioning).** The cost still prefers the truth, but is so insensitive along",
        "> that direction that gradient descent does not get there in the budget allowed.",
        "",
        "These are distinguished by one number: the 4D-Var objective evaluated at the recovered",
        "$x_0$ versus at the true $x_0$. **The test below rejects H1 and supports H2.**",
    ),
    code(
        """fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.0))
ua.nullspace_figure(axes=ax); plt.tight_layout(); plt.show()
print(ua.nullspace_summary())"""
    ),
    md(
        "**What the audit establishes.**",
        "",
        "1. **The space–time result is real.** Re-rolling the optimised $x_0$ independently",
        "   reproduces the plotted trajectory to `0.000e+00`. No observation insertion, nudging,",
        "   teacher forcing, reinitialisation or hidden DA update is involved — the whole",
        "   trajectory is one deterministic rollout from one optimised vector.",
        "",
        "2. **The analysis error is a single damped band.** It is concentrated at $k\\approx 8$,",
        "   and the U-Net's own one-step map contracts that direction by $\\approx 0.125$ —",
        "   agreeing closely with linear KS theory, $e^{(q^2-q^4)\\Delta t}\\approx 0.11$ at",
        "   $k=8$ — against $\\approx 0.78$ for a random direction of the same norm. The cost",
        "   function is therefore about $6\\times$ less sensitive to this error than to a generic",
        "   one.",
        "",
        "3. **But it is *not* an identifiability failure.** The objective at the true $x_0$ is",
        "   still $\\sim\\!14\\times$ lower than at the recovered $x_0$. The optimiser has not found",
        "   a point the cost prefers to the truth; it has simply not got there. This is",
        "   consistent with Part A.4, where neither 4D-Var method plateaus.",
        "",
        "4. **Consequently the two claims must be kept apart.** On *trajectory* reconstruction",
        "   over the observed window the U-Net is far ahead; on *analysis-state* reconstruction",
        "   at $t_0$ it is far behind. Neither statement generalises to the other, and neither",
        "   should be reported without saying which is meant.",
    ),
]

# =====================================================================  G
cells += [
    md("---", "# G. Conclusions"),
    md("## G.1 Summary table — the original sections"),
    reuse("summary_table"),
    md(
        "## G.2 Summary table — temporal geometry, noise, sparsity and horizon",
        "",
        "Everything from Parts C, D and E in one place. Rows whose experiment has not",
        "been run are omitted rather than guessed at.",
    ),
    code(
        '''import contextlib

def _endpoints(tag, xlabel, fmt="{:.2f}"):
    """First and last point of a sweep, so the table shows the RANGE, not one number."""
    try:
        d = pg.load(GEO, tag)
    except pg.NotRunYet:
        return []
    rows, rs = [], d["rows"]
    for r, lab in [(rs[0], "min"), (rs[-1], "max")]:
        v = r.get("value")
        v = v if not isinstance(v, list) else tuple(v)
        row = {"experiment": f"{xlabel} = {fmt.format(v) if isinstance(v, float) else v}"
                             f"  ({lab})"}
        for m in METHODS:
            row[STYLE[m]["short"]] = (r[m]["mean"] if m in r and "mean" in r[m]
                                      else np.nan)
        rows.append(row)
    return rows


rows2 = []
rows2 += _endpoints("C1_delta_f", "C1  $\\delta_f$")
rows2 += _endpoints("C2_delta_l", "C2  $\\delta_l$")
rows2 += _endpoints("C3_n_obs", "C3  $N$", fmt="{:.0f}")
rows2 += _endpoints("noise_law", "D  $\\sigma_y$", fmt="{:.2f}")
rows2 += _endpoints("joint_sparsity", "E  (frac, $N$)")

# long-horizon skill, from the two forecast experiments
for f, lab in [("FC_forecast", "forecast from the EXACT state"),
               ("PF_post_da_forecast", "forecast AFTER assimilation")]:
    fp = LONG / f"{f}.npz"
    if not fp.is_file():
        continue
    d = np.load(fp)
    row = {"experiment": f"skill horizon, {lab}  ($T_L$)"}
    for m in METHODS:
        k = f"{m}__skill_horizon_tu"
        row[STYLE[m]["short"]] = (float(d[k]) / float(d["T_L"])
                                  if k in d.files else np.nan)
    rows2.append(row)

if rows2:
    df2 = pd.DataFrame(rows2).set_index("experiment")
    print("Analysis rel-L2 at the ends of each sweep, plus forecast skill horizons "
          "in Lyapunov times")
    print("(the last rows are HORIZONS -- higher is better; every other row is an "
          "ERROR -- lower is better)")
    print()
    display(df2.style.format("{:.4f}", na_rep="not run"))
else:
    print("none of the Part C/D/E experiments are on disk yet")'''
    ),
    md(
        "## G.3 Takeaways",
        "",
        "### 1. The two claims that must never be merged",
        "",
        "Every comparison here splits into **analysis-state** recovery at $t_0$ and",
        "**trajectory** reconstruction over the observed window, and the ordering",
        "*reverses* between them. The U-Net recovers the trajectory to $\\sim10^{-4}$",
        "while being the worst method at $t_0$; the KAE does the opposite. Neither",
        "statement generalises to the other, and neither should be reported without",
        "saying which is meant.",
        "",
        "### 2. The U-Net is the better forward model. The KAE is the better inverse model.",
        "",
        "| | measurement |",
        "|---|---|",
        "| U-Net forecasts better | free-running rel-$L_2$ 0.0003 vs 0.0242 at 2.3 t.u.; skill to 9.29 $T_L$ vs 2.42 $T_L$ |",
        "| KAE assimilates better | analysis 0.0045 vs 0.0987 at the canonical baseline |",
        "",
        "These are not in tension, and the reason is measured rather than asserted:",
        "**the KAE's advantage is the conditioning of the inverse problem, not the",
        "fidelity of the forward map.** Inverting $e^{K\\tau}$ is linear; inverting a",
        "nonlinear autoregressive map is not.",
        "",
        "### 3. The KAE's forecasting ceiling is structural, not a training failure",
        "",
        "121 of 128 eigenvalues of $K$ decay, and",
        "$\\max\\operatorname{Re}\\lambda(K) = +0.00727$ against the measured",
        "$\\lambda_1 = +0.04432$ — **6.1$\\times$ too slow**. A finite-dimensional linear",
        "autonomous system cannot have a chaotic attractor at all. So the KAE is",
        "necessarily accurate *locally* (its 1.0 t.u. training rollout, still 20$\\times$",
        "better than persistence at 2.5 t.u.) and necessarily poor *globally* — and",
        "assimilation lives in exactly the local regime.",
        "",
        "### 4. $\\delta_f$ dominates; $N$ saturates; $\\delta_l$ barely matters",
        "",
        "The canonical schedule conflated all three. Separated:",
        "",
        "| varied | range | KAE response |",
        "|---|---|---|",
        "| $\\delta_f$ | 0.1 $\\to$ 1.8 | 0.0045 $\\to$ 0.5288 (**117$\\times$ worse**) |",
        "| $N$ | 2 $\\to$ 16 | 0.0422 $\\to$ 0.0060, saturating by $N=3$ |",
        "| $\\delta_l$ | 0.02 $\\to$ 3.10 $T_L$ | 0.0158 $\\to$ 0.0456 (2.9$\\times$ over a 140$\\times$ range) |",
        "",
        "What matters is having an observation **close to $t_0$** — not how many there",
        "are, and not how far the window extends. At $\\delta_f = 1.8$ the U-Net",
        "overtakes the KAE (0.3375 vs 0.5288), the only regime found where it does.",
        "",
        "### 5. Cost is genuinely independent of the horizon — measured properly",
        "",
        "Timed one method at a time, KAE-expm is **9.56 $\\to$ 7.38 ms/iteration across",
        "an 800$\\times$ horizon range (0.77$\\times$)**, against 124$\\times$ for KAE-RK4 and",
        "243$\\times$ for the U-Net over only 200$\\times$. The mechanism is explicit:",
        "$e^{K\\tau_i}$ is precomputed once per observation time outside the optimisation",
        "loop, leaving a $\\tau$-independent matmul and decode inside it.",
        "",
        "> The campaign's own Part C.9 reports KAE-expm rising 9.07$\\times$ over the same",
        "> range. That measurement interleaves the methods to share GPU contention, but",
        "> contention adds a roughly *constant* per-step cost, inflating a 9.5 ms method",
        "> by ~90% and a 4425 ms method by ~0.2%. Interleaving is therefore biased",
        "> against the fast method. Both numbers are shown; the isolated one is correct.",
        "",
        "### 6. Two failure modes, both reported rather than tuned away",
        "",
        "**Neither 4D-Var method converges.** Both follow power laws in the iteration",
        "count with no plateau out to $1.3\\times10^5$ iterations. Every 4D-Var number",
        "here is a number *at a stated budget*. The U-Net's $t_0$ error is a direct",
        "consequence: the cost still prefers the truth by $14.2\\times$, so it is",
        "ill-conditioning, **not** unidentifiability — the nullspace hypothesis is",
        "rejected, not assumed.",
        "",
        "**SDA diverges at high $N$.** 0% up to $N=8$, 4.2% at $N=13$, 20.8% at $N=16$.",
        "Faithful to Eq. 15: the likelihood is a sum over observation blocks, so",
        "guidance grows with $N$ while the prior score stays $O(1)$. Not re-tuned; the",
        "failure count is reported beside every survivor statistic.",
        "",
        "### 7. What this does *not* show",
        "",
        "Nothing here says the continuous KAE is a better model of Kuramoto–Sivashinsky",
        "— the forecast experiments say plainly that it is not. The claim is narrower",
        "and better supported: **for recovering an unobserved state from nearby future",
        "observations, a linear latent generator gives a well-conditioned inverse",
        "problem at a cost independent of the horizon.** That is the regime data",
        "assimilation operates in, and it is the only regime these results support.",
    ),
    md(
        "## G.4 Consistency check",
        "",
        "Every headline number in this notebook and in `DA_REVISION_REPORT.md` is",
        "re-read here from the file that produced it and compared against the value",
        "as written. A stale caption or a hand-copied table cannot survive this cell.",
        "It also re-verifies that no held-out trajectory appears in any training or",
        "validation set.",
    ),
    code(
        """import subprocess
r = subprocess.run(["python", "-m", "data_assimilation.ks.consistency_check"], capture_output=True,
                   text=True, cwd=".")
print(r.stdout)
if r.returncode:
    print(r.stderr)
    raise AssertionError("consistency check FAILED — a reported number no longer "
                         "matches the saved results")"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NB.write_text(json.dumps(nb, indent=1))
n_md = sum(c["cell_type"] == "markdown" for c in cells)
print(f"wrote {NB}: {len(cells)} cells ({n_md} markdown, {len(cells) - n_md} code)")
