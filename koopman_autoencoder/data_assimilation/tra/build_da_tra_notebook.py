"""Generate visualize_da_tra.ipynb from templates, so structure is never hand-edited."""

import json
from pathlib import Path


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def code(src):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src,
    }


cells = [
    md(
        "# Data assimilation on transonic flow (2-D, 128x64, Mach-conditioned)",
        "",
        "Recover an unobserved state $u(t_0)$ from a handful of **future**, noisy, possibly",
        "sparse observations, using each surrogate as the forward model and changing nothing",
        "about it. Every weight is frozen; only the control variable moves.",
        "",
        "| symbol | meaning |",
        "|---|---|",
        "| $t_0$ | analysis time — the state to recover, **never observed** |",
        "| $\\delta_f = \\tau_1$ | lead to the *first* observation |",
        "| $\\delta_l = \\tau_N$ | recovery horizon — lead to the *last* |",
        "| $N$ | number of observation times |",
        "| $n$ | number of held-out trajectories averaged over (the statistics sample) |",
        "",
        "**Methods, and why each is treated as it is.**",
        "",
        "| model | DA method | control | reaching $\\tau_i$ |",
        "|---|---|---|---|",
        "| Continuous KAE | 4D-Var | latent $z_0$ | one product $e^{K\\tau_i}z_0$ |",
        "| U-Net | 4D-Var | conditioning window at $t_0$ | $\\tau_i$ autoregressive steps |",
        "| FNO | 4D-Var | conditioning window at $t_0$ | $\\tau_i$ autoregressive steps |",
        "| ACDM | score-based DA | posterior trajectory | 3-frame blanket score |",
        "| ACDM-ncn | score-based DA | posterior trajectory | **same path** (see below) |",
        "",
        "> **Why the diffusion models are not run with 4D-Var.** Backpropagating through 20",
        "> denoising steps per frame exhausted 20 GB on a *four*-frame rollout at batch 1.",
        "> Instead ACDM's U-Net is used as what it already is: it takes a 3-frame window noised",
        "> to one level and outputs $\\varepsilon$ for **all three frames** (15 channels in, 15",
        "> out) — turbpred discards two of them. Keeping them gives a local joint score over a",
        "> $k=1$ Markov blanket, exactly what Algorithm 2 of Rozet & Louppe composes. No",
        "> retraining, no architecture change.",
        "",
        "> **ACDM-ncn is run on the identical path by request.** It is trained with *clean*",
        "> conditioning (ncn = no conditioning noise), so a noised window is out of",
        "> distribution for it. Section D measures that directly rather than assuming it, so",
        "> any gap is attributable.",
    ),
    code(
        """%load_ext autoreload
%autoreload 2
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import sys; sys.path.insert(0, ".")
from data_assimilation.tra import plots as P
TRA = Path("da_results_tra")
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})
print("results:", sorted(p.name for p in TRA.glob("*.json")))"""
    ),
    md("---", "# A. Headline recovery"),
    md(
        "## A.1 Analysis error at $t_0$",
        "",
        "All methods see the **identical** problem: same trajectories, same analysis times,",
        "same observation geometry, same noise realisation, same obstacle mask. The obstacle",
        "interior is excluded from the metric — it carries no physics and each model fills it",
        "differently, so scoring it would reward whichever fills it most plausibly.",
    ),
    code(
        """fig, ax = plt.subplots(figsize=(7.6, 4.4))
P.fig_headline(TRA, ax=ax); plt.show()
print(P.table(TRA))"""
    ),
    md(
        "## A.2 Are the 4D-Var methods converged?",
        "",
        "Started from **climatology** ($z=0$, which in normalised units is the dataset mean),",
        "not from the truth. An earlier version of this driver initialised from the true",
        "window; that put iteration 0 at the answer and could only move away from it, so it",
        "measured nothing. The uninformed start is the only honest one.",
    ),
    code(
        """fig, ax = plt.subplots(figsize=(7.4, 4.4))
P.fig_convergence(TRA, ax=ax); plt.show()"""
    ),
    md(
        "---",
        "# B. Protocol and verification",
        "",
        "Two codebases are involved and nothing in either was modified. What makes the",
        "comparison trustworthy is that the glue is **gated**, not assumed:",
        "",
        "| check | result |",
        "|---|---|",
        "| adapter reproduces turbpred's own `forward` | **8/8 at float32 round-off** (max 9.5e-07) |",
        "| normalisation constants agree across codebases | identical entry for entry |",
        "| data identity | `gt_interp.nc` Ma = 0.66–0.68 matches turbpred's own test split |",
        "| gradients reach the control | U-Net 1.7e-05, FNO 2.8e-05 |",
        "",
        "Re-run the gate at any time with `python -m data_assimilation.tra.verify_adapters`.",
        "",
        "**Traps handled** (each found by reading source, not by assumption): the transonic",
        "channel order is `[v_x, v_y, rho, p]` in turbpred but `[v_x, v_y, p, rho]` in the",
        "`.nc` files; simulation parameters are channels that get overwritten with true values",
        "every step; the incompressible checkpoints use `normalizeMode='karmanMixed'`; and the",
        "tracked KAE configs have drifted from their checkpoints, so each checkpoint is paired",
        "with the config its own run saved.",
    ),
    code(
        """import subprocess
r = subprocess.run(["python", "-m", "data_assimilation.tra.verify_adapters",
                    "--regimes", "tra", "--models", "UNet", "FNO", "ACDM", "ACDM-ncn"],
                   capture_output=True, text=True, cwd=".")
print("\\n".join(l for l in r.stdout.splitlines() if "PASS" in l or "FAIL" in l
                 or "MATCH" in l))"""
    ),
    md(
        "## B.1 Learning rates were chosen on a *different* Mach range",
        "",
        "Every 4D-Var learning rate is selected on `gt_extrap.nc` (Mach 0.50–0.52) and frozen",
        "before `gt_interp.nc` (Mach 0.66–0.68) is touched, so no hyper-parameter is fitted on",
        "the numbers reported above.",
    ),
    code(
        """tp = TRA / "tuning.json"
if not tp.is_file():
    print("tuning.json is not on disk yet — it is written when the whole sweep "
          "finishes.\\n  produce it with:  python -m data_assimilation.tra.exp_tra --sections tune")
else:
    t = json.loads(tp.read_text())
    for m, v in t.items():
        sweep = "  ".join(f"{r['lr']:g}:{r['rel']:.4f}" for r in v["sweep"])
        best = min(v["sweep"], key=lambda r: r["rel"])["rel"]
        spread = max(r["rel"] for r in v["sweep"]) / best
        print(f"{m:6s} chosen lr={v['lr']:<6g}  spread across the sweep {spread:5.2f}x"
              f"  ({v['chosen_on']})")
        print(f"        {sweep}")
    print("\\nA FLAT response to the learning rate indicates a badly conditioned "
          "objective,\\nnot a mis-set optimiser.")"""
    ),
    md(
        "---",
        "# C. Memory: why 4D-Var over a 2-D rollout needs checkpointing",
        "",
        "Backpropagating through 25 autoregressive U-Net steps at $128\\times64$ stores every",
        "intermediate activation and exceeds the GPU. Segmented gradient checkpointing",
        "(recompute each $\\sqrt{n}$-step segment once in the backward pass) makes it fit. The",
        "forward *values* are unchanged — the equivalence gate in Part B still passes.",
        "",
        "| model | no checkpointing | $\\sqrt{n}$ segments |",
        "|---|---|---|",
        "| U-Net | **OOM** (>22 GB) | 5.26 GiB |",
        "| FNO | 4.31 GiB | 0.97 GiB |",
    ),
    md(
        "---",
        "# D. Is ACDM's window really a score?",
        "",
        "The blanket reinterpretation is only legitimate if the network actually denoises a",
        "noised window. Test: take a **true** trajectory, noise it to level $t$, score it, and",
        "recover $\\hat x_0$ by Tweedie. A valid score returns the prior mean at maximum noise,",
        "i.e. rel-$L_2 \\to 1$.",
        "",
        "| | $\\varepsilon$-RMSE $t{=}1$ | $t{=}10$ | $t{=}19$ | Tweedie $\\hat x_0$ $t{=}1$ | $t{=}10$ | $t{=}19$ |",
        "|---|---|---|---|---|---|---|",
        "| **ACDM** | 0.758 | 0.254 | **0.036** | 0.159 | 0.644 | **0.984** |",
        "| **ACDM-ncn** | 1.184 | 0.723 | 0.626 | 0.249 | 1.497 | **13.02** |",
        "",
        "ACDM behaves correctly: its $\\varepsilon$-prediction *improves* with noise and $\\hat x_0$",
        "lands at 0.984 at maximum noise. ACDM-ncn diverges to **13$\\times$ worse than",
        "climatology**, because it never saw a noised context in training. Its DA number in",
        "Part A should be read with this in view: it is a measurement of running a",
        "clean-conditioned model on the noisy-conditioned path, not a verdict on ACDM-ncn as a",
        "forecaster.",
    ),
    md(
        "---",
        "# E. What the model can do without any assimilation",
        "",
        "Before asking how well each method *inverts*, ask how well it *predicts*. Started from",
        "the **exact** true state, with no assimilation and no optimisation, this is the floor",
        "no analysis can beat — and it separates “the model cannot represent this flow” from",
        "“the assimilation cannot find the state”.",
        "",
        "> This section is also the gate that caught a serious bug. An earlier version of this",
        "> campaign fed the models transposed fields — the `.nc` files store $(x{=}64, y{=}128)$",
        "> while the checkpoints are trained on `dataSize=[128, 64]`. Convolutions accept either",
        "> and return plausible output, so nothing errored; the U-Net's one-step error was",
        "> **0.1456 instead of 0.0025** and every baseline came out worse than persistence.",
        "> `verify_adapters` could not catch it, because it compares the adapter against",
        "> turbpred's own forward on the *same* input — both were consistently wrong. The gate",
        "> that catches it is `verify_forward_skill`: **a forward model must beat persistence**.",
    ),
    code(
        """fig, ax = plt.subplots(1, 2, figsize=(13.0, 4.4))
P.fig_forecast(TRA, ax=ax[0]); P.fig_cost(TRA, ax=ax[1])
plt.tight_layout(); plt.show()"""
    ),
    md(
        "**The U-Net is the best forward model and the worst inverse model.** One-step error",
        "0.0023 against the KAE's 0.0382 — 16$\\times$ better — yet it is the *worst* method in",
        "Part A (0.2146 vs 0.0267). This is the same split the Kuramoto–Sivashinsky campaign",
        "found, reproduced on 2-D transonic flow with different models and a different",
        "codebase: **4D-Var through a linear latent generator is well conditioned; through a",
        "nonlinear autoregressive map it is not.**",
        "",
        "The cost panel is the other half: KAE's per-iteration cost is **flat across a",
        "50$\\times$ horizon range** (2.03 $\\to$ 1.85 ms) because $e^{K\\tau}$ is one matrix",
        "product, while the U-Net grows 83$\\times$ and FNO 71$\\times$.",
    ),
    md(
        "---",
        "# F. Sweeps: what the headline depends on",
        "",
        "The headline is one operating point. These vary one thing at a time with everything",
        "else fixed, so whatever moves is attributable to the swept quantity alone. All six",
        "sweeps cover **all five methods**.",
        "",
        "> **Budget.** Sweeps run at a constant 500 iterations against the headline's 2000, so",
        "> the swept variable is never confounded with the optimisation budget. The canonical",
        "> point appears in both, making the offset measurable rather than assumed.",
    ),
    code(
        """fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.4))
for a, s in zip(ax.ravel(), ["G6_single_obs", "G1_delta_f", "G4_delta_l",
                             "G5_n_obs", "G3_sparsity", "G2_noise"]):
    P.fig_sweep(TRA, s, ax=a)
plt.tight_layout(); plt.show()"""
    ),
    code(
        """import json
M = ["KAE", "ACDM", "FNO", "UNet", "ACDM-ncn"]
for stem, lab, f in [("G6_single_obs", "tau (1 obs)", "{:.0f}"),
                     ("G1_delta_f", "delta_f", "{:.0f}"),
                     ("G4_delta_l", "delta_l", "{:.0f}"),
                     ("G5_n_obs", "N", "{:.0f}"),
                     ("G3_sparsity", "obs_frac", "{:.2f}"),
                     ("G2_noise", "noise", "{:.2f}")]:
    rows = P.load_sweep(TRA, stem)["rows"]
    present = [m for m in M if m in rows[0]]
    print(f"=== {stem} ===")
    print(f"{lab:>12s} " + " ".join(f"{m:>9s}" for m in present))
    for r in rows:
        print(f"{f.format(r['value']):>12s} " + " ".join(
            f"{r[m]['mean']:9.4f}" if m in r and r[m]['mean'] == r[m]['mean']
            else f"{'nan':>9s}" for m in present))
    print()"""
    ),
    md(
        "### The crossover the headline hides",
        "",
        "**ACDM wins when observations are close, dense and clean. KAE wins everywhere else.**",
        "",
        "| condition | ACDM | KAE | winner |",
        "|---|---|---|---|",
        "| $\\delta_f=1$ | **0.0054** | 0.0269 | ACDM 5$\\times$ |",
        "| $\\delta_f=4$ | 0.1837 | **0.0421** | KAE 4.4$\\times$ |",
        "| $\\delta_f=18$ | 0.2608 | **0.1470** | KAE 1.8$\\times$ |",
        "| 100% of sensors | **0.0055** | 0.0278 | ACDM 5$\\times$ |",
        "| 10% of sensors | 0.0956 | **0.0285** | KAE 3.4$\\times$ |",
        "| 5% of sensors | 0.1690 | **0.0269** | KAE 6.3$\\times$ |",
        "| $\\sigma_y=0.30$ | 0.0342 | **0.0295** | KAE (tied) |",
        "",
        "**The KAE is the only method flat on every axis.** It stays within 0.024–0.029 across",
        "$\\delta_l$, $N$, sparsity down to 5% of sensors, and noise up to $\\sigma_y=0.3$,",
        "moving only for $\\delta_f$ (5.5$\\times$). ACDM degrades **34$\\times$** under sparsity",
        "and **48$\\times$** under $\\delta_f$.",
        "",
        "**$N$ barely matters for anyone**: KAE is 0.0271 at $N=1$ and 0.0258 at $N=9$. One",
        "observation is nearly as good as nine — the same conclusion the KS campaign reached,",
        "that *when* you observe dominates *how often*.",
        "",
        "**ACDM-ncn sits at 2.4–4.1 everywhere**, above climatology, and worsens as conditions",
        "degrade. Its Tweedie diagnostic in Part D predicted exactly this, and the prediction",
        "holds across all six sweeps.",
    ),
    md(
        "## F.1 The recovered fields, not just the error",
        "",
        "A sweep of scalars says how much is lost; the fields say *what* is lost.",
    ),
    code(
        """P.fig_sweep_fields(TRA, "G1_delta_f", channel=0)
plt.tight_layout(); plt.show()"""
    ),
    md(
        "---",
        "# G. Do the models actually work? Three correctness checks",
        "",
        "A DA comparison is only meaningful if each surrogate can run forward at all. These",
        "three checks establish that, and together they explain the whole campaign.",
    ),
    md(
        "## G.1 Rollout from the TRUE initial condition",
        "",
        "Each model is handed the **exact** state and asked to predict, with no assimilation",
        "anywhere. This separates “the model cannot represent this flow” from “the",
        "assimilation cannot find the state”.",
    ),
    code(
        """P.fig_rollout_fields(TRA, channel=3)
plt.suptitle("Free-running rollout from the TRUE initial condition (density)", y=1.005)
plt.tight_layout(); plt.show()"""
    ),
    md(
        "**Every established method reproduces the von Kármán wake faithfully.** At frame 24",
        "the U-Net (0.0358) is visually indistinguishable from truth, including the detached",
        "vortex pair; ACDM-ncn (0.0439), ACDM (0.0748) and FNO (0.0814) follow. **The KAE is",
        "the *worst* forward model here (0.1123)** — its near-wake is over-smoothed and the",
        "shed vortices lose definition.",
        "",
        "So the baselines' poor DA numbers are **not** a model-quality problem. The U-Net",
        "predicts 17$\\times$ better than the KAE one step out, and recovers $t_0$ 8$\\times$",
        "worse. Forward accuracy and invertibility are different properties.",
    ),
    md(
        "## G.2 Rollout from a PERTURBED initial condition",
        "",
        "White noise is added to $x_0$ only, at 1%, 5% and 20% of the field standard",
        "deviation. A trained surrogate should give a slightly wrong flow, not garbage.",
    ),
    code(
        """P.fig_perturbed(TRA, kind="white", channel=3)
plt.suptitle("Rollout from a PERTURBED initial condition", y=1.003)
plt.tight_layout(); plt.show()
print(P.perturbation_table(TRA, "white"))"""
    ),
    md(
        "All five models pass: the perturbation is visible at frame 0 and **cleaned up by",
        "frame 8**, with the wake recovered. Two things stand out.",
        "",
        "**The KAE never shows the noise at all**, even at 20%: its encoder projects onto a",
        "128-dimensional latent, so grid-scale structure is filtered before any dynamics run.",
        "Its final-frame error is unchanged (0.1123 $\\to$ 0.1128) across a 20$\\times$",
        "perturbation range.",
        "",
        "**Every model damps white noise within one or two steps** (growth ratios well below",
        "1). That is the licence the 4D-Var optimiser exploits: all observations lie at frames",
        "$\\ge 1$, where a noisy $x_0$ has already been attenuated 2–15$\\times$, so the cost",
        "barely penalises it. This is why the U-Net and FNO analyses in Part F look like",
        "speckle rather than flow.",
        "",
        "| | KAE | U-Net / FNO |",
        "|---|---|---|",
        "| control dimension | **128** (latent) | **32,768** (full field) |",
        "| can represent grid-scale noise? | no | yes |",
        "| analysis at $t_0$ looks like | flow | speckle |",
        "",
        "The KAE's latent parameterisation regularises implicitly; ACDM's diffusion prior does",
        "it explicitly. The physical-space methods have **no prior on $x_0$ at all**.",
    ),
    md(
        "## G.3 Optimisation failure, or identifiability failure?",
        "",
        "The decisive test: compare the 4D-Var objective at the recovered $x_0$ with its value",
        "at the **true** $x_0$.",
        "",
        "> $J(\\text{true}) < J(\\text{recovered})$ — the cost prefers the truth and the",
        "> optimiser did not get there: **ill-conditioning**, better optimisation helps.",
        ">",
        "> $J(\\text{true}) \\ge J(\\text{recovered})$ — the optimiser found a point the cost",
        "> likes at least as much as the truth: **unidentifiability**, no amount of",
        "> optimisation helps.",
    ),
    code("""print(P.identifiability_table(TRA))"""),
    md(
        "**The two baselines fail for opposite reasons.**",
        "",
        "**FNO is unidentifiable**: its recovered state fits the observations 3.7$\\times$",
        "*better* than the truth does. The observations genuinely do not determine $x_0$",
        "through the FNO, which is why it drives the objective lower than the KAE (1.18e-04 vs",
        "2.86e-04) while being 5$\\times$ worse at recovery.",
        "",
        "**U-Net is ill-conditioned**: the truth gives a 33$\\times$ lower cost, so the",
        "optimiser is simply stuck. That is fixable in principle.",
        "",
        "> **A limitation of this comparison, stated plainly.** These 4D-Var solves carry no",
        "> background term and no prior on $x_0$ — operational 4D-Var always does. The KAE and",
        "> ACDM get regularisation for free from their parameterisations; the U-Net and FNO get",
        "> none. A smoothness penalty on $x_0$ would likely improve them substantially, and",
        "> this campaign does not measure that.",
    ),
    md(
        "## G.4 The recovered trajectory, frame by frame",
        "",
        "Each method's analysis at $t_0$ rolled forward under its own dynamics. The",
        "interesting column is frame 0 — the analysis itself — and what the flow does to it.",
    ),
    code(
        """P.fig_spacetime(TRA, channel=3)
plt.suptitle("Analysis at $t_0$ rolled forward (density)", y=1.004)
plt.tight_layout(); plt.show()
print(P.spacetime_table(TRA))"""
    ),
    md(
        "**This is the mechanism, measured.** FNO's analysis error falls **4.8$\\times$ in a",
        "single frame** (0.1407 $\\to$ 0.0289) and U-Net's 2.3$\\times$, while **the KAE's error",
        "grows** (ratio 1.465) as a genuine state error should.",
        "",
        "The first observation is at **frame 1**. By then the physical-space methods' errors",
        "have already largely vanished, so the 4D-Var cost never sees them — which is why the",
        "optimiser is free to leave speckle at $t_0$. A ratio below 1 means the dynamics",
        "*destroy* the analysis error rather than propagate it, and that is precisely the",
        "signature of an error living in unobservable directions.",
    ),
    md("## G.5 Accuracy against cost"),
    code(
        """fig, ax = plt.subplots(figsize=(6.8, 4.4))
P.fig_error_vs_cost(TRA, ax=ax); plt.show()"""
    ),
    md(
        "## G.6 Forecasting after assimilation",
        "",
        "Part E starts from the exact state and so measures the propagator alone. This starts",
        "from each method's **own analysis**, which is what a real system has.",
    ),
    code(
        """fig, ax = plt.subplots(figsize=(7.4, 4.4))
P.fig_post_da(TRA, ax=ax); plt.show()"""
    ),
    md("---", "# H. Summary table"),
    code("""print(P.summary_table(TRA))"""),
    md(
        "---",
        "# I. Conclusions",
        "",
        "### 1. Forward accuracy and invertibility are different properties",
        "",
        "| | one-step forecast | analysis at $t_0$ |",
        "|---|---|---|",
        "| U-Net | **0.0023** | 0.2146 |",
        "| KAE | 0.0382 (16$\\times$ worse) | **0.0267** (8$\\times$ better) |",
        "",
        "The U-Net is much the better model of transonic flow and much the worse inverse. Any",
        "claim about a method's DA performance that rests on its forecast skill — or the",
        "reverse — is unsupported. This reproduces the Kuramoto–Sivashinsky result on 2-D",
        "fields at $128\\times64$, with different models and a different codebase.",
        "",
        "### 2. The KAE's advantage is its parameterisation, not its physics",
        "",
        "It optimises **128 latent numbers**; the U-Net and FNO optimise **32,768 physical",
        "values** with no prior. Since the dynamics damp grid-scale error within one or two",
        "steps (G.2) and every observation lies downstream of $t_0$, the physical-space",
        "optimiser is free to fill unobservable directions with speckle that the cost cannot",
        "see. The KAE cannot represent such a state at all.",
        "",
        "### 3. The two baselines fail for opposite reasons",
        "",
        "**FNO is unidentifiable** — its analysis fits the observations 3.7$\\times$ better than",
        "the truth does. **U-Net is ill-conditioned** — the truth gives a 33$\\times$ lower cost",
        "and the optimiser does not reach it. These need different fixes, and neither is a",
        "model-quality problem.",
        "",
        "### 4. ACDM wins on accuracy, KAE on robustness and cost",
        "",
        "| | ACDM | KAE |",
        "|---|---|---|",
        "| headline | **0.0057** | 0.0267 |",
        "| $\\delta_f=4$ | 0.1837 | **0.0421** |",
        "| 5% of sensors | 0.1690 | **0.0269** |",
        "| $\\sigma_y=0.3$ | 0.0342 | **0.0295** |",
        "| wall clock | 138 s | **43 s** |",
        "| cost vs horizon | grows | **flat over 50$\\times$** |",
        "",
        "ACDM is 4.7$\\times$ more accurate when observations are close, dense and clean, and",
        "degrades 34$\\times$ under sparsity and 48$\\times$ under $\\delta_f$. **The KAE is the",
        "only method flat on every axis**, moving only for $\\delta_f$ (5.5$\\times$).",
        "",
        "### 5. What this campaign does *not* establish",
        "",
        "* The 4D-Var baselines run **without a background term**, which operational 4D-Var",
        "  always has. The KAE and ACDM are implicitly regularised; the U-Net and FNO are not.",
        "  A prior on $x_0$ would likely close much of the gap, and that is untested here.",
        "* ACDM-ncn is run on the noisy-conditioned path **by request**, outside its training",
        "  regime; its 2.4–4.1 errors measure that mismatch, not the method.",
        "* Sweeps run at 500 iterations against the headline's 2000. The canonical point",
        "  appears in both, so the offset is measured, but the sweeps are not converged runs.",
        "* Everything is one Mach range (0.66–0.68) with $n=8$–12 problems. The extrapolation",
        "  regime is used only for tuning, never as a second test set.",
        "",
        "### 6. Provenance",
        "",
        "An earlier version of this campaign was **invalid**: the fields were transposed",
        "relative to what the checkpoints expect, inflating the U-Net's one-step error",
        "58$\\times$ and making every baseline worse than persistence. Those results are",
        "retired in `da_results_tra_INVALID_transpose/`. The gate that now prevents it is",
        "`verify_forward_skill` — a forward model must beat persistence — alongside",
        "`verify_adapters`, which checks the adapter against turbpred's own forward but",
        "**cannot** catch a bad input because both sides receive it.",
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
Path("visualize_da_tra.ipynb").write_text(json.dumps(nb, indent=1))
n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
print(
    f"wrote visualize_da_tra.ipynb: {len(cells)} cells "
    f"({n_md} markdown, {len(cells)-n_md} code)"
)
