# ruff: noqa: E741
"""Generate visualize_da_tra.ipynb as a PAPER PITCH, not a report.

One idea per slide: a title, a large figure, and a single takeaway line.  Detail that a
report would put in prose goes either into the figure or into a short table beneath it.
Every cell is tagged with a RISE/nbconvert slide type so the notebook can be presented
directly (`jupyter nbconvert --to slides`), and reads top-to-bottom otherwise.

Two rules the deck follows, both learned the hard way:

  * **A number written in markdown is a transcription and will go stale.**  Anything that
    can be read off a result file is drawn or printed from that file instead.  What is left
    is checked by `data_assimilation.tra.consistency_check`, which re-reads every remaining literal from
    its source and reports anything it cannot account for.
  * **A claim about a ratio should plot the ratio.**  The crossover and the
    forward-vs-inverse dissociation used to be hand-typed tables; they are figures now.

Never hand-edit the .ipynb.  Rebuild with `python -m data_assimilation.tra.build_da_tra_slides`.
"""

import json
from pathlib import Path

SLIDE, SUB, NOTE, SKIP = "slide", "subslide", "notes", "skip"


def md(*lines, t=SLIDE):
    return {
        "cell_type": "markdown",
        "metadata": {"slideshow": {"slide_type": t}},
        "source": "\n".join(lines),
    }


def code(src, t=SUB):
    return {
        "cell_type": "code",
        "metadata": {"slideshow": {"slide_type": t}},
        "execution_count": None,
        "outputs": [],
        "source": src,
    }


def guarded(src: str) -> str:
    """Run a cell so that a section still being computed degrades to one clear line.

    Sections take hours and the deck is rebuilt while they run; a raw traceback in the
    middle of a presentation is worse than a sentence saying which command produces the
    missing file, which is what `plots.NotRunYet` carries.
    """
    body = "\n".join("    " + l for l in src.strip("\n").splitlines())
    return (
        "try:\n" + body + "\nexcept P.NotRunYet as _e:\n"
        "    print('NOT ON DISK YET —', _e)\n"
        "except Exception as _e:\n"
        "    import traceback; traceback.print_exc()"
    )


def fig(title, takeaway, src, extra=None, note=None, how=None):
    """A slide: title, HOW IT WAS RUN, figure, one-line takeaway, optional detail.

    ``how`` is the protocol in one line -- what varies, what is held fixed, how many
    problems, how many iterations.  It goes ABOVE the figure on purpose: a reader who does
    not yet know what was run cannot judge what they are looking at, and a supervisor
    reading this cold should not have to reconstruct the design from the axis labels.
    """
    cells = [md(f"## {title}", t=SLIDE)]
    if how:
        cells.append(md(f"*{how}*", t="-"))
    cells.append(code(guarded(src), t="-"))
    if extra:
        cells.append(code(guarded(extra), t="-"))
    cells.append(md(f"> **{takeaway}**" + (f"\n>\n> {note}" if note else ""), t="-"))
    return cells


# Structure is a pitch, not a catalogue. Nine sections, each with one job, each titled
# with an ASSERTION rather than a question. The regime story is not a section: it is
# carried inside every claim, because a result shown on one Mach range is not a result.
cells = []

# ============================================================ 1. THE CLAIM
cells += [
    md(
        "# Forward accuracy does not predict invertibility",
        "",
        "### Data assimilation with frozen neural surrogates on 2-D transonic flow",
        "",
        "**Continuous KAE · U-Net · FNO · ACDM · ACDM-ncn**",
        "",
        "---",
        "",
        "The best one-step forecaster in this study is the **worst** inverse model.",
        "The ordering is close to reversed, it holds on **three independent Mach regimes**,",
        "and what decides it is the **dimension of the control**, not the physics.",
        "",
        "*Every model frozen. Every method sees the identical problem.*",
        t=SLIDE,
    )
]

cells += [
    code(
        """%load_ext autoreload
%autoreload 2
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import sys; sys.path.insert(0, ".")
from data_assimilation.tra import plots as P
TRA = Path("da_results_tra")
plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.facecolor": "white"})
print("regimes on disk:", P.available_regimes())""",
        t=SKIP,
    )
]

cells += [
    md(
        "## The problem, and the one design choice that matters",
        "",
        "| | |",
        "|---|---|",
        "| **Given** | noisy observations at $t_0+\\tau_1,\\dots,t_0+\\tau_N$ |",
        "| **Recover** | $u(t_0)$ — **never observed** |",
        "| **Using** | a frozen surrogate as the forward model |",
        "",
        "$$J(c)=\\frac{1}{N}\\sum_i \\left\\| M_i\\big[G(c,\\tau_i)-y_i\\big]\\right\\|^2$$",
        "",
        "Every method minimises the same cost over the same problems. They differ in **what",
        "$c$ is**:",
        "",
        "| method | control $c$ | dimension |",
        "|---|---|---|",
        "| **KAE** | a Koopman latent $z_0$ | **128** |",
        "| **U-Net, FNO** | the conditioning frame at $t_0$ | **32,768** |",
        "| **ACDM, ACDM-ncn** | the whole sampled trajectory | trajectory prior |",
        "",
        "> That column is the paper. Everything downstream follows from it.",
        t=SLIDE,
    )
]

cells += [
    md(
        "## How to read this deck",
        "",
        "**The metric.** Relative $L_2$ of the recovered field at $t_0$ against the truth,",
        "over all four channels, with the cylinder interior excluded. **Lower is better.**",
        "Typical values run from 0.005 (excellent) to 0.3 (the field is unrecognisable).",
        "",
        "**The five methods**, coloured consistently in every figure:",
        "",
        "| | what it is | what it optimises |",
        "|---|---|---|",
        "| **KAE** | Continuous Koopman autoencoder | a 128-d latent $z_0$ |",
        "| **U-Net**, **FNO** | turbpred's deterministic surrogates | the 32,768-d field at $t_0$ |",
        "| **ACDM** | turbpred's diffusion model, run as a score-based sampler | the whole trajectory |",
        "| **ACDM-ncn** | the same, trained *without* conditioning noise | *(diagnostic only — see §7)* |",
        "",
        "**The symbols**, all in frames:",
        "",
        "| | |",
        "|---|---|",
        "| $\\delta_f$ | lead to the **first** observation — the deciding variable |",
        "| $\\delta_l$ | the recovery horizon: lead to the **last** observation |",
        "| $N$ | how many observation times |",
        "| $W$ | how many past frames the analysis is asked to determine |",
        "| $T_L$ | the Lyapunov time — 88–195 frames here, see §7 |",
        "",
        "**Error bars are clustered by trajectory**, not by problem — there are only 4–6",
        "trajectories, so 24 cases are not 24 independent ones. They are ~3.8$\\times$ wider",
        "than a naive interval, deliberately. **Hatched bars are `gt_extrap`**, the tuning",
        "regime: a stress test, never a clean test set.",
        t=SLIDE,
    )
]

cells += fig(
    "The result, on all three regimes at once",
    "Identical ordering everywhere — but read the next slide before believing it.",
    """fig, ax = plt.subplots(figsize=(10.4, 5.2))
P.fig_headline_regimes(ax=ax); plt.tight_layout(); plt.show()""",
    extra="""print(P.regime_campaign_table())""",
    note="The whole campaign — headline, six sweeps, forecast, space-time, "
    "backward, reach, post-DA, off-grid and the extensions — was re-run "
    "**per regime**, 22 sections each, **zero failures**. Hyper-parameters "
    "are frozen across all three.\\n>\\n> `gt_extrap` is hatched throughout: "
    "it is outside the training Mach range *and* it is the regime the "
    "learning rates were tuned on, so it is a stress test, never a clean "
    "test set.",
    how="**Run:** the canonical schedule $[1,3,7,15,25]$, no noise, fully observed. 2000 4D-Var iterations, 4 sampler draws, 12 problems per regime. The whole campaign is re-run per regime; hyper-parameters frozen across all three.",
)

cells += [
    md(
        "## The headline is one point, and it sits in ACDM's corner",
        "",
        "The canonical schedule is $[1,3,7,15,25]$ — so $\\delta_f=1$, the first observation is",
        "at the very next frame. That is precisely the condition ACDM needs.",
        "",
        "Laid out over the whole design space, the record is nearly even, and it separates",
        "cleanly on **one variable**:",
        "",
        "| | ACDM wins | KAE wins |",
        "|---|---|---|",
        "| first observation at frame 1 | **7 of 7** | — |",
        "| first observation later, or sparse, or a window | — | **9 of 9** |",
        "",
        "> **Neither method dominates.** ACDM is 4.7$\\times$ better on the headline problem and",
        "> 3–7$\\times$ *worse* as soon as the first observation moves, the sensors thin out, or",
        "> more than one frame is asked for.",
        "",
        "> We lead with the headline because it is the standard benchmark, not because it is",
        "> representative. The rest of the pitch is about where that single number stops",
        "> holding.",
        t=SLIDE,
    )
]

cells += fig(
    "The full record",
    "ACDM wins 7, KAE wins 9 — and the `df` column says which, every time.",
    """print(P.win_loss_table(TRA, a="KAE", b="ACDM"))""",
    how="**Run:** one row per experiment already in the deck, each at its own settings; the `df` column is that experiment's lead to the first observation.",
)

# ============================================================ 2. THE SURPRISE
cells += [md("# The surprise: the ordering is close to reversed", t=SLIDE)]

cells += [
    md(
        "## Best forecaster, worst inverse model",
        "",
        "| | one-step forecast from the exact state | analysis at $t_0$ |",
        "|---|---|---|",
        "| **U-Net** | **0.0023** — the best | 0.2146 — the worst |",
        "| **KAE** | 0.0382 &nbsp; *(16× worse)* | **0.0267** &nbsp; *(8× better)* |",
        "",
        "These are the same frozen weights, on the same trajectories, an hour apart in the",
        "same script.",
        "",
        "> If forward skill were inverse skill, the next figure would be a rising line.",
        t=SLIDE,
    )
]

cells += fig(
    "Forward skill against inverse skill, every regime",
    "The spread is roughly perpendicular to the diagonal. On all three.",
    """P.fig_across_regimes(lambda d, ax: P.fig_paradox(d, ax=ax),
                     stem="A_headline", figsize_each=(5.6, 4.7))
plt.tight_layout(); plt.show()""",
    note="One-step forecast error on the x axis, analysis error on the y. A model "
    "that inverted as well as it predicted would sit on the dotted line.",
    how="**Run:** x-axis is the free-running one-step error from the EXACT state (no assimilation, 12 problems). y-axis is the analysis error from the headline. Same frozen weights, same trajectories.",
)

# ============================================================ 3. THE MECHANISM
cells += [
    md("# The mechanism: the dynamics hide the error before it is observed", t=SLIDE)
]

cells += fig(
    "The analysis error is destroyed within one frame",
    "FNO's error falls 4.9× between $t_0$ and frame 1. The first observation is at frame 1.",
    """P.fig_spacetime(TRA, channel=3)
plt.suptitle("Analysis at $t_0$ rolled forward under each model", y=1.004)
plt.tight_layout(); plt.show()""",
    extra="""print(P.spacetime_table(TRA))""",
    note="Read the ratio table underneath. The methods whose analysis is speckle "
    "have it **erased** before the cost can see it — FNO 0.1407 $\\to$ "
    "0.0289 in one frame, U-Net 0.2188 $\\to$ 0.0953. ACDM is the "
    "counter-example that completes the argument: its analysis is already "
    "right (0.0091) and its error only **grows**, 8.3$\\times$ over the "
    "window. Nothing is erased there because there is nothing to erase.",
    how="**Run:** each method's analysis at $t_0$ is rolled forward 30 frames under its own dynamics, free-running. 2000 iterations, 3 problems. Density channel shown.",
)

cells += fig(
    "What the analysis error actually looks like",
    "KAE: smooth, in the wake. U-Net/FNO: grid-scale speckle over the whole domain.",
    """P.fig_error_map(TRA, "G1_delta_f", channel=3)
plt.show()""",
    note="This is the component the dynamics damp. It costs the 4D-Var objective "
    "almost nothing, and with a 32,768-d control nothing else forbids it.",
    how="**Run:** analysis minus truth at $t_0$, canonical schedule. All panels share one colour scale except ACDM-ncn, whose residual is ~100$\\times$ larger and would otherwise flatten the rest to grey.",
)

cells += fig(
    "Two different failures, not one",
    "U-Net: the cost prefers the truth (stuck). FNO: it prefers its own answer (ill-posed).",
    """P.fig_across_regimes(lambda d, ax: P.fig_identifiability(d, ax=ax),
                     stem="A_headline", figsize_each=(5.4, 4.6))
plt.tight_layout(); plt.show()""",
    extra="""print(P.identifiability_table(TRA))""",
    note="Only one of these is something more iterations could fix. An "
    "unidentifiable cost cannot be optimised out of.",
    how="**Run:** the 4D-Var cost evaluated at each method's own analysis, and again at the TRUE $x_0$. 2000 iterations, 8 problems, per regime.",
)

cells += fig(
    "Why $\\delta_f$, $\\delta_l$ and $N$ have to be separated",
    "The confounded sweep is the $\\delta_f$ sweep wearing a disguise.",
    """fig, ax = plt.subplots(figsize=(8.0, 4.8))
P.fig_confounded(TRA, ax=ax); plt.tight_layout(); plt.show()""",
    note="The natural way to ask *how much do observations help* is to sweep one "
    "knob — observe every $k$-th frame. That moves $\\delta_f$, "
    "$\\delta_l$ **and** $N$ together, and produces a clean-looking curve "
    "that invites the reading *more observations is better*.\n>\n> It is "
    "not. Varying $N$ **alone** (crosses, all at $\\delta_f=1$) barely "
    "moves ACDM at all; varying $\\delta_f$ alone reproduces the confounded "
    "curve almost exactly. What looked like an observation-count effect is a "
    "lead-time effect throughout.",
    how="**Run:** observe every $k$-th frame out to 25, for $k=1\\ldots12$ — which moves $\\delta_f$, $\\delta_l$ and $N$ together. 500 iterations, 8 problems. Compared against G1 (varies $\\delta_f$ alone) and G5 (varies $N$ alone).",
)

cells += [
    md(
        "## The mechanism, stated",
        "",
        "| | KAE | U-Net / FNO |",
        "|---|---|---|",
        "| control dimension | **128** | **32,768** |",
        "| can represent grid-scale noise? | no | **yes** |",
        "| prior on $x_0$ | implicit (latent) | **none** |",
        "| analysis looks like | flow | speckle |",
        "",
        "> The dynamics damp grid-scale error before the first observation, so the cost cannot",
        "> see it — and with a full-field control, nothing else forbids it.",
        "",
        "### The KAE's advantage here is its parameterisation, not its physics.",
        "",
        "> We show this directly rather than inferring it: the KAE is the **worst forward",
        "> model** in the study, and its noise robustness is an **encoder** property — see",
        "> *Why you should believe this*.",
        t=SLIDE,
    )
]

# ============================================================ 4. THE BOUNDARY
cells += [
    md(
        "# The boundary: ACDM owns the easy corner, the KAE owns everything else",
        t=SLIDE,
    )
]

cells += fig(
    "Six sweeps, five methods",
    "ACDM wins when observations are close, dense and clean. The KAE is flat on every axis.",
    """fig, ax = plt.subplots(2, 3, figsize=(16, 8))
for a, s in zip(ax.ravel(), ["G6_single_obs", "G1_delta_f", "G4_delta_l",
                             "G5_n_obs", "G3_sparsity", "G2_noise"]):
    try:
        P.fig_sweep(TRA, s, ax=a)
    except P.NotRunYet as e:
        a.text(.5, .5, f"{s}\\nnot on disk", ha="center", va="center",
               transform=a.transAxes, fontsize=9, color="crimson"); a.set_axis_off()
plt.tight_layout(); plt.show()""",
    how="**Run:** each panel varies ONE quantity with the others pinned. 500 iterations, 8 problems per point, 4 sampler draws. Bars are the standard error over problems.",
)

cells += fig(
    "The crossover, as a ratio — and the axis that never crosses",
    "Above the line the KAE wins, below it ACDM. $\\delta_l$ is the control panel.",
    """P.fig_crossover(TRA, a="ACDM", b="KAE")
plt.tight_layout(); plt.show()""",
    note="Plotting the ratio rather than tabulating it: the claim *is* the "
    "ratio.\\n>\\n> **$\\\\delta_l$ never crosses.** With $\\\\delta_f$ pinned "
    "at 1, ACDM has a close observation at every horizon and varies by only "
    "1.12$\\\\times$ across a 10$\\\\times$ range. So ACDM's weakness is "
    "specifically the **lead to the first observation** — not the horizon, "
    "not the number of observations.",
    how="**Run:** the ratio error(ACDM)/error(KAE) at each point of the sweeps two slides back — same 500 iterations and 8 problems. Shaded by whichever method is ahead.",
)

cells += fig(
    "The deciding variable, isolated — with random observation sets",
    "ACDM wins at gap 1–2 and loses from gap 4. The crossover is between them.",
    """P.fig_across_regimes(lambda d, ax: P.fig_gap(d, ax=ax),
                     stem="G22_gap", figsize_each=(5.8, 4.5))
plt.tight_layout(); plt.show()""",
    note="**Every other experiment here uses the fixed schedule $[1,3,7,15,25]$**, "
    "so none of them can separate *ACDM is good* from *ACDM is good on that "
    "schedule*. Here the observation times are **redrawn at random** for each "
    "replicate with only the nearest one pinned, and the gap is the swept "
    "variable.\n>\n> The shaded band is the spread across schedules. "
    "**ACDM's is widest exactly where it wins** ($\\times$1.61 at gap 1 "
    "against $\\times$1.06–1.08 elsewhere): it depends on *which* "
    "observations it gets, not merely how near they are.\n>\n> Targets stay "
    "strictly before all observations because U-Net and FNO cannot represent "
    "a state earlier than their conditioning frame — a limit of the "
    "baselines, not a choice.",
    how="**Run:** 5 observation times drawn AT RANDOM per replicate, with only the nearest pinned at the swept gap; 3 independent schedules per gap, 8 problems each, 500 iterations. This is the only experiment whose schedule is not fixed.",
)

cells += fig(
    "The same crossover on every regime",
    "Not a property of one Mach range.",
    """P.fig_across_regimes(lambda d, ax: P.fig_sweep(d, "G1_delta_f", ax=ax),
                     stem="G1_delta_f", figsize_each=(6.0, 4.2))
plt.tight_layout(); plt.show()""",
    how="**Run:** the $\\delta_f$ sweep, repeated independently on each regime's campaign.",
)

cells += fig(
    "What is lost, not just how much",
    "The KAE degrades by smoothing. U-Net and FNO degrade into noise.",
    """P.fig_sweep_fields(TRA, "G1_delta_f", channel=3)
plt.tight_layout(); plt.show()""",
    how="**Run:** the recovered density field at $t_0$ at each $\\delta_f$, one example problem. Percentile colour limits; cylinder interior masked.",
)

cells += fig(
    "Gap-filling at 0.5% of sensors",
    "The KAE still returns a flow field where the others return noise.",
    """P.fig_recovery_gallery(TRA, stem="G3_sparsity", channel=3)
plt.tight_layout(); plt.show()""",
    how="**Run:** the sparsity sweep's fields, from 100% of grid points down to 0.5%. Sensors are a fixed random layout, identical across observation times.",
)

# ---- the window story, in one place instead of four --------------------------
cells += [
    md(
        "## An analysis is a state, not a frame",
        "",
        "Everything above scores **one frame**, $u(t_0)$. A useful analysis should determine",
        "its neighbourhood too — and all observations lie **strictly after** $t_0$, so any",
        "frame at $t_0-j$ must be inferred **backwards**.",
        "",
        "| method | control | can it represent $t_0-j$? |",
        "|---|---|---|",
        "| **KAE** | one latent $z_0$ | **any $j$** — via $e^{-Kj\\Delta t}$ |",
        "| **ACDM / ACDM-ncn** | the whole sampled trajectory | **any $j$** — it is a *smoother* |",
        "| U-Net, FNO | a single frame at $t_0$ | **never** |",
        "",
        "> Two different reasons for the same reach, and one hard structural limit. We ask it",
        "> three ways, because the first way flatters the KAE by construction.",
        t=SLIDE,
    )
]

cells += fig(
    "(1) One solve: how far does the control reach?",
    "Reach is structural. U-Net and FNO cannot express a state before $t_0$ at all.",
    """fig, ax = plt.subplots(1, 2, figsize=(13.0, 4.6))
P.fig_backward(TRA, axes=ax)
plt.tight_layout(); plt.show()""",
    note="1 of 8 frames for U-Net and FNO; the KAE and both samplers span the "
    "whole window from a single assimilation. **Empty panels in the next "
    "figure are the finding, not a rendering failure** — those states are "
    "not in the object being solved for.",
    how="**Run:** ONE assimilation on the canonical schedule, then the recovered control is asked for $t_0-j$, $j=0\\ldots7$. 2000 iterations, 8 problems. No re-solving.",
)

cells += fig(
    "(1) The reconstructed past states",
    "One KAE latent regenerates the whole window; U-Net and FNO have nothing to span it with.",
    """P.fig_backward_fields(TRA, channel=3)
plt.suptitle('Reconstructing states BEFORE $t_0$ (density)', y=1.004)
plt.tight_layout(); plt.show()""",
    how="**Run:** the same single solve as the previous slide, shown as fields. Empty panels are states the method's control cannot express at all.",
)

cells += fig(
    "(2) Re-solve at every target — the fair version",
    "ACDM wins $t_0$ by 4.9×. The KAE is ahead by $t_0-2$ and 3.0× ahead by $t_0-8$.",
    """P.fig_across_regimes(lambda d, ax: P.fig_reach(d, ax=ax),
                     stem="G18_reach", figsize_each=(5.8, 4.5))
plt.tight_layout(); plt.show()""",
    note="The analysis time moves to $t_0-j$ and **every method solves again**, so "
    "the reversal cannot be an artefact of the KAE's representational reach — "
    "that advantage is removed by construction here.",
    how="**Run:** the analysis time moves to $t_0-j$ and EVERY method solves again from scratch; observations stay put, so the lead grows with $j$. 500 iterations, 8 problems, per regime.",
)

cells += fig(
    "(3) Window-length scaling, $W = 1 \\ldots 32$",
    "Representability is exactly $W/W$ and $1/W$. And the KAE overtakes ACDM at $t_0$ itself.",
    """P.fig_window_scaling(TRA)
plt.tight_layout(); plt.show()""",
    how="**Run:** $W$ past frames requested, $W=1,2,4,8,16,32$. 500 iterations, 12 problems, on `gt_longer` — $W=32$ plus a 25-frame horizon does not fit in a 60-frame record.",
)

cells += [
    md(
        "## What the window costs, and who it changes the answer for",
        "",
        "Error **at $t_0$ itself**, relative to $W=1$:",
        "",
        "| $W$ | KAE | U-Net | FNO | **ACDM** | who leads at $t_0$ |",
        "|---|---|---|---|---|---|",
        "| 1 | 1.00 | 1.00 | 1.00 | 1.00 | **ACDM**, significantly |",
        "| 2 | 1.00 | 1.00 | 1.00 | **1.64** | **ACDM**, significantly |",
        "| 4 | 0.99 | 1.01 | 0.99 | **8.39** | KAE — but intervals *overlap* |",
        "| 8 | 0.99 | 0.98 | 1.00 | **8.65** | **KAE**, significantly |",
        "| 16 | 0.97 | 0.98 | 1.01 | **8.74** | **KAE**, significantly |",
        "",
        "1. **The 4D-Var solve really is independent of $W$** — flat to within 3% across a",
        "   16$\\times$ range. A measurement, not an assumption in the code.",
        "2. **ACDM's window is bought with accuracy at the analysis time**, paid almost",
        "   entirely between $W=2$ and $W=4$, then saturating near 8.7$\\times$.",
        "",
        "> **This is the strongest form of the claim.** Elsewhere the KAE only overtakes ACDM",
        "> at $t_0-j$. Here it overtakes **at $t_0$** — the quantity the headline scores — so",
        "> past about two frames of window, the headline ordering itself reverses.",
        "",
        "> The crossover is honest about its own significance: the means cross at $W=4$, but",
        "> the trajectory-clustered intervals still touch there. It is only **significant from",
        "> $W=8$**.",
        t=SLIDE,
    )
]

cells += fig(
    "Three more axes, same shape of answer",
    "Off-grid times, a fixed sensor budget spent two ways, and heavy-tailed noise.",
    """fig, ax = plt.subplots(1, 2, figsize=(13.2, 4.6))
try: P.fig_joint_sparsity(TRA, ax=ax[0])
except P.NotRunYet as e: ax[0].set_axis_off()
try: P.fig_misspec(TRA, ax=ax[1])
except P.NotRunYet as e: ax[1].set_axis_off()
plt.tight_layout(); plt.show()""",
    note="**Misspecified noise is a null result** and is reported as one: Laplace "
    "against Gaussian at matched variance, every ratio within **0.96–1.08**. "
    "Every method here assumes Gaussian observation error and none can tell "
    "the difference. Whatever separates these methods, it is not robustness "
    "to the shape of the error distribution.",
    how="**Run:** left — $N\\times$obs_frac held fixed, so every point costs the same number of scalar measurements. Right — Laplace vs Gaussian noise at MATCHED variance, 3 levels, paired. 500 iterations, 8 problems.",
)

cells += fig(
    "Observations between stored frames",
    "ACDM degrades 5.2× at half a frame. The KAE, which never snaps, moves 1.09×.",
    """P.fig_offgrid(TRA)
plt.tight_layout(); plt.show()""",
    note="**Two effects, and the sweep separates them.** At $+0.25$ the nearest "
    "stored frame is unchanged, so every snapping method assimilates the "
    "*same frames* and only the observed values move — yet **ACDM already "
    "degrades 2.4$\\times$**, because an interpolated field is not a physical "
    "state and its learned prior notices. At $+0.50$ the frames snap as "
    "well.\\n>\\n> The KAE evaluates $e^{K\\\\tau}$ at any real $\\\\tau$ and "
    "never snaps, but it is **not exempt** — it still fits an interpolant "
    "that is not on the true trajectory.",
    how="**Run:** observations placed at real lead times $\\tau+$shift and generated by linear interpolation between bracketing frames. 500 iterations, 8 problems.",
)

# ============================================================ 5. THE COST
cells += [
    md("# The cost: the KAE is flat in the horizon, the baselines are not", t=SLIDE)
]

cells += fig(
    "Cost against horizon",
    "$e^{K\\tau}$ is one matrix product. The others grow ~linearly.",
    """fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
P.fig_cost(TRA, ax=ax[0]); P.fig_cost(TRA, ax=ax[1], per_iteration=True)
plt.tight_layout(); plt.show()""",
    note="Left: seconds for one **complete** solve — the only cost defined for all "
    "five. Right: per 4D-Var iteration, which the samplers do not have.",
    how="**Run:** one method at a time, with warm-up and CUDA synchronisation, so no method's timing is inflated by another's memory pressure.",
)

cells += fig(
    "Accuracy at equal wall-clock",
    "Below ~20 s only the KAE can run at all. Above it, ACDM wins with ONE draw.",
    """P.fig_across_regimes(lambda d, ax: P.fig_budget(d, ax=ax),
                     stem="G15_budget", figsize_each=(5.8, 4.6))
plt.tight_layout(); plt.show()""",
    extra="""print(P.budget_table(TRA))""",
    how="**Run:** 4D-Var swept over iterations (50–2000), samplers over draws (1–8); both plotted against MEASURED wall clock. 8 problems.",
)

cells += [
    md(
        "## The decision rule",
        "",
        "| budget | choose | why |",
        "|---|---|---|",
        "| **low** | **KAE** | the samplers cannot run at all below their single-draw cost |",
        "| **higher** | **ACDM** | a single draw already wins; extra draws buy almost nothing |",
        "| **window $>2$ frames** | **KAE** | ACDM pays 8.7$\\times$ at $t_0$ for the window; the KAE pays nothing |",
        "| any | **not U-Net** | most expensive *and* least accurate, and worse at 2000 iterations than at 500 |",
        "",
        "> Numbers on the previous slide are printed from `G15_budget.json`, not typed here.",
        t=SLIDE,
    )
]

cells += fig(
    "Forecasting from the exact state, and from your own analysis",
    "At a 50-frame lead the KAE is only THIRD. Analysis accuracy does not survive the rollout.",
    """fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
P.fig_forecast(TRA, ax=ax[0]); P.fig_post_da(TRA, ax=ax[1])
plt.tight_layout(); plt.show()""",
    note="Right panel, forecasting from each method's OWN analysis. ACDM starts "
    "far ahead (0.0105 against the KAE's 0.0258) and **FNO finishes ahead**: "
    "at a 50-frame lead FNO 0.1432, ACDM 0.1589, KAE 0.1962. The KAE's "
    "advantage is in the *inverse* problem and does not carry into a long "
    "forward rollout — consistent with it being the weakest forward model.",
    how="**Run:** left — free-running from the TRUE state, the floor no analysis can beat. Right — free-running from each method's OWN analysis, on the 240-frame record.",
)

# ============================================================ 6. WHY BELIEVE IT
cells += [md("# Why you should believe this", t=SLIDE)]

cells += [
    md(
        "## Three regimes, three different senses of held-out",
        "",
        "| regime | Mach | what kind of held-out |",
        "|---|---|---|",
        "| `gt_interp` | 0.66–0.68 | in the training gap; **excerpt of the KAE's val split** |",
        "| `gt_longer` | 0.64–0.65 | in the gap, from `test.nc` — **clean for every model** |",
        "| `gt_extrap` | 0.50–0.52 | **outside training range**, and the **tuning regime** |",
        "",
        "**Hyper-parameters are frozen across all three.** `tuning.json` and",
        "`tuning_diffusion.json` are copied into each campaign, never recomputed. Re-tuning per",
        "regime would make the regimes incomparable.",
        "",
        "> **One result worth not over-reading:** every method does *better* on `gt_extrap`",
        "> despite it being outside the training range. That is a property of the flow — a",
        "> lower-Mach wake is a less demanding field to reconstruct — not evidence that",
        "> extrapolation is easy.",
        "",
        "> Two records hold only 60 frames, so there the post-DA horizon is cut 50 $\\to$ 24 and",
        "> $W=32$ cannot run. Both cuts are recorded in the output rather than papered over.",
        t=SLIDE,
    )
]

cells += [
    md(
        "## We found a leak in our own test set, and then measured it",
        "",
        "Comparing Mach ranges is not enough. Checked **frame by frame**, `gt_interp` — the",
        "regime the headline is measured on — is a **bit-identical excerpt of `val.nc`**, the",
        "split the KAE's checkpoint selection (`best_val_loss`) ran on.",
        "",
        "`gt_longer` is disjoint from **both** `train.nc` and `val.nc`. If the KAE's result",
        "were an artefact of that overlap, it would shrink there. It does not:",
        "",
        "| method | `gt_interp` (overlaps val) | `gt_longer` (clean) | ratio |",
        "|---|---|---|---|",
        "| **KAE** | 0.0267 | 0.0283 | **1.06×** |",
        "| U-Net | 0.2146 | 0.2179 | 1.02× |",
        "| FNO | 0.1281 | 0.1169 | 0.91× |",
        "| ACDM | 0.0057 | 0.0061 | 1.07× |",
        "",
        "The KAE's margin over U-Net is **8.0×** on the compromised regime and **7.7×** on the",
        "clean one.",
        "",
        "> **The asymmetry is real, was worth finding, and changes nothing about the ordering.**",
        t=SLIDE,
    )
]

cells += fig(
    "Every regime, with trajectory-clustered intervals",
    "The intervals are wider than they look like they should be — deliberately.",
    """fig, ax = plt.subplots(figsize=(10.0, 5.2))
P.fig_regimes(TRA, ax=ax); plt.tight_layout(); plt.show()""",
    extra="""print(P.regimes_table(TRA))""",
    how="**Run:** the canonical problem, 24 problems per regime drawn stratified across trajectories, 500 iterations.",
)

cells += [
    md(
        "## 24 cases are not 24 independent cases",
        "",
        "Each case is a *(trajectory, analysis time)* pair, and the pool is small:",
        "",
        "| regime | trajectories | cases drawn |",
        "|---|---|---|",
        "| `gt_interp` | **6** | 24 |",
        "| `gt_extrap` | **6** | 24 |",
        "| `gt_longer` | **4** | 24 |",
        "",
        "Two cases from one trajectory with overlapping 25-frame observation windows are",
        "strongly dependent. Treating them as independent gives an interval roughly",
        "$\\sqrt{n/n_\\text{traj}}$ too narrow — measured at **3.8×** on a realistic synthetic.",
        "",
        "> **So the trajectory is the unit of replication**: average within, then take the",
        "> $t$-interval across trajectory means, $\\mathrm{df}=n_\\text{traj}-1$. At df 3 the",
        "> critical value is **3.18**. That is the honest cost of a small pool.",
        "",
        "> **The fix is more trajectories, not more draws.** Cases are drawn stratified so the",
        "> replication available is not wasted.",
        t=SLIDE,
    )
]

cells += fig(
    "The models are not broken: rollout from the TRUE initial condition",
    "All baselines reproduce the wake. The KAE is the WORST forward model.",
    """P.fig_rollout_fields(TRA, channel=3, frames=(0, 6, 12, 18, 24))
plt.suptitle("Free-running rollout, density — no assimilation", y=1.004)
plt.tight_layout(); plt.show()""",
    how="**Run:** free-running rollout from the exact state, no assimilation anywhere. If a method could not do this, nothing downstream would mean anything.",
)

cells += [
    md(
        "## The KAE's noise robustness is an encoder property, not a dynamical one",
        "",
        "**Every frame of the conditioning window is perturbed**, not just $t_0$ — U-Net and",
        "FNO condition on one frame, the KAE and the samplers on two, and perturbing $t_0$",
        "alone would hand the $k{=}2$ models half their input clean.",
        "",
        "| model | its own frame 0 | error at amp = 0.2 |",
        "|---|---|---|",
        "| U-Net, FNO, ACDM | $x_0+\\varepsilon$ — the raw perturbed field | **0.0514** |",
        "| **KAE** | $\\mathrm{decode}(\\mathrm{encode}(x_0+\\varepsilon))$ | **0.0421**, vs 0.0422 unperturbed |",
        "",
        "The 128-d latent is a **256$\\times$ compression** of a 32,768-d field. Grid-scale white",
        "noise is not in its range, so the encoder projects it out *before any dynamics run*:",
        "a 0.2-amplitude perturbation moves its state by **$-0.0001$**.",
        "",
        "> So the figures show the **initial condition every model was handed** at frame 0, not",
        "> a model output — otherwise the KAE looks robust when it simply never received the",
        "> perturbation. Perturb its **own latent** instead and it moves to 0.0652, and the",
        "> perturbation then *grows*, much like the U-Net's.",
        t=SLIDE,
    )
]

cells += fig(
    "Perturbed rollout, and the like-for-like control",
    "Frame 0 is the initial condition every model was handed — not a model output.",
    """P.fig_perturbed(TRA, kind="white", where="state", channel=3, amps=(0.0, 0.2))
plt.tight_layout(); plt.show()""",
    extra="""print(P.perturbation_table(TRA, "white"))""",
    how="**Run:** white noise at 20% of each channel's standard deviation added to EVERY frame of the conditioning window. Frame 0 shows the input, not a model output.",
)

cells += fig(
    "Does the perturbation grow or decay?",
    "Left: the KAE never received it. Right: perturb its own control and it does.",
    """fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.2))
P.fig_perturbed_curves(TRA, kind="white", where="state", ax=ax[0])
P.fig_perturbed_curves(TRA, kind="white", where="control", ax=ax[1])
plt.tight_layout(); plt.show()""",
    how="**Run:** error above each method's own unperturbed run, normalised by its value at frame 0. Left perturbs the physical state; right perturbs each method's own control.",
)

cells += fig(
    "Is U-Net's answer just an unconverged optimiser?",
    "No — and the way it fails is the claim: the cost falls while the analysis gets worse.",
    """fig, ax = plt.subplots(figsize=(8.0, 4.8))
P.fig_convergence_check(TRA, ax=ax); plt.tight_layout(); plt.show()""",
    note="The KAE and FNO settle (last-fifth drift 0.2% and 0.9%). **U-Net reaches "
    "its best analysis, 0.1951, at iteration 858 and then degrades to "
    "0.2290** while its objective keeps falling.\n>\n> That is not a budget "
    "problem — more iterations make it *worse*. It is the signature of an "
    "objective whose minimum is in the wrong place, which is exactly what "
    "the identifiability test measures independently.",
    how="**Run:** the canonical problem at 2000 iterations, 8 problems, with the analysis error recorded every ~33 iterations.",
)

cells += [
    md(
        "## Three gates, and the bugs that needed all three",
        "",
        "| gate | checks | what it caught |",
        "|---|---|---|",
        "| `verify_adapters` | adapter == turbpred's own forward (**4.8e-07**) | nothing — it *cannot* catch a bad input |",
        "| `verify_forward_skill` | every model **beats persistence** | fields fed to the models **transposed** |",
        "| `consistency_check` | every number on a slide, re-read from its source | an off-grid sweep that never shifted; two sweeps overwritten; a sampler returning **NaN** in every row |",
        "| `verify_no_leak` | the recovered frame is never an input | — (it passes, which is the point) |",
        "",
        "**The leak gate is the decisive one for the samplers.** ACDM reaches $t_0$ through a",
        "learned trajectory prior, so *does it just copy an observation?* is a fair question.",
        "We corrupt the ground truth at $t_0$ by 5$\\times$ the field standard deviation,",
        "re-solve with identical seeds, and require every analysis back **bit-identical**.",
        "All five are. A method that had used the target could not do that.",
        "",
        "> The transpose bug passed the first gate for hours: it compares two computations on",
        "> the *same* input, and both were consistently wrong. The U-Net's one-step error was",
        "> **0.1472** instead of **0.0023** — a 64× inflation — and every deterministic baseline",
        "> came out *worse than doing nothing*.",
        "> Retired results and a write-up: `da_results_tra_INVALID_transpose/README.md`.",
        t=SLIDE,
    )
]

cells += [
    code(
        """import subprocess
for mod in ["data_assimilation.tra.verify_adapters", "data_assimilation.tra.verify_forward_skill"]:
    r = subprocess.run(["python", "-m", mod, "--regimes", "tra"],
                       capture_output=True, text=True, cwd=".")
    print("\\n".join(l for l in r.stdout.splitlines()
                     if any(k in l for k in ("PASS", "FAIL", "MATCH", "PERSISTENCE"))))""",
        t=SUB,
    )
]

cells += fig(
    "Every number in this deck, re-read from the file that produced it",
    "UNVERIFIED means a number appears in markdown that no check can trace.",
    """fig, ax = plt.subplots(figsize=(7.8, 3.2))
P.fig_consistency(TRA, ax=ax); plt.tight_layout(); plt.show()""",
    extra="""print(P.consistency_report(TRA))""",
    note="`FIGURE-ONLY` is a verified number that only a figure shows — the "
    "preferred state.",
    how="**Run:** `python -m data_assimilation.tra.consistency_check`, live, against the current results.",
)

cells += fig(
    "The training specification",
    "Read from the checkpoints and the .nc files — nothing typed in.",
    """print(P.training_spec_table(TRA))""",
    how="**Run:** read directly from the checkpoints, the run YAMLs and the .nc files.",
)

# ============================================================ 7. WHAT WE DO NOT CLAIM
cells += [md("# What we are not claiming", t=SLIDE)]

cells += [
    md(
        "## This is the SDA *algorithm* on ACDM's denoiser — not the paper's *sampler*",
        "",
        "**Faithful to Rozet & Louppe (2023), verified line by line:** Algorithm 2 composition,",
        "Tweedie $\\hat x_0=(x-\\sigma\\varepsilon)/\\mu$, the Eq. 15 covariance",
        "$\\Sigma_y+(\\sigma^2/\\mu^2)\\Gamma$, and Algorithm 4's corrector formula.",
        "",
        "The blanket reinterpretation is legitimate: ACDM's loss is computed over **all 15",
        "channels**, so its U-Net genuinely predicts $\\epsilon$ for the whole 3-frame window.",
        "",
        "| # | deviation | forced? |",
        "|---|---|---|",
        "| 1 | blanket $k=1$ (3-frame window), not $k=2$ (5 frames) | yes — ACDM takes exactly 3 frames |",
        "| 2 | ACDM's DDPM schedule, not cosine VP | yes |",
        "| 3 | DDPM ancestral predictor, not the exponential integrator | yes |",
        "| 4 | **20 denoising steps**, not 256 | yes |",
        "| 5 | ACDM trained with **smooth-L1**, not squared $L_2$ | yes |",
        "| 6 | **corrections = 0** | **no** |",
        "| 7 | $\\Gamma=0.3$ not $10^{-2}$; $\\sigma_y$ floored at 0.05 | **no** |",
        "",
        "> Against the paper's own **2-D fluid** setup ($k=2$, $C=1$, 256 steps) we are one step",
        "> narrower in blanket, corrector off, ~13$\\times$ fewer steps. **ACDM's number here is a",
        "> lower bound on what SDA can do on this problem, not SDA's best.**",
        "",
        "> **(5) is subtle.** Score matching needs the squared-$L_2$ objective for",
        "> $\\epsilon_\\phi\\to\\mathbb{E}[\\epsilon\\mid x(t)]$. Huber's minimiser is not the",
        "> conditional mean, so the score is mildly biased and everything downstream inherits it.",
        t=SLIDE,
    )
]

cells += fig(
    "Algorithm 4's corrector is harmful here, not mis-scaled  *(ACDM only — by design)*",
    "Even at the smallest step, one corrector step is far worse.",
    """fig, ax = plt.subplots(figsize=(7.0, 4.4))
P.fig_corrector(TRA, ax=ax); plt.show()""",
    note="**Single-method by design:** Algorithm 4's corrector exists only in the "
    "sampler. It sits in tension with deviation (4): the corrector exists "
    "*because* discretisation error accumulates, and 20 steps accumulate far "
    "more than 256 — yet the corrector is what is disabled. The $\\\\tau$ "
    "sweep says it hurts here regardless; that is an empirical finding, not "
    "a resolution.",
    how="**Run:** the corrector's step size $\\tau$ and count $C$ swept on the VALIDATION regime, so the choice never touches a reported number.",
)

cells += fig(
    "A background term does not rescue U-Net or FNO  *(those two — by design)*",
    "It removes the speckle and the error barely moves — then gets worse.",
    """P.fig_background(TRA)
plt.tight_layout(); plt.show()""",
    extra="""print(P.background_table(TRA))""",
    note="**Two-method by design:** the KAE's latent already regularises and the "
    "samplers carry a learned prior, so only U-Net and FNO lacked one. The "
    "right panel is the interesting one: roughness falls monotonically with "
    "the weight — the term *does* suppress the speckle — while accuracy "
    "improves a few percent and then degrades. **The speckle is not the "
    "whole problem**; these analyses are also wrong in the resolved scales.",
    how="**Run:** a climatology background term added to the 4D-Var cost at five weights. 500 iterations, 8 problems. Right panel is spectral roughness above quarter-Nyquist.",
)

cells += fig(
    "Is ACDM's posterior calibrated?  *(samplers only — by design)*",
    "A sampler must be judged on spread, not only on mean error.",
    """P.fig_calibration(TRA)
plt.tight_layout(); plt.show()""",
    extra="""print(P.calibration_table(TRA))""",
    note="**Single-method by design:** a rank histogram needs draws, which the "
    "4D-Var methods do not produce.",
    how="**Run:** many draws per problem; the truth's rank among them is histogrammed. Flat means calibrated, U-shaped over-confident, dome-shaped under-confident.",
)

cells += [
    md(
        "## ACDM-ncn measures a broken score, not a method",
        "",
        "It is run on the noisy-conditioned path **by explicit request**. In `model_diffusion.py`",
        "its conditioning channels are trained against `noise = cat(cond, dNoise)` — that is, to",
        "**reproduce the clean conditioning frames**, not to predict $\\epsilon$.",
        "",
        "> Using them as a score uses output channels that were **never trained to be one**.",
        "",
        "> Its magnitude is correspondingly unstable: **2.70, 2.43 and 1.24** on the *same*",
        "> regime across three runs, each with a within-run interval of a few percent. That",
        "> tight interval measures spread across problems, not across sampler realisations.",
        "> **So we quote its order — 40–400× every other method — never a value.**",
        t=SLIDE,
    )
]

# ============================================================ 8. TAKEAWAY
cells += fig(
    "Every horizon here is below one Lyapunov time",
    "$T_L \\approx 88$–195 frames. The canonical horizon is 0.13–0.29 $T_L$.",
    """print(P.lyapunov_table())""",
    note="Temporal axes in frames are arbitrary until divided by a timescale of "
    "the flow. The KS campaign measured this with a twin experiment on its "
    "own solver; **the transonic solver is not in this repository**, so this "
    "is Rosenstein's method on the stored trajectories — a *lower* bound at "
    "coarse sampling.\n>\n> **This is a short-horizon study.** KS reached "
    "3.1 $T_L$; nothing here exceeds 0.57. Chaos has not had time to destroy "
    "the information the observations carry, which is why the differences "
    "between methods are about **representation** rather than "
    "predictability.\n>\n> It also explains why `gt_extrap` is *easier* "
    "despite being outside the training range: its $T_L$ is 195 frames "
    "against `gt_longer`'s 88, so it is simply a less chaotic flow.",
    how="**Run:** Rosenstein's method on the stored trajectories — nearest-neighbour pairs separated in time, followed forward, slope of mean $\\ln$ separation.",
)

cells += [md("# Takeaway", t=SLIDE)]
cells += [code(guarded("""print(P.summary_table(TRA))"""), t=SUB)]

cells += [
    md(
        "## What this study establishes",
        "",
        "1. **Forward accuracy does not predict invertibility.** The best one-step propagator",
        "   is the worst inverse model, and the ordering is close to reversed — on **three**",
        "   independent Mach regimes.",
        "2. **The control's dimension decides it, not the physics.** 128 latent numbers cannot",
        "   represent the speckle a 32,768-d control can, and the dynamics erase that speckle",
        "   before the first observation, so the cost never penalises it.",
        "3. **The baselines fail differently**: FNO is *unidentifiable* (its cost prefers its own",
        "   answer), U-Net *ill-conditioned* (its cost prefers the truth and cannot reach it).",
        "   Only one of those is fixable with more iterations.",
        "4. **ACDM owns a well-defined corner** — close, dense, clean observations, and a window",
        "   of at most two frames. The KAE owns everything else, including the neighbourhood of",
        "   the analysis and, past $W>2$, the analysis itself.",
        "5. **But the KAE's advantage is inverse-only.** It is the weakest forward model here",
        "   and only third at a 50-frame post-DA lead.",
        "",
        "### What would change our minds",
        "",
        "* A blanket at the paper's own $k=2$ with correctors on, at 256 steps — ACDM's number",
        "  here is a lower bound.",
        "* More trajectories. The pool is 4–6, and every interval is limited by that, not by",
        "  the number of draws.",
        "* A regime where the analysis error is *not* damped before the first observation; the",
        "  whole mechanism is downstream of that.",
        "",
        "> **`inc` is blocked** on a superseded checkpoint architecture. The code is",
        "> regime-general and a modern checkpoint drops in unchanged.",
        t=SLIDE,
    )
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
        "celltoolbar": "Slideshow",
        "rise": {"theme": "white", "transition": "none", "scroll": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path("visualize_da_tra.ipynb").write_text(json.dumps(nb, indent=1))
nm = sum(1 for c in cells if c["cell_type"] == "markdown")
print(
    f"wrote visualize_da_tra.ipynb: {len(cells)} cells ({nm} markdown, "
    f"{len(cells)-nm} code), slide-tagged"
)
