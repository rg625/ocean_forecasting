# Research-grade audit of the continuous-time Koopman autoencoder

Audit date: 2026-10-19. Audited code: `rares/updated_model` @ `79713ce` plus the uncommitted
config edits present in `/home/rg625/mnt/ocean_forecasting` that day. All changes live on
branch `audit/kae-research-audit` (worktree `/home/rg625/mnt/ocean_forecasting_audit`). The
original checkout and branch were not modified.

Companion files:
- `issues.json`: machine-readable issue tracker.
- `experiments.yaml`: ablation registry with decision rules.
- `PROGRESS_LOG.md`: commands, tests, limitations.
- `LITERATURE_REVIEW.md`: Phase IV, with sources.
- `results/`: raw JSON from every measurement quoted here.
- `scripts/`: the code that produced them.
- `tests/`: regression tests.

Evidence labels:
- **[measured]**: produced in this audit, with the result file named.
- **[code]**: read directly in the source.
- **[subagent]**: found by a read-only delegated audit. Spot-checked where marked.
- **[hypothesis]**: not yet tested.

---

## 1. Executive summary

**What the model is.** A convolutional encoder (ResNet + CBAM + single-query attention pooling)
maps a 2-frame window to z ∈ R^128. A continuous-time linear generator A = skew(W) + sym(D) is
stepped with one RK4 step of fixed dt = 0.1 per frame. A resize-conv decoder maps z back to
fields. It is trained and evaluated on ACDM transonic cylinder flow (`tra`, conditioned on
Mach), 2-layer QG turbulence and 1-D Kuramoto–Sivashinsky. None of these are observational
ocean data. QG is the only ocean-relevant system.

**Code quality.** The code runs, and checkpoints round-trip exactly. However, several components
silently do something other than what their configuration and the papers say:

- **Conditioning is dead.**
  - The LoRA hypernetwork is initialised with U = V = 0, so it never receives a gradient. Its
    last layer is exactly 0.0 after 200 epochs.
  - The encoder's `adaln` conditioning is not implemented, and the decoder ignores the condition.
  - Changing the Mach input changes the output by exactly 0.0 [measured]. The "physics-conditioned"
    TRA model is therefore unconditioned.
- **The current evaluation is off by one frame.** Each prediction is scored against the
  previous ground-truth frame [code + measured].
- **Configured settings are silently ignored.** `padding_mode` (circular for KS/QG) and
  `spectral` are dropped by the encoder/decoder, and the stability regulariser is
  `.detach()`ed [code + tests].
- **Debug instrumentation is always on.** It makes every training step **16.5× slower**
  (4.62 s → 0.28 s) [measured].
- **A clean clone cannot run any experiment.** The gitignore rule `*.yaml` hides 35 Hydra
  configs [code].

**Mathematical soundness.** Two claims hold up:

- RK4 at dt = 0.1 is numerically indistinguishable from exp(dt·A) for the trained generators:
  ‖P4(dt·A) − exp(dt·A)‖₂ ≤ 1.6·10⁻⁴ [measured]. The "RK4 vs exact exponential" concern is not a
  practical problem at the trained step.

Three do not:

- The generator is **not constrained to be stable**. The QG checkpoint has **63 of 128 eigenvalues
  with positive real part**, spectral abscissa +0.40 [measured], so its long rollouts must diverge.
- The TRA generator is stable but strongly non-normal: 16× transient amplification.
- On TRA, every oscillatory mode near the vortex-shedding frequency decays with an e-folding time
  of 36–41 steps, whereas the true flow is a persistent limit cycle. Long forecasts therefore
  regress to the mean: predicted fluctuation energy falls to 25% of the truth by step 240
  [measured].

**Can the results be trusted?**

*Short horizons on TRA:* yes. The KAE is genuinely skilful, with anomaly correlation ≥ 0.6 up to
about 27–29 steps (≈ one shedding period) on `interp`/`longer`, and beyond 58 steps on
`extrap` [measured].

*The 240-step headline claim (KAE has the lowest MSE of 15 models): not as stated.*
- On that case a constant field (the training-set time mean) scores MSE 15.8·10⁻³.
- A 34-second, 128-mode POD + linear-map fit with no neural network scores 13.4·10⁻³.
- The KAE scores 16.9·10⁻³ (realigned), or 14.9·10⁻³ in the paper.
- ACDM scores 22.5·10⁻³ [measured, `results/benchmark_table_tra.json`].
- At 240 steps, pointwise MSE rewards forecasts that have lost phase and regressed to the mean.
- The paper's 14.9 and 5.2 cannot be reproduced from any checkpoint on disk [subagent].

*QG results are compromised:*
- The QG test file's first 500 steps are the training trajectories (sims 0–9), up to a per-file
  z-score rescale (residual 6·10⁻⁸) [measured].
- The QG config validated on the test file [code].
- The QG model is exponentially unstable [measured].

**Largest sources of lost performance, in order:**

1. **Long horizons: dissipative or unstable linear latent dynamics.** The model cannot represent
   the persistent shedding, and its training horizon (8 steps) and checkpoint criterion
   (8-step loss) never see the failure.
2. **Short horizons: the autoencoder reconstruction floor.** Reconstruction RMSE is 0.13–0.17 in
   normalised units, which is worse than a 128-mode linear POD on `interp`/`longer` (0.12)
   [measured].
3. **Dead conditioning.** The model must infer Mach from two frames.
4. **Wasted compute.** 16.5× of every training step went to debug hooks, which limited how many
   experiments could be run.

**Strongest alternative architecture, given the evidence.** Keep a linear latent model but:

- make it **structurally stable with near-neutral oscillatory modes**: a diagonal complex or
  Hurwitz-constrained generator, propagated exactly;
- train it on **horizons of at least one shedding period** with a stop-gradient latent target;
- benchmark it against **POD-DMD and the ACDM U-Net baselines (U-Net_tn / U-Net_ut)**.

U-Net_tn / U-Net_ut beat ACDM on Tra_ext/Tra_int in the ACDM paper ([LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) §3).

For QG the linear-latent family is the wrong tool for pointwise skill beyond a few decorrelation
times. Evaluate it on statistics (Thermalizer protocol), or pair it with a nonlinear corrector.

**Five highest-priority actions:**

1. Re-score every KAE result with the fixed evaluator and add persistence, climatology and
   POD-DMD rows to every table.
2. Rebuild the QG test split from independent seeds, and give TRA a validation split that is not
   one of the reported test cases.
3. Retrain the conditioned TRA model with the LoRA fix, or drop the conditioning claim. Verify
   Mach sensitivity > 0.
4. Replace the unconstrained generator with a stable parameterisation propagated by
   `matrix_exp`, and train on ≥ 25–60-step horizons with checkpoint selection on the deployment
   horizon (E1–E3 in `experiments.yaml`).
5. Report horizon-resolved RMSE/ACC, fluctuation-energy ratio and spectra, with ≥ 3 seeds.

---

## 2. Repository map (what matters for the KAE)

| Path (under `koopman_autoencoder/`) | Role |
|---|---|
| `train.py` | Hydra entry point. Builds datasets, model, AdamW (wd 1e-3), CosineWarmup and Trainer. Saves `final_model.pth` with the full config |
| `models/autoencoder.py` | `KoopmanAutoencoder`: encode → latent rollout → decode. Also `forward_theoretical` (matrix_exp), `forward_stabilized` (energy injection) and `assimilate` |
| `models/modules.py` | Koopman operators: `ContinuousKoopmanOperator` (skew+sym, RK4), `DiscreteKoopmanOperator` (I+skew+sym, learned backward map), `AzencotKoopmanOperator`, plus the LoRA hypernetwork |
| `models/networks.py` | `ConvEncoder`, `ConvDecoder`, `HistoryEncoder` (per-frame encoder + transformer), CBAM, attention pooling |
| `models/loss.py` | `KoopmanLoss`: reconstruction, prediction, physics (Sobolev/temporal/FFT), latent and stability terms |
| `models/azencot_loss.py` | Loss for the discrete/Azencot operators |
| `models/dataloader.py` | NetCDF datasets, hard-coded normalisation tables, samplers, collate |
| `models/trainer.py` | Training loop, validation, checkpointing |
| `models/utils.py` | `load_datasets`, `load_checkpoint`, config loading |
| `evaluate.py`, `models/metrics_utils.py` | TRA/INC rollouts written to `autoreg_pde_diffusion/src/results/sampling/<case>/*.npz` |
| `../autoreg_pde_diffusion/src/plot_loss_all.py` | Produces the MSE tables from those npz files |
| `qg_eval.py`, `evaluate_ks.py` | QG/KS evaluation. `qg_eval.py` is currently broken [subagent] |
| `data_assimilation/{ks,tra}` | Latent 4D-Var vs SDA/UNet experiments. Uses `best_model.pth` |
| `configs/` | Hydra configs. **35 of them are untracked** because of `.gitignore:17 *.yaml` |
| `data/acdm/128_tra`, `data/qg`, `data/ks` | Datasets (68 GB local) |
| `model_outputs_*` | Checkpoints. The evaluated TRA model is `model_outputs_tra/continous_linear_128/run-20260821_031121/checkpoints/epoch_199.pth`; DA uses `best_model.pth` from the same run [subagent] |

Environment: conda env `samudra` (Python 3.10.13, torch 2.5.1+cu121, tensordict 0.9.1) and one
RTX 4090. During the audit the GPU was shared with another of your jobs (`diagnostics.py`, about
99% utilisation), so absolute timings are inflated. A/B comparisons were interleaved.

---

## 3. Architecture reconstruction

Notation: x_t ∈ R^{C×H×W} (TRA: C = 4 [v_x, v_y, p, ρ], H×W = 64×128), z ∈ R^d (d = 128),
B = batch, T = horizon.

**Data → model** [code, `dataloader.py`]:
- Fields are z-scored with **hard-coded** per-variable constants (`TRA_MEAN/STD`, etc.,
  `dataloader.py:25-60`), chosen by variable name, and the obstacle cells are zeroed.
- TRA train frames are 0.05 apart. `subsample: 2` gives Δt = 0.1, which matches the model's
  hard-coded `base_dt = 0.1` (`autoencoder.py:404`).
- Windows: 2 input frames, 8 target frames, all overlapping start positions within each
  simulation (exhaustive).

**Encoder** (`networks.py:211-340`):

    z_p = Pool( CBAM( ResNet( [x_t, grid] ) ) )            # [B, d]
    z_h = Transformer( Pool(...)(x_{t-1}) )[:, -1]         # history token
    z_0 = (z_h + z_p) / 2                                  # autoencoder.py:227

- Pooling is a single learned query attending over the 8×16 feature map, so z is a convex
  combination of per-pixel value vectors.
- The first conv is 3×1 even for 2-D fields (`networks.py:269-275`).
- `cond_type: adaln` computes an embedding that is **never used** (`networks.py:323-337`).

**Latent dynamics** (`modules.py:300-382`):

    A = ½(W − Wᵀ) + ½(D + Dᵀ) [+ LoRA(c)],   z_{k+1} = P4(dt·A) z_k,
    P4(X) = I + X + X²/2 + X³/6 + X⁴/24,   dt = 0.1

- D is an *unconstrained* symmetric matrix (`modules.py:231-233, 340-349`), so A can have
  Re λ > 0.
- The code comment "Stable Linear Mode" is not enforced.
- Initialisation is W orthogonal and D = 0, which gives a neutral start.

**Decoder** (`networks.py:348-452`):

    x̂ = Conv( [ResBlock → Upsample×2 → Conv]×3 ( SiLU(Linear(z)).view(256, 8, 16) ) )

- 2-D dropout (p = 0.15) is applied to the last feature map during training.
- The decoder's `expansion_map` (14k parameters) is unused.

**Loss as implemented** (`loss.py:489-569`; TRA uses `loss/tra.yaml`: loss_type l1, α = β = 1,
physics weight 1 with all γ = 1, stability weight 0):

    L = Σ_v mean(w ⊙ |D(z̃_0) − x_t|)                          # recon, L1, w = 1 + 2 tanh|∇x|
      + α Σ_v mean_t mean((D(z_k) − x_{t+k})²)                 # prediction: ALWAYS squared
      + phys(D(z_k), x_{t+k})                                  # temporal diff, Sobolev, 2-D FFT
      + β [ ‖z_k − E_p(x_{t+k})‖² + linearity(fwd+bwd, 1 step)
            + ‖Δ²_t E_p(x_{t+k})‖² + 2·FFT_t(z, E_p(x)) ]       # latent terms, encoder NOT detached
      + 0 · stability                                          # detached (BUG-03)

- z̃_0 = z_0 + 0.01·ε is used for reconstruction only (`autoencoder.py:496`).
- In training, every rollout step also adds 0.005·ε (`autoencoder.py:314-315`).
- Inputs get 0.005·ε noise and are clamped to ±5 (`trainer.py:264-287`).
- **Latent chart mismatch.** The latent targets use `present_encoding` (a single frame,
  `trainer.py:239`), while the rollout starts from the history-averaged z_0. They are different
  maps (ENC-01).

**Optimisation and selection:**
- AdamW (lr 2e-4, wd 1e-3), linear warmup 20 epochs then cosine decay to 2e-5 over 180.
- Gradient clipping at 10.
- Validation every 10 epochs. `best_model.pth` is chosen by validation `total_loss` at the
  8-step training horizon.
- Forecasts use `epoch_199`, while DA uses `best_model`, whose validation loss bottomed around
  epoch 40 [subagent].

**Inference** (`evaluate.py`, `metrics_utils.py:207`): plain `model(x, T)`, i.e. the same RK4
rollout, under `eval()`/`inference_mode`. The paper describes exp(K·τ). Both maps agree to
≤ 1.6·10⁻⁴ for this checkpoint.

Shapes per training step (B = 32, TRA): input 4×[32, 2, 64, 128] → z [32, 128] → rollout
[32, 8, 128] → decode [256, 4, 64, 128]. There are 11.17 M trainable parameters. Peak memory is
6.7 GB at B = 16 in fp32 [measured].

---

## 4. Confirmed bugs

Severity reflects impact on the *validity of reported results*. Every item in this section has a
regression test in `tests/test_audit_bugs.py`, unless it is a data or protocol item, which cannot
be unit-tested. All 10 tests fail on `79713ce` (`results/tests_before_fixes.txt`) and pass on the
audit branch (`results/tests_after_fixes.txt`).

| ID | Sev. | Location (original code) | Defect → consequence | Status |
|---|---|---|---|---|
| **BUG-01** | Critical | `evaluate.py:149-154` | Commit `a5f4796` stitches `[last input, pred[:-1]]`, so saved frame k holds x̂_{t+k} but is scored against x_{t+k−1}. The current KAE npz files are all misaligned, while the January ones are aligned: the tables mix conventions. Effect (same convention, ×10⁻³): interp 7.62 → **6.37**, extrap 1.82 → **1.59**, longer 16.97 → **16.91** [measured]. Verified independently: frame 0 equals GT[1] to 1.5·10⁻⁷ on fluid cells | **Fixed** `0bfe883` (stitch all input frames; also correct for 1-frame inputs) |
| **BUG-02** | Critical | `modules.py:79-82` (`KoopmanHypnet.reset_parameters`) used by the LoRA paths (`:236-242, 461-463, 634-637`) | U = V = 0, so ∂(UVᵀ)/∂U = V = 0 and ∂/∂V = U = 0, and the hypernetwork never moves. The evaluated checkpoint's last layer is exactly 0.0 and all hypernetwork gradients are zero [measured, `results/spectrum_tra_run20260821_ep199.json`, `results/smoke_profile_baseline.json`]. With the encoder's `adaln` unimplemented and the decoder ignoring the condition, **Mach sensitivity = 0.0** [measured] | **Fixed** `55f7512` (U random, V = 0: identical initial dynamics, and gradient flows from step 1; the whole chain trains from step 2). Needs retraining. The `adaln` gap is not fixed: it is an architectural choice (E5) |
| **BUG-03** | Medium | `loss.py:563` | `total_loss + l_stab.detach()` makes `stability_weight` a no-op, although it is set to 0.1 in `loss/qg.yaml` | **Fixed** `8e9645c` |
| **BUG-04** | Low | `trainer.py:72-75` | `save_latest_every` receives `num_visual_batches` | **Fixed** `b9a66eb` |
| **PERF-01** | High | `trainer.py:579-586` | Autograd anomaly mode is turned on every epoch and a NaN hook (one host sync each) sits on all 165 modules: **4.62 s vs 0.28 s per step** [measured, interleaved ×3]. The non-finite-loss guard already exists | **Fixed** `b9a66eb`. Now opt-in via `training.debug_nan_checks=true` |
| **BUG-05** | High (QG) | `dataloader.py:109-120`, `utils.py:204-236` | QG variables `q_1`/`q_2` fall through to the INC table and raise `KeyError`: QG training cannot run with the current code [measured]. Each split also builds its own normaliser | **Fixed** `63fbfb5`. Unknown variables use training-split statistics, and val/test share the train-fitted normaliser. TRA/KS/INC behaviour is unchanged |
| **BUG-06** | Medium | `networks.py:225-283, 357-419`, `autoencoder.py:123-133` | `conv_kwargs` (`padding_mode: circular` for KS/QG, `reflect` for TRA) and `spectral: false` are swallowed by `**kwargs`. Res blocks are always spectral-normed and zero-padded. KS's "CRITICAL FIX: periodic boundaries" never took effect | **Fixed (opt-in)** `175c7a4`. The `honor_conv_kwargs` flag defaults to False so that existing checkpoints load: strict load, bit-identical outputs [measured, `scripts/compat_check.py`]. A warning is now emitted when settings are ignored |
| **REPRO-01** | High | `.gitignore:17` | `*.yaml` hides 35 model/loss/training/scheduler configs that the tracked experiment configs need | **Fixed on the audit branch** `9897d96` (configs tracked). Your uncommitted `.gitignore` also ignores `*.md` and `*.sh` |
| **DATA-02** | High (QG) | `configs/data/qg.yaml` | `val_file == test_file == qg_test.nc`, so early stopping and checkpoint selection used the test set | **Fixed** `63fbfb5`: now `qg_val.nc`, which I verified is independent of train |

**Confirmed, not fixed** (they need a decision from you, a data rebuild or a retrain):

| ID | Sev. | Evidence | Why it matters / action |
|---|---|---|---|
| **DATA-01** | Critical (QG) | Each `qg_test.nc` sim i (i = 0–9), at t < 500, equals `0.9948·qg_train` sim i + 6·10⁻⁴, with residual 6·10⁻⁸ [measured]. Each file is z-scored with its own statistics (≈ 0.5% scale mismatch) | Any QG forecast starting before t = 500 is evaluated on training data. Regenerate the test set with new seeds, or score only start times ≥ 500 plus a decorrelation gap. Store raw (unnormalised) fields |
| **EVAL-02** | Critical (claims) | Training climatology 15.83 and POD-DMD2 13.41 beat KAE 16.91 and ACDM 22.51 at 240 steps [measured, §8] | The 240-step MSE ranking measures mean regression, not skill. Report persistence, climatology, ACC, energy ratio and spectra alongside |
| **STAB-01** | High | QG generator: 63/128 eigenvalues with Re λ > 0, abscissa +0.40, so ‖exp(k·dt·A)‖ ≈ 7·10¹⁷ at k = 1000. KS (the DA model): 7 unstable eigenvalues and 80× transient growth. TRA: stable, but 16× transient growth and κ(V) = 533 [measured, `results/spectrum_*.json`] | There is no structural stability guarantee. Long QG rollouts cannot be bounded. Non-normal growth amplifies encoder error and conditions the DA adjoint |
| **DYN-01** | High | On TRA, the modes at the shedding frequency (ω ≈ 1.16–1.31) have Re λ ≈ −0.24 to −0.28 (e-fold 36–41 steps), and the slowest modes are real-valued. Fluctuation-energy ratio: 0.93 at lead 10, 0.60 at 100, 0.25 at 240 [measured] | Long forecasts converge to a near-steady field, whereas the truth is a persistent limit cycle (period ≈ 25 frames, constant amplitude over 1000 frames). `forward_stabilized` (γ = 0.017 energy injection) is a symptom of this, not a fix |
| **EVAL-03** | Medium | TRA `val.nc` (Ma 0.66–0.68) is the `interp` evaluation case. DA loads `best_model.pth`, selected on it, and tunes on `gt_extrap` [subagent; Ma values verified] | Selection bias on a reported test case. Hold out separate Mach values (e.g. 0.60 and 0.75) for validation |
| **REPRO-02** | High | The paper's KAE numbers (14.9, 5.2) match no checkpoint on disk. Run dirs named in `evaluate.py:99-101` no longer exist. The evaluated run saved only its model YAML, not its loss/data/training config. There is no code for the solver-comparison experiments [subagent] | Re-derive every reported number from a recorded config and commit hash (see `experiments.yaml` registry fields) |
| **CLAIM-01** | High | Frame dropping is hard-wired off (`trainer.py:290`: `0.0 if ... else 0.0`), `base_dt = 0.1` is hard-coded (`autoencoder.py:404`), and the QG data are 0.05 apart but trained with dt = 0.1 | The "irregular sampling / arbitrary τ" claims are untested for the TRA model. For QG, the eigenvalue time units are off by 2× |
| **REPRO-03** | Medium | `KoopmanAutoencoder.forward` always creates CUDA events and calls `torch.cuda.synchronize()` 4× per call (`autoencoder.py:387-514`) | The model cannot run on CPU (`RuntimeError: No CUDA GPUs`) [measured], and every forward blocks the host. Gate the timers behind a flag |
| **DATA-03** | Medium | `gt_longer_1000.nc` and `gt_*_subsample_1.nc` label frames 0.1 apart but are 0.05 apart (shedding period ≈ 50 frames vs ≈ 25) [measured] | A model applied to them at dt = 0.1 runs at half speed. My first diagnostic made exactly this mistake (§12). Fix the `t` coordinate |
| **ENC-01** | Medium | Rollout start z_0 = (E_hist + E_present)/2, but the latent targets are E_present(x_{t+k}) (`trainer.py:315-322` vs `autoencoder.py:221-228`) | The latent loss ties together two different charts. Encode the targets with the same map, or drop the averaging (`no_history` ablation exists) |
| **LOSS-01** | Medium | `PredictionLoss` is always squared error (`loss.py:290`) while `loss_type: l1` applies to reconstruction only. The gradient weighting `1 + 2 tanh` is capped at 3, while the comment says 10 (`loss.py:231`) | The config does not mean what it says. Make it explicit before ablating L1 vs L2 |
| **LOSS-02** | Medium | Smoothness penalty ‖Δ²_t E(x)‖² on non-detached encoder outputs (`loss.py:410-416`). Latent energy term weighted 0.0 (`:406`). The latent FFT "phase" term duplicates the time-domain MSE (Parseval) | The penalty biases the encoder toward slow, low-curvature latents, i.e. against oscillatory Koopman modes, consistent with DYN-01 [hypothesis, E4]. There was no collapse: ‖z‖ ≈ 10 throughout rollouts [measured] |
| **DEAD-01** | Low | Continuous `eigen` mode raises AssertionError (`_forward_eigen` → `_get_effective_linear_map` asserts LINEAR, `modules.py:325`) [measured]. `qg_eval.py` raises TypeError, and `build_kae` passes `is_continuous=cfg.model` [subagent] | Remove or fix |
| **EVAL-04** | Medium | `evaluate.py:420-426` swallows exceptions, and `load_checkpoint` returns an untrained model with only a warning when the file is missing (`utils.py:258-260`) [subagent; loader verified] | Stale or untrained results can enter tables silently. Make both fatal |
| **MISC** | Low | Discrete init `D = −0.01·ones` is a rank-one −1.28 eigenvalue, not "slight contraction" (`modules.py:478`). The backward init pinv(Kᵀ) equals K for orthogonal K (`:491`). The obstacle (1.06% of pixels) is included in metric means. `TRA_STD` v_y copies v_x. QG/KS use bf16 autocast through the latent rollout | Minor. Listed in `issues.json` |

---

## 5. Data and evaluation audit

**Splits.**
- *TRA:* split by simulation and Mach number. Train Ma = 0.53–0.63 ∪ 0.69–0.89, val 0.66–0.68,
  test 0.50–0.52 and 0.64–0.65. There is no temporal leakage, but val = `interp` (EVAL-03).
- *KS:* 80/10/10 disjoint simulations, no overlap.
- *QG:* train 900 sims × 500 steps, val 100 independent sims, but the test sims overlap train
  (DATA-01).

**Normalisation.** The constants are hard-coded and identical at train and eval time, and inverse
normalisation is correct [subagent + code]. The stored TRA stds differ from the training data:
v_x 0.217 vs 0.251, v_y 0.217 vs 0.169. This is harmless but undocumented.

**Masking.** Obstacle cells are zeroed in inputs, targets and decoder output, so their loss is
exactly zero. They still count in the denominator of means (≈ 1% dilution, uniform across models).

**Time step.** TRA (0.05 × 2) and KS (0.1) match the model's dt = 0.1. QG does not (0.05 × 1).

**Corrected evaluation protocol** (implemented in `scripts/tra_rollout_diagnostics.py` and
`scripts/benchmark_table.py`):
1. Align by ground-truth index (BUG-01). Score predicted frames only.
2. Report curves against lead time, not only rollout means: RMSE (fluid cells), ACC against the
   *training* climatology, and the fluctuation-energy ratio.
3. Always include persistence, training climatology and POD-DMD rows.
4. Use multiple start times per trajectory (10 cases for `longer_1000`).
5. Select checkpoints on a validation split that is not a reported test case, using a rollout
   metric at the deployment horizon.
6. Use ≥ 3 seeds and paired comparisons on identical cases.
7. For QG/KS, report statistical metrics (spectra, autocorrelation, distributional distances)
   and valid-prediction time in Lyapunov or decorrelation times, not RMSE at 10⁴ steps
   ([LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) §4d).

---

## 6. Mathematical audit

**Formulation.** The model is a continuous-time linear generator with a fixed-step RK4
discretisation. In effect it learns the discrete map K = P4(0.1·A), and A is identifiable only
through K. Because ‖dt·A‖ is small (max |λ|·dt ≤ 0.36 TRA, 0.09 KS, 0.12 QG), P4 and exp agree to
10⁻⁴ to 10⁻⁷ [measured]. RK4's stability boundary (|hω| < 2.83) and its damping of oscillatory
modes (|P4(iωh)| ≈ 0.994 at hω = 1) are therefore irrelevant at the trained step. They would
matter for the paper's arbitrary-τ queries only if they were evaluated with the same RK4 at large
τ, which they are not. **Recommendation:** train with `torch.linalg.matrix_exp(dt·A)` anyway. It
is cheap at 128×128, exact, and the only principled way to support variable dt.

**Stability.** Skew + symmetric gives d/dt‖z‖² = 2zᵀ sym(A) z. Contractivity needs sym(A) ⪯ 0,
which is not imposed:
- TRA: max eig(sym A) = +1.72, so the norm can grow transiently (16× at 50 steps) even though
  every eigenvalue is stable.
- QG: unstable outright.

The fix is structural: D = −(LLᵀ + εI) gives ‖exp(tA)‖ ≤ 1 with no transient growth. A diagonal
complex spectrum, λ_k = −softplus(a_k) + iω_k (modal basis absorbed into encoder/decoder), gives
exact exponentials in O(d) and lets modes approach the imaginary axis. This is what a limit cycle
needs.

**Do not over-correct to strict dissipativity.** The TRA failure (DYN-01) is *too much* damping at
the shedding frequency. The parameterisation must permit Re λ → 0⁻ for a few oscillatory modes,
and training must reward it. That requires long-horizon losses (§11) and a loss without the
smoothness penalty.

**Koopman consistency** [measured, `results/tra_rollout_*.json`]:
- One-step relative latent error ‖K z_0 − E(x_1)‖/‖E(x_1)‖ is 1.8–3%.
- It rises to 13–28% at 10 steps, and 33–38% plateau from 50 steps.
- The latent norm stays ≈ 10, so there is no collapse.
- Linearity therefore holds over roughly 10 steps, which matches the 8-step training horizon and
  nothing longer.

**Representation.**
- The reconstruction floor (RMSE 0.13–0.17 normalised) is above a 128-mode linear POD's
  projection error (0.12) on `interp`/`longer`, and similar on `extrap` (0.075–0.097 vs 0.088)
  [measured].
- A 14 M-parameter nonlinear autoencoder that reconstructs no better than linear POD is a red
  flag. Plausible causes [hypothesis]: single-query attention pooling, which forces z to be a
  convex mixture of local features; spectral norm on every res block, which limits Lipschitz
  constant and sharpness; the 3×1 first conv; decoder dropout; and the noise injected into z_0
  before decoding.
- Test with E6.

**Training objective.**
- The prediction horizon is 8 steps. The forecast deployment horizons are 60 and 240 steps,
  roughly 2–10 shedding periods.
- Nothing in the loss or the checkpoint criterion sees the long-horizon behaviour (energy decay,
  phase drift), so the train/test mismatch is structural.
- Multi-step latent unrolls are almost free. The latent rollout is 3 ms of a 72 ms forward
  [measured], so latent-only losses can extend to 60+ steps with decoded supervision on a random
  subset of steps.

---

## 7. State-of-the-art comparison (summary; full entries in `LITERATURE_REVIEW.md`)

Published numbers from different datasets are **not comparable**. Only the ACDM protocol
(Tra/Inc) and the Thermalizer protocol (QG) give like-for-like comparisons.

| Family | Formulation | Long-horizon behaviour | Cost | Fit to these data | Recommended experiment |
|---|---|---|---|---|---|
| POD + linear map (DMD, BOP-DMD) | a_{t+1} = K a_t on POD coefficients | Exact; decays or persists per Re(ω) | Seconds | TRA: strong (§8). QG/KS: poor | **Mandatory baseline** (done here for TRA) |
| Periodic Koopman (Lange et al. 2021, JMLR) | z = cos/sin(ωt) + nonlinear decoder | No decay by construction | Negligible | TRA long horizon | Baseline for `longer` |
| Discrete or continuous KAE (Lusch 2018, Azencot 2020, this repo) | dz/dt = Az or z_{t+1} = Kz | Stable only if constrained; non-normal transients | Lowest | TRA short horizon | Fix + E1–E4 |
| Diagonal complex / SSM-style linear latent (LRU, S5) | λ = −e^ν + iθ, exact exp | Guaranteed \|λ\| ≤ 1; ring init near the axis | O(d) | Drop-in for A | **E1** |
| AR U-Net / FNO with unrolling or noise (ACDM benchmark) | x_{t+1} = f(x_t) | Needs unrolling; can blow up | Low–moderate | TRA SOTA at short T (U-Net_tn beats ACDM on Tra_ext/int) | **Mandatory baseline** (public code and weights) |
| Diffusion / refinement (ACDM, PDE-Refiner, Thermalizer) | Conditional denoising per step | Best spectra; Thermalizer stable to 10⁵ QG steps | 20–100× U-Net | TRA (ACDM), QG (Thermalizer) | Compare on statistics, not long-T MSE |
| Measure-aware linear latent (PFNN, ICLR 2025) | Linear latent + invariant-measure constraints | Designed for chaotic statistics | Low | QG/KS | Use its metrics |
| Ocean emulators (OceanNet, Samudra, NeuralOM) | Global AR emulators | Operational | High | Not this scale | Context only |

---

## 8. Baseline benchmark (TRA, measured)

**MSE ×10⁻³**, single convention: physical units, obstacle zeroed, predicted frames only
(`results/benchmark_table_tra.json`). This is the *same* metric for every row. One checkpoint per
model and no seeds, so differences under about 10% should not be read as rankings.

| Model | interp (T=60) | extrap (T=60) | longer (T=240) |
|---|---|---|---|
| Persistence (last input frame) | 23.44 | 17.64 | 28.78 |
| Training climatology (constant field) | 16.76 | 17.39 | **15.83** |
| **POD(128) + linear one-step map** (34 s CPU fit) | 8.60 | 6.04 | 15.16 |
| **POD(128) + linear 2-frame map** | 6.06 | 3.86 | **13.41** |
| KAE continuous, `run-20260821` ep199 (realigned) | 6.37 | **1.59** | 16.91 |
| (same, as scored by the current pipeline) | 7.62 | 1.82 | 16.97 |
| KAE continuous, January files (rk4 = exp) | 6.71 | 2.60 | 23.01 |
| KAE Azencot (realigned) | 8.09 | 5.98 | 24.37 |
| ACDM (r20) | **2.81** | 1.99 | 22.51 |
| ACDM-ncn | 2.82 | 4.19 | 22.35 |
| *Paper (KAE)* | *5.2* | — | *14.9* |

Horizon-resolved skill: anomaly correlation (ACC) ≥ 0.6 up to the given lead, with RMSE in
normalised units (`results/tra_rollout_*.json`, `results/linear_baselines_r128.json`):

| Case | KAE: first lead with ACC < 0.6 | POD-DMD2: first lead with ACC < 0.6 | Persistence: same | KAE RMSE at L1 / L10 / L30 | POD-DMD2 at L1 / L10 / L30 |
|---|---|---|---|---|---|
| interp | 27 | 53 | 5 | 0.140 / 0.193 / 0.618 | 0.118 / 0.277 / 0.449 |
| extrap | > 58 | > 58 | 6 | 0.087 / 0.112 / 0.250 | 0.093 / 0.204 / 0.352 |
| longer | 28 | 27 | 5 | 0.141 / 0.175 / 0.577 | 0.128 / 0.269 / 0.555 |
| longer_1000 (10 cases, stride 2) | 29 | n/a | 5 | 0.146 / 0.172 / 0.573 | n/a |

**Interpretation.**
- The KAE is best-in-class on `extrap` and strong over leads 2–15 everywhere.
- On `interp` beyond about 15 steps, and on `longer`, a linear POD model is as good or better.
- At 240 steps no model beats climatology by much, and ACDM is the only model that keeps
  realistic fluctuation energy, which MSE penalises.
- The KAE's long-horizon "win" over ACDM is a mean-regression effect.

**Not run** (see `experiments.yaml`): ACDM-benchmark U-Net_tn/U-Net_ut/FNO, Lange periodic Koopman,
BOP-DMD, a corrected KAE retrain, multi-seed runs, any QG/KS forecast benchmark (the QG test split
is invalid; KS was out of scope given compute), LSiM and spectral metrics.

---

## 9. Performance profile (measured; GPU shared with another job)

`results/smoke_profile_baseline.json` and `results/smoke_profile_after_fixes.json`. TRA model,
B = 16, fp32, RTX 4090, interleaved A/B ×3.

| Configuration | Median s/step |
|---|---|
| As the trainer runs it (anomaly mode + 165 NaN hooks) | **4.62** |
| Anomaly mode off | 1.04 |
| Both off (audit-branch default) | **0.28** |

**Forward breakdown** (model timers, 72 ms total): decode 48.6 ms (8 frames × B), encode
10.7 ms, latent rollout 3.1 ms. Forward + loss 137 ms, backward 145 ms. Peak memory 6.7 GB.
**Inference:** a 600-step rollout and decode for 2 trajectories took 2.3 s.

**Bottlenecks after PERF-01, in order:**
1. The decoder: every predicted frame is decoded at full resolution.
2. Per-forward `torch.cuda.synchronize()` ×4 plus CUDA-event timing (REPRO-03).
3. The second (disturbed) latent rollout, which is computed even when `stability_weight` is 0.
4. 4 dataloader workers on 32 cores.

**Prioritised efficiency plan** (benefits estimated unless marked):

| Change | Benefit | Risk | Effort |
|---|---|---|---|
| PERF-01 (done) | **16.5× [measured]** | None (debug-only) | Done |
| Gate the timing syncs behind a flag | Small–moderate | None | Trivial |
| Skip the disturbed rollout when `stability_weight` = 0 | Small | None | Trivial |
| bf16 autocast for encoder/decoder only, latent rollout in fp32 | ~1.5–2× [estimate] | Validate rollout drift | Low |
| Decode a random subset of rollout steps when unrolling long horizons | Makes 60-step training affordable | Changes the objective (E2) | Low |
| `torch.compile` of encoder/decoder | 1.2–1.5× [estimate] | Compile time; dynamic shapes | Low |

---

## 10. Recommended architecture

There is not yet enough evidence to name a single winner. The shortlist below, and the
experiment that decides between its entries, are specified in `experiments.yaml`.

**Candidate A: Stable modal KAE (primary).**
- Encoder: same backbone, but replace single-query pooling with flatten + linear, or multi-query
  pooling; E6 decides.
- Latent: d = 128, organised as d/2 complex modes.
- Dynamics: λ_k = −softplus(a_k) + iω_k, propagated exactly as z(t+τ) = exp(λτ) ⊙ z(t) for any τ.
  ω is initialised to cover the shedding frequency and its harmonics (2π/2.5 per time unit and
  multiples); a_k is initialised small (ring init).
- Conditioning: λ(c) = λ_0 + Δλ(c) via a small MLP on the Mach embedding. This is
  well-conditioned and needs no LoRA.
- Loss:
  - reconstruction;
  - latent rollout loss against **stop-gradient** encoder targets over 60 steps;
  - decoded loss on a random subset of ~8 of those steps;
  - no smoothness penalty;
  - optional energy-ratio penalty on decoded fluctuations.
- Selection: validation ACC/RMSE at leads 60 and 240 on held-out Mach values.

Why: it removes STAB-01 by construction and makes τ continuous. It also makes the generator, and
therefore the DA adjoint, cheap and well-conditioned. It attacks DYN-01 directly by letting
shedding modes sit near the imaginary axis while training sees one or more periods.

**Candidate B: Hurwitz-dense KAE (fallback if modal is too restrictive).**
- A = S − (LLᵀ + εI), with S skew, propagated by `matrix_exp`. Otherwise as A.
- It keeps cross-mode coupling, at the cost of a 128³ exponential per batch, which is negligible.

**Candidate C: Linear latent + nonlinear residual.**
- z_{k+1} = exp(dt·Λ) z_k + g_θ(z_k), with g small and penalised.
- Pursue only if A and B both plateau above U-Net_tn at T = 60. Then compare against the ACDM
  U-Net baselines directly, since a KAE that needs a nonlinear residual loses its main advantage
  (linear, cheap, analysable dynamics).

For **QG/KS**, a linear latent model cannot reproduce chaotic pointwise trajectories beyond a few
Lyapunov times. Claim and evaluate statistical fidelity (spectra, invariant-measure distances) and
DA performance, not long-horizon RMSE.

---

## 11. Prioritised action plan

**Immediate fixes: the current implementation or protocol is wrong (§16 A).**

| P | Issue | Accuracy | Stability | Speed | Effort | Validation cost | Risk | Evidence | Next action |
|---|---|---|---|---|---|---|---|---|---|
| 1 | BUG-01 eval alignment | Correct numbers | — | — | Done | Re-run `evaluate.py` + tables (≈ 1 h) | None | Measured | Re-score all KAE npz files; cherry-pick `0bfe883` |
| 2 | DATA-01/02 QG splits | Valid QG results | — | — | Data regen | Hours | None | Measured | Regenerate `qg_test.nc` from new seeds; cherry-pick `63fbfb5` |
| 3 | EVAL-02 baselines in tables | Honest claims | — | — | Low | Minutes | None | Measured | Add persistence, climatology and POD-DMD rows (scripts exist) |
| 4 | PERF-01 | — | — | **16.5×** | Done | — | None | Measured | Cherry-pick `b9a66eb` |
| 5 | BUG-02 + `adaln` gap | Conditioning works | — | — | Done/Low | 1 retrain | Low | Measured | Retrain TRA; check Mach sensitivity > 0 and E5 |
| 6 | REPRO-01/02 | Reproducibility | — | — | Low | — | None | Code | Track configs; save the full resolved config + git hash per run |
| 7 | EVAL-03 TRA validation split | Unbiased selection | — | — | Low | Retrain | None | Measured | Hold out Ma 0.60/0.75 from training as validation |
| 8 | BUG-03/05/06, REPRO-03, EVAL-04 | Configs mean what they say | — | Small | Done/Low | Tests | Low | Tests | Cherry-pick; make missing checkpoints fatal |

**Experiments: strongly motivated (§16 B) or plausible (§16 C).** See §12.

**Major redesign (§16 D).** Candidate C, or switch to an AR U-Net/diffusion family, only if the
E1–E3 gates fail.

---

## 12. Experiment roadmap (summary of `experiments.yaml`)

Stage 0 (baseline, ~1 GPU-day with PERF-01):
- Retrain the current TRA KAE with the fixes and a new validation split, 3 seeds. This is the
  "corrected baseline".
- Record: per-lead RMSE/ACC, energy ratio, recon floor, spectrum, Mach sensitivity.

Stage 1: dynamics. Single changes against the corrected baseline, 3 seeds each.
- **E1:** stable modal generator with `matrix_exp`.
- **E2:** training horizon 8 → 30 → 60 latent steps, with decoded subset.
- **E3:** checkpoint selection on validation ACC at lead 60 vs `total_loss`.
- **Gate:** proceed with the best of E1×E2 if it raises the ACC ≥ 0.6 horizon on `interp` above
  POD-DMD2's 53 steps *and* the energy ratio at 240 above 0.6. Otherwise stop pursuing long-horizon
  claims and position the model for short-horizon / DA use.

Stage 2: representation and loss.
- **E4:** remove the smoothness penalty; stop-grad latent targets.
- **E5:** conditioning in encoder and operator vs none.
- **E6:** pooling → flatten, spectral norm off (`honor_conv_kwargs`), decoder dropout off.
  Gate on reconstruction floor < POD projection floor.
- **E7:** history encoder vs single frame vs delay embedding (ENC-01).

Stage 3: external baselines on identical splits.
- ACDM U-Net_tn/U-Net_ut, Lange periodic Koopman, BOP-DMD.
- For QG: Thermalizer UNet ± thermalizer on statistical metrics.

Stage 4: efficiency.
- E8: bf16 split precision.
- E9: `torch.compile`.
- Each must not change the per-lead metrics by more than seed noise.

---

## 13. Implementation changelog (branch `audit/kae-research-audit`)

| Commit | Files | Change | Tests run |
|---|---|---|---|
| `77ef4c7` | 9 configs + `eval.sh` | Snapshot of your uncommitted config edits (baseline). The pre-commit hook added an EOF newline to `tra.yaml` | — |
| `fe2b53e` | `audit_report/**` | Diagnostics, results and 10 regression tests | 10/10 **fail** on original code |
| `9897d96` | 35 configs | Track configs hidden by `*.yaml` (whitespace normalised by hooks) | — |
| `0bfe883` | `evaluate.py` | BUG-01 | test_bug01 |
| `55f7512` | `models/modules.py` | BUG-02 (`_build_lora_hypnet`) | test_bug02 ×3 operator types |
| `8e9645c` | `models/loss.py` | BUG-03 | test_bug03 |
| `b9a66eb` | `trainer.py`, `config_classes.py`, `train.py` | BUG-04, PERF-01 (`debug_nan_checks`). Also carries the `honor_conv_kwargs` config/`train.py` wiring, because those files were edited together | test_bug04, test_perf01 |
| `63fbfb5` | `dataloader.py`, `utils.py`, `configs/data/qg.yaml` | BUG-05, DATA-02 | test_bug05 |
| `175c7a4` | `networks.py`, `autoencoder.py` | BUG-06 (opt-in) | test_bug06 ×2 |
| `b66f5be`, later | `audit_report/**` | Post-fix results, compatibility check, report | 10/10 **pass**; `test_discrete_loss.py` passes |

Post-fix checks:
- **Checkpoint compatibility:** `run-20260821_031121/epoch_199` loads strictly in both trees, with
  identical output checksums [measured].
- **Gradient flow after the fix:** parameters without gradient at step 1 fall from 152 k to 17 k.
  The remainder are the unused decoder `expansion_map`, `re_predictor` (`re_weight` is null) and
  the earlier hypernetwork layers. The latter, and the Mach embedding, get gradient from step 2,
  as expected for LoRA [measured].

**Failures encountered during the audit**, recorded so nobody repeats them:

1. My first long-rollout diagnostic used `gt_longer_1000.nc` at stride 1. Its frames are really
   0.05 apart (DATA-03). That run suggested the KAE was worse than persistence at lead 1; **this
   was an artefact**. On correctly strided data the KAE beats persistence at lead 1 (0.146 vs
   0.170). The invalid result file was deleted.
2. The model cannot run on CPU (REPRO-03).
3. A light profiling run was used because the GPU was shared.

---

## 14. Reproducibility instructions

All commands run from the audit worktree. Python is `/home/rg625/anaconda3/envs/samudra/bin/python`.

```bash
cd /home/rg625/mnt/ocean_forecasting_audit
PY=/home/rg625/anaconda3/envs/samudra/bin/python

# Tests (CPU, about 3 s)
cd koopman_autoencoder && $PY -m pytest ../audit_report/tests -q -p no:cacheprovider && cd ..

# Operator spectrum of any continuous checkpoint (CPU)
$PY audit_report/scripts/operator_spectrum.py <ckpt.pth> --json out.json

# Long-horizon diagnostics (GPU; stride 2 for the *_1000 / *_subsample_1 files)
RUN=/home/rg625/mnt/ocean_forecasting/koopman_autoencoder/model_outputs_tra/continous_linear_128/run-20260821_031121
D=/home/rg625/mnt/ocean_forecasting/koopman_autoencoder/data/acdm/128_tra
$PY audit_report/scripts/tra_rollout_diagnostics.py --run $RUN --data $D/gt_interp.nc --steps 58 --t0 1
$PY audit_report/scripts/tra_rollout_diagnostics.py --run $RUN --data $D/gt_longer_1000.nc --stride 2 --steps 450 --t0 10 50 90 130 170

# Linear POD baselines (CPU, about 1 min) and the unified benchmark table
CUDA_VISIBLE_DEVICES="" $PY audit_report/scripts/linear_baselines.py --rank 128 --json audit_report/results/linear_baselines_r128.json
$PY audit_report/scripts/benchmark_table.py --pod audit_report/results/linear_baselines_r128.json

# Smoke test, gradient flow and profile (GPU)
$PY audit_report/scripts/smoke_gradflow_profile.py --repo $PWD/koopman_autoencoder --steps 12 --batch 16

# Checkpoint compatibility between two code trees
$PY audit_report/scripts/compat_check.py /home/rg625/mnt/ocean_forecasting/koopman_autoencoder
$PY audit_report/scripts/compat_check.py $PWD/koopman_autoencoder

# Baseline training (as configured; now about 16x faster per step)
cd koopman_autoencoder && $PY train.py --config-name experiment/tra/continous_linear_128
#   debug hooks back on:      ... training.debug_nan_checks=true
#   honour padding/spectral:  ... +model.honor_conv_kwargs=true   (new architecture; old checkpoints will not load)

# Re-score with the fixed evaluator (writes npz under autoreg_pde_diffusion/src/results/sampling)
bash training_scripts/eval.sh
```

Training and evaluation paths in the configs are absolute and point at the main checkout's data
directory. The worktree does not contain the data.

---

## 15. Limitations of this audit

- No model was retrained, so every proposed architectural or loss change is **unvalidated**.
- One checkpoint per model and no seeds. Differences under about 10% are not rankings.
- The GPU was shared throughout, so absolute timings are inflated. The ratios are robust
  (interleaved).
- I did not run the ACDM U-Net/FNO baselines, LSiM, spectra or any QG/KS forecast benchmark.
- DA experiments were reviewed only for dt and checkpoint use.
- Literature details marked "unverified" or "(memory)" in `LITERATURE_REVIEW.md` should be
  checked before citing. The ACDM Table 1 numbers there came from a garbled PDF extraction.

---

## 16. Final answer

> *If our objective is to maximise scientifically valid ocean-forecasting accuracy over the required prediction horizons while maintaining practical computational efficiency, what exactly should we change in this repository, in what order, and why?*

**A. Necessary, because the current implementation or protocol is wrong.**

1. **Fix the evaluator.** Cherry-pick `0bfe883` and re-score every KAE table. Every current KAE
   number is scored against the wrong frame.
2. **Fix the data splits.**
   - Regenerate QG test data from independent seeds, since the current test set is the training
     set for t < 500.
   - Validate QG on `qg_val.nc` (`63fbfb5`).
   - Validate TRA on held-out Mach values that are not the `interp` test case.
3. **Fix the conditioning.**
   - Cherry-pick `55f7512` and retrain, or stop claiming Mach/Re conditioning: the evaluated model
     ignores Mach exactly.
   - Implement or remove the encoder's `adaln` path.
4. **Remove the 16.5× training slowdown** (`b9a66eb`). This alone multiplies the experimental
   budget.
5. **Make runs reproducible.**
   - Track all configs (`9897d96`) and narrow the `.gitignore`.
   - Save the full resolved config and git hash in every run directory.
   - Make missing checkpoints and evaluation exceptions fatal.
6. **Make configs mean what they say:** `honor_conv_kwargs`, the detached stability term,
   L1 vs L2 in the prediction loss, and the QG dt (0.05, not 0.1).
7. **Rebuild the comparison tables.**
   - Always add persistence, climatology and POD-DMD rows.
   - Report per-lead RMSE/ACC and energy ratio.
   - Stop presenting 240-step MSE as forecast skill, since climatology and a linear POD model beat
     the KAE there.

**B. Strongly motivated by mathematics and the evidence collected here.**

1. Parameterise the generator so it is **stable by construction yet able to hold near-neutral
   oscillatory modes**, either diagonal complex (Candidate A) or Hurwitz-dense (Candidate B).
   Propagate with the exact exponential. This removes the QG instability and the KS/TRA transient
   growth, and makes arbitrary-τ and DA use principled.
2. **Train on the horizons you deploy.**
   - Latent unrolls of 30–60 steps (about 3 ms per 8 steps, so cheap).
   - Decoded supervision on a random subset of steps.
   - Checkpoint selection on validation ACC/RMSE at those horizons.
   - The current 8-step objective cannot see the damping that dominates long-horizon error.
3. **Stop-gradient latent targets, encoded with the same map that starts the rollout.** Drop the
   smoothness penalty on encoder outputs: it pushes against exactly the oscillatory modes the
   flow needs.

**C. Plausible, but they need controlled experiments** (`experiments.yaml` E4–E9):
- the encoder pooling and spectral-norm choices behind the reconstruction floor being worse than
  linear POD;
- history encoding vs delay embedding;
- latent dimension;
- L1 vs L2 prediction loss;
- noise injection;
- bf16 encoder/decoder;
- `torch.compile`.

Each must beat seed noise on per-lead metrics.

**D. Only if A–C do not reach the target.** If the stable, long-horizon-trained KAE still does not
beat POD-DMD2 beyond 30 steps on `interp`, and U-Net_tn at T = 60, then the linear-latent
hypothesis does not hold for this flow at useful horizons. Two options remain:
- Add a constrained nonlinear latent residual (Candidate C).
- Concede short-horizon accuracy to AR U-Nets/diffusion and position the KAE where its structure
  pays: cheap, differentiable, well-conditioned dynamics for data assimilation and for the
  statistics of QG. Evaluate the latter with the Thermalizer protocol and invariant-measure
  metrics.

**The path, in order:**
1. A1–A7 (days).
2. Corrected baseline with 3 seeds.
3. E1 × E2 × E3 (about 1 week of single-GPU time with the speed fix).
4. Gate against POD-DMD2 and U-Net_tn.
5. E4–E7.
6. External baselines.
7. Then, and only then, architectural expansion.

---

## Addendum (2026-10-19, branch `impl/ct-kae`): corrections and phase-2 fixes

**Correction: the KAE baseline numbers above are not reproducible (REPRO-04).**
- §1, §8 and §16 quote KAE MSE 6.37 / 1.59 / 16.91 (interp / extrap / longer). Those figures
  were computed from the saved prediction files written on 24 August.
- Re-running the same checkpoint (`run-20260821_031121/epoch_199`, identical to
  `final_model.pth`, same config) with the repository code gives **11.43 / 2.12 / 18.62**
  under the same scorer.
- The first predicted frame already differs from the saved file by 0.07, in fp32 and TF32
  alike. No other TRA checkpoint on disk reproduces the files.
- The model code was rewritten in `a5f4796` (24 September), after that evaluation.
- The audit's horizon diagnostics (ACC < 0.6 at lead 27–29, energy ratio 0.25 at 240) were
  computed with the current code, so they are consistent with the reproducible numbers. The
  MSE table was not.
- Conclusions that change:
  - The legacy KAE is **not** competitive on `extrap` (2.12 vs U-Net_tn 1.42 and ACDM
    ≈ 2.0–2.2).
  - Its `interp` gap to ACDM is about 4×, not about 2×.
- See `EXPERIMENT_RESULTS.md` §2.

**Also found in phase 2:**

| ID | Finding | Resolution |
|---|---|---|
| REPRO-05 | The repository `.gitignore` (`*.json`, `*.txt`, `*.sh`) had silently excluded `issues.json` and the raw results from the audit commits | Force-added in `b5e5876` |
| PROV-01 | ACDM provenance verified: official checkpoints; Tra_int matches the paper after the frame-convention conversion | The paper has no Tra_long numbers. U-Net_tn and U-Net_ut, not ACDM, are the strongest TRA baselines |

**Fixed in phase 2** (tests in `tests/test_phase1_fixes.py`, 9/9 fail on the previous code):

| Issue | Fix | Commit |
|---|---|---|
| REPRO-03 | CPU-safe timings | `0505f1f` |
| CLAIM-01 | Configurable `dt` and `drop_prob` | `0505f1f`, `7013450` |
| DEAD-01 | Fail-fast continuous eigen mode | `0505f1f` |
| LOSS-01 | Explicit `pred_loss_type` | `73a2d81` |
| REPRO-02 | Run metadata in checkpoints | `7013450` |
| EVAL-04 | Fatal missing checkpoints / evaluation errors | `7013450` |
| ENC-01 | Consistent latent targets (opt-in) | `7013450` |
| EVAL-03 | Parameter-separated TRA validation split | `34870f7` |

The evaluated legacy checkpoint still loads strictly with bit-identical outputs.
