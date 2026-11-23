# Literature review v2: ACDM verification and improvements for a conditional continuous-time KAE (2026-10-19)

Scope:
- Part 1 verifies the ACDM Tra benchmark from primary sources: arXiv HTML and the official code.
- Part 2 is a targeted review of ways to improve, not replace, the model z = E(x,c); dz/dt = A(c) z; x̂ = D(z,c) on Tra.
- Part 3 ranks the most promising ideas against three known failure modes:
  - shedding modes are damped, with e-folding ≈ 36–41 steps;
  - Mach is ignored;
  - Tra_int MSE is 6.37e-3, against 2.81e-3 for ACDM.

Conventions used in this file:
- **[verified]**: read in the primary source this session.
- **[code]**: read in the official GitHub code this session.
- **[derived]**: my own arithmetic from verified quantities.
- **UNVERIFIED**: could not be checked; the reason is given each time.

This file supersedes §D5 of `ocean_forecasting_audit/audit_report/LITERATURE_REVIEW.md`. That section's Table 1 values came from a garbled extraction. They turn out to be correct for the rows it listed, but it omitted rows and conventions.

---

## 1. ACDM verification

### 1.1 Provenance

**Provenance.** The values were parsed programmatically from the `<table>` elements of `https://arxiv.org/html/2309.01745v3` (v3, 11 Dec 2024), and cross-checked against `…v2` (29 Jan 2024) and `…v1` (4 Sep 2023).
- The v2 and v3 tables are **numerically identical**. Only the captions differ: "TABLE I" vs "Table 1".
- The evaluation conventions were read from the official code at `github.com/tum-pbs/autoreg-pde-diffusion`, branch `main`, last push 2024-12-13:
  - `src/plot_loss_all.py`
  - `src/sample_models_tra.py`
  - `src/training_tra.py`
  - `src/training_diffusion_tra.py`
  - `src/turbpred/model.py`
  - `src/turbpred/data_transformations.py`
  - `src/plot_color_and_name_mapping.py`

**Bibliographic status.**
- Kohl, Chen, Thuerey, "Benchmarking Autoregressive Conditional Diffusion Models for Turbulent Flow Simulation", arXiv:2309.01745 (v1 title: "Turbulent Flow Simulation using Autoregressive Conditional Diffusion Models").
- **Journal version verified via Crossref**: *Neural Networks* 199 (2026) 108641, DOI 10.1016/j.neunet.2026.108641, issue date July 2026, same three authors.
- The arXiv page carries no journal-ref.
- An ICML 2024 workshop poster also exists (icml.cc/virtual/2024/36841).
- **UNVERIFIED:** whether the journal tables are identical to arXiv v3. The Elsevier full text is paywalled and was not opened.

### 1.2 Verified Tra results (Table 1 of v2/v3)

Units: MSE ×10⁻³ and LSiM ×10⁻¹. The ± value is the standard deviation over (3 training seeds) × (5 samples for probabilistic models) × (6 test sequences) of the per-trajectory MSE (see §1.3).

| Paper name | Code ID (npz / checkpoint) | Tra_ext MSE | Tra_ext LSiM | Tra_int MSE | Tra_int LSiM |
|---|---|---|---|---|---|
| ResNet | `resnet-m2` | 2.3 ± 0.9 | 1.4 ± 0.2 | 1.8 ± 1.0 | 1.0 ± 0.3 |
| ResNet_dil. | `dil-resnet-m2` | 1.7 ± 1.0 | 1.2 ± 0.3 | 1.7 ± 1.4 | 1.0 ± 0.5 |
| FNO16 | `fno-16modes-m2` | 4.8 ± 1.2 | 3.4 ± 1.1 | 5.5 ± 2.6 | 2.6 ± 1.1 |
| FNO32 | `fno-32modes-m2` | 4.9 ± 1.9 | 3.6 ± 0.9 | 6.8 ± 3.4 | 3.1 ± 1.1 |
| TF_MGN | `tf-mgn` | 3.9 ± 1.0 | 1.8 ± 0.3 | 6.3 ± 4.4 | 2.2 ± 0.7 |
| TF_Enc | `tf-enc` | 3.3 ± 1.2 | 1.8 ± 0.3 | 6.2 ± 4.2 | 2.2 ± 0.7 |
| TF_VAE | `tf-vae` | 4.1 ± 0.9 | 2.4 ± 0.2 | 7.2 ± 3.0 | 2.7 ± 0.6 |
| U-Net | `unet-m2` | 3.1 ± 2.1 | 3.9 ± 2.8 | 2.3 ± 2.0 | 3.3 ± 2.8 |
| U-Net_ut | `unet-m8` | **1.6 ± 0.7** | **1.1 ± 0.2** | **1.5 ± 1.5** | **1.0 ± 0.5** |
| U-Net_tn | `unet-m2-noise0.01` | **1.4 ± 0.8** | **1.1 ± 0.3** | **1.8 ± 1.1** | **1.0 ± 0.4** |
| Refiner | `refiner-r4_std0.000001` (see note) | 5.4 ± 2.1 | 2.3 ± 0.5 | 7.1 ± 2.1 | 3.0 ± 1.7 |
| ACDM_ncn | `acdm-r20_ncn` | 4.1 ± 1.9 | 1.9 ± 0.6 | 2.8 ± 1.3 | 1.7 ± 0.4 |
| ACDM | `acdm-r20` | **2.3 ± 1.4** | **1.3 ± 0.3** | **2.7 ± 2.1** | **1.3 ± 0.6** |

**Tra_long has no MSE/LSiM numbers anywhere in the paper.**
- Tra_long (Ma 0.64–0.65, T=240) is used only for qualitative and statistical figures:
  - posterior samples (Figs. 5–6);
  - temporal and spatial spectra (Fig. 7);
  - the rate-of-change stability metric ‖(sᵗ − sᵗ⁻¹)/Δt‖₁ (Fig. 10, and ablation Figs. 20, 22, 23, 25, 27, 28).
- The paper's narrative on Tra_long:
  - ResNet_dil, FNO16 and U-Net "diverge … and mostly settle into a stable but wrong state of a mean flow prediction without vortices";
  - TF_Enc "slightly undershoots";
  - U-Net_tn shows "early deterioration signs" towards the end;
  - ACDM and U-Net_ut remain stable.
- Every Tra_long MSE quoted in the repo owner's preprints or in `FINAL_AUDIT.md` (e.g. ACDM 22.51e-3) is a **re-evaluation, not a paper number**.

**Other verified ablation rows that are useful baselines** (same units, Tra_ext / Tra_int MSE):
- Table 5, training rollout m:
  - U-Net m=4: 1.6 ± 1.0 / **1.1 ± 1.0**. This is the best Tra_int MSE in the paper.
  - m=8: 1.6 / 1.5. This is identical to U-Net_ut, which confirms U-Net_ut = m8.
  - m=16: 2.2 / 2.4.
- Table 6, training noise n:
  - n=1e-2: 1.4 / 1.8, which equals U-Net_tn.
  - n=1e-1: 1.8 / 2.2.
  - n=1: 4.0 / 11.4.
- Table 4, ACDM diffusion steps R:
  - R=20: 2.3 / 2.7, which equals Table 1.
  - R=50: 2.3 / 2.4.
  - R=10: 3.8 / 6.2.

**Internal inconsistency (Refiner).** The Table 1 Refiner row (5.4 / 2.3 / 7.1 / 3.0) does **not** match any row of Table 9 (Refiner R∈{2,4,8} × σ∈{1e-3…1e-7}). The code's choice for Tra, `refiner-r4_std0.000001`, is listed in Table 9 as 3.5 ± 1.1 / 2.1 ± 0.5 / 8.8 ± 0.9 / 4.3 ± 2.1. UNVERIFIED which configuration produced the Table 1 row. Do not quote Refiner as a tight baseline.

**Version differences.** v1 (Sep 2023) Table 1 contained only TF_MGN, TF_Enc, TF_VAE, U-Net and ACDM, with slightly different values:

| Model | v1 Tra_ext MSE | v1 Tra_int MSE |
|---|---|---|
| ACDM | 2.5 ± 1.5 | 2.7 ± 2.5 |
| TF_MGN | 3.5 | 6.1 |
| TF_VAE | 4.0 | 7.2 |

In v1 Table 2, ACDM appears as 2.2 ± 1.3 / 2.7 ± 2.5. Always cite v2/v3, which are identical.

**Previous review cross-check.** The old §D5 values for U-Net_tn, U-Net_ut, ResNet_dil, ACDM, FNO16, Refiner and TF_Enc all match v3, so the garbling did not corrupt those rows. That table omitted ResNet, FNO32, TF_MGN, TF_VAE, U-Net and ACDM_ncn, and it did not state that Tra_long has no numbers.

### 1.3 Exact MSE convention (code-verified)

Source: `src/plot_loss_all.py`, `src/sample_models_tra.py`, `turbpred/model.py` [code].

1. **Units: physical (dataset) units, not normalised.**
   - Ground truth is saved with `testSet.transform = lambda x: x` (raw).
   - Predictions are de-normalised before saving: `predSamples = predSamples*normStd + normMean`.
   - Tra statistics (`normalizeMode="machMixed"`, `data_transformations.py`):

     | Field | mean | std |
     |---|---|---|
     | v_x | 0.560642 | 0.216987 |
     | v_y | −0.000129 | 0.216987 |
     | ρ | 0.903352 | 0.145391 |
     | p | 0.637941 | 0.119944 |
     | Ma | 0.70 | 0.118322 |

   - v_y uses v_x's std; this is the "normalised by magnitude" rule.
   - [derived] A normalised-unit MSE would be larger by 1/std² ≈ 21.2 (v_x, v_y), 47.3 (ρ) and 69.5 (p).
2. **Obstacle.** If `obsMask` is present (it is for Tra), both GT and prediction are multiplied by the binary mask. Obstacle pixels contribute zero error but **stay in the denominator** of the spatial mean.
3. **Fields.** `getLossRelevantFields("interp"/"extrap"/"longer") = (0,4)`, i.e. channels v_x, v_y, ρ, p. **The Mach channel (index 4) is excluded.**
4. **Reduction.** `mse = (gt − pred)²` has shape [seeds=3, evals=5, sequences, T, 4, 128, 64]. `mseScalar = mean over (T, fields, H, W)` gives one scalar per (seed, eval, sequence).
   - Reported value = mean of `mseScalar` × 10³.
   - Reported ± = **std** of `mseScalar`. This is the spread across seeds, samples and sequences, not a standard error.
   - Deterministic models are repeated 5× (`np.repeat`), so for them the std is effectively over seeds × sequences.
5. **Frames: the conditioning frames are included.** `timeMinMax = (0,240)` slices all T frames of the saved prediction. The saved prediction holds the **ground-truth input frames copied verbatim**:
   - `forwardDirect`: `prediction += [d[:,i]]` for `i < prevSteps`.
   - Diffusion forward: `prediction[:,i] = d[:,i]`.
   - k = 1 copied frame for U-Net, ResNet, FNO and Refiner (arch `unet`, `resnet`, `fno`, `refiner`).
   - k = 2 for ACDM and ACDM_ncn (arch `direct-ddpm+Prev`).
   - For the TF models, frame 0 is the autoencoder reconstruction D(E(x₀)), so it has a non-zero error.
6. **Sequences.**
   - Tra_ext: Ma 0.50–0.52 (sims 0–2), frames [500,620) and [620,740) at stride 2, giving 6 sequences of T=60.
   - Tra_int: Ma 0.66–0.68 (sims 16–18), same frames, 6 sequences.
   - Tra_long: Ma 0.64–0.65 (sims 14–15), frames [0,480) and [480,960) at stride 2, giving 4 sequences of T=240.
   - Code: `filterFrame=[(500,750)], sequenceLength=[[60,2]]` and `[(0,1000)], [[240,2]]`.

**Comparison with the audit convention ("physical units, obstacle zeroed, mean over 4 fields, H, W and predicted frames only", `FINAL_AUDIT.md` §8):**

| Aspect | ACDM paper | Audit convention | Same? |
|---|---|---|---|
| Units | physical (de-normalised with the stats above) | physical | yes |
| Obstacle | zeroed in GT and pred; pixels kept in the mean | zeroed; 1.06% of pixels kept in the mean | yes |
| Fields | v_x, v_y, ρ, p (Mach excluded) | 4 fields | yes |
| Spatial | mean over 128×64 | mean over H, W | yes |
| Temporal | mean over **all T frames, including the k copied input frames** | predicted frames only | **no** |
| Aggregation | mean over 3 seeds × 5 samples × sequences; ± = std | 1 checkpoint (per §8) | differs |
| Temporal stride | 2 exported frames per model step | must also be 2 | check |

[derived] Conversion: paper_MSE = ((T−k)/T) × predicted-only_MSE.

| T | k=1 | k=2 |
|---|---|---|
| 60 | ×0.983 | ×0.967 |
| 240 | ×0.996 | ×0.992 |

Paper values converted to predicted-only, approximately (inputs are rounded to 2 significant figures):

| Model | Tra_int | Tra_ext |
|---|---|---|
| ACDM | 2.79 | 2.38 |
| U-Net_tn | 1.83 | 1.42 |
| U-Net_ut | 1.53 | 1.63 |
| FNO16 | 5.59 | 4.88 |

The audit's single-checkpoint ACDM re-evaluation (Tra_int 2.81, Tra_ext 1.99) agrees with the paper on Tra_int to within rounding. Tra_ext is lower than the paper's 3-seed mean, consistent with the large seed and sample spread (± 1.4).

**Local deviation to check.** The working-tree (uncommitted) `autoreg_pde_diffusion/src/sample_models_tra.py` uses `sequenceLength=[[120,1]]` (ext/int) and `[[480,1]]` (long) instead of the official `[[60,2]]` / `[[240,2]]`.
- The official checkpoints were trained at stride 2 (`sequenceLength=[m,2]`).
- Any npz produced with stride 1 therefore runs the models at half their training Δt and must not be compared with the paper.
- The 2.81 value matches the paper, which suggests it was produced at stride 2. This should be confirmed from the npz shape: T=60 vs 120.

### 1.4 Training data of the official Tra models [code + verified]

- **Simulation** (App. A.2):
  - SU2, SA-DDES, Re = 10⁴, Ma ∈ {0.50, 0.51, …, 0.90} (41 sims).
  - Steady RANS initialisation (1000 iterations), then 150 000 unsteady iterations.
  - Every 50th step is exported after the first 100 000, giving T = 1000 exported frames per sim.
  - Non-dimensional solver step: 0.002·D/U∞.
  - RBF-resampled (linear basis, 5 nearest points) to a 128×64 near-field grid.
- **Training split:** 33 sims, `filterSim=[[0,1,2,14,15,16,17,18]], excludefilterSim=True`, i.e. Ma ∈ {0.53,…,0.63} ∪ {0.69,…,0.90}. Excluded: 0.50–0.52 (ext), 0.64–0.65 (long), 0.66–0.68 (int).
- **Frames:** all exported frames `filterFrame=[(0,1000)]`, at **temporal stride 2** ("Training and test sequences use a temporal stride of 2").
  - [derived] One model step = 2 × 50 × 0.002 = 0.2 D/U∞.
  - With ≈ 8 shedding periods in 240 steps (paper), the period is ≈ 30 steps, so St ≈ 1/(30 × 0.2) ≈ 0.17.
- **Random sequence offset** (`randSeqOffset=True`): each training sequence start is shifted by up to half the sequence length on every load. The paper says this is "especially crucial for the oscillating cylinder flows when training models with longer rollouts", to avoid correlating vortex phase with rollout position. This matters for multi-step KAE training (§3, idea 5).
- **Training windows:** `sequenceLength=[m,2]`.

  | Models | m | Notes |
  |---|---|---|
  | U-Net, ResNet, FNO, Refiner | 2 | 1 input + 1 target |
  | U-Net_ut | 8 | 1 input + 7 unrolled |
  | ACDM | 3 | 2 conditioning + 1 target |
  | TF variants | 60 | transformer window 30; rollout faded in, `fadeInSeqLen=[300,1200]` |

- **Optimiser and epochs:**
  - Adam, lr 1e-4.
  - Epochs: U-Net 1000, FNO 2000, ACDM 3100, Refiner 3000, TF 5000.
  - Batch 64, reduced for unrolled models: m8 → 16.
- **Mach input:** a constant, spatially expanded, normalised channel ((Ma−0.7)/0.118). During rollout, the predicted Mach channel is **overwritten with the true value** at every step.
  - [derived] Tra_int (0.66–0.68) differs from the nearest training Mach (0.63 or 0.69) by only 0.08–0.25 normalised units.

### 1.5 Model-name mapping (code IDs → paper names)

| Code ID | Paper name | Meaning |
|---|---|---|
| `unet-m2` | U-Net | Plain next-step U-Net, k=1 input, trained on 2-frame windows (one-step MSE) |
| `unet-m8` | U-Net_ut | Same U-Net trained with an 8-frame window: MSE on 7 unrolled steps with gradients through the rollout. Paper also calls it U-Net_m8 |
| `unet-m2-noise0.01` | U-Net_tn | One-step U-Net with Gaussian input noise of std n = 10⁻² (normalised units) during training. Paper also calls it U-Net_n1e-2 |
| `acdm-r20` | ACDM | DDPM with R = 20 steps, linear schedule, k = 2 conditioning frames, "noisy" conditioning integration (conditioning noised along the diffusion chain) |
| `acdm-r20_ncn` | ACDM_ncn | Same, with "clean" conditioning (no conditioning noise) |
| `refiner-r4_std0.000001` | Refiner | PDE-Refiner with R = 4 refinement steps and noise σ = 10⁻⁶ (σ = 10⁻⁵ is used for Iso) |
| `tf-mgn`, `tf-enc`, `tf-vae` | TF_MGN, TF_Enc, TF_VAE | Latent-space transformers on a CNN autoencoder latent (latentSize 32 / 31 / 63, plus the simulation parameter): MeshGraphNet-style transformer (Han et al. 2021 adapted to grids), transformer encoder, and VAE variant |
| `resnet-m2`, `dil-resnet-m2` | ResNet, ResNet_dil. | Stachenfeld-style (dilated) ResNet, one-step |
| `fno-16modes-m2`, `fno-32modes-m2` | FNO16, FNO32 | 4-layer FNO with (16,8) / (32,16) modes |

Probabilistic models (ACDM, ACDM_ncn, Refiner, TF_VAE) are evaluated with 5 samples per seed. The MSE is computed **per sample and then averaged**, not on the ensemble mean, so ACDM pays the double penalty.

### 1.6 Availability [verified]

- **Code:** github.com/tum-pbs/autoreg-pde-diffusion, MIT licence, 120 stars, last push 2024-12-13.
- **Data and checkpoints:** mediaTUM, DOI 10.14459/2024mp1734798, which resolves to mediatum.ub.tum.de/1734798 (HTTP 200). rsync password m1734798.001 is per the README.
  - `128_tra.zip`: 12.1 GB.
  - `checkpoints.zip`: 1.9 GB, all architectures and data sets.
  - `checkpoints_acdm_tra.zip`: 75 MB.
- **Local copy:** all 13 Tra model IDs × 3 seeds (`_00`, `_01`, `_02`) are present in `autoreg_pde_diffusion/pretrained_models/models_tra/`. A full 3-seed × 5-sample re-evaluation under the paper's convention is therefore possible without retraining.

### 1.7 Consequences for the KAE comparison

- Quote ACDM, U-Net_tn and U-Net_ut together. On Tra_ext and Tra_int the strongest models are the unrolled and noise-trained U-Nets, with U-Net_m4 best on Tra_int at 1.1. ACDM is mid-pack on MSE and better on LSiM, stability and spectra.
- Report both conventions, or convert with (T−k)/T. Use 3 seeds.
- For Tra_long, the paper defines no MSE. Under `FINAL_AUDIT.md`, a **constant training climatology (15.83e-3) beats both the KAE (16.91) and ACDM (22.51)**. Long-horizon MSE on Tra therefore mostly measures decorrelation and mean-regression, not skill. Report the rate-of-change curve (paper Fig. 10), spectra and ACC-vs-lead instead.

---

## 2. Targeted literature review (improving the conditional continuous-time KAE)

Classes: (1) directly applicable; (2) moderate adaptation; (3) major redesign; (4) inappropriate for this data or budget. Code status: "verified" means the GitHub API returned 200 this session.

### 2A. Parameter-dependent / conditional Koopman

**A1. Korda & Mezić, "Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control".** Automatica 93 (2018). [arXiv:1611.03537](https://arxiv.org/abs/1611.03537) [verified abstract].
- Idea: lift z = ψ(x), then z⁺ = A z + B u, with (A, B) by least squares (EDMD with inputs). Inputs enter **additively and linearly**, and the dictionary acts on the state only.
- Code: none verified. Their MATLAB code is distributed with the paper (not checked).
- Class **(1)** for the piece that matters: a Mach-dependent **affine forcing** dz/dt = A(c) z + b(c), equivalently A(c)(z − z*(c)), which lets the latent fixed point (mean flow) move with Mach. Mach is constant per sequence, so B·c is a constant forcing.

**A2. Proctor, Brunton, Kutz, "Dynamic mode decomposition with control (DMDc)".** SIAM J. Appl. Dyn. Syst. 15 (2016). [arXiv:1409.6358](https://arxiv.org/abs/1409.6358) [verified abstract].
- Idea: x⁺ = A x + B u, solved jointly via SVD of [X; Υ]. It disambiguates intrinsic dynamics from actuation.
- Code: `pydmd/dmdc.py` in PyDMD (verified file).
- Class **(2)**. Useful as a POD-level baseline with Mach as "input". It cannot represent a Mach-dependent *frequency*, because B only shifts.

**A3. Peitz, Otto, Rowley, "Data-Driven Model Predictive Control using Interpolated Koopman Generators".** [arXiv:2003.07094](https://arxiv.org/abs/2003.07094) (SIAM J. Appl. Dyn. Syst. 2020 per my memory, UNVERIFIED). [verified text]
- Idea: for control-affine systems the **Koopman generator is affine in the input** (their Thm 3.2): L_u = L₀ + Σ uᵢ(Lᵢ − L₀). Finite-time operators interpolate only to first order in t.
- Implication: interpolate **generators** (A(c) = A₀ + (c−c₀)A₁), not step maps.
- Code: none found.
- Class **(1)**. The cheapest principled A(c), which here would be a bilinear model. Caveat: Mach enters compressible NS non-affinely (shocks above ≈ 0.66), so it is a first-order model in c.

**A4. Andreuzzi, Demo, Rozza, "A Dynamic Mode Decomposition Extension for the Forecasting of Parametric Dynamical Systems".** SIAM J. Appl. Dyn. Syst. 22(3) (2023). [arXiv:2110.09155](https://arxiv.org/abs/2110.09155) [verified abstract].
- Idea: POD-reduce all (parameter, time) snapshots, run DMD per training parameter, then **regress/interpolate the forecast reduced coefficients** across parameters to reach unseen μ.
- Code: `pydmd/paramdmd.py` (ParametricDMD, verified file).
- Class **(2)**. A strong, cheap *baseline* for Tra_int: interpolation between Ma 0.63 and 0.69. It interpolates outputs, not dynamics, so it cannot fix phase drift.

**A5. Guo, Korda, Kevrekidis, Li, "Learning Parametric Koopman Decompositions for Prediction and Control".** [arXiv:2310.01124](https://arxiv.org/abs/2310.01124) (v2; venue UNVERIFIED, none on arXiv) [verified text].
- Idea: a shared NN dictionary Ψ(x) and an NN matrix-valued K(u; θ_K) with Ψ(x_{n+1}) ≈ K(u_n)Ψ(x_n). They show that linear/bilinear parameter dependence (Σ hᵢ(u)Kᵢ) is a special case. The continuous-time form is K̃(u) = (K(u) − I)/Δt.
- Code: [GUOYUE-Cynthia/Learning-Parametric-Koopman-Decompositions](https://github.com/GUOYUE-Cynthia/Learning-Parametric-Koopman-Decompositions) (verified).
- Class **(1)**: the theory and justification for an NN-generated A(c) with a parameter-independent encoder.

**A6. Geneva & Zabaras, "Transformers for modeling physical systems".** Neural Networks (2022). [arXiv:2010.03957](https://arxiv.org/abs/2010.03957) [verified text].
- Closest published precedent for *this exact setup*:
  - CNN Koopman embedding of 2-D cylinder flow (u_x, u_y, p; 64×128) to a **128-d** latent;
  - Re ~ U(100, 750);
  - 27 / 6 / 7 train / val / test sims of 400 steps.
- "Two auxiliary fully-connected networks are used to predict the **diagonal and off-diagonal elements of the Koopman operator for each viscosity ν**". K is a learnable **banded** matrix (they report this "helped … preventing the model from overfitting to high-frequency fluctuations"), trained with a K^j multi-step loss + ‖K‖² decay.
- Code: [zabaras/transformer-physx](https://github.com/zabaras/transformer-physx) (verified; the README lists the cylinder example as unfinished, per search snippet).
- Class **(1)**: a hypernetwork-generated, banded, ν-dependent Koopman matrix on cylinder wakes. They used it only as an embedding for a transformer, so no stand-alone KAE accuracy is reported.

**A7. Somasekharan, Cao, Pan, "Deep Koopman Sensing".** [arXiv:2609.15762](https://arxiv.org/abs/2609.15762) (Sep 2026, v2; no venue) [verified text].
- Almost the same model as ours:
  - z = E(x, μ), x̂ = G(z, μ), s = e_ω(μ) (MLP);
  - **A(μ) = Q(s) − Q(s)ᵀ − diag(d(s)∘d(s))**, with Q, d from MLPs;
  - K(μ; Δt) = exp(A(μ)Δt), which is non-expansive by construction.
- Setup: 2-D cylinder on 64×64, Re ∈ [59.5, 990.5], 90 / 11 trajectories of 100 steps. Encoder Conv(4→128→256→512, stride 2) → flatten → FC → r. **r = 32** is the main setting.
- Finding: an unconstrained **MLP latent stepper beats the Koopman model open-loop on 3 of 4 benchmarks**. Cylinder final-time MSE with approximate init: MLP 3.01e-2 vs Koopman 5.33e-2 (Table 6; units are their normalisation). The Koopman model is best under EKF/EnKF assimilation.
- Code: none found.
- Class **(1)** as a reference design. It also warns that a dissipative skew+diag generator gives up open-loop accuracy.

**A8. Wang, Dong, Arık, Yu, "Koopman Neural Forecaster (KNF)".** ICLR 2023. [arXiv:2210.03675](https://arxiv.org/abs/2210.03675) [verified abstract].
- Idea: a global K (shared), a local K computed by attention over the lookback window (per-sample), and a feedback K updated from prediction residuals.
- Code: google-research/KNF (verified in the previous review).
- Class **(4)** as a whole: a time-series method with no PDE fields, and the benchmark gives only 2 input frames, so no lookback window exists for a local operator. The *global + per-parameter residual* idea is already covered by A3/A5.

**A9. Vlachas, Vlachas, Chatzi, "Beyond Static Models: Hypernetworks for Adaptive and Generalizable Forecasting in Complex Parametric Dynamical Systems (PHLieNet)".** [arXiv:2506.19609](https://arxiv.org/abs/2506.19609) [verified abstract].
- Idea: parameter → embedding → hypernetwork that emits the weights of a (nonlinear) propagator, i.e. "interpolating in the space of models".
- Code: none found.
- Class **(2)** if the emitted object is A(c) or (γ(c), ω(c)) rather than a full network. It is evidence that model-space interpolation generalises across parameters better than input-concatenation.

**A10. Iwata & Kawahara, meta-learning Koopman spectra** ([arXiv:2102.04683](https://arxiv.org/abs/2102.04683), not opened). Class **(4)**: needs per-task context sequences, which the benchmark does not provide.

### 2B. Structured continuous generators

**B1. Lusch, Kutz, Brunton (2018), auxiliary eigenvalue network.** Nat. Commun. 9:4950. [arXiv:1712.09707](https://arxiv.org/abs/1712.09707) [verified text]. Code: [BethanyL/DeepKoopman](https://github.com/BethanyL/DeepKoopman) (verified).
- **Verification of the claim.** K is block-diagonal with 2×2 Jordan blocks B(μ,ω) = exp(μΔt)[[cos ωΔt, −sin ωΔt],[sin ωΔt, cos ωΔt]]. The eigenvalues λ± = μ ± iω are produced by an auxiliary network Λ(y), and "we often parameterize the eigenvalues by the radius λ(‖y‖²)".
- So their (μ, ω) depend on the **latent state (radius)**, which handles continuous spectra (pendulum, cylinder mean-field model). They do **not** depend on an external parameter.
- It is therefore the closest precedent for a per-block (γ, ω) parameterisation with an exact per-block exponential. Making (γ, ω) a function of Mach instead of radius is a small change (as A6 does for ν).
- Class **(1)**.

**B2. Pan & Duraisamy, "Physics-informed probabilistic learning of linear embeddings of nonlinear dynamics with guaranteed stability".** SIAM J. Appl. Dyn. Syst. 19(1) (2020), DOI 10.1137/19M1267246. [arXiv:1906.03663](https://arxiv.org/abs/1906.03663) [verified text].
- Continuous-time KAE with **K_stable = tridiagonal**: diagonal −σᵢ², super-diagonal ζᵢ, sub-diagonal −ζᵢ. K is negative semi-definite, so Re λ ≤ 0. They note a 2×2-block reduction is possible.
- Tested on a cylinder wake at Re = 100 (transient). Uses ADVI for uncertainty.
- Code: none found.
- Class **(1)**. Like A7, it has *strict* dissipation unless σᵢ → 0 can be reached.

**B3. Gillis & Sharma, "On computing the distance to stability for matrices using linear dissipative Hamiltonian systems".** Automatica 85 (2017). [arXiv:1611.00595](https://arxiv.org/abs/1611.00595) [verified abstract].
- Idea: **A is Hurwitz-stable (in their sense) iff A = (J − R)Q with J = −Jᵀ, R ⪰ 0, Q ≻ 0.** This port-Hamiltonian form covers **all** stable matrices, including non-normal ones and ones whose symmetric part is indefinite.
- Our current S + D form covers only those with D = sym(A) ⪯ 0 when D is constrained. Leaving D unconstrained loses the guarantee.
- Code: none for ML.
- Class **(1)**: parameterise J(c), R(c) = L Lᵀ, Q = M Mᵀ + εI.

**B4. Morandin, Nicodemus, Unger, "Port-Hamiltonian Dynamic Mode Decomposition".** [arXiv:2204.13474](https://arxiv.org/abs/2204.13474) (SIAM J. Sci. Comput. 2023 per my memory, UNVERIFIED) [verified abstract].
- Idea: least-squares fit of a continuous-time pH system (J − R)Q with guaranteed passivity.
- Code: not checked.
- Class **(2)**: an initialiser for A from encoded training latents.

**B5. Stable LDS learning.**
- Mamakoukas, Xherija, Murphey, "Memory-Efficient Learning of Stable Linear Dynamical Systems", NeurIPS 2020. [arXiv:2006.03937](https://arxiv.org/abs/2006.03937). Uses the Gillis–Sharma-type discrete characterisation A = S⁻¹OCS. Code: [giorgosmamakoukas/MemoryEfficientStableLDS](https://github.com/giorgosmamakoukas/MemoryEfficientStableLDS) (verified).
- Mamakoukas et al., "Learning Stable Models for Prediction and Control" (DISKO). [arXiv:2005.04291](https://arxiv.org/abs/2005.04291). Stability constraints improve multi-step prediction error bounds.
- Bevanda et al. (diffeomorphic stable Koopman) [arXiv:2112.04085](https://arxiv.org/abs/2112.04085) and Fan et al. [arXiv:2401.08153](https://arxiv.org/abs/2401.08153) (from the previous review).
- Class **(2)**. Discrete-time; useful for theory and initialisation, not needed if B3 is used.

**B6. LRU (Orvieto et al., ICML 2023, [arXiv:2303.06349](https://arxiv.org/abs/2303.06349)) and S5 (Smith et al., ICLR 2023, [arXiv:2208.04933](https://arxiv.org/abs/2208.04933); code [lindermanlab/S5](https://github.com/lindermanlab/S5) verified).** Details verified in the previous review.
- λ = exp(−exp(ν) + iθ) (LRU) or ZOH exp(ΔΛ) (S5) for complex-diagonal Λ.
- Ring initialisation near |λ| = 1 helps long memory **only when stability is enforced**.
- Class **(1)**. A complex-diagonal A(c) = diag(−softplus(γ_k(c)) + iω_k(c)) with an exact O(d) exponential is a direct substitute for dense S + D.

**B7. Kerg et al., "Non-normal RNN (nnRNN)".** NeurIPS 2019. [arXiv:1905.12080](https://arxiv.org/abs/1905.12080) [verified abstract].
- Idea: Schur form A = P(Λ + T)Pᵀ with P orthogonal, Λ block-normal (unit-modulus eigenvalues) and T strictly upper-triangular. Spectrum control is decoupled from **non-normal transient amplification**.
- Code: `nkerg/nnRNN` returned 404. Repo not found; UNVERIFIED.
- Class **(2)**. A non-normal part could let a *damped* spectrum still give transient amplitude. For a statistically stationary limit cycle this is not the main need.

### 2C. Long-horizon training, amplitude-preserving losses, mean regression

**C1. Lange, Brunton, Kutz, "From Fourier to Koopman".** JMLR 22 (2021). [arXiv:2004.00574](https://arxiv.org/abs/2004.00574); code [helange23/from_fourier_to_koopman](https://github.com/helange23/from_fourier_to_koopman) (verified).
- Idea: the latent is exactly periodic, cos/sin(ω_k t). The ω_k are found by a global FFT-based objective (avoiding local minima of the time-domain loss). A nonlinear decoder handles the rest. There is **no amplitude decay by construction**.
- Class **(2)**: allocate a few *exactly neutral* blocks (γ = 0) for the shedding fundamental and harmonics, and initialise ω from the FFT of the training data.

**C2. Subich, Husain, Separovic, Yang, "Fixing the Double Penalty in Data-Driven Weather Forecasting Through a Modified Spherical Harmonic Loss Function".** ICML 2025 (PMLR 267). [arXiv:2501.19374](https://arxiv.org/abs/2501.19374) [verified text]. Code: [csubich/graphcast](https://github.com/csubich/graphcast) branch `amse` (repo verified; branch not checked).
- Idea: per wavenumber k, MSE = PSD_k(x) + PSD_k(y) − 2√(PSD_k(x)PSD_k(y))·Coh_k. Under MSE the optimum amplitude ratio equals the coherence, so models **shrink unpredictable scales**. AMSE = Σ_k (√PSD_k(x) − √PSD_k(y))² + 2·max(PSD_k(x), PSD_k(y))·(1 − Coh_k) decouples amplitude from coherence. It is zero iff x = y and has the same Taylor expansion as MSE.
- Class **(2)**. Use the 2-D FFT on the 128×64 grid, or apply it per latent Koopman mode in time.
- Caveat: AMSE **raises evaluation MSE** wherever phase is uncertain.

**C3. Mean regression under MSE.** [derived, standard]
- Take a single oscillatory mode a·cos(ωt + φ) predicted as r·a·cos(ωt + φ + δ). Then E[MSE] = (a²/2)(1 + r² − 2r·E[cos δ]), minimised at **r* = E[cos δ]**. If the phase error grows diffusively (δ ~ N(0, Dt)), r*(t) = exp(−Dt/2). So an MSE-trained linear model learns **exponential amplitude decay at the rate at which it loses phase**.
- [derived] Applied to our model: e-folding 36–41 steps means D ≈ 2/38 ≈ 0.05 rad²/step, i.e. ≈ 0.23 rad phase-error std after 1 step growing to ≈ 1.8 rad after 60 steps. The learned damping should be read as a **phase-uncertainty hedge, not a numerical artefact**.
- Literature naming this effect:
  - "double penalty" (C2);
  - Chattopadhyay, Sun, Hassanzadeh, [arXiv:2304.07029](https://arxiv.org/abs/2304.07029) (spectral bias → instability and smoothing);
  - McCabe et al. TMLR 2023 (previous review F2);
  - ACDM's own observation that next-step models "settle into … a mean flow prediction without vortices" on Tra_long.
- Fixes in the literature:
  - (a) improve phase accuracy (better dynamics or conditioning);
  - (b) amplitude/coherence-split losses (C2);
  - (c) probabilistic models scored per sample (ACDM);
  - (d) evaluate with spectra and ACC instead of MSE.

**C4. Shaffer, Vorenberg, Hsieh, "Spectrally Informed Learning of Fluid Flows".** [arXiv:2408.14407](https://arxiv.org/abs/2408.14407) (2024; venue UNVERIFIED) [verified text].
- Builds on the Lusch-style KAE. It regularises the **operator spectrum**: relative power per eigenfrequency (from eigenvalues of Ω and amplitudes b = Φ†z) is matched to a prior temporal power spectrum, and an explicit low-eigenfrequency bias is added.
- Code: none found.
- Class **(2)**: use the measured shedding spectrum per Mach as the prior.

**C5. Choi, Krishnan, Park, "Koopman AutoEncoder via Singular Value Decomposition for Data-Driven Long-Term Prediction".** IEEE MLSP 2024. [arXiv:2408.11303](https://arxiv.org/abs/2408.11303) [verified abstract].
- A loss on the singular values of K pushes eigenvalues toward the unit circle and improves long-term prediction.
- Code: none found.
- Class **(1)**, as a cheap regulariser. Only useful once phase is accurate (see C3).

**C6. Unrolled training.**
- List, Chen, Bali, Thuerey, "Differentiability in Unrolled Training of Neural Physics Simulators on Transient Dynamics", [arXiv:2402.12971](https://arxiv.org/abs/2402.12971) (CMAME 2025 per my memory, UNVERIFIED); code [tum-pbs/unrolling](https://github.com/tum-pbs/unrolling) (verified). It separates two effects of unrolling: training-distribution shift and long-term gradients. Fully differentiable unrolling performed best among pure prediction setups.
- ACDM Table 5: U-Net m = 4 or 8 is best and m = 16 is worse.
- Lusch et al. use a multi-step prediction loss over S_p steps.
- Geneva & Zabaras use K^j F(φ₀) over the full window.
- Class **(1)**: for a linear latent, z(t) = exp(tA(c)) z₀ for **all** horizons is one batched `matrix_exp`, or O(d) per block, so 60–120-step losses are cheap. Combine with ACDM-style random start offsets.

### 2D. Encoder/decoder design for KAEs on 2-D flows

| Work | Grid | Encoder → latent | Latent dim | Notes |
|---|---|---|---|---|
| Geneva & Zabaras 2022 (A6) [verified] | 64×128, 3 fields + ν channel | 5-layer conv, then FC | 128 | ν fed to the encoder **and** to the K hypernetworks |
| Somasekharan et al. 2026 (A7) [verified] | 64×64, u, v, p, cell type | 3 strided convs → flatten (512×8×8) → FC 512 → 256 → r | 32 (main) | encoder and decoder both conditioned on μ via an embedding |
| Morton et al. 2018, NeurIPS [verified text; arXiv:1805.07472](https://arxiv.org/abs/1805.07472), code [sisl/deep_flow_control](https://github.com/sisl/deep_flow_control) (verified) | cylinder Re = 50 CFD | ResNet convs + FC | 32 | A by **least squares from the first 16 encoded steps** (Takeishi-style), trained on 32-step sequences |
| Pan & Duraisamy 2020 (B2) | cylinder Re = 100 | MLP on POD-reduced data per my memory (UNVERIFIED) | small | tridiagonal stable K |
| Gupta, Schmid, Sipp, Sayadi, Rigas, "Mori–Zwanzig latent space Koopman closure" [arXiv:2310.10745](https://arxiv.org/abs/2310.10745) [verified text] | cylinder Re = 100 (Nektar++) | nonlinear AE | r ∈ {2, 8, 16} | 2 latent modes suffice on the limit cycle. A Koopman-AE alone fails on the transient. MZ memory terms improve it. Rollouts to 1000 time units (≈ 180 shedding cycles). Numbers in figures only, not extracted |
| Solera-Rico et al. 2026 [arXiv:2607.24569](https://arxiv.org/abs/2607.24569) [verified abstract] | controlled wakes (truck, fluidic pinball) | CAE vs POD (LSTM predictors) | — | CAE latents are "more irregular and with broadband spectral content", and long-horizon forecasts degrade faster than with POD latents |

Takeaways:
- All published cylinder KAEs flatten to a 32–128-d latent.
- Only A6 and A7 condition the operator on the flow parameter.
- None of them reports results on the ACDM Tra data.
- Solera-Rico et al. support adding a **latent-smoothness / spectral constraint** on the encoder so that latents are predictable, not just compressive.

### 2E. Recent KAE results on cylinder wake / vortex shedding (2023–2026), with numbers

- **Repo owner's preprint arXiv:2602.02832** (previous review §0): this is the only KAE with ACDM-Tra numbers. It quotes Tra_long 17.0e-3 against "ACDM ~22.2e-3", and its own table lists 22.6e-3, an internal inconsistency. That ACDM figure is a re-evaluation, since the paper has no Tra_long MSE (§1.2).
- **Deep Koopman Sensing 2026 (A7):** cylinder, final-time MSE open-loop, Koopman 5.33e-2 vs MLP 3.01e-2 vs pDMD 2.54e-1 (their normalisation; r = 32).
- **MZ-AE 2023/24 (above):** Re = 100 limit cycle, r = 2, stable over ≈ 180 cycles. Numbers are figures only (UNVERIFIED values).
- **Halder, Eiximeno, Lehmkuhl, "Koopman β-VAE" 2025** [arXiv:2507.18754](https://arxiv.org/abs/2507.18754) [verified abstract]: large-scale turbulence ROMs. Flow case and numbers UNVERIFIED (abstract only).
- **No paper was found that reports a conditional KAE on a transonic or compressible cylinder with Mach dependence.** Searches: "Koopman autoencoder Mach", "parametric Koopman transonic", "hypernetwork Koopman Reynolds cylinder".

---

## 3. Most promising ideas for this model

Baseline facts used below (from `FINAL_AUDIT.md`):

| Metric | Value |
|---|---|
| KAE Tra_int MSE (predicted frames, physical units) | 6.37e-3 |
| ACDM Tra_int MSE (same convention) | 2.81e-3 |
| Paper ACDM Tra_int | 2.7 (≈ 2.79 predicted-only) |
| Paper U-Net_tn / U-Net_ut Tra_int | 1.8 / 1.5 |
| KAE Tra_ext | 1.59 (best) |
| KAE first lead with ACC < 0.6 on Tra_int | 27 steps |
| Shedding period | ≈ 30 model steps |
| Shedding-mode e-folding | 36–41 steps |

[derived] At the dt = 0.1 latent step mentioned in the previous review, ω ≈ 2π/(30 × 0.1) ≈ 2.1 and hω ≈ 0.21, so |P4(iωh)| − 1 ≈ −(hω)⁶/144 ≈ −6e-7 per step. **RK4 is not the source of the 36–41-step damping. The damping is in the learned Re λ.**

### Idea 1. Mach-conditioned block-diagonal oscillator generator (Lusch blocks with Geneva–Zabaras/Guo conditioning)

**Form.**
- A(c) = blockdiag_k [[−γ_k(c), −ω_k(c)], [ω_k(c), −γ_k(c)]], plus optionally a small dense dissipative remainder.
- γ_k = softplus(·) ≥ 0 and ω_k come from a small MLP (or FiLM/hypernetwork) of the normalised Mach.
- Exact per-block exponential: e^{−γt}·Rot(ωt).
- **Feed c to the generator, and initially *not* to the encoder.** Otherwise E can absorb Mach and A can ignore it.

**Why it may help.** On Tra_int the main loss is phase drift: ACC drops below 0.6 at lead 27 ≈ 1 period. A frequency error Δω gives a phase error Δω·t, and MSE then learns damping (§2C, C3). Making ω explicitly Mach-dependent lets the model hit the interpolated shedding frequency at Ma 0.66–0.68. The current model, which ignores Mach, effectively uses one averaged frequency.

**Failure modes addressed.** Ignoring Mach; frequency mismatch, which feeds the damping.

**Assumptions.**
- The shedding frequency (and the harmonic structure) varies measurably with Ma over 0.53–0.90.
- The variation is smooth enough to interpolate across the 0.63–0.69 gap.
- This is **unverified**. It must be measured first: see the validating experiment.

**Cost.** Low: O(d) exponential, a small MLP, and a retrain.

**Likely failure modes.**
- If St is nearly Mach-independent, the gain is nil.
- Above Ma ≈ 0.66, shock–vortex interaction may change the spectrum non-smoothly, so interpolation can be biased.
- The encoder can still leak Mach. Check with a gradient-attribution test or by shuffling c at test time.

**Validating experiment.**
1. *Before training*, compute the dominant temporal frequency per training Mach (FFT of v_y at a wake probe, or of the leading POD coefficient) on all 33 training sims plus held-out 0.64–0.68. If f(Ma) varies by more than ≈ 2% across 0.63 → 0.69, the idea is worth pursuing. [derived] 2% of ω over 60 steps ≈ 0.02 × 2π × 2 ≈ 0.25 rad.
2. After training, plot learned ω₁(c) against the measured f(Ma).
3. Report Tra_int/Tra_ext MSE (3 seeds), ACC-vs-lead and phase error-vs-lead.
4. Ablation: shuffle c at test time. The MSE must get worse; if it does not, Mach is still ignored.

### Idea 2. Mach-dependent affine term / latent mean (Korda–Mezić, DMDc, Peitz)

**Form.** dz/dt = A(c)(z − z*(c)), or A(c) z + b(c). z*(c) is a learned latent fixed point, the "mean flow at this Mach". Decode the mean as D(z*(c), c).

**Why.** Mean ρ and p fields and shock positions shift with Mach. If the latent attractor centre does not move with c, the model's mean flow at Ma 0.66–0.68 is an average of neighbours. That is a bias error that does not decay and is directly visible in MSE. The damped long-horizon state also converges to z*(c), so getting z* right lowers the asymptotic MSE (cf. climatology 15.8e-3 on Tra_long).

**Failure mode addressed.** Ignoring Mach (mean component); long-horizon mean bias.

**Assumptions.** The per-Mach time-mean field varies smoothly in Ma.

**Cost.** Negligible.

**Failure modes.** Redundant if the decoder already uses c well. It can be confounded with idea 1 when both are added, so ablate separately.

**Validation.**
- Decompose Tra_int MSE into (time-mean bias)² + fluctuation error, per field.
- Compare the per-Mach climatology error of the decoded z*(c) with the true time-mean at Ma 0.66–0.68 (held out).

### Idea 3. Stable-by-construction generator with neutral modes allowed (Gillis–Sharma pH form or LRU/S5 complex-diagonal), plus near-axis initialisation

**Form.**
- Option (a): A(c) = (J(c) − R(c))Q, with R = LLᵀ ⪰ 0 (rank-limited) and Q = MMᵀ + εI. This covers all stable A, including non-normal ones.
- Option (b): complex-diagonal λ_k(c) = −softplus(ν_k(c)) + iω_k(c), with LRU-style ring init close to the imaginary axis (small γ).
- Allow γ_k → 0 exactly for a few "limit-cycle" modes, for example a hard-zero subset as in Lange C1.

**Why.**
- The flow is a statistically stationary limit cycle: the true Koopman spectrum on the attractor has eigenvalues *on* the imaginary axis (fundamental and harmonics).
- The A7 and B2 forms (skew − diag(d²)) and an unconstrained sym part either force strict dissipation everywhere or allow growth.
- Neither represents "neutral shedding + dissipative transients" well, and LRU shows that near-unit-modulus init matters for long memory once stability is enforced.

**Failure mode addressed.** Shedding damping as a structural bias; Tra_long stability.

**Assumptions.** The damping is partly due to parameterisation and initialisation, not only to the MSE hedge in C3.

**Cost.** Low to moderate.

**Failure modes.** Under pure MSE, the optimiser will still push γ up to hedge phase error (C3). This must be paired with idea 1 or idea 4, or the gain shows only in spectra and ACC, not in MSE.

**Validation.**
- Eigenvalue plot per Mach.
- Amplitude ratio ‖x̂(t)′‖/‖x(t)′‖ of the fluctuations vs lead.
- Paper Fig. 10-style rate-of-change on Tra_long.
- MSE at T = 60 and T = 240.

### Idea 4. Long-horizon closed-form multi-step loss with ACDM-style random offsets and a horizon curriculum

**Form.**
- Loss Σ_{t=1}^{H} w_t ‖D(e^{tA(c)} E(x₀, c), c) − x_t‖², with H grown 8 → 30 → 60 (→ 120).
- One batched exponential per sample gives all t, or O(d) per block.
- Random window starts, as with ACDM's `randSeqOffset`, so phase is uncorrelated with the window position.
- Use detached or normalised latent targets to avoid scale collapse (previous review §4c).

**Why.**
- If training horizons are shorter than the evaluation horizon (60), errors in ω are barely penalised during training: a phase error of Δω·8 is small.
- Training at H ≈ 30–60 directly penalises frequency error over a full period.
- ACDM Table 5 shows that moderate unrolling (m = 4–8) gives the best Tra_int for U-Net. List et al. 2024 show that differentiable unrolling helps pure-prediction models.

**Failure mode addressed.** Phase drift on Tra_int (lead 27), and therefore the damping.

**Assumptions.** The encoder gives a sufficiently accurate initial phase from 1–2 frames.

**Cost.** Moderate: memory grows with H through the decoder, mitigated by subsampling t.

**Failure modes.**
- Long-horizon MSE again rewards damping (C3). Use w_t that emphasises 1–2 periods, or pair the long-horizon term with idea 5.
- Longer windows reduce the number of distinct training samples per sim (only 33 sims × 500 strided frames).

**Validation.**
- Sweep H ∈ {8, 30, 60}.
- Report the Tra_int MSE curve vs lead, the phase-error slope (rad/step) and learned γ₁.
- The prediction is that γ₁ falls as the phase error falls.

### Idea 5. Amplitude/coherence-split loss (AMSE-style) on time or on space-time spectra

**Form.**
- Add λ·AMSE (C2) computed over the 2-D spatial FFT per frame, and/or over the temporal FFT of the predicted window at a few probes.
- Alternatively add a direct latent-amplitude penalty (|z_k(t)| on neutral blocks matching |E(x_t)_k|), keeping the MSE term.

**Why.** Removes the incentive to shrink amplitude when phase is uncertain, so predictions keep realistic shedding energy. Spectra, LSiM and stability then look like ACDM's.

**Failure mode addressed.** Damping (as a symptom).

**Assumptions.** The evaluation values amplitude and statistics, i.e. LSiM, spectra and ACC, not only MSE.

**Cost.** Low.

**Failure modes.** **It will likely increase MSE** unless phase is fixed (ideas 1 and 4), because MSE-optimal amplitude = E[cos δ]. ACDM, which samples rather than averages, already pays this penalty and still reaches 2.8e-3, which shows that the phase must be good. Use it as a secondary objective, gated in after phase accuracy is achieved.

**Validation.**
- Report MSE, LSiM, the temporal spectrum at a wake probe (paper Fig. 7 style) and the amplitude ratio vs lead.
- Success means the spectra match without more than ≈ 10% MSE loss.

### Idea 6. Operator-level spectral prior per Mach (Shaffer et al.; Lange FFT initialisation)

**Form.**
- Initialise ω_k(c) from the FFT peaks of the training sims (fundamental and harmonics 2f, 3f) by Mach.
- Optionally regularise the operator spectrum toward the measured per-Mach spectrum, as in C4.

**Why.**
- Linear-latent training from random ω is prone to local minima (the motivation for Lange's global FFT objective).
- Data-driven initialisation puts the model near the right frequencies so that the phase-sensitive long-horizon loss (idea 4) is optimisable.

**Failure mode addressed.** Frequency mismatch; optimisation instability.

**Cost.** Low.

**Failure modes.** A broadband transonic spectrum at high Ma may not have clean peaks. Regularising toward training spectra cannot help extrapolation (Tra_ext).

**Validation.** Compare learned vs measured frequency per Mach, and run a convergence-speed comparison with random init.

### Idea 7. Parametric-DMD / bilinear-generator baselines (Andreuzzi; Peitz) as diagnostics

**Form.**
- ParametricDMD (PyDMD) on POD-128 coefficients across the 33 training Machs, interpolated to 0.66–0.68.
- A bilinear POD model A(c) = A₀ + cA₁ fitted by least squares on generators.

**Why.** The audit's POD-128 2-frame linear map already gets Tra_int 6.06 (≈ KAE 6.37). If a Mach-interpolated linear POD model gets well below 6, the gain from idea 1 is real and cheap, and the remaining gap to ACDM is nonlinear.

**Cost.** Minutes on CPU.

**Failure modes.** POD truncation error floors the result.

**Validation.** Tra_int / Tra_ext MSE under the audit convention.

### Idea 8 (lower priority). Conditioning the decoder via FiLM and keeping a spatial bottleneck

**Form.** FiLM(c) in decoder blocks. Optionally a CNN latent with a few channels at low resolution instead of a flatten-to-128 bottleneck, with the latent operator acting per channel via a 1×1 conv.

**Why.** Reconstruction floor (RMSE 0.13–0.17 normalised, `FINAL_AUDIT.md`), shock-position sharpness at high Ma.

**Cost.** Moderate (3). This is a change of latent topology, and it moves away from the published 32–128-d flattened designs (A6, A7, Morton).

**Validation.** Reconstruction-only MSE per Mach before any dynamics work.

**Recommended order.**
1. Measure f(Ma) and the per-Mach mean (no training).
2. Idea 7 (minutes).
3. Ideas 1 + 2 + 6 together: one retrain, 3 seeds.
4. Idea 4 sweep.
5. Idea 3.
6. Idea 5 as a fine-tuning stage.

Report every result under both MSE conventions (§1.3), with 3 seeds, plus ACC-vs-lead and spectra.

---

## Sources opened this session

- **ACDM:**
  - arxiv.org/html/2309.01745v1, v2, v3; arxiv.org/abs/2309.01745.
  - Crossref record for DOI 10.1016/j.neunet.2026.108641.
  - github.com/tum-pbs/autoreg-pde-diffusion (README; `src/plot_loss_all.py`, `sample_models_tra.py`, `training_tra.py`, `training_diffusion_tra.py`, `turbpred/model.py`, `turbpred/data_transformations.py`, `turbpred/turbulence_dataset.py`, `plot_color_and_name_mapping.py`).
  - doi.org/10.14459/2024mp1734798 → mediatum.ub.tum.de/1734798.
  - [ICML 2024 listing](https://icml.cc/virtual/2024/36841) (search result only).
- **Full texts (HTML/ar5iv):** 1712.09707, 1906.03663, 2010.03957, 2310.01124v2, 2609.15762v2, 2408.14407v1, 2501.19374v2, 1805.07472, 2310.10745v3, 2003.07094.
- **Abstracts (arXiv API):** 1611.03537, 1409.6358, 1611.00595, 2006.03937, 2005.04291, 2204.13474, 2110.09155, 1905.12080, 2210.03675, 2304.07029, 2402.12971, 2408.11303, 2506.19609, 2507.18754, 2607.24569, 2303.06349, 2208.04933, 2004.00574, 2112.04085, 2401.08153.
- **GitHub API existence checks (HTTP 200):** zabaras/transformer-physx, GUOYUE-Cynthia/Learning-Parametric-Koopman-Decompositions, sisl/deep_flow_control, csubich/graphcast, tum-pbs/unrolling, BethanyL/DeepKoopman, lindermanlab/S5, helange23/from_fourier_to_koopman, giorgosmamakoukas/MemoryEfficientStableLDS, PyDMD/PyDMD (`pydmd/paramdmd.py`, `dmdc.py`, `bopdmd.py`), dynamicslab/pykoopman, erichson/koopmanAE, Koopman-Laboratory/KoopmanLab.
- **Returned 404:** nkerg/nnRNN.
- **Local (read-only):**
  - `ocean_forecasting_audit/audit_report/LITERATURE_REVIEW.md`;
  - `ocean_forecasting_impl/audit_report/FINAL_AUDIT.md` §8;
  - `autoreg_pde_diffusion/src/{plot_loss_all.py, sample_models_tra.py}` (diffed against upstream);
  - `autoreg_pde_diffusion/pretrained_models/models_tra/`.
