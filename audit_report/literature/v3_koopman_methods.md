# Koopman and linear-latent methods for long-horizon accuracy: literature for the conditional CT-KAE on Tra

Date: 2026-10-22. Scope: Koopman autoencoders (KAE), DMD-family methods and linear-latent models. The focus is on reducing long-horizon error growth, phase drift, amplitude decay, blur and harmonic error.

Verification policy:
- "Verified" means I downloaded the arXiv PDF, extracted the text with pdftotext, and read the relevant method and results sections. The text copies are in `/tmp/claude-1003/-home-rg625-mnt-ocean-forecasting/563da406-8943-47fe-a8f7-28eddbd027a5/scratchpad/koop/`.
- Venue claims are checked against the PDF header, a proceedings page, or JMLR/ICLR listings, as noted. Anything else is marked UNVERIFIED.
- Numbers quoted below come from the papers' own tables or text. Where a table's layout was garbled by text extraction, I quote only the paper's text claims.

Constraint codes used in the verdicts:
- **L**: latent dynamics remain exactly linear.
- **CT**: continuous time and zero-shot dt.
- **Ma**: compatible with Mach conditioning.
- **FO**: forecasting-only training, with no DA objective.
- **H8**: 2 conditioning frames and about 8 supervised frames.
- **Eff**: the latent stays small.

Verdicts are ✔ (compatible), ~ (compatible with caveats) or ✘ (incompatible).

---

## 0. A quantitative reading of the existing diagnostics (motivates the ranking)

**Damping is invisible inside the training window but large at the test horizon.**
- The median damping is γ ≈ 0.086 per time unit, with dt = 0.1 per frame.
- At lead 8 (the end of the supervised window), a median mode keeps e^{-0.086·0.8} = **0.93** of its amplitude. The MSE penalty for spurious damping during training is therefore only a few percent.
- At lead 30 it keeps 0.77, and at lead 58 it keeps e^{-0.086·5.8} = **0.61**.
- The global energy ratio at lead 58 is 0.94. So the energetic modes (the shedding pair) are almost neutral, and the strongly damped modes are low-energy ones: harmonics and finer structure.
- This matches the error budget. At lead 30, about 32% of the excess error is in POD modes 3-10 and about 49% is outside the span of the true trajectory. Shrinking harmonic and fine-scale latents makes a nonlinear decoder output smoothed, off-manifold fields.
- Hypothesis H1: a large part of the growing excess error comes from spurious damping of non-leading modes. That damping is learned under an 8-frame horizon that cannot see it.

**Koopman theory says these modes should not decay on the attractor.**
- For post-transient (statistically stationary) dynamics, the Koopman operator on L²(μ) is unitary. Its spectrum lies on the unit circle, or on the imaginary axis in continuous time (Colbrook, mpEDMD, §2.1).
- For a stable limit cycle with base frequency ω₀, the Koopman eigenvalues form the lattice {i·n·ω₀ + m·λ_Floquet}. The pure-oscillation part is ±i·n·ω₀ (Mauroy & Mezić 2018; dymad tutorial).
- An autoencoder Koopman model of the Re=100 cylinder limit cycle learned Re(λ) ≈ 8e-5, i.e. almost exactly neutral (MZ-AE, Gupta et al.).
- Unless Tra sequences contain transients, gammas of order 0.1 are an artefact rather than physics.

**Free per-mode frequencies can dephase the harmonics.**
- In our block-diagonal A, each oscillator has an independent ω_k.
- If harmonic content is spread over modes whose frequencies are not exact integer multiples of the fundamental, the relative phase between fundamental and harmonics drifts with lead. A nonlinear decoder renders this as distorted, misplaced structure, i.e. outside-span error. This happens even when the fundamental frequency is accurate, which ours is (0.2-1.5%).
- This hypothesis (H2) is not tested in any paper I found (see §3). It can be checked cheaply (see §4).

**Zero-cost tests of H1 and H2 on the existing checkpoint.** Run these before any retraining:
- (a) At inference, set γ_k → 0, or γ_k → α·γ_k with α ∈ {0, 0.25, 0.5}, and re-evaluate the 60- and 240-step errors.
- (b) Histogram ω_k/ω_1(Ma) for modes with non-negligible decoder sensitivity, and check for clustering near integers. Then snap each ω_k to n_k·ω_1 and re-evaluate.
- (c) Measure the phase error of the shedding pair at leads 1-5 (the intercept) versus its slope. This separates initial-phase encoding error from frequency error.

If (a) helps, the cheap regularisers in §5 (#1) are the right first move. If (a) makes things worse, damping is protecting the decoder from spurious modes, and the spectral-hygiene items (ResDMD-style residuals, whitening) come first.

---

## 1. Spectral structure: neutrality, unitarity and stability of the generator

### 1.1 Eigenvalue initialisation and regularisation ("eigeninit" and "eigenloss")
- **Citation:** J. W. Miller, C. O'Neill, N. C. Constantinou, O. Azencot. *Eigenvalue initialisation and regularisation for Koopman autoencoders.* arXiv 2212.12086 (2022). https://arxiv.org/abs/2212.12086. Verified from the PDF. No peer-reviewed venue is visible (UNVERIFIED).
- **Mechanism:**
  - Eigeninit samples the initial Koopman matrix from a chosen eigenvalue distribution, for example radii uniform on [0, 1] or [0.9, 1].
  - Eigenloss adds λ·Σ_j(|λ_j| − 1)² to the loss, pulling the eigenvalues onto the unit circle.
  - The authors explicitly recommend a stronger weight for measure-preserving systems.
- **Evidence (Table 1, cumulative long-term test error), on the cylinder u-velocity ("Fluid"):**

  | Variant | Error |
  |---|---|
  | Plain KAE | 1.25 |
  | Eigeninit | **0.58** |
  | Eigenloss | 0.82 |
  | Both | 0.96 |
  | Consistent KAE | 1.00 |

  - The headline claim is up to 3× lower cumulative error and up to 5× faster convergence.
  - Effects on pendulum and cyclone data are mixed: the combination is sometimes worse.
- **Constraints:** L ✔ · CT ✔ (use Σ γ_k² or Σ(1 − e^{−γ_k Δt})²) · Ma ✔ (apply to A(Ma) at sampled Ma) · FO ✔ · H8 ✔ · Eff ✔.
- **Targets:** damping, and therefore the harmonic and outside-span error that grows with lead.
- **Caveat:** single seeds and small models; the improvement is not monotone across datasets.

### 1.2 mpEDMD: measure-preserving EDMD
- **Citation:** M. J. Colbrook. *The mpEDMD algorithm for data-driven computations of measure-preserving dynamical systems.* SIAM J. Numer. Anal. 61(3), 2023. arXiv 2209.02244, https://arxiv.org/abs/2209.02244. Verified from the PDF. The venue comes from the author's GitHub and a search result; I did not open the SIAM page.
- **Mechanism:**
  - The Koopman matrix is fitted by an orthogonal Procrustes problem in the Gram-matrix (G = Ψ*WΨ) inner product. The truncated operator is therefore exactly unitary with respect to the data measure, and its eigenvalues lie on the unit circle.
  - The paper proves convergence of the spectral measures and spectra.
  - Key detail: unitarity must hold in the *data-measure* inner product, through G, not the Euclidean one.
- **Evidence:**
  - Turbulent boundary layer (Re > 6×10⁴), forecasting 5 s (5× the observation window).
  - mpEDMD conserves TKE (relative error of time-averaged TKE ≤ 0.001) and matches the wavenumber spectrum.
  - EDMD is unstable. piDMD is stable but has the wrong TKE.
- **Constraints:** L ✔ · CT ✔ (skew-adjoint generator, γ = 0) · Ma ✔ · FO ✔ · H8 ✔ · Eff ✔.
- **Note:** it is a DMD algorithm, not a KAE. The transferable idea is "neutral generator plus whitened latent", combined with DPNets in §1.4.
- **Targets:** damping and amplitude decay. It does not target phase.

### 1.3 Neural Koopman prior with orthogonal K / skew-symmetric generator
- **Citation:** A. Frion, L. Drumetz, M. Dalla Mura, G. Tochon, A. Aïssa El Bey. *Neural Koopman prior for data assimilation.* IEEE Trans. Signal Processing (the PDF header says Transactions on Signal Processing), 2024. arXiv 2309.05317, https://arxiv.org/abs/2309.05317. Verified from the PDF.
- **Mechanism:**
  - A soft orthogonality penalty ‖KᵀK − I‖ is applied to K. Theorem 1 states that K ∈ SO(d) corresponds to a skew-symmetric continuous-time generator, which gives periodic dynamics.
  - K is initialised to the identity.
  - It is used for long-term continuous-time interpolation and extrapolation of remote-sensing series.
- **Evidence:**
  - Orthogonality mainly helped backward prediction and irregular-sampling interpolation (their Table II).
  - On low-dimensional examples, their model interpolates to high frequency far better than DeepKoopman and cKAE when trained at low frequency.
  - No fluid results.
- **Constraints:** L ✔ · CT ✔ · Ma ✔ · FO ~ (the paper targets DA, but the prior is trained as a forecaster) · H8 ✔ · Eff ✔.
- **Targets:** damping.

### 1.4 DPNets: metric distortion and covariance whitening of the representation
- **Citation:** V. Kostic, P. Novelli, R. Grazzi, K. Lounici, M. Pontil. *Learning invariant representations of time-homogeneous stochastic dynamical systems.* ICLR 2024. arXiv 2307.09912, https://arxiv.org/abs/2307.09912. Verified from the PDF header, which reads "Published as a conference paper at ICLR 2024".
- **Mechanism:**
  - The encoder is trained to maximise a projection score (VAMP-like) with the relaxation R(C) = tr(C² − C − ln C). This term is zero only when the latent covariance C equals I.
  - The authors identify "metric distortion" (a mismatch between latent norms and L²(μ) norms) as a source of spurious spectra.
  - A continuous-time (generator) variant exists.
- **Evidence:** experiments include passive-scalar transport past a cylinder (non-normal dynamics). I did not extract their numeric table, so the magnitudes are UNVERIFIED.
- **Constraints:** L ✔ · CT ✔ · Ma ~ (apply the covariance penalty per Ma batch) · FO ✔ · H8 ✔ · Eff ✔.
- **Relevance:** whitening is what makes a neutral, Euclidean-orthogonal generator equivalent to the measure-preserving one in 1.2. It also prevents a few latent directions from dominating.
- **Targets:** spurious modes and damping.

### 1.5 Uni-Flow: norm-preserving Koopman latent with a diffusion refiner
- **Citation:** X. Xue et al. (incl. P. V. Coveney). *Uni-Flow: a unified autoregressive-diffusion model for complex multiscale flows.* arXiv 2602.15592 (Feb 2026), https://arxiv.org/abs/2602.15592. Verified from the PDF. Preprint.
- **Mechanism:**
  - The Koopman latent uses L_ortho = ‖KᵀK − I‖²_F plus a backward-consistency term ‖ψ(Kᵀφ(u_{n+1})) − u_n‖² (Eqs. 8-11).
  - High-resolution detail is restored by a separate DDIM refiner with about 40 steps.
  - The authors also state that conventional Koopman formulations were "unable to provide stable and sufficiently expressive latent dynamics" for their turbulent-inflow case.
- **Evidence:** compared against neural operators on hemodynamics and turbulent inflow. There is no clean ablation of the orthogonality term.
- **Constraints:**
  - The orthogonality and backward terms: L ✔ · CT ✔ · Ma ✔ · FO ✔ · H8 ✔ · Eff ✔.
  - The diffusion refiner: not end-to-end, adds stochastic compute, and is outside the "efficient" spirit. ✘/~.
- **Targets:** damping (orthogonality); outside-span blur (refiner).

### 1.6 Caution from the course-correction paper
Fathi et al. (ICLR 2024, §2.6 below) report in an appendix that constraining K to be **skew-symmetric failed** ("similar to those with the diagonal matrix … lack of expressivity"). Verified from the PDF, appendix "Low Dimensional Latents…".

Hard neutrality on *all* modes is therefore risky. Soft penalties, or a hybrid with a neutral subspace and a small damped subspace, are safer.

---

## 2. Consistency, re-encoding and closure

### 2.1 Consistent Koopman autoencoder (forward-backward)
- **Citation:** O. Azencot, N. B. Erichson, V. Lin, M. W. Mahoney. *Forecasting sequential data using consistent Koopman autoencoders.* ICML 2020, PMLR 119. arXiv 2003.02236, https://arxiv.org/abs/2003.02236. Verified from the PDF; venue from mlanthology and the ICML virtual page.
- **Mechanism:**
  - A backward operator D is learned alongside the forward C.
  - Losses cover forward and backward multi-step prediction, plus a consistency penalty Σ_k ‖D_{k*}C_{*k} − I_k‖² that pushes D ≈ C⁻¹.
  - The penalty pushes eigenvalues toward the unit circle and away from unstable modes.
- **Evidence:** Re=100 cylinder, κ = 10 latent, 18 seeds, relative prediction error at T = 100:

  | Data | DAE (avg / max) | cKAE (avg / max) |
  |---|---|---|
  | Clean | 0.156 / 0.348 | 0.112 / 0.196 |
  | Noisy | 0.343 / 0.982 | 0.151 / 0.218 |

  - **Negative result:** cKAE does worse when diffusion makes the dynamics non-invertible (sphere-flow experiment).
- **Constraints:** L ✔ · CT ✔ (in our model the backward operator is exactly e^{−τA}, so no extra D is needed) · Ma ✔ · FO ✔ (backward prediction is a training regulariser on forecasting data) · H8 ✔ (uses the same 10-frame window) · Eff ✔.
- **Targets:** damping (see §5 #1). It also reduces variance across seeds.

### 2.2 Koopman Invertible Autoencoder (KIA)
- **Citation:** K. Tayal, A. Renganathan, R. Ghosh, X. Jia, V. Kumar. ICDM 2023 (per the search result and UMN listing). arXiv 2309.10291, https://arxiv.org/abs/2309.10291. Only the abstract was checked; the PDF was downloaded but not read in depth.
- **Mechanism:** forward and backward dynamics with an invertible design.
- **Evidence:** "300% improvement" in long-term prediction on a pendulum, plus climate data. UNVERIFIED beyond the abstract.
- **Constraints:** as 2.1.
- **Targets:** damping and stability.

### 2.3 Temporally-consistent KAE (tcKAE)
- **Citation:** I. Nayak, A. Chakrabarti, M. Kumar, F. L. Teixeira, D. Goswami. arXiv 2403.12335 (v3, July 2025), https://arxiv.org/abs/2403.12335. Verified from the PDF. The venue is UNVERIFIED (preprint).
- **Mechanism:**
  - Adds L_tc, which compares latent predictions of the *same* future time reached from *different* start times (K^k z_n versus K^{k−q} z_{n+q}). The comparison is between predictions, not against the encoded target.
  - Justification: a subspace is Koopman-invariant iff Ψ(x_{n+k}) = K^k Ψ(x_n) for all n and k (Theorem 1).
  - L_tc is switched on after a warm-up of e_s epochs.
- **Evidence:** flow past a cylinder (about 29 samples per period), relative test error (%) averaged over 2500 extrapolation steps (Table 3):

  | Setting | DAE | cKAE | tcKAE |
  |---|---|---|---|
  | N_train = 28, clean | 10.18 | 7.15 | **3.81** |
  | N_train = 28, 30 dB noise | 20.72 | 18.38 | **13.31** |
  | N_train = 40, clean | 1.695 | 1.616 | **0.874** |

  - The gains are largest in the low-data regime.
- **Constraints:** L ✔ · CT ✔ (start points are encoder windows at different frames within the 10-frame clip) · Ma ✔ · FO ✔ · H8 ✔ (no horizon extension) · Eff ~ (several encoder passes per clip).
- **Targets:** latent drift off the encoder manifold, which shows up as outside-span error, plus phase consistency.

### 2.4 Course correcting Koopman representations: periodic re-encoding
- **Citation:** M. Fathi, C. Gehring, J. Pilault, D. Kanaa, P.-L. Bacon, R. Goroshin. ICLR 2024. arXiv 2310.15386, https://arxiv.org/abs/2310.15386. Verified from the PDF and the ICLR proceedings page.
- **Mechanism:**
  - Every k steps, decode the latent and re-encode it. In continuous time, this is done every Δt.
  - Because encoder and decoder are not exact inverses, this projects the drifting latent back onto the encoder's image ("course correction").
  - It can be used at inference only, or also during training.
  - The authors argue that globally linear latent trajectories cannot cross, so they cannot represent multiple attractors.
- **Evidence:**
  - Pendulum, Duffing, Lotka-Volterra and Lorenz (100- and 1000-step MSE), plus D4RL locomotion (300 steps).
  - "Periodic reencoding consistently improves the accuracy". It is robust to periods of 10-100, and works even for models trained without re-encoding (their Fig. 8).
  - Re-encoding at every step is worse.
  - The table numbers were garbled by text extraction, so only the textual claims are verified.
- **Constraints:**
  - ~L: each segment is exactly linear, but the overall map is piecewise.
  - CT ✔ · Ma ✔ · FO ✔ · H8 ✔ (no retraining needed).
  - Eff ~: m/k extra encoder and decoder passes for an m-step rollout, still far cheaper than AR diffusion.
  - It **breaks** "any lead is one matrix exponential".
- **Targets:** outside-span drift and phase. It also needs two decoded frames for our 2-frame encoder.
- **Value for us:** primarily a **diagnostic**. If re-encoding every 8 frames sharply lowers the lead-58 error, the error is latent off-manifold drift, which training regularisers can fix. If it does not, the decoder or representation is the bottleneck.

### 2.5 Mori-Zwanzig latent Koopman closure (MZ-AE)
- **Citation:** P. Gupta, P. J. Schmid, T. Sayadi, D. Sipp, G. Rigas. arXiv 2310.10745 (v3, May 2025), https://arxiv.org/abs/2310.10745. Verified from the PDF. Venue UNVERIFIED.
- **Mechanism:**
  - The Markov term is a linear Koopman operator on autoencoder observables.
  - A non-Markovian memory term Σ_k Ω_k z_{n−k} is added. It is either *linear* (GFD-constrained regression, or a linear RNN) or an LSTM.
- **Evidence:** Re=100 cylinder limit cycle with only r = 2 latent observables:
  - The Koopman-AE is about 3 orders of magnitude better in relative MSE than rank-2 DMD.
  - The linear memory gives roughly another order of magnitude.
  - It is stable over 1000 time units (about 180 shedding cycles), whereas the plain Koopman-AE goes unstable long term.
  - The learned Re(λ) is about 8e-5 (marginally stable), and Im(λ) matches St = 0.179.
  - The memory benefit vanishes as r → 16.
- **Constraints:** L ✔ for the linear memory (it is an augmented linear system) · CT ~ (the kernels are discrete-time) · Ma ✔ · FO ✔ · **H8 ✘** (memory needs q > 2 past latent states; q up to 24 was used) · Eff ✔.
- **Takeaways:** (i) on a limit cycle the learned generator should be almost neutral; (ii) closure matters mostly when the latent is tiny, which ours (256) is not.

### 2.6 Projected Koopman dynamics
- **Citation:** C. R. Constante-Amores, A. J. Fox, C. E. Pérez De Jesús, M. D. Graham. arXiv 2407.16542 (2024), https://arxiv.org/abs/2407.16542. Verified from the PDF. The draft uses the JFM style; the venue is UNVERIFIED.
- **Mechanism:** each step lifts h → Ψ(h), applies K, then projects back to h. This amounts to re-encoding every step and is a nonlinear map.
- **Evidence:** for the MFE turbulence model, all |λ_k| ≈ 0.99, so pure linear evolution decays. With projection, the trajectories track well (qualitative).
- **Constraints:** **L ✘** · AR ✘.
- **Value:** independent confirmation that a pure linear Koopman rollout of a mixing or attracting system tends to decay. This is the same symptom as our γ.

### 2.7 Deep Koopman for the cylinder wake
- **Citation:** J. Morton, F. D. Witherden, A. Jameson, M. J. Kochenderfer. *Deep dynamical modeling and control of unsteady fluid flows.* NeurIPS 2018. arXiv 1805.07472, https://arxiv.org/abs/1805.07472. Verified from the PDF.
- **Mechanism:** the A matrix is obtained *per sequence* by least squares on the encodings of the first 16 steps, then applied recursively. Training uses 32-step sequences.
- **Evidence:** Re=50 cylinder, 128×256 grid. Mean relative L1 error stays below 0.2% over 128 steps (about 8 shedding periods).
- **Constraints:** **H8 ✘** (needs a 16-frame lookback to fit A) · Ma ~.
- **Value:** shows that very low long-horizon error is achievable for a laminar wake, but with a long conditioning window.

### 2.8 Koopman Neural Forecaster (KNF)
- **Citation:** R. Wang, Y. Dong, S. Ö. Arik, R. Yu. ICLR 2023. arXiv 2210.03675, https://arxiv.org/abs/2210.03675. Verified from the PDF header.
- **Mechanism:**
  - A global learned K_g, plus a local K_tl (transformer attention computed over the lookback window), plus a diagonal feedback operator K_tc. K_tc is produced by an MLP from the lookback-window prediction residuals.
  - Measurement functions are *predefined* (polynomial, sinusoidal, etc.).
  - The loss includes L_back, which fits the lookback window.
- **Evidence:** M4-weekly sMAPE ablation:

  | Variant | sMAPE |
  |---|---|
  | base with learned measurements | 14.18 |
  | base with predefined measurements | 9.12 |
  | +RevIN | 8.44 |
  | +K_g | 7.50 |
  | +feedback | 7.25 |

  The data are non-stationary time series, not fluids.
- **Constraints:** L ~ (the operator varies per sample but is linear along the trajectory) · CT ~ · Ma ✔ · FO ✔ · **H8 ✘** for the local and feedback operators (they need a lookback of several steps; with 2 frames there is no residual to feed back) · Eff ✔.
- **Transferable piece:** a per-trajectory correction δA(z₀, Ma) (§5, "other ideas").

### 2.9 Koopa
- **Citation:** Y. Liu, C. Li, J. Wang, M. Long. NeurIPS 2023. arXiv 2305.18803, https://arxiv.org/abs/2305.18803. Verified from the PDF footer ("37th Conference on Neural Information Processing Systems (NeurIPS 2023)").
- **Mechanism:**
  - A Fourier filter splits time-invariant components (dominant frequencies) from time-variant ones.
  - The time-invariant Koopman predictor uses a globally learned K. The time-variant one uses eDMD fitted on lookback snapshots.
  - Blocks are stacked residually, and the reconstruction loss is dropped in favour of a pure forecasting loss.
- **Evidence:** "competitive" accuracy with 77.3% less training time and 76.0% less memory. It is a time-series benchmark, not fluids.
- **Constraints:** L ~ · H8 ✘ (the local DMD needs a lookback).
- **Transferable insight:** dropping the reconstruction loss, i.e. a pure forecasting objective, which we already do.

---

## 3. Limit cycles, phase and harmonics

### 3.1 Koopman spectrum of limit cycles: the lattice structure (theory)
- **Citation:** A. Mauroy, I. Mezić. *Global computation of phase-amplitude reduction for limit-cycle dynamics.* Chaos 28 (2018), DOI 10.1063/1.5030175. arXiv 1803.07379, https://arxiv.org/abs/1803.07379. Only the abstract and DOI were checked.
- **Also:** dymad documentation, "Spectral Analysis Part 2", https://dymad.readthedocs.io/en/latest/examples/sa_lco/part2.html. Opened and quoted: "±nω₀i, n=1,2,… are all Koopman eigenvalues … a lattice of eigenvalues composing the above two types."
- **Related:** Mauroy, Mezić, Moehlis, Physica D 261 (2013), on isostables for fixed points (verified via search listing).
- **Content:**
  - The isochron (phase) eigenfunction has eigenvalue iω₀ and the isostable (amplitude) eigenfunctions have Floquet exponents λ < 0.
  - Products of eigenfunctions give the lattice i·n·ω₀ + m·λ.
- **Implication:** on the attractor, the harmonics are *exactly* integer multiples of ω₀ and neutral. This is a hard structural prior our free (ω_k, γ_k) blocks do not encode.
- **Not found:** I found **no published KAE that hard-codes integer-multiple harmonic frequencies**. The closest are 3.2 (a single phase pair) and 3.3 (a few base frequencies plus a nonlinear decoder). The intervention in §5 #2 is therefore my extrapolation, not an established method.

### 3.2 Phase autoencoder
- **Citation:** K. Yawata, K. Fukami, K. Taira, H. Nakao. *Phase autoencoder for limit-cycle oscillators.* arXiv 2403.06992 (2024), https://arxiv.org/abs/2403.06992. Verified from the PDF. The journal is UNVERIFIED (formatted as AIP/Chaos).
- **Mechanism:**
  - A 3-D latent. (Y1, Y2) are normalised onto the unit circle *in the encoder* (division by R), so the decoder always receives a unit-norm phase.
  - They evolve as an exact rotation at a learned ω for all τ.
  - Y3 decays as e^{λτ}, capturing the deviation from the cycle.
  - Losses: reconstruction, multi-step phase and deviation losses with weights α_k, and an auxiliary loss.
  - The paper shows (Y1 + iY2) is a Koopman eigenfunction with eigenvalue e^{iωτ}.
- **Evidence:** recovers the asymptotic phase and phase-sensitivity function on low-dimensional oscillators. No forecasting-error benchmarks; no fluid fields.
- **Constraints:** L ✔ · CT ✔ (defined for all τ) · Ma ✔ (ω(Ma)) · FO ✔ · H8 ✔ · Eff ✔.
- **Targets:** phase drift (the phase is exactly neutral) and amplitude decay. The decoder always sees an on-cycle input, which addresses outside-span blur from shrinking latents.

### 3.3 From Fourier to Koopman
- **Citation:** H. Lange, S. L. Brunton, J. N. Kutz. *From Fourier to Koopman: Spectral methods for long-term time series prediction.* JMLR 22(41):1-38, 2021. https://jmlr.org/papers/v22/20-406.html and arXiv 2004.00574. Verified from the PDF; venue confirmed on JMLR.
- **Mechanism:**
  - The model is x(t) ≈ f(Ω(ω⃗ t)): a *small set* of frequencies drives a nonlinear decoder f, which generates harmonics implicitly.
  - Fitting ω is highly non-convex, and the loss surface sharpens with horizon. The authors compute the global optimum per frequency from the FFT of the residual, then refine by gradient descent.
  - A phase-drift variant adds a penalty |φ̂_t − φ̂_{t−1}| to model process noise.
- **Evidence:**
  - On cylinder (Re about 100), cavity, Kolmogorov and video data in PCA space, with a temporal split, the Fourier algorithm beats DMD and FB-DMD on 3 of 4 tasks (their Table 3, cumulative error per test quarter).
  - The Koopman (nonlinear-decoder) variant beats the Fourier variant on synthetic tasks.
  - LSTMs are frequency-biased over long horizons.
- **Constraints:** L ✔ (latent = rotations) · CT ✔ · Ma ✔ · FO ✔ · H8 ~ (they fit on long series; the FFT trick needs long records, but only for initialising ω(Ma), not at test time) · Eff ✔.
- **Targets:** phase drift (frequency accuracy) and harmonics, which the decoder generates from a few base phases and so keeps phase-locked.

### 3.4 Linearly-recurrent autoencoder networks (LRAN)
- **Citation:** S. E. Otto, C. W. Rowley. SIAM J. Appl. Dyn. Syst. 18(1), 2019 (venue from memory; UNVERIFIED in this session). arXiv 1712.01378, https://arxiv.org/abs/1712.01378. Verified from the PDF.
- **Mechanism:**
  - An autoencoder with linear recurrence, trained with a decaying-weight multi-step loss (δ^τ) and relative errors.
  - Known eigenvalues can be **fixed** in K: a fixed 2×2 block [[σ, ω], [−ω, σ]] with the remaining K̃ trainable.
- **Evidence:** cylinder wake from the unstable equilibrium to the limit cycle:
  - KDMD's second eigenvalue is almost exactly the harmonic 2λ₁, while the LRAN's second eigenvalue sits close to λ₁.
  - With its nonlinear decoder, the LRAN's mean relative error is about an order of magnitude smaller than the nonlinearly reconstructed KDMD ROM.
- **Lesson:** with a nonlinear decoder, harmonics do not need dedicated latent modes. The decoder can synthesise them from the fundamental, so phase-locking is automatic.
- **Constraints:** all ✔ (H8 ✔: T = 20 steps were used in training).
- **Targets:** harmonics and phase.

### 3.5 LKIS-DMD
- **Citation:** N. Takeishi, Y. Kawahara, T. Yairi. *Learning Koopman invariant subspaces for dynamic mode decomposition.* NeurIPS 2017. arXiv 1710.04340. Verified from the PDF (abstract and method).
- **Mechanism:** learns delay-embedded observables that minimise the residual of a least-squares linear fit.
- **Evidence:** on limit cycles, the eigenvalues lie near the unit circle.
- **Constraints:** H8 ~ (needs delay embedding).
- **Novelty for us:** low; included because the brief asked for it.

---

## 4. Decoder-side and other items (lower relevance or incompatible)

- **HAVOK.**
  - Citation: S. L. Brunton, B. W. Brunton, J. L. Proctor, E. Kaiser, J. N. Kutz. *Chaos as an intermittently forced linear system.* Nature Communications 8, 2017. arXiv 1608.05306. Verified from the PDF.
  - Mechanism: a linear model in the leading delay coordinates, forced by the last delay coordinate v_r.
  - Verdict: **forecasting ✘**, because v_r is unknown in the future. **H8 ✘** (long Hankel windows).
  - Not usable except as a diagnostic of intermittency.
- **Deep adversarial Koopman.**
  - Citation: K. Balakrishnan, D. Upadhyay. arXiv 2006.05547 (2020). Verified from the PDF.
  - Mechanism: a WGAN-GP discriminator plus image-gradient losses on a KAE.
  - Evidence: on Kuramoto-Sivashinsky and Gray-Scott, the full model ("adv + grad") does best at later times (figures only; no tables). Either loss alone can be *worse* than vanilla late in the rollout.
  - Constraints: L/CT/Ma/FO/H8/Eff ✔.
  - Targets: outside-span blur.
  - Risk: sharper but misplaced structure raises MSE (double penalty). The evidence is weak.
- **Koopman neural operator (KNO).**
  - Citation: W. Xiong et al. arXiv 2301.10022. Verified at the abstract and method level only.
  - Mechanism: a linear operator on truncated low Fourier modes, plus a separate convolutional "high-frequency information complement" path.
  - Relation to our work: this is structurally the "detail latent" we tried, which collapsed.
  - Applied autoregressively. Quantitative claims UNVERIFIED.
- **Structured Koopman autoencoders (static and dynamic factors).**
  - Citation: N. Berman, I. Naiman, O. Azencot. ICLR 2023 (oral). arXiv 2303.17264. Venue verified via iclr.cc.
  - Mechanism: a spectral loss structures K so that subsets of eigenvalues carry separate factors.
  - The claim that static factors use eigenvalue 1 is **UNVERIFIED**.
  - Relevance: a time-invariant "content" code would be a legitimate Koopman observable (λ = 0 in continuous time) that could lower the lead-1 error. The risk is that it freezes phase-dependent detail, the same window-exploitation failure seen with the detail latent.
- **KoVAE.**
  - Citation: I. Naiman, N. B. Erichson, P. Ren, O. Azencot, M. W. Mahoney. ICLR 2024. arXiv 2310.02619. Verified from the PDF header.
  - Mechanism: a linear Koopman prior in a VAE, with optional eigenvalue constraints.
  - Purpose: generative modelling, not deterministic long-horizon MSE.
  - Low relevance.
- **Koopman β-VAE for LES turbulence.**
  - Citation: R. Halder, B. Eiximeno, O. Lehmkuhl. arXiv 2507.18754. Verified from the abstract.
  - Mechanism: a Koopman-style loss denoises the latents, but the latent dynamics are forecast by an **LSTM**.
  - Verdict: L ✘.
- **Koopman ensembles.**
  - Citation: A. Frion et al. EUSIPCO 2024. arXiv 2403.06757. Search-level only.
  - Purpose: uncertainty quantification. An ensemble mean would reduce MSE but costs K× compute.
- **Sparsity-promoting Koopman-invariant subspaces.**
  - Citation: S. Pan, N. Arnold-Medabalimi, K. Duraisamy. J. Fluid Mech. 917, A18 (2021). arXiv 2002.10637. Verified from the PDF (method).
  - Mechanism: prunes modes by their *a posteriori* maximal normalised deviation from linear evolution, then selects a sparse set. Tested on cylinder wakes at Re = 70, 100, 130 with transients.
  - Relevance: the same idea as ResDMD below.
- **ResDMD.**
  - Citation: M. J. Colbrook, A. Townsend. arXiv 2111.14889. Verified from the PDF. The CPAM venue is UNVERIFIED (search did not confirm).
  - Mechanism: residuals ‖ψ∘F − λψ‖ / ‖ψ‖, computed with an extra Gram matrix for K*K, flag spurious eigenpairs (spectral pollution) without assuming measure preservation.
  - Use for us: a **post-hoc per-mode residual** r_k = E‖E(x_{t+τ})_k − (e^{τA}E(x_t))_k‖ / ‖E(x_t)_k‖ on validation data. This identifies which oscillators are spurious and should stay damped or be pruned before neutrality is enforced.

Related prior work by the user (not external evidence):
- arXiv 2602.02832, CT-KAE for fluids.
- arXiv 2603.05560, CT-KAE for ocean QG (ICLR 2026 workshop).

---

## 5. Ranked shortlist for our model

Honest framing: no paper reports a fix for exactly our failure mode, a high-dimensional CT-KAE with an 8-frame window on transonic wakes. The rankings rest on (i) the γ arithmetic in §0, (ii) consistent evidence that unit-circle or neutral spectra cut long-horizon KAE error on cylinder flows (Miller et al.: 1.25 → 0.58-0.82; Azencot et al.: 0.156 → 0.112 clean and 0.343 → 0.151 noisy; MZ-AE: Re λ ≈ 0), and (iii) limit-cycle Koopman theory. Run the §0 zero-cost checks first; they decide between #1 and #2.

### #1. Remove the window-invisible damping: backward-prediction loss plus γ penalty, with residual-gated neutrality
**Sources:** cKAE (2.1), Uni-Flow backward term (1.5), eigenloss and eigeninit (1.1), mpEDMD (1.2), ResDMD and Pan (§4).

**Sketch:**
- (a) **Backward loss inside the same 10-frame clip.** Encode the last two frames (x_8, x_9). Propagate with e^{−τA(Ma)} for τ = 1..8 frames. Decode, and apply the MSE against x_7 … x_0 with weight w_b of about 0.25-0.5.
  - Because e^{+γτ} amplifies backwards, any γ the data do not support gets penalised. The penalty scales the same way at lead 8 backward as damping costs at lead 58 forward.
  - No new data and no horizon extension are needed. It is a forecasting-only regulariser.
- (b) **γ penalty.** Add λ_γ·mean_k(γ_k(Ma)²), with λ_γ set so that the penalty equals about 1-5% of the loss. Alternatively, initialise the softplus so that γ starts near 0 (eigeninit with radius near 1).
- (c) **Gating.** Compute per-mode ResDMD-style residuals on validation data. Allow neutrality only for modes with low residual. A small set of high-residual or "transient" modes may keep free γ (a hybrid neutral-plus-damped latent, as in the phase autoencoder).

**Expected effect:**
- This targets the growth slope, not the lead-1 floor.
- If H1 holds (test it by zeroing or scaling γ at inference), I would expect a meaningful cut in the POD 3-10 and outside-span parts of the excess at leads 30-58. A plausible range is 20-40% of the 60-step gap to the U-Net, i.e. 3.49 → about 2.8-3.1.
- This is a guess calibrated on the 1.4-2× long-horizon reductions in the cylinder papers, which used much simpler data. 240-step rollouts should benefit more than 60-step ones.

**Main risk:** damping may be *protecting* the decoder from spurious, poorly-encoded modes. Undamped spurious modes would persist and add off-manifold content. Gating (c) and the inference-time γ-scaling test exist to detect this. Fathi et al. also found hard skew-symmetric K failed, so keep the constraint soft.

### #2. Phase-locked harmonic generator: the lattice prior
**Sources:** limit-cycle Koopman lattice (3.1), phase autoencoder (3.2), Fourier-to-Koopman (3.3), LRAN (3.4).

**Sketch:**
- Replace the free ω_k with ω_k(Ma) = n_k·ω₁(Ma) for a "harmonic bank" of oscillator pairs. Use n_k ∈ {1, …, N_h} with several pairs per harmonic for spatial richness, and set γ = 0 on the bank.
- ω₁(Ma) comes from a small MLP, or is anchored to the shedding frequency estimated from the training data's FFT per Ma (Lange-style initialisation).
- Keep a small free block (for example 32 of 256 dimensions) with free (ω, γ) for incommensurate content: shock oscillation or a second base frequency ω₂(Ma).
- If ω₂ exists, extend the lattice to n·ω₁ + m·ω₂.
- Optionally normalise the fundamental pair to the unit circle in the encoder, as in the phase autoencoder, and let the decoder FiLM on Ma set the amplitude.

**Expected effect:**
- Harmonic and fundamental phases stay locked, so a frequency error produces a *coherent* phase shift of the whole pattern rather than shape distortion.
- This should reduce the outside-span share (49-53%) and the POD 3-10 share of the excess, and remove amplitude loss in the bank.
- The shedding-pair phase error (26-37° vs 14° for the U-Net) improves only if part of it comes from fundamental-harmonic interaction in the decoder. Diagnostic (c) in §0 tells you how much.
- **Zero-training check first:** snap the learned ω_k to the nearest n·ω₁ on the existing checkpoint (§0 b). If the clustering is poor, the model is not using a harmonic representation and the gain is uncertain.
- With strong evidence I would expect this to rival #1. Without it, treat this as the highest-upside but **unvalidated** option.

**Main risk:**
- At high Ma the Tra wake may not be a clean limit cycle (shock-vortex interaction, possibly quasi-periodic or chaotic content). A rigid lattice would then underfit, which the free block mitigates.
- Zero-shot Ma also relies on ω₁(Ma) interpolating well, but that is easier with one function than with 128.
- No published KAE uses this exact construction.

### #3. Multi-start temporal consistency (tcKAE-style) inside the window
**Source:** tcKAE (2.3).

**Sketch:**
- In each 10-frame clip, encode windows starting at frames 0, 2 and 4 (each pair of frames gives z^{(s)}).
- Propagate every z^{(s)} to all later frames in the clip.
- Add L_tc = Σ_{s<s'} Σ_t ‖e^{(t−s)A}z^{(s)} − e^{(t−s')A}z^{(s')}‖², stopping gradients on neither side, and switch it on after warm-up.
- This complements the existing latent-consistency loss, which compares against the encoder of the target.

**Expected effect:**
- It makes the encoder image closer to an invariant subspace, so propagated latents stay on-manifold. That reduces decoder extrapolation, i.e. outside-span error.
- The tcKAE cylinder gains (2×) were in low-data regimes. Tra has few Mach trajectories, so moderate gains (about 5-15% on the long-lead error) are plausible.
- It is cheap and orthogonal to #1 and #2.

**Main risk:** it overlaps with the existing consistency loss, so the marginal gain may be small. It also costs about 2-3× encoder passes per clip.

### #4. Latent whitening, so that "neutral" means energy-conserving
**Sources:** DPNets (1.4), mpEDMD (1.2).

**Sketch:**
- Add γ_w·R(C), with R(C) = tr(C² − C − ln C), on the batch covariance of z, per Ma batch or pooled.
- Alternatively, use a cheaper version: ‖C − I‖²_F on standardised latents.
- With C ≈ I, a neutral block-diagonal A preserves the data-measure norm (mpEDMD's condition). Latent "energy" is then meaningful, and the amplitude loss can act in a whitened space.

**Expected effect:** small on its own. It mostly makes #1 and #2 well-posed: it prevents the model from hiding decay in badly scaled directions and stops a few directions from dominating.

**Main risk:** whitening can conflict with the decoder's preferred latent geometry and slow convergence. Use a small weight.

### #5. Periodic re-encoding as a diagnostic, and optionally an inference mode
**Source:** Fathi et al. (2.4).

**Sketch:**
- At test time, every k ∈ {8, 16, 30} frames, decode frames (t − 1, t) from the latent (closed form, so cheap), re-encode with the Mach FiLM, and continue.
- Compare lead-30 and lead-58 errors with and without re-encoding.

**Expected effect:**
- If the error drops a lot, the growth is latent off-manifold drift, which #1, #3 and #4 should fix in a constraint-compatible way.
- If not, the bottleneck is the decoder or representation. In that case, decoder-side work (generative refiner, gradient or adversarial losses) or a larger latent is where to look, with weak evidence for MSE gains.

**Main risk:** as a deployed method, it violates "one matrix exponential at any lead" and makes the predictor piecewise-linear. Lead-1 reconstruction errors may also compound through repeated re-encoding.

### Other ideas (not ranked; evidence weak or constraint conflicts)
- **Per-trajectory generator correction** A(Ma) + δA(z₀, Ma), adapted from KNF's local and feedback operators. It stays exactly linear along a trajectory and in closed form. On a stationary attractor it adds little beyond Ma conditioning, though it may help phase speed if Tra amplitude varies within a Mach number.
- **Static content code** (λ = 0 latent, from Berman et al.). It could cut the 0.26 lead-1 error but risks the same collapse as the detail latent.
- **Decaying-weight or strided supervision** (Otto-Rowley's δ^τ weighting, or supervising the same 8 frames at strided leads). This conflicts with the "similar supervised horizon" constraint, so check with the user.

**On the lead-1 gap (0.26 vs 0.02-0.03):** none of the Koopman literature I found addresses it directly. It is an encoder/decoder bottleneck issue (Otto-Rowley and Fathi both stress that decoder choice dominates), not a dynamics issue. With 3.49 − 1.53 ≈ 1.96 of average excess and a floor of about 0.25, the growth term is where most of the recoverable error lies.
