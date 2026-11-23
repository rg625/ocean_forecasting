# Remedies for long-horizon degradation in neural PDE / fluid surrogates (2021-2026), with a compatibility check against the conditional CT-KAE

Search date: 2026-10-22. Each paper below was downloaded from arXiv, or opened as HTML, PDF or journal page, and the quoted numbers were read from the paper text. Table values pulled from PDF text were cross-checked against the HTML version wherever the PDF extraction was garbled (ACDM Table 1, DySLIM Table 1). Anything not checked against a primary source is marked **UNVERIFIED**.

Abbreviations for the hard constraints: **LIN** = exactly linear latent dynamics; **CT** = continuous time; **MA** = Mach-conditioned / zero-shot Ma; **FO** = forecasting-only objective (no DA); **E2E** = end-to-end; **HOR** = supervised horizon about 2 to 8 frames; **EFF** = efficient latent / closed-form inference.

Diagnosed error components: **BLUR** = excess error outside the span of the true trajectory's POD modes (49% at lead 30); **HARM** = POD modes 3-10 (32%); **PHASE** = shedding-pair phase drift (8% at lead 30; 26-37 deg at lead 58); **DAMP** = amplitude or energy loss (includes the detail latent collapsing after the 8-step window).

---

## 0. One framing point that changes how most of this literature transfers

Most remedies in this literature (pushforward, input-noise injection, unrolled training, Thermalizer, the FouRKS spectrum correction, latent-generative "uncertainty knobs") target **input distribution shift under autoregressive feedback**. The model is fed its own outputs, drifts off the training distribution and diverges. Brandstetter et al. state this explicitly ("distribution shift problem ... domain adaptation problem"), and so do Stachenfeld et al. ("the model is fed its most recent prediction back in as input").

**The CT-KAE does not have this failure mode.** The encoder only ever sees the two true conditioning frames. The latent is propagated by exp(A(Ma)τ) with gamma >= 0, so it cannot blow up, and nothing is fed back. So:
- Methods whose only mechanism is "train on your own perturbed outputs" (pushforward, input-noise injection, diverted-chain or unrolled training with gradients cut) have **nothing to act on**. The one exception is the decoder, which sees propagated latents that may drift off the encoder manifold. That is the only place a "distribution shift" exists in our model.
- Our degradation is **model bias**: frequency and phase error, harmonics that are not phase-locked, and an MSE-optimal blur once phase is uncertain. It is not instability. The literature that matches our problem is therefore (a) phase and frequency identification, (b) spectral structure of the Koopman generator, and (c) the reconstruction/dynamics trade-off in latent models. It is not the stability literature.
- Corollary: under phase uncertainty, the MSE-optimal prediction **is** the blurred conditional mean. Anything that only "sharpens" without reducing phase error will tend to **increase** MSE (double penalty; see Subich 2025 and Oommen 2025 below). Reducing PHASE/HARM error is probably the main way to reduce BLUR under an MSE metric.

---

## 1. Same-benchmark evidence: the ACDM paper (Kohl, Chen, Thuerey)

**Citation:** Kohl, Chen, Thuerey, "Benchmarking Autoregressive Conditional Diffusion Models for Turbulent Flow Simulation", arXiv:2309.01745v3 (Dec 2024). Opened at https://arxiv.org/pdf/2309.01745 and https://arxiv.org/html/2309.01745v3.

Table 1 values (MSE x1e-3, Tra_ext / Tra_int), read from the HTML version:

| Model | Tra_ext | Tra_int |
|---|---|---|
| U-Net (plain one-step) | 3.1 ± 2.1 | 2.3 ± 2.0 |
| U-Net_ut (unrolled, m = 8) | 1.6 ± 0.7 | 1.5 ± 1.5 |
| U-Net_tn (input noise, n ≈ 1e-2) | 1.4 ± 0.8 | 1.8 ± 1.1 |
| ResNet_dil | 1.7 ± 1.0 | 1.7 ± 1.4 |
| **Refiner (PDE-Refiner re-impl.)** | **5.4 ± 2.1** | **7.1 ± 2.1** |
| TF_Enc (latent transformer) | 3.3 ± 1.2 | 6.2 ± 4.2 |
| TF_MGN / TF_VAE (latent) | 3.9 / 4.1 | 6.3 / 7.2 |
| ACDM | 2.3 ± 1.4 | 2.7 ± 2.1 |

Takeaways for us:
- On Tra, input-noise training (σ ≈ 1e-2) is worth as much as 8-step unrolling for an **autoregressive** U-Net: 3.1 → 1.4 extrap, 2.3 → 1.8 interp. The strong U-Net we compare against is a stabilised U-Net.
- **PDE-Refiner is substantially worse than a plain U-Net on this exact benchmark** (5.4 / 7.1). This is the strongest available negative evidence against spectral-refinement remedies for Tra.
- Every **latent** baseline in the paper is worse than our CT-KAE on Tra: TF_Enc scores 3.3 / 6.2 against our 1.98 / 3.49. Our gap is specifically to the stabilised pixel-space U-Net and ResNet_dil.

---

## 2. Method-by-method review

### 2.1 Pushforward trick and temporal bundling (MP-PDE)
- **Citation:** Brandstetter, Worrall, Welling, "Message Passing Neural PDE Solvers", ICLR 2022 (spotlight). https://arxiv.org/abs/2202.03376 (PDF read).
- **Mechanism:** *Pushforward*: unroll 2 steps but backpropagate only through the last one ("we found it important not to backpropagate through the first unroll step"). This is an adversarial-style stability loss that makes the input perturbations look like the model's own errors. *Temporal bundling*: predict K future steps per call, which cuts the number of calls (and distribution shifts) by a factor of K.
- **Evidence:** 1D shock-formation E1 (nx = 100), accumulated error: FNO-RNN 11.93 → FNO with bundling + pushforward 0.54. Survival-time ablation (their Fig. 5b): pushforward gives "far higher survival times" than no pushforward or Gaussian noise (σ = 0.01). Counter-evidence: PDE-Refiner (Lippe et al., Table 5) finds pushforward does **not** help on KS (75.4 s, the same as the baseline).
- **Compatibility:** LIN ok, CT ok, MA ok, FO ok, E2E ok, HOR ok. **Needs autoregressive stepping: yes. Mechanism is absent in our model.** Bundling is already maximal for us, because closed-form propagation predicts all leads at once. The nearest analogue is re-encoding a decoded prediction and propagating again (an "encode → decode → re-encode" cycle). That would break closed-form any-lead inference and so violates EFF.
- **Targets:** none of our components directly. **Verdict: not useful.**

### 2.2 Training-noise injection
- **Citation:** Stachenfeld et al., "Learned Coarse Models for Efficient Turbulence Simulation", ICLR 2022. https://arxiv.org/abs/2112.15275 (PDF read). Sanchez-Gonzalez et al. 2020 is cited there and was not re-opened.
- **Mechanism:** Gaussian noise (σ = 0.01) is added to the inputs during training, so the model learns to map slightly-off states back onto the data manifold.
- **Evidence:** Mixed. Noise worsens one-step error but stabilises long rollouts. Per their Fig. 4: "In KS-1D the noise harmed performance; in Incomp-2D the noise particularly benefits the FNO rollouts; in CompDecay-3D the noise mainly stabilized rollouts." On Tra (ACDM paper), U-Net_tn takes MSE from 3.1 to 1.4 (extrap).
- **Compatibility (input-space noise):** compatible with all constraints, but its mechanism (feedback distribution shift) is absent. On our encoder it acts only as a generic smoothness regulariser.
- **Compatibility (latent-space noise, see 2.11):** also compatible with all constraints. This is the version that touches our one real distribution shift (decoder fed propagated latents).
- **Targets:** BLUR only indirectly. **Verdict: input noise is low value; latent noise is moderate (see 2.11).**

### 2.3 PDE-Refiner
- **Citation:** Lippe, Veeling, Perdikaris, Turner, Brandstetter, "PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers", NeurIPS 2023 (the arXiv v2 page lists no venue; NeurIPS 2023 is from memory and was not re-checked). https://arxiv.org/abs/2308.05732 (PDF read).
- **Mechanism:** After the MSE prediction, K denoising refinement steps with exponentially decreasing noise σ_k. The model must predict the noise, so it is forced to model low-amplitude (high-wavenumber) components that MSE ignores.
- **Evidence:** KS, corr > 0.9 time: MSE U-Net 75.4 s; 4x parameters ≈ 79.7; 5-model ensemble 79.7; pushforward 75.4; Sobolev k = 0 / 1 / 2: 71.4 / 66.9 / 8.7; MNO 66.6; PDE-Refiner with 3 steps 97.5 ± 0.5. Kolmogorov 2D, corr > 0.8 time: MSE U-Net 9.663 vs PDE-Refiner 10.659 (+10%). **On Tra (ACDM paper) Refiner is worse: 5.4 / 7.1 vs U-Net 3.1 / 2.3.** A follow-up, PDE-SpectralRefiner (Luo, arXiv:2506.10711, single author, read), reports small gains for a U-Net on NS (unrolled MSE 0.248 → 0.223) but **worse** results for FNO (0.311 → 0.388). That is weak evidence.
- **Compatibility:** LIN would hold for the latent if refinement were applied only as a decoder-side post-processor, but it adds 3-4 extra network calls per output frame (hurts EFF) and is stochastic. Nothing would feed back, so its claimed stability benefit does not apply to us. What remains is spectral sharpening, which under MSE means double-penalty risk.
- **Targets:** BLUR (high-frequency content) in principle. **Verdict: not recommended. There is direct negative evidence on Tra.**

### 2.4 Spectral, Fourier and Sobolev losses against spectral bias
- **Citation A:** Chattopadhyay, Sun, Hassanzadeh, "Challenges of learning multi-scale dynamics with AI weather models: Implications for stability and one solution", arXiv:2304.07029v2 (Dec 2024; earlier title "Long-term instabilities of deep learning-based digital twins of the climate system: The cause and a solution"). PDF read.
  - **Mechanism:** FouRKS has three parts: an RK4 integrator layer, a spectral regulariser penalising high-wavenumber (k ≥ k_T) zonal Fourier-amplitude mismatch (λ = 0.8, k_T = 30 for QG), and a self-supervised spectrum correction during inference that re-optimises the last two layers every 2 days.
  - **Evidence:** "the spectral regularizer [alone] pushes the RMSE error to grow faster than the regular U-NET". The combination of RK4 and the regulariser is stable to only about 30 days and blows up by about 60. Only the full FouRKS, with **test-time** correction, stays bounded (about 200 days; 300,000-day QG runs; 10 years on ERA5).
  - **Compatibility:** a loss on the decoded closed-form frames is compatible with all constraints. The test-time correction violates FO in spirit (it optimises during inference) and hurts EFF.
- **Citation B:** Li, Liu-Schiaffini, Kovachki et al., "Learning Dissipative Dynamics in Chaotic Systems" (MNO), NeurIPS 2022. https://arxiv.org/abs/2106.06898 (read). Proposes Sobolev losses plus dissipativity regularisation. In PDE-Refiner's comparison, both Sobolev and MNO underperform plain MSE on KS (see 2.3).
- **Targets:** BLUR and HARM in principle. **Verdict: weak to negative for MSE.** Two independent studies find spectral or Sobolev losses alone shorten accurate rollout time or raise RMSE. Dissipativity is already guaranteed by our gamma >= 0.

### 2.5 Invariant-measure and statistics regularisation
**(a) DySLIM**
- **Citation:** Schiff, Wan, Parker, Hoyer, Kuleshov, Sha, Zepeda-Núñez, "DySLIM: Dynamics Stable Learning by Invariant Measure for Chaotic Systems", ICML 2024 (PMLR 235). https://arxiv.org/abs/2402.04467 (PDF and HTML read).
- **Mechanism:** Adds an MMD (sum of multiscale inverse-quadratic kernels on raw states) between the true and model-evolved sample distributions. The *conditional* term λ2·MMD((S^ℓ)#μ, (S_θ^ℓ)#μ) compares the batch of true states at lag ℓ with the batch of model states at lag ℓ. **ℓ equals the training unroll length** (ℓ = 1 for the 1-step objective; up to 5 for curriculum and 10 for pushforward on Kolmogorov). The unconditional λ1 term "becomes uninformative as the dimension ... increases", so the Kolmogorov runs use λ1 = 0.
- **Evidence (Kolmogorov flow, Table 1, x1e-2, Base → DySLIM):**
  - 1-step: MELR 2.77 → 1.84, covRMSE 7.93 → 7.30, Wass1 16.2 → 5.55, TCM 5.39 → 2.45, but MELRw 0.44 → 0.85 (worse).
  - Curriculum: MELR 5.35 → 1.64, Wass1 9.66 → 4.76.
  - Pushforward: TCM 3.68 → 0.72.
  - Short-term cosine similarity is also reported as improved (their Fig. 3). The baselines "become highly dissipative and quickly veer towards the mean".
- **Compatibility:** LIN, CT, MA, FO, E2E ok. **HOR ok**: the conditional term only needs true samples at the same leads already supervised (≤ 8). EFF ok, because the closed form gives all 8 leads at once.
- **Translation:** per Mach batch, MMD between decoded predictions at leads 1-8 and the true frames at the same leads, as a set-level loss. Optionally use downsampled or feature-space kernels.
- **Targets:** DAMP (regression to the mean) and BLUR (distributional sharpness). It does not fix PHASE.

**(b) Jiang, Lu, Orlova, Willett**
- **Citation:** "Training neural operators to preserve invariant measures of chaotic attractors", NeurIPS 2023. https://arxiv.org/abs/2306.01187 (PDF read).
- **Mechanism:** A Sinkhorn optimal-transport loss between distributions of expert summary statistics, or a contrastively learned feature loss, over windows of predictions, added to relative RMSE. **The roll-out step h was grid-searched over {1..5} and set to 1**, so the statistics are matched on one-step predictions collected over a window of length K+1.
- **Evidence:** only Lorenz-96 and KS. Large improvements in histogram and energy-spectrum errors with comparable 1-step RMSE. No fluid benchmark.
- **Compatibility:** compatible with all constraints including HOR (h = 1). The paper's limitations section says it assumes trajectories are on the attractor and "cannot handle explicit time dependence ... or control parameters". Mach as a control parameter would need per-Ma statistics.
- **Targets:** DAMP, BLUR.

**(c) Thermalizer**
- **Citation:** Pedersen, Zanna, Bruna, "Thermalizer: Stable autoregressive neural emulation of spatiotemporal chaos", ICML 2025. https://arxiv.org/abs/2503.18731 (PDF read).
- **Mechanism:** A diffusion model of the invariant measure is used at inference time to denoise drifting emulator states ("thermalization").
- **Evidence:** extends stable horizons "by two orders of magnitude" (Kolmogorov, QG; up to 1e5 steps). But MSE against the reference saturates at about unity: it preserves statistics, not pointwise skill.
- **Compatibility:** inference-time non-linear correction of decoded states. It does not change the linear latent, but it is a post-processor, hurts EFF, and does not help MSE.
- **Targets:** DAMP and statistics only. **Verdict: not for our metric.**

**Grey-area variant for us (flagged):** because our rollouts are closed form, we could cheaply decode at leads τ ∈ [8, 240] and match **per-Ma climatological statistics** (mean field, variance field, temporal PSD at probes) taken from the training trajectories. This is unsupervised with respect to pointwise targets beyond 8 frames, but it does use information from long training trajectories. **Flag: grey area w.r.t. HOR.**

### 2.6 Double-penalty and phase-aware losses
- **Subich et al., "Fixing the Double Penalty in Data-Driven Weather Forecasting Through a Modified Spherical Harmonic Loss Function", ICML 2025.** https://arxiv.org/abs/2501.19374 (read). Already covered by the earlier review. AMSE separates the per-wavenumber PSD-amplitude error from decorrelation (coherence). GraphCast fine-tuning moved effective resolution from 1,250 km to 160 km. The appendix shows ensemble RMSE skill with "degradation at short lead times". Under an MSE metric this sharpens while raising pointwise error.
- **FreDF** (Wang et al., "FreDF: Learning to Forecast in the Frequency Domain", ICLR 2025; https://arxiv.org/abs/2402.02399, read): L = α·‖FFT_t(Ŷ) − FFT_t(Y)‖₁ + (1−α)·L_time, where the FFT runs along the **forecast-time axis**. The motivation is that multi-step direct-forecast labels are autocorrelated and frequency components are close to independent. Evidence is on time-series benchmarks (ETT and similar) only, with no fluid data.
  - Compatible with all constraints (the closed form outputs the 8-frame sequence directly).
  - Targets PHASE and HARM weakly, because an 8-frame temporal FFT has very coarse frequency resolution (about 1 bin per shedding period).
- **DILATE** (Le Guen & Thome 2019, arXiv:1909.09020, abstract read): soft-DTW shape term plus temporal-distortion term. It predates 2021 and is time-series only. Listed for completeness.
- **Verdict:** phase-aware losses in the literature target *timing of sudden changes* in 1D series or *spectral amplitude* in weather. None was found that targets slow frequency drift of a periodic flow. See 2.9 and 2.10 for what actually addresses that.

### 2.7 Generative and diffusion approaches, and refinement post-processors
- **ACDM** (see section 1): Tra 2.3 / 2.7. This is better than plain U-Net on extrap but worse than U-Net_ut/tn.
- **DYffusion** (Rühling Cachay, Zhao, Joren, Yu, NeurIPS 2023; https://arxiv.org/abs/2306.01984, read): time-conditioned stochastic interpolator plus forecaster in place of Gaussian diffusion. Continuous-time sampling. Navier-Stokes (221x42 grid, 64-step trajectories): ensemble-mean MSE 0.022 vs 0.027 (Dropout) and 0.028 (Perturbation). **Trained with horizon h = 16**, which is beyond our HOR.
- **Oommen, Bora, Zhang, Karniadakis, "Integrating Neural Operators with Diffusion Models Improves Spectral Representation in Turbulence Modeling"** (arXiv:2409.08477v2, Feb 2025; read). A diffusion model is conditioned on the neural-operator output as a post-processor. Energy-spectrum error improves by an order of magnitude, **but "the field error of NO is slightly better than NO+DM"**: recovered frequencies "do not occur at all the correct locations ... due to misalignment or phase errors". In autoregressive mode the diffusion correction helps only at long leads, and even at 120τ the AR-only relative MSE (2.25e-3) beats the diffusion-corrected one (2.46e-3).
- **Latent Generative Solvers** (Chen & Deng, arXiv:2602.11229v3, Sep 2026; abstract read): VAE plus input-noised flow-matching transformer in latent space. On 20-step rollouts, L2RE 56.1% → 30.2% vs U-AFNO. It is autoregressive and stochastic.
- **Compatibility:** a generative **decoder** conditioned on the linear latent z(τ) keeps LIN, CT, MA, FO and E2E (if trained jointly), but breaks determinism and hurts EFF (many denoising calls per frame). Generative *dynamics* breaks LIN.
- **Targets:** BLUR in the perceptual and spectral sense. The evidence says it **does not reduce MSE** and often slightly increases it. It would raise correlation only if phase were already right. **Verdict: only if the paper's headline metrics move to spectra or LSiM. Not for MSE.**

### 2.8 Stability via Jacobian, Lipschitz or dissipativity regularisation
- MNO dissipativity (above) and FouRKS RK4 address bounded-ness. Our latent is already stable by construction (gamma >= 0, block-diagonal oscillators), so this whole class targets a problem we do not have.
- The one residual Lipschitz question is the **decoder's** sensitivity to latent error. Latent-noise injection penalises it implicitly, a known Tikhonov-equivalence result (Bishop 1995, from memory, **not re-opened**). See 2.11.
- **Verdict: not a priority.**

### 2.9 Benchmark studies of remedies (APEBench, List et al.)
- **APEBench** (Koehler, Niedermayr, Westermann, Thuerey, NeurIPS 2024 Datasets & Benchmarks; https://arxiv.org/abs/2411.00180, read): systematic taxonomy of unrolled training (T steps, branch length B) and "diverted-chain" training with a differentiable solver. Finding: "unrolling improves accuracy at the highest difficulty", with large gains for local (ResNet) architectures and marginal gains for FNO.
- **List, Chen, Bali, Thuerey, "Differentiability in Unrolled Training of Neural Physics Simulators on Transient Dynamics"** (arXiv:2402.12971v2, Oct 2024; read). They disentangle the two effects of unrolling. Unrolling **without** temporal gradients (pure distribution-shift exposure) gives on average a 33% error reduction over one-step training. Long-term gradients add about another 15%.
- **For us:** the 33% "distribution shift" component has no analogue in our model. The "long-term gradient" component is what our closed-form 8-frame loss already provides, with exact gradients through every lead. **We are already at the "fully differentiable unrolled" end of their spectrum**, so these studies suggest there is no cheap win left in *how* the 8 frames are supervised. That shifts attention to *what* the 8 frames cover (2.10) and to the *structure* of A (2.12).
- **Compatibility:** n/a (training-paradigm studies).

### 2.10 Frequency identification in linear spectral models: why 8 frames under-determine phase
- **Citation:** Lange, Brunton, Kutz, "From Fourier to Koopman: Spectral Methods for Long-term Time Series Prediction", JMLR 22(41):1-38, 2021. https://jmlr.org/papers/v22/20-406.html; arXiv:2004.00574 (PDF read).
- **Mechanism and evidence:** For x_t ≈ A·Ω(ωt), "the magnitude of gradients [w.r.t. ω] grows linearly with time". The loss is 2π/t-periodic in ω, so it is highly non-convex over long windows. Their fix is a global FFT search to locate the basin followed by local gradient descent. Fluid examples (Kolmogorov 2D experiment, cylinder, cavity) are qualitative; the paper notes the inductive bias suits "oscillator flows" but not "noise amplifiers".
- **Implication for us (our inference):** with 8 supervised frames, the gradient on the shedding frequency is weighted by t ≤ 8Δt. A 1% frequency error costs only a few degrees of phase inside the window, so MSE hardly sees it. Rough estimate: 26-37 deg at lead 58 with a 0.2-1.5% frequency error implies a shedding period of very roughly 5-10 frames.
  - The same error is 26-37 deg at lead 58, and (scaling linearly) about 100+ deg by lead 240.
  - Our error is still well inside the convex basin (< 180 deg at lead 58), so longer-*span* supervision would refine ω by plain gradient descent, without needing Lange's FFT search.
- **Compatibility:** the method itself (fitting frequencies over long windows) violates HOR if used as pointwise supervision. The **stride trick** below (shortlist #2) keeps 8 supervised frames but spans more physical time. **Flag: grey area w.r.t. HOR.**

### 2.11 Latent-space noise and the reconstruction/dynamics trade-off in latent surrogates
- **Robertson, Oommen, ..., Dingreville, "Beyond Compression: Training Latent Representations for Stable Long-Horizon Rollout in Neural Surrogate Solvers"**, arXiv:2609.30198 (Sep 2026; read).
  - **Claim:** instability "does not stem from the latent representation itself ... it arises when they are trained solely for reconstruction". Interventions that help long rollout "often degrade conventional training metrics, including reconstruction and one-step prediction accuracy".
  - **Interventions:**
    - a Koopman next-step reconstruction term during autoencoder training;
    - "Hamming" noise, i.e. constant-variance Gaussian noise added to the latent after encoding and before the Koopman map (σ = 0.1);
    - dynamics-input noise;
    - 1-3 extra unroll steps of fine-tuning.
  - **Evidence:** spinodal decomposition (cumulative Table 1, 15-step rollout error):
    - baseline 0.982
    - +Koopman 0.772
    - +Hamming noise 0.768, about 20% better than the KL-regularised variant (0.906)
    - +dynamic noise 0.601
    - +roll fine-tuning about 0.58
    - Overall "~40%" reduction; one-step error got worse in several steps.
  - Benchmarks are phase-field and active matter (the Well). They are **not fluid**, and the dynamics model is a nonlinear AViT.
- **LE-PDE** (Wu, Maruyama, Leskovec, NeurIPS 2022; https://arxiv.org/abs/2206.07681, read): multi-step loss plus latent consistency plus reconstruction. Ablation (2D NS): full 0.1861; no multi-step 0.2156; no consistency 0.2316; **no reconstruction 0.2025**; horizon M = 1 / 3 / 4 / 6 gives 0.286 / 0.201 / 0.186 / 0.192 (best M = 4). Best d_z = 256 for 2D.
- **tcKAE** (Nayak, Goswami, Kumar, Teixeira, "Temporally Consistent Koopman Autoencoders for Forecasting Dynamical Systems", Scientific Reports 2025, doi 10.1038/s41598-025-05222-7; https://arxiv.org/abs/2403.12335, read).
  - **Mechanism:** consistency between latent predictions of the same target time made from *different start times* (K^k z_n vs K^{k+q} z_{n-q}), with no target needed.
  - **Evidence:** cylinder flow (FEATool, u-velocity), 2500-sample extrapolation, relative error %: Ntrain = 40 clean: DAE 1.695, cKAE 1.616, **tcKAE 0.874**; Ntrain = 28 clean: 10.18 / 7.15 / **3.81**. The authors say "with sufficient training data, the performance gap ... narrows significantly" and note "secular growth in relative error", i.e. phase drift remains.
- **VAMO** (Liao, Guilleminot, Tarokh, arXiv:2609.16621, Sep 2026; abstract and results read): variational latent with structured Gaussian perturbations and a neural-operator transition. Reports 51-87% relative L2 reduction on 2D incompressible NS and 77.6% on compressible NS vs the strongest baseline. The transition is non-linear and autoregressive.
- **Lusch, Kutz, Brunton** (Nat. Commun. 2018; arXiv:1712.09707, abstract read): an auxiliary network parametrises continuous frequencies. The abstract confirms this; that the frequency is a function of the latent amplitude is from memory, **UNVERIFIED** here. This breaks exact LIN.
- **Otto & Rowley, LRAN** (arXiv:1712.01378, abstract read; published SIADS 2019, from memory): linear-recurrent autoencoder on an unstable cylinder wake; "accurate low-dimensional models".
- **Compatibility of the transferable pieces:** Hamming/latent noise, tcKAE multi-start consistency and the reconstruction anchor all satisfy LIN, CT, MA, FO, E2E, HOR and EFF.
- **Targets:** BLUR via decoder robustness, and the lead-1 floor via the reconstruction anchor. tcKAE weakly targets PHASE (it enforces time-homogeneity across start times).
- **Caveat (our reasoning):** with an MSE decoder, noise on z teaches the decoder E[x | z + ε], which is *smoother*. Latent noise can therefore **increase** blur while improving robustness. Use a small σ and anneal it.

### 2.12 Koopman spectral structure of the cylinder wake (theory that bears on HARM and PHASE)
- **Citation:** Bagheri, "Koopman-mode decomposition of the cylinder wake", J. Fluid Mech. 726:596-623, 2013 (pre-2021, but the theory underpins shortlist #1). PDF read.
- **Content:** near the Hopf bifurcation the Koopman eigenvalues form a **lattice** λ_{m,n} = mσ + i·nω. On the limit cycle "the eigenvalues are on the imaginary axis and integer multiples of the fundamental frequency". Harmonics arise from interactions of the first harmonic with itself and with the mean-flow mode.
- **Implication (our inference):** a free block-diagonal A has to learn each harmonic's frequency independently, so small independent errors let harmonics drift out of phase with the fundamental. This matches the HARM component (32%) growing with lead. A lattice parametrisation ties them together and pools the frequency gradient: harmonic n contributes n·t to ∂/∂ω₁.
- **Search result:** no ML paper was found that imposes this lattice as an architectural constraint in a Koopman autoencoder and reports numbers (searched; **none found**). Shortlist #1 is therefore an extrapolation from theory, not a validated method.
- **Risk:** the transonic Tra wake may contain non-lattice content (shock-related, low-frequency modulation, possibly quasi-periodic at some Ma). Keep some free oscillators.

### 2.13 Context: the user's own prior work surfaced by search
- Grozavescu, Meunier, Zhang, Girolami, "Invertible continuous latent dynamic for long-term data assimilation in complex physical systems", arXiv:2602.02832v4 (Sep 2026). This is the CT-KAE paper. A related ICLR 2026 ocean paper (arXiv:2603.05560) was listed in search results but not opened. Noted only so it is not mistaken for independent evidence.

---

## 3. Compatibility summary table

Y = compatible; N = violates; G = grey area.

| Method | LIN | CT | MA | FO | E2E | HOR | EFF | Needs AR stepping? | Targets | Evidence strength for MSE on wake-type flow |
|---|---|---|---|---|---|---|---|---|---|---|
| Pushforward / bundling | Y | Y | Y | Y | Y | Y | Y | **yes; mechanism absent** | none | n/a |
| Input-noise injection | Y | Y | Y | Y | Y | Y | Y | mechanism needs AR | (BLUR, weak) | strong for AR U-Net on Tra; n/a for us |
| PDE-Refiner / SpectralRefiner | Y (post-proc.) | Y | Y | Y | Y | Y | N (3-4x calls) | no (as post-proc.) | BLUR | **negative on Tra** |
| Spectral / Sobolev loss | Y | Y | Y | Y | Y | Y | Y | no | BLUR, HARM | negative (2 studies) |
| FouRKS test-time correction | Y | Y | Y | G | Y | Y | N | yes | DAMP | stability only |
| DySLIM conditional MMD (ℓ ≤ 8) | Y | Y | Y | Y | Y | **Y** | Y | no | DAMP, BLUR | moderate (Kolmogorov stats and cosine) |
| Long-lead statistics matching (closed form) | Y | Y | Y | Y | Y | **G** | Y | no | DAMP, detail collapse | indirect (DySLIM, Jiang) |
| Jiang OT / contrastive (h = 1) | Y | Y | G (needs per-Ma stats) | Y | Y | Y | Y | no | DAMP, BLUR | KS / L96 only |
| Thermalizer | Y | Y | Y | Y | N (separate model) | Y | N | yes | DAMP / statistics | no MSE benefit |
| AMSE / FreDF | Y | Y | Y | Y | Y | Y | Y | no | BLUR (AMSE), PHASE (FreDF, weak) | AMSE raises MSE; FreDF not fluid |
| Diffusion decoder / post-processor | Y | Y | Y | Y | Y | Y | N | no | BLUR (spectral) | MSE slightly worse (Oommen) |
| Latent (Hamming) noise | Y | Y | Y | Y | Y | Y | Y | no | BLUR (robustness) | moderate, non-fluid |
| tcKAE multi-start consistency | Y | Y | Y | Y | Y | Y | Y | no | PHASE (weak), BLUR | cylinder, low-data regime |
| Reconstruction anchor (LE-PDE) | Y | Y | Y | Y (auxiliary AE) | Y | Y | Y | no | lead-1 floor, BLUR | LE-PDE ablation (NS) |
| Stride-randomised 8-frame supervision | Y | **Y (exploits CT)** | Y | Y | Y | **G** | Y | no | PHASE, HARM, DAMP | indirect (Lange; Stachenfeld Δt) |
| Harmonic-lattice generator | Y | Y | Y | Y | Y | Y | Y | no | HARM, PHASE | theory only (Bagheri) |

---

## 4. Ranked shortlist for the conditional CT-KAE

Ranking criteria: (i) the method targets our *measured* error components rather than stability; (ii) it fully respects the constraints; (iii) it fits the evidence that sharpening alone hurts MSE.

### #1. Harmonic-lattice (phase-locked) parametrisation of A(Ma)
- **Sketch:**
  - Split the 128 oscillator blocks into a *lattice set* and a *free set*.
  - In the lattice set, block j has frequency ω_j = n_j·ω₁(Ma) (n_j ∈ {1, ..., H}, H ≈ 6-10, several blocks per n so different spatial structures can share a harmonic) and damping γ_j = m_j·σ(Ma) + γ_j⁰ (γ_j⁰ ≥ 0 learned; or simply free γ_j ≥ 0).
  - ω₁(Ma) and σ(Ma) come from a small MLP, or a low-order polynomial for smooth zero-shot Ma.
  - The free set (for example 25-50% of blocks) keeps the current unconstrained parametrisation for non-lattice content.
  - Optionally relax the lattice with a penalty λ·Σ(ω_j − n_j·ω₁)² instead of a hard tie, and anneal λ upward.
  - Initialise ω₁(Ma) from the current model's measured shedding frequency.
- **Expected effect and reasoning:**
  - The theory says the post-transient wake spectrum *is* this lattice (Bagheri 2013).
  - Tying harmonics to ω₁ removes independent frequency errors, so harmonics stay phase-coherent with the fundamental. That targets HARM (32% at lead 30) and the coherent part of BLUR (misplaced fine structure).
  - Harmonic n also feeds n·t into the gradient on ω₁, which sharpens the fundamental estimate from the same 8 frames, targeting PHASE.
  - The model gets fewer free parameters, which may also help zero-shot Ma.
  - My guess (not evidence) is a meaningful cut in the excess error at leads 30-60; most likely no change at lead 1.
- **Main risk:**
  - The transonic wake at higher Ma may not be a clean single-frequency limit cycle (shock motion, modulation, quasi-periodicity), so a hard lattice could under-fit. Mitigate with the free set and a soft penalty.
  - **No published ML validation was found.** This is an extrapolation from theory.

### #2. Stride-randomised 8-frame supervision (exploits continuous time) — GREY w.r.t. HOR
- **Sketch:**
  - Keep the 2 native-Δt conditioning frames.
  - Supervise 8 target frames at leads τ_k = k·s·Δt, k = 1..8, with stride s drawn per batch from {1, 2, 3, 4} (curriculum: start with s = 1 only, then widen the range).
  - Optionally mix strides within a sample, e.g. leads {1, 2, 3, 4, 6, 8, 12, 16}.
  - Normalise the per-frame MSE by lead so long leads do not dominate early training.
  - Inference is unchanged.
- **Expected effect and reasoning:**
  - Lange et al. show that frequency gradients scale with t and that MSE over a short window barely constrains ω. Doubling or quadrupling the physical span at the same frame count directly attacks PHASE (8% plus the phase part of HARM and BLUR).
  - The "detail" latent that collapses after 8 native steps would be trained to stay useful out to 32+ steps, attacking DAMP.
  - Our current error (26-37 deg at lead 58) is well inside the convex basin, so the non-convexity Lange warns about should not bite for s ≤ 4.
  - Stachenfeld et al.'s U-shaped error vs Δt shows that step size is a first-order knob.
  - LE-PDE's horizon sweep (M = 1 → 4 gives a large gain, M = 6 slightly worse) suggests diminishing returns in *frame count*. Stride changes span, not count.
- **Main risk:**
  - **Constraint interpretation.** Same number of supervised frames and conditioning frames, but the supervised physical horizon grows to about 32 native steps. This should be disclosed and possibly paired with a baseline trained on the same strided targets (U-Net_ut cannot easily do this; that asymmetry is arguably the CT model's legitimate advantage).
  - Possible lead-1 degradation if s = 1 is under-sampled.

### #3. Decoder-side robustness and anchoring: small latent noise + multi-start latent consistency + reconstruction anchor
- **Sketch (three cheap auxiliary terms, all within the 8-frame window):**
  - (a) Latent noise: z₀ ← z₀ + ε, ε ~ N(0, σ²·diag(var(z))), with σ ≈ 0.02-0.05, annealed to 0. Optionally add ε to z(τ) before decoding.
  - (b) tcKAE-style term ‖e^{A(τ+q)}z(t−q) − e^{Aτ}z(t)‖², using pairs of start times inside the same 10-frame clip. No target is needed.
  - (c) Reconstruction anchor L_rec = ‖D(E(x_{t+k−1}, x_{t+k})) − x_{t+k}‖² on target frames. This is an auxiliary autoencoding loss, not data assimilation, and keeps the decoder able to express detail. Check that it fits the user's definition of "forecasting-only".
- **Expected effect and reasoning:**
  - The decoder is the only component that sees distribution-shifted inputs (propagated latents drift off the encoder manifold). (a) and (b) shrink that drift and make the decoder tolerant of it.
  - Robertson et al. found latent noise worth about 20% at 15-step rollouts.
  - tcKAE halved cylinder extrapolation error, but in a tiny-data regime, with gains that "narrow significantly" given enough data.
  - LE-PDE found the reconstruction term worth about 8% on 2D NS.
  - Targets BLUR and the lead-1 floor (0.26 vs 0.02).
  - Expect modest gains. This is the safest option, not the biggest lever.
- **Main risk:**
  - With an MSE decoder, latent noise teaches E[x | z + ε], which is smoother, so it can *increase* blur. Keep σ small and annealed, and monitor the outside-span diagnostic.
  - The tcKAE gain may vanish at Tra data scale.

### #4. Within-horizon distributional regularisation of decoded frames (DySLIM conditional MMD, ℓ ≤ 8)
- **Sketch:**
  - Per mini-batch grouped by Ma bin, compute MMD² between the set of decoded predictions {D(e^{A(Ma)τ_k}z₀)} and the true frames {x(τ_k)} at the same leads.
  - Kernel: Σ_q σ_q²/(σ_q² + ‖u − v‖²) on 2x-downsampled fields, or on POD/feature projections.
  - Weight λ₂ tuned so pointwise MSE at lead 8 does not regress.
  - A grey-area extension applies the same MMD at leads 16-240 against per-Ma pooled true frames. No pairing is needed, so there is no pointwise supervision.
- **Expected effect and reasoning:**
  - DySLIM on Kolmogorov cut MELR and Wass1 substantially and reduced the baselines' "veer towards the mean" (exactly our DAMP and detail-collapse symptom), with improved short-term cosine similarity.
  - Would mainly help correlation and energy (our correlation at step 58 is 0.69 vs U-Net 0.85) and somewhat less MSE.
- **Main risk:**
  - The double penalty: sharper but misplaced structure can raise MSE (as in AMSE and Oommen).
  - MMD in high dimension is weak (DySLIM dropped the unconditional term for this reason).
  - The long-lead extension is HOR-grey.

### #5 (conditional on metric choice). Generative refinement decoder
- **Sketch:** a small conditional flow-matching or diffusion decoder p(x | z(τ), Ma), trained jointly with or after the CT-KAE, used only when spectra or LSiM are reported.
- **Expected effect:** better spectra and visual sharpness at long leads. MSE is likely flat or worse (Oommen: NO+DM field error slightly worse than NO; PDE-Refiner much worse on Tra).
- **Main risk:** breaks determinism and EFF, and is unlikely to close the MSE gap. **Not recommended if MSE stays the headline metric.**

### Explicitly de-prioritised
- Pushforward, input-noise injection, unrolled-without-gradient training, Thermalizer and FouRKS test-time correction: the mechanism (autoregressive feedback) is absent in our model.
- PDE-Refiner: negative evidence on Tra itself.
- Plain spectral or Sobolev losses: two independent negative results for rollout accuracy.
- Jacobian or dissipativity regularisation: the latent is already stable by construction.

### Honest assessment of evidence strength
- None of the top three has direct evidence on a cylinder-wake benchmark at Tra's data scale with a linear-latent model.
  - #1 rests on Koopman theory (Bagheri 2013) plus our own HARM/PHASE diagnostics.
  - #2 rests on an analytic gradient argument (Lange 2021) plus general Δt evidence.
  - #3 rests on non-fluid latent-model results (Robertson 2026) and a small-data cylinder KAE (tcKAE).
- The most cost-effective experiment is probably #1 + #2 together, since they target the same PHASE → HARM → BLUR cascade, with the outside-span / POD-band / phase diagnostics re-run at leads 30 and 58 to confirm the mechanism.

---

## 5. Sources opened
- https://arxiv.org/pdf/2309.01745 and https://arxiv.org/html/2309.01745v3 (ACDM)
- https://arxiv.org/abs/2202.03376 (MP-PDE)
- https://arxiv.org/abs/2112.15275 (Stachenfeld)
- https://arxiv.org/abs/2308.05732 (PDE-Refiner)
- https://arxiv.org/abs/2506.10711 (PDE-SpectralRefiner)
- https://arxiv.org/abs/2304.07029 (Chattopadhyay/Sun/Hassanzadeh)
- https://arxiv.org/abs/2106.06898 (MNO)
- https://arxiv.org/abs/2402.04467 and HTML v2 (DySLIM)
- https://arxiv.org/abs/2306.01187 (Jiang et al.)
- https://arxiv.org/abs/2503.18731 (Thermalizer)
- https://arxiv.org/abs/2501.19374 (AMSE)
- https://arxiv.org/abs/2402.02399 (FreDF)
- https://arxiv.org/abs/1909.09020 (DILATE, abstract)
- https://arxiv.org/abs/2306.01984 (DYffusion)
- https://arxiv.org/abs/2409.08477 (Oommen et al.)
- https://arxiv.org/abs/2602.11229 (LGS)
- https://arxiv.org/abs/2411.00180 (APEBench)
- https://arxiv.org/abs/2402.12971 (List et al.)
- https://jmlr.org/papers/v22/20-406.html and https://arxiv.org/abs/2004.00574 (Lange et al.)
- https://arxiv.org/abs/2609.30198 (Robertson et al.)
- https://arxiv.org/abs/2206.07681 (LE-PDE)
- https://arxiv.org/abs/2403.12335 (tcKAE)
- https://arxiv.org/abs/2609.16621 (VAMO)
- https://arxiv.org/abs/1712.09707 (Lusch, abstract)
- https://arxiv.org/abs/1712.01378 (Otto & Rowley, abstract)
- https://www.mech.kth.se/~shervin/pdfs/2013_jfm_koopman_cylinder.pdf (Bagheri 2013 JFM)
- https://arxiv.org/abs/2602.02832 (the user's own CT-KAE paper; context only)
