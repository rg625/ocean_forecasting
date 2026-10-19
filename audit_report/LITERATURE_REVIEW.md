# Literature review for the KAE forecasting audit (as of 2026-10-19)

Scope: continuous/discrete Koopman autoencoders (KAE), generator learning, DMD, neural operators, ocean emulators, long-horizon stability. Each entry was checked against an arXiv/journal/proceedings page or the Semantic Scholar record, and code links were checked against the GitHub API. "unverified" means I could not confirm the item this session. "(memory)" marks a detail I recall from reading the paper but did not re-open this session.

---

## 0. Identification of the local PDF and the authors' own prior work (read this first)

**Local PDF `22567_Scaling_Continuous_Time_.pdf`** is an anonymised NeurIPS-format submission titled *"Scaling Continuous-Time Koopman Autoencoders for High-Dimensional Fluid Dynamics"*. Its abstract claims closed-form inference z(τ)=exp(K_cont τ)z(0), with rollout training, forward-backward consistency, latent regularisation and physics-conditioned LoRA, and a 110x inference speedup over diffusion and operator baselines. No public version under that exact title was found. The closest public versions are the repo owner's own preprints:

- **arXiv:2602.02832**, Grozavescu, Meunier, Zhang, Girolami. v1 was 2 Feb 2026 and v4 was 29 Sep 2026. The current title is *"Invertible continuous latent dynamic for long-term data assimilation in complex physical systems"*; the v1/v2 title was *"Koopman Autoencoders with Continuous-Time Latent Dynamics for Fluid Dynamics Forecasting"*. No venue or code is listed. v2 states: K_cont(φ)=K0+N_ψ(φ) with a LoRA hypernetwork, **RK4 during training and the matrix exponential at inference**, and "close agreement" between the two (Sec. 6.2). The stability argument is "dissipative initialisation, spectral constraints" plus eigenvalues observed mostly in the left half-plane (App. E.3). There is no hard Re(λ)≤0 constraint. ACDM Inc/Tra results: Tra_long (T=240) MSE 17.0e-3 vs ACDM ~22.2e-3, but the table lists 22.6e-3, **an internal inconsistency to fix**. The paper also concedes that ACDM is better at short horizons on Tra_int and Tra_ext. Per-step cost is quoted as 1.67 µs in the text but 1.04e-3 ms in Table 2 (another inconsistency). v4 adds KS and data assimilation (DA) and states that "a strong forecaster is not necessarily a strong assimilator".
- **arXiv:2603.05560**, Grozavescu, Zhang, Girolami, Meunier, Mar 2026, *"Towards Efficient and Stable Ocean State Forecasting: A Continuous-Time Koopman Approach"*. It uses 2-layer QG at 64x64 with Δt=5 h, and **the data setup follows Pedersen et al. 2025 (Thermalizer)**. Generator = skew + sym, RK4 for training and exp for inference. It compares against an AR ViT over 2083 days (~10k steps): RMSE 0.938 vs 0.993, KE drift −49.4% vs +23.3%, enstrophy drift −34.6% vs +37.0%. ACC is near zero for both models. The paper admits that fine scales are dissipated.

**Audit implication.** Both preprints claim exp(Kτ) at inference. The audited code uses a single RK4 step per frame with dt=0.1 (see §5a). Whether the published numbers came from RK4 or expm, and how far apart the two maps are for the trained A, must be checked. The QG KE drift of −49% is what a dissipative linear generator produces (see §5a and §5d).

---

## 1. Shortlist

### A. Koopman autoencoders

**A1. Deep learning for universal linear embeddings of nonlinear dynamics.** Lusch, Kutz, Brunton. Nat. Commun. 9 (2018) 4950. [arXiv:1712.09707](https://arxiv.org/abs/1712.09707), DOI 10.1038/s41467-018-07210-0. Code: [BethanyL/DeepKoopman](https://github.com/BethanyL/DeepKoopman) (verified).
- Arch: MLP AE. The latent is a set of Koopman eigenfunctions. An **auxiliary network outputs eigenvalues (μ±iω) as functions of the latent radius**, which handles continuous spectra. K is block-diagonal 2x2 Jordan/rotation blocks of exp(μΔt)[cos ωΔt, −sin; sin, cos].
- Objective: reconstruction + multi-step prediction in state space + multi-step linearity in latent space (the target is the encoder output, **not detached**) + L∞ penalty.
- Data: pendulum, Duffing, fluid flow on an attractor (cylinder-wake reduced model). Low-dimensional, horizons of tens to hundreds of steps.
- Limitations: low-dimensional states only, and the eigenvalue network makes the dynamics nonlinear.
- Relevance: this is the canonical **block-diagonal (μ, ω) parameterisation**, a direct alternative to dense skew+sym. It also shows that a non-detached latent-linearity loss can be trained when reconstruction anchors the encoder.
- Borrow: the 2x2 rotation-block structure with exact exponential per block.

**A2. Learning Koopman Invariant Subspaces for DMD.** Takeishi, Kawahara, Yairi. NeurIPS 2017. [arXiv:1710.04340](https://arxiv.org/abs/1710.04340). Code: none found.
- Formulation: the encoder g is learned so that DMD on g(x) is accurate. K is obtained in closed form by least squares (pseudo-inverse) inside the loss, not as a free parameter.
- Data: toy systems and small fluid examples. Short horizons.
- Relevance: an "operator by regression" alternative, K = Y X⁺ per batch, which removes K from the optimiser and ties it to the encoder (related to the VAMP/DPNets line).
- Engineering: batch least squares on 128-d latents is cheap. Stability is not controlled.

**A3. Linearly-Recurrent Autoencoder Networks for Learning Dynamics.** Otto, Rowley. SIAM J. Appl. Dyn. Syst. 18 (2019) 558–593. [arXiv:1712.01378](https://arxiv.org/abs/1712.01378), DOI 10.1137/18M1177846. Code: none found.
- Formulation: z_{t+1}=K z_t. Multi-step prediction loss with **normalised (relative) errors** across the horizon, which avoids trivial small-latent solutions. Shows that EDMD is a special case.
- Data: cylinder wake and other low-dimensional fluid systems.
- Relevance: the **relative/normalised latent loss** is a cheap guard against latent-scale shrinkage when the encoder is not detached (§5c).

**A4. Forecasting Sequential Data using Consistent Koopman Autoencoders.** Azencot, Erichson, Lin, Mahoney. ICML 2020 (per Semantic Scholar; the arXiv page lists no venue). [arXiv:2003.02236](https://arxiv.org/abs/2003.02236). Code: [erichson/koopmanAE](https://github.com/erichson/koopmanAE) (verified).
- Formulation: forward K and backward D with z_{t+1}=Kz_t and z_{t−1}=Dz_t. Consistency penalty Σ_k ‖D_{k*}C_{*k} − I_k‖² + ‖C_{k*}D_{*k} − I_k‖² (approximately encouraging D ≈ K⁻¹).
- Data: pendulum, cylinder flow, a sphere-flow surrogate, climate (SST) data. Horizons are tens to hundreds of steps. Metric: relative prediction error.
- Limitations: the consistency term is a soft constraint. Results are mostly on low-to-moderate dimensions.
- Relevance: **already a variant in this repo**. The audit should check that the repo's backward operator and consistency loss match this definition, including the sub-block sum.

**A5. VAMPnets for deep learning of molecular kinetics.** Mardt, Pasquali, Wu, Noé. Nat. Commun. 9 (2018) 5. [arXiv:1710.06012](https://arxiv.org/abs/1710.06012). Code: VAMPnet in [deeptime-ml/deeptime](https://github.com/deeptime-ml/deeptime) (repo verified).
- Formulation: the encoder maximises the VAMP-2 score ‖C00^{-1/2} C01 C11^{-1/2}‖_F². K is obtained in closed form from **whitened covariances**.
- Relevance: the whitening makes the objective **invariant to latent scale and immune to collapse by construction** (§5c). The repo owner's QG preprint already uses "covariance whitening".
- Engineering: covariance inverses are needed per batch, which is fine at d=128.

**A6. From Fourier to Koopman: Spectral Methods for Long-term Time Series Prediction.** Lange, Brunton, Kutz. JMLR 22 (2021) 41:1–38. [arXiv:2004.00574](https://arxiv.org/abs/2004.00574). Code: [helange23/from_fourier_to_koopman](https://github.com/helange23/from_fourier_to_koopman) (verified).
- Formulation: the latent is constrained to **purely periodic modes**, z_t = [cos(ω_k t), sin(ω_k t)], with frequencies found by an FFT-based global objective. A nonlinear decoder handles the rest. There is no growth or decay by construction.
- Strength: very long horizons for quasi-periodic signals (energy and traffic data, fluid flows).
- Limitation: it cannot represent transients or broadband chaos.
- Relevance: a strong **baseline for the periodic Tra/Inc vortex-shedding cases**, where the shedding is near limit-cycle. A KAE should at least match it on Tra_long.

**A7. Koopman Neural (Operator) Forecaster for Time Series with Temporal Distribution Shifts.** Wang, Dong, Arik, Yu. ICLR 2023 (the proceedings title adds "Operator"). [arXiv:2210.03675](https://arxiv.org/abs/2210.03675). Code: [google-research/KNF](https://github.com/google-research/google-research/tree/master/KNF) (path verified).
- Arch: a measurement function built from a fixed dictionary, plus a global K, a local (attention-based) K, and a feedback-loop K updated by prediction residuals.
- Data: M4, crypto, sports trajectories. Not PDEs.
- Relevance: low. This is a time-series method. The idea of a local operator plus residual feedback is mildly relevant to Mach conditioning.

**A8. Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors.** Liu, Li, Wang, Long. NeurIPS 2023. [arXiv:2305.18803](https://arxiv.org/abs/2305.18803). Code: [thuml/Koopa](https://github.com/thuml/Koopa) (verified).
- Arch: Fourier filter splitting time-invariant and time-variant parts. A learned global K handles the invariant part and a **per-window K is fitted by least squares** (DMD-style) for the variant part.
- Data: standard long-term time-series benchmarks. Not PDEs.
- Relevance: low for these fields. It is a reminder that per-window closed-form K needs no backprop through K.

**A9. Temporally Consistent Koopman Autoencoders (tcKAE).** Nayak, Chakrabarty, Kumar, Teixeira, Goswami. Sci. Rep. 15 (2025) 22127. [arXiv:2403.12335](https://arxiv.org/abs/2403.12335). Code: none found.
- Formulation: consistency regularisation so that multi-step predictions of the same target from different start times agree, e.g. K^j z_{t−j} ≈ K^i z_{t−i}.
- Data: pendulum, kinetic plasma, fluid flow. The method targets low-data and noisy regimes.
- Relevance: an alternative to the second-difference latent smoothness penalty, and one with a Koopman-spectral justification. Cheap to add.

**A10. Stability-constrained Koopman generators (2021–2026).**
- **Bevanda et al., "Diffeomorphically Learning Stable Koopman Operators".** IEEE Control Syst. Lett. 6 (2022). [arXiv:2112.04085](https://arxiv.org/abs/2112.04085). Uses an **unconstrained parameterisation of Hurwitz matrices**, so stability holds regardless of approximation error.
- **Fan, Yi, Rye, Shi, Manchester, "Learning Stable Koopman Embeddings for Identification and Control".** Automatica (per Semantic Scholar), [arXiv:2401.08153](https://arxiv.org/abs/2401.08153). A direct parameterisation with built-in stability, turning learning into an unconstrained problem, with a link to contraction.
- **Mondal et al., "Efficient Dynamics Modeling in Interactive Environments with Koopman Theory".** ICLR 2024. [arXiv:2306.11941](https://arxiv.org/abs/2306.11941). A Koopman latent model with convolution-parallel long-range prediction and stability analysis. In their RL setting, latent consistency losses in two latent spaces "promote collapse"; they use stop-grad targets rather than EMA (reported in search snippet, not opened in full).
- **Petersen et al., "Reduced-Order Surrogates for Forced Flexible Mesh Coastal-Ocean Models".** [arXiv:2602.05416](https://arxiv.org/abs/2602.05416), 2026, Ocean Modelling DOI listed. A KAE with forcing and eigenvalue regularisation for stability, with horizons up to 1 yr. Code: none found.
- Relevance: the repo's A = skew(W)+sym(D) is **one line away from a guaranteed-dissipative generator**: set D = −(LLᵀ + εI) (§5b). The papers above show that such parameterisations do not hurt fit in their domains.

**A11. Information Shapes Koopman Representation.** Cheng, Yuan, Yang, Zhang, Cheng, He, Sun. ICLR 2026. [arXiv:2510.13025](https://arxiv.org/abs/2510.13025). Code: [Wenxuan52/InformationKoopman](https://github.com/Wenxuan52/InformationKoopman) (verified).
- Formulation: information-bottleneck Lagrangian. Over-weighting latent mutual information causes **collapse onto a few dominant modes**, and a **von Neumann entropy** regulariser on the latent covariance keeps modes diverse.
- Relevance: directly addresses latent mode collapse in KAEs (§5c). The repo's QG preprint "eigenvalue repulsion" plays a similar role.

**A12. Learning Chaos in a Linear Way (PFNN).** Cheng et al. ICLR 2025. [arXiv:2503.14702](https://arxiv.org/abs/2503.14702). Code: none found.
- Formulation: AE + linear evolution operators trained to respect **contraction onto the attractor and measure invariance**, without knowing the invariant measure.
- Data: Lorenz, KS, Navier–Stokes (per search summary). Metrics are statistical (invariant-measure and spectral).
- Relevance: the closest published "linear latent for chaos" baseline on **KS**. It supports judging the KS and QG results by statistics, not RMSE.

**A13. Learning Propagators for Sea Surface Height Forecasts Using Koopman Autoencoders.** Brettin, Zanna, Barnes. GRL 52(4) 2025, DOI 10.1029/2024GL112835. Code: [andrewbrettin/koopman_autoencoders_ssh_prediction](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction) (verified).
- Formulation: conv AE + latent propagator. Joint learning of reduction and time-stepping beats learning them separately (vs LIM-style). Daily and monthly SSH anomalies in the North Pacific and Atlantic. **Real-ocean (reanalysis/model) data, regional.**
- Propagator detail (linear? continuous?) unverified from the abstract page.
- Relevance: the only peer-reviewed KAE for ocean SSH found. A natural citation and an ocean baseline family.

### B. Continuous-time / generator learning and structured linear recurrences

**B1. gEDMD.** Klus, Nüske, Peitz, Niemann, Clementi, Schütte, "Data-driven approximation of the Koopman generator: Model reduction, system identification, and control". Physica D 406 (2020) 132416. [arXiv:1909.10638](https://arxiv.org/abs/1909.10638). Code: none found for the paper. gEDMD implementations exist in community libraries (e.g. deeptime); not verified which.
- Formulation: estimates the generator L directly from time derivatives (dψ/dt = ∇ψ·b + ½ a:∇²ψ) via least squares on a dictionary. No time discretisation of the operator.
- Relevance: the theoretical basis for learning a generator rather than a step map. It needs derivatives or small Δt. With Δt=0.1 on subsampled data, it supports exp-based (not RK4-based) training.

**B2. Neural ODEs / Latent ODEs.** Chen et al., NeurIPS 2018, [arXiv:1806.07366](https://arxiv.org/abs/1806.07366). Rubanova, Chen, Duvenaud, NeurIPS 2019 (venue per my memory, S2 shows arXiv), [arXiv:1907.03907](https://arxiv.org/abs/1907.03907).
- Relevance: the general nonlinear latent-ODE baseline for irregular sampling. The KAE is the linear special case, where an exact solution exists, so **using an ODE solver (RK4) for a linear ODE throws away the main advantage**.

**B3. Resurrecting RNNs for Long Sequences (LRU).** Orvieto, Smith, Gu, Fernando, Gulcehre, Pascanu, De. ICML 2023 (per S2). [arXiv:2303.06349](https://arxiv.org/abs/2303.06349). Code: none found as an official repo (the google-deepmind repo name could not be verified).
- Formulation (verified in the PDF): a dense linear recurrence is diagonalised to complex diagonal Λ (any diagonalisable A = PΛP⁻¹, with P absorbed into the in/out projections). The parameterisation is **λ_j = exp(−exp(ν_log,j) + iθ_j)**, which keeps |λ|<1 for any parameter value and decouples magnitude from frequency, making Adam easier. **Ring initialisation** samples eigenvalues uniformly on an annulus r_min≤|λ|≤r_max, with **small max phase** for very long-range tasks. Pushing r_max toward 1 (0.99) helps only when stability is enforced.
- Normalisation: γ_j = sqrt(1−|λ_j|²) prevents forward-pass blow-up when |λ|→1.
- They also note the exp parameterisation "resembles ZOH" discretisation of continuous SSMs.
- Relevance: the modern template for §5b.

**B4. S4/S5.** Smith, Warrington, Linderman, "Simplified State Space Layers". ICLR 2023 (per S2). [arXiv:2208.04933](https://arxiv.org/abs/2208.04933). Code: [lindermanlab/S5](https://github.com/lindermanlab/S5) (verified). S4 code: [state-spaces/s4](https://github.com/state-spaces/s4) (verified).
- Formulation: continuous-time diagonal complex A (HiPPO-N initialised), with **exact ZOH discretisation Ā = exp(ΔA)** and learnable per-mode Δ, which allows variable step sizes. Parallel scan.
- Relevance: the exact discretisation of a diagonal continuous-time generator costs O(d), so there is no reason to use RK4. The learnable Δ and variable-Δt handling directly match the "query any τ" claim.

### C. DMD family

**C1. On Dynamic Mode Decomposition: Theory and Applications (exact DMD).** Tu, Rowley, Luchtenburg, Brunton, Kutz. J. Comput. Dyn. 1 (2014) 391–421. [arXiv:1312.0041](https://arxiv.org/abs/1312.0041), DOI 10.3934/jcd.2014.1.391.
- Formulation: K = Y X⁺ via SVD truncation. Modes and eigenvalues give continuous-time ω = log(λ)/Δt.
- Relevance: a mandatory linear baseline on raw or POD-reduced fields for Tra/Inc. It sets the floor a KAE must beat.

**C2. EDMD.** Williams, Kevrekidis, Rowley. J. Nonlinear Sci. 25 (2015) 1307–1346. [arXiv:1408.4408](https://arxiv.org/abs/1408.4408).
- Formulation: Galerkin projection of the Koopman operator onto a dictionary. The KAE replaces the dictionary with a learned encoder.

**C3. HAVOK, "Chaos as an intermittently forced linear system".** Brunton, Brunton, Proctor, Kaiser, Kutz. Nat. Commun. 8 (2017) 19. [arXiv:1608.05306](https://arxiv.org/abs/1608.05306).
- Formulation: Hankel/delay embedding, with a linear model in the leading delay coordinates **forced by the last coordinate**. Chaos cannot be captured by a closed finite linear system.
- Relevance: the theoretical reason a closed linear latent cannot reproduce KS/QG chaos without forcing, noise, or nonlinearity (§5d). The repo's history encoder is a learned delay embedding.

**C4. Optimized DMD / BOP-DMD.** Askham, Kutz, "Variable Projection Methods for an Optimized DMD", SIAM J. Appl. Dyn. Syst. 17 (2018), [arXiv:1704.02343](https://arxiv.org/abs/1704.02343). Sashidhar, Kutz, "Bagging, optimized DMD for robust, stable forecasting with spatial and temporal UQ", Phil. Trans. R. Soc. A 380 (2022), [arXiv:2107.10878](https://arxiv.org/abs/2107.10878).
- Formulation: fit x(t) ≈ Φ exp(Ωt) b directly in **continuous time** by variable projection. This handles irregular Δt and noise. BOP-DMD bags the fits for UQ and can **constrain Re(ω)≤0**.
- Relevance: a continuous-time, exp-based, stability-constrainable linear baseline with UQ. Its exp(Ωt) structure is exactly what the KAE claims.

**C5. PyDMD / PyKoopman libraries.** [PyDMD/PyDMD](https://github.com/PyDMD/PyDMD) and [dynamicslab/pykoopman](https://github.com/dynamicslab/pykoopman), both verified.
- Use them for the DMD/EDMD/BOP-DMD baselines without writing new code.

**C6. ResDMD and spectral pollution.** Colbrook, Townsend, "Rigorous data-driven computation of spectral properties of Koopman operators", CPAM 77 (2024), [arXiv:2111.14889](https://arxiv.org/abs/2111.14889). Colbrook, "The Multiverse of DMD Algorithms", [arXiv:2312.00137](https://arxiv.org/abs/2312.00137).
- Content: residual-based detection of spurious eigenvalues and pseudospectra. For non-normal operators, transients differ from what the spectrum predicts.
- Relevance: the eigenvalue plots of the learned A should be supplemented by **pseudospectra or ‖exp(tA)‖ curves**, and an encoder-residual check of the learned modes.

### D. Neural operators and long-rollout stabilisers

**D1. Fourier Neural Operator.** Li et al. ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895). Code: [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator) (verified).
- Formulation: (Kv)(x) = F⁻¹(R·F v)(x) with truncated modes. Autoregressive one-step map.
- In ACDM, FNO16/FNO32 are among the **weakest** on Tra (MSE ~4.8–6.8e-3; see D5).
- Relevance: a standard baseline that already has numbers on the exact Tra protocol.

**D2. Koopman Neural Operator (KNO).** Xiong, Huang, Zhang, Deng, Sun, Tian. [arXiv:2301.10022](https://arxiv.org/abs/2301.10022), v2 May 2024. S2 lists J. Comput. Phys. (journal ref unverified on the arXiv page). Code: [Koopman-Laboratory/KoopmanLab](https://github.com/Koopman-Laboratory/KoopmanLab) (verified; described as "A library for Koopman Neural Operator with Pytorch").
- Formulation: encoder → Koopman operator approximated in Fourier space, applied as powers → decoder.
- Data: 5 PDEs (Navier–Stokes, Rayleigh–Bénard, etc.) plus ERA5 water vapour and western-boundary-current data.
- No ACDM-benchmark results found.
- Relevance: the closest "Koopman + operator" baseline, and easy to run through KoopmanLab.

**D3. PDE-Refiner.** Lippe, Veeling, Perdikaris, Turner, Brandstetter. NeurIPS 2023 (per S2). [arXiv:2308.05732](https://arxiv.org/abs/2308.05732). Code: [pdearena/pdearena](https://github.com/pdearena/pdearena) (verified; the paper cites microsoft/pdearena, which now resolves elsewhere).
- Finding: MSE-trained models fit only high-amplitude low wavenumbers. Errors in neglected low-amplitude high frequencies later feed back into low frequencies.
- Formulation: refinement with diffusion-style denoising steps.
- KS setup: L∈[0.9,1.1]·64, 256 points, Δt∈[0.18,0.22]. Metric: **"high-correlation time"**, when mean Pearson correlation drops below 0.8 or 0.9. Result: about 75 s for the MSE U-Net vs about 100 s for Refiner. Pushforward did not help on KS.
- Cost: about 4x the MSE model.
- Relevance: the protocol (correlation time) and the spectral diagnosis are directly applicable to the KS and Tra decoders. "Refiner" is already a baseline in ACDM.

**D4. Thermalizer: Stable autoregressive neural emulation of spatiotemporal chaos.** Pedersen, Zanna, Bruna. ICML 2025, PMLR 267:48635–48661. [arXiv:2503.18731](https://arxiv.org/abs/2503.18731). Code: [Chris-Pedersen/thermalizer](https://github.com/Chris-Pedersen/thermalizer) (verified; the repo already has a `thermalizer/` dir).
- Systems: **Kolmogorov flow and 2-layer QG (not KS)**. QG is solved pseudo-spectrally (AB3) in PyTorch at 256² and downsampled to 64², with a 2x64x64 state.
- Emulators: UNet (~48M) and DRN (~0.5M), predicting residuals, 450k training trajectories.
- Thermalizer: a diffusion denoiser with a noise-level classifier. When the predicted noise level exceeds s_init, the state is re-noised to that level and denoised down to s_stop. About 140 such interventions per 10k-step QG trajectory.
- Metrics: MSE (MSE≈1 means decorrelated but stable), KE spectra, autocorrelation, time to instability.
- Results: plain emulators blow up at ~2k steps, while thermalised runs are stable to 1e5 steps. QG KE is somewhat low.
- Relevance: **the repo's QG follows this setup**, so this gives a directly comparable protocol and baselines (UNet/DRN ± thermalizer) on identical data. Thermalisation could also be applied to KAE decoded states.

**D5. ACDM benchmark: Benchmarking Autoregressive Conditional Diffusion Models for Turbulent Flow Simulation.** Kohl, Chen, Thuerey. Neural Networks 199 (2026) 108641 (DOI 10.1016/j.neunet.2026.108641 per S2; earlier ICML 2024 workshop listing seen in search). [arXiv:2309.01745](https://arxiv.org/abs/2309.01745) (v3 Dec 2024). Code, data and models: [tum-pbs/autoreg-pde-diffusion](https://github.com/tum-pbs/autoreg-pde-diffusion) (verified).

*Tra data:*
- SU2 with SA-DDES, Re=1e4, Ma 0.50–0.90 in steps of 0.01 (41 sequences).
- RBF-resampled to **128x64**. Fields: vx, vy, ρ, p. Export every 50th solver step with temporal stride 2.
- Normalised to standard normal (velocity by magnitude). The cylinder interior is masked.
- Train: Ma 0.53–0.63 ∪ 0.69–0.90 (33 sequences).
- Test: **Tra_ext Ma 0.50–0.52, T=60. Tra_int Ma 0.66–0.68, T=60. Tra_long Ma 0.64–0.65, T=240** (4 sequences, ~8 shedding periods).

*Inc data:* PhiFlow, train Re 200–900. Inc_low Re 100–180 (T=60), Inc_high 920–1000 (T=60), Inc_var with Re ramping 200→900 over T=250.

*Baselines and metrics:*
- Baselines: ResNet, dilated ResNet, FNO16/32, TF_MGN/TF_Enc/TF_VAE (latent transformers, rollout m=60 with input window k=30), U-Net, U-Net_ut (unrolled m=2–16), U-Net_tn (training noise), Refiner, ACDM, ACDM_ncn.
- Metrics: **MSE and LSiM averaged over the rollout**, 3 training seeds, and 5 samples for probabilistic models. Also spatial and temporal spectra, temporal stability as ‖dx/dt‖ (rate of change), Pearson correlation over time (Inc_var), and % stable runs for very long rollouts.

*Table 1 Tra values (MSE e-3 / LSiM e-1, ext | int; partially garbled in extraction, **verify against the PDF before quoting**):*

| Method | Tra_ext MSE (e-3) | Tra_ext LSiM (e-1) | Tra_int MSE (e-3) | Tra_int LSiM (e-1) |
|---|---|---|---|---|
| U-Net_tn | 1.4 | 1.1 | 1.8 | 1.0 |
| U-Net_ut | 1.6 | 1.1 | 1.5 | 1.0 |
| ResNet_dil | 1.7 | 1.2 | 1.7 | 1.0 |
| ACDM | 2.3 | 1.3 | 2.7 | 1.3 |
| FNO16 | 4.8 | 3.4 | 5.5 | 2.6 |
| Refiner | 5.4 | 2.3 | 7.1 | 3.0 |
| TF_Enc | 3.3 | 1.8 | 6.2 | 2.2 |

*Timing (Appendix C):* inference for a T=1000 sequence: ACDM R=20 takes 194 s, U-Net 9.4 s, TF_Enc 0.6 s, FNO16 2.3 s. Training on Tra takes ACDM 42 h, U-Net 19 h, TF 38–42 h. **Note that the strongest Tra baselines are U-Net with training noise or unrolling, not ACDM.** A KAE comparison that omits U-Net_tn/ut overstates the result.

### E. Ocean-specific (labelled by regime)

**E1. OceanNet** *(regional, operational-style, reanalysis SSH)*. Chattopadhyay, Gray, Wu, Lowe, He. Sci. Rep. 14 (2024) 21181. [arXiv:2310.00813](https://arxiv.org/abs/2310.00813), DOI 10.1038/s41598-024-72145-0. Code: unverified (the guessed repo name was not found).
- Arch: FNO with a **predictor-evaluate-corrector integrator** to suppress autoregressive error growth, plus a **spectral regulariser** against small-scale spectral bias.
- Data: Gulf Stream / NW Atlantic SSH, seasonal (~120-day per my memory) forecasts of Loop Current eddies. Claims ~500,000x lower cost than the numerical model.
- Relevance: the spectral regulariser is analogous to the repo's FFT losses. PEC is a cheap stabiliser.

**E2. Samudra** *(global, climate-scale, emulating the OM4 model)*. Dheeshjith, Subel, Adcroft, Busecke, Fernandez-Granda, Gupta, Zanna. GRL 52(10) 2025, DOI 10.1029/2024GL114318. [arXiv:2412.03795](https://arxiv.org/abs/2412.03795). Code: [m2lines/Samudra](https://github.com/m2lines/Samudra) and [suryadheeshjith/Samudra](https://github.com/suryadheeshjith/Samudra) (both verified).
- Arch: ConvNeXt UNet over multi-depth T, S, SSH, u, v.
- Results: stable for centuries, about 150x faster than OM4, but struggles with the magnitude of forcing trends.
- Relevance: context only. It shows that stable century rollouts are achievable with nonlinear AR models plus careful training, so "stability" alone is not a differentiator for KAE.

**E3. NeuralOM** *(global S2S, 0.5°, 93 ocean variables)*. Gao, Shu, Wu, et al. AAAI 2026. [arXiv:2505.21020](https://arxiv.org/abs/2505.21020). Code: [YuanGao-YG/NeuralOM](https://github.com/YuanGao-YG/NeuralOM) (verified).
- Arch: progressive residual correction + physics-guided graph network. 60-day rollouts with a 13.3% lower RMSE than the best baseline (FourCastNet, CirT) at 60 days.
- Relevance: context only. A different scale and data regime.

**E4. Brettin, Zanna, Barnes (A13)** *(regional, real-ocean SSH)*, and **the repo owner's CT-KAE QG preprint (§0)** *(idealised QG)*. No other continuous-time Koopman ocean forecasting paper was found.

### F. Long-horizon training, stability, QG emulation, invariant measures

**F1. Message Passing Neural PDE Solvers.** Brandstetter, Worrall, Welling. ICLR 2022 (spotlight). [arXiv:2202.03376](https://arxiv.org/abs/2202.03376). Code: [brandstetter-johannes/MP-Neural-PDE-Solvers](https://github.com/brandstetter-johannes/MP-Neural-PDE-Solvers) (verified).
- **Pushforward trick** (memory, consistent with the abstract's zero-stability/domain-adaptation framing): unroll k steps without gradient, then train the last step on the model's own distribution. **Temporal bundling**: predict several steps at once.
- Relevance: a cheaper alternative to backprop through 6–8 RK4 steps. Note, however, that PDE-Refiner found pushforward did not help on KS.

**F2. Towards Stability of Autoregressive Neural Operators.** McCabe, Harrington, Subramanian, Brown. TMLR 2023 (per S2). [arXiv:2306.10619](https://arxiv.org/abs/2306.10619).
- Content: links rollout blow-up to **spectral amplification of high frequencies**. Proposes architectural fixes (frequency-domain damping/normalisation).
- Relevance: diagnostic framework for the decoder side.

**F3. QG learning in pyqg-type models.**
- Ross, Li, Perezhogin, Fernandez-Granda, Zanna, "Benchmarking of machine learning ocean subgrid parameterizations in an idealized model". JAMES 15 (2023), DOI 10.1029/2022MS003258. Code: [m2lines/pyqg_parameterization_benchmarks](https://github.com/m2lines/pyqg_parameterization_benchmarks) / [Cambridge-ICCS/...](https://github.com/Cambridge-ICCS/pyqg_parameterization_benchmarks) (verified). Online evaluation of 148 closures with 19 physical/climatological metrics (spectra, energy budgets, distributional distances). [pyqg/pyqg](https://github.com/pyqg/pyqg) (verified).
- Frezat, Le Sommer, Fablet, Balarac, Lguensat, "A Posteriori Learning for QG Turbulence Parametrization". JAMES 14 (2022), DOI 10.1029/2022MS003124. **A-posteriori (unrolled through the solver) training** beats a-priori training for long-term stability and statistics.
- Relevance: the **metric suite** from Ross et al. (distributional distances of KE and enstrophy spectra, energy transfer) should replace RMSE at multi-year QG horizons.

**F4. Invariant-measure training.**
- Jiang, Lu, Orlova, Willett, "Training neural operators to preserve invariant measures of chaotic attractors". NeurIPS 2023. [arXiv:2306.01187](https://arxiv.org/abs/2306.01187). Uses OT (MMD/Sinkhorn) on summary statistics and contrastive features.
- Schiff et al., "DySLIM". ICML 2024, PMLR 235. [arXiv:2402.04467](https://arxiv.org/abs/2402.04467). Code in google-research/swirl-dynamics (ergodic project; not separately verified). Measure-matching regulariser tested on Lorenz-63, **KS** and Kolmogorov.
- Relevance: provides metrics (MMD/energy distance on attractor statistics) and optional regularisers for KS/QG.

**F5. Chaotic-system forecasting metrics.**
- Pathak, Hunt, Girvan, Lu, Ott, PRL 120 (2018) 024102. KS with a parallel reservoir.
- Vlachas et al., Neural Networks 126 (2020), [arXiv:1910.05266](https://arxiv.org/abs/1910.05266).
- Both standardise **valid prediction time (VPT) in Lyapunov times** (time until normalised RMSE exceeds a threshold, e.g. 0.5; threshold value from memory) and power-spectrum comparison for KS.

**F6. Score-based Data Assimilation (SDA).** Rozet, Louppe. NeurIPS 2023 (per S2). [arXiv:2306.10574](https://arxiv.org/abs/2306.10574). Code: [francois-rozet/sda](https://github.com/francois-rozet/sda) (verified).
- Formulation: a trajectory score from short-segment scores. Observation models are applied zero-shot at inference.
- Experiments: Lorenz-63 and 2D Kolmogorov (memory; not re-verified).
- Relevance: the repo's DA comparator. KAE 4D-Var is deterministic MAP, while SDA gives posterior samples. **Compare with posterior-aware metrics (spread/skill, CRPS) as well as RMSE.**

**F7. Latent-target collapse references (self-supervised and world models).**
- Grill et al., BYOL (NeurIPS 2020, [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)): EMA target plus predictor.
- Chen & He, SimSiam (CVPR 2021, [arXiv:2011.10566](https://arxiv.org/abs/2011.10566)): **stop-gradient is essential**; without it, collapse occurs.
- Bardes, Ponce, LeCun, VICReg (ICLR 2022, [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)): variance hinge + covariance decorrelation, with no stop-grad needed.
- Hansen, Su, Wang, TD-MPC2 ([arXiv:2310.16828](https://arxiv.org/abs/2310.16828); ICLR 2024, venue unverified on the page): latent consistency ‖z'_t − **sg**(h(s'_t))‖² with **SimNorm** latent normalisation, "essential to training stability".
- Kostić et al., DPNets (ICLR 2024 per S2, [arXiv:2307.09912](https://arxiv.org/abs/2307.09912)): Koopman/transfer representations via a VAMP-like covariance-normalised score.

---

## 2. Comparison matrix

| Family (examples) | Formulation | Data needs | Forecast strengths | Long-horizon behaviour | Physical fidelity | Compute (inference) | Impl. complexity | Suitability tra / QG / KS | Published evidence |
|---|---|---|---|---|---|---|---|---|---|
| **Linear DMD / EDMD / BOP-DMD** (C1–C5) | x≈Φe^{Ωt}b; K=YX⁺ | Low (tens of snapshots) | Periodic, limit-cycle flows | Exact exp; decays or persists per Re(ω); BOP can constrain Re(ω)≤0 | Linear modes only; no chaos | Negligible | Very low (PyDMD) | tra: good for shedding · QG: poor · KS: poor | Extensive on wakes; no ACDM numbers |
| **Periodic Koopman (Lange)** (A6) | z_t = cos/sin(ω t), nonlinear decoder | Low–moderate | Very long quasi-periodic | No drift, no decay by construction | Cannot do transients or broadband | Negligible | Low (code exists) | tra: strong baseline · QG/KS: unsuitable | JMLR; energy and fluid examples |
| **Discrete KAE** (A1, A3, A4, A9) | z_{t+1}=Kz_t | Moderate | Short–mid horizons; interpretable spectra | Stable only if ρ(K)≤1; non-normal transients | Mean-reverting on chaos | Negligible latent; decoder-bound | Low | tra ok · QG/KS: statistics degrade | Many small/medium-scale papers |
| **Continuous KAE (this repo, A10)** | dz/dt=Az; exp(τA) | Moderate | Arbitrary-τ queries, cheap DA adjoints | Contractive if sym part ≤0; otherwise unbounded | Same as discrete; dissipates fine scales (QG KE −49%) | Lowest (1 matvec + decoder) | Low–moderate | tra: competitive at long T · QG: stats only · KS: stats only | Owner's preprints only (no peer-reviewed venue found) |
| **Diagonal complex / SSM-style linear latent** (B3, B4) | λ=exp(−e^ν+iθ) or ZOH exp(ΔΛ) | Moderate–high | Long memory, exact stable discretisation | Guaranteed \|λ\|≤1; ring init near unit circle | Same linear-latent limits | O(d) per step | Low | Drop-in for A; all three | Strong on sequence tasks; few PDE-forecast papers |
| **Measure-aware linear latent** (A12 PFNN, F4) | Linear operators + contraction/measure constraints | Moderate–high (long trajectories) | Statistics on chaotic attractors | Designed for invariant measure | Better spectra and statistics | Low | Moderate–high | QG/KS: best-matched · tra: unnecessary | ICLR 2025, NeurIPS 2023, ICML 2024 |
| **Koopman neural operator** (D2) | Encoder→Fourier-space K→decoder | Moderate–high | Mesh-independence, zero-shot | Reported good; no ACDM evidence | Moderate | Low | Moderate (KoopmanLab) | tra/QG/KS all feasible | JCP (per S2) |
| **AR neural operators / U-Nets** (D1, D5) | x_{t+1}=f_θ(x_t) | High | Best short-horizon accuracy on Tra (U-Net_tn/ut) | Needs unrolling/noise; can blow up | High at short T | Low–moderate (O(T) calls) | Low (ACDM repo) | tra: SOTA baseline · QG/KS: blow up ~2k steps without help | ACDM Table 1; Thermalizer |
| **Diffusion / refinement** (ACDM, Refiner, Thermalizer, D3–D5) | Conditional denoising per step | High | Spectral fidelity, uncertainty | Thermalizer: stable to 1e5 steps on QG | Best spectra | High (20–100x U-Net) | Moderate–high (code exists) | tra: ACDM · QG: Thermalizer · KS: Refiner | NeurIPS 2023, ICML 2025, Neural Networks 2026 |
| **Ocean emulators** (OceanNet, Samudra, NeuralOM) | FNO/UNet/GNN AR | Very high (reanalysis / GCM) | Operational-like skill | Samudra: centuries stable | Real-ocean statistics | Moderate | High | Not applicable at this scale (cite as context) | Sci. Rep. 2024, GRL 2025, AAAI 2026 |

---

## 3. Most relevant for this repo (ranked)

**Caveat.** Published RMSE/MSE numbers from different datasets, normalisations, horizons or rollout averaging are **not comparable**. Only numbers produced with the ACDM protocol on Tra/Inc are directly comparable. That protocol is: the same 128x64 normalised fields, the same Ma splits (Tra_ext 0.50–0.52 / Tra_int 0.66–0.68 at T=60, Tra_long 0.64–0.65 at T=240), MSE and LSiM averaged over the full rollout, 3 training seeds. The same applies to the Thermalizer protocol on QG.

1. **ACDM benchmark baselines on Tra/Inc (D5), especially U-Net_tn and U-Net_ut, plus ACDM and FNO.** Models, data and code are public. They are the strongest short-horizon Tra models, and **U-Net_tn beats ACDM on Tra_ext and Tra_int in Table 1**, so the KAE must be compared against them, not only against ACDM. Report MSE and LSiM on all three test sets with 3 seeds, plus spectra and ‖dx/dt‖ stability. Fix the 22.2 vs 22.6 inconsistency in the preprint.
2. **Thermalizer QG protocol (D4).** The repo's QG follows Pedersen et al. Use their UNet/DRN emulators (± thermalizer) as baselines with their metrics: KE spectra, autocorrelation, MSE→1 decorrelation and time to instability. Add Ross et al. (F3) distributional metrics. This turns the "stable but −49% KE drift" result into a comparison on equal footing.
3. **Stable diagonal/structured generator (A10, B3, B4) as an ablation of the repo's own operator.**
   - Option (i): D = −(LLᵀ+εI) with skew S, which is guaranteed contractive.
   - Option (ii): complex-diagonal λ_k = −e^{ν_k} + iω_k (continuous-time) with exact exp, as in LRU/S5 with ring init near the imaginary axis.
   - Cheap to implement. It removes the RK4/exp question and the "not guaranteed stable" weakness at once.
4. **Linear spectral baselines: BOP-DMD/exact DMD on POD coefficients and Lange's periodic Koopman (C1, C4, A6).** Cheap, and essential for Tra/Inc vortex shedding. If a 20-mode BOP-DMD gets close on Tra_long, the KAE's long-horizon advantage is mostly the linear-periodic structure.
5. **PFNN / DySLIM / invariant-measure evaluation for KS and QG (A12, F4).** The literature evaluates linear-latent models on chaos by statistics. Use them as KS/QG baselines or, at minimum, adopt their metrics.
6. **SDA (F6) for DA, plus KNO (D2) as a Koopman-operator baseline.** For DA, compare on identical observation operators and report posterior calibration. SDA gives samples while 4D-Var gives a point estimate, so RMSE alone favours neither fairly. KNO via KoopmanLab is the cheapest "other Koopman" baseline.

---

## 4. Specific notes

### (a) Single RK4 step vs exact exponential for a linear latent generator

**Effective map.** For dz/dt=Az, one classical RK4 step of size h gives z_{n+1}=P4(hA)z_n with P4(x)=1+x+x²/2+x³/6+x⁴/24. This is the degree-4 Taylor truncation of exp(hA), with local error ‖(hA)^5‖/120 (standard numerical analysis). For a linear ODE, RK4 is not a "solver" in any useful sense: the exact solution costs one `torch.linalg.matrix_exp` of a 128x128 matrix per batch, which is cheap and differentiable. For a diagonal parameterisation it costs O(d). This is how S4/S5 (ZOH, exp(ΔΛ)) and LRU (exp parameterisation) discretise.

**Mismatch between training and inference.** When the model is trained through P4(hA) but evaluated with exp(τA), as both preprints state, it is run with a different map. My calculation, |P4(iωh)| for purely oscillatory modes:

| hω | \|P4(iωh)\| | \|P4\|^250 | rel. err vs exp |
|---|---|---|---|
| 0.5 | 0.99990 | 0.974 | 2.6e-4 |
| 1.0 | 0.99391 | 0.217 | 8e-3 |
| 1.5 | 0.94143 | 3e-7 | 6e-2 |
| 2.0 | 0.74536 | 1e-32 | 2.6e-1 |

RK4 becomes unstable for hω > 2√2 ≈ 2.83.

So **RK4 numerically damps oscillatory modes that exp(hA) keeps exactly undamped**. A skew-only generator is energy-preserving under exp but dissipative under P4. Over 250 steps, any mode with hω ≳ 1 (|ω| ≳ 10 at dt=0.1) behaves completely differently under the two maps.

For real eigenvalues, P4(hλ) vs exp(hλ):

| hλ | P4(hλ) | exp(hλ) |
|---|---|---|
| −1 | 0.375 | 0.368 |
| −2 | 0.333 | 0.135 |
| ≈ −2.79 | ≈ 1.0 | 0.06 |

At hλ ≈ −2.79, **stiff decaying modes become neutrally stable or unstable under RK4**. The optimiser can therefore exploit P4 artefacts, for example:
- using large-|ω| modes as learned damping;
- leaving a positive-real-part mode that RK4's damping happens to cancel.

The "close agreement" reported in the preprint holds only if h·ρ(A) ≲ 0.5.

**Audit actions.**
- Compute eig(A) for every trained model (and every Mach via LoRA). Report max h|λ| and max Re(λ).
- Report ‖P4(hA)^n − exp(hA)^n‖ for n up to 250.
- Re-run the evaluations with expm.
- Prefer training with exact matrix_exp. This also makes the "arbitrary τ / variable Δt" claim true by construction: the QG preprint tests 1 h and 10 h, and RK4 at 2x the step leaves the accurate region much sooner.

### (b) Diagonal/complex-eigenvalue parameterisation vs dense skew+sym

**Dense A = S + D, with S skew and D symmetric.** Re⟨z, Az⟩ = zᵀDz, so d/dt‖z‖² = 2zᵀDz. Hence:
- **if D ≼ 0, ‖exp(tA)‖₂ ≤ 1 for all t**, giving no transient growth despite non-normality;
- Re(λ(A)) ≤ λ_max(D).

With D unconstrained, as in this repo, neither holds, and non-normal transient growth (pseudospectra; Colbrook C6) is possible even when all eigenvalues sit in the left half-plane. The minimal fix is D = −(LLᵀ+εI), or −softplus on the eigenvalues of D. This is a Hurwitz/dissipative parameterisation in the spirit of Bevanda et al. (A10) and port-Hamiltonian forms. Caveat: a strictly contractive A drives the latent to 0, i.e. the decoder's mean field. For chaotic QG/KS this predicts a decay of energy, which matches the −49% KE drift seen in the preprint, unless a set of modes sits exactly on the imaginary axis.

**Complex diagonal (Lusch A1 2x2 blocks; LRU/S5 B3–B4).**
- Parameterise λ_k = −exp(ν_k) + iω_k in continuous time (or exp(−e^ν+iθ) in discrete time). Stability is guaranteed, and magnitude and frequency are decoupled, which LRU shows eases Adam optimisation.
- Exact exp costs O(d). Variable Δt is trivial.
- Initialise near the imaginary axis with small damping (LRU ring init with r_max→0.99 helps long range only when stability is enforced) and small phases for long horizons. Add γ-normalisation if magnitudes approach 1.
- Over ℂ, a diagonal complex A loses no generality for diagonalisable A, since the change of basis can be absorbed into the encoder and decoder. LRU makes this argument for the in/out projections.
- It does lose explicit non-normal transient dynamics. Those transients may matter for Tra start-up but are unlikely to matter on the attractor.
- LoRA conditioning on Mach becomes conditioning of (ν_k, ω_k), which is more interpretable: e.g. shedding frequency vs Mach.

### (c) Encoder-target latent losses without stop-gradient: collapse and fixes

**Literature.**
- In siamese and self-predictive learning, predicting an online-encoded target **without stop-gradient collapses** (SimSiam). BYOL uses EMA target + predictor, TD-MPC2 uses sg on targets plus SimNorm normalisation, and VICReg avoids sg with variance and covariance terms.
- In Koopman learning, Mondal et al. report that latent consistency terms promote collapse in their setting. Cheng et al. (ICLR 2026) show collapse onto a few dominant modes and fix it with a von Neumann-entropy regulariser. VAMPnets/DPNets avoid it through covariance whitening.
- Classic KAEs (Lusch, Azencot, Otto–Rowley) do train without sg. **The reconstruction loss prevents total collapse**, because z must remain decodable.

**The residual failure mode in this repo's setup is scale/rank shrinkage.** The latent losses include:
- trajectory matching to encoded futures (not detached);
- forward/backward consistency;
- a second-difference smoothness penalty;
- a latent FFT loss.

All of these are quadratic in z, so scaling the encoder output by c<1 (with the decoder's first linear layer scaling by 1/c) reduces them by c² at no reconstruction cost. Spectral-normed encoder layers do not prevent this, because the final projection and attention pooling are free. The smoothness penalty also biases toward low-frequency or degenerate latents, which conflicts with oscillatory Koopman modes.

**Diagnostics.** Track per-dimension latent std, effective rank (entropy of the covariance eigenvalues), and the ratio of latent loss to ‖z‖².

**Fixes, cheapest first.**
1. Normalise latent losses by target variance (Otto & Rowley-style relative error) or by batch ‖z‖².
2. Stop-grad on the encoded-future target, as in TD-MPC2. Keep gradients through the rollout and pixel losses, which already anchor the encoder.
3. EMA target encoder (BYOL).
4. VICReg-style variance hinge (std ≥ 1 per dimension) + covariance decorrelation, or whitening as in VAMPnets and the owner's QG preprint.
5. Von Neumann entropy (A11).

Run an ablation with sg on vs off.

### (d) Evaluation for chaotic systems (KS, QG)

**Trajectory skill.**
- Report **valid prediction time in Lyapunov times** (Pathak 2018; Vlachas 2020): time until normalised RMSE exceeds a threshold. Compute λ_max for the repo's KS domain length and the QG configuration rather than quoting a literature value, since λ_max depends on L.
- Alternatively, report **high-correlation time** (corr < 0.8 / 0.9; PDE-Refiner).
- Beyond 1–3 Lyapunov times, RMSE saturates at √2·σ for a decorrelated but on-attractor state, while a model that **collapses to the mean scores a lower RMSE (σ)**. The QG preprint's RMSE 0.938 < 0.993 with ACC ≈ 0 is this regime. **Long-horizon RMSE must not be used as evidence of skill.**

**Statistics.**
- Time-averaged KE and enstrophy spectra (QG), and spatial power spectra (KS).
- Autocorrelation functions.
- PDFs of q, u and vorticity.
- KE and enstrophy time series and drift.
- Distributional distances (Wasserstein, MMD, energy distance) on these quantities, as in Jiang et al. 2023, DySLIM and Ross et al. 2023.
- For QG, use the Thermalizer protocol (spectra, autocorrelation, MSE→1 behaviour, time to instability) on identical data.

**Ensembles and DA.** Use CRPS/spread-skill when comparing to probabilistic models (ACDM, SDA, Thermalizer).

**Theory caveat.** On a mixing attractor, the Koopman operator has only trivial eigenfunctions plus continuous spectrum (HAVOK; Lusch's continuous-spectrum treatment; Colbrook). A finite linear latent therefore either decays to the mean or is quasi-periodic. Its long-term statistics can only be right if the decoder maps quasi-periodic latents onto the attractor, so claims should be framed statistically.

---

### Sources (primary pages opened or checked)
arXiv: 2602.02832, 2603.05560, 2309.01745 (v3 HTML), 2003.02236, 1712.09707, 2403.12335, 2303.06349 (PDF), 2503.18731 (v2 HTML), 2308.05732 (v2 HTML), 2202.03376, 2301.10022, 2306.10574, 2306.11941, 2510.13025, 2503.14702, 2602.05416, 2310.16828. NOAA repository record for Brettin et al. (GRL 2025). Semantic Scholar batch metadata for all arXiv IDs listed. GitHub API existence checks for all code links marked "verified".
