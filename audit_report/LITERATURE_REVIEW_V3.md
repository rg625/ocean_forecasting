# Literature review v3: reducing long-horizon error in the CT-KAE (2026-10-22)

Question: the fair-horizon CT-KAE (2 -> 8, latent 256) trails the stabilised U-Net mainly
through error that grows with lead time. What does the literature do about it, and what
fits our constraints?

Constraints:
- exactly linear latent dynamics;
- continuous time and zero-shot time steps;
- Mach conditioning;
- forecasting-only training;
- end-to-end;
- about 2 -> 8 supervised frames;
- efficient latent.

Full source-verified write-ups (about 47 papers, constraint table per method):
- [literature/v3_koopman_methods.md](literature/v3_koopman_methods.md): Koopman and linear-latent methods.
- [literature/v3_surrogate_remedies.md](literature/v3_surrogate_remedies.md): general neural-PDE remedies.

## 1. Framing: our model does not accumulate error the usual way

Most long-rollout remedies target autoregressive drift: the model consumes its own
imperfect output and wanders off the training distribution. That covers pushforward,
input-noise training, unrolling and Thermalizer-type test-time correction. Our encoder
only ever sees true frames, and every lead is one exact matrix exponential, so that
failure mode does not exist here.

What grows with lead is model bias. Excess error over the U-Net, interpolation
(`results/phase_drift.json`):

| Lead | Outside the true trajectory's POD span | POD modes 3-10 | Shedding pair | Shedding-pair phase error |
|---|---|---|---|---|
| 30 | 49% | 32% | 8% | — |
| 58 | 53% | 13% | 33% | 26-37 deg (U-Net 14 deg) |

The shedding frequency itself is accurate (0.2-1.5%).

## 2. Same-benchmark evidence (ACDM paper, confirmed by our own re-scoring)

Our re-scoring of the official checkpoints (`results/benchmark_suite_baselines.json`,
MSE x1e-3, interp / extrap):

| Model | interp | extrap |
|---|---|---|
| U-Net, unrolled 8 steps | 1.53 | 1.66 |
| U-Net, input noise | 1.83 | 1.42 |
| ACDM | 3.02 | 2.22 |
| **CT-KAE 2 -> 8 (ours, 3 seeds)** | **3.49** | **1.98** |
| Latent transformers TF_Enc / TF_MGN / TF_VAE | 6.4-7.4 | 3.4-4.1 |
| PDE-Refiner | 7.50 | 5.61 |

- Our model is already the best latent model on Tra. Only the stabilised pixel-space
  U-Nets are clearly ahead.
- Spectral refinement (PDE-Refiner) is strongly negative on this exact benchmark.
  Sobolev and spectral losses were also negative or neutral for MSE elsewhere
  (PDE-Refiner paper on KS; Chattopadhyay et al.).

## 3. Checks we ran on the existing model (no retraining; `scripts/infer_ablate.py`)

- **Damping is not the culprit at inference.** Scaling every oscillator's damping by
  1 / 0.5 / 0.25 / 0 gives:

  | Damping scale | interp | extrap | longer | Energy at step 58 (interp) |
  |---|---|---|---|---|
  | 1 | **3.43** | **1.67** | **16.8** | 0.94 |
  | 0.5 | 3.58 | 1.74 | 17.2 | 1.00 |
  | 0.25 | 3.68 | 1.79 | 17.5 | 1.04 |
  | 0 | 3.81 | 1.85 | 17.9 | 1.07 |

  The learned damping is MSE-optimal hedging of uncertain components. The modes that
  carry most of the latent are already nearly neutral (gamma about 0.005). This
  demotes the Koopman review's top recommendation (remove damping) unless a
  training-time variant behaves differently.
- **The dynamics contain no harmonics.** On the test trajectories, the share of
  temporal latent variance at or above 1.9x the shedding frequency is 0.5% (extrap)
  and 0% (longer). Nothing sits at 3x or above. Harmonics of the wake, which Koopman
  theory places on the lattice n * omega_shed (Bagheri, JFM 2013), are produced only by
  the decoder's nonlinearity. Snapping frequencies post hoc cannot test this, because
  the model was not trained with that structure; a lattice must be trained.

## 4. Ranked plan (all keep linear latent, continuous time, Ma, forecasting-only, 8 frames)

1. **Harmonic-lattice generator.** Put a bank of oscillator pairs at
   omega = n * omega_1(Ma), n = 1..N, nearly neutral, with omega_1(Ma) learned and
   initialised from the training-data shedding frequency. Keep about 25-50% of the
   oscillators free for content off the lattice.
   - Both reviews rank it first.
   - Grounded in limit-cycle Koopman theory (Bagheri 2013; Mauroy-Mezic) and
     Fourier-to-Koopman (Lange, Brunton, Kutz, JMLR 2021).
   - Targets the POD 3-10 share and phase coherence between fundamental and harmonics.
   - Risk: no published Koopman autoencoder validates it, and the transonic wake may be
     partly quasi-periodic (hence the free bank).
2. **Multi-start temporal consistency (tcKAE, Nayak et al. 2024) plus forward-backward
   consistency (Azencot et al., ICML 2020), inside the 8-frame window.** Encode
   windows starting at frames 0, 2 and 4, and penalise disagreement of their
   propagated latents. Optionally add a backward-prediction term.
   - Keeps propagated latents on the encoder's manifold, so it targets the
     outside-span share.
   - Evidence: cylinder flow, 1.62% -> 0.87%, but in small-data regimes. Expect a
     moderate gain.
3. **Within-window distribution matching (DySLIM-style MMD at the supervised leads
   only).**
   - On Kolmogorov flow it stopped collapse toward the mean.
   - Likely helps energy and correlation more than MSE.
4. **Grey area, the user's decision: stride-randomised 8-frame supervision.** Same frame
   count, but spread over up to about 4x the physical time. The frequency gradient
   grows with time span (Lange et al.), which targets phase drift. It stretches the
   fair-horizon rule.

## 5. Ruled out, with reasons

| Remedy | Why it is ruled out |
|---|---|
| Pushforward, input noise, unrolling, Thermalizer | Address autoregressive drift, which we do not have. |
| PDE-Refiner, spectral and Sobolev losses | Negative on Tra or for MSE. |
| Generative or diffusion decoders | Fix spectra but not MSE (Oommen et al. 2025); break efficiency. |
| KNF / Koopa local operators, Hankel embeddings, Mori-Zwanzig memory | Need a lookback longer than 2 frames. |
| HAVOK forcing | Unknown at forecast time. |
| Periodic re-encoding (Fathi et al.) | Breaks the single closed-form propagation; diagnostic use only. |

## 6. Honest expectation

No paper fixes exactly our failure mode at our data scale. A realistic target for items
1-2 is part of the gap to ACDM on interpolation (3.49 -> about 3.0). Nothing found
suggests a linear-latent model reaching the stabilised U-Net (1.53) under these
constraints.
