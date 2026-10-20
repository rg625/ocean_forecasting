# Experiment results (TRA benchmark)

Every number here was measured on this branch unless it is marked *paper*. Reproduction
commands are in `IMPLEMENTATION_PLAN.md` §4.

Hardware: one RTX 4090, shared throughout with an unrelated job (`diagnostics.py`, ≈ 99%
utilisation, 3.5 GB), so wall-clock figures are pessimistic.

**Metric** (identical for every row; `audit_report/scripts/benchmark_suite.py::score`):
- MSE ×10⁻³ in physical units, obstacle zeroed, averaged over the 4 fields, H, W and the
  **predicted frames** (frames 2..T−1). The paper averages over all frames including the
  conditioning frames, i.e. its values are (T−k)/T of ours (≈ 0.967× for ACDM at T = 60).
- ACC: anomaly correlation against the training climatology.
- E: predicted / true fluctuation energy about the true time mean. E < 1 means regression to
  the mean.
- ± is the std over the 3 official seeds (× 5 samples for diffusion models).

## 1. Verified baselines (official ACDM-paper checkpoints, re-scored)

Source: `results/benchmark_suite_baselines.json`. Provenance:
`pretrained_models/models_tra/*_{00,01,02}` sampled with the official script; see
`LITERATURE_REVIEW_V2.md` §ACDM verification. The paper's Tra_int ACDM value of 2.7 (its
convention) converts to ≈ 2.79 in ours, which matches.

| Model | interp (T=60) | extrap (T=60) | longer (T=240) | longer E@238 |
|---|---|---|---|---|
| U-Net_ut (`unet-m8`) | **1.53 ± 0.16** | 1.66 ± 0.05 | 22.37 ± 1.02 | 1.17 |
| U-Net_tn (`unet-m2-noise0.01`) | 1.83 ± 0.17 | **1.42 ± 0.07** | 22.56 ± 1.60 | 1.18 |
| Dil-ResNet | 1.78 ± 0.68 | 1.71 ± 0.09 | 22.14 ± 1.34 | 1.32 |
| ACDM (run A / run B, independent sampling) | 3.02 ± 0.80 / 2.81 ± 0.71 | 2.22 ± 0.49 / 1.99 ± 0.44 | 22.82 ± 1.59 / 22.51 ± 1.93 | 1.12 / 1.11 |
| ACDM_ncn | 3.00 ± 0.48 | 4.25 ± 1.86 | 22.95 ± 3.22 | 1.20 |
| FNO16 | 5.72 ± 1.14 | 4.98 ± 0.22 | 20.95 ± 0.42 | 0.97 |
| TF_MGN | 6.46 ± 0.11 | 3.85 ± 0.43 | 19.04 ± 0.54 | 0.65 |
| Training climatology (constant field) | 16.76 | 17.39 | **15.83** | 0.17 |
| Persistence | 23.44 | 17.64 | 28.78 | 1.07 |
| POD(128) + 2-frame linear map (POD-DMD2) | 6.06 | 3.86 | 13.41 | ≈ 0.05 |

**Reading Tra_long.**
- No model beats a constant climatology on 240-step MSE except the mean-regressing POD
  model.
- All neural baselines keep realistic fluctuation energy (E ≈ 1), and MSE punishes them for
  it.
- The ACDM paper reports **no** Tra_long numbers; every Tra_long value is a re-evaluation.
- Long-horizon claims below are therefore judged on MSE **and** E **and** ACC together.

## 2. E0: the existing KAE checkpoint, reproduced (REPRO-04)

`results/E0_legacy_run20260821_*.json`, from `scripts/eval_legacy_tra.py`: current code,
strict load, same scorer.

| Checkpoint | interp | extrap | longer | ACC < 0.6 at lead (interp / longer) | longer E@238 |
|---|---|---|---|---|---|
| run-20260821 `epoch_199` (forecast checkpoint) | 11.43 | 2.12 | 18.62 | 27 / 28 | 0.27 |
| run-20260821 `best_model` (DA checkpoint) | 10.76 | 4.16 | 24.73 | 30 / 29 | 1.00 |
| *saved npz (24 Aug), realigned* | *6.20* | *1.55* | *16.91* | — | — |

The saved prediction files, and therefore the audit's earlier "6.37 / 1.59 / 16.91" and the
paper's KAE numbers, are **not reproducible**:
- No checkpoint on disk regenerates them. The closest, `epoch_199`, which has the same
  weights and config as `final_model.pth`, differs by 0.07 at the first predicted frame,
  in both fp32 and TF32.
- The model code was rewritten in `a5f4796` after that evaluation.
- Only the reproducible rows are used as the KAE baseline from here on.

## 3. Validation references (held-out Ma 0.60 / 0.75; used for every decision)

`results/val_baselines.json`. These are the same windows and metric as `train_ctkae.validate`:
60-step rollouts from 5 starts and 240-step rollouts from 3 starts per simulation.

| Baseline | val MSE60 | val MSE60, first 10 leads | val MSE240 |
|---|---|---|---|
| Persistence | 25.17 | 15.73 | 28.05 |
| Training climatology | 16.35 | 16.18 | 16.76 |
| POD-DMD2 (r = 128, train sims only) | 7.94 | 1.16 | 13.40 |

## 4. CT-KAE pilots (6k steps, seed 0, one change each vs P0; validation only)

`results/pilots_summary.json`, logs in `koopman_autoencoder/model_outputs_ctkae/P*/log.jsonl`.

Setup:
- P0 = block generator, full Mach conditioning, K: 10 → 60, 8 decoded leads,
  w_lat = 0.1, stride 2.
- Each pilot took about 31 min at 0.31 s/step on the shared GPU.
- Validation is on held-out Ma 0.60 and 0.75.
- Table columns: best MSE60 is the minimum over checkpoints every 1k steps; final is at 6k
  steps. E60 and E240 are the energy ratios at 60 and 240 steps, final checkpoint.
- References on the same windows: POD-DMD2 7.94 (60) and 13.40 (240); climatology 16.35 and
  16.76.

| Pilot | Change vs P0 | Best MSE60 (step) | Final MSE60 | Final MSE60, first 10 | Final E60 | Final E240 |
|---|---|---|---|---|---|---|
| P0 | block generator (reference) | 4.46 (2k) | 4.61 | 0.60 | 0.98 | 0.78 |
| P1 | dense generator (legacy form) | 4.88 (2k) | 5.35 | — | 0.96 | 0.71 |
| P2 | hurwitz generator (J − R) | **4.20** (3k) | 4.68 | — | 0.97 | 0.80 |
| P3 | no conditioning (FiLM, generator, μ all off) | 4.56 (3k) | 4.86 | — | 0.96 | 1.08 |
| P4 | + amplitude loss (w_amp = 0.1) | 4.34 (3k) | **4.61** | — | **0.99** | 0.68 |
| P5 | training horizon K = 10 (not 60) | 5.94 (4k) | 6.06 | — | 0.84 | — |
| P6 | strides {1, 2, 3} (Δt 0.05 / 0.10 / 0.15) | 4.54 (3k) | 4.68 | 0.62 | 0.97 | 0.83 |

Conclusions (single seed: differences < ~0.3 are within noise and need the multi-seed runs):

1. **Training horizon is the dominant lever.** K = 10 gives +30% MSE60 and a lower energy
   ratio. This is direct evidence for the audit's DYN-01 diagnosis: the legacy 8-step
   objective taught the model to hedge by damping.
2. **The stable generators (block, hurwitz) beat the unconstrained dense form.** Removing
   the possibility of growth does not cost accuracy.
3. **Every pilot keeps E60 ≥ 0.96 and E240 0.7–1.1.** The legacy model keeps 0.25 at 240.
   So the redesign removes the mean regression at the training horizon.
4. **Conditioning has only a small effect on this validation set.** The held-out Mach values
   sit between training neighbours 0.01 away. Whether conditioning matters must be judged
   on the wide-gap `interp` and the `extrap` test cases (§6).
5. **Amplitude loss and stride augmentation cost nothing measurable.** Stride augmentation
   adds genuine Δt generalisation, tested in §7.
6. **MSE60 bottoms out around 2–3k steps and creeps up slightly afterwards while E rises.**
   Under MSE, keeping amplitude costs error once phase drifts, which is the same trade-off
   the neural baselines show on Tra_long.

**Decision.** Full runs use the block generator with amplitude loss and strides {1, 2, 3}
(3 seeds), the hurwitz generator with strides {1, 2, 3} (2 seeds), 12k steps each, selected
on val_mse60. The corrected legacy model is retrained on the same split as E1.

## 5. Full runs: validation (12k steps, strides {1,2,3}, selection on val_mse60)

`results/full_runs_val_summary.json`.

| Config | Seeds | Best val MSE60 (step) | Val MSE60, first 10 | E60 at best |
|---|---|---|---|---|
| block + w_amp 0.1 | 0 / 1 / 2 | 4.04 (3k) / 4.33 (4k) / 4.09 (4k); **mean 4.15** | 0.72 / 0.66 / 0.68 | 0.82 / 0.91 / 0.85 |
| hurwitz | 0 / 1 | 4.37 (3k) / 4.50 (4k); mean 4.43 | 0.91 / 0.85 | 0.82 / 0.89 |

All three block seeds beat both hurwitz seeds on validation, so **block is the selected
formulation**. In every run the best val MSE60 occurs at 3–4k steps. Later checkpoints keep
more energy (final E60 0.97–0.99) at slightly higher MSE.

## 6. Test benchmark (pre-registered checkpoint: best.pth by val_mse60)

Sources: `koopman_autoencoder/model_outputs_ctkae/*/eval/{best,last}.json` (scored with
`benchmark_suite.score`, ground truth asserted equal to the ACDM dicts),
`results/E1_legacy_heldout_best.json` and `results/benchmark_suite_baselines.json`.
MSE ×10⁻³, mean ± std over seeds.

| Model | interp | extrap | longer | longer E@238 | longer: first lead ACC < 0.6 |
|---|---|---|---|---|---|
| **CT-KAE block** (3 seeds) | 4.07 ± 0.29 | 2.18 ± 0.14 | 16.5 ± 2.0 | 0.81 / 0.78 / 1.73 | 43 / 48 / 46 |
| CT-KAE hurwitz (2 seeds) | 4.65 | 2.23 | 13.9 | 0.24 / 0.36 | 48 / 43 |
| CT-KAE block, last.pth (no early stopping, 3 seeds) | 5.59 ± 0.29 | 2.56 ± 0.23 | 18.1 ± 0.5 | 0.62 / 0.60 / 0.53 | 45 / 48 / 44 |
| E1: corrected legacy KAE, same split, ~2 h | 8.18 | 3.83 | 22.77 | 0.66 | 31 |
| E0: legacy checkpoint, reproducible | 11.43 | 2.12 | 18.62 | 0.27 | 28 |
| ACDM (official, 3 seeds × 5 samples; two sampling runs) | 2.81–3.02 | 1.99–2.22 | 22.5–22.8 | 1.11–1.12 | 50–52 |
| U-Net_ut (official) | **1.53** | 1.66 | 22.37 | 1.17 | 54 |
| U-Net_tn (official) | 1.83 | **1.42** | 22.56 | 1.18 | 56 |
| Training climatology | 16.76 | 17.39 | 15.83 | 0.17 | 1 |
| POD-DMD2 (r = 128) | 6.06 | 3.86 | 13.41 | ≈ 0.05 | 27 |

**Verdict against ACDM**, a fair comparison on identical cases with an identical scorer:

- **interp:** the CT-KAE does **not** beat ACDM (4.07 ± 0.29 vs 2.81–3.02).
- **extrap:** it is **on par** (2.18 ± 0.14 vs 1.99–2.22, within the spread of two ACDM
  sampling runs).
- **longer:** it has **lower MSE than ACDM** (16.5 vs 22.5), but part of that is
  amplitude hedging. E@238 is about 0.8 for two seeds, and the correlation horizon is
  shorter than ACDM's (43–48 vs 50–52 steps). This is not a demonstration of better
  long-horizon dynamics.
- The official U-Net_ut and U-Net_tn baselines beat both ACDM and the CT-KAE on interp and
  extrap.

**Disclosed differences:**
- The CT-KAE trains on 30 Mach values; Ma 0.60 and 0.75 are held out for validation.
  ACDM and the U-Nets train on 32 or 33.
- The CT-KAE is deterministic; ACDM is scored per sample.

**Improvement over the legacy model:** interp 11.43 → 4.07 (2.8×), longer 18.6 → 16.5, with
the energy ratio at 238 steps going from 0.27 to about 0.8. Against E1 (the legacy
architecture retrained on the same split with all fixes): 8.18 → 4.07 on interp and
3.83 → 2.18 on extrap.

## 7. Diagnosis of the remaining gap: a reconstruction floor, not the dynamics

MSE ×10⁻³ per lead (test; CT-KAE block seed 0, best.pth):

| Lead | 1 | 5 | 10 | 20 | 30 | 40 | 58 |
|---|---|---|---|---|---|---|---|
| ACDM, interp | 0.03 | 0.12 | 0.29 | 0.90 | 2.35 | 4.42 | 8.46 |
| U-Net_ut, interp | 0.02 | 0.06 | 0.11 | 0.31 | 1.12 | 2.20 | 4.90 |
| CT-KAE, interp | **0.66** | 0.75 | 1.00 | 2.34 | 3.48 | 4.59 | 8.71 |
| ACDM, extrap | 0.02 | 0.09 | 0.24 | 0.78 | 1.57 | 3.03 | 6.51 |
| CT-KAE, extrap | **0.31** | 0.33 | 0.51 | 0.91 | 1.80 | 2.90 | **5.43** |

The CT-KAE error is almost flat over leads 1–5 and 20–30× above ACDM at lead 1. This is
the signature of an autoencoder reconstruction floor. At long leads it matches ACDM on interp
and beats it on extrap, so the continuous-time dynamics are competitive. The latent
bottleneck (d = 256) reconstructs worse than a 128-mode linear POD (projection floor
≈ 0.46·10⁻³ in these units).

**Next experiment:** representation pilots R0–R3, run in a queue:
- latent dimension 512 and 1024;
- reconstruction weight ×5;
- all on the selected configuration, compared on validation only.

## 8. Ablations on test (single seed, analysis only, not used for selection)

| Pilot (6k steps) | interp | extrap | longer | Reading |
|---|---|---|---|---|
| P0: block, Δt 0.1 only | 4.40 | **1.71** | 15.58 | — |
| P3: P0 without any Mach conditioning | 5.43 | 2.04 | 18.94 | Conditioning helps on the wide-gap / extrapolation cases (−19% interp, −16% extrap) even though validation barely showed it |
| P5: P0 with a 10-step horizon | 5.18 | 2.44 | 15.93 | Long-horizon training helps on test too |

## 9. Zero-shot generalisation

**Temporal resolution.** P0 was trained only at Δt = 0.1. Queried at Δt = 0.05 on
`gt_*_subsample_1` (0.05-spaced truth; the input window keeps 0.1 spacing):

| Case | Half-step leads (times never trained on) | Whole-step leads |
|---|---|---|
| interp | 4.408 | 4.400 |
| extrap | 1.720 | 1.707 |

Exact propagation generalises to unseen time offsets at no loss.

**Mach number.** The test cases themselves are zero-shot in Mach: interp 0.66–0.68 lies in a
training gap 0.06 wide, and extrap 0.50–0.52 lies below the training range. Output
sensitivity to the conditioning value (P0; rms change versus using the true Ma on an interp
trajectory):

| Ma used | 0.50 | 0.55 | 0.60 | 0.66 (true) | 0.70 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|---|
| rms change | 0.57 | 0.43 | 0.25 | 0 | 0.16 | 0.42 | 0.61 |

The response is smooth and monotonic in |ΔMa|. The no-conditioning model (P3) is exactly 0
everywhere, as expected.

## 10. Representation pilots (6k steps, seed 0, validation only)

`results/repr_summary.json`. Base configuration: block generator, w_amp = 0.1, strides {1,2,3}.

| Run | Change | Best val MSE60 (step) | First-10 at end | Final E60 | Final MSE240 | Peak mem |
|---|---|---|---|---|---|---|
| R0 | latent 256 (reference) | 4.48 (2k) | 0.58 | 0.97 | 16.92 | 12.5 GB |
| R1 | latent 512 | 4.12 (2k) | 0.56 | 0.99 | 16.55 | 12.8 GB |
| R2 | latent 1024 | 4.50 (2k) | **0.49** | 0.97 | 16.51 | 13.4 GB |
| R3 | w_rec = 5 | 4.40 (6k) | 0.53 | 0.95 | 16.89 | 12.5 GB |

**Neither a larger latent nor a heavier reconstruction weight closes the floor.** At most
they lower the early-lead error by about 15%, and the change in MSE60 is within single-seed
noise. The first-step gap to ACDM is 20–30×. The bottleneck is therefore the encoder/decoder
design and the joint training, not the latent width.

## 11. Follow-up (20 Oct): ACDM training setting, mu ablation, curriculum, data assimilation

Test MSE x1e-3. Single seed per row unless stated. Evaluations are in
`koopman_autoencoder/model_outputs_ctkae/<run>/eval/`.

| Model | interp | extrap | longer | Lead-1 interp | Energy at 58 (interp) |
|---|---|---|---|---|---|
| ACDM setting (2->1, 33 sims, batch 64), with mu (`A_mu_s0`, last) | 10.67 | 7.28 | 27.58 | 0.08 | 0.59 |
| ACDM setting, without mu (`A_nomu_s0`, last) | 12.30 | 6.06 | 24.83 | 0.08 | 1.69 |
| 2->60, with mu (`F_block_s0`, best) | 3.84 | 2.23 | 14.96 | 0.66 | 0.74 |
| 2->60, without mu (`N_nomu_s0_20261020_181054`, best) | 3.86 | 1.90 | 15.25 | 0.67 | 0.75 |
| Curriculum 1-step -> 60 (`C_curr_s0`, best) | 4.45 | 1.81 | 14.50 | 0.49 | 0.68 |

- **ACDM's 2->1 setting is unsuitable for a linear latent model.** Lead-10 error is 3.08
  against 1.00 for 2->60 and 0.29 for ACDM. It does show that the reconstruction floor
  comes from the multi-step objective competing with reconstruction.
- **mu has no measurable effect** in the 2->60 setting; the results are within the spread of
  the 3 mu seeds. It can be dropped.
- **The curriculum** gives better extrapolation and lead-1 error, slightly worse
  interpolation (single seed).
- **Aborted or crashed runs**, marked with ABORTED files: `A_mu_s1` was dropped by decision;
  `N_nomu_s0_20261020_162641`, `N_nomu_s1` and `C_curr_s1` crashed with out-of-memory while
  sharing the GPU with DA.

**Data assimilation v2** (`results/da_v2_summary.json`; `scripts/run_da_v2.sh`; KAE slot only,
same DA code for every model; learning rate tuned on gt_extrap over 0.03-1.0). Mean relative
error over each sweep's points:

| Sweep | Original KAE | CT-KAE curriculum | CT-KAE 2->60 | CT-KAE ACDM setting |
|---|---|---|---|---|
| df9 delta_l / noise / sparsity / misspec / n_obs | 0.085 / 0.078 / 0.086 / 0.078 / 0.079 | 0.094 / 0.094 / 0.100 / 0.099 / 0.095 | 0.130 / 0.129 / 0.108 / 0.132 / 0.126 | 0.22-0.31 |
| df1 delta_l / noise / sparsity / misspec / n_obs | 0.027 / 0.028 / 0.031 / 0.028 / 0.026 | 0.027 / 0.028 / 0.037 / 0.029 / 0.027 | 0.032 / 0.033 / 0.042 / 0.034 / 0.031 | 0.15-0.27 |
| delta_f sweep / single obs (canonical) | 0.070 / 0.089 | 0.074 / 0.091 | 0.094 / 0.105 | 0.19 / 0.19 |
| Decoder floor / encode-init / climatology-init | 0.041 / 0.074 / 0.079 | 0.032 / 0.096 / 0.102 | 0.041 / 0.131 / 0.130 | 0.014 / 0.208 / 0.221 |
| Tuned learning rate | 0.3 | 1.0 (grid top) | 1.0 (grid top) | 1.0 (grid top) |

**The original KAE remains the best DA model.** The curriculum CT-KAE ties it at df1 and is
15-25% behind at df9. The gap is in fitting observations 9-25 steps ahead, not in
representation.

Superseded: `model_outputs_ctkae/da/` (v1). Its "canonical" sweeps actually ran at df9,
because `--pin-delta-f` defaults to 9, and its learning-rate grid was too narrow.
