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
