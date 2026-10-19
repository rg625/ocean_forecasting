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

## 4. CT-KAE pilots (6k steps, one change each vs P0; validation only)

*(filled in below as runs finish)*
