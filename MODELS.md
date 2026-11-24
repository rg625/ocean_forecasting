# Which model for which system

Decided November 2026. One model per system; everything else in the repository is a baseline or
an ablation.

| System | Model | Why |
|---|---|---|
| Kuramoto–Sivashinsky (KS) | Original KAE (`models/autoencoder.py`), latent 512 | Best clean δf = 9 DA (0.015 vs 0.030 for the best CT-KAE) and better forecasts than every CT-KAE variant |
| Transonic flow (Tra) | CT-KAE back001 (`models/ct_kae.py`) + LPM mirror readout | Best δf = 9 DA (0.088 vs 0.103 for the original KAE, same protocol) and lowest long-rollout MSE |
| Incompressible flow (Inc) and other 2-D flows | CT-KAE (as Tra) | Not yet available: `models/ct_data.py` loads Tra only; an Inc loader and a trained model are needed |

## KS: original KAE

- Train: `python train.py --config-path configs --config-name experiment/ks/continous_linear_128 experiment.ks.model.latent_dim=512` (see `run_latent_sweep.sh`).
- Checkpoint used in the paper: `koopman_autoencoder/model_outputs_ks_latentdim/dz_512/run-20260906_140222` (not in git).
- The encoder's LayerNorm (`latent_norm`, default on) must stay on: every published KAE checkpoint was trained with it.
- DA: `data_assimilation/ks/` (warm start, revised objective by default).

## Tra: CT-KAE back001

- Config: `koopman_autoencoder/configs/ctkae/tra_back001.yaml` (fully resolved; identical to the
  model and training settings of the `AB_back001` runs).
- Train: `python train_ctkae.py --config=configs/ctkae/tra_back001.yaml seed=0` (the `=` is
  required: with a space the file is ignored and the defaults are used).
- Checkpoints (not in git): `model_outputs_ctkae/AB_back001_s{0,1,2}_*` in the
  `ocean_forecasting_impl` folder; use `last.pth`.
- Forecast readout: LPM mirror average, `koopman_autoencoder/models/readout.py` (`lpm`).
- DA: `python -m data_assimilation.tra.exp_tra --kae-kind ctkae --kae-run RUN --kae-ckpt last.pth --kae-mirror --kae-avg lpm`.
  Not yet run with the revised objective's background (`--kae-objective bg` falls back to the
  unweighted misfit for the CT-KAE).

## History

Results, reports, figures and the research scripts (`audit_report/`) of the research branches
are not on `master`. They are kept under
the tags `archive/*`, e.g. `archive/impl-ct-kae` (CT-KAE study, KS matched runs) and
`archive/rares-updated_model`.
