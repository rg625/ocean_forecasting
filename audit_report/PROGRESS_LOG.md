# Audit progress log

## Repository and environment
- Main checkout: `/home/rg625/mnt/ocean_forecasting`, branch `rares/updated_model` @ `79713ce`.
  It has 24 modified tracked files (configs, DA scripts, notebook, autoreg_pde_diffusion) and many
  untracked files. **It was not modified.**
- Audit worktree: `/home/rg625/mnt/ocean_forecasting_audit`, branch `audit/kae-research-audit`,
  created from `79713ce`. Your uncommitted koopman_autoencoder config edits are snapshotted in
  `77ef4c7`.
- Hardware: RTX 4090, shared during the whole audit with another job (`diagnostics.py`, ~99%
  utilisation). Disk at 100% (13 GB free), so no large files were written. 125 GB RAM, 32 cores.

## Stages
| Stage | Status | Main outputs |
|---|---|---|
| 1 Locate repo / baseline | done | this log, `77ef4c7` |
| 2 Map pipeline | done | FINAL_AUDIT.md §2–3 |
| 3 Math and data audit | done | §4–6, `results/spectrum_*.json`, data checks in §5 |
| 4 Reproduce bugs | done | `tests/test_audit_bugs.py` (10 fail on original) |
| 5 Literature | done | `LITERATURE_REVIEW.md` (delegated) |
| 6 Profile | done | `results/smoke_profile_*.json` |
| 7 Fix high-confidence issues | done | commits `0bfe883`..`175c7a4`; 10/10 tests pass |
| 8 Corrected baseline | **not run** | needs retraining (E0); about 1 GPU-day after PERF-01 |
| 9 Ablations | **planned only** | `experiments.yaml` |
| 10 Architecture | recommendation | FINAL_AUDIT.md §10 |
| 11 Report | done | `FINAL_AUDIT.md` |

## Files inspected
- **Read in full:** `models/{autoencoder,modules,loss,dataloader,trainer,networks,rbf,lr_schedule,fourier,register_models}.py`, `train.py`, and the relevant configs.
- **Partly read:** `models/utils.py`, `evaluate.py`, `plot_loss_all.py`.
- **Read by a subagent:** the evaluation, DA and paper claims.

## Commands (key)
- Data checks: inline xarray scripts (split overlap, QG twin test, shedding periods, dt labels).
- `scripts/operator_spectrum.py` on 4 checkpoints.
- `scripts/tra_rollout_diagnostics.py` on 4 TRA evaluation sets.
- `scripts/linear_baselines.py --rank 128`.
- `scripts/benchmark_table.py`.
- `scripts/smoke_gradflow_profile.py`, before and after the fixes.
- `scripts/compat_check.py` on both trees.
- `pytest audit_report/tests` (before: 10 failed; after: 10 passed).
- `test_discrete_loss.py` (passes).

## Mistakes caught during the audit
- `gt_longer_1000.nc` is labelled dt = 0.1 but is really 0.05. The first rollout diagnostic used
  it at stride 1 and wrongly suggested the KAE was worse than persistence at lead 1. Corrected
  with `--stride 2`; the invalid file was deleted before commit.
- One profiling run was interrupted by you (GPU contention). It was re-run with a smaller batch.

## Outstanding questions for you
1. Which checkpoint produced the paper's 14.9 / 5.2? None on disk reproduces them.
2. Was the QG test set intended to continue the training trajectories?
3. Should the audit fixes be cherry-picked onto `rares/updated_model`?
   They are independent commits. `175c7a4` is opt-in and does not change existing behaviour.

## Next actions
E0 (corrected baseline, 3 seeds) → E1 × E2 × E3 → gate against POD-DMD2 / U-Net_tn.
