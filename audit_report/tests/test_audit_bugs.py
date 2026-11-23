"""Regression tests for the confirmed bugs in audit_report/FINAL_AUDIT.md (IDs in names).

Each test fails on the original code (commit 79713ce) and passes after the fix.
Runs on CPU in a few seconds:

    cd koopman_autoencoder && python -m pytest ../audit_report/tests -q
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

KAE = Path(__file__).resolve().parents[2] / "koopman_autoencoder"
sys.path.insert(0, str(KAE))

from models.modules import KoopmanOperator  # noqa: E402
from models.loss import KoopmanLoss  # noqa: E402
from models.dataloader import MeanStdNormalizer, TRA_MEAN  # noqa: E402


# --------------------------------------------------------------------------- #
# BUG-01  evaluate.py scores prediction x_{t+k} against ground truth x_{t+k-1}
# --------------------------------------------------------------------------- #
def test_bug01_eval_array_is_time_aligned_with_ground_truth():
    from evaluate import tensordict_to_eval_array_with_cond

    L_in, T, H, W = 2, 6, 3, 4
    # ground-truth frame index i has constant value i; the dataset puts frames
    # [0, L_in) in the input window and the model predicts frames L_in, L_in+1, ...
    # Unbatched, exactly as KoopmanEvaluator.run_validation_rollouts passes them.
    frame = lambda i: torch.full((H, W), float(i))  # noqa: E731
    inp = TensorDict(
        {
            v: torch.stack([frame(i) for i in range(L_in)], 0)
            for v in ["v_x", "v_y", "rho", "p"]
        },
        batch_size=[L_in],
    )
    inp["cond_input"] = torch.full((L_in,), 0.7)
    pred = TensorDict(
        {
            v: torch.stack([frame(i) for i in range(L_in, L_in + T)], 0)
            for v in ["v_x", "v_y", "rho", "p"]
        },
        batch_size=[T],
    )
    arr = tensordict_to_eval_array_with_cond(inp, pred)
    got = arr[0, :, 0, 0, 0]
    assert arr.shape[1] == T
    np.testing.assert_array_equal(got, np.arange(T, dtype=np.float32))


# --------------------------------------------------------------------------- #
# BUG-02  LoRA hypernetwork initialised with U = V = 0 -> zero gradient forever
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("operator_type", ["continuous", "discrete", "azencot"])
def test_bug02_lora_hypernetwork_receives_gradient(operator_type):
    torch.manual_seed(0)
    D, C, rank = 8, 4, 2
    op = KoopmanOperator(
        latent_dim=D,
        cond_embedding_dim=C,
        rank=rank,
        operator_type=operator_type,
        is_continuous=operator_type == "continuous",
    )
    z, cond = torch.randn(5, D), torch.randn(5, C)
    out = op(z, cond=cond, dt=0.1)
    # at initialisation the conditioned operator must still equal the base operator
    base = op.dynamics.hypnet
    op.dynamics.hypnet = None  # same call without the LoRA path
    ref = op(z, cond=cond, dt=0.1)
    op.dynamics.hypnet = base
    torch.testing.assert_close(out, ref)
    (out**2).sum().backward()
    g = sum(
        float(p.grad.abs().sum())
        for p in op.dynamics.hypnet.parameters()
        if p.grad is not None
    )
    assert g > 0, "conditioning hypernetwork receives no gradient (dead LoRA init)"


# --------------------------------------------------------------------------- #
# BUG-03  stability regulariser is added as l_stab.detach(): weight has no effect
# --------------------------------------------------------------------------- #
def test_bug03_stability_term_contributes_gradient():
    torch.manual_seed(0)
    B, T, D = 2, 3, 4
    lat = torch.randn(B, T, D)
    dist = (lat + 0.1 * torch.randn(B, T, D)).requires_grad_(True)
    dz, dzd = torch.randn(B, D), torch.randn(B, D)
    empty = TensorDict({}, batch_size=[B])
    crit = KoopmanLoss(stability_weight=1.0, beta=0.0, physics_weight=0.0)
    res = crit(
        koopman_operator=None,
        x_recon=empty,
        x_preds=TensorDict({}, batch_size=[B, T]),
        latent_pred=lat,
        x_true=empty,
        x_future=TensorDict({}, batch_size=[B, T]),
        disturbed_latents=dist,
        dz_dt=dz,
        dz_dt_disturbed=dzd,
    )
    assert res.metrics["stability"] > 0
    res.total_loss.backward()
    assert dist.grad is not None and float(dist.grad.abs().sum()) > 0


# --------------------------------------------------------------------------- #
# BUG-04  Trainer stores num_visual_batches into save_latest_every
# --------------------------------------------------------------------------- #
def test_bug04_trainer_keeps_save_latest_every(tmp_path):
    from models.trainer import Trainer

    t = Trainer(
        model=None,
        train_loader=None,
        val_loader=None,
        optimizer=None,
        criterion=None,
        lr_scheduler=None,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        num_epochs=1,
        patience=1,
        log_epoch=1,
        save_latest_every=10,
        num_visual_batches=1,
        precision="float32",
    )
    assert t.save_latest_every == 10
    assert t.num_visual_batches == 1


# --------------------------------------------------------------------------- #
# PERF-01 autograd anomaly mode + per-module NaN hooks are always on (16x slower)
# --------------------------------------------------------------------------- #
def test_perf01_debug_instrumentation_is_opt_in(tmp_path):
    from models.trainer import Trainer

    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    t = Trainer(
        model=model,
        train_loader=None,
        val_loader=None,
        optimizer=None,
        criterion=None,
        lr_scheduler=None,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        num_epochs=1,
        patience=1,
        log_epoch=1,
        precision="float32",
    )
    assert getattr(t, "debug_nan_checks", None) is False
    t._setup_debug_instrumentation()
    assert not torch.is_anomaly_enabled()
    assert all(len(m._forward_hooks) == 0 for m in model.modules())
    t.debug_nan_checks = True
    t._setup_debug_instrumentation()
    assert torch.is_anomaly_enabled()
    assert any(len(m._forward_hooks) > 0 for m in model.modules())
    torch.autograd.set_detect_anomaly(False)


# --------------------------------------------------------------------------- #
# BUG-05  MeanStdNormalizer crashes on QG variables (q_1/q_2), and a data-fitted
#         normaliser must not be refitted on validation/test data
# --------------------------------------------------------------------------- #
def test_bug05_normalizer_handles_unknown_vars_and_is_not_refit():
    torch.manual_seed(0)
    train = TensorDict({"q_1": torch.randn(4, 10, 3, 3) * 2 + 1}, batch_size=[4, 10])
    val = TensorDict({"q_1": torch.randn(4, 10, 3, 3) * 5 - 3}, batch_size=[4, 10])
    n = MeanStdNormalizer()
    n.fit(train)
    m, s = float(n.means["q_1"]), float(n.stds["q_1"])
    assert abs(m - float(train["q_1"].mean())) < 1e-5
    assert abs(s - float(train["q_1"].std())) < 1e-4
    n.fit(val)  # second fit (validation split) must keep the training statistics
    assert float(n.means["q_1"]) == m and float(n.stds["q_1"]) == s
    out = n.transform(val.clone())
    torch.testing.assert_close(out["q_1"], (val["q_1"] - m) / (s + n.EPSILON))
    # hard-coded tables are still used for the known datasets
    n2 = MeanStdNormalizer()
    n2.fit(
        TensorDict(
            {"rho": torch.randn(2, 3, 3), "v_x": torch.randn(2, 3, 3)}, batch_size=[2]
        )
    )
    assert float(n2.means["rho"]) == pytest.approx(TRA_MEAN["rho"], rel=1e-6)


# --------------------------------------------------------------------------- #
# BUG-06  conv_kwargs (padding_mode) and `spectral` silently ignored by encoder and
#         decoder blocks; opt-in flag honours them, legacy default warns.
# --------------------------------------------------------------------------- #
def _tiny_kae(**kw):
    from models.autoencoder import KoopmanAutoencoder

    return KoopmanAutoencoder(
        data_variables={"u": 1},
        input_frames=1,
        height=16,
        width=16,
        latent_dim=8,
        cond_embedding_dim=None,
        hidden_dims=[8, 8],
        is_continuous=True,
        operator_type="continuous",
        rank=None,
        use_attention=False,
        **kw
    )


def test_bug06_conv_kwargs_honoured_when_requested():
    m = _tiny_kae(
        spectral=False, padding_mode="circular", padding=1, honor_conv_kwargs=True
    )
    convs = [
        c
        for c in list(m.encoder.modules()) + list(m.decoder.modules())
        if isinstance(c, torch.nn.Conv2d) and c.kernel_size != (1, 1)
    ]
    assert convs and all(c.padding_mode == "circular" for c in convs)
    assert not any(hasattr(c, "weight_orig") for c in m.modules())  # no spectral norm


def test_bug06_legacy_default_warns_about_ignored_kwargs():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = _tiny_kae(spectral=False, padding_mode="circular", padding=1)
    assert any("ignored" in str(x.message) for x in w)
    # legacy architecture unchanged: res-block convs still spectral-normed, zero padding
    assert any(hasattr(c, "weight_orig") for c in m.modules())
