"""Regression tests for the second batch of legacy-model fixes (issues.json IDs).

cd koopman_autoencoder && python -m pytest ../audit_report/tests -q
"""

import sys
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

KAE = Path(__file__).resolve().parents[2] / "koopman_autoencoder"
sys.path.insert(0, str(KAE))

from models.autoencoder import KoopmanAutoencoder  # noqa: E402
from models.loss import KoopmanLoss  # noqa: E402
from models.modules import KoopmanOperator  # noqa: E402
from models.trainer import Trainer  # noqa: E402
from models.utils import load_checkpoint  # noqa: E402


def _tiny(input_frames=1, dt=0.1):
    return KoopmanAutoencoder(
        data_variables={"u": 1},
        input_frames=input_frames,
        height=16,
        width=16,
        latent_dim=8,
        cond_embedding_dim=None,
        hidden_dims=[8, 8],
        is_continuous=True,
        operator_type="continuous",
        rank=None,
        use_attention=False,
        disturb_std=None,
        dt=dt,
        transformer_config=None if input_frames == 1 else _tf(),
    )


def _tf():
    from models.networks import TransformerConfig

    return TransformerConfig(num_layers=1, nhead=2, ff_mult=1, dropout=0.0)


def _batch(B=2, T_in=1, T=3):
    g = torch.Generator().manual_seed(0)
    return (
        TensorDict(
            {"u": torch.randn(B, T_in, 16, 16, generator=g)}, batch_size=[B, T_in]
        ),
        T,
    )


# REPRO-03: forward must run on CPU (no unconditional CUDA events / synchronize)
def test_repro03_forward_runs_on_cpu():
    m = _tiny().eval()
    x, T = _batch()
    with torch.no_grad():
        out = m(x, T)
    assert out.x_preds["u"].shape == (2, T, 16, 16)
    assert torch.isfinite(out.x_preds["u"]).all()


# CLAIM-01: the rollout uses the configured dt, not a hard-coded 0.1
def test_claim01_rollout_uses_configured_dt():
    m = _tiny(dt=0.25).eval()
    x, _ = _batch()
    with torch.no_grad():
        out = m(x, 1)
        z0 = m.encode(x)
        ref = m.koopman_operator(z0, dt=0.25)
    torch.testing.assert_close(out.z_preds[:, 0], ref)
    assert m.koopman_operator.dt_train == 0.25


# EVAL-04: a missing checkpoint is an error, not a silently untrained model
def test_eval04_missing_checkpoint_raises(tmp_path):
    m = _tiny()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(
            str(tmp_path / "nope.pth"), m, torch.optim.SGD(m.parameters(), lr=0.1)
        )


# REPRO-02: checkpoints carry the run metadata, and round-trip strictly
def test_repro02_checkpoint_has_metadata_and_round_trips(tmp_path):
    m = _tiny()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    meta = {
        "config": {"model": {"latent_dim": 8}},
        "git": {"commit": "abc", "dirty": False},
    }
    t = Trainer(
        model=m,
        train_loader=None,
        val_loader=None,
        optimizer=opt,
        criterion=None,
        lr_scheduler=None,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        num_epochs=1,
        patience=1,
        log_epoch=1,
        precision="float32",
        run_metadata=meta,
    )
    t.save_checkpoint(epoch=0, val_loss=1.0, is_best=True)
    ck = torch.load(tmp_path / "checkpoints" / "best_model.pth", weights_only=False)
    assert ck["run_metadata"] == meta
    m2 = _tiny()
    load_checkpoint(
        str(tmp_path / "checkpoints" / "best_model.pth"),
        m2,
        torch.optim.SGD(m2.parameters(), lr=0.1),
        strict=True,
    )
    for (n, a), (_, b) in zip(m.state_dict().items(), m2.state_dict().items()):
        torch.testing.assert_close(a, b, msg=n)


# LOSS-01: explicit prediction-loss distance; default keeps the legacy squared error
@pytest.mark.parametrize("ptype,expected", [(None, 4.0), ("l2", 4.0), ("l1", 2.0)])
def test_loss01_prediction_distance(ptype, expected):
    B, T = 1, 2
    pred = TensorDict({"u": torch.full((B, T, 2, 2), 2.0)}, batch_size=[B, T])
    true = TensorDict(
        {"u": torch.zeros(B, T, 2, 2), "seq_length": torch.full((B, T), T)},
        batch_size=[B, T],
    )
    crit = KoopmanLoss(
        alpha=1.0, loss_type="l1", pred_loss_type=ptype, weighting_type="uniform"
    )
    val, _ = crit.pred_loss(pred, true)
    # uniform weights are 1/T and the loss is a mean over t, so the value is d / T
    assert float(val) == pytest.approx(expected / T)


# DEAD-01: continuous eigen mode fails loudly at construction
def test_dead01_continuous_eigen_mode_raises():
    with pytest.raises(NotImplementedError):
        KoopmanOperator(
            latent_dim=4,
            mode="eigen",
            operator_type="continuous",
            is_continuous=True,
            rank=None,
        )


# ENC-01: window targets use the same encode() map as the rollout start
def test_enc01_window_latent_targets_match_encode():
    torch.manual_seed(0)
    m = _tiny(input_frames=2).eval()
    B, T = 2, 3
    full = torch.randn(B, 2 + T, 16, 16)
    inp = TensorDict({"u": full[:, :2]}, batch_size=[B, 2])
    tgt = TensorDict({"u": full[:, 2:]}, batch_size=[B, T])
    with torch.no_grad():
        z = Trainer.window_latent_encoding(inp, tgt, m)
        for j in range(T):
            win = TensorDict({"u": full[:, 1 + j : 3 + j]}, batch_size=[B, 2])
            torch.testing.assert_close(z[:, j], m.encode(win))
