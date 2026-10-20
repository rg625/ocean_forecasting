"""Tests for the conditional continuous-time KAE (models/ct_kae.py, models/ct_data.py).

cd koopman_autoencoder && python -m pytest ../audit_report/tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

KAE = Path(__file__).resolve().parents[2] / "koopman_autoencoder"
sys.path.insert(0, str(KAE))

from models.ct_data import DT_RAW, TraWindows, denormalise, to_acdm_layout  # noqa: E402
from models.ct_kae import CTKAE, CTKAEConfig  # noqa: E402

GENS = ["block", "hurwitz", "dense"]


def tiny(gen="block", **kw):
    torch.manual_seed(0)
    cfg = CTKAEConfig(
        height=16,
        width=32,
        latent_dim=8,
        widths=[8, 16],
        blocks_per_stage=1,
        cond_embed=8,
        generator=gen,
        dense_rank=2,
        **kw
    )
    return CTKAE(cfg)


def batch(B=3, n=2):
    g = torch.Generator().manual_seed(1)
    win = torch.randn(B, n, 4, 16, 32, generator=g)
    mask = torch.ones(16, 32)
    mask[6:10, 4:8] = 0
    ma = torch.tensor([0.55, 0.66, 0.8])[:B]
    return win, mask, ma


def perturb_heads(m):
    """Give conditional heads non-zero weights so conditioning is active in the test."""
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "head" in name or "film" in name or "mu_head" in name:
                p.add_(0.1 * torch.randn_like(p))


@pytest.mark.parametrize("gen", GENS)
def test_shapes_finite_and_masking(gen):
    m = tiny(gen).eval()
    win, mask, ma = batch()
    taus = torch.tensor([0.1, 0.2, 0.3]).expand(3, -1)
    z0, zs = m(win, mask, ma, taus)
    assert z0.shape == (3, 8) and zs.shape == (3, 3, 8)
    x = m.decode(zs, mask, ma.repeat_interleave(3))
    assert x.shape == (3, 3, 4, 16, 32) and torch.isfinite(x).all()
    assert (x[..., mask == 0] == 0).all()


@pytest.mark.parametrize("gen", GENS)
def test_exact_propagation_matches_matrix_exponential(gen):
    m = tiny(gen).double()
    perturb_heads(m)
    _, _, ma = batch()
    z0 = torch.randn(3, 8, dtype=torch.float64)
    taus = torch.tensor(
        [[0.1, 0.35, 2.0], [0.05, 0.6, 7.3], [1.0, 1.1, 1.2]], dtype=torch.float64
    )
    zs = m.propagate(z0, taus, ma)
    e = m.embed_cond(ma)
    A = m.gen.matrix(e, 3)
    mu = m.fixed_point(e, 3)
    for b in range(3):
        for k in range(3):
            ref = mu[b] + torch.linalg.matrix_exp(A[b] * taus[b, k]) @ (z0[b] - mu[b])
            torch.testing.assert_close(zs[b, k], ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("gen", GENS)
def test_semigroup_and_uniform_grid(gen):
    """z(t1 + t2) = propagate(propagate(z, t1), t2); uniform-grid fast path == general path."""
    m = tiny(gen).double()
    perturb_heads(m)
    _, _, ma = batch()
    z0 = torch.randn(3, 8, dtype=torch.float64)
    t1, t2 = 0.3, 0.45
    a = m.propagate(
        m.propagate(z0, torch.full((3, 1), t1, dtype=torch.float64), ma)[:, 0],
        torch.full((3, 1), t2, dtype=torch.float64),
        ma,
    )[:, 0]
    b = m.propagate(z0, torch.full((3, 1), t1 + t2, dtype=torch.float64), ma)[:, 0]
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
    grid = 0.1 * torch.arange(1, 6, dtype=torch.float64).expand(3, -1)
    fast = m.propagate(z0, grid, ma)
    slow = torch.stack(
        [m.propagate(z0, grid[:, k : k + 1], ma)[:, 0] for k in range(5)], 1
    )
    torch.testing.assert_close(fast, slow, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("gen", ["block", "hurwitz"])
def test_stable_generators_never_grow(gen):
    m = tiny(gen).double()
    perturb_heads(m)
    with torch.no_grad():  # push the unconstrained parameters far from init
        for p in m.gen.parameters():
            p.add_(3 * torch.randn_like(p))
    ma = torch.linspace(0.3, 1.2, 7, dtype=torch.float64)
    A = m.gen.matrix(m.embed_cond(ma), 7)
    for t in [0.05, 0.5, 5.0, 50.0]:
        n = torch.linalg.matrix_norm(torch.linalg.matrix_exp(A * t), ord=2)
        assert (n <= 1 + 1e-9).all(), (gen, t, n.max())


@pytest.mark.parametrize("gen", GENS)
def test_conditioning_receives_gradient_and_changes_output(gen):
    m = tiny(gen)
    win, mask, ma = batch()
    taus = torch.tensor([0.1, 0.5]).expand(3, -1)
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    for _ in range(
        2
    ):  # zero-init heads: grads from step 1, sensitivity after an update
        opt.zero_grad()
        z0, zs = m(win, mask, ma, taus)
        loss = (m.decode(zs, mask, ma.repeat_interleave(2)) - 1.0).pow(2).mean() + (
            zs - 1
        ).pow(2).mean()
        loss.backward()
        heads = [
            p.grad.abs().sum()
            for n, p in m.named_parameters()
            if ("head" in n or "film" in n or "embed" in n) and p.grad is not None
        ]
        assert len(heads) > 0 and float(sum(heads)) > 0
        opt.step()
    m.eval()
    with torch.no_grad():
        a = m.decode(
            m(win, mask, torch.full((3,), 0.55), taus)[1], mask, torch.full((6,), 0.55)
        )
        b = m.decode(
            m(win, mask, torch.full((3,), 0.80), taus)[1], mask, torch.full((6,), 0.80)
        )
    assert float((a - b).abs().max()) > 1e-6


def test_checkpoint_round_trip_and_deterministic_eval(tmp_path):
    m = tiny("block")
    perturb_heads(m)
    torch.save(
        {"model": m.state_dict(), "model_config": m.config_dict()}, tmp_path / "c.pth"
    )
    ck = torch.load(tmp_path / "c.pth", weights_only=False)
    m2 = CTKAE(CTKAEConfig(**ck["model_config"]))
    m2.load_state_dict(ck["model"], strict=True)
    m.eval(), m2.eval()
    win, mask, ma = batch()
    taus = torch.tensor([0.1, 0.2]).expand(3, -1)
    with torch.no_grad():
        o1, o2, o3 = (
            m(win, mask, ma, taus)[1],
            m2(win, mask, ma, taus)[1],
            m(win, mask, ma, taus)[1],
        )
    torch.testing.assert_close(o1, o2, rtol=0, atol=0)
    torch.testing.assert_close(o1, o3, rtol=0, atol=0)


def _fake_windows(S=2, T=40):
    w = TraWindows.__new__(TraWindows)
    t = torch.arange(T, dtype=torch.float32)
    w.x = (
        (
            t[None, :, None, None, None]
            + 1000 * torch.arange(S)[:, None, None, None, None]
        )
        .expand(S, T, 4, 2, 2)
        .clone()
    )
    w.S, w.T, w.device = S, T, torch.device("cpu")
    w.ma = torch.tensor([0.6, 0.7])
    w.mask = torch.ones(2, 2)
    return w


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_sampler_time_alignment(stride):
    """Window = frames t-(n-1)s..t, targets = t+s..t+Ks, taus = k*s*DT_RAW; never out of range."""
    w = _fake_windows()
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        win, tgt, taus, ma, s = w.sample(4, 2, 5, [stride], g)
        t_now = win[:, -1, 0, 0, 0] % 1000
        np.testing.assert_allclose(win[:, 0, 0, 0, 0] % 1000, t_now - s)
        for k in range(5):
            np.testing.assert_allclose(tgt[:, k, 0, 0, 0] % 1000, t_now + (k + 1) * s)
            np.testing.assert_allclose(taus[:, k], (k + 1) * s * DT_RAW, rtol=1e-6)
        sim = (win[:, -1, 0, 0, 0] // 1000).long()
        np.testing.assert_allclose(ma, w.ma[sim])


def test_fixed_windows_alignment():
    w = _fake_windows()
    # start 35 needs frame 35 + 4*2 = 43 >= T = 40 and must be skipped; 2 sims x 2 starts
    win, tgt, taus, ma = w.fixed_windows(2, 4, 2, starts=[5, 30, 35])
    assert win.shape[0] == 4
    t_now = win[:, -1, 0, 0, 0] % 1000
    np.testing.assert_allclose(tgt[:, 0, 0, 0, 0] % 1000, t_now + 2)
    np.testing.assert_allclose(tgt[:, -1, 0, 0, 0] % 1000, t_now + 8)


def test_acdm_layout_round_trip():
    x = torch.randn(2, 3, 4, 64, 128)
    ma = torch.tensor([0.5, 0.6])
    a = to_acdm_layout(x, ma)
    assert a.shape == (2, 3, 5, 128, 64)
    phys = denormalise(x)
    np.testing.assert_allclose(
        a[:, :, 2], phys[:, :, 3].transpose(-1, -2).numpy(), rtol=1e-6
    )  # rho
    np.testing.assert_allclose(
        a[:, :, 3], phys[:, :, 2].transpose(-1, -2).numpy(), rtol=1e-6
    )  # p
    np.testing.assert_allclose(a[:, 0, 4, 0, 0], ma.numpy())


@pytest.mark.parametrize("gen", GENS)
def test_no_fixed_point_is_pure_linear(gen):
    """fixed_point='none': z(tau) = exp(tau A) z0 exactly (no offset anywhere)."""
    torch.manual_seed(0)
    cfg = CTKAEConfig(
        height=16,
        width=32,
        latent_dim=8,
        widths=[8, 16],
        blocks_per_stage=1,
        cond_embed=8,
        generator=gen,
        dense_rank=2,
        fixed_point="none",
    )
    m = CTKAE(cfg).double()
    perturb_heads(m)
    assert not any(n == "mu0" for n, _ in m.named_parameters())
    _, _, ma = batch()
    z0 = torch.randn(3, 8, dtype=torch.float64)
    taus = torch.tensor([[0.1, 0.7]] * 3, dtype=torch.float64)
    zs = m.propagate(z0, taus, ma)
    A = m.generator_matrix(ma)
    for b in range(3):
        for k in range(2):
            torch.testing.assert_close(
                zs[b, k],
                torch.linalg.matrix_exp(A[b] * taus[b, k]) @ z0[b],
                rtol=1e-9,
                atol=1e-9,
            )


@pytest.mark.parametrize("gen", GENS)
def test_augmented_generator_reproduces_affine_rollout(gen):
    """With a fixed point mu(Ma), the model is exactly linear in [z; 1]."""
    m = tiny(gen).double()
    perturb_heads(m)
    with torch.no_grad():
        m.mu0.add_(torch.randn_like(m.mu0))
    _, _, ma = batch()
    z0 = torch.randn(3, 8, dtype=torch.float64)
    taus = torch.tensor([[0.2, 1.3]] * 3, dtype=torch.float64)
    zs = m.propagate(z0, taus, ma)
    Aug = m.augmented_generator(ma)
    z_aug = torch.cat([z0, torch.ones(3, 1, dtype=torch.float64)], -1)
    for b in range(3):
        for k in range(2):
            out = torch.linalg.matrix_exp(Aug[b] * taus[b, k]) @ z_aug[b]
            torch.testing.assert_close(out[:-1], zs[b, k], rtol=1e-9, atol=1e-9)
            torch.testing.assert_close(out[-1], torch.tensor(1.0, dtype=torch.float64))


def test_horizon_schedule_hold_then_ramp():
    from omegaconf import OmegaConf
    import train_ctkae

    tc = OmegaConf.create(
        {
            "steps": 1000,
            "k_start": 1,
            "k_max": 61,
            "curriculum_frac": 0.5,
            "k_hold_frac": 0.2,
        }
    )
    assert train_ctkae.horizon_at(1, tc) == 1
    assert train_ctkae.horizon_at(200, tc) == 1  # still holding
    assert train_ctkae.horizon_at(450, tc) == 31  # half-way up the ramp
    assert (
        train_ctkae.horizon_at(700, tc) == 61 and train_ctkae.horizon_at(1000, tc) == 61
    )
    tc0 = OmegaConf.create(
        {"steps": 1000, "k_start": 10, "k_max": 60, "curriculum_frac": 0.5}
    )
    assert train_ctkae.horizon_at(250, tc0) == 35  # legacy schedule unchanged
