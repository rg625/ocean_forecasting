"""The paper's KS KAE (models/autoencoder.KoopmanAutoencoder: attention + spectral-norm conv
encoder, transformer history encoder, conv decoder) with the CT-KAE's block generator in place
of its dense one.

The paper KAE's continuous generator is K = skew(W) + sym(D), a dense D x D matrix. Here
K = blockdiag([[-g_j, -w_j], [w_j, -g_j]]), j = 1..D/2: independent damped oscillators, with
g_j = softplus(.) >= 0 (models/ct_kae.BlockGenerator; ``signed`` frees the sign) and the CT-KAE's
initialisation (g log-uniform in gamma_init, w log-spaced over (0.05, omega_max)); with
init="kae" it starts from the KAE's initial spectrum instead (w = eigen-frequencies of skew(W),
W orthogonal; g = 0, free in sign), so only the generator's block structure differs. Everything
else is the KAE's except the propagation: exact exp(dt K) in training (closed form per
oscillator, as the CT-KAE; the KAE uses an RK4 step), K for matrix_exp in the DA
(da_ks.get_generator calls _get_effective_linear_map), the loss and the training pipeline.
So a run with this operator differs from the KAE control only in the generator (structure,
initialisation and exact propagation).

    blockify(model)          swap the operator of a built KoopmanAutoencoder (before training or
                             before load_state_dict)
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.modules import ContinuousKoopmanOperator, KoopmanMode


def _inv_softplus(x: Tensor) -> Tensor:
    return x + torch.log(-torch.expm1(-x))


class BlockContinuousKoopmanOperator(ContinuousKoopmanOperator):
    def __init__(
        self,
        config,
        gamma_init=(0.01, 0.3),
        omega_max: float = 12.0,
        signed: bool = False,
        init: str = "ct",
    ):
        super().__init__(config)
        assert config.mode == KoopmanMode.LINEAR and config.latent_dim % 2 == 0
        assert self.hypnet is None, "KS: unconditioned generator"
        del self.W_param, self.D_param
        m = config.latent_dim // 2
        if init == "kae":
            # the KAE's initial generator in block form: K0 = skew(W), W orthogonal, D = 0
            # has eigenvalues +-i lambda_j (lambda_j <= 1); start at w_j = lambda_j, g_j = 0,
            # decay free in sign (sym(D) is free in sign)
            W = torch.empty(config.latent_dim, config.latent_dim)
            nn.init.orthogonal_(W, gain=1.0)
            lam = torch.linalg.eigvals(0.5 * (W - W.T)).imag.abs().sort().values
            w0 = lam[1::2]  # one of each +-lambda pair (robust to a lambda ~ 0 pair)
            g0, signed = torch.zeros(m), True
        else:
            g0 = torch.exp(
                torch.empty(m).uniform_(
                    math.log(gamma_init[0]), math.log(gamma_init[1])
                )
            )
            w0 = torch.exp(torch.linspace(math.log(0.05), math.log(omega_max), m))
            w0 = w0[torch.randperm(m)]
        self.signed = signed
        self.g_raw = nn.Parameter(g0.clone() if signed else _inv_softplus(g0))
        self.w = nn.Parameter(w0)
        i = torch.arange(m)
        self.register_buffer("i0", 2 * i)
        self.register_buffer("i1", 2 * i + 1)

    def rates(self):
        return (self.g_raw if self.signed else F.softplus(self.g_raw)), self.w

    def _get_effective_linear_map(self, cond_encoded):
        g, w = self.rates()
        D = self.config.latent_dim
        K = g.new_zeros(D, D)
        K = K.index_put((self.i0, self.i0), -g)
        K = K.index_put((self.i1, self.i1), -g, accumulate=True)
        K = K.index_put((self.i0, self.i1), -w, accumulate=True)
        K = K.index_put((self.i1, self.i0), w, accumulate=True)
        return K

    def forward(self, z: Tensor, cond, dt) -> Tensor:
        """exact z(t + dt) = exp(dt K) z, per oscillator e^{-g dt} R(w dt) (the CT-KAE's
        propagation; equals torch.matrix_exp(dt K) used by the DA, unlike an RK4 step)
        """
        assert dt is not None, "Continuous operator needs time step"
        g, w = self.rates()
        t = dt if isinstance(dt, Tensor) else z.new_tensor(float(dt))
        t = t.reshape(-1, 1) if t.ndim else t
        a, c, s = torch.exp(-g * t), torch.cos(w * t), torch.sin(w * t)
        z0, z1 = z[..., self.i0], z[..., self.i1]
        out = torch.stack([a * (c * z0 - s * z1), a * (s * z0 + c * z1)], -1)
        return out.flatten(-2).to(z.dtype)


def blockify(model, **kw):
    """Replace model.koopman_operator.dynamics by the block operator (same config)."""
    op = model.koopman_operator
    assert op.operator_type == "continuous", op.operator_type
    dev = next(model.parameters()).device
    op.dynamics = BlockContinuousKoopmanOperator(op.dynamics.config, **kw).to(dev)
    return model


def load_kae_block(load_kae, run_dir, ckpt, device, meta_path):
    """models_io.load_kae for a train_matched.py kae_block run: da_ks.build_model constructs
    the KoopmanAutoencoder, blockified before its (strict) load_state_dict."""
    import json

    import data_assimilation.ks.da_ks as DK

    kw = json.loads(open(meta_path).read()).get("block_kw", {})
    orig = DK.KoopmanAutoencoder
    DK.KoopmanAutoencoder = lambda **a: blockify(orig(**a), **kw)
    try:
        return load_kae(run_dir, ckpt, device)
    finally:
        DK.KoopmanAutoencoder = orig
