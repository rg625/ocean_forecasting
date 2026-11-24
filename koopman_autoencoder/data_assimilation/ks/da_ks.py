# mypy: disable-error-code="var-annotated"
"""
da_ks.py — Latent-space Data Assimilation (4D-Var) for the Kuramoto–Sivashinsky (KS) model.

Idea (variational data assimilation in the Koopman latent space):
Because the latent dynamics are linear and continuous, the *exact* propagator is
    z(tau) = expm(K_cont * tau) @ z(0)
a single matrix product, independent of the horizon tau. We NEVER observe the state at
the analysis time t0. We only observe one or more FUTURE states y_i = u(t0 + tau_i).
Data assimilation recovers the unknown past state by optimising a free latent z0:

    min_{z0}  sum_i || decoder( expm(K*tau_i) @ z0 ) - y_i ||^2

z0 is initialised randomly (no knowledge of the past) and gradient descent pulls it,
through the forward operator and decoder, to a past state whose future matches the
observations.

WHAT TO MEASURE
---------------
* Headline (physically meaningful): analysis error in FIELD space at the unobserved t0,
      || decoder(z0) - u(t0) ||^2 ,   compared to the autoencoder reconstruction floor.
  This is exactly the quantity DA is meant to minimise (recover the STATE).
* Diagnostic only: latent distance || z0 - encoder(u(t0)) ||.  The encoder/decoder are
  NOT a bijection (the decoder has a null space), so the latent code is non-unique and
  this quantity is NOT expected to reach zero. It is kept purely for inspection.

Notes
-----
* No input-history frames are needed anywhere: z0 is a free variable, the decoder maps
  one latent -> one field, and each observation is a single frame. Feeding past frames
  would leak the answer DA is supposed to infer.
* Multiple, irregularly spaced observations (`--obs-offsets`) over-determine the state
  and drive the field recovery toward the AE floor — demonstrating the paper's
  "irregularly sampled observations" claim, still with one matrix_exp per time.

Run in the training conda env, e.g.:
    python -m data_assimilation.ks.da_ks --sim 0 --t0 100 --obs-offsets 1,3,7,15,25 --iters 1500 --lr 1e-2
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

# --- Local imports (same style as evaluate_ks.py) ---
from models.autoencoder import KoopmanAutoencoder
from models.networks import TransformerConfig
from models.dataloader import KS_MEAN, KS_STD

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("da_ks")


# ----------------------------------------------------------------------------
# Architecture handling
# ----------------------------------------------------------------------------
# Fallback matches configs/model/continous_linear_128_ks.yaml, used only if the
# run's final_model.pth (which stores the resolved config) is unavailable.
KS_ARCH_FALLBACK: Dict = {
    "height": 64,
    "width": 1,
    "hidden_dims": [64, 128, 256],
    "block_size": 1,
    "kernel_size": 3,
    "conv_kwargs": {"padding": 1, "padding_mode": "circular"},
    "latent_dim": 128,
    "cond_embedding_dim": None,
    "cond_type": None,
    "operator_mode": "linear",
    "transformer": {
        "num_layers": 1,
        "nhead": 2,
        "ff_mult": 2,
        "max_len": 1000,
        "dropout": 0.1,
    },
    "predict_cond": False,
    "cond_grad_enabled": False,
    "is_continuous": True,
    "rank": None,
    "use_attention": True,
    "spectral": True,
    "input_sequence_length": 4,
    "variables": {"u": 1},
}


def load_arch(run_dir: Path) -> Dict:
    """Read the architecture from the run's final_model.pth if present, else fall back."""
    final_path = run_dir / "final_model.pth"
    arch = dict(KS_ARCH_FALLBACK)
    if final_path.is_file():
        try:
            ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
            m, d = ckpt["config"]["model"], ckpt["config"]["data"]
            for k in [
                "height",
                "width",
                "hidden_dims",
                "block_size",
                "kernel_size",
                "conv_kwargs",
                "latent_dim",
                "cond_embedding_dim",
                "cond_type",
                "operator_mode",
                "transformer",
                "predict_cond",
                "cond_grad_enabled",
                "is_continuous",
                "rank",
                "use_attention",
                "spectral",
            ]:
                if k in m:
                    arch[k] = m[k]
            arch["input_sequence_length"] = d["input_sequence_length"]
            arch["variables"] = dict(d["variables"])
            logger.info(f"Loaded architecture from {final_path}")
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Could not read arch from {final_path} ({e}); using fallback."
            )
    else:
        logger.warning(f"{final_path} not found; using hardcoded KS fallback arch.")
    return arch


def build_model(
    arch: Dict, ckpt_path: Path, device: torch.device
) -> KoopmanAutoencoder:
    """Instantiate KoopmanAutoencoder and load weights from a checkpoint (.pth)."""
    tcfg = TransformerConfig(**{k: arch["transformer"][k] for k in arch["transformer"]})
    model = KoopmanAutoencoder(
        data_variables=dict(arch["variables"]),
        input_frames=arch["input_sequence_length"],
        height=arch["height"],
        width=arch["width"],
        latent_dim=arch["latent_dim"],
        cond_embedding_dim=arch["cond_embedding_dim"],
        cond_type=arch["cond_type"],
        operator_mode=arch["operator_mode"],
        hidden_dims=list(arch["hidden_dims"]),
        block_size=arch["block_size"],
        kernel_size=arch["kernel_size"],
        transformer_config=tcfg,
        use_checkpoint=False,
        predict_cond=arch["predict_cond"],
        cond_grad_enabled=arch["cond_grad_enabled"],
        disturb_std=None,  # eval: no stochastic latent noise
        is_continuous=arch["is_continuous"],
        rank=arch["rank"],
        cond_expansion_type=None,  # KS is unconditioned
        use_attention=arch["use_attention"],
        spectral=arch["spectral"],
        **dict(arch["conv_kwargs"]),
    ).to(device)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["model_state_dict"] if "model_state_dict" in state else state
    model.load_state_dict(sd, strict=True)
    logger.info(f"Loaded weights from {ckpt_path}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # freeze the whole model; only z0 is optimised
    return model


# ----------------------------------------------------------------------------
# Normalisation (deterministic KS constants — identical for train/val)
# ----------------------------------------------------------------------------
def _ks_stats(device: torch.device):
    def _f(v):
        return float(v.item()) if torch.is_tensor(v) else float(v)

    return torch.tensor(_f(KS_MEAN["u"]), device=device), torch.tensor(
        _f(KS_STD["u"]), device=device
    )


# ----------------------------------------------------------------------------
# Koopman propagator:  z(tau) = expm(K * tau) @ z0   (row-vector convention)
# ----------------------------------------------------------------------------
def get_generator(model: KoopmanAutoencoder) -> torch.Tensor:
    """Extract the constant continuous-time generator K = skew(W) + sym(D)  [D, D]."""
    dyn = model.koopman_operator.dynamics
    return dyn._get_effective_linear_map(None).detach()  # cond=None for KS


def evolve(z0: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """z(tau) for row vectors: z0 [B, D], phi = expm(K*tau) [D, D] -> [B, D].

    EXACT propagator: one matrix product, independent of the horizon tau.
    `phi` is precomputed once per observation time (K is constant during z0 optimisation).
    """
    return z0 @ phi.T


def propagate_rollout(
    z0: torch.Tensor, model: KoopmanAutoencoder, n_steps: int, dt: float = 0.1
) -> torch.Tensor:
    """BASELINE propagator: autoregressive latent rollout of `n_steps` sequential steps.

    Uses the model's own operator (RK4 step of dz/dt = K z). Cost scales linearly with
    the horizon (n_steps = tau / dt) and backprop unrolls all steps -- this is the method
    the exact matrix-exponential is meant to beat.
    """
    z = z0
    for _ in range(n_steps):
        z = model.koopman_operator(z, cond=None, dt=dt)
    return z


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """MSE over observed entries. mask broadcasts over the batch; None => dense MSE."""
    if mask is None:
        return F.mse_loss(pred, target)
    diff = (pred - target) ** 2 * mask
    return diff.sum() / mask.expand_as(pred).sum().clamp_min(1.0)


# ----------------------------------------------------------------------------
# Encoding helpers
# ----------------------------------------------------------------------------
def encode_frame(model: KoopmanAutoencoder, frame_bhw: torch.Tensor) -> torch.Tensor:
    """Single-frame present-encoding. frame_bhw: [B, H, W] (normalised) -> z [B, D]."""
    from tensordict import TensorDict

    x = TensorDict(
        {"u": frame_bhw.unsqueeze(1)},  # [B, T=1, H, W]
        batch_size=[frame_bhw.shape[0], 1],
    )
    return model.present_encoding(x, cond_input=None)


def auto_init_scale(model, u_norm, sim, device, n=256) -> float:
    """Estimate the per-element std of encoder latents to scale the random init."""
    T = u_norm.shape[1]
    idx = np.linspace(0, T - 1, min(n, T)).astype(int)
    frames = u_norm[sim, idx].to(device)  # [n, H, W]
    with torch.no_grad():
        z = encode_frame(model, frames)  # [n, D]
    return float(z.std().item())


# ----------------------------------------------------------------------------
# Main DA routine
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="KS latent-space data assimilation (4D-Var)."
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path("model_outputs_ks/continous_linear_128/rollout_10"),
    )
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Checkpoint .pth (default: <run-dir>/checkpoints/best_model.pth)",
    )
    ap.add_argument("--val", type=Path, default=Path("data/ks/val.nc"))
    ap.add_argument(
        "--sim", type=int, default=0, help="Index into the 'sim' dimension."
    )
    ap.add_argument(
        "--t0",
        type=int,
        default=100,
        help="Analysis frame — the UNOBSERVED state we recover.",
    )
    ap.add_argument(
        "--obs-offsets",
        type=str,
        default="20",
        help="Comma-separated FUTURE observation offsets in frames, e.g. '1,3,7,15,25'. "
        "tau_i = offset_i * dt. Multiple/irregular offsets over-determine the state.",
    )
    ap.add_argument("--dt", type=float, default=0.1, help="Physical time per frame.")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    ap.add_argument(
        "--init-scale",
        type=float,
        default=None,
        help="Std of the random z0 init (default: auto from encoder latents).",
    )
    ap.add_argument(
        "--reg",
        type=float,
        default=0.0,
        help="Optional L2 (min-norm) prior weight on z0.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("da_ks_result.npz"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    offsets: List[int] = [
        int(o) for o in str(args.obs_offsets).split(",") if o.strip() != ""
    ]
    assert offsets and all(
        o > 0 for o in offsets
    ), "obs-offsets must be positive frame offsets"
    ckpt = args.ckpt or (args.run_dir / "checkpoints" / "best_model.pth")
    logger.info(f"Device={device} | ckpt={ckpt}")
    logger.info(
        f"Analysis t0={args.t0} | observation offsets (frames)={offsets} "
        f"| tau={[round(o * args.dt, 3) for o in offsets]}"
    )

    # 1. Model
    arch = load_arch(args.run_dir)
    model = build_model(arch, ckpt, device)

    # 2. Data (raw -> normalised with deterministic KS constants)
    ds = xr.open_dataset(args.val)
    u_raw = torch.from_numpy(ds["u"].values).float()  # [sim, t, x=H, y=W]
    x_coord = ds["x"].values
    ds.close()
    n_sim, n_t = u_raw.shape[0], u_raw.shape[1]
    assert 0 <= args.sim < n_sim, f"sim {args.sim} out of range [0,{n_sim})"
    assert (
        0 <= args.t0 < n_t and args.t0 + max(offsets) < n_t
    ), f"t0/offsets out of range (n_t={n_t})"

    mean, std = _ks_stats(device)
    u_norm = (u_raw.to(device) - mean) / (std + 1e-8)  # [sim, t, H, W]

    # 3. References (never used as optimisation inputs)
    frame_t0 = u_norm[args.sim, args.t0].unsqueeze(0)  # [1, H, W]  true analysis state
    with torch.no_grad():
        z0_true = encode_frame(model, frame_t0)  # latent diagnostic reference
        # autoencoder reconstruction floor: best possible field error at t0
        ae_floor = float(F.mse_loss(model.decode(z0_true)["u"], frame_t0).item())
    z0_true_norm = float(torch.linalg.vector_norm(z0_true).item())
    logger.info(
        f"AE reconstruction floor MSE(decode(enc(u(t0))), u(t0)) = {ae_floor:.4e}"
    )

    # 4. Observations and Koopman propagators (computed ONCE — cost independent of tau)
    K = get_generator(model)  # [D, D]
    phis = [torch.matrix_exp(K * (o * args.dt)) for o in offsets]
    y_obs = [u_norm[args.sim, args.t0 + o].unsqueeze(0) for o in offsets]

    # 5. Initialise the unknown latent and optimise it
    if args.init_scale is None:
        scale = auto_init_scale(model, u_norm, args.sim, device)
        logger.info(f"Auto init scale (encoder latent std) = {scale:.4f}")
    else:
        scale = args.init_scale
    z0 = (torch.randn(1, arch["latent_dim"], device=device) * scale).requires_grad_(
        True
    )

    opt = (torch.optim.Adam if args.optimizer == "adam" else torch.optim.SGD)(
        [z0], lr=args.lr
    )

    hist = {
        "iter": [],
        "obs_loss": [],
        "analysis_mse_t0": [],
        "latent_dist": [],
        "latent_dist_rel": [],
    }

    for it in range(args.iters):
        opt.zero_grad()
        obs_loss = sum(
            F.mse_loss(model.decode(evolve(z0, p))["u"], y) for p, y in zip(phis, y_obs)
        ) / len(phis)
        total = obs_loss + args.reg * (z0**2).mean()
        total.backward()
        opt.step()

        with torch.no_grad():
            analysis_mse = float(F.mse_loss(model.decode(z0)["u"], frame_t0).item())
            lat = float(torch.linalg.vector_norm(z0 - z0_true).item())
        hist["iter"].append(it)
        hist["obs_loss"].append(float(obs_loss.item()))
        hist["analysis_mse_t0"].append(analysis_mse)
        hist["latent_dist"].append(lat)
        hist["latent_dist_rel"].append(lat / (z0_true_norm + 1e-8))

        if it % max(1, args.iters // 10) == 0 or it == args.iters - 1:
            logger.info(
                f"[{it:04d}] obs_loss={obs_loss.item():.4e} "
                f"| analysis_MSE@t0={analysis_mse:.4e} (floor {ae_floor:.1e}) "
                f"| [diag] ||z0-z0_true||_rel={lat / (z0_true_norm + 1e-8):.3f}"
            )

    # 6. Final fields (denormalised) for plotting
    with torch.no_grad():
        z0_final = z0.detach()
        u_t0_recon = (
            (model.decode(z0_final)["u"] * std + mean).cpu().numpy()[0]
        )  # [H, W]
        u_obs_pred = np.stack(
            [
                (model.decode(evolve(z0_final, p))["u"] * std + mean).cpu().numpy()[0]
                for p in phis
            ]
        )  # [n_obs, H, W]
    u_t0_true = u_raw[args.sim, args.t0].numpy()  # [H, W]
    u_obs_true = np.stack(
        [u_raw[args.sim, args.t0 + o].numpy() for o in offsets]
    )  # [n_obs, H, W]

    # denormalised relative field error at t0
    rel_t0 = float(
        np.linalg.norm(u_t0_recon - u_t0_true) / (np.linalg.norm(u_t0_true) + 1e-12)
    )

    np.savez(
        args.out,
        # histories
        iter=np.array(hist["iter"]),
        obs_loss=np.array(hist["obs_loss"]),
        analysis_mse_t0=np.array(hist["analysis_mse_t0"]),
        latent_dist=np.array(hist["latent_dist"]),
        latent_dist_rel=np.array(hist["latent_dist_rel"]),
        # scalars
        ae_floor=ae_floor,
        analysis_rel_t0_final=rel_t0,
        # latents (diagnostic)
        z0_true=z0_true.detach().cpu().numpy()[0],
        z0_final=z0_final.cpu().numpy()[0],
        # fields (denormalised, squeezed to 1D along x for KS)
        x=np.asarray(x_coord).squeeze(),
        u_t0_true=u_t0_true.squeeze(),
        u_t0_recon=u_t0_recon.squeeze(),
        u_obs_true=u_obs_true.squeeze(),  # [n_obs, H] (or [H] if single)
        u_obs_pred=u_obs_pred.squeeze(),
        # metadata
        sim=args.sim,
        t0=args.t0,
        obs_offsets=np.array(offsets),
        dt=args.dt,
        init_scale=scale,
        optimizer=args.optimizer,
        lr=args.lr,
        iters=args.iters,
        reg=args.reg,
    )
    logger.info(f"Saved results -> {args.out}")
    logger.info(
        f"STATE recovery @ t0: analysis_MSE {hist['analysis_mse_t0'][0]:.3e} -> "
        f"{hist['analysis_mse_t0'][-1]:.3e} (AE floor {ae_floor:.1e}) | "
        f"denorm rel-L2 {rel_t0:.3f} | obs_loss -> {hist['obs_loss'][-1]:.3e}"
    )
    logger.info(
        f"[diag] latent ||z0-z0_true||_rel {hist['latent_dist_rel'][0]:.3f} -> "
        f"{hist['latent_dist_rel'][-1]:.3f}  (expected to stay high: decoder null space)"
    )


if __name__ == "__main__":
    main()
