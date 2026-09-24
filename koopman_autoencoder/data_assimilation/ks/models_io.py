"""Loading the frozen forecast models used by the DA campaign."""

from __future__ import annotations

from pathlib import Path

import torch

from data_assimilation.ks.unet_ks import UNet1d, UNet1dConfig


def load_kae(run_dir: Path, ckpt: Path | None, device: torch.device):
    """Return (model, K, latent_dim). All weights frozen; K is the continuous generator."""
    from data_assimilation.ks.da_ks import build_model, load_arch, get_generator

    arch = load_arch(run_dir)
    ckpt = ckpt or (run_dir / "checkpoints" / "best_model.pth")
    model = build_model(arch, ckpt, device)  # already eval() + requires_grad_(False)
    K = get_generator(model)
    return model, K, int(arch["latent_dim"])


def load_unet(ckpt: Path, device: torch.device) -> UNet1d:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = state.get("config", {})
    model = UNet1d(
        UNet1dConfig(
            hidden_dims=tuple(cfg.get("hidden_dims", (64, 128, 256))),
            blocks_per_level=int(cfg.get("blocks", 2)),
            residual=bool(cfg.get("residual", True)),
        )
    ).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def kae_latent_scale(model, data, n: int = 512) -> float:
    """Per-element std of encoder latents over the data pool -> uninformed init scale."""
    from tensordict import TensorDict

    idx_t = torch.linspace(100, data.n_t - 1, min(n, data.n_t - 100)).long()
    idx_s = torch.arange(min(8, data.n_sim))
    frames = data.u[idx_s][:, idx_t].reshape(-1, data.X)  # [N, X]
    zs = []
    for i in range(0, frames.shape[0], 256):
        f = frames[i : i + 256].unsqueeze(-1).unsqueeze(1)  # [b, T=1, X, 1]
        x = TensorDict({"u": f.squeeze(-1).unsqueeze(-1)}, batch_size=[f.shape[0], 1])
        zs.append(model.present_encoding(x, cond_input=None))
    return float(torch.cat(zs).std())
