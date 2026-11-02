"""Train a KS model with the paper's KS KAE training settings, exactly: the main repository's
train.py (dataset, dataloader, windows, KoopmanLoss, AdamW, CosineWarmup, Trainer), run
unmodified from its directory on the configuration stored in the paper's checkpoint
(model_outputs_ks/continous_linear_128/rollout_10/final_model.pth). Only the model class is
swapped (and, for back001, its backward-consistency term added to the loss):

    python train_matched.py ARCH OUT_DIR [--latent D] [--w-back W] [--seed S]
      ARCH = kae : the paper's KoopmanAutoencoder (control: today's code, same settings)
             ct  : models/ct_kae_matched.CTKAEMatched (CT-KAE: block generator, exact exp)

W&B is disabled (WANDB_MODE=disabled); everything is written under OUT_DIR.
"""

import argparse
import copy
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
MAIN = Path("/home/rg625/mnt/ocean_forecasting/koopman_autoencoder")
IMPL = Path(__file__).resolve().parents[3] / "koopman_autoencoder"
PAPER_CKPT = MAIN / "model_outputs_ks/continous_linear_128/rollout_10/final_model.pth"
os.chdir(MAIN)
sys.path.insert(0, str(MAIN))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import models  # noqa: E402,F401  (the main repository's package)

_load("models.ct_kae", IMPL / "models/ct_kae.py")
_load("models.ct_kae_ks", IMPL / "models/ct_kae_ks.py")
CM = _load("models.ct_kae_matched", IMPL / "models/ct_kae_matched.py")
import train as TR  # noqa: E402  (main repository's train.py)
from models.loss import KoopmanLoss  # noqa: E402
from models.trainer import Trainer  # noqa: E402


class MatchedTrainer(Trainer):
    """Trainer, unchanged except that a delay-window model supplies the latent targets
    of the target frames itself (window ending at each frame; see CTKAEMatched)."""

    def true_latent_encoding(self, target_td, model_module):
        if hasattr(model_module, "window_latents"):
            return model_module.window_latents(target_td)
        return super().true_latent_encoding(target_td, model_module)


class LossWithBack(torch.nn.Module):
    """KoopmanLoss plus w_back * CTKAEMatched.back_term(true_latents)"""

    def __init__(self, base: KoopmanLoss, model_ref: list, w_back: float):
        super().__init__()
        self.base, self.model_ref, self.w_back = base, model_ref, w_back
        self.cfg, self.latent_loss = base.cfg, base.latent_loss

    def forward(self, **kw):
        res = self.base(**kw)
        if self.w_back > 0 and kw.get("true_latents") is not None:
            lb = self.model_ref[0].back_term(kw["true_latents"])
            res.total_loss = res.total_loss + self.w_back * lb
            res.metrics["loss_back"] = float(lb.detach())
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arch", choices=["kae", "ct"])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--latent", type=int, default=None)
    ap.add_argument("--w-back", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None, help="smoke tests only")
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    st = torch.load(PAPER_CKPT, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(st["config"])
    cfg["output_dir"] = str(a.out_dir.resolve())
    cfg["ckpt"] = None
    if a.latent is not None:
        cfg["model"]["latent_dim"] = a.latent
    if a.epochs is not None:
        cfg["training"]["num_epochs"] = a.epochs
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "matched_run.json").write_text(
        json.dumps(
            {
                "arch": a.arch,
                "w_back": a.w_back,
                "seed": a.seed,
                "paper_ckpt": str(PAPER_CKPT),
                "config": cfg,
            },
            indent=1,
        )
        + "\n"
    )

    model_ref: list = []
    if a.arch == "ct":

        def factory(data_variables, input_frames, latent_dim, hidden_dims, **_):
            m = CM.CTKAEMatched(data_variables, input_frames, latent_dim, hidden_dims)
            model_ref.append(m)
            return m

        TR.KoopmanAutoencoder = factory
    if a.w_back > 0:
        assert a.arch == "ct", "w_back is a CT-KAE term"
        TR.KoopmanLoss = lambda **kw: LossWithBack(
            KoopmanLoss(**kw), model_ref, a.w_back
        )  # noqa: E731
    TR.Trainer = MatchedTrainer
    TR.main(OmegaConf.create(cfg))


if __name__ == "__main__":
    main()
