import torch
import torch.optim as optim
from functools import partial

from turbpred.params import DataParams, ModelParamsDecoder
from turbpred.model_diffusion import DiffusionModel
from .autoencoder import KoopmanAutoencoder
from models.metrics_utils import run_diffusion_rollout, kae_rollout_wrapper
from models.utils import (
    load_checkpoint,
)

MODEL_REGISTRY = {}


def register_model(name):
    def decorator(fn):
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


@register_model("KAE")
def build_kae(cfg, ckpt_path, metadata, val_dataset, device="cuda", rollout_steps=50):
    model = KoopmanAutoencoder(
        data_variables=cfg.data.variables,
        input_frames=cfg.data.input_sequence_length,
        height=cfg.model.height,
        width=cfg.model.width,
        latent_dim=cfg.model.latent_dim,
        re_embedding_dim=cfg.model.re_embedding_dim,
        re_cond_type=cfg.model.re_cond_type,
        operator_mode=cfg.model.operator_mode,
        hidden_dims=cfg.model.hidden_dims,
        transformer_config=cfg.model.transformer,
        use_checkpoint=cfg.training.use_checkpoint,
        predict_re=cfg.model.predict_re,
        re_grad_enabled=cfg.model.re_grad_enabled,
        is_continuous=cfg.model.is_continuous,
        **cfg.model.conv_kwargs,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr_scheduler.lr)
    if ckpt_path:
        model, _, _, _ = load_checkpoint(
            ckpt_path, model=model, optimizer=optimizer, strict=True
        )
    model.eval()

    # ✅ Bind rollout_steps now
    rollout_fn = partial(
        kae_rollout_wrapper,
        metadata=metadata,
        rollout_steps=rollout_steps,
        dataset=val_dataset,
    )
    return {"model": model, "rollout_fn": rollout_fn}


@register_model("Diffusion")
def build_diffusion(
    cfg, ckpt_path, metadata, val_dataset, device="cuda", rollout_steps=50
):
    p_md = ModelParamsDecoder(
        arch="direct-ddpm+Prev",
        diffSteps=20,
        diffSchedule="linear",
        diffCondIntegration="noisy",
        trainingNoise=0.0,
    )

    p_d = DataParams(
        batch=64,
        augmentations=["normalize"],
        sequenceLength=[rollout_steps, 2],
        randSeqOffset=True,
        dataSize=[128, 64],
        dimension=2,
        simFields=["pres"],
        simParams=["rey"],
        normalizeMode="incMixed",
    )
    model = DiffusionModel(p_d, p_md, dimension=0, condChannels=8)
    model.training = False
    model.inferenceConditioningIntegration = "clean" if "ncn" in ckpt_path else "noisy"

    loaded = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(loaded["stateDictDecoder"], strict=True)
    model.to(device).eval()

    # ✅ Bind metadata, dataset, rollout_steps now
    rollout_fn = partial(
        run_diffusion_rollout,
        metadata=metadata,
        rollout_steps=rollout_steps,
        dataset=val_dataset,
    )
    return {"model": model, "rollout_fn": rollout_fn}
