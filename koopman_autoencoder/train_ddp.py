import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from pathlib import Path
from datetime import datetime
import wandb
import logging
from omegaconf import OmegaConf, DictConfig, open_dict
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import hydra
import sys
import random
import numpy as np
from typing import Mapping, Any

# Refactored project imports
from models.autoencoder import KoopmanAutoencoder
from models.loss import KoopmanLoss
from models.lr_schedule import CosineWarmup
from models.dataloader import (
    create_dataloaders,
    create_ddp_dataloaders,
    QuantileNormalizer,
    MeanStdNormalizer,
)
from models.trainer import Trainer
from models.metrics import Metric
from models.utils import (
    load_checkpoint,
    load_datasets,
)
from models.config_classes import Config


# --- Rank-Aware Logging ---
# This filter ensures that only the main process (Rank 0) prints to the console.
class RankFilter(logging.Filter):
    def filter(self, record):
        try:
            return int(os.environ.get("RANK", 0)) == 0
        except (ValueError, TypeError):
            return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.addFilter(RankFilter())


def set_seed(seed: int):
    """
    Sets seeds for reproducibility.
    In DDP, this should be called with (base_seed + rank) to ensure
    different noise on different GPUs while maintaining determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_ddp():
    """Initializes the distributed process group."""
    if not dist.is_available():
        raise RuntimeError("Torch distributed is not available.")

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    # Sync barrier to ensure all processes have started before continuing
    dist.barrier()
    logger.info(f"DDP setup: Rank {rank}/{world_size}, Local Rank {local_rank}")
    return rank, local_rank, world_size


def find_and_extract_config(node: Mapping[str, Any], key_to_find="output_dir"):
    """
    Recursively searches for a specific key in a nested DictConfig.
    """

    if key_to_find in node:
        return node

    if not isinstance(node, (dict, DictConfig)):
        return None

    for key, value in node.items():
        if isinstance(value, (dict, DictConfig)):
            result = find_and_extract_config(value, key_to_find)
            if result is not None:
                return result
    return None


def main(cfg: DictConfig):
    """Main function to set up and run the training pipeline."""

    # --- Config Handling ---
    # Hoist nested experiment configs to root if necessary
    if "experiment" in cfg:
        if "output_dir" in cfg:
            with open_dict(cfg):
                del cfg["experiment"]
        else:
            logger.info(
                "Config 'output_dir' missing at root. Searching nested 'experiment' config..."
            )
            nested_config = find_and_extract_config(cfg.experiment, "output_dir")

            if nested_config:
                logger.info("Found nested configuration. Merging to root level.")
                with open_dict(cfg):
                    cfg = OmegaConf.merge(cfg, nested_config)

            with open_dict(cfg):
                if "experiment" in cfg:
                    del cfg["experiment"]

    # Merge with strict schema
    cfg = OmegaConf.merge(OmegaConf.structured(Config), cfg)

    if OmegaConf.is_missing(cfg, "output_dir"):
        logger.warning("Config 'output_dir' is still missing. Defaulting to 'outputs'.")
        cfg.output_dir = "outputs"

    # --- Distributed Training Setup ---
    is_ddp = "WORLD_SIZE" in os.environ
    if is_ddp:
        rank, local_rank, world_size = setup_ddp()
    else:
        rank, local_rank, world_size = 0, 0, 1

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # --- Deterministic Seeding ---
    # IMPORTANT: Offset seed by rank to prevent identical noise/dropout on all GPUs
    base_seed = getattr(cfg, "seed", 42)
    set_seed(base_seed + rank)

    # --- W&B Initialization (Rank 0 only) ---
    if rank == 0:
        try:
            wandb.init(
                project="koopman-autoencoder",
                config=OmegaConf.to_container(cfg, resolve=True),
                name=f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                resume="allow",
            )
            logger.info("Weights & Biases initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}. Running without logging.")
            wandb.init(mode="disabled")

    # --- Output Directory ---
    run_name = wandb.run.name if (wandb.run and wandb.run.name) else "local_run"
    output_dir = (
        Path(cfg.output_dir) / run_name
        if rank == 0
        else Path(cfg.output_dir) / "local_run"
    )

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

    # Wait for Rank 0 to create the directory structure
    if is_ddp:
        dist.barrier()

    # --- Dataset and DataLoader Creation ---
    logger.info("Loading datasets...")
    try:
        train_dataset, val_dataset, test_dataset = load_datasets(cfg)
        logger.info(f"Datasets loaded. Type: {cfg.data.dataset_type}")
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        logger.critical(f"Dataset loading failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Creating dataloaders...")
    if is_ddp:
        train_loader, val_loader, test_loader = create_ddp_dataloaders(
            train_dataset, val_dataset, test_dataset, cfg.training, rank, world_size
        )
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            test_dataset,
            cfg.training,
            num_workers=cfg.data.num_workers,
        )
    logger.info("Dataloaders created.")

    # --- Model Initialization ---
    logger.info("Initializing KoopmanAutoencoder model...")
    try:
        model = KoopmanAutoencoder(
            data_variables=cfg.data.variables,
            input_frames=cfg.data.input_sequence_length,
            height=cfg.model.height,
            width=cfg.model.width,
            latent_dim=cfg.model.latent_dim,
            cond_embedding_dim=cfg.model.cond_embedding_dim,
            cond_type=cfg.model.cond_type,
            operator_mode=cfg.model.operator_mode,
            hidden_dims=cfg.model.hidden_dims,
            transformer_config=cfg.model.transformer,
            use_checkpoint=cfg.training.use_checkpoint,
            predict_cond=cfg.model.predict_cond,
            cond_grad_enabled=cfg.model.cond_grad_enabled,
            disturb_std=cfg.model.disturb_std,
            is_continuous=cfg.model.is_continuous,
            rank=cfg.model.rank,
            cond_expansion_type=cfg.data.selection_param,
            use_attention=cfg.model.use_attention,
            **cfg.model.conv_kwargs,
        ).to(device)

        if is_ddp:
            # Convert BatchNorm to SyncBatchNorm for better stats across GPUs
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            # find_unused_parameters=False is faster and safer if your graph is connected
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    except Exception as e:
        logger.critical(f"Model initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # --- Optimizer, Loss, Metrics, and Scheduler ---
    model_params = model.module.parameters() if is_ddp else model.parameters()
    optimizer = optim.AdamW(model_params, lr=cfg.lr_scheduler.lr, weight_decay=1e-4)

    if cfg.loss.get("ssim_weight", None) is not None:
        criterion = KoopmanLoss(to_unit_range=train_dataset.to_unit_range, **cfg.loss)
    else:
        criterion = KoopmanLoss(**cfg.loss)

    eval_metrics = Metric(**cfg.metric) if cfg.metric else None

    # --- Learning Rate Scheduler ---
    logger.info("Initializing learning rate scheduler...")
    try:
        scheduler_args = OmegaConf.to_container(cfg.lr_scheduler, resolve=True)
        scheduler_args.pop("lr", None)

        if "warmup" not in scheduler_args:
            # Fallback to a default if absolutely necessary
            scheduler_args["warmup"] = getattr(cfg.lr_scheduler, "warmup", 0)

        lr_scheduler = CosineWarmup(optimizer=optimizer, **scheduler_args)
        logger.info("Learning rate scheduler initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize LR scheduler: {e}", exc_info=True)
        sys.exit(1)

    # --- Checkpoint Loading ---
    start_epoch = 0
    if cfg.ckpt:
        logger.info(f"Attempting to load checkpoint from: {cfg.ckpt}")
        model_to_load = model.module if is_ddp else model

        try:
            # DDP: load to specific map_location to avoid VRAM spikes on GPU 0
            map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank} if is_ddp else device
            logger.info(f"Currently training on: {map_location}.")

            # Note: You may need to adapt your load_checkpoint function
            # to accept map_location if it doesn't already.
            # Assuming standard torch.load inside load_checkpoint supports it via kwargs or internally.
            _, _, _, start_epoch = load_checkpoint(
                cfg.ckpt, model_to_load, optimizer, strict=True
            )

            # Ensure optimizer state is on the correct device after loading
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

            # Reset LR if needed or resume scheduler state.
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.lr_scheduler.lr
                param_group["initial_lr"] = cfg.lr_scheduler.lr

            # Re-initialize scheduler
            lr_scheduler = CosineWarmup(
                optimizer=optimizer,
                warmup=0,
                decay=cfg.lr_scheduler.decay,
                final_lr=cfg.lr_scheduler.final_lr,
                last_epoch=cfg.lr_scheduler.last_epoch,
            )
            logger.info(
                f"Checkpoint loaded. Resuming at epoch {start_epoch} with LR {optimizer.param_groups[0]['lr']:.6e}"
            )
        except RuntimeError as e:
            logger.critical(
                f"Fatal checkpoint loading error: {e}. Exiting.", exc_info=True
            )
            sys.exit(1)

    # --- Trainer Initialization ---
    logger.info("Initializing Trainer...")
    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            lr_scheduler=lr_scheduler,
            device=device,
            output_dir=output_dir,
            eval_metrics=eval_metrics,
            start_epoch=start_epoch,
            num_epochs=cfg.training.num_epochs,
            patience=cfg.training.patience,
            log_epoch=cfg.log_epoch,
            save_latest_every=cfg.training.save_latest_every,
            num_visual_batches=cfg.training.num_visual_batches,
            precision=cfg.training.precision,
        )
        logger.info("Trainer initialized.")
    except Exception as e:
        logger.critical(f"Trainer initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # --- Run Training ---
    logger.info(f"Starting training from epoch {start_epoch}...")
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        try:
            history = trainer.run()
            logger.info("Training finished successfully.")
        except Exception as e:
            logger.critical(f"Unhandled error during training: {e}", exc_info=True)
            sys.exit(1)

    # --- Final Evaluation and Saving (Rank 0 only) ---
    if rank == 0:
        logger.info("Evaluating final model on the test set...")
        test_metrics = trainer.evaluate(
            test_loader, epoch=trainer.current_epoch, mode="test"
        )
        logger.info(f"Test Metrics: {test_metrics}")
        if wandb.run:
            wandb.log({"test/final_metrics": test_metrics})

        logger.info("Saving final model and artifacts...")
        final_model_path = output_dir / "final_model.pth"
        model_to_save = model.module if is_ddp else model

        save_data = {
            "model_state_dict": model_to_save.state_dict(),
            "epoch": trainer.current_epoch,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "history": history,
            "test_metrics": test_metrics,
            "data_variables": cfg.data.variables,
        }

        normalizer = train_dataset.normalizer
        if isinstance(normalizer, QuantileNormalizer):
            save_data["normalization_stats"] = {
                "q_lows": normalizer.q_lows,
                "q_highs": normalizer.q_highs,
            }
        elif isinstance(normalizer, MeanStdNormalizer):
            save_data["normalization_stats"] = {
                "means": normalizer.means,
                "stds": normalizer.stds,
            }

        torch.save(save_data, final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        if wandb.run:
            wandb.save(str(final_model_path))
            wandb.finish()

    # --- Cleanup ---
    if is_ddp:
        dist.destroy_process_group()

    logger.info("Script finished.")


@hydra.main(version_base=None, config_path="configs", config_name="experiment/base")
def hydra_train(cfg: DictConfig):
    """
    Main entry point for Hydra.
    """
    # When using TorchRun, we rely on the main() function to handle logic.
    main(cfg)


if __name__ == "__main__":
    hydra_train()
