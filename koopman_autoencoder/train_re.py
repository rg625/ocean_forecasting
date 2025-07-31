# main.py
import torch
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import wandb
import logging
import argparse
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import OmegaConfBaseException
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Refactored project imports
from models.config_classes import Config
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
from models.utils import load_datasets  # Removed unused load_checkpoint import

# Configure logging for clear and informative output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    logger.info(f"DDP setup: Rank {rank}/{world_size}, Local Rank {local_rank}")
    return rank, local_rank, world_size


def main(cfg: DictConfig):
    """Main function to set up and run the training pipeline."""
    # --- Distributed Training Setup ---
    is_ddp = "WORLD_SIZE" in os.environ
    rank, local_rank, world_size = setup_ddp() if is_ddp else (0, 0, 1)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # --- W&B Initialization (only on main process) ---
    if rank == 0:
        try:
            wandb.init(
                project="koopman-autoencoder",
                config=OmegaConf.to_container(cfg, resolve=True),
                name=f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            logger.info("Weights & Biases initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}. Running without logging.")
            wandb.init(mode="disabled")

    # --- Output Directory (only on main process) ---
    output_dir = (
        Path(cfg.output_dir) / wandb.run.name
        if rank == 0 and wandb.run.name is not None
        else Path(cfg.output_dir) / "local_run"
    )
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

    # --- Dataset and DataLoader Creation ---
    logger.info("Loading datasets...")
    try:
        train_dataset, val_dataset, test_dataset = load_datasets(cfg)
        logger.info(f"Datasets loaded. Type: {cfg.data.dataset_type}")
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        logger.critical(f"Dataset loading failed: {e}", exc_info=True)
        exit(1)

    logger.info("Creating dataloaders...")
    if is_ddp:
        train_loader, val_loader, test_loader = create_ddp_dataloaders(
            train_dataset, val_dataset, test_dataset, cfg.training, rank, world_size
        )
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset, cfg.training
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
            operator_mode=cfg.model.operator_mode,
            hidden_dims=cfg.model.hidden_dims,
            transformer_config=cfg.model.transformer,
            use_checkpoint=cfg.training.use_checkpoint,
            predict_re=cfg.model.predict_re,
            re_grad_enabled=cfg.model.re_grad_enabled,
            **cfg.model.conv_kwargs,
        ).to(device)
        if is_ddp:
            model = DDP(model, device_ids=[local_rank])
    except Exception as e:
        logger.critical(f"Model initialization failed: {e}", exc_info=True)
        exit(1)

    # --- CORRECTED AND ROBUST SETUP FOR FINE-TUNING ---

    model_to_load = model.module if is_ddp else model

    # 1. Load ONLY the Model Weights from the Checkpoint
    # We do not need the old optimizer state for fine-tuning.
    start_epoch = 0
    if cfg.ckpt:
        logger.info(f"Loading model weights from checkpoint: {cfg.ckpt}")
        try:
            # Load the entire checkpoint dictionary to the CPU first
            checkpoint = torch.load(cfg.ckpt, map_location="cpu")

            # Load only the model's state dictionary.
            # This completely avoids issues with optimizer state mismatch.
            model_to_load.load_state_dict(checkpoint["model_state_dict"])

            # The epoch number is for logging continuity.
            # The fine-tuning training process (optimizer, lr_scheduler) starts fresh.
            start_epoch = checkpoint.get("epoch", -1) + 1
            logger.info(
                f"Successfully loaded model weights. Fine-tuning will start from epoch {start_epoch}."
            )
        except (KeyError, FileNotFoundError, RuntimeError) as e:
            logger.critical(
                f"Fatal checkpoint loading error: {e}. Exiting.", exc_info=True
            )
            exit(1)

    # 2. Freeze Parameters for Fine-Tuning
    logger.info("Freezing model parameters for fine-tuning the Reynolds predictor...")
    for param in model_to_load.parameters():
        param.requires_grad = False

    if (
        hasattr(model_to_load, "re_predictor")
        and model_to_load.re_predictor is not None
    ):
        for param in model_to_load.re_predictor.parameters():
            param.requires_grad = True
        logger.info("Successfully unfroze the 're_predictor' network.")
    else:
        logger.error(
            "'re_predictor' not found or is None. Cannot proceed with fine-tuning."
        )
        exit(1)

    # 3. Create the REAL optimizer with ONLY the trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.critical(
            "No trainable parameters found after freezing. Ensure 're_predictor' was unfrozen correctly."
        )
        exit(1)

    optimizer = optim.Adam(trainable_params, lr=cfg.lr_scheduler.lr)
    logger.info("Optimizer initialized with only the 're_predictor' parameters.")

    # 4. Initialize Loss, Metrics, and LR Scheduler with the new optimizer
    criterion = KoopmanLoss(**cfg.loss)
    eval_metrics = Metric(**cfg.metric) if cfg.metric else None

    logger.info("Initializing learning rate scheduler...")
    try:
        scheduler_args = OmegaConf.to_container(cfg.lr_scheduler, resolve=True)
        scheduler_args.pop("lr", None)
        lr_scheduler = CosineWarmup(optimizer=optimizer, **scheduler_args)
        logger.info("Learning rate scheduler initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize LR scheduler: {e}", exc_info=True)
        exit(1)

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
        )
        logger.info("Trainer initialized.")
    except Exception as e:
        logger.critical(f"Trainer initialization failed: {e}", exc_info=True)
        exit(1)

    # --- Run Training ---
    logger.info(f"Starting training from epoch {start_epoch}...")
    try:
        history = trainer.run()
        logger.info("Training finished successfully.")
    except Exception as e:
        logger.critical(f"Unhandled error during training: {e}", exc_info=True)
        exit(1)

    # --- Final Evaluation and Saving (on main process) ---
    if rank == 0:
        logger.info("Evaluating final model on the test set...")
        test_metrics = trainer.evaluate(
            test_loader, epoch=trainer.current_epoch, mode="test"
        )
        logger.info(f"Test Metrics: {test_metrics}")
        wandb.log({"test/final_metrics": test_metrics})

        logger.info("Saving final model and artifacts...")
        final_model_path = output_dir / "final_model.pth"
        model_to_save = model.module if is_ddp else model

        # FIX: Include the fine-tuned optimizer's state in the final save file.
        # This makes the final model a proper, resumable checkpoint.
        save_data = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),  # <-- CORRECTED
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
        logger.info(f"Final model and optimizer state saved to {final_model_path}")
        wandb.save(str(final_model_path))

    if is_ddp:
        dist.destroy_process_group()
    if rank == 0:
        wandb.finish()
    logger.info("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Koopman Autoencoder for Re prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file for fine-tuning",
    )
    args, unknown_args = parser.parse_known_args()

    try:
        base_config = OmegaConf.structured(Config)
        file_config = OmegaConf.load(args.config)
        cli_config = OmegaConf.from_cli(unknown_args)

        cfg = OmegaConf.merge(base_config, file_config, cli_config)
        OmegaConf.resolve(cfg)

        logger.info(
            f"Configuration loaded and merged for fine-tuning:\n{OmegaConf.to_yaml(cfg)}"
        )
        main(cfg)

    except (FileNotFoundError, OmegaConfBaseException) as e:
        logger.critical(f"Configuration setup failed: {e}", exc_info=True)
        exit(1)
