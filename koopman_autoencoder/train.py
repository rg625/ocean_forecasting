import torch
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import wandb
import logging
import argparse
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from typing import Dict, List

# Assuming these imports are correctly defined in your project
from models.cnn import TransformerConfig
from models.config_classes import Config
from models.autoencoder import KoopmanAutoencoder
from models.loss import KoopmanLoss
from models.lr_schedule import CosineWarmup
from models.dataloader import create_dataloaders
from models.trainer import Trainer
from models.metrics import Metric
from models.dataloader import QGDatasetQuantile
from models.utils import (
    load_checkpoint,
    get_dataset_class_and_kwargs,
    load_datasets,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(config_path: str):
    """
    Main function to load configuration, set up model, data, optimizer,
    and run the training loop for the Koopman Autoencoder.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    try:
        # Load configuration using Omegaconf
        # The base_config from the dataclass provides defaults and structure
        base_config = OmegaConf.structured(Config)
        file_config = OmegaConf.load(config_path)

        # Merge the loaded file config with the structured base config
        # This will validate against the dataclass schema
        cfg = OmegaConf.merge(base_config, file_config)

        # You can add CLI overrides if needed (e.g., from argparse)
        # cli_config = OmegaConf.from_cli()
        # cfg = OmegaConf.merge(cfg, cli_config)

        # Resolve interpolations and defaults
        OmegaConf.resolve(cfg)

        logger.info(
            f"Configuration loaded from {config_path}:\n{OmegaConf.to_yaml(cfg)}"
        )

    except (FileNotFoundError, ValueError, OmegaConfBaseException) as e:
        logger.error(f"Configuration error: {e}")
        exit(1)  # Exit if config is invalid

    # Initialize W&B
    try:
        # wandb.config can directly take an Omegaconf object, but resolving it
        # to a plain dict is often cleaner for logging.
        wandb.init(
            project="koopman-autoencoder",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        logger.info("Weights & Biases initialized.")
    except Exception as e:
        logger.error(
            f"Failed to initialize Weights & Biases: {e}. Continuing without W&B logging."
        )
        # Optionally, set a flag to disable wandb logging throughout if it fails to init

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Prepare output directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.output_dir) / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created at: {output_dir}")

    # Get dataset class and kwargs
    try:
        dataset_class, dataset_kwargs = get_dataset_class_and_kwargs(cfg)
    except ValueError as e:
        logger.error(f"Dataset configuration error: {e}")
        exit(1)

    # Load datasets
    try:
        train_dataset, val_dataset, test_dataset = load_datasets(
            cfg, dataset_class, dataset_kwargs
        )
    except RuntimeError as e:
        logger.error(f"Dataset loading failed: {e}")
        exit(1)

    # Crucial: Get the data variables from the dataset for model initialization
    # Assuming all datasets (train, val, test) have the same 'variables' attribute
    # and it's a list of strings, consistent with what KoopmanAutoencoder expects.
    data_variables = train_dataset.variables
    if not data_variables:
        logger.error(
            "No data variables found in the training dataset. Cannot initialize model."
        )
        exit(1)
    logger.info(f"Data variables identified: {data_variables}")

    # Create DataLoaders
    try:
        # Pass Omegaconf object directly
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            config=cfg,  # Pass the OmegaConf object
        )
        logger.info("DataLoaders created successfully.")
    except Exception as e:
        logger.error(f"Failed to create DataLoaders: {e}")
        exit(1)

    # Initialize model
    try:
        model = KoopmanAutoencoder(
            input_frames=cfg.data.input_sequence_length,
            input_channels=cfg.model.input_channels,  # This should match len(data_variables)
            height=cfg.model.height,
            width=cfg.model.width,
            latent_dim=cfg.model.latent_dim,
            hidden_dims=cfg.model.hidden_dims,
            use_checkpoint=cfg.training.use_checkpoint,
            # TransformerConfig expects a plain dict for its kwargs, not Omegaconf.DictConfig
            # OmegaConf.to_container converts DictConfig to a plain dict
            transformer_config=TransformerConfig(
                **OmegaConf.to_container(cfg.model.transformer, resolve=True)
            ),
            predict_re=cfg.model.predict_re,
            data_variables=data_variables,  # Pass the data_variables
            **cfg.model.conv_kwargs,  # Omegaconf DictConfig can be unpacked like a dict
        ).to(device)
        logger.info("KoopmanAutoencoder model initialized.")
        logger.debug(model)  # Log model architecture for debugging
    except Exception as e:
        logger.error(f"Failed to initialize KoopmanAutoencoder: {e}")
        exit(1)

    # Initialize optimizer and loss function
    try:
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr_scheduler.lr)

        criterion = KoopmanLoss(
            alpha=cfg.loss.alpha,
            beta=cfg.loss.beta,
            re_weight=cfg.loss.re_weight,
            weighting_type=cfg.loss.weighting_type,
            sigma_blur=cfg.loss.sigma_blur,
        )
        logger.info("Optimizer and Loss function initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize optimizer or loss: {e}")
        exit(1)

    # Initialize evaluation metrics
    try:
        eval_metrics = Metric(
            mode=cfg.metric.type, variable_mode=cfg.metric.variable_mode
        )
        logger.info("Evaluation metrics initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize metrics: {e}")
        exit(1)

    # Initialize learning rate scheduler
    try:
        lr_scheduler = CosineWarmup(
            optimizer=optimizer,
            warmup=cfg.lr_scheduler.warmup,
            decay=cfg.lr_scheduler.decay,
            final_lr=cfg.lr_scheduler.final_lr,
        )
        logger.info("Learning rate scheduler initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize LR scheduler: {e}")
        exit(1)

    # Log model architecture to W&B
    try:
        wandb.watch(model, log="all")
        logger.info("Model architecture logged to W&B.")
    except Exception as e:
        logger.warning(
            f"Failed to log model to W&B (perhaps W&B not initialized correctly): {e}"
        )

    # Load checkpoint if specified
    start_epoch: int = 0
    history: Dict[str, Dict[str, List[float]]] = {
        "total_loss": {"train": [], "val": []},
        "latent_loss": {"train": [], "val": []},
        "re_loss": {"train": [], "val": []},
        "recon_loss_sum": {"train": [], "val": []},
        "pred_loss_sum": {"train": [], "val": []},
        f"{cfg.metric.type}_{cfg.metric.variable_mode}": {
            "train": [],
            "val": [],
        },
    }

    if cfg.ckpt is not None:  # Accessing optional field directly via dot notation
        try:
            model, optimizer, history, start_epoch = load_checkpoint(
                checkpoint_path=cfg.ckpt, model=model, optimizer=optimizer
            )
            # After loading, move model/optimizer to correct device
            model.to(device)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        except RuntimeError as e:  # Catch specific error from load_checkpoint
            logger.error(f"Checkpoint loading failed fatally: {e}. Exiting.")
            exit(1)
        except Exception as e:
            logger.warning(
                f"An unexpected error occurred during checkpoint loading: {e}. Starting from scratch."
            )
            # Fallback to default if a non-critical error occurred
            # Re-initialize model and optimizer to their initial states
            model = KoopmanAutoencoder(
                input_frames=cfg.data.input_sequence_length,
                input_channels=cfg.model.input_channels,
                height=cfg.model.height,
                width=cfg.model.width,
                latent_dim=cfg.model.latent_dim,
                hidden_dims=cfg.model.hidden_dims,
                use_checkpoint=cfg.training.use_checkpoint,
                transformer_config=TransformerConfig(
                    **OmegaConf.to_container(cfg.model.transformer, resolve=True)
                ),
                predict_re=cfg.model.predict_re,
                data_variables=data_variables,
                **cfg.model.conv_kwargs,
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr_scheduler.lr)

    # Initialize Trainer
    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            eval_metrics=eval_metrics,
            lr_scheduler=lr_scheduler,
            device=device,
            num_epochs=cfg.training.num_epochs,
            patience=cfg.training.patience,
            output_dir=output_dir,
            start_epoch=start_epoch,
            log_epoch=cfg.log_epoch,
        )
        logger.info("Trainer initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Trainer: {e}")
        exit(1)

    # Run training loop
    try:
        history = trainer.run_training_loop()
        logger.info("Training loop completed.")
    except Exception as e:
        logger.critical(
            f"An unhandled error occurred during training: {e}", exc_info=True
        )
        exit(1)

    # Evaluate the model on the test set
    logger.info("Evaluating model on the test set...")
    try:
        test_metrics = trainer.evaluate(test_loader, mode="test")
        logger.info("\nTest Set Results:")
        logger.info(f"Loss: {test_metrics.get('total_loss', 'N/A'):.4f}")
        logger.info(f"Reconstruction Loss: {test_metrics.get('recon_loss', 'N/A'):.4f}")
        logger.info(f"Prediction Loss: {test_metrics.get('pred_loss', 'N/A'):.4f}")
        wandb.log({"test_metrics": test_metrics})
    except Exception as e:
        logger.error(f"Error during test set evaluation: {e}")
        # test_metrics = {"evaluation_error": str(e)}  # Record error in metrics

    # Save final model and log it to W&B
    final_model_path = output_dir / "final_model.pth"
    try:
        # Save normalization stats and variables from the training dataset
        # This will depend on the type of dataset used (QGDatasetBase, QGDatasetQuantile, MultipleSims)
        # All these derive from QGDatasetBase and have .variables and .mins/.maxs (or .q_lows/.q_highs)
        # It's safest to save the specific normalization params based on the dataset type.
        save_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": trainer.current_epoch,
            "variables": train_dataset.variables,
            "test_metrics": test_metrics,
            "history": history,
            "config": OmegaConf.to_container(
                cfg, resolve=True
            ),  # Save the resolved config
        }
        if isinstance(train_dataset, QGDatasetQuantile):
            save_data["q_lows"] = train_dataset.q_lows
            save_data["q_highs"] = train_dataset.q_highs
        else:  # Covers QGDatasetBase and MultipleSims
            save_data["means"] = train_dataset.means
            save_data["stds"] = train_dataset.stds

        torch.save(save_data, final_model_path)

        logger.info(f"Final model saved to: {final_model_path}")
        wandb.save(str(final_model_path))
        logger.info("Final model uploaded to W&B.")
    except Exception as e:
        logger.error(f"Failed to save or upload final model: {e}")

    # Finalize W&B run
    wandb.finish()
    logger.info("W&B run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Koopman Autoencoder with configuration file"
    )
    # OmegaConf can automatically parse CLI arguments that match the config structure
    # For example: python main.py --training.num_epochs 100 --model.latent_dim 256
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    main(args.config)
