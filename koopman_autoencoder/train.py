import torch
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import wandb  # Import Weights & Biases
import yaml
from models.autoencoder import KoopmanAutoencoder
from models.loss import KoopmanLoss
from models.lr_schedule import CosineWarmup
from models.dataloader import QGDataset, create_dataloaders
from models.trainer import Trainer
from models.utils import load_checkpoint


def main(config_path):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize W&B
    wandb.init(project="koopman-autoencoder", config=config)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare output directory
    output_dir = Path(config["output_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    data_dir = Path(config["data"]["data_dir"])
    train_dataset = QGDataset(
        data_dir / config["data"]["train_file"],
        max_sequence_length=config["data"]["max_sequence_length"],
    )
    val_dataset = QGDataset(
        data_dir / config["data"]["val_file"],
        max_sequence_length=config["data"]["max_sequence_length"],
    )
    test_dataset = QGDataset(
        data_dir / config["data"]["test_file"],
        max_sequence_length=config["data"]["max_sequence_length"],
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        config=config,
    )

    # Initialize model
    model = KoopmanAutoencoder(
        input_channels=config["model"]["input_channels"],
        height=config["model"]["height"],
        width=config["model"]["width"],
        latent_dim=config["model"]["latent_dim"],
        hidden_dims=config["model"]["hidden_dims"],
        use_checkpoint=config["training"]["use_checkpoint"],
        **config["model"]["conv_kwargs"],
    ).to(device)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config["lr_scheduler"]["lr"])
    criterion = KoopmanLoss(alpha=config["loss"]["alpha"], beta=config["loss"]["beta"])

    lr_scheduler = CosineWarmup(
        optimizer=optimizer,
        warmup=config["lr_scheduler"]["warmup"],
        decay=config["lr_scheduler"]["decay"],
        final_lr=config["lr_scheduler"]["final_lr"],
    )
    # Log model architecture to W&B
    wandb.watch(model, log="all")

    # Load checkpoint if specified
    start_epoch = 0
    history = {}
    if config["ckpt"] is not None:
        model, optimizer, history, start_epoch = load_checkpoint(
            checkpoint_path=config["ckpt"], model=model, optimizer=optimizer
        )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        device=device,
        num_epochs=config["training"]["num_epochs"],
        patience=config["training"]["patience"],
        output_dir=output_dir,
        start_epoch=start_epoch,
        log_epoch=config["log_epoch"],
    )

    # Run training loop
    history = trainer.run_training_loop()

    # Evaluate the model on the test set
    test_metrics = trainer.evaluate(test_loader, mode="test")
    print("\nTest Set Results:")
    print(f"Loss: {test_metrics['total_loss']:.4f}")
    print(f"Reconstruction Loss: {test_metrics['recon_loss']:.4f}")
    print(f"Prediction Loss: {test_metrics['pred_loss']:.4f}")

    # Save final model and log it to W&B
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "means": train_dataset.mins,
            "stds": train_dataset.maxs,
            "variables": train_dataset.variables,
            "test_metrics": test_metrics,
            "history": history,
        },
        output_dir / "final_model.pth",
    )
    wandb.save(str(output_dir / "final_model.pth"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Koopman Autoencoder with configuration file"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    main(args.config)
