import torch
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from tqdm import tqdm
import wandb


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        num_epochs=100,
        patience=10,
        output_dir=None,
    ):
        """
        Initializes the Trainer class.

        Args:
            model: PyTorch model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            optimizer: Optimizer for training.
            criterion: Loss function.
            device: Device to train on ('cpu' or 'cuda').
            num_epochs: Maximum number of epochs to train.
            patience: Number of epochs to wait for improvement before early stopping.
            output_dir: Directory to save model checkpoints and logs.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_recon_loss": [],
            "val_recon_loss": [],
            "train_pred_loss": [],
            "val_pred_loss": [],
        }
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(exist_ok=True)

    def train_step(self, input, target):
        """
        Performs a single training step.

        Args:
            input: Input tensor.
            target: Target tensor.

        Returns:
            Loss value for the step.
        """
        self.model.train()
        seq_length = target.size(1)
        input, target = input.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()

        # Forward pass
        x_recon, x_preds, z_preds, latent_pred_differences = self.model(
            input, seq_length=seq_length
        )

        # Compute loss
        loss, recon_loss, pred_loss, latent_loss = self.criterion(
            x_recon, x_preds, latent_pred_differences, input[:, -1], target
        )

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_pred_loss = 0
        total_latent_loss = 0

        with torch.no_grad():
            for input, target in dataloader:
                input, target = input.to(self.device), target.to(self.device)
                x_recon, x_preds, z_preds, latent_pred_differences = self.model(
                    input, seq_length=target.size(1)
                )
                loss, recon_loss, pred_loss, latent_loss = self.criterion(
                    x_recon, x_preds, latent_pred_differences, input[:, -1], target
                )
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_pred_loss += pred_loss.item()
                total_latent_loss += latent_loss.item()

        n_batches = len(dataloader)
        return {
            "loss": total_loss / n_batches,
            "recon_loss": total_recon_loss / n_batches,
            "pred_loss": total_pred_loss / n_batches,
            "latent_loss": total_latent_loss / n_batches,
        }

    def save_checkpoint(self, epoch, val_loss):
        """
        Saves the model checkpoint.

        Args:
            epoch: Current epoch.
            val_loss: Validation loss at the current epoch.
        """
        if self.output_dir:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "history": self.history,
                },
                self.output_dir / "best_model.pth",
            )

    def log_metrics(self, epoch, train_metrics, val_metrics):
        """
        Logs metrics to W&B and updates the history.

        Args:
            epoch: Current epoch.
            train_metrics: Training metrics for the epoch.
            val_metrics: Validation metrics for the epoch.
        """
        self.history["train_loss"].append(train_metrics["loss"])
        self.history["val_loss"].append(val_metrics["loss"])
        self.history["train_recon_loss"].append(train_metrics["recon_loss"])
        self.history["val_recon_loss"].append(val_metrics["recon_loss"])
        self.history["train_pred_loss"].append(train_metrics["pred_loss"])
        self.history["val_pred_loss"].append(val_metrics["pred_loss"])

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_recon_loss": train_metrics["recon_loss"],
                "val_recon_loss": val_metrics["recon_loss"],
                "train_pred_loss": train_metrics["pred_loss"],
                "val_pred_loss": val_metrics["pred_loss"],
            }
        )

    def plot_training_history(self):
        """
        Plots and saves the training history.
        """
        if self.output_dir:
            plt.figure(figsize=(12, 6))
            plt.plot(self.history["train_loss"], label="Train Loss")
            plt.plot(self.history["val_loss"], label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training History")
            plt.legend()
            plt.savefig(self.output_dir / "training_history.png")
            plt.close()

            with open(self.output_dir / "training_history.yaml", "w") as f:
                yaml.dump(self.history, f)

    def run_training_loop(self):
        """
        Runs the full training loop with validation and early stopping.

        Returns:
            Dictionary with training history.
        """
        progress_bar = tqdm(range(self.num_epochs), desc="Training", unit="epoch")
        for epoch in progress_bar:
            # Training step
            for input, target in self.train_loader:
                self.train_step(input, target)

            # Evaluate on train and validation sets
            train_metrics = self.evaluate(self.train_loader)
            val_metrics = self.evaluate(self.val_loader)

            # Log metrics and update progress bar
            self.log_metrics(epoch, train_metrics, val_metrics)
            progress_bar.set_postfix(
                {
                    "Train Loss": f"{train_metrics['loss']:.4f}",
                    "Val Loss": f"{val_metrics['loss']:.4f}",
                }
            )

            # Early stopping and checkpoint saving
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics["loss"])
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    progress_bar.write(
                        f"Early stopping triggered after {epoch + 1} epochs"
                    )
                    break

        # Save training history
        self.plot_training_history()
        return self.history
