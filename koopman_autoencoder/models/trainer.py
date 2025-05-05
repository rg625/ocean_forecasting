import torch
from tensordict import TensorDict
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from tqdm import tqdm
import wandb
from models.utils import (
    average_losses,
    accumulate_losses,
    denormalize_and_visualize,
    tensor_dict_to_json,
)


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
        self.history: dict[str, dict[str, list[float]]] = {}
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
        input, target = input.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()

        # Forward pass
        x_recon, x_preds, z_preds, latent_pred_differences = self.model(
            input, seq_length=target["seq_length"]
        )

        # Compute loss
        losses = self.criterion(
            x_recon, x_preds, latent_pred_differences, input[:, -1], target
        )
        loss = losses["total_loss"]

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        return losses

    def evaluate(self, dataloader, epoch):
        """
        Evaluates the model on the given dataloader.

        Args:
            dataloader: DataLoader to evaluate on.
            epoch: Current epoch.

        Returns:
            TensorDict: Averaged losses over the dataset.
        """
        self.model.eval()
        total_losses = TensorDict({}, batch_size=[])

        with torch.no_grad():
            for input, target in dataloader:
                input, target = input.to(self.device), target.to(self.device)
                x_recon, x_preds, z_preds, latent_pred_differences = self.model(
                    input, seq_length=target["seq_length"]
                )
                losses = self.criterion(
                    x_recon, x_preds, latent_pred_differences, input[:, -1], target
                )

                # Accumulate losses
                total_losses = accumulate_losses(total_losses, losses)

                # Visualization for the first batch
                if epoch == 0 and self.output_dir:
                    input_denorm = dataloader.denormalize(input)
                    target_denorm = dataloader.denormalize(target)
                    x_preds_denorm = dataloader.denormalize(x_preds)

                    denormalize_and_visualize(
                        epoch,
                        input_denorm,
                        target_denorm,
                        x_recon,
                        x_preds_denorm,
                        self.output_dir,
                    )
                break  # Break after the first batch for visualization

        # Average losses over batches
        n_batches = len(dataloader)
        total_losses = average_losses(total_losses, n_batches)

        return total_losses

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

    def log_metrics(self, step, losses, prefix="train_"):
        """
        Logs metrics to W&B.

        Args:
            step: Current step (iteration or epoch).
            losses: Losses for the step as a dictionary.
            prefix: Prefix for logging (e.g., 'train_' or 'val_').
        """
        # Prepare the W&B log dictionary
        wandb_log_dict = {"step": step}

        for key, value in losses.items():
            if isinstance(value, dict):  # For nested dictionaries
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, TensorDict):  # Check if it's a TensorDict
                        wandb_log_dict[f"{prefix}{key}_{sub_key}"] = (
                            tensor_dict_to_json(sub_value)
                        )
                    elif isinstance(sub_value, torch.Tensor):  # Check if it's a tensor
                        wandb_log_dict[f"{prefix}{key}_{sub_key}"] = (
                            sub_value.item()
                            if sub_value.numel() == 1
                            else sub_value.cpu().numpy().tolist()
                        )
                    else:  # Handle scalars or other types
                        wandb_log_dict[f"{prefix}{key}_{sub_key}"] = sub_value
            elif isinstance(value, TensorDict):  # For top-level TensorDicts
                for sub_key, sub_value in value.items():
                    wandb_log_dict[f"{prefix}{key}_{sub_key}"] = tensor_dict_to_json(
                        sub_value
                    )
            elif isinstance(value, torch.Tensor):  # For top-level tensors
                wandb_log_dict[f"{prefix}{key}"] = (
                    value.item() if value.numel() == 1 else value.cpu().numpy().tolist()
                )
            else:  # For other types (e.g., scalars)
                wandb_log_dict[f"{prefix}{key}"] = value

        # Log to W&B
        wandb.log(wandb_log_dict)

    def plot_training_history(self):
        """
        Plots and saves the training history.
        """
        if self.output_dir:
            plt.figure(figsize=(12, 6))
            for key in self.history.keys():
                plt.plot(self.history[key]["train"], label=f"Train {key}")
                plt.plot(self.history[key]["val"], label=f"Val {key}")
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
        global_step = 0  # Track the global training step
        for epoch in progress_bar:
            # Training step
            for input, target in self.train_loader:
                losses = self.train_step(input, target)
                self.log_metrics(global_step, losses, prefix="train_")
                global_step += 1

            # Evaluate on train and validation sets
            train_losses = self.evaluate(self.train_loader, epoch=epoch)
            val_losses = self.evaluate(self.val_loader, epoch=epoch)

            # Log metrics and update progress bar
            self.log_metrics(epoch, train_losses, prefix="epoch_train_")
            self.log_metrics(epoch, val_losses, prefix="epoch_val_")
            progress_bar.set_postfix(
                {
                    "Train Loss": f"{train_losses['total_loss']:.4f}",
                    "Val Loss": f"{val_losses['total_loss']:.4f}",
                }
            )

            # Early stopping and checkpoint saving
            if val_losses["total_loss"] < self.best_val_loss:
                self.best_val_loss = val_losses["total_loss"]
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_losses["total_loss"])
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
