import torch
import numpy as np
from torch import Tensor
from torch.optim import Optimizer
from tensordict import TensorDict
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from tqdm import tqdm
import wandb
from typing import Optional, Union
from models.autoencoder import KoopmanAutoencoder
from models.loss import KoopmanLoss
from models.lr_schedule import CosineWarmup
from torch.utils.data import DataLoader
from models.visualization import denormalize_and_visualize
from models.metrics import Metric
from models.utils import (
    average_losses,
    accumulate_losses,
    tensor_dict_to_json,
)


class Trainer:
    def __init__(
        self,
        model: KoopmanAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: KoopmanLoss,
        eval_metrics: Metric,
        lr_scheduler: CosineWarmup,
        device: torch.device,
        num_epochs: int = 100,
        patience: int = 10,
        output_dir: Optional[Union[Path, str]] = "/home/koopman/",
        start_epoch: int = 0,
        log_epoch: int = 10,
    ):
        """
        Initializes the Trainer class.

        Args:
            model: PyTorch model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            optimizer: Optimizer for training.
            eval_metrics: Evaluation metrics for validation.
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
        self.eval_metrics = eval_metrics
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: dict[str, dict[str, list[float]]] = {}
        self.output_dir = Path(output_dir) if output_dir else None
        self.start_epoch = start_epoch
        self.log_epoch = log_epoch
        if self.output_dir:
            self.output_dir.mkdir(exist_ok=True)

    def train_step(self, input: TensorDict, target: TensorDict):
        """
        Performs a single training step.

        Args:
            input: Input TensorDict.
            target: Target TensorDict.

        Returns:
            Losses for the step (including total loss).
        """
        self.model.train()
        input, target = input.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()

        # Forward pass
        out = self.model(input, seq_length=target["seq_length"])

        # Compute loss
        losses = self.criterion(
            out.x_recon, out.x_preds, out.z_preds, input[:, -1], target, out.reynolds
        )
        losses["total_loss"].backward()

        self.optimizer.step()

        # Log gradient and parameter norms
        grad_norms = {
            f"gradient_norms/{name}": param.grad.norm(2).item()
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }
        param_norms = {
            f"parameter_norms/{name}": param.norm(2).item()
            for name, param in self.model.named_parameters()
        }

        wandb.log(grad_norms)
        wandb.log(param_norms)

        # Compute evaluation metric if available
        if self.eval_metrics:
            target_denorm = self.train_loader.denormalize(target)
            preds_denorm = self.train_loader.denormalize(out.x_preds)
            metric_value = self.eval_metrics.compute_distance(
                target_denorm, preds_denorm
            )
            losses[f"{self.eval_metrics.mode}_{self.eval_metrics.variable_mode}"] = (
                float(np.mean(metric_value))
            )

        return losses

    def evaluate(self, dataloader: DataLoader, mode: str = "train"):
        """
        Evaluates the model on the given dataloader.

        Args:
            dataloader: DataLoader to evaluate on.
            mode: train/val/test
        Returns:
            TensorDict: Averaged losses over the dataset.
        """
        self.model.eval()
        total_losses = TensorDict({}, batch_size=[])

        with torch.no_grad():
            for input, target in dataloader:
                input, target = input.to(self.device), target.to(self.device)
                out = self.model(input, seq_length=target["seq_length"])
                losses = self.criterion(
                    out.x_recon,
                    out.x_preds,
                    out.z_preds,
                    input[:, -1],
                    target,
                    out.reynolds,
                )

                # Accumulate losses
                total_losses = accumulate_losses(total_losses, losses)

                # Visualization for the first batch
                if self.output_dir:
                    denormalize_and_visualize(
                        input=dataloader.denormalize(input),
                        target=dataloader.denormalize(target),
                        x_recon=dataloader.denormalize(out.x_recon),
                        x_preds=dataloader.denormalize(out.x_preds),
                        output_dir=self.output_dir,
                        mode=mode,
                    )
                break  # Break after the first batch for visualization

        n_batches = len(dataloader)
        total_losses = average_losses(total_losses, n_batches)

        # Compute metric if available
        if self.eval_metrics is not None:
            target_denorm = dataloader.denormalize(target)
            preds_denorm = dataloader.denormalize(out.x_preds)
            metric_value = self.eval_metrics.compute_distance(
                target_denorm, preds_denorm
            )
            metric_mean = float(np.mean(metric_value))
            total_losses[
                f"{self.eval_metrics.mode}_{self.eval_metrics.variable_mode}"
            ] = metric_mean

        return total_losses

    def save_checkpoint(self, epoch: int, val_loss: Tensor):
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

    def log_metrics(self, step: int, losses: dict, mode: str = "train"):
        """
        Logs metrics to W&B.

        Args:
            step: Current step (iteration or epoch).
            losses: Losses for the step as a dictionary.
            mode: Mode for logging (e.g., 'train' or 'val').
        """
        # Prepare the W&B log dictionary
        wandb_log_dict = {"step": step}

        for key, value in losses.items():
            if isinstance(value, dict):  # For nested dictionaries
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, TensorDict):  # Check if it's a TensorDict
                        wandb_log_dict[f"loss/{mode}/{key}_{sub_key}"] = (
                            tensor_dict_to_json(sub_value)
                        )
                    elif isinstance(sub_value, torch.Tensor):  # Check if it's a tensor
                        wandb_log_dict[f"loss/{mode}/{key}_{sub_key}"] = (
                            sub_value.item()
                            if sub_value.numel() == 1
                            else sub_value.cpu().numpy().tolist()
                        )
                    else:  # Handle scalars or other types
                        wandb_log_dict[f"loss/{mode}/{key}_{sub_key}"] = sub_value
            elif isinstance(value, TensorDict):  # For top-level TensorDicts
                for sub_key, sub_value in value.items():
                    wandb_log_dict[f"loss/{mode}/{key}_{sub_key}"] = (
                        tensor_dict_to_json(sub_value)
                    )
            elif isinstance(value, torch.Tensor):  # For top-level tensors
                wandb_log_dict[f"loss/{mode}/{key}"] = (
                    value.item() if value.numel() == 1 else value.cpu().numpy().tolist()
                )
            else:  # For other types (e.g., scalars)
                wandb_log_dict[f"loss/{mode}/{key}"] = value

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
        progress_bar = tqdm(
            range(self.start_epoch, self.num_epochs), desc="Training", unit="epoch"
        )
        global_step = self.start_epoch * len(self.train_loader)
        for epoch in progress_bar:
            # Training step
            for input, target in self.train_loader:
                losses = self.train_step(input, target)
                self.log_metrics(step=global_step, losses=losses, mode="train")
                global_step += 1

            self.lr_scheduler.step()

            # Evaluate on train and validation sets
            if (epoch % self.log_epoch == 0) or (epoch == self.start_epoch):
                train_losses = self.evaluate(dataloader=self.train_loader, mode="train")
                val_losses = self.evaluate(dataloader=self.val_loader, mode="val")
                self.log_metrics(step=epoch, losses=val_losses, mode="val")

            progress_bar.set_postfix(
                {
                    "Train Loss": f"{train_losses['total_loss']:.4f}",
                }
            )

            # Early stopping and checkpoint saving
            if val_losses["total_loss"] and (
                val_losses["total_loss"] < self.best_val_loss
            ):
                self.best_val_loss = val_losses["total_loss"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    progress_bar.write(
                        f"Early stopping triggered after {epoch + 1} epochs"
                    )
                    break
            self.save_checkpoint(epoch, val_losses["total_loss"])

        # Save training history
        self.plot_training_history()
        return self.history
