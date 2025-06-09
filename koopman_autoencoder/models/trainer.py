import torch
import numpy as np
from torch import Tensor
from torch.optim import Optimizer
from tensordict import TensorDict
import torch.distributed as dist
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from tqdm import tqdm
import wandb
import logging
from typing import Optional, Union, Dict, List

# Local imports
from models.autoencoder import KoopmanAutoencoder
from models.loss import KoopmanLoss  # Now using the provided KoopmanLoss
from models.lr_schedule import CosineWarmup
from torch.utils.data import DataLoader
from models.visualization import denormalize_and_visualize
from models.metrics import Metric
from models.utils import (
    average_losses,
    accumulate_losses,
    # tensor_dict_to_json_compatible,
)

logger = logging.getLogger(__name__)


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
        output_dir: Optional[Union[Path, str]] = None,
        start_epoch: int = 0,
        log_epoch: int = 1,
    ):
        """
        Initializes the Trainer class.

        Args:
            model (KoopmanAutoencoder): PyTorch model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            optimizer (Optimizer): Optimizer for training.
            criterion (KoopmanLoss): Loss function.
            eval_metrics (Metric): Evaluation metrics for validation.
            lr_scheduler (CosineWarmup): Learning rate scheduler.
            device (torch.device): Device to train on ('cpu' or 'cuda').
            num_epochs (int): Maximum number of epochs to train.
            patience (int): Number of epochs to wait for improvement before early stopping.
            output_dir (Optional[Union[Path, str]]): Directory to save model checkpoints and logs.
                                                     If None, checkpoints/logs will not be saved.
            start_epoch (int): The epoch to start training from (useful for resuming).
            log_epoch (int): Frequency (in epochs) to perform validation and log metrics.
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
        self.log_epoch = log_epoch

        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # --- History Initialization (Adapted for KoopmanLoss structure) ---
        # The structure of self.history should reflect the scalar values and sub-dictionaries
        # returned by KoopmanLoss.forward() after detachment.
        # It's good to pre-define all keys you expect to track for plotting.
        self.history: Dict[str, Dict[str, List[float]]] = {
            "total_loss": {"train": [], "val": []},
            "latent_loss": {"train": [], "val": []},
            "re_loss": {"train": [], "val": []},
            "recon_loss_sum": {"train": [], "val": []},
            "pred_loss_sum": {"train": [], "val": []},
            f"{self.eval_metrics.mode}_{self.eval_metrics.variable_mode}": {
                "train": [],
                "val": [],
            },
        }
        # Optionally, you could also track per-variable losses, but that makes history complex.
        # Example for per-variable:
        # for var in self.model.data_variables:
        #    self.history[f"recon_loss_{var}"] = {"train": [], "val": []}
        #    self.history[f"pred_loss_{var}"] = {"train": [], "val": []}
        # For simplicity in plotting, summing is often sufficient.

        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_epoch = start_epoch
        self.current_epoch = start_epoch

    @staticmethod
    def is_main_process() -> bool:
        """Checks if the current process is the main (rank 0) process in a DDP setup."""
        return (
            not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        )

    def train_step(self, input: TensorDict, target: TensorDict) -> Dict[str, Tensor]:
        """
        Performs a single training step (one batch).

        Args:
            input (TensorDict): Input data for the model.
            target (TensorDict): Target data for the model.

        Returns:
            Dict[str, Tensor]: Dictionary of raw (scalar) loss tensors for the step.
                               These are detached from the graph, ready for accumulation.
        """
        self.model.train()
        input, target = input.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()

        # Forward pass through the model
        # KoopmanAutoencoder.forward expects a scalar seq_length, already handled by dataloader.py fix
        out = self.model(input, seq_length=target["seq_length"])

        # --- Prepare arguments for KoopmanLoss.forward() ---
        # x_true: This should be the ground truth input used for reconstruction loss.
        # The KoopmanAutoencoder.forward gets its history and present frame from 'input' TensorDict.
        # The KoopmanLoss expects x_true (for recon_loss) and x_future (for rollout_loss).
        # x_true should correspond to the last frame of the input sequence.

        # Extract the last frame of the input TensorDict for x_true
        x_true_recon_input = TensorDict(
            {
                var: input[var][:, -1]  # Get last time step (B, H, W) for each variable
                for var in self.model.data_variables  # Use model's expected data variables
            },
            batch_size=input.batch_size,
        )

        # Pass x_future (ground truth future states) directly as 'target' (it contains the future sequence)
        # reynolds will be 'out.reynolds' which can be None if not predicted

        # Compute loss using the new KoopmanLoss signature
        # KoopmanLoss.forward(x_recon, x_preds, latent_pred, x_true, x_future, reynolds)
        losses_from_criterion = self.criterion(
            x_recon=out.x_recon,  # Reconstructed current frame
            x_preds=out.x_preds,  # Predicted future frames
            latent_pred=out.z_preds,  # Latent space predictions
            x_true=x_true_recon_input,  # Ground truth for reconstruction (last input frame)
            x_future=target,  # Ground truth for future predictions (target sequence)
            reynolds=out.reynolds,  # Predicted Reynolds numbers
        )

        losses_from_criterion["total_loss"].backward()
        self.optimizer.step()

        # Prepare losses for accumulation: extract scalar components and detach
        processed_losses = {
            "total_loss": losses_from_criterion["total_loss"].detach(),
            "latent_loss": torch.tensor(
                losses_from_criterion["latent_loss"], device=self.device
            ),
            "re_loss": torch.tensor(
                losses_from_criterion["re_loss"], device=self.device
            ),
            "recon_loss_sum": torch.tensor(
                sum(losses_from_criterion["recon_loss"].values()), device=self.device
            ),
            "pred_loss_sum": torch.tensor(
                sum(losses_from_criterion["pred_loss"].values()), device=self.device
            ),
        }

        return processed_losses

    def evaluate(self, dataloader: DataLoader, mode: str = "val") -> Dict[str, float]:
        """
        Evaluates the model on the given dataloader.

        Args:
            dataloader (DataLoader): DataLoader to evaluate on.
            mode (str): Evaluation mode, typically "val" or "test".

        Returns:
            Dict[str, float]: Averaged losses and metrics over the dataset.
        """
        self.model.eval()
        total_losses_accumulated: Dict[str, Tensor] = (
            {}
        )  # Initialize as dict for accumulate_losses
        all_metric_values: Dict[str, List[float]] = {}

        visualization_done_for_run = False

        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(dataloader):
                input, target = input.to(self.device), target.to(self.device)

                out = self.model(input, seq_length=target["seq_length"])

                # --- Prepare arguments for KoopmanLoss.forward() ---
                x_true_recon_input = TensorDict(
                    {var: input[var][:, -1] for var in self.model.data_variables},
                    batch_size=input.batch_size,
                )

                losses_from_criterion = self.criterion(
                    x_recon=out.x_recon,
                    x_preds=out.x_preds,
                    latent_pred=out.z_preds,
                    x_true=x_true_recon_input,
                    x_future=target,
                    reynolds=out.reynolds,
                )

                # Accumulate losses (detached from graph)
                # Convert scalars from .item() back to tensors for accumulation if needed,
                # or ensure accumulate_losses handles scalar Python floats directly.
                # Given current accumulate_losses expects Tensors, convert explicitly.
                processed_losses = {
                    "total_loss": losses_from_criterion["total_loss"].detach(),
                    "latent_loss": torch.tensor(
                        losses_from_criterion["latent_loss"], device=self.device
                    ),
                    "re_loss": torch.tensor(
                        losses_from_criterion["re_loss"], device=self.device
                    ),
                    "recon_loss_sum": torch.tensor(
                        sum(losses_from_criterion["recon_loss"].values()),
                        device=self.device,
                    ),
                    "pred_loss_sum": torch.tensor(
                        sum(losses_from_criterion["pred_loss"].values()),
                        device=self.device,
                    ),
                }
                total_losses_accumulated = accumulate_losses(
                    total_losses_accumulated, processed_losses
                )

                # --- Metric Computation ---
                if self.eval_metrics is not None:
                    target_denorm = dataloader.denormalize(target)
                    preds_denorm = dataloader.denormalize(out.x_preds)

                    target_unit = dataloader.to_unit_range(target_denorm)
                    preds_unit = dataloader.to_unit_range(preds_denorm)

                    metric_result = self.eval_metrics.compute_distance(
                        target_unit, preds_unit
                    )

                    metric_key = (
                        f"{self.eval_metrics.mode}_{self.eval_metrics.variable_mode}"
                    )
                    if metric_key not in all_metric_values:
                        all_metric_values[metric_key] = []

                    if isinstance(metric_result, (list, np.ndarray)):
                        all_metric_values[metric_key].extend(metric_result)

                # --- Visualization ---
                if (
                    self.output_dir
                    and self.is_main_process()
                    and not visualization_done_for_run
                ):
                    denormalize_and_visualize(
                        input=dataloader.denormalize(input),
                        target=dataloader.denormalize(target),
                        x_recon=dataloader.denormalize(out.x_recon),
                        x_preds=dataloader.denormalize(out.x_preds),
                        output_dir=self.output_dir,
                        mode=mode,
                        # epoch=self.current_epoch # Pass current epoch for unique filenames
                    )
                    visualization_done_for_run = True

        n_batches = len(dataloader)
        averaged_losses_py_floats = average_losses(total_losses_accumulated, n_batches)

        # Compute and add averaged metrics
        for metric_key, values_list in all_metric_values.items():
            if values_list:
                averaged_losses_py_floats[metric_key] = float(np.mean(values_list))
            else:
                averaged_losses_py_floats[metric_key] = float("nan")

        return averaged_losses_py_floats

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        Saves the model checkpoint.

        Args:
            epoch (int): Current epoch.
            val_loss (float): Validation loss at the current epoch.
            is_best (bool): True if this is the best model so far.
        """
        if not self.output_dir:
            logger.warning("Output directory not set. Skipping checkpoint saving.")
            return

        # Save a checkpoint of the latest model every epoch (optional, but good for resume)
        # You could make this conditional (e.g., save_latest_every_N_epochs)
        # latest_checkpoint_path = self.output_dir / f"epoch_{epoch:04d}.pth"

        # Data to save
        save_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "history": self.history,  # Save the entire history for resume
        }

        if self.is_main_process():
            # try:
            #     torch.save(save_dict, latest_checkpoint_path)
            #     logger.info(f"Saved latest checkpoint to: {latest_checkpoint_path}")
            #     wandb.save(str(latest_checkpoint_path))  # Log to W&B
            # except Exception as e:
            #     logger.error(f"Failed to save latest checkpoint: {e}")

            # Save best model separately if this is the best so far
            if is_best:
                best_checkpoint_path = self.output_dir / "best_model.pth"
                try:
                    torch.save(save_dict, best_checkpoint_path)
                    logger.info(
                        f"Saved BEST model to: {best_checkpoint_path} (Val Loss: {val_loss:.4f})"
                    )
                    wandb.save(str(best_checkpoint_path))  # Log to W&B
                except Exception as e:
                    logger.error(f"Failed to save best model checkpoint: {e}")

    def log_metrics(self, step: int, losses: Dict[str, float], mode: str = "train"):
        """
        Logs metrics to W&B.

        Args:
            step (int): Current step (iteration for train, epoch for val/test).
            losses (Dict[str, float]): Losses/metrics for the step as a dictionary (Python floats).
            mode (str): Mode for logging (e.g., 'train', 'val', 'test').
        """
        wandb_log_dict = {"step": float(step)}

        for key, value in losses.items():
            wandb_log_dict[f"{mode}/{key}"] = value

        # Log specific norms if in train mode (and not too frequently)
        # This is handled in run_training_loop to control frequency better

        if self.is_main_process():
            wandb.log(wandb_log_dict)
            logger.info(
                f"Epoch {self.current_epoch}, {mode.capitalize()} Losses: {losses}"
            )

    def plot_training_history(self):
        """
        Plots and saves the training history.
        """
        if not self.output_dir:
            logger.warning(
                "Output directory not set. Skipping plotting training history."
            )
            return

        plt.figure(figsize=(12, 6))

        for key, data_dict in self.history.items():
            # Only plot if there's data for this key
            if "train" in data_dict and data_dict["train"]:
                plt.plot(
                    range(self.start_epoch, self.current_epoch + 1),
                    data_dict["train"],
                    label=f"Train {key}",
                )
            if "val" in data_dict and data_dict["val"]:
                plt.plot(
                    range(self.start_epoch, self.current_epoch + 1),
                    data_dict["val"],
                    label=f"Val {key}",
                )

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training History")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        history_plot_path = self.output_dir / "training_history.png"
        try:
            plt.savefig(history_plot_path)
            logger.info(f"Training history plot saved to: {history_plot_path}")
        except Exception as e:
            logger.error(f"Failed to save training history plot: {e}")
        plt.close()

        history_yaml_path = self.output_dir / "training_history.yaml"
        try:
            with open(history_yaml_path, "w") as f:
                yaml.dump(self.history, f, default_flow_style=False)
            logger.info(f"Training history saved to: {history_yaml_path}")
        except Exception as e:
            logger.error(f"Failed to save training history to YAML: {e}")

    def run_training_loop(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Runs the full training loop with validation and early stopping.

        Returns:
            Dict[str, Dict[str, List[float]]]: The complete training history.
        """
        progress_bar = tqdm(
            range(self.start_epoch, self.num_epochs), desc="Training", unit="epoch"
        )

        global_step = self.start_epoch * len(self.train_loader)

        # Initial evaluation and logging at start_epoch (or resume)
        logger.info("Performing initial evaluation...")
        # Evaluate train and val datasets to get initial losses/metrics
        train_losses_initial = self.evaluate(dataloader=self.train_loader, mode="train")
        val_losses_initial = self.evaluate(dataloader=self.val_loader, mode="val")

        # Log initial metrics
        self.log_metrics(
            step=self.current_epoch, losses=train_losses_initial, mode="train"
        )
        self.log_metrics(step=self.current_epoch, losses=val_losses_initial, mode="val")

        # Populate history with initial values from evaluation
        for key, value in train_losses_initial.items():
            if key in self.history:
                self.history[key]["train"].append(value)
        for key, value in val_losses_initial.items():
            if key in self.history:
                self.history[key]["val"].append(value)

        # Check for initial best loss for early stopping logic
        if val_losses_initial["total_loss"] < self.best_val_loss:
            self.best_val_loss = val_losses_initial["total_loss"]
            self.patience_counter = 0
            self.save_checkpoint(self.current_epoch, self.best_val_loss, is_best=True)
        else:
            self.patience_counter += 1

        logger.info("Training...")
        for epoch in progress_bar:
            self.current_epoch = (
                epoch  # Update current_epoch for logging and visualization
            )

            # Training phase
            self.model.train()  # Ensure model is in train mode
            epoch_train_losses_accumulated: Dict[str, Tensor] = (
                {}
            )  # Accumulate Tensor losses for the epoch

            for batch_idx, (input, target) in enumerate(self.train_loader):
                batch_losses = self.train_step(input, target)
                epoch_train_losses_accumulated = accumulate_losses(
                    epoch_train_losses_accumulated, batch_losses
                )

                # Log gradient/parameter norms (less frequently, e.g., every N batches)
                if (
                    self.is_main_process() and global_step % 100 == 0
                ):  # Log every 100 steps
                    grad_norms = {
                        f"train/grad_norms/{name}": param.grad.norm(2).item()
                        for name, param in self.model.named_parameters()
                        if param.grad is not None
                        and param.grad.norm(2).item() < float("inf")  # Avoid inf values
                    }
                    param_norms = {
                        f"train/param_norms/{name}": param.norm(2).item()
                        for name, param in self.model.named_parameters()
                    }
                    wandb.log({**grad_norms, **param_norms, "step": global_step})

                global_step += 1

            # Update LR after each epoch
            self.lr_scheduler.step()

            # --- End of Epoch Evaluation and Logging ---
            # Average training losses for the epoch (convert to Python floats)
            train_losses_epoch_avg = average_losses(
                epoch_train_losses_accumulated, len(self.train_loader)
            )
            # Log these epoch-averaged training losses
            self.log_metrics(
                step=self.current_epoch, losses=train_losses_epoch_avg, mode="train"
            )

            # Populate history with epoch-averaged training losses
            for key, value in train_losses_epoch_avg.items():
                if key in self.history:
                    # Append only if it's the first time for this epoch (or handle resume carefully)
                    # For current logic, this is safe because it's called once per epoch loop
                    if len(self.history[key]["train"]) <= (
                        self.current_epoch - self.start_epoch
                    ):
                        self.history[key]["train"].append(value)

            # Only evaluate validation set and perform early stopping/checkpointing if at a log_epoch interval
            if (self.current_epoch % self.log_epoch == 0) or (
                self.current_epoch == self.num_epochs - 1
            ):
                val_losses_epoch_avg = self.evaluate(
                    dataloader=self.val_loader, mode="val"
                )
                self.log_metrics(
                    step=self.current_epoch, losses=val_losses_epoch_avg, mode="val"
                )

                # Populate history with epoch-averaged validation losses
                for key, value in val_losses_epoch_avg.items():
                    if key in self.history:
                        if len(self.history[key]["val"]) <= (
                            self.current_epoch - self.start_epoch
                        ):
                            self.history[key]["val"].append(value)

                # Check for early stopping and checkpoint saving based on validation loss
                current_val_total_loss = val_losses_epoch_avg.get(
                    "total_loss", float("inf")
                )
                if current_val_total_loss < self.best_val_loss:
                    self.best_val_loss = current_val_total_loss
                    self.patience_counter = 0
                    self.save_checkpoint(
                        self.current_epoch, self.best_val_loss, is_best=True
                    )
                else:
                    self.patience_counter += 1
                    logger.info(
                        f"Patience counter: {self.patience_counter}/{self.patience}"
                    )
                    if self.patience_counter >= self.patience:
                        progress_bar.write(
                            f"Early stopping triggered after {self.current_epoch + 1} epochs"
                        )
                        break

            # Always save the latest checkpoint at the end of each epoch loop
            # This is separate from 'best_model.pth' and is useful for simple resume
            self.save_checkpoint(
                self.current_epoch,
                val_losses_epoch_avg.get("total_loss", float("nan")),
                is_best=False,
            )

            progress_bar.set_postfix(
                {
                    "Train Loss": f"{train_losses_epoch_avg.get('total_loss', float('nan')):.4f}",
                    "Val Loss": f"{val_losses_epoch_avg.get('total_loss', float('nan')):.4f}",
                    "Best Val Loss": f"{self.best_val_loss:.4f}",
                    "Patience": f"{self.patience_counter}/{self.patience}",
                }
            )

        # Ensure history is fully populated before plotting
        self.plot_training_history()
        return self.history
