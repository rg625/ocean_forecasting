# models/trainer.py
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
import logging
from typing import Optional, Union, Dict, List

# Local imports
from .autoencoder import KoopmanAutoencoder
from .loss import KoopmanLoss
from .lr_schedule import CosineWarmup
from .dataloader import DataLoaderWrapper
from .metrics import Metric
from .utils import accumulate_losses, average_losses
from .visualization import denormalize_and_visualize
import wandb

logger = logging.getLogger(__name__)


class Trainer:
    """A robust training orchestrator for PyTorch models with DDP support."""

    def __init__(
        self,
        model: KoopmanAutoencoder,
        train_loader: DataLoaderWrapper,
        val_loader: DataLoaderWrapper,
        optimizer: Optimizer,
        criterion: KoopmanLoss,
        lr_scheduler: CosineWarmup,
        device: torch.device,
        output_dir: Union[Path, str],
        num_epochs: int,
        patience: int,
        log_epoch: int,
        start_epoch: int = 0,
        save_latest_every: int = 1,
        num_visual_batches: int = 1,
        eval_metrics: Optional[Metric] = None,
    ):
        self.model = model
        self.train_loader, self.val_loader = train_loader, val_loader
        self.optimizer, self.criterion, self.lr_scheduler = (
            optimizer,
            criterion,
            lr_scheduler,
        )
        self.eval_metrics = eval_metrics
        self.device = device
        self.num_epochs, self.patience, self.log_epoch = num_epochs, patience, log_epoch
        self.save_latest_every, self.num_visual_batches = (
            num_visual_batches,
            num_visual_batches,
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.start_epoch = start_epoch
        self.current_epoch = start_epoch
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._init_history()

    def _init_history(self):
        self.history: Dict[str, Dict[str, List[float]]] = {
            "total_loss": {"train": [], "val": []},
            "loss_recon": {"train": [], "val": []},
            "loss_pred": {"train": [], "val": []},
            "loss_latent": {"train": [], "val": []},
            "loss_re": {"train": [], "val": []},
        }
        if self.eval_metrics:
            metric_key = (
                f"metric_{self.eval_metrics.mode}_{self.eval_metrics.variable_mode}"
            )
            self.history[metric_key] = {"val": []}

    @staticmethod
    def is_main_process() -> bool:
        return (
            not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        )

    def _gather_and_average_metrics(
        self, metrics: Dict[str, float]
    ) -> Dict[str, float]:
        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() == 1
        ):
            return metrics

        # Ensure all metrics are floats for tensor conversion
        metric_values = [float(v) for v in metrics.values()]
        metric_tensor = torch.tensor(metric_values, device=self.device)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)

        return {key: val.item() for key, val in zip(metrics.keys(), metric_tensor)}

    def _run_one_epoch(self) -> Dict[str, float]:
        """Runs a single training epoch."""
        self.model.train()
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.current_epoch)

        epoch_losses: Dict[str, Tensor] = {}
        for input_td, target_td in self.train_loader:
            input_td, target_td = input_td.to(self.device), target_td.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            model_module = (
                self.model.module
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                else self.model
            )
            out = self.model(input_td, target_td["seq_length"])

            x_true_recon = TensorDict(
                {k: input_td[k][:, -1] for k in model_module.data_variables.keys()},
                batch_size=input_td.batch_size[0],
            )
            loss_dict = self.criterion(
                out.x_recon,
                out.x_preds,
                out.z_preds,
                x_true_recon,
                target_td,
                out.reynolds,
            )

            loss_dict["total_loss"].backward()
            self.optimizer.step()

            detached_losses = {
                k: v.detach() for k, v in loss_dict.items() if isinstance(v, Tensor)
            }
            epoch_losses = accumulate_losses(epoch_losses, detached_losses)

        return average_losses(epoch_losses, len(self.train_loader))

    def evaluate(
        self, dataloader: DataLoaderWrapper, epoch: int, mode: str = "val"
    ) -> Dict[str, float]:
        """Evaluates the model on a given dataloader."""
        self.model.eval()
        total_losses: Dict[str, Tensor] = {}
        all_metric_values: List[float] = []

        with torch.no_grad():
            for i, (input_td, target_td) in enumerate(dataloader):
                input_td, target_td = input_td.to(self.device), target_td.to(
                    self.device
                )

                model_module = (
                    self.model.module
                    if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                    else self.model
                )
                out = self.model(input_td, target_td["seq_length"])

                x_true_recon = TensorDict(
                    {k: input_td[k][:, -1] for k in model_module.data_variables.keys()},
                    batch_size=input_td.batch_size[0],
                )
                loss_dict = self.criterion(
                    out.x_recon,
                    out.x_preds,
                    out.z_preds,
                    x_true_recon,
                    target_td,
                    out.reynolds,
                )
                detached_losses = {
                    k: v.detach() for k, v in loss_dict.items() if isinstance(v, Tensor)
                }
                total_losses = accumulate_losses(total_losses, detached_losses)

                if self.eval_metrics and not out.x_preds.is_empty():
                    preds_denorm = dataloader.denormalize(out.x_preds)
                    target_denorm = dataloader.denormalize(target_td)
                    metric_val = self.eval_metrics(
                        dataloader.to_unit_range(target_denorm),
                        dataloader.to_unit_range(preds_denorm),
                    )
                    all_metric_values.extend(np.atleast_1d(metric_val.cpu().numpy()))

                if self.is_main_process() and i < self.num_visual_batches:
                    denormalize_and_visualize(
                        input=dataloader.denormalize(input_td),
                        target=dataloader.denormalize(target_td),
                        x_recon=dataloader.denormalize(out.x_recon),
                        x_preds=dataloader.denormalize(out.x_preds),
                        output_dir=self.output_dir,
                        mode=f"{mode}_epoch{epoch:04d}_batch{i:03d}",
                    )

        final_metrics = average_losses(total_losses, len(dataloader))
        if self.eval_metrics:
            metric_key = (
                f"metric_{self.eval_metrics.mode}_{self.eval_metrics.variable_mode}"
            )
            final_metrics[metric_key] = (
                float(np.mean(all_metric_values)) if all_metric_values else float("nan")
            )

        return final_metrics

    def _log_metrics(self, metrics: Dict, epoch: int, mode: str):
        if not self.is_main_process() or not metrics:
            return

        log_data = {f"{mode}/{k}": v for k, v in metrics.items()}
        log_data["epoch"] = epoch
        if mode == "train":
            log_data["lr"] = self.optimizer.param_groups[0]["lr"]
        wandb.log(log_data)

        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Epoch {epoch:04d} [{mode.upper()}] {metrics_str}")

        for key, value in metrics.items():
            if key in self.history and mode in self.history[key]:
                self.history[key][mode].append(value)

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        if not self.is_main_process():
            return

        model_to_save = (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )
        state = {
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        cp_dir = self.output_dir / "checkpoints"
        cp_dir.mkdir(exist_ok=True)
        if is_best:
            torch.save(state, cp_dir / "best_model.pth")
        if self.save_latest_every > 0 and epoch % self.save_latest_every == 0:
            torch.save(state, cp_dir / "latest_model.pth")

    def run(self) -> Dict:
        logger.info(
            f"Starting training from epoch {self.start_epoch}/{self.num_epochs}"
        )

        for epoch in (
            pbar := tqdm(range(self.start_epoch, self.num_epochs), desc="Epochs")
        ):
            self.current_epoch = epoch

            train_metrics = self._run_one_epoch()

            # Step the scheduler AFTER the training epoch
            self.lr_scheduler.step()

            avg_train_metrics = self._gather_and_average_metrics(train_metrics)
            self._log_metrics(avg_train_metrics, epoch, "train")

            if epoch % self.log_epoch == 0 or epoch == self.num_epochs - 1:
                val_metrics = self.evaluate(self.val_loader, epoch, "val")
                avg_val_metrics = self._gather_and_average_metrics(val_metrics)
                self._log_metrics(avg_val_metrics, epoch, "val")

                if self.is_main_process():
                    current_val_loss = float(
                        avg_val_metrics.get("total_loss", float("inf"))
                    )
                    is_best = current_val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = current_val_loss
                        self.patience_counter = 0
                        self.save_checkpoint(epoch, current_val_loss, is_best=True)
                    else:
                        self.patience_counter += 1

                    self.save_checkpoint(epoch, current_val_loss, is_best=False)

                    if self.patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch}.")
                        break

            pbar.set_postfix(
                train_loss=avg_train_metrics.get("total_loss"),
                val_loss=(
                    self.history["total_loss"]["val"][-1]
                    if self.history["total_loss"]["val"]
                    else -1
                ),
                best_val=self.best_val_loss,
            )

        if self.is_main_process():
            self.plot_and_save_history()
        logger.info("Training finished.")
        return self.history

    def plot_and_save_history(self):
        if not self.is_main_process():
            return
        fig, ax = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

        loss_keys = [k for k in self.history if "loss" in k]
        metric_keys = [k for k in self.history if "metric" in k]

        for key in loss_keys:
            if self.history[key]["train"]:
                ax[0].plot(self.history[key]["train"], label=f"Train {key}", alpha=0.8)
            if self.history[key]["val"]:
                val_epochs = range(
                    self.start_epoch,
                    self.start_epoch + len(self.history[key]["val"]) * self.log_epoch,
                    self.log_epoch,
                )
                ax[0].plot(
                    val_epochs,
                    self.history[key]["val"],
                    label=f"Val {key}",
                    linestyle="--",
                )
        ax[0].set_ylabel("Loss")
        ax[0].set_title("Training & Validation Loss")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        if any(self.history[k]["val"] for k in metric_keys):
            for key in metric_keys:
                if self.history[key]["val"]:
                    val_epochs = range(
                        self.start_epoch,
                        self.start_epoch
                        + len(self.history[key]["val"]) * self.log_epoch,
                        self.log_epoch,
                    )
                    ax[1].plot(
                        val_epochs,
                        self.history[key]["val"],
                        label=f"Val {key}",
                        linestyle="--",
                    )
            ax[1].set_ylabel("Metric Value")
            ax[1].set_title("Validation Metrics")
            ax[1].legend()
            ax[1].grid(True, alpha=0.3)

        ax[-1].set_xlabel("Epoch")
        plt.tight_layout()
        fig.savefig(self.output_dir / "training_history.png", dpi=300)
        with open(self.output_dir / "training_history.yaml", "w") as f:
            yaml.dump(self.history, f, indent=2)
        plt.close(fig)
