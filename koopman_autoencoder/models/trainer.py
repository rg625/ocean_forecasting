# models/trainer.py

import torch
import numpy as np
from torch import Tensor
from torch.optim import Optimizer
from tensordict import TensorDict
import torch.distributed as dist
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from tqdm import tqdm
import logging
from typing import Optional, Union, Dict, List
import wandb
from einops import rearrange

# Local imports
from .autoencoder import KoopmanAutoencoder
from .loss import KoopmanLoss
from .lr_schedule import CosineWarmup
from .dataloader import DataLoaderWrapper
from .metrics import Metric
from .utils import accumulate_losses, average_losses
from .visualization import denormalize_and_visualize
import random

seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
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
        precision: Optional[str] = "bfloat16",
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
        self.scaler = None

        if precision == "float16":
            self.autocast_dtype = torch.float16
            self.scaler = GradScaler(device=self.device)
            logger.info("Using float16 mixed precision with GradScaler.")

        elif precision == "bfloat16":
            # bfloat16 is only available on Ampere and newer GPUs
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.autocast_dtype = torch.bfloat16
                # GradScaler is not required for bfloat16
                logger.info("Using bfloat16 mixed precision.")
            else:
                logger.warning(
                    "bfloat16 not supported on this device, falling back to float32."
                )

        elif precision in ["float32", None]:
            self.autocast_dtype = torch.float32
            logger.info("Using float32 full precision.")
        else:
            raise ValueError(f"Unsupported precision: '{precision}'")

    def _init_history(self):
        self.history: Dict[str, Dict[str, List[float]]] = {
            "total_loss": {"train": [], "val": []},
            "loss_recon": {"train": [], "val": []},
            "loss_pred": {"train": [], "val": []},
            "loss_latent": {"train": [], "val": []},
            "loss_phys": {"train": [], "val": []},  # Changed from loss_grad
            # You don't need to add the sub-metrics (like latent_energy) here
            # because the new _log_metrics will add them automatically.
        }

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

    def true_latent_encoding(self, target_td: TensorDict, model_module):

        original_batch_size = target_td.batch_size
        B, T = original_batch_size
        td_to_encode = target_td.select(*model_module.data_variables.keys())

        # 1. Create a new Python dictionary with rearranged (flattened) tensors.
        squashed_dict = {
            key: rearrange(tensor.unsqueeze(2), "b t ... -> (b t) ...")
            for key, tensor in td_to_encode.items()
        }
        # 2. Create a new TensorDict with the correct flattened batch size.
        squashed_td_to_encode = TensorDict(squashed_dict, batch_size=[B * T, 1])

        # Flatten the Reynolds number tensor to match the squashed batch dimension.
        cond_target_flat = None
        if "cond_target" in target_td:
            cond_target_flat = rearrange(target_td["cond_target"], "b t -> (b t)")

        # Get the "true" latent vectors by encoding the ground-truth future states.
        true_latents_flat = model_module.present_encoding(
            squashed_td_to_encode, cond_input=cond_target_flat
        )
        return rearrange(true_latents_flat, "(b t) d -> b t d", b=B)

    def _run_one_epoch(self) -> Dict[str, float]:
        """Runs a single training epoch with Crash Protection."""
        self.model.train()
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.current_epoch)

        epoch_losses: Dict[str, Tensor] = {}

        # Track batches to avoid division by zero if we skip too many
        valid_batches = 0

        for input_td, target_td in self.train_loader:
            input_td, target_td = input_td.to(self.device), target_td.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            model_module = (
                self.model.module
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                else self.model
            )

            # --- 1. SAFER NOISE INJECTION ---
            # Reduced from 0.02 to 0.005 (0.5% of standard deviation)
            # This is enough to stabilize, but safe for sensitive fluids.
            noise_level = 0.005

            # Create noise
            noise = (
                torch.randn_like(input_td.select(*model_module.data_variables.keys()))
                * noise_level
            )

            # Apply noise ONLY to data variables
            noisy_input_td = input_td.clone()
            for k in model_module.data_variables.keys():
                noisy_input_td[k] = noisy_input_td[k] + noise[k]

            # --- 2. INPUT CLAMPING (CRITICAL) ---
            # Even normalized data shouldn't exceed +/- 5 sigma.
            # Noise can sometimes push outlier pixels to +/- 20, which kills RBFs/SiLU.
            noisy_input_td = noisy_input_td.apply(lambda t: torch.clamp(t, -5.0, 5.0))

            with autocast(
                device_type=str(self.device),
                dtype=self.autocast_dtype,
                enabled=self.autocast_dtype is not None,
            ):

                # Forward Pass
                out = self.model(noisy_input_td, target_td["seq_length"])
                # out = self.model(input_td, target_td["seq_length"])

                x_true_recon = TensorDict(
                    {k: input_td[k][:, -1] for k in model_module.data_variables.keys()},
                    batch_size=input_td.batch_size[0],
                )

                loss = self.criterion(
                    model_module.koopman_operator,
                    out.x_recon,
                    out.x_preds,
                    out.z_preds,
                    x_true_recon,  # Compare against CLEAN input
                    target_td,  # Compare against CLEAN target
                    self.true_latent_encoding(
                        target_td=target_td, model_module=model_module
                    ),
                    out.reynolds,
                    out.disturbed_latents,
                    out.dz_dt,
                    out.dz_dt_disturbed,
                )
                loss_dict = loss.metrics

            # --- 3. LOSS SPIKE GUARD ---
            total_loss = loss.total_loss

            # Check for NaN/Inf
            if not torch.isfinite(total_loss):
                logger.warning(
                    f"Epoch {self.current_epoch}: Loss is {total_loss.item()}. SKIPPING BATCH."
                )
                continue

            # Check for Massive Spikes (Model Collapse Prevention)
            # If loss is > 5.0 (assuming normalized data), something is wrong.
            # Normal loss is ~0.02. A spike to 1.0+ is a collapse risk.
            if total_loss.item() > 100.0 and self.current_epoch > 5:
                logger.warning(
                    f"Epoch {self.current_epoch}: Loss spike detected ({total_loss.item():.4f}). SKIPPING BATCH to prevent collapse."
                )
                continue

            # --- 4. BACKWARD & CLIP ---
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)

                # Gradient Clipping is essential for SiLU/Koopman
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

            # detached_losses = {
            #     k: v.detach() for k, v in loss_dict.items() if isinstance(v, Tensor)
            # }
            epoch_losses = accumulate_losses(epoch_losses, loss_dict)
            valid_batches += 1

        # Prevent division by zero if all batches failed
        if valid_batches == 0:
            logger.error("All batches failed or were skipped!")
            return {}

        return average_losses(epoch_losses, valid_batches)

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
                    koopman_operator=model_module.koopman_operator,
                    x_recon=out.x_recon,
                    x_preds=out.x_preds,
                    latent_pred=out.z_preds,
                    x_true=x_true_recon,
                    x_future=target_td,
                    true_latents=self.true_latent_encoding(
                        target_td=target_td, model_module=model_module
                    ),
                    reynolds=out.reynolds,
                    disturbed_latents=out.disturbed_latents,
                    dz_dt=out.dz_dt,
                    dz_dt_disturbed=out.dz_dt_disturbed,
                ).metrics
                # detached_losses = {
                #     k: v.detach() for k, v in loss_dict.items() if isinstance(v, Tensor)
                # }
                total_losses = accumulate_losses(total_losses, loss_dict)

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
                        mode=f"{mode}",
                    )
                # TODO: Add number of batches break

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
            # Automatically initialize the key if it doesn't exist
            if key not in self.history:
                self.history[key] = {"train": [], "val": []}

            # Ensure the specific mode list exists (e.g. if a metric is val-only)
            if mode not in self.history[key]:
                self.history[key][mode] = []

            self.history[key][mode].append(value)

        wandb.log(self.model.timings)

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
            torch.save(state, cp_dir / f"epoch_{epoch}.pth")

    def run(self) -> Dict:
        logger.info(
            f"Starting training from epoch {self.start_epoch}/{self.num_epochs}"
        )

        for epoch in (
            pbar := tqdm(range(self.start_epoch, self.num_epochs), desc="Epochs")
        ):
            torch.autograd.set_detect_anomaly(True)
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

        # Categorize keys
        all_keys = list(self.history.keys())
        loss_keys = sorted([k for k in all_keys if "loss" in k])
        # External metrics (usually from self.eval_metrics)
        metric_keys = sorted([k for k in all_keys if "metric" in k])
        # Internal detailed metrics (e.g., latent_energy, phys_time, recon_u)
        detail_keys = sorted(
            [k for k in all_keys if k not in loss_keys and k not in metric_keys]
        )

        # Determine how many subplots we need
        rows = 1
        if metric_keys:
            rows += 1
        if detail_keys:
            rows += 1

        fig, ax = plt.subplots(rows, 1, figsize=(15, 6 * rows), sharex=True)
        if rows == 1:
            ax = [ax]  # Ensure ax is iterable if only 1 row

        # Helper to plot a group of keys on a specific axis
        def plot_group(keys, axis, title):
            has_data = False
            for key in keys:
                # Plot Train
                if self.history[key].get("train"):
                    axis.plot(
                        self.history[key]["train"], label=f"Train {key}", alpha=0.7
                    )
                    has_data = True

                # Plot Val
                if self.history[key].get("val"):
                    val_len = len(self.history[key]["val"])
                    # Calculate correct x-axis indices for validation steps
                    val_epochs = [
                        self.start_epoch + (i + 1) * self.log_epoch
                        for i in range(val_len)
                    ]
                    # Handle case where validation might have run one extra time or differently
                    # simple fallback: linspace
                    if len(val_epochs) != val_len:
                        val_epochs = list(range(val_len))

                    axis.plot(
                        val_epochs,
                        self.history[key]["val"],
                        label=f"Val {key}",
                        linestyle="--",
                        linewidth=2,
                    )
                    has_data = True

            if has_data:
                axis.set_ylabel("Value")
                axis.set_title(title)
                axis.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                axis.grid(True, alpha=0.3)
                if (
                    self.cfg
                    and hasattr(self.cfg, "log_scale_plots")
                    and self.cfg.log_scale_plots
                ):
                    axis.set_yscale("log")

        # --- Plotting ---
        current_ax_idx = 0

        # 1. Main Losses
        if loss_keys:
            plot_group(loss_keys, ax[current_ax_idx], "Loss Functions")
            current_ax_idx += 1

        # 2. Detailed Physics/Latent Metrics
        if detail_keys:
            plot_group(
                detail_keys, ax[current_ax_idx], "Detailed Diagnostics (Physics/Latent)"
            )
            current_ax_idx += 1

        # 3. Validation Metrics
        if metric_keys:
            plot_group(metric_keys, ax[current_ax_idx], "Evaluation Metrics")
            current_ax_idx += 1

        ax[-1].set_xlabel("Epoch")
        plt.tight_layout()

        # Save
        fig.savefig(
            self.output_dir / "training_history.png", dpi=300, bbox_inches="tight"
        )

        # Save YAML
        with open(self.output_dir / "training_history.yaml", "w") as f:
            yaml.dump(self.history, f, indent=2)

        plt.close(fig)
