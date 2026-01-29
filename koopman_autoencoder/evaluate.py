import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset

# --- Local Imports ---
from models.autoencoder import KoopmanAutoencoder
from models.dataloader import create_dataloaders
from models.utils import (
    load_checkpoint,
    load_datasets,
    load_config,
)
from models.metrics_utils import run_kae_rollout

# --- Configuration ---


@dataclass
class EvalConfig:
    """
    Configuration dataclass to manage experiment parameters.
    Matches the naming and path logic found in the reference notebook.
    """

    # Experiment Identifiers
    model_arch: str = "discrete"  # Matches notebook typo/folder naming
    model_type: str = "mlp"
    dimension: int = 128
    regime: str = "full"
    ckpt_index: int = 199

    # Paths
    base_output_dir: Path = Path("./model_outputs_full")
    config_dir: str = "experiment"
    result_dir: Path = Path("./results/sampling/lowRey")

    # Data Parameters
    rollout_steps: int = 240
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def experiment_name(self) -> str:
        return f"{self.model_type}_{self.model_arch}_{self.dimension}"

    @property
    def config_path(self) -> str:
        # e.g. experiment/stable/continous_linear_64
        return f"{self.config_dir}/{self.regime}/{self.experiment_name}"

    @property
    def checkpoint_path(self) -> Path:
        # Matches notebook: ./model_outputs_{REGIME}/{MODEL}_{TYPE}_{DIMENSION}/checkpoints/epoch_{CKPT_INDEX}.pth
        # Use base_output_dir corresponding to the regime (stable or full)
        return (
            self.base_output_dir
            / self.experiment_name
            / "checkpoints"
            / f"epoch_{self.ckpt_index}.pth"
        )


# --- Logging Setup ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# --- Helper Functions ---


def tensordict_to_eval_array_with_re(
    input_seq_td: TensorDict,
    predicted_seq_td: TensorDict,
    variables: List[str] = ["v_x", "v_y", "rho", "p"],
) -> np.ndarray:
    """
    Ported directly from the notebook.
    Converts TensorDict sequences to NumPy array with Re channel.
    """

    def ensure_batch(td_var):
        arr = td_var.cpu().numpy()
        if arr.ndim == 3:  # [T, H, W] -> [1, T, H, W]
            arr = arr[None, ...]
        # Swap W/H to match notebook's output expectations
        return arr.transpose(0, 1, 3, 2)

    input_arrs = [ensure_batch(input_seq_td[var]) for var in variables]
    predicted_arrs = [ensure_batch(predicted_seq_td[var]) for var in variables]

    # Stack channels: (B, T, C, H, W)
    input_arr = np.stack(input_arrs, axis=2)
    predicted_arr = np.stack(predicted_arrs, axis=2)

    B = input_arr.shape[0]
    H, W = input_arr.shape[-2:]

    # Concatenate last 2 input frames with predicted sequence (excluding last 2 of pred)
    # This matches the notebook's temporal stitching logic
    last_input_frame = input_arr[:, -2:, :, :, :]
    predicted_arr_sliced = predicted_arr[:, :-2, :, :, :]

    full_seq = np.concatenate([last_input_frame, predicted_arr_sliced], axis=1)

    # Add Re channel
    re_vals = input_seq_td["cond_input"].cpu().numpy()
    if re_vals.ndim == 0:
        re_vals = np.full(B, re_vals, dtype=np.float32)
    elif re_vals.ndim == 1 and re_vals.shape[0] != B:
        re_vals = np.tile(re_vals, B)[:B]

    re_channel = np.zeros((B, full_seq.shape[1], 1, H, W), dtype=np.float32)
    for b in range(B):
        re_channel[b, :, 0, :, :] = re_vals[b]

    # Final concat [B, T, C+1, H, W]
    full_seq_with_re = np.concatenate([full_seq, re_channel], axis=2)
    return cast(np.ndarray, full_seq_with_re)


class KoopmanEvaluator:
    def __init__(self, config: EvalConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        self.model: Optional[KoopmanAutoencoder] = None
        self.train_loader: Optional[DataLoader] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

        self._initialize()

    def _initialize(self):
        logger.info(f"Initializing Evaluator: {self.cfg.experiment_name}")
        logger.info(f"Using Regime: {self.cfg.regime} from {self.cfg.base_output_dir}")

        # 1. Load Config
        self.exp_cfg = load_config(self.cfg.config_path)

        # 2. Load Datasets
        self.train_dataset, self.val_dataset, test_dataset = load_datasets(self.exp_cfg)
        assert self.val_dataset is not None, "Expected validation dataset to be loaded"
        assert self.train_dataset is not None, "Expected training dataset to be loaded"
        self.train_loader, _, _ = create_dataloaders(
            self.train_dataset,
            self.val_dataset,
            test_dataset,
            self.exp_cfg.training,
            num_workers=self.exp_cfg.data.num_workers,
        )
        assert self.train_loader is not None, "Expected train loader to be created"

        # 3. Build Model
        self.model = KoopmanAutoencoder(
            data_variables=self.exp_cfg.data.variables,
            input_frames=self.exp_cfg.data.input_sequence_length,
            height=self.exp_cfg.model.height,
            width=self.exp_cfg.model.width,
            latent_dim=self.exp_cfg.model.latent_dim,
            cond_embedding_dim=self.exp_cfg.model.cond_embedding_dim,
            cond_type=self.exp_cfg.model.cond_type,
            operator_mode=self.exp_cfg.model.operator_mode,
            hidden_dims=self.exp_cfg.model.hidden_dims,
            transformer_config=self.exp_cfg.model.transformer,
            use_checkpoint=self.exp_cfg.training.use_checkpoint,
            predict_cond=self.exp_cfg.model.predict_cond,
            cond_grad_enabled=self.exp_cfg.model.cond_grad_enabled,
            disturb_std=None,
            is_continuous=self.exp_cfg.model.is_continuous,
            rank=self.exp_cfg.model.rank,
            cond_expansion_type=self.exp_cfg.data.selection_param,
            **self.exp_cfg.model.conv_kwargs,
        ).to(self.device)

        # 4. Load Weights
        ckpt_path = self.cfg.checkpoint_path
        if ckpt_path.exists():
            logger.info(f"Loading Checkpoint: {ckpt_path}")
            optimizer = optim.Adam(
                self.model.parameters(), lr=self.exp_cfg.lr_scheduler.lr
            )
            self.model, _, _, _ = load_checkpoint(
                str(ckpt_path),
                model=self.model,
                optimizer=optimizer,
                strict=True,
            )
        else:
            logger.error(f"CRITICAL: Checkpoint NOT found at {ckpt_path}")
            sys.exit(1)

        self.model.eval()
        logger.info(
            f"Model Initialized. Continuous: {self.model.koopman_operator.is_continuous}"
        )

    def run_validation_rollouts(self) -> np.ndarray:
        assert self.val_dataset is not None
        assert self.train_dataset is not None
        assert self.train_loader is not None
        assert self.model is not None

        val_dataset = self.val_dataset

        logger.info(f"Starting Rollouts for {len(val_dataset)} samples...")

        eval_arrays = []

        with torch.inference_mode():
            for idx in tqdm(range(len(self.val_dataset)), desc="Processing Val Set"):
                input_seq, _, metadata = self.val_dataset[idx, self.cfg.rollout_steps]

                # Match notebook metadata attachment
                input_seq["obstacle_mask"] = (
                    metadata["obstacle_mask"][0]
                    .repeat(*input_seq.batch_size, 1, 1)
                    .to(self.device)
                )
                input_seq["cond_input"] = (
                    metadata["cond_input"][0]
                    .repeat(*input_seq.batch_size)
                    .to(self.device)
                )

                # Use masked input for rollout as per notebook
                masked_input = self.train_dataset.apply_mask(input_seq)

                total_predicted_seq = run_kae_rollout(
                    self.model,
                    masked_input,
                    self.cfg.rollout_steps,
                    return_xpreds=False,
                )

                predicted_seq = total_predicted_seq.x_preds.squeeze(0).cpu()

                # Denormalize
                input_denorm = self.train_loader.denormalize(input_seq).cpu()
                pred_denorm = self.train_loader.denormalize(predicted_seq)

                # Format and append
                arr = tensordict_to_eval_array_with_re(
                    input_seq_td=input_denorm,
                    predicted_seq_td=pred_denorm,
                    variables=(
                        ["v_x", "v_y", "rho", "p"]
                        if "rho" in self.exp_cfg.data.variables
                        else ["v_x", "v_y", "p"]
                    ),
                )
                eval_arrays.append(arr)

        return np.concatenate(eval_arrays, axis=0)

    def save_results(self, data: np.ndarray):
        self.cfg.result_dir.mkdir(parents=True, exist_ok=True)
        # filename = f"{self.cfg.model_type}_{self.cfg.model_arch}_{self.cfg.dimension}_{self.cfg.regime}_on_stable_only.npz"
        filename = f"{self.cfg.model_type}_{self.cfg.model_arch}_{self.cfg.dimension}_{self.cfg.regime}.npz"
        save_path = self.cfg.result_dir / filename

        # Wrap in extra dims to match notebook: [None, None, ...]
        final_data = data[None, None, ...]

        logger.info(f"Saving shape {final_data.shape} to {save_path}")
        np.savez_compressed(save_path, final_data)
        logger.info("✅ Save Complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Koopman Autoencoder Evaluation (Notebook Sync)"
    )

    # Support for the bash script pattern: stable/discrete_mlp_1024
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Full config path (e.g. stable/discrete_linear_128).",
    )

    parser.add_argument("--dim", type=int, default=64, help="Model dimension")
    parser.add_argument(
        "--type", type=str, default="linear", help="Model type (linear/mlp)"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="continous",
        help="Architecture (continous/discrete)",
    )
    parser.add_argument(
        "--regime", type=str, default="tra", help="Regime (stable/full)"
    )
    parser.add_argument("--ckpt", type=int, default=199, help="Checkpoint index")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Base path to logs. If None, derived from regime.",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./results/sampling/lowRey", help="Output path"
    )
    parser.add_argument("--rollout_steps", type=int, default=60, help="Rollout steps")

    args = parser.parse_args()

    dim_val = args.dim
    type_val = args.type
    arch_val = args.arch
    regime_val = args.regime
    rollout_steps = args.rollout_steps

    # Logic to parse --config_name if provided from bash loop
    if args.config_name:
        try:
            parts = args.config_name.strip("/").split("/")
            if len(parts) > 1:
                regime_val = parts[0]
                exp_name = parts[-1]
            else:
                exp_name = parts[0]

            exp_parts = exp_name.split("_")
            if len(exp_parts) >= 3:
                arch_val = exp_parts[0]
                type_val = exp_parts[1]
                dim_val = int(exp_parts[2])
        except Exception as e:
            logger.error(f"Failed to parse config_name '{args.config_name}': {e}")
            sys.exit(1)

    # Automatically set base_dir based on regime if not provided
    # Matches notebook logic: f"./model_outputs_{REGIME}"
    base_dir_val = (
        Path(args.base_dir) if args.base_dir else Path(f"./model_outputs_{regime_val}")
    )

    config = EvalConfig(
        model_arch=arch_val,
        model_type=type_val,
        dimension=dim_val,
        regime=regime_val,
        ckpt_index=args.ckpt,
        base_output_dir=base_dir_val,
        result_dir=Path(args.out_dir),
        rollout_steps=rollout_steps,
        device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",
    )

    try:
        evaluator = KoopmanEvaluator(config)
        results = evaluator.run_validation_rollouts()
        evaluator.save_results(results)
    except Exception:
        logger.exception("Evaluation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
