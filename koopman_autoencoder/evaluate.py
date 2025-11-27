import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, cast

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from tensordict import TensorDict

# --- Local Imports ---
# Assuming the script is run from the root where 'src' or 'models' are accessible
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
    Replaces global variables and hardcoded paths.
    """

    # Experiment Identifiers
    model_arch: str = "discrete"  # "discrete" or "continuous"
    model_type: str = "mlp"  # "linear", "mlp", "eigen"
    dimension: int = 1024
    ckpt_index: int = 999

    # Paths
    base_output_dir: Path = Path("./model_outputs_stable")
    config_dir: str = "experiment"
    result_dir: Path = Path(
        "/home/rg625/mnt/ocean_forecasting/autoreg_pde_diffusion/src/results/sampling/lowRey/"
    )

    # Data Parameters
    initial_sample_index: int = 0
    rollout_steps: int = 60
    subsample: int = 1
    max_sequence_length: int = 58

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def experiment_name(self) -> str:
        return f"{self.model_arch}_{self.model_type}_{self.dimension}"

    @property
    def config_path(self) -> str:
        return f"{self.config_dir}/{self.experiment_name}"

    @property
    def checkpoint_path(self) -> Path:
        # Construct path dynamically
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


def process_batch_metadata(
    input_seq: TensorDict, metadata: Dict, device: torch.device
) -> TensorDict:
    """
    Attaches necessary metadata (masks, conditions) to the input sequence batch.
    Handles dimension expansion to match batch size [B, ...].
    """

    # 1. Handle Obstacle Mask
    if "obstacle_mask" in metadata:
        mask = metadata["obstacle_mask"][0]  # Take first from loader batch
        input_seq["obstacle_mask"] = mask.repeat(*input_seq.batch_size, 1, 1).to(device)

    # 2. Handle Conditional Input
    if "cond_input" in metadata:
        cond = metadata["cond_input"][0]
        input_seq["cond_input"] = cond.repeat(*input_seq.batch_size).to(device)

    return input_seq.to(device)


def format_predictions_with_re(
    input_seq_td: TensorDict,
    predicted_seq_td: TensorDict,
    variables: List[str] = ["v_x", "v_y", "p"],
) -> np.ndarray:
    """
    Converts TensorDict outputs to a unified NumPy array with an injected Reynolds number channel.

    Logic:
    1. Extracts variables (v_x, v_y, p).
    2. Transposes to [B, T, H, W] (assuming input is [B, T, W, H] or vice versa based on original code).
    3. Stacks variables into channels.
    4. Concatenates the last 2 frames of input with the prediction (minus last 2 frames).
    5. Injects a constant Reynolds number channel.

    Returns:
        np.ndarray: Shape [B, Total_Steps, C+1, H, W]
    """

    def ensure_batch_numpy(tensor_data):
        arr = tensor_data.detach().cpu().numpy()
        # Original code did transpose(-2, -1). Assuming layout [B, T, W, H] -> [B, T, H, W]
        # Adjust this if your source data is already H, W.
        if arr.ndim == 4:  # [B, T, D1, D2]
            arr = arr.transpose(0, 1, 3, 2)
        elif arr.ndim == 3:  # [T, D1, D2] -> Add Batch
            arr = arr[None, ...].transpose(0, 1, 3, 2)
        return arr

    # 1. Convert specific variables to numpy arrays
    input_vars = [ensure_batch_numpy(input_seq_td[var]) for var in variables]
    pred_vars = [ensure_batch_numpy(predicted_seq_td[var]) for var in variables]

    # 2. Stack channels -> [B, T, C, H, W]
    input_stack = np.stack(input_vars, axis=2)
    pred_stack = np.stack(pred_vars, axis=2)

    # 3. Temporal Stitching
    # Take last 2 frames of input context
    context_frames = input_stack[:, -2:, ...]
    # Take predictions (cutting off last 2 steps to match context length logic if intended)
    future_frames = pred_stack[:, :-2, ...]

    full_sequence = np.concatenate(
        [context_frames, future_frames], axis=1
    )  # [B, T_total, C, H, W]

    # 4. Inject Reynolds Number Channel
    B, T, C, H, W = full_sequence.shape

    re_vals = input_seq_td["cond_input"].cpu().numpy()

    # Handle scalar or batch Re values
    if re_vals.ndim == 0:
        re_vals = np.full((B,), re_vals)
    elif re_vals.ndim > 1:
        re_vals = re_vals.flatten()[:B]

    # Create channel
    re_channel = np.zeros((B, T, 1, H, W), dtype=np.float32)
    for b in range(B):
        re_channel[b, :, 0, :, :] = re_vals[b]

    # 5. Final Concatenation
    final_output = np.concatenate([full_sequence, re_channel], axis=2)
    return cast(np.ndarray, final_output)


# --- Main Evaluator Class ---


class KoopmanEvaluator:
    def __init__(self, config: EvalConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        self.model = None
        self.train_loader = None
        self.val_dataset = None
        self.train_dataset = None  # Needed for normalizer

        self._initialize()

    def _initialize(self):
        """Loads configuration, data, and model weights."""
        logger.info(
            f"Initializing Evaluator for Experiment: {self.cfg.experiment_name}"
        )

        # 1. Load Experiment Config (Hydra/Omegaconf)
        try:
            self.exp_cfg = load_config(self.cfg.config_path)
        except Exception as e:
            logger.error(f"Failed to load config at {self.cfg.config_path}")
            raise e

        # Apply overrides from EvalConfig to Experiment Config
        self.exp_cfg.data.subsample = self.cfg.subsample
        self.exp_cfg.data.max_sequence_length = self.cfg.max_sequence_length

        # 2. Load Datasets
        logger.info("Loading Datasets...")
        self.train_dataset, self.val_dataset, test_dataset = load_datasets(self.exp_cfg)
        self.train_loader, _, _ = create_dataloaders(
            self.train_dataset, self.val_dataset, test_dataset, self.exp_cfg.training
        )

        # 3. Build Model
        logger.info(f"Building Model ({self.cfg.model_arch})...")
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
            is_continuous=self.exp_cfg.model.is_continuous,
            cond_expansion_type=self.exp_cfg.data.selection_param,
            **self.exp_cfg.model.conv_kwargs,
        ).to(self.device)

        # 4. Load Weights
        if self.cfg.checkpoint_path.exists():
            logger.info(f"Loading Checkpoint: {self.cfg.checkpoint_path}")
            # Optimizer is needed for load_checkpoint signature, though not used for eval
            dummy_optimizer = optim.Adam(
                self.model.parameters(), lr=self.exp_cfg.lr_scheduler.lr
            )
            self.model, _, _, _ = load_checkpoint(
                self.cfg.checkpoint_path,
                model=self.model,
                optimizer=dummy_optimizer,
                strict=True,
            )
        else:
            logger.warning(
                f"Checkpoint not found at {self.cfg.checkpoint_path}. Using random weights!"
            )

        self.model.eval()

        # Log model specifics
        logger.info(f"Model Continuous: {self.model.koopman_operator.is_continuous}")
        if hasattr(self.model.koopman_operator, "dt_train"):
            logger.info(f"dt_train: {self.model.koopman_operator.dt_train}")

    def run_validation_rollouts(self) -> np.ndarray:
        """
        Iterates through the validation dataset using the collect-then-process
        pattern to match the original script's logic exactly.
        """

        assert self.val_dataset is not None, "Validation dataset is not initialized."
        assert len(self.val_dataset) != 0, "Validation dataset is empty."  # type: ignore[unreachable]

        logger.info(f"Starting Rollouts. Total samples: {len(self.val_dataset)}")

        # Step 1: Collect sequences
        all_input_seqs = []
        all_ground_truths = []
        all_predicted_seqs = []

        # Disable gradient calculation for efficiency
        with torch.inference_mode():
            for idx in tqdm(range(len(self.val_dataset)), desc="Running val rollouts"):

                # Fetch Data
                input_seq, ground_truth, metadata = self.val_dataset[
                    idx, self.cfg.rollout_steps
                ]

                # Attach obstacle mask & Re_input
                # We use the helper logic here but ensure it matches the snippet's direct access style where possible
                # Note: process_batch_metadata uses expand() which is safe.
                input_seq = process_batch_metadata(input_seq, metadata, self.device)

                # Apply Mask for model
                masked_input = self.train_dataset.apply_mask(input_seq)

                # Run KAE rollout
                total_predicted_seq = run_kae_rollout(
                    self.model,
                    masked_input,
                    self.cfg.rollout_steps,
                    return_xpreds=False,
                )

                # Get predicted sequence and move to CPU immediately (as per snippet)
                predicted_seq = total_predicted_seq.x_preds.squeeze(0).cpu()

                # Store sequences (Denormalize inputs here as well)
                # Ensure input_seq is on cpu before/after denorm to avoid mixed device issues in list
                input_denorm = self.train_loader.denormalize(input_seq).cpu()
                pred_denorm = self.train_loader.denormalize(
                    predicted_seq
                )  # predicted_seq is already cpu

                all_input_seqs.append(input_denorm)
                all_ground_truths.append(ground_truth)
                all_predicted_seqs.append(pred_denorm)

        # Step 2: Convert to evaluation-ready NumPy arrays
        eval_arrays = []
        logger.info("Formatting predictions...")
        for input_seq_td, pred_td in zip(all_input_seqs, all_predicted_seqs):
            arr = format_predictions_with_re(input_seq_td, pred_td)
            eval_arrays.append(arr)

        # Combine into a single array
        full_results = np.concatenate(eval_arrays, axis=0)
        logger.info(f"All validation sequences shape: {full_results.shape}")

        return full_results

    def save_results(self, data: np.ndarray, filename: str = "KAE.npz"):
        """Saves the result array to disk."""
        self.cfg.result_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.cfg.result_dir / filename

        # Expand dims to [1, 1, N, T, C, H, W] as requested in the snippet
        # snippet: np.savez_compressed(save_path, eval_arrays[None, None, ...])
        final_data = data[None, None, ...]

        logger.info(f"Saving results shape {final_data.shape} to {save_path}")
        np.savez_compressed(save_path, final_data)
        logger.info(f"✅ Saved all validation rollouts to {save_path}")


# --- Entry Point ---


def main():
    parser = argparse.ArgumentParser(description="Koopman Autoencoder Evaluation")

    # 1. Config Name Argument (Overrides individual args)
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Full config name (e.g. discrete_mlp_1024). Overrides dim/type/arch args.",
    )

    # 2. Individual Arguments (Fallback)
    parser.add_argument("--dim", type=int, default=1024, help="Model dimension")
    parser.add_argument(
        "--type", type=str, default="mlp", help="Model type (linear/mlp)"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="discrete",
        help="Architecture (discrete/continuous)",
    )

    # 3. Other args
    parser.add_argument("--ckpt", type=int, default=999, help="Checkpoint index")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    # Logic to handle config_name vs individual args
    if args.config_name:
        # Strip potential folder prefixes (e.g., experiment/discrete_mlp_1024 -> discrete_mlp_1024)
        clean_name = Path(args.config_name).name
        try:
            # Expected format: architecture_type_dimension
            parts = clean_name.split("_")
            # Assuming standard naming: discrete_linear_1024 or continuous_mlp_128
            arch_val = parts[0]
            type_val = parts[1]
            dim_val = int(parts[2])
        except (IndexError, ValueError):
            logger.error(
                f"Could not parse config name '{args.config_name}'. Expected format: arch_type_dim (e.g. discrete_mlp_1024)"
            )
            sys.exit(1)
    else:
        arch_val = args.arch
        type_val = args.type
        dim_val = args.dim

    # Create kwargs for Config
    config_kwargs = {
        "dimension": dim_val,
        "model_type": type_val,
        "model_arch": arch_val,
        "ckpt_index": args.ckpt,
        "device": f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",
    }

    if args.out_dir is not None:
        config_kwargs["result_dir"] = Path(args.out_dir)

    # Create Config
    config = EvalConfig(**config_kwargs)

    try:
        evaluator = KoopmanEvaluator(config)
        results = evaluator.run_validation_rollouts()

        # Save using dynamic name based on model architecture
        output_filename = f"{config.experiment_name}.npz"
        evaluator.save_results(results, filename=output_filename)

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user.")
    except Exception:
        logger.exception("An error occurred during evaluation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
