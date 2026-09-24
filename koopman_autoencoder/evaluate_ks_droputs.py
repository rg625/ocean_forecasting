# mypy: disable-error-code="arg-type, index, union-attr"
import argparse
import logging
import sys
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast, Tuple

import matplotlib

matplotlib.use("Agg")  # Safe for headless server environments
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset

# --- Local Imports ---
from models.autoencoder import KoopmanAutoencoder
from models.dataloader import create_dataloaders
from models.utils import load_checkpoint, load_datasets, load_config
from models.metrics_utils import run_kae_rollout

# --- Configuration ---


@dataclass
class EvalConfig:
    model_arch: str = "discrete"
    model_type: str = "mlp"
    dimension: int = 128
    regime: str = "ks"
    ckpt_index: int = 199

    dropout: str = "0.1"  # The dropout probability this model was trained with
    eval_rollout_steps: int = 100

    base_output_dir: Path = Path("./model_outputs")
    config_dir: str = "experiment"
    result_dir: Path = Path("./results/sampling")

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def experiment_name(self) -> str:
        return f"{self.model_type}_{self.model_arch}_{self.dimension}"

    @property
    def config_path(self) -> str:
        # e.g., experiment/ks/continous_linear_128
        return f"{self.config_dir}/{self.regime}/{self.experiment_name}"

    @property
    def checkpoint_path(self) -> Path:
        # Dynamically find the latest run directory inside the dropout folder
        print(f"base_output_dir: {self.base_output_dir}")
        exp_dir = (
            self.base_output_dir / self.experiment_name / f"rollout_{self.dropout}"
        )
        print(f"exp_dir: {exp_dir}")
        return exp_dir / "checkpoints" / f"epoch_{self.ckpt_index}.pth"


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


def tensordict_to_eval_array_with_cond(
    input_seq_td: TensorDict,
    predicted_seq_td: TensorDict,
    variables: List[str] = ["v_x", "v_y", "rho", "p"],
) -> np.ndarray:
    def ensure_batch(td_var):
        arr = td_var.cpu().numpy()
        if arr.ndim == 3:
            arr = arr[None, ...]
        return arr.transpose(0, 1, 3, 2)

    input_arrs = [ensure_batch(input_seq_td[var]) for var in variables]
    predicted_arrs = [ensure_batch(predicted_seq_td[var]) for var in variables]

    input_arr = np.stack(input_arrs, axis=2)
    predicted_arr = np.stack(predicted_arrs, axis=2)

    B = input_arr.shape[0]
    H, W = input_arr.shape[-2:]

    last_input_frame = input_arr[:, -1:, :, :, :]
    predicted_arr_sliced = predicted_arr[:, :-1, :, :, :]

    full_seq = np.concatenate([last_input_frame, predicted_arr_sliced], axis=1)

    if "cond_input" in input_seq_td.keys():
        re_vals = input_seq_td["cond_input"].cpu().numpy()
        if re_vals.ndim == 0:
            re_vals = np.full(B, re_vals, dtype=np.float32)
        elif re_vals.ndim == 1 and re_vals.shape[0] != B:
            re_vals = np.tile(re_vals, B)[:B]

        re_channel = np.zeros((B, full_seq.shape[1], 1, H, W), dtype=np.float32)
        for b in range(B):
            re_channel[b, :, 0, :, :] = re_vals[b]

        full_seq_with_re = np.concatenate([full_seq, re_channel], axis=2)
        return cast(np.ndarray, full_seq_with_re)

    return cast(np.ndarray, full_seq)


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
        logger.info(
            f"Initializing Evaluator for Dropout {self.cfg.dropout}: {self.cfg.experiment_name}"
        )

        self.exp_cfg = load_config(self.cfg.config_path)

        # ====================================================================
        # MANUAL INTERVENTION: config.data overrides
        # ====================================================================
        self.exp_cfg.data.max_sequence_length = self.cfg.eval_rollout_steps - 4
        self.exp_cfg.data.subsample = 1
        # ====================================================================

        self.train_dataset, self.val_dataset, test_dataset = load_datasets(self.exp_cfg)
        self.train_loader, _, _ = create_dataloaders(
            self.train_dataset,
            self.val_dataset,
            test_dataset,
            self.exp_cfg.training,
            num_workers=self.exp_cfg.data.num_workers,
        )

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
            spectral=self.exp_cfg.model.spectral,
            cond_expansion_type=self.exp_cfg.data.selection_param,
            use_attention=self.exp_cfg.model.use_attention,
            **self.exp_cfg.model.conv_kwargs,
        ).to(self.device)

        ckpt_path = self.cfg.checkpoint_path
        print(f"ckpt_path: {ckpt_path}")
        if ckpt_path.exists():
            logger.info(f"Loading Checkpoint: {ckpt_path}")
            optimizer = optim.Adam(
                self.model.parameters(), lr=self.exp_cfg.lr_scheduler.lr
            )
            self.model, _, _, _ = load_checkpoint(
                str(ckpt_path), model=self.model, optimizer=optimizer, strict=True
            )
        else:
            logger.error(f"CRITICAL: Checkpoint NOT found at {ckpt_path}")
            sys.exit(1)

        self.model.eval()

    def run_validation_rollouts(self) -> Tuple[np.ndarray, List[float]]:
        max_samples = min(500, len(self.val_dataset))
        logger.info(f"Starting Rollouts for {max_samples} samples...")

        eval_arrays = []
        sample_mses = []

        with torch.inference_mode():
            for idx in tqdm(range(max_samples), desc="Processing Val Set"):
                input_seq, target_seq, metadata = self.val_dataset[
                    idx, self.cfg.eval_rollout_steps
                ]

                if "obstacle_mask" in metadata:
                    input_seq["obstacle_mask"] = (
                        metadata["obstacle_mask"][0]
                        .repeat(*input_seq.batch_size, 1, 1)
                        .to(self.device)
                    )
                if "cond_input" in metadata:
                    input_seq["cond_input"] = (
                        metadata["cond_input"][0]
                        .repeat(*input_seq.batch_size)
                        .to(self.device)
                    )

                masked_input = self.train_dataset.apply_mask(input_seq)

                total_predicted_seq = run_kae_rollout(
                    self.model,
                    masked_input,
                    self.cfg.eval_rollout_steps,
                    return_xpreds=False,
                )

                predicted_seq = total_predicted_seq.x_preds.squeeze(0).cpu()

                input_denorm = self.train_loader.denormalize(input_seq).cpu()
                pred_denorm = self.train_loader.denormalize(predicted_seq)

                target_denorm = self.train_loader.denormalize(target_seq).cpu()

                sample_mse_avg = 0.0
                for var in self.exp_cfg.data.variables:
                    pred_v = pred_denorm[var]
                    gt_v = target_denorm[var]

                    if gt_v.ndim == pred_v.ndim + 1 and gt_v.shape[0] == 1:
                        gt_v = gt_v.squeeze(0)
                    elif pred_v.ndim == gt_v.ndim + 1 and pred_v.shape[0] == 1:
                        pred_v = pred_v.squeeze(0)

                    t_pred = pred_v.shape[0]
                    t_gt = gt_v.shape[0]
                    min_t = min(t_pred, t_gt)

                    pred_aligned = pred_v[:min_t]
                    gt_aligned = gt_v[:min_t]

                    mse_val = torch.mean((pred_aligned - gt_aligned) ** 2).item()
                    sample_mse_avg += mse_val

                sample_mse_avg /= len(self.exp_cfg.data.variables)
                sample_mses.append(sample_mse_avg)

                arr = tensordict_to_eval_array_with_cond(
                    input_seq_td=input_denorm,
                    predicted_seq_td=pred_denorm,
                    variables=self.exp_cfg.data.variables,
                )
                eval_arrays.append(arr)

        return np.concatenate(eval_arrays, axis=0), sample_mses

    def save_results(self, data: np.ndarray):
        self.cfg.result_dir.mkdir(parents=True, exist_ok=True)
        filename = f"rollout_{self.cfg.dropout}_{self.cfg.model_type}_{self.cfg.model_arch}_{self.cfg.dimension}_{self.cfg.regime}.npz"
        save_path = self.cfg.result_dir / filename

        final_data = data[None, None, ...]
        logger.info(f"Saving shape {final_data.shape} to {save_path}")
        np.savez_compressed(save_path, final_data)
        logger.info("✅ Save Complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Koopman Autoencoder Evaluation across Dropouts"
    )

    parser.add_argument("--dim", type=int, default=128, help="Model dimension")
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
        "--regime", type=str, default="ks", help="Regime context (e.g. ks, tra)"
    )
    parser.add_argument("--ckpt", type=int, default=199, help="Checkpoint index")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")

    parser.add_argument("--base_dir", type=str, default=None, help="Base path to logs.")
    parser.add_argument("--out_dir", type=str, default="./results/sampling/ks")

    # NEW ARGUMENT FOR DROPOUTS
    parser.add_argument(
        "--dropouts",
        nargs="+",
        type=str,
        required=True,
        help="List of dropout probabilities (e.g. 0.1 0.2 0.3)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Number of steps to evaluate over (default: 100)",
    )

    args = parser.parse_args()

    base_dir_val = (
        Path(args.base_dir) if args.base_dir else Path(f"./model_outputs_{args.regime}")
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_report = {}

    for dropout_val in args.dropouts:
        logger.info("=" * 60)
        logger.info(
            f"Evaluating Model Trained with Dropout: {dropout_val} (Eval steps: {args.eval_steps})"
        )
        logger.info("=" * 60)

        config = EvalConfig(
            model_arch=args.arch,
            model_type=args.type,
            dimension=args.dim,
            regime=args.regime,
            ckpt_index=args.ckpt,
            dropout=dropout_val,
            eval_rollout_steps=args.eval_steps,
            base_output_dir=base_dir_val,
            result_dir=out_dir,
            device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",
        )

        try:
            evaluator = KoopmanEvaluator(config)
            results_data, sample_mses = evaluator.run_validation_rollouts()
            evaluator.save_results(results_data)

            # --- Calculate Publication Statistics ---
            arr_mses = np.array(sample_mses)
            mean_mse = np.mean(arr_mses)
            std_mse = np.std(arr_mses, ddof=1)
            n_samples = len(arr_mses)

            # 95% Confidence Interval (1.96 * standard error)
            margin_of_error = 1.96 * (std_mse / np.sqrt(n_samples))

            logger.info(
                f"Dropout {dropout_val} | Mean MSE: {mean_mse:.6f} | 95% CI: +/- {margin_of_error:.6f}"
            )

            # Store with dropout converted to float for sorting/plotting
            results_report[float(dropout_val)] = {
                "mean": mean_mse,
                "std": std_mse,
                "ci_margin": margin_of_error,
                "ci_lower": mean_mse - margin_of_error,
                "ci_upper": mean_mse + margin_of_error,
                "label": dropout_val,
            }

        except Exception:
            logger.exception(f"Evaluation failed for dropout {dropout_val}")
            continue

    # --- Generate Publication-Ready Report and Plot ---
    if results_report:
        logger.info("=" * 60)
        logger.info("Compiling Report and Generating Plot...")

        # Sort values by dropout probability
        sorted_dropouts_floats = sorted(results_report.keys())
        labels = [results_report[d]["label"] for d in sorted_dropouts_floats]
        means = [results_report[d]["mean"] for d in sorted_dropouts_floats]
        ci_lower = [results_report[d]["ci_lower"] for d in sorted_dropouts_floats]
        ci_upper = [results_report[d]["ci_upper"] for d in sorted_dropouts_floats]

        # 1. Save Highly Detailed CSV Report
        csv_path = out_dir / f"publication_mse_report_dropout_{args.regime}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Dropout", "Mean MSE", "Std Dev", "95% CI Lower", "95% CI Upper"]
            )
            for d in sorted_dropouts_floats:
                writer.writerow(
                    [
                        results_report[d]["label"],
                        results_report[d]["mean"],
                        results_report[d]["std"],
                        results_report[d]["ci_lower"],
                        results_report[d]["ci_upper"],
                    ]
                )
        logger.info(f"Saved CSV Report to: {csv_path}")

        # 2. Generate Shaded Area Line Plot
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            sorted_dropouts_floats,
            means,
            marker="o",
            linestyle="-",
            color="#1f77b4",
            linewidth=2,
            label="Mean MSE",
        )
        ax.fill_between(
            sorted_dropouts_floats,
            ci_lower,
            ci_upper,
            color="#1f77b4",
            alpha=0.2,
            label="95% Confidence Interval",
        )

        ax.set_title(
            f"Evaluation MSE vs. Dropout Probability\n(Evaluated over {args.eval_steps} steps)",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Dropout Probability", fontsize=12, fontweight="bold")
        ax.set_ylabel("Denormalized MSE", fontsize=12, fontweight="bold")

        # Use exact string labels provided by user on the x-axis
        ax.set_xticks(sorted_dropouts_floats)
        ax.set_xticklabels(labels)
        ax.tick_params(axis="both", which="major", labelsize=10)

        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="upper right", frameon=True, fontsize=10)

        plot_path_pdf = out_dir / f"publication_mse_plot_dropout_{args.regime}.pdf"
        plot_path_png = out_dir / f"publication_mse_plot_dropout_{args.regime}.png"

        plt.savefig(plot_path_pdf, format="pdf", dpi=300, bbox_inches="tight")
        plt.savefig(plot_path_png, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved PDF Plot to: {plot_path_pdf}")
        logger.info("Evaluation Pipeline Complete!")


if __name__ == "__main__":
    main()
