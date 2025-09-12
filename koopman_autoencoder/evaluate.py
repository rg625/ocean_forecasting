# evaluate_models.py (Hardened / Fixed)
import os
import logging
import argparse
import yaml
import warnings
import numpy as np
from typing import Dict, Tuple, Any, cast, Optional, Callable, Literal

import torch
import torch.nn as nn
import torch.optim as optim

# Config helpers
from omegaconf import DictConfig

# Matplotlib setup
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

# --- Import Project Modules (may raise ImportError if project structure not on PYTHONPATH) ---
try:
    from models.config_classes import Config
    from models.autoencoder import KoopmanAutoencoder
    from models.dataloader import create_dataloaders
    from models.utils import load_checkpoint, load_datasets, load_config
    from models.metrics_utils import (
        run_evaluation,
        kae_rollout_wrapper,
        run_diffusion_rollout,
    )
    from models.register_models import (
        MODEL_REGISTRY,
    )
except Exception:
    # Provide helpful log if imports fail; allow the script to continue so user sees the message.
    # Many errors here are environment/setup related (PYTHONPATH, missing package).
    raise

# Diffusion Model Imports (assuming they are in the python path)
try:
    from turbpred.params import DataParams, ModelParamsDecoder
    from turbpred.model_diffusion import DiffusionModel
except Exception:
    # We'll only raise at runtime if a diffusion model is actually required.
    pass

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use seaborn style but be resilient if seaborn not available
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("default")
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.transforms.functional"
)

# =====================================================================================
# SCRIPT CONFIGURATION
# =====================================================================================
SCRIPT_CONFIG = {
    "rollout_steps": 60,
    "input_len": 2,
    "plotting": {
        "cmap": sns.color_palette("icefire", as_cmap=True),
        "err_cmap": "coolwarm",
        "frame_stride": 5,
        "max_frames_to_plot": 8,
    },
    "latex": {
        "metric_map": {"L2": "MSE", "LSIM": "LSiM"},
        "metric_order": ["MSE", "LSiM"],
        "model_display_map": {
            "ACDM_ncn": r"$ACDM_{ncn}$",
            "ACDM": "ACDM",
            "KAE": "KAE",
        },
    },
}

# =====================================================================================
# HELPER FUNCTIONS
# =====================================================================================


def get_split_for_re(re_val: int) -> str:
    """Determines the data split ('train', 'val', 'test') for a given Reynolds number."""
    if re_val in [490, 500]:
        return "test"
    elif (100 <= re_val <= 190) or (900 <= re_val <= 1000):
        return "val"
    elif 200 <= re_val <= 890:
        return "train"
    else:
        raise ValueError(
            f"Reynolds number {re_val} does not fall into any defined data split."
        )


def load_kae_model(model_config: dict, device: torch.device) -> Tuple[nn.Module, Any]:
    """Loads a Koopman Autoencoder model and its configuration with robust error handling."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "configs")

    # Relative paths as in YAML
    cfg_path = model_config.get("config_path")
    checkpoint_path = model_config.get("checkpoint_path")

    # Construct absolute paths only for existence check
    abs_cfg_path = (
        os.path.join(base_dir, cfg_path)
        if cfg_path and not os.path.isabs(cfg_path)
        else cfg_path
    )
    abs_checkpoint_path = (
        os.path.join(script_dir, checkpoint_path)
        if checkpoint_path and not os.path.isabs(checkpoint_path)
        else checkpoint_path
    )

    # Check if files exist, but keep original relative paths for loading
    if not abs_cfg_path or not os.path.exists(abs_cfg_path):
        logger.warning(f"KAE config path not found, skipping model: {cfg_path}")
        return None, None
    if not abs_checkpoint_path or not os.path.exists(abs_checkpoint_path):
        logger.warning(
            f"KAE checkpoint path not found, skipping model: {checkpoint_path}"
        )
        return None, None

    # Load and merge structured Config if present, otherwise just load the file
    cfg = load_config(cfg_path)

    try:
        input_frames = int(cfg.data.input_sequence_length)
    except Exception:
        input_frames = int(cfg.data.input_sequence_length)

    try:
        if cfg.model.re_cond_type not in (None, "late_fusion", "adaln"):
            raise ValueError(f"Invalid re_cond_type: {cfg.model.re_cond_type}")
        if cfg.model.operator_mode not in ("linear", "eigen", "mlp"):
            raise ValueError(f"Invalid operator_mode: {cfg.model.operator_mode}")

        # Cast to satisfy mypy
        re_cond_type = cast(
            Optional[Literal["late_fusion", "adaln"]], cfg.model.re_cond_type
        )
        operator_mode = cast(Literal["linear", "eigen", "mlp"], cfg.model.operator_mode)

        model = KoopmanAutoencoder(
            data_variables=cfg.data.variables,
            input_frames=input_frames,
            height=cfg.model.height,
            width=cfg.model.width,
            latent_dim=cfg.model.latent_dim,
            re_embedding_dim=cfg.model.re_embedding_dim,
            re_cond_type=re_cond_type,
            operator_mode=operator_mode,
            hidden_dims=cfg.model.hidden_dims,
            transformer_config=cfg.model.transformer,
            use_checkpoint=False,
            predict_re=cfg.model.predict_re,
            re_grad_enabled=cfg.model.re_grad_enabled,
            is_continuous=cfg.model.is_continuous,
            **cfg.model.conv_kwargs,
        ).to(device)
    except Exception:
        logger.exception(
            "Failed to instantiate KoopmanAutoencoder. Check config for missing fields."
        )
        raise

    # Optimizer placeholder to pass to load_checkpoint
    optimizer = optim.Adam(
        model.parameters(), lr=getattr(cfg, "lr_scheduler", {}).get("lr", 1e-3)
    )

    # Load checkpoint
    try:
        if checkpoint_path is None:
            raise FileNotFoundError("Checkpoint path is missing")
        model, _, _, _ = load_checkpoint(
            checkpoint_path, model=model, optimizer=optimizer
        )
    except FileNotFoundError:
        logger.exception(f"Checkpoint not found at {checkpoint_path}")
        raise
    except Exception as e:
        logger.exception(f"Failed to load_checkpoint for KAE: {e}")
        # Attempt a strict=False state_dict load if a raw state_dict is present
        try:
            loaded = torch.load(checkpoint_path, map_location=device)
            state_dict = None
            if isinstance(loaded, dict):
                # typical fields: 'state_dict', 'model', 'model_state_dict'
                for k in ("state_dict", "model", "model_state_dict", "stateDict"):
                    if k in loaded:
                        state_dict = loaded[k]
                        break
                # fallback entire dict as state dict (if it's a flat mapping)
                if state_dict is None and all(
                    isinstance(v, torch.Tensor) for v in loaded.values()
                ):
                    state_dict = loaded
            if state_dict:
                model.load_state_dict(state_dict, strict=False)
            else:
                raise RuntimeError("No recognizable state_dict found in checkpoint.")
        except Exception as e2:
            logger.exception(f"Fallback checkpoint loading failed: {e2}")
            raise

    model.eval()
    return model, cfg


def load_acdm_model(model_config: dict, device: torch.device) -> nn.Module:
    """Loads an ACDM (diffusion) model with defensive checks."""
    checkpoint_path = model_config.get("checkpoint_path")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"ACDM checkpoint path not found: {checkpoint_path}")

    # Default model parameters; user can expand to read from config if needed
    p_md = ModelParamsDecoder(
        arch="direct-ddpm+Prev",
        diffSteps=20,
        diffSchedule="linear",
        diffCondIntegration="noisy",
        trainingNoise=0.0,
    )
    p_d = DataParams(
        batch=64,
        augmentations=["normalize"],
        sequenceLength=[[SCRIPT_CONFIG["rollout_steps"], 2]],
        randSeqOffset=True,
        dataSize=[128, 64],
        dimension=2,
        simFields=["pres"],
        simParams=["rey"],
        normalizeMode="incMixed",
    )
    model = DiffusionModel(p_d, p_md, dimension=0, condChannels=8)
    model.training = False
    model.inferenceConditioningIntegration = (
        "clean" if model_config.get("model_type") == "acdm_ncn" else "noisy"
    )

    loaded = torch.load(checkpoint_path, map_location=device)
    # Try a couple of likely keys to find the decoder state dict
    state_dict = None
    if isinstance(loaded, dict):
        for candidate in (
            "stateDictDecoder",
            "state_dict_decoder",
            "decoder_state_dict",
            "state_dict",
        ):
            if candidate in loaded:
                state_dict = loaded[candidate]
                break
        if state_dict is None:
            # If loaded itself looks like a state dict mapping
            if all(isinstance(v, torch.Tensor) for v in loaded.values()):
                state_dict = loaded

    if state_dict is None:
        raise KeyError(f"No decoder state dict found in checkpoint: {checkpoint_path}")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logger.warning(f"Strict load failed for ACDM decoder: {e}. Trying strict=False")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


def safe_first_re_from_loader(loader) -> Any:
    """Attempt to read the first Re value from a loader.dataset in a safe way."""
    try:
        ds = getattr(loader, "dataset", None)
        if ds is None:
            return None
        # Allow dataset.Re to be a tensor-like or list-like
        Re_attr = getattr(ds, "Re", None)
        if Re_attr is None:
            # maybe dataset has attributes or metadata
            if hasattr(ds, "__len__") and len(ds) > 0:
                first = ds[0]
                # try common keys
                if isinstance(first, dict):
                    if "Re" in first:
                        return (
                            int(first["Re"].item())
                            if hasattr(first["Re"], "item")
                            else int(first["Re"])
                        )
                    for k in ("rey", "re"):
                        if k in first:
                            return (
                                int(first[k].item())
                                if hasattr(first[k], "item")
                                else int(first[k])
                            )
                # fallback: none
            return None
        # If it's a tensor
        if hasattr(Re_attr, "__len__") and len(Re_attr) > 0:
            first = Re_attr[0]
            try:
                return int(first.item()) if hasattr(first, "item") else int(first)
            except Exception:
                return None
        # Single scalar
        try:
            return int(Re_attr)
        except Exception:
            return None
    except Exception:
        logger.exception("Failed to extract Re from loader.dataset.")
        return None


def generate_comparison_plots(
    ground_truth, all_predictions: Dict, variable: str, re_val: int
):
    """Plotting for comparison between GT and predictions. Defensive: only run if tensors exist."""
    cfg: DictConfig = SCRIPT_CONFIG["plotting"]
    logger.info(f"Generating comparison plots for '{variable}' at Re={re_val}")
    model_names = list(all_predictions.keys())
    gt_seq = ground_truth.get(variable) if isinstance(ground_truth, dict) else None
    if gt_seq is None:
        logger.warning(
            "Ground truth sequence not available for plotting; skipping plot."
        )
        return
    # Ensure sequences are tensors
    try:
        pass
    except Exception:
        logger.exception("Torch not importable during plotting.")
        return

    all_seqs = [gt_seq] + [
        p.get(variable) for p in all_predictions.values() if p.get(variable) is not None
    ]
    if not all_seqs:
        logger.warning(
            "No sequences available to determine color limits; skipping plot."
        )
        return

    try:
        vmin = min(s.min() for s in all_seqs if s is not None).item()
        vmax = max(s.max() for s in all_seqs if s is not None).item()
    except Exception:
        # fallback numeric conversion
        vmin = float(min(np.nanmin(s.cpu().numpy()) for s in all_seqs if s is not None))
        vmax = float(max(np.nanmax(s.cpu().numpy()) for s in all_seqs if s is not None))

    norm = Normalize(vmin=vmin, vmax=vmax)
    errors = [
        gt_seq - p.get(variable)
        for p in all_predictions.values()
        if p.get(variable) is not None
    ]
    max_abs_err = max(torch.abs(e).max() for e in errors).item() if errors else 1.0
    err_norm = Normalize(vmin=-max_abs_err, vmax=max_abs_err)

    num_rows = 1 + (2 * len(model_names))
    indices = range(0, gt_seq.shape[0], int(cfg["frame_stride"]))
    num_cols = min(len(list(indices)), int(cfg["max_frames_to_plot"]))

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(2.5 * num_cols, 2.2 * num_rows),
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
    )
    # Normalize axes indexing when num_rows/cols may be 1
    if num_rows == 1:
        axes = axes.reshape((1, num_cols))
    for j, frame_idx in enumerate(list(indices)[:num_cols]):
        ax = axes[0, j]
        ax.imshow(
            gt_seq[frame_idx].cpu().T, cmap=cfg["cmap"], norm=norm, origin="lower"
        )
        ax.set_title(f"t={frame_idx}")
        ax.axis("off")
    axes[0, 0].text(
        -0.2,
        0.5,
        f"Ground Truth\nRe={re_val}",
        fontsize=12,
        fontweight="bold",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="center",
    )
    for i, name in enumerate(model_names):
        pred_seq = all_predictions[name].get(variable)
        if pred_seq is None:
            continue
        err_seq = gt_seq - pred_seq
        pred_row, err_row = 1 + 2 * i, 2 + 2 * i
        for j, frame_idx in enumerate(list(indices)[:num_cols]):
            axes[pred_row, j].imshow(
                pred_seq[frame_idx].cpu().T, cmap=cfg["cmap"], norm=norm, origin="lower"
            )
            axes[pred_row, j].axis("off")
            axes[err_row, j].imshow(
                err_seq[frame_idx].cpu().T,
                cmap=cfg["err_cmap"],
                norm=err_norm,
                origin="lower",
            )
            axes[err_row, j].axis("off")
        axes[pred_row, 0].text(
            -0.2,
            0.5,
            f"{name}\nPrediction",
            transform=axes[pred_row, 0].transAxes,
            ha="right",
            va="center",
            fontweight="bold",
        )
        axes[err_row, 0].text(
            -0.2,
            0.5,
            f"{name}\nError",
            transform=axes[err_row, 0].transAxes,
            ha="right",
            va="center",
            fontweight="bold",
        )
    fig.suptitle(f"Rollout Comparison for '{variable.upper()}'", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0.05, 0, 0.95, 0.95])
    filename = f"comparison_rollout_{variable}_Re{re_val}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comparison plot to {filename}")


def generate_comparison_latex_table_by_split(all_results: dict) -> str:
    """
    Generate a LaTeX table comparing quantitative results across data splits.
    """
    cfg: DictConfig = SCRIPT_CONFIG["latex"]
    split_names = list(all_results.keys())
    if not split_names:
        return "No results to generate table."

    # pick the first split that has models
    model_names = []
    for split in split_names:
        if all_results.get(split):
            model_names = list(all_results[split].keys())
            if model_names:
                break
    if not model_names:
        return "No model results to generate table."

    metric_order = cfg.get("metric_order", ["MSE", "LSiM"])

    def format_val(mean, std):
        """Format mean ± std in scientific notation scaled by 1e-4."""
        if mean is None or std is None or np.isnan(mean) or np.isnan(std):
            return "-"
        scale, precision = 1e-4, 2
        return f"${mean / scale:.{precision}f} \\pm {std / scale:.{precision}f}$"

    # build tabular column specification: 1st col=split, 2nd col=method, rest=metrics
    column_spec = "@{}lc" + "c" * len(metric_order) + "@{}"

    # table header
    header = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        r"\textbf{Data Split} & \textbf{Method} & "
        + " & ".join([f"{m} $(\\times 10^{{-4}})$" for m in metric_order])
        + r" \\ \midrule",
    ]

    rows = []
    for split_name in split_names:
        for i, model_name in enumerate(model_names):
            display_name = cfg["model_display_map"].get(model_name, model_name)

            # only show split name once per block using multirow
            row_prefix = (
                rf"\multirow{{{len(model_names)}}}{{*}}{{{split_name}}}"
                if i == 0
                else ""
            )

            # add horizontal midrule between split blocks
            if i == 0 and split_name != split_names[0]:
                rows.append(r"\midrule")

            row_data = [row_prefix, display_name]

            for display_metric in metric_order:
                internal_key = next(
                    (k for k, v in cfg["metric_map"].items() if v == display_metric),
                    None,
                )
                metrics = all_results.get(split_name, {}).get(model_name, {})
                mean, std = (np.nan, np.nan)
                if internal_key and metrics:
                    mean_std = metrics.get(internal_key, {}).get("all")
                    if mean_std:
                        mean, std = mean_std
                row_data.append(format_val(mean, std))

            rows.append(" & ".join(row_data) + r" \\")

    # final table
    table = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Quantitative Comparison by Data Split}",
        r"\label{tab:quantitative_split_comparison}",
        *header,
        "\n".join(rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(table)


def generate_comparison_latex_table_by_re(all_results_by_re: dict) -> str:
    cfg: DictConfig = SCRIPT_CONFIG["latex"]
    if not all_results_by_re:
        return "No results to generate table."

    re_values = sorted(all_results_by_re.keys())
    first_re = next((re for re in re_values if all_results_by_re.get(re)), None)
    if not first_re:
        return "No valid results found to create table."
    model_names = sorted(all_results_by_re[first_re].keys())

    def format_val(mean, std):
        if mean is None or std is None or np.isnan(mean) or np.isnan(std):
            return "-"
        scale, precision = 1e-4, 2
        return f"${mean / scale:.{precision}f} \pm {std / scale:.{precision}f}$"

    # header rows
    header1 = (
        r"\multirow{2}{*}{\textbf{Method}} & "
        + " & ".join(
            [f"\multicolumn{{2}}{{c}}{{\textbf{{Re={re}}}}}" for re in re_values]
        )
        + r" \\"
    )
    cmidrules = " ".join(
        [f"\\cmidrule(lr){{{2+i*2}-{3+i*2}}}" for i in range(len(re_values))]
    )
    header2_parts = [
        f"{metric} $(\\times 10^{{-4}})$"
        for re in re_values
        for metric in cfg["metric_order"]
    ]
    header2 = " & ".join(["", *header2_parts]) + r" \\ \midrule"

    # data rows
    rows = []
    for model_name in model_names:
        display_name = cfg["model_display_map"].get(model_name, model_name)
        row_data = [display_name]
        for re in re_values:
            for display_metric in cfg["metric_order"]:
                internal_key = next(
                    (k for k, v in cfg["metric_map"].items() if v == display_metric),
                    None,
                )
                metrics = all_results_by_re.get(re, {}).get(model_name, {})
                mean, std = (np.nan, np.nan)
                if internal_key and metrics:
                    mean_std = metrics.get(internal_key, {}).get("all")
                    if mean_std:
                        mean, std = mean_std
                row_data.append(format_val(mean, std))
        rows.append(" & ".join(row_data) + r" \\")

    # build table
    table = [
        r"\begin{table*}[h!]",
        r"\centering",
        r"\caption{Quantitative comparison of prediction accuracy.}",
        r"\label{tab:quantitative_comparison}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}l " + " ".join(["cc"] * len(re_values)) + r"@{}}",
        r"\toprule",
        header1,
        cmidrules,
        header2,
        "\n".join(rows),
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
    ]
    return "\n".join(table)


def generate_plots_for_re(
    models_with_rollout_fns: Dict[str, Dict[str, Any]],
    loader,
    re_val: int,
    initial_sample_index: int = 300,
    rollout_steps: int = 61,
):
    """
    Generates comparison plots for given models, each with its own rollout function.
    This function is defensive: if data/predictions are incompatible, it logs and returns.
    """
    logger.info("Attempting to generate plots for Re=%s", re_val)

    # 1. Obtain a single ground-truth sequence and initial condition
    try:
        ds = getattr(loader, "dataset", None)
        if not ds or len(ds) <= initial_sample_index:
            logger.warning("Dataset unavailable or too small. Skipping plotting.")
            return

        # Fetch the data sample required for rollouts
        input_data, ground_truth, metadata = ds[initial_sample_index, rollout_steps]
        # Make the initial condition usable by rollout functions
        initial_condition = {
            "v_x": input_data["v_x"],
            "v_y": input_data["v_y"],
            "obstacle_mask": metadata["obstacle_mask"][0].repeat(
                input_data["v_x"].shape[0], 1, 1, 1
            ),
            "Re_input": metadata["Re_input"][0].repeat(input_data["v_x"].shape[0]),
        }

    except Exception as e:
        logger.exception(f"Could not retrieve a data sample for plotting: {e}")
        return

    # 2. Generate predictions for each model using its specific rollout function
    all_predictions: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, entry in models_with_rollout_fns.items():
        model = entry.get("model")
        rollout_fn = entry.get("rollout_fn")

        if not model or not rollout_fn:
            logger.warning(
                f"Skipping model '{name}' due to missing model or rollout_fn."
            )
            continue

        try:
            model.eval()
            with torch.no_grad():
                # The rollout function is now self-contained
                prediction = rollout_fn(model, initial_condition)
                all_predictions[name] = prediction
            logger.info(f"Successfully generated rollout for model '{name}'.")
        except Exception:
            logger.exception(f"Failed to run rollout for model '{name}'.")
            all_predictions[name] = {}

    # 3. Plot the results if we have ground truth and at least one prediction
    if not all_predictions:
        logger.warning("No predictions were generated; skipping plot creation.")
        return

    var_to_plot = next(iter(ground_truth.keys()), None)
    if var_to_plot:
        generate_comparison_plots(ground_truth, all_predictions, var_to_plot, re_val)
    else:
        logger.warning("Ground truth dictionary is empty; cannot generate plots.")


def main(args):
    """Main script to load, evaluate, and compare all models defined in the YAML config."""
    if not os.path.exists(args.eval_config):
        logger.error("Evaluation config not found: %s", args.eval_config)
        return

    with open(args.eval_config, "r") as f:
        try:
            eval_config = yaml.safe_load(f)
        except Exception:
            logger.exception("Failed to parse YAML evaluation config.")
            return

    if not isinstance(eval_config, dict):
        logger.error("Evaluation config is not a mapping/dictionary.")
        return

    evaluations = eval_config.get("evaluations", [])
    if not evaluations:
        logger.error("No 'evaluations' list found in the eval_config.")
        return

    # --- Setup Dataloaders ---
    base_kae_cfg = None
    models_to_plot: Dict = {}
    try:
        # Try to find any kae config if available, but tolerate absence
        kae_entry = next((m for m in evaluations if m.get("model_type") == "kae"), None)
        if kae_entry:
            base_kae_cfg_path = kae_entry.get("config_path")
            base_kae_cfg = load_config(base_kae_cfg_path)

            # harmonize attribute naming for rollouts
            try:
                # prefer explicit attr names if present
                if hasattr(base_kae_cfg.data, "max_sequence_length"):
                    rollout_steps: int = int(cast(int, SCRIPT_CONFIG["rollout_steps"]))
                    base_kae_cfg.data.max_sequence_length = rollout_steps
                else:
                    setattr(
                        base_kae_cfg.data,
                        "max_sequence_length",
                        SCRIPT_CONFIG["rollout_steps"],
                    )
            except Exception:
                logger.debug(
                    "Setting max_seq_length on base_kae_cfg failed; continuing."
                )
        # Use load_datasets with base_kae_cfg if available; otherwise pass None and rely on function defaults.
        assert base_kae_cfg is not None

        # tell mypy that it's a Config
        cfg: Config = base_kae_cfg

        # now safe to pass to functions
        train_ds, val_ds, test_ds = load_datasets(cfg)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds, cfg.training
        )
    except Exception as e:
        logger.exception("Failed to load datasets. Error: %s", e)
        return

    all_results_split: Dict = {}
    all_results_re: Dict = {}

    # Evaluate across the three static splits
    splits_to_evaluate = {
        "Train": train_loader,
        "Validation": val_loader,
        "Test": test_loader,
    }

    for split_name, loader in splits_to_evaluate.items():
        # If loader is None or its dataset empty -> skip gracefully
        if loader is None:
            logger.info(f"Skipping '{split_name}' split because loader is None.")
            continue
        if getattr(loader, "dataset", None) is None:
            logger.info(
                f"Skipping '{split_name}' split because loader.dataset is None."
            )
            continue
        # Some dataloaders wrap empty datasets but still exist; check length if possible
        try:
            if len(loader.dataset) == 0:
                logger.info(f"Skipping '{split_name}' split as its dataset is empty.")
                continue
        except Exception:
            # If dataset doesn't implement __len__, try to access first element safely
            try:
                _ = loader.dataset[0]
            except Exception:
                logger.info(
                    f"Skipping '{split_name}' split as dataset appears not indexable or empty."
                )
                continue

        # Try to extract a first Re value for logging/plotting; safe fallback to 'unknown'
        re_val = safe_first_re_from_loader(loader)
        logger.info(
            f"\n{'='*30} Evaluating on '{split_name}' Split (Re={re_val}) {'='*30}"
        )
        all_results_split[split_name] = {}
        all_results_re[re_val] = {}

        for model_config in evaluations:
            model_name = model_config.get("name", "<unnamed>")
            model_type = model_config.get("model_type", "").lower()
            logger.info(f"\n--- Evaluating model: {model_name} ({model_type}) ---")
            model = None
            rollout_fn = None

            try:
                if model_type == "kae":
                    try:
                        model, cfg = load_kae_model(model_config, DEVICE)
                        rollout_fn = kae_rollout_wrapper
                    except Exception as e:
                        logger.exception(
                            f"Failed to load KAE model '{model_name}': {e}"
                        )
                        all_results_split[split_name][model_name] = {}
                        all_results_re[re_val][model_name] = {}
                        continue
                elif model_type in ("acdm", "acdm_ncn"):
                    try:
                        model = load_acdm_model(model_config, DEVICE)
                        rollout_fn = cast(
                            Optional[Callable[..., Any]], run_diffusion_rollout
                        )
                    except Exception as e:
                        logger.exception(
                            f"Failed to load ACDM model '{model_name}': {e}"
                        )
                        all_results_split[split_name][model_name] = {}
                        all_results_re[re_val][model_name] = {}
                        continue
                else:
                    logger.warning(
                        f"Unknown model type '{model_type}' for model '{model_name}'. Skipping."
                    )
                    continue

                # If the rollout_fn or run_evaluation isn't available for some reason, skip gracefully
                if rollout_fn is None:
                    logger.warning(
                        f"No rollout function available for model '{model_name}'. Skipping evaluation."
                    )
                    all_results_split[split_name][model_name] = {}
                    all_results_re[re_val][model_name] = {}
                    continue

                # Run evaluation in try/except so single model failure doesn't stop the loop
                try:
                    input_len: int = int(cast(int, SCRIPT_CONFIG["input_len"]))
                    output_len: int = int(cast(int, SCRIPT_CONFIG["rollout_steps"]))

                    metrics = run_evaluation(
                        model=model,
                        loader=loader,
                        input_len=input_len,
                        output_len=output_len,
                        rollout_fn=rollout_fn,
                    )
                    all_results_split[split_name][model_name] = metrics or {}
                    all_results_re[re_val][model_name] = metrics or {}
                except Exception as e:
                    logger.exception(
                        f"run_evaluation failed for model '{model_name}' on split '{split_name}': {e}"
                    )
                    # store empty metrics so table generation knows model was attempted
                    all_results_split[split_name][model_name] = metrics or {}
                    all_results_re[re_val][model_name] = metrics or {}
                    continue

            except Exception as e:
                logger.exception(
                    f"Unexpected error while setting up model '{model_name}': {e}"
                )
                all_results_split[split_name][model_name] = {}
                all_results_re[re_val][model_name] = {}
                continue

        if args.generate_plots:
            logger.info(f"Preparing models for plotting on split '{split_name}'...")
            # We need a single metadata sample for the model registry builders
            try:
                _, _, metadata_for_plot = loader.dataset[
                    300, 61
                ]  # Use the same index as plotting
            except Exception as e:
                logger.error(
                    f"Cannot get metadata for plotting, skipping plots. Error: {e}"
                )
                continue

            for mcfg in evaluations:
                name = mcfg.get("name")
                model_type = mcfg.get("model_type", "").lower()

                # Use a display name for keys if available, otherwise use model type
                key = name if name else model_type
                if not key:
                    logger.warning(
                        "Skipping a model in eval_config with no name or type."
                    )
                    continue

                try:
                    builder_fn = None
                    if model_type == "kae":
                        builder_fn = MODEL_REGISTRY.get("KAE")
                    elif model_type in ("acdm", "acdm_ncn"):
                        builder_fn = MODEL_REGISTRY.get("Diffusion")

                    if builder_fn:
                        models_to_plot[name] = builder_fn(
                            cfg=base_kae_cfg,  # Use the globally loaded KAE config
                            ckpt_path=mcfg.get("checkpoint_path"),
                            metadata=metadata_for_plot,
                            val_dataset=loader.dataset,
                            device=DEVICE,
                            rollout_steps=SCRIPT_CONFIG["rollout_steps"],
                        )
                        logger.info(f"Successfully built '{name}' for plotting.")
                    else:
                        logger.warning(
                            f"No model builder found in registry for type '{model_type}'."
                        )

                except Exception as e:
                    logger.exception(
                        f"Failed to build model '{name}' for plotting. Error: {e}"
                    )

            if models_to_plot:
                # rollout_steps: int = int(cast(int, SCRIPT_CONFIG["rollout_steps"]))
                generate_plots_for_re(
                    models_with_rollout_fns=models_to_plot,
                    loader=loader,
                    re_val=re_val,
                    initial_sample_index=300,  # Standardized index
                    rollout_steps=rollout_steps,
                )
            else:
                logger.warning("No models were successfully built for plotting.")

    # --- Final Report Generation ---
    logger.info(
        "\n"
        + "=" * 80
        + "\nAll evaluations complete. Generating LaTeX summary table.\n"
        + "=" * 80
    )
    try:
        latex_table_split = generate_comparison_latex_table_by_split(all_results_split)
    except Exception:
        logger.exception("Failed to generate LaTeX table by split.")
        latex_table_split = "Failed to create LaTeX table by split."

    try:
        latex_table_re = generate_comparison_latex_table_by_re(all_results_re)
    except Exception:
        logger.exception("Failed to generate LaTeX table by Re.")
        latex_table_re = "Failed to create LaTeX table by Re."

    print(latex_table_split)
    print(latex_table_re)

    output_filename_split = "metrics_table_split_comparison.tex"
    output_filename_re = "metrics_table_re_comparison.tex"
    try:
        with open(output_filename_split, "w") as f:
            f.write(latex_table_split)
        logger.info(f"LaTeX table saved to {output_filename_split}")
        with open(output_filename_re, "w") as f:
            f.write(latex_table_re)
        logger.info(f"LaTeX table saved to {output_filename_re}")
    except Exception:
        logger.exception("Failed to save LaTeX file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated multi-model evaluation pipeline."
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        required=True,
        help="Path to the YAML evaluation configuration file.",
    )
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        help="Generate and save joint rollout plots.",
    )
    args = parser.parse_args()
    main(args)
