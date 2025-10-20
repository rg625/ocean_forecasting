# evaluate_models_no_re_no_plot.py
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

from omegaconf import DictConfig

# --- Import Project Modules ---
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
except Exception:
    raise

# Diffusion Model Imports
try:
    from turbpred.params import DataParams, ModelParamsDecoder
    from turbpred.model_diffusion import DiffusionModel
except Exception:
    pass

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.transforms.functional"
)

# =====================================================================================
# SCRIPT CONFIGURATION
# =====================================================================================
SCRIPT_CONFIG = {
    "rollout_steps": 60,
    "input_len": 2,
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


def load_kae_model(model_config: dict, device: torch.device) -> Tuple[nn.Module, Any]:
    """Loads a Koopman Autoencoder model with robust error handling."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "configs")

    cfg_path = model_config.get("config_path")
    checkpoint_path = model_config.get("checkpoint_path")

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

    if not abs_cfg_path or not os.path.exists(abs_cfg_path):
        logger.warning(f"KAE config path not found: {cfg_path}")
        return None, None
    if not abs_checkpoint_path or not os.path.exists(abs_checkpoint_path):
        logger.warning(f"KAE checkpoint path not found: {checkpoint_path}")
        return None, None

    cfg = load_config(cfg_path)

    try:
        input_frames = int(cfg.data.input_sequence_length)
    except Exception:
        input_frames = int(cfg.data.input_sequence_length)

    try:
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
        logger.exception("Failed to instantiate KoopmanAutoencoder.")
        raise

    optimizer = optim.Adam(
        model.parameters(), lr=getattr(cfg, "lr_scheduler", {}).get("lr", 1e-3)
    )

    try:
        if checkpoint_path is None:
            raise FileNotFoundError("Checkpoint path is missing")
        model, _, _, _ = load_checkpoint(
            checkpoint_path, model=model, optimizer=optimizer, strict=True
        )
    except Exception as e:
        logger.exception(f"Failed to load_checkpoint for KAE: {e}")
        # fallback
        loaded = torch.load(checkpoint_path, map_location=device)
        state_dict = None
        if isinstance(loaded, dict):
            for k in ("state_dict", "model", "model_state_dict", "stateDict"):
                if k in loaded:
                    state_dict = loaded[k]
                    break
            if state_dict is None and all(
                isinstance(v, torch.Tensor) for v in loaded.values()
            ):
                state_dict = loaded
        if state_dict:
            model.load_state_dict(state_dict, strict=False)
        else:
            raise RuntimeError("No recognizable state_dict found in checkpoint.")

    model.eval()
    return model, cfg


def load_acdm_model(model_config: dict, device: torch.device) -> nn.Module:
    """Loads an ACDM (diffusion) model with defensive checks."""
    checkpoint_path = model_config.get("checkpoint_path")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"ACDM checkpoint path not found: {checkpoint_path}")

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
        if state_dict is None and all(
            isinstance(v, torch.Tensor) for v in loaded.values()
        ):
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


def main(args):
    """Main script to load, evaluate, and compare all models."""
    if not os.path.exists(args.eval_config):
        logger.error("Evaluation config not found: %s", args.eval_config)
        return

    with open(args.eval_config, "r") as f:
        try:
            eval_config = yaml.safe_load(f)
        except Exception:
            logger.exception("Failed to parse YAML evaluation config.")
            return

    evaluations = eval_config.get("evaluations", [])
    if not evaluations:
        logger.error("No 'evaluations' list found in the eval_config.")
        return

    # --- Setup Dataloaders ---
    try:
        kae_entry = next((m for m in evaluations if m.get("model_type") == "kae"), None)
        base_kae_cfg = load_config(kae_entry.get("config_path")) if kae_entry else None
        cfg: Config = base_kae_cfg
        cfg.data.max_sequence_length = SCRIPT_CONFIG["rollout_steps"]

        train_ds, val_ds, test_ds = load_datasets(cfg)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds, cfg.training
        )
    except Exception as e:
        logger.exception("Failed to load datasets. Error: %s", e)
        return

    all_results_split: Dict = {}
    splits_to_evaluate = {
        "Train": train_loader,
        "Validation": val_loader,
        "Test": test_loader,
    }

    for split_name, loader in splits_to_evaluate.items():
        if loader is None or getattr(loader, "dataset", None) is None:
            continue
        if len(loader.dataset) == 0:
            continue

        all_results_split[split_name] = {}

        for model_config in evaluations:
            model_name = model_config.get("name", "<unnamed>")
            model_type = model_config.get("model_type", "").lower()
            logger.info(f"Evaluating model: {model_name} ({model_type})")
            model = None
            rollout_fn = None

            try:
                if model_type == "kae":
                    model, cfg = load_kae_model(model_config, DEVICE)
                    rollout_fn = kae_rollout_wrapper
                elif model_type in ("acdm", "acdm_ncn"):
                    model = load_acdm_model(model_config, DEVICE)
                    rollout_fn = cast(
                        Optional[Callable[..., Any]], run_diffusion_rollout
                    )
                else:
                    logger.warning(f"Unknown model type '{model_type}'. Skipping.")
                    continue

                if rollout_fn is None:
                    continue

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
            except Exception as e:
                logger.exception(f"Failed evaluation for model '{model_name}': {e}")
                all_results_split[split_name][model_name] = {}

    # --- Generate LaTeX Table ---
    cfg_latex: DictConfig = SCRIPT_CONFIG["latex"]

    def generate_comparison_latex_table(all_results: dict) -> str:
        if not all_results:
            return "No results to generate table."
        split_names = list(all_results.keys())
        model_names = []
        for split in split_names:
            if all_results.get(split):
                model_names = list(all_results[split].keys())
                if model_names:
                    break
        if not model_names:
            return "No model results to generate table."

        metric_order = cfg_latex.get("metric_order", ["MSE", "LSiM"])

        def format_val(mean, std):
            if mean is None or std is None or np.isnan(mean) or np.isnan(std):
                return "-"
            scale, precision = 1e-4, 2
            return f"${mean / scale:.{precision}f} \\pm {std / scale:.{precision}f}$"

        column_spec = "@{}lc" + "c" * len(metric_order) + "@{}"
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
                display_name = cfg_latex["model_display_map"].get(
                    model_name, model_name
                )
                row_prefix = (
                    rf"\multirow{{{len(model_names)}}}{{*}}{{{split_name}}}"
                    if i == 0
                    else ""
                )
                if i == 0 and split_name != split_names[0]:
                    rows.append(r"\midrule")

                row_data = [row_prefix, display_name]
                for display_metric in metric_order:
                    internal_key = next(
                        (
                            k
                            for k, v in cfg_latex["metric_map"].items()
                            if v == display_metric
                        ),
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

    latex_table = generate_comparison_latex_table(all_results_split)
    print(latex_table)

    try:
        output_filename = "metrics_table_split_comparison.tex"
        with open(output_filename, "w") as f:
            f.write(latex_table)
        logger.info(f"LaTeX table saved to {output_filename}")
    except Exception:
        logger.exception("Failed to save LaTeX file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-model evaluation pipeline.")
    parser.add_argument(
        "--eval_config",
        type=str,
        required=True,
        help="Path to the YAML evaluation configuration file.",
    )
    args = parser.parse_args()
    main(args)
