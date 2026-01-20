# models/utils.py

import json
import os
from tensordict import TensorDict
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from pathlib import Path
from typing import Dict, Any, Type, Tuple, Union, List, Optional
import logging
from omegaconf import OmegaConf, DictConfig
from hydra import initialize, compose

from .metrics import Metric
from .config_classes import Config
from .dataloader import (
    QGDatasetBase,
    QGDatasetMultiSim,
    SingleSimOverfit,
    AbstractNormalizer,
    MeanStdNormalizer,
    QuantileNormalizer,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def tensor_dict_to_json(data: Union[TensorDict, Tensor]):
    """Recursively converts a TensorDict or Tensor to JSON-serializable types."""
    if isinstance(data, Tensor):
        return data.item() if data.numel() == 1 else data.cpu().numpy().tolist()
    if isinstance(data, TensorDict):
        return {key: tensor_dict_to_json(value) for key, value in data.items()}
    raise TypeError(f"Unsupported type for JSON conversion: {type(data)}")


def accumulate_losses(
    total_losses: Dict[str, Tensor], losses: Dict[str, Tensor]
) -> Dict[str, Tensor]:
    """Accumulates loss values from a dictionary into a running total."""
    for key, value in losses.items():
        if not isinstance(value, Tensor):
            continue
        if key not in total_losses:
            total_losses[key] = value.clone()
        else:
            total_losses[key] += value
    return total_losses


def average_losses(total_losses: Dict[str, Tensor], n_batches: int) -> Dict[str, float]:
    """Averages accumulated losses and converts to floats."""
    if n_batches == 0:
        return {k: 0.0 for k in total_losses}
    return {key: (value / n_batches).item() for key, value in total_losses.items()}


# def load_config(
#     config_path: Union[str, None], cli_args: Optional[List[str]] = None
# ) -> Config:
#     """
#     Load a Hydra config into the structured Config dataclass.
#     Works in notebooks and scripts.

#     Args:
#         config_path: Path to the experiment YAML relative to the configs root, e.g., "experiment/128_inc"
#         cli_args: Optional list of CLI-style overrides.

#     Returns:
#         Config: Structured and fully resolved configuration.
#     """
#     # Repo root (assumes this file is in models/)
#     repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#     configs_root_abs = os.path.join(repo_root, "../configs")

#     # Make configs_root relative to current working dir (Hydra requires relative paths)
#     configs_root = os.path.relpath(configs_root_abs, start=os.getcwd())

#     # Strip .yaml if present
#     if config_path is not None:
#         config_name = os.path.splitext(config_path)[0]
#     else:
#         config_name = ""

#     # Initialize Hydra from the relative configs root
#     with initialize(config_path=configs_root, version_base=None):
#         cfg_dict = compose(config_name=config_name, overrides=cli_args or [])

#     # Extract 'experiment' if present
#     cfg_dict = cfg_dict.get("experiment", cfg_dict)

#     # Merge into structured dataclass
#     cfg: Config = OmegaConf.merge(OmegaConf.structured(Config()), cfg_dict)
#     OmegaConf.resolve(cfg)

#     return cfg


def _find_config_root(cfg: Any) -> Optional[DictConfig]:
    """
    Recursively searches for the actual configuration root within a nested DictConfig.
    It identifies the root by looking for keys that MUST exist in your Config schema,
    specifically 'output_dir' or 'data'.
    """
    if not isinstance(cfg, (dict, DictConfig)):
        raise NotImplementedError("Config needs to be read by OmegaConf")

    # Check if this node looks like the config root
    # (We check for 'data' because 'output_dir' might sometimes be missing/defaulted)
    if "data" in cfg or "output_dir" in cfg:
        return cfg

    # Recursively search children
    for key in cfg:
        val = cfg[key]
        if isinstance(val, (dict, DictConfig)):
            found = _find_config_root(val)
            if found is not None:
                return found
    return None


def load_config(
    config_path: Union[str, None], cli_args: Optional[List[str]] = None
) -> Config:
    """
    Load a Hydra config into the structured Config dataclass.
    Works in notebooks and scripts.

    Args:
        config_path: Path to the experiment YAML relative to the configs root, e.g., "experiment/stable/128_inc"
        cli_args: Optional list of CLI-style overrides.

    Returns:
        Config: Structured and fully resolved configuration.
    """

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    configs_root_abs = os.path.join(repo_root, "../configs")

    try:
        configs_root = os.path.relpath(configs_root_abs, start=os.getcwd())
    except ValueError:
        configs_root = configs_root_abs

    if config_path is not None:
        config_name = os.path.splitext(config_path)[0]
    else:
        config_name = ""

    with initialize(config_path=configs_root, version_base=None):
        cfg_dict = compose(config_name=config_name, overrides=cli_args or [])

    real_config = _find_config_root(cfg_dict)

    if real_config is None:
        real_config = cfg_dict

    cfg: Config = OmegaConf.merge(OmegaConf.structured(Config()), real_config)

    if OmegaConf.is_missing(cfg, "output_dir"):
        cfg.output_dir = "outputs"

    OmegaConf.resolve(cfg)

    return cfg


def get_dataset_class(dataset_type_str: str) -> Type[QGDatasetBase]:
    """Maps a string from the config to an actual dataset class."""
    class_map = {
        "QGDatasetBase": QGDatasetBase,
        "QGDatasetMultiSim": QGDatasetMultiSim,
        "SingleSimOverfit": SingleSimOverfit,
    }
    dataset_class = class_map.get(dataset_type_str)
    if dataset_class is None:
        raise ValueError(f"Unknown dataset_type: '{dataset_type_str}'")
    return dataset_class


def get_normalizer(cfg: Config) -> AbstractNormalizer:
    """Instantiates a normalizer based on the config."""
    norm_type = cfg.data.normalization.type
    if norm_type == "MeanStdNormalizer":
        return MeanStdNormalizer()
    elif norm_type == "QuantileNormalizer":
        return QuantileNormalizer(quantile_range=cfg.data.quantile_range)
    else:
        raise ValueError(f"Unknown normalization type: '{norm_type}'")


def load_datasets(
    cfg: Config, ignore_cond: Optional[bool] = False
) -> Tuple[QGDatasetBase, QGDatasetBase, QGDatasetBase]:
    """
    Loads the training, validation, and test datasets based on the provided config.
    """
    base_data_dir = Path(cfg.data.data_dir)
    try:
        DatasetClass = get_dataset_class(cfg.data.dataset_type)

        common_args: Dict[str, Any] = {}

        if cfg.data.static_variables:
            common_args["static_variables"] = cfg.data.static_variables
        if cfg.data.control_parameters:
            common_args["control_parameters"] = cfg.data.control_parameters
        if cfg.data.selection_param:
            common_args["selection_param"] = cfg.data.selection_param

        train_dataset = DatasetClass(
            data_path=base_data_dir / cfg.data.train_file,
            normalizer=get_normalizer(cfg),
            input_sequence_length=cfg.data.input_sequence_length,
            max_sequence_length=cfg.data.max_sequence_length,
            variables=cfg.data.variables,
            subsample=cfg.data.subsample,
            exhaustive=cfg.data.exhaustive,
            select_cond=None if ignore_cond else cfg.data.train_select_cond,
            **common_args,  # This will now pass all the new args
        )
        val_dataset = DatasetClass(
            data_path=base_data_dir / cfg.data.val_file,
            normalizer=get_normalizer(cfg),
            input_sequence_length=cfg.data.input_sequence_length,
            max_sequence_length=cfg.data.max_sequence_length,
            variables=cfg.data.variables,
            subsample=cfg.data.subsample,
            exhaustive=cfg.data.exhaustive,
            select_cond=None if ignore_cond else cfg.data.val_select_cond,
            **common_args,  # This will now pass all the new args
        )
        test_dataset = DatasetClass(
            data_path=base_data_dir / cfg.data.test_file,
            normalizer=get_normalizer(cfg),
            input_sequence_length=cfg.data.input_sequence_length,
            max_sequence_length=cfg.data.max_sequence_length,
            variables=cfg.data.variables,
            subsample=cfg.data.subsample,
            exhaustive=cfg.data.exhaustive,
            select_cond=None if ignore_cond else cfg.data.test_select_cond,
            **common_args,  # This will now pass all the new args
        )

        logger.info(
            f"Successfully loaded datasets with type '{cfg.data.dataset_type}'."
        )
        return train_dataset, val_dataset, test_dataset
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to load datasets: {e}", exc_info=True)
        raise


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optimizer,
    strict: Optional[bool] = True,
) -> Tuple[nn.Module, Optimizer, Dict[str, Any], int]:
    """
    Loads a model, optimizer, history, and start epoch from a checkpoint.
    Returns the initial state if the checkpoint is not found.
    """
    cp_path = Path(checkpoint_path)
    if not cp_path.is_file():
        logger.warning(f"Checkpoint file not found: {cp_path}. Starting from scratch.")
        return model, optimizer, {}, 0

    try:
        checkpoint = torch.load(cp_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError:
                print("Optimizer state incompatible, reinitializing optimizer")

        start_epoch = checkpoint.get("epoch", -1) + 1
        history = checkpoint.get("history", {})

        logger.info(
            f"Checkpoint loaded from {cp_path}. Resuming from epoch {start_epoch}."
        )
        return model, optimizer, history, start_epoch
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {cp_path}: {e}", exc_info=True)
        raise RuntimeError("Critical error loading checkpoint.") from e


def compute_all_metrics(
    target: TensorDict,
    prediction: TensorDict,
    loader,
    variables: List[str],
    custom_min_max: Optional[Dict[str, Tuple[float, float]]] = None,
) -> dict:
    """
    Computes all metrics (L2, SSIM, PSNR, VI) for each variable and for 'all'.
    Applies manual normalization for custom-computed variables like 'vort'.

    Args:
        target (TensorDict): Ground truth data (denormalized).
        prediction (TensorDict): Model predictions (denormalized).
        loader: The dataset or loader that contains to_unit_range().
        variables (List[str]): Variables to evaluate.
        custom_min_max (dict, optional): For manual normalization, e.g. {'vort': (min, max)}

    Returns:
        dict: Nested structure {metric_mode: {variable_name: (mean, std)}}
    """
    results: Dict = {}

    # --- Prepare data ---
    target_norm = {}
    prediction_norm = {}

    for var in variables:
        if custom_min_max and var in custom_min_max:
            # Use manual normalization
            vmin, vmax = custom_min_max[var]
            target_norm[var] = (target[var] - vmin) / (vmax - vmin + 1e-8)
            prediction_norm[var] = (prediction[var] - vmin) / (vmax - vmin + 1e-8)
        else:
            # Use loader's to_unit_range
            target_norm[var] = loader.to_unit_range(target)[var]
            prediction_norm[var] = loader.to_unit_range(prediction)[var]

    # Convert to TensorDict
    target_td = TensorDict(target_norm, batch_size=target.batch_size)
    pred_td = TensorDict(prediction_norm, batch_size=prediction.batch_size)

    # --- Run metrics ---
    for mode in Metric.VALID_MODES:
        results[mode] = {}

        for var in variables + ["all"]:
            variable_mode = "all" if var == "all" else "single"
            metric_fn = Metric(
                mode=mode,
                variable_mode=variable_mode,
                variable_name=None if variable_mode == "all" else var,
            )

            dist = metric_fn(target_td, pred_td)  # [B, T]
            results[mode][var] = (dist.mean().item(), dist.std().item())

    return results


def cuda_timer():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


def elapsed_time(start, end):
    return start.elapsed_time(end) / 1000.0  # ms → seconds


def surgically_transfer_checkpoint(
    old_checkpoint: Dict[str, Any],
    new_model: nn.Module,
    old_model_for_mapping: nn.Module,
    new_optimizer: Optimizer,
) -> Tuple[nn.Module, Optimizer]:
    """
    Performs model surgery to transfer weights and optimizer state from an old
    checkpoint to a new model architecture.

    This utility is essential when you've modified a model's architecture (e.g.,
    added a new layer) and want to resume training without starting from scratch.
    It preserves both the learned weights and the optimizer's momentum for all
    unchanged layers.

    Args:
        old_checkpoint (Dict[str, Any]): The loaded state dictionary from the old checkpoint file.
        new_model (nn.Module): An instance of the *new* model architecture.
        old_model_for_mapping (nn.Module): An instance of the *old* model architecture,
                                           used only to create a stable parameter name map.
        new_optimizer (optim.Optimizer): An instance of the optimizer configured for the *new* model.

    Returns:
        Tuple[nn.Module, optim.Optimizer]: The new model and optimizer, now populated
                                           with the transferred states.

    Raises:
        KeyError: If the checkpoint is missing required keys.
    """
    # 1. Transfer Model Weights
    # ---------------------------
    logger.info("Step 1: Transferring model weights...")
    old_model_state = old_checkpoint["model_state_dict"]

    # Load weights with strict=False to accommodate architectural changes
    missing_keys, unexpected_keys = new_model.load_state_dict(
        old_model_state, strict=False
    )

    if unexpected_keys:
        logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")
    if not missing_keys:
        logger.warning("No missing keys found. Model architectures may be identical.")
    else:
        logger.info(f"Successfully ignored missing keys (new layers): {missing_keys}")

    # 2. Reconstruct Optimizer State
    # --------------------------------
    logger.info("Step 2: Reconstructing optimizer state...")

    # Load the old model's state to create a name-based map
    old_model_for_mapping.load_state_dict(old_model_state)

    # Create mappings from parameter names to the actual parameter objects
    # This is more robust than using id(p), as names are consistent.
    old_params_by_name = {
        name: p for name, p in old_model_for_mapping.named_parameters()
    }
    new_params_by_name = {name: p for name, p in new_model.named_parameters()}

    old_optimizer_state = old_checkpoint["optimizer_state_dict"]["state"]
    new_optimizer_state = {}

    num_transferred = 0
    # Iterate over the old parameters to find their state
    for name, old_param in old_params_by_name.items():
        if name in new_params_by_name:
            # If the parameter still exists, find its state in the old optimizer
            if old_param in old_optimizer_state:
                # Get the corresponding parameter in the new model
                new_param = new_params_by_name[name]
                # Assign the old state to the new parameter
                new_optimizer_state[new_param] = old_optimizer_state[old_param]
                num_transferred += 1

    total_new_params = len(list(new_model.parameters()))
    logger.info(
        f"Transferred optimizer state for {num_transferred} / {total_new_params} parameters."
    )

    # Update the new optimizer's state dictionary
    new_optimizer.state = new_optimizer_state

    return new_model, new_optimizer


def save_timing_to_json(timing_data, model_name, filename="benchmarks.json"):
    # Load existing data if file exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Initialize model key if not present
    if model_name not in data:
        data[model_name] = []

    # Append new timing entry
    data[model_name].append(timing_data)

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
