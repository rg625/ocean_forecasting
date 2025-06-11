# models/utils.py
from tensordict import TensorDict
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from pathlib import Path
import yaml
from typing import Dict, Any, Type, Tuple, Union
import logging

# Re-importing locally to make this file self-contained and reflect fix
from .config_classes import Config
from .dataloader import (
    QGDatasetBase,
    QGDatasetMultiSim,
    SingleSimOverfit,
    AbstractNormalizer,
    MeanStdNormalizer,
    QuantileNormalizer,
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


def load_config(config_path: str) -> Any:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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


def load_datasets(cfg: Config) -> Tuple[QGDatasetBase, QGDatasetBase, QGDatasetBase]:
    """
    Loads the training, validation, and test datasets based on the provided config.
    """
    base_data_dir = Path(cfg.data.data_dir)
    try:
        DatasetClass = get_dataset_class(cfg.data.dataset_type)

        # CORRECTED: Instead of using a helper dictionary, we pass all arguments
        # as explicit keywords. This is unambiguous to the mypy type checker.
        train_dataset = DatasetClass(
            data_path=base_data_dir / cfg.data.train_file,
            normalizer=get_normalizer(cfg),
            input_sequence_length=cfg.data.input_sequence_length,
            max_sequence_length=cfg.data.max_sequence_length,
            variables=cfg.data.variables,
        )
        val_dataset = DatasetClass(
            data_path=base_data_dir / cfg.data.val_file,
            normalizer=get_normalizer(cfg),
            input_sequence_length=cfg.data.input_sequence_length,
            max_sequence_length=cfg.data.max_sequence_length,
            variables=cfg.data.variables,
        )
        test_dataset = DatasetClass(
            data_path=base_data_dir / cfg.data.test_file,
            normalizer=get_normalizer(cfg),
            input_sequence_length=cfg.data.input_sequence_length,
            max_sequence_length=cfg.data.max_sequence_length,
            variables=cfg.data.variables,
        )

        logger.info(
            f"Successfully loaded datasets with type '{cfg.data.dataset_type}'."
        )
        return train_dataset, val_dataset, test_dataset
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to load datasets: {e}", exc_info=True)
        raise


def load_checkpoint(
    checkpoint_path: str, model: nn.Module, optimizer: Optimizer
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
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint.get("epoch", -1) + 1
        history = checkpoint.get("history", {})

        logger.info(
            f"Checkpoint loaded from {cp_path}. Resuming from epoch {start_epoch}."
        )
        return model, optimizer, history, start_epoch
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {cp_path}: {e}", exc_info=True)
        raise RuntimeError("Critical error loading checkpoint.") from e
