from tensordict import TensorDict
import torch
from torch import nn
from torch import Tensor
from models.dataloader import QGDatasetBase, QGDatasetQuantile, MultipleSims
from pathlib import Path
import torch.optim as optim
import yaml
from typing import Dict, Any, Type, Tuple
import logging
from models.config_classes import Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def tensor_dict_to_json(tensor_dict: TensorDict):
    """
    Convert a TensorDict or Tensor to a JSON-serializable dictionary or list.

    Args:
        tensor_dict (TensorDict or torch.Tensor): Input TensorDict or tensor.

    Returns:
        dict or list or scalar: JSON-serializable dictionary, list, or scalar.
    """
    if isinstance(tensor_dict, Tensor):
        # Handle tensors: return as a Python scalar if it's a single value, otherwise convert to a list
        return (
            tensor_dict.item()
            if tensor_dict.numel() == 1
            else tensor_dict.cpu().numpy().tolist()
        )
    elif isinstance(tensor_dict, TensorDict):
        # Handle TensorDict: recursively convert each item to JSON-serializable format
        return {key: tensor_dict_to_json(value) for key, value in tensor_dict.items()}
    else:
        raise TypeError(
            f"Unsupported type for tensor_dict_to_json: {type(tensor_dict)}"
        )


def accumulate_losses(total_losses: dict, losses: dict) -> dict:
    """
    Accumulates losses over batches.

    Args:
        total_losses: TensorDict to store accumulated losses.
        losses: Current batch losses as a TensorDict.

    Returns:
        Updated total_losses TensorDict.
    """
    for key, value in losses.items():
        if key not in total_losses:
            total_losses[key] = value
        else:
            total_losses[key] += value
    return total_losses


def average_losses(total_losses: dict, n_batches: int) -> dict:
    """
    Averages the losses over the number of batches.

    Args:
        total_losses: TensorDict with accumulated losses.
        n_batches: Total number of batches.

    Returns:
        TensorDict with averaged losses.
    """
    for key in total_losses.keys():
        total_losses[key] /= n_batches
    return total_losses


def load_config(config_path: str) -> Any:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_class_and_kwargs(
    cfg: Config,
) -> Tuple[Type[QGDatasetBase], Dict[str, Any]]:
    """
    Determines the dataset class and its specific kwargs based on Omegaconf config.
    """
    dataset_type = cfg.data.dataset_type
    dataset_kwargs: Dict[str, Any] = {
        "input_sequence_length": cfg.data.input_sequence_length,
        "max_sequence_length": cfg.data.max_sequence_length,
        "variables": cfg.data.variables,
    }

    if dataset_type == "QGDatasetBase":
        return QGDatasetBase, dataset_kwargs
    elif dataset_type == "QGDatasetQuantile":
        dataset_kwargs["quantile_range"] = cfg.data.quantile_range
        return QGDatasetQuantile, dataset_kwargs
    elif dataset_type == "MultipleSims":
        return MultipleSims, dataset_kwargs
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


# In models/utils.py, inside the load_datasets function


def load_datasets(
    cfg: Config, dataset_class: Type[QGDatasetBase], dataset_kwargs: Dict[str, Any]
) -> Tuple[QGDatasetBase, QGDatasetBase, QGDatasetBase]:
    base_data_dir = Path(cfg.data.data_dir)

    train_data_path = base_data_dir / cfg.data.train_file
    val_data_path = base_data_dir / cfg.data.val_file
    test_data_path = base_data_dir / cfg.data.test_file

    if not train_data_path.is_file():
        raise FileNotFoundError(f"Train data file not found: {train_data_path}")
    if not val_data_path.is_file():
        raise FileNotFoundError(f"Validation data file not found: {val_data_path}")
    if not test_data_path.is_file():
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")

    try:
        # Change this line: Pass data_path as the first positional argument
        train_dataset = dataset_class(str(train_data_path), **dataset_kwargs)
        val_dataset = dataset_class(str(val_data_path), **dataset_kwargs)
        test_dataset = dataset_class(str(test_data_path), **dataset_kwargs)

        logger.info(f"Train dataset loaded from: {train_data_path}")
        logger.info(f"Validation dataset loaded from: {val_data_path}")
        logger.info(f"Test dataset loaded from: {test_data_path}")

        return train_dataset, val_dataset, test_dataset
    except Exception as e:
        logger.error(f"Failed to instantiate datasets. Error: {e}", exc_info=True)
        raise RuntimeError(f"Dataset instantiation failed: {e}")


def load_checkpoint(
    checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer
) -> Tuple[nn.Module, optim.Optimizer, Dict[str, Any], int]:
    """
    Loads a model and optimizer state from a checkpoint.
    Returns model, optimizer, history, and start_epoch.
    """
    if not Path(checkpoint_path).is_file():
        logger.warning(
            f"Checkpoint file not found at: {checkpoint_path}. Starting training from scratch."
        )
        return model, optimizer, {}, 0  # Return initial state if checkpoint not found

    try:
        # Load to CPU first to prevent CUDA memory issues if loading on different GPU setups
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1  # Start from the next epoch
        history = checkpoint.get("history", {})

        logger.info(
            f"Checkpoint loaded successfully from {checkpoint_path}. Resuming from epoch {start_epoch}."
        )
        return model, optimizer, history, start_epoch
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        # Decide if this should be a fatal error or fallback to training from scratch
        # For critical checkpoint loading failures, it's safer to raise
        raise RuntimeError(f"Critical error loading checkpoint: {e}")
