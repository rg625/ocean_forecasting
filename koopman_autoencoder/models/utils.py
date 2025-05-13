from tensordict import TensorDict
import torch
from torch import Tensor
from models.autoencoder import KoopmanAutoencoder
from models.dataloader import QGDatasetBase, QGDatasetQuantile, DiffusionReaction
from pathlib import Path
from torch.optim import Optimizer
import yaml
from typing import Any


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


def accumulate_losses(total_losses: dict, losses: dict):
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


def average_losses(total_losses: dict, n_batches: int):
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


def load_checkpoint(
    checkpoint_path: str, model: KoopmanAutoencoder, optimizer: Optimizer
):
    """
    Loads training state from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    history = checkpoint.get("history", {})
    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"Resuming training from epoch {start_epoch}.")
    return model, optimizer, history, start_epoch


def load_config(config_path: str) -> Any:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_class_and_kwargs(config: dict):
    norm_cfg = config.get("data", {}).get("normalization", {})
    norm_type = norm_cfg.get("type", None)

    if norm_type == "quantile":
        return QGDatasetQuantile, {
            "quantile_range": norm_cfg.get("quantiles", (2.5, 97.5))
        }
    elif norm_type == "diff":
        return (
            DiffusionReaction,
            {},
        )  # , {"max_samples": norm_cfg.get("max_samples", 90)}
    return QGDatasetBase, {}


def load_datasets(config: dict, dataset_class, kwargs: dict):
    data_dir = Path(config["data"]["data_dir"])
    return [
        dataset_class(
            data_dir / config["data"][split + "_file"],
            max_sequence_length=config["data"]["max_sequence_length"],
            **kwargs,
        )
        for split in ["train", "val", "test"]
    ]
