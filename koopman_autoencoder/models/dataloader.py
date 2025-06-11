# models/dataloader.py
import torch
import xarray as xr
import numpy as np
import random
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple
from abc import ABC, abstractmethod
from omegaconf import DictConfig

from tensordict import TensorDict, stack as stack_tensordict
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


class DatasetConfigurationError(Exception):
    """Custom exception for dataset configuration errors."""

    pass


# --- Normalizer Classes ---
class AbstractNormalizer(ABC):
    """Abstract base class for data normalizers."""

    def __init__(self):
        self.normalized_vars: List[str] = []

    @abstractmethod
    def fit(self, data: TensorDict):
        pass

    @abstractmethod
    def transform(self, data: TensorDict) -> TensorDict:
        pass

    @abstractmethod
    def inverse_transform(self, data: TensorDict) -> TensorDict:
        pass


class MeanStdNormalizer(AbstractNormalizer):
    """Normalizes data using Z-score (mean/standard deviation)."""

    EPSILON = 1e-8

    def __init__(self):
        super().__init__()
        self.means: Optional[TensorDict] = None
        self.stds: Optional[TensorDict] = None

    def fit(self, data: TensorDict):
        self.normalized_vars = list(data.keys())
        self.means = TensorDict(
            {key: torch.mean(tensor).float() for key, tensor in data.items()},
            batch_size=[],
        )
        self.stds = TensorDict(
            {key: torch.std(tensor).float() for key, tensor in data.items()},
            batch_size=[],
        )
        logger.info("Fitted MeanStdNormalizer.")

    def transform(self, data: TensorDict) -> TensorDict:
        if self.means is None:
            raise DatasetConfigurationError("Normalizer has not been fitted.")

        def norm_fn(d: torch.Tensor, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return (d - m) / (s + self.EPSILON)

        data_to_transform = data.select(*self.normalized_vars, strict=False)
        transformed_data = data_to_transform.apply(norm_fn, self.means, self.stds)
        data.update(transformed_data)
        return data

    def inverse_transform(self, data: TensorDict) -> TensorDict:
        if self.means is None:
            raise DatasetConfigurationError(
                "Normalizer has not been fitted (means is None)."
            )
        if self.stds is None:
            raise DatasetConfigurationError(
                "Normalizer has not been fitted (stds is None)."
            )

        denormalized = {}
        for key, tensor in data.items():
            if key in self.normalized_vars:
                mean = self.means[key].to(tensor.device)
                std = self.stds[key].to(tensor.device)
                denormalized[key] = tensor * std + mean
            else:
                denormalized[key] = tensor
        return TensorDict(denormalized, batch_size=data.batch_size)


class QuantileNormalizer(AbstractNormalizer):
    """Normalizes data to the range [-1, 1] using specified quantiles."""

    EPSILON = 1e-8

    def __init__(self, quantile_range: Tuple[float, float] = (2.5, 97.5)):
        super().__init__()
        if not (0 <= quantile_range[0] < quantile_range[1] <= 100):
            raise ValueError("Quantile range must be valid.")
        self.quantile_range = quantile_range
        self.q_lows: Dict[str, torch.Tensor] = {}
        self.q_highs: Dict[str, torch.Tensor] = {}

    def fit(self, data: TensorDict):
        self.normalized_vars = list(data.keys())
        q_lows: Dict[str, torch.Tensor] = {}
        q_highs: Dict[str, torch.Tensor] = {}
        for key, tensor in data.items():
            flat = tensor.numpy().flatten()
            low, high = np.percentile(flat, self.quantile_range)
            if np.isclose(high, low, atol=self.EPSILON):
                logger.warning(f"Quantile range for '{key}' is near zero.")
            q_lows[key] = torch.tensor([low], dtype=torch.float32)
            q_highs[key] = torch.tensor([high], dtype=torch.float32)
        self.q_lows = TensorDict(q_lows, [])
        self.q_highs = TensorDict(q_highs, [])
        logger.info("Fitted QuantileNormalizer.")

    def transform(self, data: TensorDict) -> TensorDict:
        if self.q_lows is None:
            raise DatasetConfigurationError(
                "Normalizer has not been fitted (q_lows is None)."
            )
        if self.q_highs is None:
            raise DatasetConfigurationError(
                "Normalizer has not been fitted (q_highs is None)."
            )

        def norm_fn(
            d: torch.Tensor, low: torch.Tensor, high: torch.Tensor
        ) -> torch.Tensor:
            return 2 * (d - low) / (high - low + self.EPSILON) - 1

        data_to_transform = data.select(*self.normalized_vars, strict=False)
        transformed_data = data_to_transform.apply(norm_fn, self.q_lows, self.q_highs)
        data.update(transformed_data)
        return data

    def inverse_transform(self, data: TensorDict) -> TensorDict:
        if self.q_lows is None:
            raise DatasetConfigurationError(
                "Normalizer has not been fitted (q_lows is None)."
            )
        if self.q_highs is None:
            raise DatasetConfigurationError(
                "Normalizer has not been fitted (q_highs is None)."
            )

        # After these checks, mypy knows both attributes are valid TensorDicts.
        denormalized = {}
        for key, tensor in data.items():
            if key in self.normalized_vars:
                low = self.q_lows[key].to(tensor.device)
                high = self.q_highs[key].to(tensor.device)
                denormalized[key] = ((tensor + 1) / 2) * (high - low) + low
            else:
                denormalized[key] = tensor
        return TensorDict(denormalized, batch_size=data.batch_size)


# --- Base Dataset Class ---
class QGDatasetBase(Dataset):
    """Base class for handling quasi-geostrophic simulation data."""

    def __init__(
        self,
        data_path: Union[str, Path],
        normalizer: AbstractNormalizer,
        input_sequence_length: int,
        max_sequence_length: int,
        variables: Dict[str, int],
    ):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        self.input_sequence_length = input_sequence_length
        self.max_sequence_length = max_sequence_length
        self.normalizer = normalizer
        self.data_vars = list(variables.keys())

        self._load_data()
        self._prepare_data()

    def _load_data(self):
        """Loads data from NetCDF, validates variables, and infers batch dimensions."""
        with xr.open_dataset(self.data_path) as ds:
            self._data = ds
            missing_vars = [v for v in self.data_vars if v not in ds.data_vars]
            if missing_vars:
                raise DatasetConfigurationError(
                    f"Vars {missing_vars} not in {self.data_path}"
                )

            td_tensors = {
                var: torch.from_numpy(ds[var].values).float() for var in self.data_vars
            }

            sample_var_name = self.normalizer_vars[0]
            sample_tensor_shape = td_tensors[sample_var_name].shape

            num_feature_dims = 2
            num_batch_dims = len(sample_tensor_shape) - num_feature_dims
            batch_size = list(sample_tensor_shape[:num_batch_dims])
            self.raw_data_td = TensorDict(td_tensors, batch_size=batch_size)

            self.mins = TensorDict(
                {key: torch.min(tensor) for key, tensor in self.raw_data_td.items()},
                batch_size=[],
            )
            self.maxs = TensorDict(
                {key: torch.max(tensor) for key, tensor in self.raw_data_td.items()},
                batch_size=[],
            )

    def _prepare_data(self):
        """Fits the normalizer and transforms the data."""
        self.normalizer.fit(self.raw_data_td.select(*self.normalizer_vars))
        self.stacked_data = self.normalizer.transform(self.raw_data_td)

    @property
    def normalizer_vars(self) -> List[str]:
        return self.data_vars

    def __len__(self) -> int:
        time_dim_index = self.raw_data_td.batch_dims - 1
        num_timesteps = self.raw_data_td.batch_size[time_dim_index]
        total_len = (
            num_timesteps - self.input_sequence_length - self.max_sequence_length + 1
        )
        return int(max(0, total_len))

    def __getitem__(self, idx: Union[int, Tuple[int, int]]):
        start_idx, target_length = (
            (idx, self.max_sequence_length) if isinstance(idx, int) else idx
        )
        if not (0 <= target_length <= self.max_sequence_length):
            raise ValueError(
                f"target_length must be in [0, {self.max_sequence_length}]"
            )

        input_end = start_idx + self.input_sequence_length
        target_end = input_end + target_length

        input_seq = self.stacked_data[..., start_idx:input_end, :, :]
        target_seq = self.stacked_data[..., input_end:target_end, :, :]

        # Create a metadata dictionary. The custom collate function will handle this.
        metadata = {"seq_length": target_length}

        # Return a 3-tuple that the custom_collate_fn expects
        return input_seq, target_seq, metadata

    def denormalize(self, data: TensorDict) -> TensorDict:
        return self.normalizer.inverse_transform(data)

    def to_unit_range(self, data: TensorDict) -> TensorDict:
        scaled = {}
        for var, tensor in data.items():
            if var in self.normalizer.normalized_vars:
                min_val, max_val = self.mins[var].to(tensor.device), self.maxs[var].to(
                    tensor.device
                )
                scaled[var] = (tensor - min_val) / (max_val - min_val + 1e-8)
            else:
                scaled[var] = tensor
        return TensorDict(scaled, batch_size=data.batch_size)


# --- Multi-Simulation Dataset ---
class QGDatasetMultiSim(QGDatasetBase):
    """Dataset for multiple simulations, handling indexing and metadata."""

    def __init__(self, *args, **kwargs):
        self.num_sims: int = 0
        self.Re: Optional[torch.Tensor] = None
        self.obstacle_mask: Optional[torch.Tensor] = None
        self.master_index: List[Tuple[int, int]] = []
        super().__init__(*args, **kwargs)

    def _load_data(self):
        super()._load_data()
        if "sim" not in self._data.dims:
            raise DatasetConfigurationError("Expected 'sim' dimension in the dataset.")
        self.num_sims = self._data.sizes["sim"]
        if "Re" in self._data:
            self.Re = torch.from_numpy(self._data["Re"].values).float()
        if "obstacle_mask" in self.data_vars:
            self.obstacle_mask = self.raw_data_td.get("obstacle_mask")

    @property
    def normalizer_vars(self) -> List[str]:
        return [v for v in self.data_vars if v != "obstacle_mask"]

    def _prepare_data(self):
        self._compute_master_index()
        super()._prepare_data()

    def _compute_master_index(self):
        self.master_index = []
        num_timesteps = self._data.sizes["t"]
        valid_starts = (
            num_timesteps - self.input_sequence_length - self.max_sequence_length + 1
        )
        if valid_starts > 0:
            for sim_idx in range(self.num_sims):
                self.master_index.extend([(sim_idx, i) for i in range(valid_starts)])
        if not self.master_index:
            logger.warning("No valid sequences generated from dataset.")

    def __len__(self) -> int:
        return len(self.master_index)

    def __getitem__(self, idx: Union[int, Tuple[int, int]]):
        flat_idx, target_length = (
            (idx, self.max_sequence_length) if isinstance(idx, int) else idx
        )
        if not (0 <= flat_idx < len(self)):
            raise IndexError("Index out of bounds.")

        sim_idx, start_idx = self.master_index[flat_idx]

        input_end = start_idx + self.input_sequence_length
        target_end = input_end + target_length

        input_seq = self.stacked_data[sim_idx, start_idx:input_end]
        target_seq = self.stacked_data[sim_idx, input_end:target_end]

        # Create a metadata dictionary to be handled by the custom collate function.
        metadata = {"seq_length": target_length}
        if self.Re is not None:
            metadata["Re"] = self.Re[sim_idx]

        if self.obstacle_mask is not None:
            mask = (
                self.obstacle_mask
                if self.obstacle_mask.ndim == 2
                else self.obstacle_mask[sim_idx]
            )
            # Add a time dimension to the mask to make it broadcastable with the input sequence.
            input_seq.set("obstacle_mask", mask.unsqueeze(0), inplace=True)

        return input_seq, target_seq, metadata


# --- Overfitting Dataset ---
class SingleSimOverfit(QGDatasetMultiSim):
    """Specialized dataset using only the first simulation for overfitting."""

    def _compute_master_index(self):
        super()._compute_master_index()
        self.master_index = [item for item in self.master_index if item[0] == 0]
        logger.info(f"SingleSimOverfit: Using sim 0. Samples: {len(self.master_index)}")


# --- Samplers ---
class RandomLengthBatchSampler(Sampler[List[Tuple[int, int]]]):
    def __init__(
        self, dataset_len, batch_size, max_seq_len, shuffle=True, drop_last=False
    ):
        self.dataset_len, self.batch_size, self.max_seq_len = (
            dataset_len,
            batch_size,
            max_seq_len,
        )
        self.shuffle, self.drop_last = shuffle, drop_last

    def __iter__(self):
        indices = list(range(self.dataset_len))
        if self.shuffle:
            random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
            target_len = random.randint(1, self.max_seq_len)
            yield [(idx, target_len) for idx in batch_indices]

    def __len__(self):
        return (
            self.dataset_len // self.batch_size
            if self.drop_last
            else (self.dataset_len + self.batch_size - 1) // self.batch_size
        )


class FixedLengthBatchSampler(Sampler[List[Tuple[int, int]]]):
    def __init__(
        self, dataset_len, batch_size, seq_len, shuffle=False, drop_last=False
    ):
        self.dataset_len, self.batch_size, self.seq_len = (
            dataset_len,
            batch_size,
            seq_len,
        )
        self.shuffle, self.drop_last = shuffle, drop_last

    def __iter__(self):
        indices = list(range(self.dataset_len))
        if self.shuffle:
            random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
            yield [(idx, self.seq_len) for idx in batch_indices]

    def __len__(self):
        if self.drop_last:
            return self.dataset_len // self.batch_size
        return (self.dataset_len + self.batch_size - 1) // self.batch_size


# --- DataLoader Setup ---
class DataLoaderWrapper(DataLoader):
    def __init__(self, dataset: QGDatasetBase, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.dataset: QGDatasetBase

    def denormalize(self, x: TensorDict) -> TensorDict:
        return self.dataset.denormalize(x)

    def to_unit_range(self, x: TensorDict) -> TensorDict:
        return self.dataset.to_unit_range(x)


def custom_collate_fn(batch: List[Tuple[TensorDict, TensorDict, Dict]]):
    """
    Collates data and metadata into final batched TensorDicts.
    This function now correctly handles the metadata dictionary.
    """
    input_tds, target_tds, meta_dicts = zip(*batch)

    # Stack the TensorDicts for data variables
    batched_inputs = stack_tensordict(input_tds)
    batched_targets = stack_tensordict(target_tds)

    # Collate the metadata and add it to the final batched TensorDicts
    for key in meta_dicts[0].keys():
        if isinstance(meta_dicts[0][key], torch.Tensor):
            meta_tensor = torch.stack([d[key] for d in meta_dicts])
        else:
            meta_tensor = torch.tensor([d[key] for d in meta_dicts])

        ####################################
        ### THIS IS THE CORRECTED SECTION ###
        ####################################

        # Get the target batch dimensions (e.g., [1, 10])
        target_batch_size = batched_targets.batch_size

        # Reshape the metadata tensor from [batch_size] to [batch_size, 1]
        meta_tensor_reshaped = meta_tensor.unsqueeze(1)

        # Explicitly expand the metadata tensor to match the target's batch dimensions.
        # This changes the shape from [batch_size, 1] to [batch_size, seq_len].
        meta_tensor_expanded = meta_tensor_reshaped.expand(*target_batch_size)

        # Set the expanded tensor. The shapes now match exactly.
        batched_targets.set(key, meta_tensor_expanded)

    return batched_inputs, batched_targets


def create_dataloaders(
    train_dataset: QGDatasetBase,
    val_dataset: QGDatasetBase,
    test_dataset: QGDatasetBase,
    training_cfg: DictConfig,
):
    bs = training_cfg.batch_size
    if training_cfg.random_sequence_length:
        train_sampler = RandomLengthBatchSampler(
            len(train_dataset),
            bs,
            train_dataset.max_sequence_length,
            shuffle=True,
            drop_last=True,
        )
    else:
        train_sampler = FixedLengthBatchSampler(
            len(train_dataset),
            bs,
            train_dataset.max_sequence_length,
            shuffle=True,
            drop_last=True,
        )
    val_sampler = FixedLengthBatchSampler(
        len(val_dataset), bs, val_dataset.max_sequence_length, shuffle=False
    )
    test_sampler = FixedLengthBatchSampler(
        len(test_dataset), bs, test_dataset.max_sequence_length, shuffle=False
    )

    train_loader = DataLoaderWrapper(
        train_dataset, batch_sampler=train_sampler, collate_fn=custom_collate_fn
    )
    val_loader = DataLoaderWrapper(
        val_dataset, batch_sampler=val_sampler, collate_fn=custom_collate_fn
    )
    test_loader = DataLoaderWrapper(
        test_dataset, batch_sampler=test_sampler, collate_fn=custom_collate_fn
    )
    return train_loader, val_loader, test_loader


def create_ddp_dataloaders(
    train_dataset: QGDatasetBase,
    val_dataset: QGDatasetBase,
    test_dataset: QGDatasetBase,
    training_cfg: DictConfig,
    rank: int,
    world_size: int,
):
    bs = training_cfg.batch_size
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoaderWrapper(
        train_dataset,
        batch_size=bs,
        sampler=train_sampler,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoaderWrapper(
        val_dataset, batch_size=bs, sampler=val_sampler, collate_fn=custom_collate_fn
    )
    test_loader = DataLoaderWrapper(
        test_dataset, batch_size=bs, sampler=test_sampler, collate_fn=custom_collate_fn
    )
    return train_loader, val_loader, test_loader
