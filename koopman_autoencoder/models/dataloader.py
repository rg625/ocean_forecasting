import torch
from tensordict import TensorDict
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import xarray as xr
import random
import numpy as np
from tensordict import stack as stack_tensordict
from typing import Optional, List, Any, Union
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define a small constant for numerical stability
EPS = 1e-8


class QGDatasetBase(Dataset):
    """
    Base Dataset class for Quasi-Geostrophic (QG) model data.
    Handles loading, basic validation, and Z-score normalization.
    """

    def __init__(
        self,
        data_path: str,
        input_sequence_length: int = 2,
        max_sequence_length: int = 2,
        variables: Optional[List[str]] = None,
    ):
        """
        Initializes the QGDatasetBase.

        Args:
            data_path (str): Path to the xarray dataset file (e.g., .nc).
            input_sequence_length (int): The length of the input sequence for the model.
            max_sequence_length (int): The maximum length of the target sequence to predict.
            variables (Optional[List[str]]): List of variables to load from the dataset.
                                             If None, all data variables will be used.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        try:
            self.data = xr.open_dataset(data_path)
        except Exception as e:
            raise IOError(f"Failed to open xarray dataset from {data_path}: {e}")

        if not isinstance(input_sequence_length, int) or input_sequence_length <= 0:
            raise ValueError("input_sequence_length must be a positive integer.")
        if not isinstance(max_sequence_length, int) or max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be a positive integer.")
        if input_sequence_length + max_sequence_length > len(
            self.data.sizes.get("t", [])
        ):
            logger.warning(
                f"Combined sequence length ({input_sequence_length + max_sequence_length}) "
                f"exceeds available time steps ({len(self.data.sizes.get('t', []))}). "
                "This might lead to an empty dataset or errors."
            )

        self.input_sequence_length = input_sequence_length
        self.max_sequence_length = max_sequence_length

        if variables is None:
            self.variables = list(self.data.data_vars.keys())
        else:
            invalid_vars = [var for var in variables if var not in self.data.data_vars]
            if invalid_vars:
                raise ValueError(f"Variables not found in dataset: {invalid_vars}")
            self.variables = variables

        if not self.variables:
            raise ValueError("No variables selected or found in the dataset.")

        self._normalize()  # perform normalization

    def _normalize(self):
        """
        Performs Z-score normalization (mean 0, std 1) on the selected variables.
        Stores means, standard deviations, min, and max values.
        Handles constant data to prevent division by zero.
        """
        self.means = TensorDict(
            {
                var: torch.tensor(np.mean(self.data[var].values), dtype=torch.float32)
                for var in self.variables
            },
            batch_size=[],
        )
        self.stds = TensorDict(
            {
                var: torch.tensor(np.std(self.data[var].values), dtype=torch.float32)
                for var in self.variables
            },
            batch_size=[],
        )

        self.stacked_data = TensorDict({}, batch_size=[])
        for var in self.variables:
            data_tensor = torch.tensor(self.data[var].values, dtype=torch.float32)
            mean = self.means[var]
            std = self.stds[var]

            if std < EPS:  # Handle constant data
                logger.warning(
                    f"Variable '{var}' has a standard deviation close to zero. Normalizing to zero."
                )
                self.stacked_data[var] = torch.zeros_like(data_tensor)
            else:
                self.stacked_data[var] = (data_tensor - mean) / (std + EPS)

        self.mins = TensorDict(
            {
                var: torch.tensor(np.min(self.data[var].values), dtype=torch.float32)
                for var in self.variables
            },
            batch_size=[],
        )
        self.maxs = TensorDict(
            {
                var: torch.tensor(np.max(self.data[var].values), dtype=torch.float32)
                for var in self.variables
            },
            batch_size=[],
        )

    def __len__(self):
        """
        Returns the total number of possible input-target sequence pairs.
        """
        if not self.stacked_data:
            return 0

        # Assume all variables have the same time dimension length
        time_dim_length = next(iter(self.stacked_data.values())).shape[0]

        # Ensure there's enough data for at least one sequence
        if time_dim_length < self.input_sequence_length + self.max_sequence_length:
            return 0

        return (
            time_dim_length - self.input_sequence_length - self.max_sequence_length + 1
        )

    def __getitem__(self, idx: Union[int, tuple]):
        """
        Retrieves an input-target sequence pair.

        Args:
            idx (Union[int, tuple]): The index of the starting point for the sequence.
                                     If a tuple (start_idx, target_length), it allows for
                                     variable target sequence lengths.

        Returns:
            tuple[TensorDict, TensorDict]: A tuple containing the input TensorDict
                                          and the target TensorDict.
        """
        if isinstance(idx, tuple):
            start_idx, target_length = idx
        else:
            start_idx = idx
            target_length = self.max_sequence_length

        # Basic bounds checking for start_idx and target_length
        if not (0 <= start_idx < len(self)):
            raise IndexError(
                f"Index {start_idx} out of bounds for dataset length {len(self)}."
            )

        # Calculate available time steps for target sequence
        current_time_dim_length = next(iter(self.stacked_data.values())).shape[0]
        max_possible_target_length = (
            current_time_dim_length - start_idx - self.input_sequence_length
        )

        if target_length > max_possible_target_length:
            logger.warning(
                f"Requested target_length ({target_length}) at start_idx {start_idx} "
                f"exceeds available data ({max_possible_target_length}). "
                "Adjusting target_length to maximum possible."
            )
            target_length = max_possible_target_length

        if target_length <= 0:
            raise ValueError(
                f"Calculated target_length is {target_length}. Cannot create empty target sequence. "
                "Check input/max sequence lengths relative to data."
            )

        input_seq = TensorDict(
            {
                var: self.stacked_data[var][
                    start_idx : start_idx + self.input_sequence_length
                ]
                for var in self.variables
            },
            batch_size=[],
        )
        target_seq = TensorDict(
            {
                var: self.stacked_data[var][
                    start_idx
                    + self.input_sequence_length : start_idx
                    + self.input_sequence_length
                    + target_length
                ]
                for var in self.variables
            },
            batch_size=[],
        )
        target_seq["seq_length"] = torch.tensor(target_length, dtype=torch.int64)
        return input_seq, target_seq

    def denormalize(self, x: TensorDict) -> TensorDict:
        """
        Denormalizes a TensorDict from Z-score (mean 0, std 1) back to original scale.

        Args:
            x (TensorDict): A TensorDict containing normalized tensors.

        Returns:
            TensorDict: Denormalized version of the input TensorDict.
        """
        denormalized_data = {}
        for var, tensor in x.items():
            if var == "seq_length":
                denormalized_data[var] = tensor  # Pass through non-data variables
                continue

            if var not in self.means or var not in self.stds:
                logger.warning(
                    f"Denormalization: Variable '{var}' not found in stored means/stds. Passing through."
                )
                denormalized_data[var] = tensor
                continue

            device = tensor.device
            means = self.means[var].to(device)
            stds = self.stds[var].to(device)

            if stds.item() < EPS:  # Handle constant data during denormalization
                denormalized_data[var] = means.expand_as(
                    tensor
                )  # All values become the mean
            else:
                denormalized_data[var] = tensor * stds + means
        return TensorDict(denormalized_data, batch_size=x.batch_size)


class QGDatasetQuantile(QGDatasetBase):
    """
    Dataset that normalizes using specified quantiles instead of mean/std.
    Normalizes data to [-1, 1] using the quantile range.
    """

    def __init__(
        self,
        data_path: str,
        input_sequence_length: int = 2,
        max_sequence_length: int = 2,
        variables: Optional[List[str]] = None,
        quantile_range: tuple = (2.5, 97.5),
    ):
        """
        Initializes the QGDatasetQuantile.

        Args:
            data_path (str): Path to the xarray dataset file (e.g., .nc).
            input_sequence_length (int): The length of the input sequence for the model.
            max_sequence_length (int): The maximum length of the target sequence to predict.
            variables (Optional[List[str]]): List of variables to load from the dataset.
                                             If None, all data variables will be used.
            quantile_range (tuple): A tuple (low_percentile, high_percentile) for normalization.
                                    e.g., (2.5, 97.5) will normalize data between the 2.5th and 97.5th percentiles.
        """
        if not isinstance(quantile_range, tuple) or len(quantile_range) != 2:
            raise ValueError(
                "quantile_range must be a tuple of two numbers (low, high)."
            )
        if not (0 <= quantile_range[0] < quantile_range[1] <= 100):
            raise ValueError(
                "Quantile range values must be between 0 and 100, and low < high."
            )

        self.quantile_range = quantile_range
        self.q_lows: Optional[TensorDict] = None
        self.q_highs: Optional[TensorDict] = None
        super().__init__(
            data_path, input_sequence_length, max_sequence_length, variables
        )

    def _normalize(self):
        """
        Overrides base class normalization to use quantile normalization.
        Normalizes each variable to the range [-1, 1] based on its specified quantile range.
        Handles constant data to prevent division by zero.
        """
        raw_data_td = TensorDict(
            {var: torch.FloatTensor(self.data[var].values) for var in self.variables},
            batch_size=[],
        )

        self.stacked_data, self.q_lows, self.q_highs = self._quantile_normalize_static(
            raw_data_td, quantile_range=self.quantile_range
        )

        # Min/Max are still useful for to_unit_range, even if not used for normalization here
        self.mins = TensorDict(
            {
                var: torch.tensor(np.min(self.data[var].values), dtype=torch.float32)
                for var in self.variables
            },
            batch_size=[],
        )
        self.maxs = TensorDict(
            {
                var: torch.tensor(np.max(self.data[var].values), dtype=torch.float32)
                for var in self.variables
            },
            batch_size=[],
        )

    @staticmethod
    def _quantile_normalize_static(tensor_dict: TensorDict, quantile_range=(2.5, 97.5)):
        """
        Static method to normalize each variable in the tensor_dict to [-1, 1]
        using the specified quantile range.
        Handles constant data to prevent division by zero.
        """
        q_lows = {}
        q_highs = {}
        normalized = {}

        for var, tensor in tensor_dict.items():
            data_flat = tensor.numpy().flatten()
            q_low = np.percentile(data_flat, quantile_range[0])
            q_high = np.percentile(data_flat, quantile_range[1])

            q_lows[var] = torch.tensor([q_low], dtype=torch.float32)
            q_highs[var] = torch.tensor([q_high], dtype=torch.float32)

            diff = q_high - q_low
            if diff < EPS:  # Handle constant data: normalize to 0
                logger.warning(
                    f"Quantile normalization: Variable '{var}' has a quantile range close to zero. Normalizing to zero."
                )
                normalized[var] = torch.zeros_like(tensor)
            else:
                norm = 2 * (tensor - q_low) / (diff + EPS) - 1
                normalized[var] = norm

        return (
            TensorDict(normalized, batch_size=[]),
            TensorDict(q_lows, batch_size=[]),
            TensorDict(q_highs, batch_size=[]),
        )

    def denormalize(self, tensor_dict: TensorDict) -> TensorDict:
        """
        Denormalize data from [-1, 1] back to the original range using stored quantiles.

        Args:
            tensor_dict (TensorDict): Dictionary with normalized tensors.

        Returns:
            TensorDict: Denormalized version.
        """
        if self.q_lows is None or self.q_highs is None:
            raise AttributeError(
                "Quantile lows/highs are not set. Normalization might not have occurred correctly."
            )

        denormalized_data = {}
        for var, tensor in tensor_dict.items():
            if var == "seq_length":
                denormalized_data[var] = tensor  # Pass through non-data variables
                continue

            if var not in self.q_lows or var not in self.q_highs:
                logger.warning(
                    f"Denormalization: Variable '{var}' not found in stored quantiles. Passing through."
                )
                denormalized_data[var] = tensor
                continue

            q_low = self.q_lows[var].to(tensor.device)
            q_high = self.q_highs[var].to(tensor.device)

            diff = q_high - q_low
            if diff < EPS:  # Handle constant data during denormalization
                denormalized_data[var] = q_low.expand_as(
                    tensor
                )  # All values become the q_low (which is q_high)
            else:
                denorm = ((tensor + 1) / 2) * (diff + EPS) + q_low
                denormalized_data[var] = denorm

        return TensorDict(denormalized_data, batch_size=tensor_dict.batch_size)


class MultipleSims(QGDatasetBase):
    """
    Dataset class for handling multiple simulations within a single xarray dataset.
    Assumes a 'sim' dimension for different simulations and 't' for time.
    Adds 'Re' and 'obstacle_mask' to the output if present.
    """

    def __init__(
        self,
        data_path: str,
        input_sequence_length: int = 2,
        max_sequence_length: int = 2,
        variables: Optional[List[str]] = None,
    ):
        """
        Initializes the MultipleSims dataset.

        Args:
            data_path (str): Path to the xarray dataset file (e.g., .nc).
            input_sequence_length (int): The length of the input sequence for the model.
            max_sequence_length (int): The maximum length of the target sequence to predict.
            variables (Optional[List[str]]): List of variables to load from the dataset.
                                             If None, all data variables will be used.
        """
        super().__init__(
            data_path, input_sequence_length, max_sequence_length, variables
        )

        if "sim" not in self.data.sizes:
            raise ValueError(
                f"Dataset at {data_path} must have a 'sim' dimension for MultipleSims dataset."
            )
        self.num_sims = self.data.sizes["sim"]

        self.Re: Optional[torch.Tensor] = None
        if "Re" in self.data.variables:
            # Re values are typically per simulation, not time-varying
            if "sim" in self.data["Re"].dims:
                self.Re = torch.tensor(self.data["Re"].values, dtype=torch.float32)
            else:
                logger.warning(
                    "Found 'Re' variable but no 'sim' dimension. It will be treated as a single value."
                )
                self.Re = torch.tensor(
                    [self.data["Re"].values], dtype=torch.float32
                )  # Wrap in a list for consistency
        else:
            logger.info("Variable 'Re' not found in dataset.")

        self.obstacle: Optional[torch.Tensor] = None
        if "obstacle_mask" in self.data.variables:
            # Obstacle mask is typically static across time and simulations
            self.obstacle = torch.tensor(
                self.data["obstacle_mask"].values, dtype=torch.float32
            )
        else:
            logger.info(
                "Variable 'obstacle_mask' not found in dataset. It will not be included."
            )

        logger.info(f"Initialized MultipleSims with {self.num_sims} simulations.")

        # Log means and stds for verification
        logger.debug(f"Means: {self.means.items()}")
        logger.debug(f"Stds: {self.stds.items()}")

    def __len__(self):
        """
        Returns the total number of possible input-target sequence pairs across all simulations.
        Assumes all simulations have the same number of time steps (len(self.data['t'])).
        If simulations have varying time steps, this method would need a more complex
        calculation (e.g., iterating through each sim's length).
        """
        if "t" not in self.data.sizes:
            logger.error(
                "Dataset missing 't' (time) dimension. Cannot calculate dataset length."
            )
            return 0

        time_dim_length = self.data.sizes["t"]

        # Ensure there's enough data for at least one sequence
        if time_dim_length < self.input_sequence_length + self.max_sequence_length:
            return 0

        return self.num_sims * (
            time_dim_length - self.input_sequence_length - self.max_sequence_length + 1
        )

    def __getitem__(self, idx: Union[int, tuple]):
        """
        Retrieves an input-target sequence pair from a specific simulation.

        Args:
            idx (Union[int, tuple]): The global index for the sequence.
                                     If a tuple (global_idx, target_length), it allows for
                                     variable target sequence lengths.

        Returns:
            tuple[TensorDict, TensorDict]: A tuple containing the input TensorDict
                                          and the target TensorDict.
        """
        if isinstance(idx, tuple):
            global_idx, target_length = idx
        else:
            global_idx = idx
            target_length = self.max_sequence_length

        if not (0 <= global_idx < len(self)):
            raise IndexError(
                f"Global index {global_idx} out of bounds for dataset length {len(self)}."
            )

        # Calculate which simulation and index in the simulation to sample from
        # Ensure 't' dimension exists for this calculation
        if "t" not in self.data.sizes:
            raise KeyError(
                "Dataset missing 't' (time) dimension. Cannot index into simulations."
            )

        time_dim_length = self.data.sizes["t"]

        # Number of possible starting points per simulation
        possible_starts_per_sim = (
            time_dim_length - self.input_sequence_length - self.max_sequence_length + 1
        )
        if possible_starts_per_sim <= 0:
            raise ValueError(
                f"Not enough time steps ({time_dim_length}) to form a sequence "
                f"(input: {self.input_sequence_length}, max_target: {self.max_sequence_length})."
            )

        sim_idx = global_idx // possible_starts_per_sim
        start_idx = global_idx % possible_starts_per_sim

        # Basic bounds checking for start_idx and target_length for this specific simulation
        max_possible_target_length = (
            time_dim_length - start_idx - self.input_sequence_length
        )
        if target_length > max_possible_target_length:
            logger.warning(
                f"Requested target_length ({target_length}) at sim {sim_idx}, start_idx {start_idx} "
                f"exceeds available data ({max_possible_target_length}). "
                "Adjusting target_length to maximum possible."
            )
            target_length = max_possible_target_length

        if target_length <= 0:
            raise ValueError(
                f"Calculated target_length is {target_length} for sim {sim_idx}, start_idx {start_idx}. "
                "Cannot create empty target sequence. Check input/max sequence lengths relative to data."
            )

        # Get the input sequence
        input_data_dict = {}
        for var in self.variables:
            # Assuming 'obstacle_mask' is handled separately and not part of the standard variables for normalization
            if var != "obstacle_mask":
                if (
                    "sim" in self.stacked_data[var].dims()
                ):  # Check if it has a sim dimension at first axis
                    input_data_dict[var] = self.stacked_data[var][
                        sim_idx, start_idx : start_idx + self.input_sequence_length
                    ]
                else:  # Data that does not have a sim dimension (e.g. constant fields not in variables list or misconfigured)
                    logger.warning(
                        f"Variable '{var}' does not have a 'sim' dimension. Taking entire slice."
                    )
                    input_data_dict[var] = self.stacked_data[var][
                        start_idx : start_idx + self.input_sequence_length
                    ]

        input_seq = TensorDict(input_data_dict, batch_size=[])

        if self.obstacle is not None:
            input_seq["obstacle_mask"] = (
                self.obstacle.clone()
            )  # Ensure it's a clone if it's mutable and might be changed downstream

        # Get the target sequence (variable length)
        target_data_dict = {}
        for var in self.variables:
            if var != "obstacle_mask":
                if "sim" in self.stacked_data[var].dims():
                    target_data_dict[var] = self.stacked_data[var][
                        sim_idx,
                        start_idx
                        + self.input_sequence_length : start_idx
                        + self.input_sequence_length
                        + target_length,
                    ]
                else:
                    logger.warning(
                        f"Variable '{var}' does not have a 'sim' dimension. Taking entire slice."
                    )
                    target_data_dict[var] = self.stacked_data[var][
                        start_idx
                        + self.input_sequence_length : start_idx
                        + self.input_sequence_length
                        + target_length,
                    ]

        target_seq = TensorDict(target_data_dict, batch_size=[])

        # Add metadata to the target TensorDict
        target_seq["seq_length"] = torch.tensor(target_length, dtype=torch.int64)
        if self.Re is not None:
            if sim_idx < len(self.Re):
                target_seq["Re"] = self.Re[sim_idx].item()
            else:
                logger.warning(
                    f"Re index {sim_idx} out of bounds for self.Re. Skipping Re for this item."
                )
                target_seq["Re"] = torch.tensor(
                    0.0, dtype=torch.float32
                )  # Default value if Re is missing for sim_idx
        else:
            target_seq["Re"] = torch.tensor(
                0.0, dtype=torch.float32
            )  # Default value if Re is globally missing

        return input_seq, target_seq

    def denormalize(self, x: TensorDict) -> TensorDict:
        """
        Denormalizes a TensorDict from Z-score (mean 0, std 1) back to original scale
        for MultipleSims dataset, also passing through 'Re' and 'obstacle_mask'.

        Args:
            x (TensorDict): A TensorDict containing normalized tensors.

        Returns:
            TensorDict: Denormalized version of the input TensorDict.
        """
        denormalized_data = {}
        for var, tensor in x.items():
            if var in ["seq_length", "Re", "obstacle_mask"]:
                denormalized_data[var] = tensor  # Passthrough
                continue

            if var not in self.means or var not in self.stds:
                logger.warning(
                    f"Denormalization: Variable '{var}' not found in stored means/stds. Passing through."
                )
                denormalized_data[var] = tensor
                continue

            device = tensor.device
            means = self.means[var].to(device)
            stds = self.stds[var].to(device)

            if stds.item() < EPS:  # Handle constant data during denormalization
                denormalized_data[var] = means.expand_as(tensor)
            else:
                denormalized_data[var] = tensor * stds + means
        return TensorDict(denormalized_data, batch_size=x.batch_size)


class BatchSampler(Sampler):
    """
    Custom BatchSampler for PyTorch DataLoader that supports variable target sequence lengths
    per batch and ensures all samples within a batch have the same target length.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        max_sequence_length: int,
        random_sequence_length: bool = True,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Initializes the BatchSampler.

        Args:
            dataset_size (int): The total number of samples in the dataset.
            batch_size (int): The number of samples per batch.
            max_sequence_length (int): The maximum possible target sequence length.
            random_sequence_length (bool): If True, randomly selects a target sequence length
                                           between 1 and max_sequence_length for each batch.
            shuffle (bool): If True, shuffles the dataset indices before creating batches.
            drop_last (bool): If True, drops the last incomplete batch.
        """
        if not isinstance(dataset_size, int) or dataset_size < 0:
            raise ValueError("dataset_size must be a non-negative integer.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(max_sequence_length, int) or max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be a positive integer.")

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.random_sequence_length = random_sequence_length
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        """
        Generates batches of indices, with an associated target sequence length.
        """
        indices = list(range(self.dataset_size))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue

            # All samples in a batch will have the same target_length
            target_length = (
                random.randint(1, self.max_sequence_length)
                if self.random_sequence_length
                else self.max_sequence_length
            )
            yield [(idx, target_length) for idx in batch_indices]

    def __len__(self):
        """
        Returns the number of batches in an epoch.
        """
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return (self.dataset_size + self.batch_size - 1) // self.batch_size


class DataLoaderWrapper(DataLoader):
    """
    A custom DataLoader that provides convenient denormalization and unit-range scaling methods
    by leveraging methods present in its underlying dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(
            self.dataset, (QGDatasetBase, QGDatasetQuantile, MultipleSims)
        ):
            logger.warning(
                "DataLoaderWrapper is designed for QGDatasetBase subclasses. "
                "Denormalization and to_unit_range methods might not be available."
            )

    def denormalize(self, x: TensorDict) -> TensorDict:
        """
        Denormalizes a TensorDict using the dataset's denormalization method.

        Args:
            x (TensorDict): The TensorDict to denormalize.

        Returns:
            TensorDict: The denormalized TensorDict.

        Raises:
            AttributeError: If the dataset does not have a `denormalize` method.
        """
        if hasattr(self.dataset, "denormalize"):
            return self.dataset.denormalize(x)
        else:
            raise AttributeError("The dataset does not have a `denormalize` method.")

    def to_unit_range(self, x: TensorDict) -> TensorDict:
        """
        Maps variables in the given TensorDict to the [0, 1] range using dataset-wide min/max values.

        Args:
            x (TensorDict): The TensorDict to scale.

        Returns:
            TensorDict: The scaled TensorDict with values in [0, 1].

        Raises:
            AttributeError: If the dataset does not have `mins` or `maxs` attributes.
        """
        if not hasattr(self.dataset, "mins") or not hasattr(self.dataset, "maxs"):
            raise AttributeError(
                "Dataset does not have `mins` or `maxs` attributes for unit range scaling."
            )

        scaled_data = {}
        for var, tensor in x.items():
            if var in ["seq_length", "Re", "obstacle_mask"]:
                scaled_data[var] = tensor  # Passthrough metadata variables
                continue

            if var not in self.dataset.mins or var not in self.dataset.maxs:
                logger.warning(
                    f"Unit range scaling: Variable '{var}' not found in stored mins/maxs. Passing through."
                )
                scaled_data[var] = tensor
                continue

            min_val = self.dataset.mins[var].to(tensor.device)
            max_val = self.dataset.maxs[var].to(tensor.device)

            diff = max_val - min_val
            if diff < EPS:  # Handle constant data: map to 0.5 (mid-point)
                logger.warning(
                    f"Unit range scaling: Variable '{var}' has a min/max range close to zero. Mapping to 0.5."
                )
                scaled_data[var] = torch.full_like(tensor, 0.5)
            else:
                scaled_data[var] = (tensor - min_val) / (diff + EPS)
        return TensorDict(scaled_data, batch_size=x.batch_size)


def custom_collate_fn(batch: List[tuple]) -> tuple[TensorDict, TensorDict]:
    """
    Custom collate function for DataLoader to stack lists of TensorDicts into single TensorDicts.

    Args:
        batch (List[tuple]): A list of (input_seq, target_seq) tuples, where each
                             input_seq and target_seq are TensorDicts.

    Returns:
        tuple[TensorDict, TensorDict]: A tuple containing a batched input TensorDict
                                      and a batched target TensorDict.
    """
    input_seqs, target_seqs = zip(*batch)
    input_tensordict = stack_tensordict(list(input_seqs))
    target_tensordict = stack_tensordict(list(target_seqs))
    return input_tensordict, target_tensordict


def create_dataloaders(
    train_dataset: QGDatasetBase,
    val_dataset: QGDatasetBase,
    test_dataset: QGDatasetBase,
    config: Any,
) -> tuple[DataLoaderWrapper, DataLoaderWrapper, DataLoaderWrapper]:
    """
    Creates standard PyTorch DataLoaders with custom BatchSampler and collate function.

    Args:
        train_dataset (QGDatasetBase): Training dataset.
        val_dataset (QGDatasetBase): Validation dataset.
        test_dataset (QGDatasetBase): Test dataset.
        config (Any): Configuration object containing training parameters like batch_size.

    Returns:
        tuple[DataLoaderWrapper, DataLoaderWrapper, DataLoaderWrapper]: Train, validation, and test DataLoaders.
    """
    try:
        batch_size = config["training"]["batch_size"]
        random_sequence_length = config["training"]["random_sequence_length"]
    except KeyError as e:
        raise KeyError(f"Missing required key in config for dataloaders: {e}")

    train_batch_sampler = BatchSampler(
        dataset_size=len(train_dataset),
        batch_size=batch_size,
        max_sequence_length=train_dataset.max_sequence_length,
        random_sequence_length=random_sequence_length,
        shuffle=True,
        drop_last=True,
    )
    val_batch_sampler = BatchSampler(
        dataset_size=len(val_dataset),
        batch_size=batch_size,
        max_sequence_length=val_dataset.max_sequence_length,
        random_sequence_length=random_sequence_length,
        shuffle=True,  # Shuffle validation batches for consistency in reporting across runs if using partial validation
        drop_last=True,  # Drop last for consistent batch sizes in validation too
    )
    test_batch_sampler = BatchSampler(
        dataset_size=len(test_dataset),
        batch_size=batch_size,
        max_sequence_length=test_dataset.max_sequence_length,
        random_sequence_length=random_sequence_length,
        shuffle=False,  # Typically no shuffle for test set
        drop_last=False,  # Don't drop last for test set to ensure all samples are evaluated
    )

    train_loader = DataLoaderWrapper(
        train_dataset, batch_sampler=train_batch_sampler, collate_fn=custom_collate_fn
    )
    val_loader = DataLoaderWrapper(
        val_dataset, batch_sampler=val_batch_sampler, collate_fn=custom_collate_fn
    )
    test_loader = DataLoaderWrapper(
        test_dataset, batch_sampler=test_batch_sampler, collate_fn=custom_collate_fn
    )

    return train_loader, val_loader, test_loader


def create_ddp_dataloaders(
    train_dataset: QGDatasetBase,
    val_dataset: QGDatasetBase,
    test_dataset: QGDatasetBase,
    config: Any,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoaderWrapper, DataLoaderWrapper, DataLoaderWrapper]:
    """
    Creates DistributedDataParallel (DDP) PyTorch DataLoaders using DistributedSampler.

    Args:
        train_dataset (QGDatasetBase): Training dataset.
        val_dataset (QGDatasetBase): Validation dataset.
        test_dataset (QGDatasetBase): Test dataset.
        config (Any): Configuration object containing training parameters like batch_size.
        rank (int): Current process rank (for DDP).
        world_size (int): Total number of processes (for DDP).

    Returns:
        tuple[DataLoaderWrapper, DataLoaderWrapper, DataLoaderWrapper]: Train, validation, and test DDP DataLoaders.
    """
    try:
        batch_size = config["training"]["batch_size"]
    except KeyError as e:
        raise KeyError(f"Missing required key in config for DDP dataloaders: {e}")

    # Note: DistributedSampler is mutually exclusive with batch_sampler.
    # We pass the batch_size directly to DataLoader and let DistributedSampler handle indexing.
    # Random sequence length per batch is not directly supported by DistributedSampler.
    # If varying sequence length per sample within a batch is needed, logic must be in dataset's __getitem__
    # with padding handled by collate_fn. For constant length per batch, it can be passed via __getitem__ if needed.

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoaderWrapper(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        drop_last=True,  # Recommended for DDP to ensure uniform batch sizes
    )
    val_loader = DataLoaderWrapper(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        drop_last=False,  # Typically don't drop last for evaluation
    )
    test_loader = DataLoaderWrapper(
        test_dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        drop_last=False,  # Typically don't drop last for evaluation
    )

    return train_loader, val_loader, test_loader
