import torch
import xarray as xr
import numpy as np
import random
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple
from abc import ABC, abstractmethod
from omegaconf import DictConfig, ListConfig

from tensordict import TensorDict, stack as stack_tensordict
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

INC_MEAN = {"v_x": 0.444969, "v_y": 0.000299, "p": 0.000586, "Re": 550.0}
INC_STD = {"v_x": 0.206128, "v_y": 0.206128, "p": 0.003942, "Re": 262.678467}


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
        # self.means = TensorDict(
        #     {key: val for key, val in INC_MEAN.items()},
        #     batch_size=[],
        # )
        # self.stds = TensorDict(
        #     {key: val for key, val in INC_STD.items()},
        #     batch_size=[],
        # )
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
                "NormalNormalizer has not been fitted (means is None)."
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
                denormalized[key] = (tensor * std) + mean
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
        subsample: int = 1,
        **kwargs,
    ):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        self.input_sequence_length = input_sequence_length
        self.max_sequence_length = max_sequence_length
        self.normalizer = normalizer
        self.data_vars = list(variables.keys())
        self.subsample = subsample

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

        # Calculate the total number of original data points needed for one full sequence
        required_index_span = (
            (self.input_sequence_length + self.max_sequence_length - 1) * self.subsample
        ) + 1

        if num_timesteps < required_index_span:
            return 0

        total_len = num_timesteps - required_index_span + 1
        return int(max(0, total_len))

    def __getitem__(self, idx: Union[int, Tuple[int, int]]):
        """
        Retrieves a single input/target sequence pair at a given index.

        The `idx` corresponds to the starting time index in the original `stacked_data`.
        """
        # Determine the starting index and the desired target length
        start_idx, target_length = (
            (idx, self.max_sequence_length) if isinstance(idx, int) else idx
        )

        # Validate that the target_length is within the allowed bounds
        if not (0 <= target_length <= self.max_sequence_length):
            raise ValueError(
                f"target_length must be in [0, {self.max_sequence_length}]"
            )

        # Calculate the end point for the input sequence slice.
        # To get `L` points with a step of `S`, the slice must be `data[t : t + L*S : S]`.
        input_end = start_idx + self.input_sequence_length * self.subsample
        input_seq = self.stacked_data[..., start_idx : input_end : self.subsample, :, :]

        # The target sequence starts immediately after the input sequence's window.
        target_start = input_end
        target_end = target_start + target_length * self.subsample
        target_seq = self.stacked_data[
            ..., target_start : target_end : self.subsample, :, :
        ]

        # Create a metadata dictionary. A custom collate function can handle this.
        # Note: In QGDatasetMultiSim, this is a (data, destination) tuple
        metadata = {"seq_length": (target_length, "target")}

        # Return a 3-tuple that a custom_collate_fn would expect
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
    """
    Dataset for multiple simulations, handling dynamic, static, and scalar
    control parameters in a generic, scalable way.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        normalizer: AbstractNormalizer,
        input_sequence_length: int,
        max_sequence_length: int,
        variables: Dict[str, int],
        static_variables: Optional[Dict[str, int]] = None,
        subsample: int = 1,
        **kwargs,
    ):
        # Store the keys for dynamic and static variables
        self.select_cond = kwargs.pop("select_cond", None)
        self.dynamic_keys = list(variables.keys())
        self.static_keys = list(static_variables.keys()) if static_variables else []
        self.subsample = subsample
        all_variables_to_load = {**variables, **(static_variables or {})}

        self.control_parameters: List[str] = kwargs.pop("control_parameters", [])
        self.selection_param: Optional[str] = kwargs.pop("selection_param", None)

        if self.selection_param and self.selection_param not in self.control_parameters:
            logger.warning(
                f"selection_param '{self.selection_param}' is not in control_parameters list."
            )
            self.control_parameters.append(self.selection_param)

        # This will hold all loaded control parameter tensors (e.g., {"Re": tensor, "Ma": tensor})
        self.control_params: Dict[str, torch.Tensor] = {}

        # These attributes will be populated in _load_data
        self.num_sims: int = 0
        self.master_index: List[Tuple[int, int]] = []
        self.static_tensors: Dict[str, torch.Tensor] = {}
        self.obstacle_mask: Optional[torch.Tensor] = None  # Initialize as None

        super().__init__(
            data_path,
            normalizer,
            input_sequence_length,
            max_sequence_length,
            variables=all_variables_to_load,
            subsample=subsample,
            **kwargs,  # Pass remaining kwargs (if any)
        )

    def _load_data(self):
        """
        Loads data, correctly separating dynamic, static, and control variables
        and performing generic filtering.
        This method completely overrides the base class's _load_data.
        """
        with xr.open_dataset(self.data_path) as ds:
            self._data = ds

            if "sim" not in self._data.dims:
                raise DatasetConfigurationError(
                    "Expected 'sim' dimension in the dataset."
                )
            self.num_sims = self._data.sizes["sim"]

            # --- MODULAR CHANGES ---
            # 2. Load all specified control parameters
            for param_name in self.control_parameters:
                if param_name in self._data:
                    self.control_params[param_name] = torch.from_numpy(
                        self._data[param_name].values
                    ).float()
                else:
                    logger.warning(
                        f"Control parameter '{param_name}' not found in dataset."
                    )

            # 3. Handle selection/filtering generically
            if self.selection_param and self.select_cond is not None:
                if self.selection_param not in self.control_params:
                    raise ValueError(
                        f"Selection parameter '{self.selection_param}' was specified but not found in dataset."
                    )

                control_tensor = self.control_params[self.selection_param]
                mask = self._build_selection_mask(
                    tensor=control_tensor,
                    selection=self.select_cond,
                    name=self.selection_param,
                )

                selected_indices = mask.nonzero(as_tuple=True)[0]
                if len(selected_indices) == 0:
                    raise ValueError(
                        f"No simulations found with {self.selection_param} criteria: {self.select_cond}"
                    )

                logger.info(
                    f"Filtering {self.num_sims} simulations → {len(selected_indices)} "
                    f"with {self.selection_param} in {self.select_cond}"
                )

                # Filter the dataset view
                ds = ds.isel(sim=selected_indices)
                self.num_sims = len(selected_indices)

                # CRITICAL: Filter all loaded control parameter tensors
                for param_name, tensor in self.control_params.items():
                    self.control_params[param_name] = tensor[selected_indices]

            # --- END MODULAR CHANGES ---

            # --- Segregated Variable Loading (Unchanged) ---
            # 1. Load ONLY DYNAMIC variables for the main TensorDict
            dynamic_tensors = {
                var: torch.from_numpy(ds[var].values).float()
                for var in self.dynamic_keys
            }

            # 2. Load STATIC variables into a separate dictionary
            self.static_tensors = {
                var: torch.from_numpy(ds[var].values).float()
                for var in self.static_keys
            }
            if "obstacle_mask" in self.static_tensors:
                logger.info("Obstacle mask found and processed from static variables.")
                mask_tensor = self.static_tensors["obstacle_mask"]
                self.obstacle_mask = (
                    mask_tensor[0] if mask_tensor.ndim > 2 else mask_tensor
                )

            # --- Setup raw_data_td (Unchanged) ---
            sample_var_name = self.dynamic_keys[0]
            sample_tensor_shape = dynamic_tensors[sample_var_name].shape
            num_feature_dims = 2  # (x, y)
            num_batch_dims = len(sample_tensor_shape) - num_feature_dims
            batch_size = list(sample_tensor_shape[:num_batch_dims])

            self.raw_data_td = TensorDict(dynamic_tensors, batch_size=batch_size)

            self.mins = TensorDict(
                {key: torch.min(tensor) for key, tensor in self.raw_data_td.items()},
                batch_size=[],
            )
            self.maxs = TensorDict(
                {key: torch.max(tensor) for key, tensor in self.raw_data_td.items()},
                batch_size=[],
            )

    def _build_selection_mask(
        self, tensor: torch.Tensor, selection, name: str
    ) -> torch.Tensor:
        """Helper to build a boolean mask given selection criteria. (Unchanged)"""
        mask = torch.zeros_like(tensor, dtype=torch.bool)

        if isinstance(selection, (list, tuple, ListConfig)) and all(
            isinstance(item, (list, tuple, ListConfig)) for item in selection
        ):
            logger.info(f"Filtering simulations with {name} in intervals: {selection}")
            for rng in selection:
                vmin, vmax = rng
                mask |= (tensor >= vmin) & (tensor <= vmax)

        elif isinstance(selection, (float, int)):
            logger.info(f"Filtering simulations with {name} == {selection}")
            mask = tensor == selection

        elif isinstance(selection, (list, tuple, ListConfig)):
            if len(selection) == 2 and all(
                isinstance(v, (float, int)) for v in selection
            ):
                vmin, vmax = selection
                logger.info(
                    f"Filtering simulations with {name} in interval [{vmin}, {vmax}]"
                )
                mask = (tensor >= vmin) & (tensor <= vmax)
            else:
                logger.info(f"Filtering simulations with {name} in list: {selection}")
                mask = torch.isin(tensor, torch.tensor(list(selection)))

        else:
            raise ValueError(f"Invalid type for '{name}' selection: {type(selection)}")

        return mask

    def denormalize(self, data: TensorDict) -> TensorDict:
        denormalized = self.normalizer.inverse_transform(data)
        return self.apply_mask(denormalized)

    def to_unit_range(self, data: TensorDict) -> TensorDict:
        """Scales data to [0, 1] range based on global min/max."""
        # This now correctly uses the scaled data from the super() call
        scaled_data = super().to_unit_range(data=data)
        return self.apply_mask(scaled_data)

    @property
    def normalizer_vars(self) -> List[str]:
        """Specifies that only dynamic variables should be normalized. (Unchanged)"""
        return self.dynamic_keys

    def _prepare_data(self):
        """Computes the master index before preparing the data. (Unchanged)"""
        self._compute_master_index()
        # This now calls the QGDatasetBase._prepare_data
        super()._prepare_data()

    def _compute_master_index(self):
        """Creates a master list of all possible (sim, start_index) pairs."""
        self.master_index = []
        if "t" in self._data.sizes:
            num_timesteps = self._data.sizes["t"]
        elif "time" in self._data.sizes:
            num_timesteps = self._data.sizes["time"]
        else:
            raise ValueError("Missing time dimension")

        required_length_in_steps = self.input_sequence_length + self.max_sequence_length
        if required_length_in_steps == 0:
            logger.warning("Total sequence length (input+max) is 0.")
            return

        # The total number of *original data points* needed for one full sequence
        required_index_span = (
            (self.input_sequence_length + self.max_sequence_length - 1) * self.subsample
        ) + 1

        # The number of valid starting positions
        valid_starts = num_timesteps - required_index_span + 1

        if valid_starts > 0:
            for sim_idx in range(self.num_sims):
                self.master_index.extend([(sim_idx, i) for i in range(valid_starts)])
        if not self.master_index:
            logger.warning(
                f"No valid sequences generated from dataset. "
                f"Sims: {self.num_sims}, Timesteps: {num_timesteps}, "
                f"Required span: {required_index_span}, Valid starts: {valid_starts}"
            )

    # def _compute_master_index(self):
    #     """
    #     Creates a master list of all possible (sim, start_index) pairs,
    #     generating non-overlapping sequences based on the full sequence length.
    #     """
    #     self.master_index = []
    #     if "t" in self._data.sizes:
    #         num_timesteps = self._data.sizes["t"]
    #     elif "time" in self._data.sizes:
    #         num_timesteps = self._data.sizes["time"]
    #     else:
    #         raise ValueError("Missing time dimension")

    #     # The step size (or stride) between the start of consecutive non-overlapping sequences.
    #     # This is the span of indices in the original data that one full sequence
    #     # (input + max_target) covers.
    #     stride = (
    #         self.input_sequence_length + self.max_sequence_length
    #     ) * self.subsample

    #     # The last possible starting index must allow for a full sequence to be drawn.
    #     # A sequence starting at `s` will need data up to index `s + stride - 1`.
    #     # So, `s + stride - 1 < num_timesteps` => `s <= num_timesteps - stride`.
    #     last_possible_start = num_timesteps - stride

    #     if last_possible_start >= 0:
    #         for sim_idx in range(self.num_sims):
    #             # Iterate with a step size equal to the stride for non-overlapping samples.
    #             # The `stop` for range is exclusive, so we use `last_possible_start + 1`.
    #             start_indices = range(0, last_possible_start + 1, stride)
    #             self.master_index.extend([(sim_idx, i) for i in start_indices])

    #     if not self.master_index:
    #         logger.warning(
    #             "No valid non-overlapping sequences generated from the dataset. "
    #             "Check sequence lengths, subsampling rate, and total timesteps."
    #         )

    def __len__(self) -> int:
        return len(self.master_index)

    def apply_mask(self, x: TensorDict) -> TensorDict:
        """
        Applies an obstacle mask by dynamically broadcasting it to the shape of
        each tensor in the TensorDict. This is the most robust method.
        (Unchanged)
        """
        if self.obstacle_mask is None:
            return x

        def broadcasting_mask_apply(tensor: torch.Tensor) -> torch.Tensor:
            """
            A closure that applies the correctly reshaped mask to a tensor.
            """
            assert self.obstacle_mask is not None, "Cannot apply None masking"
            mask = self.obstacle_mask.to(tensor.device)
            mask_shape = mask.shape
            tensor_shape = tensor.shape
            mask_rank = mask.ndim
            tensor_rank = tensor.ndim

            if tensor_rank >= mask_rank and tensor_shape[-mask_rank:] == mask_shape:
                new_shape = (1,) * (tensor_rank - mask_rank) + mask_shape
                broadcastable_mask = mask.view(new_shape)

                return tensor * broadcastable_mask
            else:
                return tensor

        return x.apply(broadcasting_mask_apply)

    def __getitem__(self, idx: Union[int, Tuple[int, int]]):
        """
        Retrieves a sample and prepares a single metadata dictionary where each
        value is a (data, destination_marker) tuple.
        """
        flat_idx, target_length = (
            (idx, self.max_sequence_length) if isinstance(idx, int) else idx
        )
        if not (0 <= flat_idx < len(self)):
            raise IndexError("Index out of bounds.")

        sim_idx, start_idx = self.master_index[flat_idx]

        input_end = start_idx + self.input_sequence_length * self.subsample
        input_seq = self.stacked_data[sim_idx, start_idx : input_end : self.subsample]

        target_start = input_end
        target_end = target_start + target_length * self.subsample
        target_seq = self.stacked_data[
            sim_idx, target_start : target_end : self.subsample
        ]

        # --- Prepare a single metadata dictionary with destination markers ---
        metadata = {
            "seq_length": (target_length, "target"),
        }

        if self.obstacle_mask is not None:
            metadata["obstacle_mask"] = (self.obstacle_mask, "input")

        # 4. Loop through all loaded control params and add them to metadata
        for param_name, param_tensor in self.control_params.items():
            value = param_tensor[sim_idx]
            # metadata[f"{param_name}_target"] = (value, "target")
            # metadata[f"{param_name}_input"] = (value, "input")

            # If this is the main selection param, add the generic 'cond' key
            if param_name == self.selection_param:
                metadata["cond_target"] = (value, "target")
                metadata["cond_input"] = (value, "input")

        input_seq = self.apply_mask(input_seq)
        target_seq = self.apply_mask(target_seq)

        # This returns the standard 3-tuple
        return input_seq, target_seq, metadata

    def create_subset_for_condition(
        self, param_name: str, param_value: Union[float, int]
    ) -> Subset:
        """
        Creates a torch.utils.data.Subset containing only the samples
        for a specific value of a given control parameter.
        """
        if param_name not in self.control_params:
            raise ValueError(
                f"Cannot create subset for '{param_name}' because it was not "
                f"loaded. Available parameters are: {list(self.control_params.keys())}"
            )

        logger.info(f"Creating a subset for {param_name} = {param_value}...")

        # Get the 1D tensor for the requested parameter
        param_tensor = self.control_params[param_name]

        # Find which simulation indices correspond to the desired value
        sim_indices_with_value = (param_tensor == param_value).nonzero(as_tuple=True)[0]

        if len(sim_indices_with_value) == 0:
            logger.warning(
                f"No simulations found for {param_name} = {param_value} in this dataset split. "
                "Returning an empty subset."
            )
            return Subset(self, [])

        # Find all master_index entries that belong to these simulation indices
        subset_indices = [
            i
            for i, (sim_idx, _) in enumerate(self.master_index)
            if sim_idx in sim_indices_with_value
        ]

        logger.info(
            f"Found {len(subset_indices)} samples for {param_name} = {param_value}."
        )
        return Subset(self, subset_indices)


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
        # Ensure the dataset attribute is correctly typed for methods
        self.dataset: QGDatasetBase | QGDatasetMultiSim

    def denormalize(self, x: TensorDict) -> TensorDict:
        return self.dataset.denormalize(x)

    def to_unit_range(self, x: TensorDict) -> TensorDict:
        return self.dataset.to_unit_range(x)


def custom_collate_fn(batch: List[Tuple[TensorDict, TensorDict, Dict]]):
    """
    Collates data and routes metadata to the input or target TensorDict
    based on a destination marker in the metadata values.
    """
    input_tds, target_tds, meta_dicts = zip(*batch)

    batched_inputs = stack_tensordict(input_tds)
    batched_targets = stack_tensordict(target_tds)

    # Check if there's any metadata to process
    if not meta_dicts or not meta_dicts[0]:
        return batched_inputs, batched_targets

    # Process each key from the metadata dictionaries
    for key in meta_dicts[0].keys():
        # --- 1. Unpack data and destination from the batch ---
        # The value for each item is a (data, destination) tuple.
        try:
            data_list = [d[key][0] for d in meta_dicts]
            destination_marker = meta_dicts[0][key][1]  # 'input' or 'target'
        except KeyError:
            logger.warning(
                f"Metadata key '{key}' not found in all batch items. Skipping."
            )
            continue
        except (TypeError, IndexError):
            logger.warning(f"Metadata for key '{key}' has malformed value. Skipping.")
            continue

        # --- 2. Batch the data ---
        if isinstance(data_list[0], torch.Tensor):
            try:
                meta_tensor = torch.stack(data_list)
            except RuntimeError as e:
                logger.warning(f"Could not stack metadata for key '{key}'. Error: {e}")
                continue
        else:
            try:
                meta_tensor = torch.tensor(data_list)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not convert metadata to tensor for key '{key}'. Error: {e}"
                )
                continue

        # --- 3. Identify the destination TensorDict ---
        if destination_marker == "input":
            destination_td = batched_inputs
        elif destination_marker == "target":
            destination_td = batched_targets
        else:
            # Skip any items with an unknown destination
            logger.warning(
                f"Unknown destination '{destination_marker}' for key '{key}'. Skipping."
            )
            continue

        # --- 4. Expand and assign to the correct destination ---
        try:
            # Check if destination_td is empty (batch size 0)
            if destination_td.batch_size[0] == 0:
                continue

            seq_len = destination_td.batch_size[1]

            meta_tensor_reshaped = meta_tensor.unsqueeze(1)
            expand_shape = list(meta_tensor_reshaped.shape)
            expand_shape[1] = seq_len
            meta_tensor_expanded = meta_tensor_reshaped.expand(*expand_shape)

            destination_td.set(key, meta_tensor_expanded)

        except Exception as e:
            logger.warning(
                f"Failed to expand and assign metadata for key '{key}'. Error: {e}"
            )

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
