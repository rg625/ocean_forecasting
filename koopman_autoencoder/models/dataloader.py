import torch
from tensordict import TensorDict
from torch.utils.data import Dataset, DataLoader, Sampler
import xarray as xr
import random
import numpy as np
from tensordict import stack as stack_tensordict
from typing import Optional, List, Any


class QGDatasetBase(Dataset):
    def __init__(
        self,
        data_path: str,
        max_sequence_length: int = 2,
        variables: Optional[List[str]] = None,
    ):
        self.data = xr.open_dataset(data_path)
        self.max_sequence_length = max_sequence_length
        if variables is None:
            variables = list(self.data.data_vars.keys())
        self.variables = variables

        self._normalize()  # perform normalization

    def _normalize(self):
        self.means = TensorDict(
            {
                var: torch.tensor(
                    np.mean(self.data[var].values, axis=(0, 1, 2)), dtype=torch.float32
                )
                for var in self.variables
            },
            batch_size=[],
        )
        self.stds = TensorDict(
            {
                var: torch.tensor(
                    np.std(self.data[var].values, axis=(0, 1, 2)), dtype=torch.float32
                )
                for var in self.variables
            },
            batch_size=[],
        )
        self.stacked_data = TensorDict(
            {
                var: (
                    torch.tensor(self.data[var].values, dtype=torch.float32)
                    - self.means[var]
                )
                / (self.stds[var] + 1e-8)
                for var in self.variables
            },
            batch_size=[],
        )

    def __len__(self):
        return (
            len(next(iter(self.stacked_data.values())))
            - 2
            - self.max_sequence_length
            + 1
        )

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            start_idx, target_length = idx
        else:
            start_idx = idx
            target_length = self.max_sequence_length

        input_seq = TensorDict(
            {
                var: self.stacked_data[var][start_idx : start_idx + 2]
                for var in self.variables
            },
            batch_size=[],
        )
        target_seq = TensorDict(
            {
                var: self.stacked_data[var][
                    start_idx + 2 : start_idx + 2 + target_length
                ]
                for var in self.variables
            },
            batch_size=[],
        )
        target_seq["seq_length"] = torch.tensor(target_length, dtype=torch.int64)
        return input_seq, target_seq

    def denormalize(self, x):
        denormalized = {}
        for var, tensor in x.items():
            if var == "seq_length":
                continue
            device = tensor.device
            means = self.means[var].to(device)
            stds = self.stds[var].to(device)
            denormalized[var] = tensor * stds + means
        return TensorDict(denormalized, batch_size=x.batch_size)


class QGDatasetQuantile(QGDatasetBase):
    def __init__(
        self,
        data_path: str,
        max_sequence_length: int = 2,
        variables: Optional[List[str]] = None,
        quantile_range: tuple = (2.5, 97.5),
    ):
        """
        Dataset that normalizes using specified quantiles instead of mean/std.
        Normalizes to [-1, 1] using the quantile range.
        """
        self.quantile_range = quantile_range
        self.q_lows: Optional[TensorDict] = None
        self.q_highs: Optional[TensorDict] = None
        super().__init__(data_path, max_sequence_length, variables)

    def _normalize(self):
        raw_data = TensorDict(
            {var: torch.FloatTensor(self.data[var].values) for var in self.variables},
            batch_size=[],
        )

        self.stacked_data, self.q_lows, self.q_highs = self.quantile_normalize(
            raw_data, quantile_range=self.quantile_range
        )

    @staticmethod
    def quantile_normalize(tensor_dict: TensorDict, quantile_range=(2.5, 97.5)):
        """
        Normalize each variable in the tensor_dict to [-1, 1] using the specified quantile range.
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

            norm = 2 * (tensor - q_low) / (q_high - q_low) - 1
            normalized[var] = norm

        return (
            TensorDict(normalized, batch_size=[]),
            TensorDict(q_lows, batch_size=[]),
            TensorDict(q_highs, batch_size=[]),
        )

    def denormalize(self, tensor_dict: TensorDict):
        """
        Denormalize data from [-1, 1] back to the original range using stored quantiles.

        Parameters:
            tensor_dict: TensorDict
                Dictionary with normalized tensors.

        Returns:
            TensorDict: Denormalized version.
        """
        assert self.q_lows is not None, "Quantile lows must not be None"
        assert self.q_highs is not None, "Quantile highs must not be None"

        denormalized = {}
        for var, tensor in tensor_dict.items():
            if var == "seq_length":
                continue

            q_low = self.q_lows[var].to(tensor.device)
            q_high = self.q_highs[var].to(tensor.device)
            denorm = ((tensor + 1) / 2) * (q_high - q_low) + q_low
            denormalized[var] = denorm

        return TensorDict(denormalized, batch_size=tensor_dict.batch_size)


class DiffusionReaction(Dataset):
    def __init__(
        self,
        data_path: str,
        max_sequence_length: int = 2,
        variables: Optional[List[str]] = None,
    ):
        self.data = xr.open_dataset(data_path)
        self.max_sequence_length = max_sequence_length

        if variables is None:
            variables = list(self.data.data_vars.keys())
        self.variables = variables

        # Perform normalization across all simulations and time steps
        self._normalize()

        # Track the number of simulations (sim dimension)
        self.num_sims = self.data.sizes["sim"]

    def _normalize(self):
        # Stack the data across simulations (sim, t, x, y)
        self.stacked_data = TensorDict(
            {var: torch.FloatTensor(self.data[var].values) for var in self.variables},
            batch_size=[],
        )

        # Calculate means and stds across simulations, time, height, and width
        self.means = TensorDict(
            {
                var: torch.FloatTensor([np.mean(self.data[var].values, axis=(0, 1, 2))])
                for var in self.variables
            },
            batch_size=[],
        )

        self.stds = TensorDict(
            {
                var: torch.FloatTensor([np.std(self.data[var].values, axis=(0, 1, 2))])
                for var in self.variables
            },
            batch_size=[],
        )

        # Normalize the stacked data
        self.stacked_data = TensorDict(
            {
                var: (self.stacked_data[var] - self.means[var]) / self.stds[var]
                for var in self.variables
            },
            batch_size=[],
        )

    def __len__(self):
        # Each simulation has a different number of time steps (sim, t, x, y)
        return self.num_sims * (len(self.data["t"]) - 2 - self.max_sequence_length + 1)

    def __getitem__(self, idx):
        # If idx is a tuple, we extract the first element which is the index we want
        if isinstance(idx, tuple):
            idx, target_length = idx
        else:
            target_length = (
                self.max_sequence_length
            )  # Use default max_sequence_length if not provided

        # Calculate which simulation and index in the simulation to sample from
        sim_idx = idx // (len(self.data["t"]) - 2 - self.max_sequence_length + 1)
        start_idx = idx % (len(self.data["t"]) - 2 - self.max_sequence_length + 1)

        # Get the input sequence (fixed length of 2)
        input_seq = TensorDict(
            {
                var: self.stacked_data[var][sim_idx, start_idx : start_idx + 2]
                for var in self.variables
            },
            batch_size=[],
        )

        # Get the target sequence (variable length)
        target_length = self.max_sequence_length
        target_seq = TensorDict(
            {
                var: self.stacked_data[var][
                    sim_idx, start_idx + 2 : start_idx + 2 + target_length
                ]
                for var in self.variables
            },
            batch_size=[],
        )

        # Add the sequence length to the target
        target_seq["seq_length"] = torch.tensor(target_length, dtype=torch.int64)

        return input_seq, target_seq

    def denormalize(self, x):
        denormalized = {}
        for var, tensor in x.items():
            if var == "seq_length":
                continue
            device = tensor.device
            means = self.means[var].to(device)
            stds = self.stds[var].to(device)
            denormalized[var] = tensor * stds + means
        return TensorDict(denormalized, batch_size=x.batch_size)


# OVERFIT EXPERIMENTS ONLY
# class DiffusionReaction(QGDatasetBase):
#     def __init__(
#         self,
#         data_path: str,
#         max_sequence_length: int = 2,
#         variables: Optional[List[str]] = None,
#         sim: int = 0,
#     ):
#         super().__init__(data_path, max_sequence_length, variables)

#         # Override self.data to only keep the selected simulation
#         self.data = self.data.isel(sim=sim)

#         # Now re-run normalization with the filtered data
#         self._normalize()


class BatchSampler(Sampler):
    def __init__(
        self,
        dataset_size,
        batch_size,
        max_sequence_length,
        shuffle=True,
        drop_last=False,
    ):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        indices = list(range(self.dataset_size))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
            target_length = random.randint(1, self.max_sequence_length)
            yield [(idx, target_length) for idx in batch_indices]

    def __len__(self):
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return (self.dataset_size + self.batch_size - 1) // self.batch_size


class DataLoaderWrapper(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def denormalize(self, x):
        if hasattr(self.dataset, "denormalize"):
            return self.dataset.denormalize(x)
        else:
            raise AttributeError("The dataset does not have a `denormalize` method.")


def custom_collate_fn(batch):
    input_seqs, target_seqs = zip(*batch)
    input_tensordict = stack_tensordict(input_seqs)
    target_tensordict = stack_tensordict(target_seqs)
    return input_tensordict, target_tensordict


def create_dataloaders(
    train_dataset: QGDatasetBase,
    val_dataset: QGDatasetBase,
    test_dataset: QGDatasetBase,
    config: Any,
):
    train_batch_sampler = BatchSampler(
        len(train_dataset),
        config["training"]["batch_size"],
        train_dataset.max_sequence_length,
        shuffle=True,
        drop_last=True,
    )
    val_batch_sampler = BatchSampler(
        len(val_dataset),
        config["training"]["batch_size"],
        val_dataset.max_sequence_length,
        shuffle=False,
        drop_last=False,
    )
    test_batch_sampler = BatchSampler(
        len(test_dataset),
        config["training"]["batch_size"],
        test_dataset.max_sequence_length,
        shuffle=False,
        drop_last=False,
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
