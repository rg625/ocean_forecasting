import torch
from tensordict import TensorDict
from torch.utils.data import Dataset, DataLoader, Sampler
import xarray as xr
import random
import numpy as np
from tensordict import stack as stack_tensordict


class QGDataset(Dataset):
    def __init__(self, data_path, max_sequence_length=2, variables=None):
        """
        Custom Dataset for QG (Quasi-Geostrophic) data with variable target sequence lengths.
        Parameters:
            data_path: str
                Path to the dataset file.
            max_sequence_length: int
                Maximum length of the target sequence.
            variables: list of str
                Variables to extract from the dataset. If None, all variables will be used.
        """
        # self.data = xr.open_dataset(data_path)
        # self.max_sequence_length = max_sequence_length
        # if variables is None:
        #     variables = list(self.data.data_vars.keys())
        # self.variables = variables

        # # Stack data into a dictionary of tensors with dimensions [time, height, width]
        # self.stacked_data = TensorDict(
        #     {var: torch.FloatTensor(self.data[var].values) for var in variables}
        # )

        # # Compute the min and max values for normalization for each variable
        # # self.mins = TensorDict(
        # #     {var: self.stacked_data[var].amin(dim=(0, 1, 2)) for var in variables}
        # # )
        # # self.maxs = TensorDict(
        # #     {var: self.stacked_data[var].amax(dim=(0, 1, 2)) for var in variables}
        # # )
        # self.means = TensorDict(
        #     {var: self.stacked_data[var].mean(dim=(0, 1, 2)) for var in variables}
        # )
        # self.stds = TensorDict(
        #     {var: self.stacked_data[var].std(dim=(0, 1, 2)) for var in variables}
        # )
        # self.stacked_data = TensorDict(
        #     {
        #         var: (self.stacked_data[var] - self.means[var]) / self.stds[var]
        #         for var in variables
        #     }
        # )

        # # Normalize data to the range [-1, 1]
        # self.stacked_data = TensorDict(
        #     {
        #         var: 2
        #         * (self.stacked_data[var] - self.mins[var])
        #         / (self.maxs[var] - self.mins[var])
        #         - 1
        #         for var in variables
        #     }
        # )

        self.data = xr.open_dataset(data_path)
        self.max_sequence_length = max_sequence_length
        if variables is None:
            variables = list(self.data.data_vars.keys())
        self.variables = variables

        # Stack data into a dictionary of tensors with dimensions [time, height, width]
        self.stacked_data = TensorDict(
            {var: torch.FloatTensor(self.data[var].values) for var in variables}
        )

        # Compute the means and stds of the raw (unnormalized) data for each variable
        self.means = TensorDict(
            {
                var: torch.FloatTensor(np.mean(self.data[var].values, axis=(0, 1, 2)))
                for var in variables
            }
        )
        self.stds = TensorDict(
            {
                var: torch.FloatTensor(np.std(self.data[var].values, axis=(0, 1, 2)))
                for var in variables
            }
        )

        # Normalize the data to have mean 0 and std 1
        self.stacked_data = TensorDict(
            {
                var: (self.stacked_data[var] - self.means[var]) / self.stds[var]
                for var in variables
            }
        )

    def __len__(self):
        """
        Length of the dataset.
        """
        # Ensure we can extract an input sequence of length 2 and at least one target step
        return (
            len(next(iter(self.stacked_data.values())))
            - 2
            - self.max_sequence_length
            + 1
        )

    def __getitem__(self, idx):
        """
        Get a sample with a fixed input sequence length of 2 and a target sequence length
        determined by the batch.

        Parameters:
            idx: tuple (start_idx, target_length)
                Index of the starting point of the sequence and the target sequence length.
        Returns:
            tuple: (input_sequence, target_sequence)
        """
        if isinstance(idx, tuple):
            start_idx, target_length = idx
        else:
            # For standard indexing (used during initialization)
            start_idx = idx
            target_length = self.max_sequence_length

        # Extract the input sequence (fixed length of 2) and target sequence for each variable
        input_seq = TensorDict(
            {
                var: self.stacked_data[var][start_idx : start_idx + 2]
                for var in self.variables
            }
        )
        target_seq = TensorDict(
            {
                var: self.stacked_data[var][
                    start_idx + 2 : start_idx + 2 + target_length
                ]
                for var in self.variables
            }
        )

        # Add sequence length to the target sequence
        target_seq["seq_length"] = torch.tensor(target_length, dtype=torch.int64)

        # Return the input sequence and target sequence as dictionaries
        return input_seq, target_seq

    # def denormalize(self, x):
    #     """
    #     Denormalize the data from [-1, 1] back to the original range.
    #     Parameters:
    #         x: dict of torch.Tensor
    #             Normalized tensor dictionary.
    #     Returns:
    #         dict of torch.Tensor: Denormalized tensor dictionary.
    #     """
    #     denormalized = {}
    #     for var, tensor in x.items():
    #         if var == "seq_length":  # Skip denormalizing sequence length
    #             # denormalized[var] = tensor
    #             continue

    #         device = tensor.device  # Get the device of the input tensor
    #         mins = self.mins[var].to(device)  # Move mins to the same device as tensor
    #         maxs = self.maxs[var].to(device)  # Move maxs to the same device as tensor
    #         denormalized[var] = 0.5 * (tensor + 1) * (maxs - mins) + mins

    #     return TensorDict(denormalized, batch_size=x.batch_size)

    def denormalize(self, x):
        """
        Denormalize the data from mean 0, std 1 back to the original range.
        Parameters:
            x: dict of torch.Tensor
                Normalized tensor dictionary.
        Returns:
            dict of torch.Tensor: Denormalized tensor dictionary.
        """
        denormalized = {}
        for var, tensor in x.items():
            if var == "seq_length":  # Skip denormalizing sequence length
                continue

            device = tensor.device  # Get the device of the input tensor
            means = self.means[var].to(
                device
            )  # Move means to the same device as tensor
            stds = self.stds[var].to(device)  # Move stds to the same device as tensor
            denormalized[var] = tensor * stds + means

        return TensorDict(denormalized, batch_size=x.batch_size)


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
    """
    Custom DataLoader that wraps the PyTorch DataLoader and exposes the `denormalize`
    method from the underlying dataset.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the DataLoaderWrapper.
        """
        super().__init__(*args, **kwargs)

        print("Dataset means:")
        for var, mean in self.dataset.means.items():
            print(f"  {var}: {mean.item():.10f}")

        print("Dataset stds:")
        for var, std in self.dataset.stds.items():
            print(f"  {var}: {std.item():.10f}")

    def denormalize(self, x):
        """
        Expose the `denormalize` method from the underlying dataset.

        Args:
            x: The tensor or tensordict to denormalize.

        Returns:
            The denormalized tensor or tensordict.
        """

        if hasattr(self.dataset, "denormalize"):
            return self.dataset.denormalize(x)
        else:
            raise AttributeError("The dataset does not have a `denormalize` method.")


def custom_collate_fn(batch):
    """
    Custom collation function to handle TensorDict objects in batches.

    Args:
        batch: List of tuples (input_seq, target_seq) from the dataset.

    Returns:
        Tuple of stacked TensorDicts (inputs, targets).
    """
    input_seqs, target_seqs = zip(*batch)  # Unzip the input and target sequences
    input_tensordict = stack_tensordict(input_seqs)  # Stack TensorDicts for input
    target_tensordict = stack_tensordict(target_seqs)  # Stack TensorDicts for target
    return input_tensordict, target_tensordict


def create_dataloaders(train_dataset, val_dataset, test_dataset, config):
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
