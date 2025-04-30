import torch
from torch.utils.data import Dataset
import xarray as xr
import random

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
        self.data = xr.open_dataset(data_path)
        self.max_sequence_length = max_sequence_length

        if variables is None:
            variables = list(self.data.data_vars.keys())
        self.variables = variables

        # Stack data into a tensor with dimensions [time, variables, height, width]
        self.stacked_data = torch.stack([
            torch.FloatTensor(self.data[var].values)
            for var in variables
        ], dim=1)

        # Compute the min and max values for normalization in [-1, 1]
        self.mins = self.stacked_data.amin(dim=(0, 2, 3))
        self.maxs = self.stacked_data.amax(dim=(0, 2, 3))

        # Normalize data to the range [-1, 1]
        self.stacked_data = 2 * (self.stacked_data - self.mins[None, :, None, None]) / \
                            (self.maxs[None, :, None, None] - self.mins[None, :, None, None]) - 1

    def __len__(self):
        """
        Length of the dataset.
        """
        # Ensure we can extract an input sequence of length 2 and at least one target step
        return len(self.stacked_data) - 2 - self.max_sequence_length + 1

    def __getitem__(self, idx):
        """
        Get a sample with a fixed input sequence length of 2 and a variable target sequence length.

        Parameters:
            idx: int
                Index of the starting point of the sequence.

        Returns:
            tuple: (input_sequence, target_sequence, target_length)
        """
        # Randomly sample a target sequence length between 1 and the maximum target length
        target_length = random.randint(1, self.max_sequence_length)

        # Extract the input sequence (fixed length of 2)
        input_seq = self.stacked_data[idx:idx + 2]

        # Extract the target sequence (variable length)
        target_seq = self.stacked_data[idx + 2:idx + 2 + target_length]

        # Return the input sequence, target sequence, and the target length
        return input_seq, target_seq, target_length

    def denormalize(self, x):
        """
        Denormalize the data from [-1, 1] back to the original range.

        Parameters:
            x: torch.Tensor
                Normalized tensor.

        Returns:
            torch.Tensor: Denormalized tensor.
        """
        device = x.device  # Get the device of the input tensor
        mins = self.mins.to(device)  # Move mins to the same device as x
        maxs = self.maxs.to(device)  # Move maxs to the same device as x
        return 0.5 * (x + 1) * (maxs[None, :, None, None] - mins[None, :, None, None]) + mins[None, :, None, None]
    