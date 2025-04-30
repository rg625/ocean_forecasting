import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr

class QGDataset(Dataset):
    def __init__(self, data_path, sequence_length=2, variables=None):
        self.data = xr.open_dataset(data_path)
        self.sequence_length = sequence_length

        if variables is None:
            variables = list(self.data.data_vars.keys())
        self.variables = variables

        self.stacked_data = torch.stack([
            torch.FloatTensor(self.data[var].values)
            for var in variables
        ], dim=1)

        self.means = self.stacked_data.mean(dim=(0, 2, 3))
        self.stds = self.stacked_data.std(dim=(0, 2, 3))
        self.stacked_data = (self.stacked_data - self.means[None, :, None, None]) / self.stds[None, :, None, None]

    def __len__(self):
        return len(self.stacked_data) - self.sequence_length + 1

    def __getitem__(self, idx):
        data_seq = self.stacked_data[idx:idx + self.sequence_length]
        return data_seq[0], data_seq[1:]

    def denormalize(self, x):
        device = x.device  # Get the device of the input tensor
        means = self.means.to(device)  # Move means to the same device as x
        stds = self.stds.to(device)    # Move stds to the same device as x
        return x * stds[None, :, None, None] + means[None, :, None, None]
    