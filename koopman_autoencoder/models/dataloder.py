import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import xarray as xr
import random
import numpy as np

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
            
        # Extract the input sequence (fixed length of 2)
        input_seq = self.stacked_data[start_idx:start_idx + 2]
        
        # Extract the target sequence (of specified length)
        target_seq = self.stacked_data[start_idx + 2:start_idx + 2 + target_length]
        
        # Return the input sequence and target sequence
        return input_seq, target_seq
        
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


class BatchSampler(Sampler):
    """
    Custom batch sampler that creates batches where all samples have the same target sequence length.
    """
    def __init__(self, dataset_size, batch_size, max_sequence_length, shuffle=True, drop_last=False):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.shuffle = shuffle
        self.drop_last = drop_last
        
    def __iter__(self):
        # For each batch, randomly choose a target sequence length
        batches = []
        indices = list(range(self.dataset_size))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Create batches with uniform target sequence length
        for i in range(0, len(indices), self.batch_size):
            if i + self.batch_size > len(indices) and self.drop_last:
                continue
                
            # For this batch, choose a random target sequence length
            target_length = random.randint(1, self.max_sequence_length)
            
            # Get indices for this batch
            batch_indices = indices[i:i + self.batch_size]
            
            # Yield (index, target_length) tuples for each item in the batch
            batch = [(idx, target_length) for idx in batch_indices]
            batches.append(batch)
            
        # If shuffle is True, also shuffle the order of batches
        if self.shuffle:
            random.shuffle(batches)
            
        for batch in batches:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return self.dataset_size // self.batch_size
        else:
            return (self.dataset_size + self.batch_size - 1) // self.batch_size


# Example usage
def create_dataloaders(train_dataset, val_dataset, test_dataset, config):
    train_batch_sampler = BatchSampler(
        len(train_dataset), 
        config['training']['batch_size'], 
        train_dataset.max_sequence_length,
        shuffle=True,
        drop_last=True
    )
    
    val_batch_sampler = BatchSampler(
        len(val_dataset), 
        config['training']['batch_size'], 
        val_dataset.max_sequence_length,
        shuffle=False,
        drop_last=False
    )
    
    test_batch_sampler = BatchSampler(
        len(test_dataset), 
        config['training']['batch_size'], 
        test_dataset.max_sequence_length,
        shuffle=False,
        drop_last=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=train_batch_sampler,
        # Cannot use batch_size with batch_sampler
        # Cannot use shuffle with batch_sampler
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler
    )
    
    return train_loader, val_loader, test_loader