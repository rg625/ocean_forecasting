import torch
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length target sequences without padding.

    Parameters:
        batch: list of tuples
            Each tuple contains (input_sequence, target_sequence, target_length).

    Returns:
        tuple: (input_batch, target_batch, target_lengths)
            Where input_batch and target_batch are lists of tensors.
    """
    input_seqs, target_seqs, target_lengths = zip(*batch)

    # Return input sequences, target sequences, and target lengths as lists
    return list(input_seqs), list(target_seqs), list(target_lengths)