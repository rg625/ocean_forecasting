import torch
from torch import nn
from cnn import ConvEncoder, ConvDecoder

class KoopmanOperator(nn.Module):
    def __init__(self, latent_dim, spatial_dim):
        """
        Koopman operator block for residual predictions in latent space.

        Parameters:
            latent_dim: int
                Dimensionality of the latent space.
            spatial_dim: int
                Spatial dimension of the latent representation.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_dim = spatial_dim
        self.flattened_dim = latent_dim * spatial_dim * spatial_dim
        self.koopman_operator = nn.Linear(self.flattened_dim, self.flattened_dim, bias=False)

    def forward(self, z):
        """
        Apply Koopman operator for residual prediction.

        Parameters:
            z: torch.Tensor
                Latent representation of shape (batch_size, latent_dim, spatial_dim, spatial_dim).

        Returns:
            torch.Tensor:
                Flattened predicted latent representation.
        """
        # Ensure the input is flattened
        batch_size = z.size(0)
        z = z.view(batch_size, -1)  # Flatten to (batch_size, latent_dim * spatial_dim * spatial_dim)
        return self.koopman_operator(z)


class KoopmanOperator(nn.Module):
    def __init__(self, latent_dim, spatial_dim):
        """
        Koopman operator block for residual predictions in latent space.

        Parameters:
            latent_dim: int
                Dimensionality of the latent space.
            spatial_dim: int
                Spatial dimension of the latent representation.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_dim = spatial_dim
        self.flattened_dim = latent_dim * spatial_dim * spatial_dim
        self.koopman_operator = nn.Linear(self.flattened_dim, self.flattened_dim, bias=False)

    def forward(self, z):
        """
        Apply Koopman operator for residual prediction.

        Parameters:
            z: torch.Tensor
                Latent representation of shape (batch_size, latent_dim, spatial_dim, spatial_dim).

        Returns:
            torch.Tensor:
                Flattened predicted latent representation.
        """
        # Ensure the input is flattened
        batch_size = z.size(0)
        z = z.view(batch_size, -1)  # Flatten to (batch_size, latent_dim * spatial_dim * spatial_dim)
        return self.koopman_operator(z)


class KoopmanAutoencoder(nn.Module):
    def __init__(self, input_channels=2, latent_dim=32, hidden_dims=[64, 128, 64], 
                 height=64, width=64, block_size=2, kernel_size=3, **conv_kwargs):
        """
        Koopman Autoencoder built using ConvEncoder, ConvDecoder, and KoopmanOperator blocks.

        Parameters:
            input_channels: int
                Number of input channels in the data.
            latent_dim: int
                Dimensionality of the latent space.
            hidden_dims: list of int
                List of hidden dimensions for encoder/decoder layers.
            height: int
                Height of the input data.
            width: int
                Width of the input data.
            block_size: int
                Number of convolutional layers in a block.
            kernel_size: int
                Size of the convolution kernel.
            conv_kwargs: dict
                Additional arguments for convolutional layers.
        """
        super().__init__()

        # Initialize Encoder
        self.encoder = ConvEncoder(
            C=input_channels, H=height, W=width, D=latent_dim, hiddens=hidden_dims,
            block_size=block_size, kernel_size=kernel_size, **conv_kwargs
        )

        # Compute spatial dimensions after encoding
        self.spatial_dim = height // (2 ** len(hidden_dims))

        if self.spatial_dim <= 0 or width % (2 ** len(hidden_dims)) != 0:
            raise ValueError("Input dimensions must be divisible by 2 ** len(hidden_dims).")

        # Initialize Decoder
        self.decoder = ConvDecoder(
            C=input_channels, H=height, W=width, D=latent_dim, hiddens=hidden_dims,
            block_size=block_size, kernel_size=kernel_size, **conv_kwargs
        )

        # Initialize Koopman Operator
        self.koopman_operator = KoopmanOperator(latent_dim, self.spatial_dim)

    def encode(self, x):
        """
        Encode the input data into the latent space.

        Parameters:
            x: torch.Tensor
                Input tensor.

        Returns:
            torch.Tensor: Flattened latent representation.
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode the latent representation back to the input space.

        Parameters:
            z: torch.Tensor
                Latent representation.

        Returns:
            torch.Tensor: Reconstructed input.
        """
        return self.decoder(z)

    def forward(self, x, seq_length):
        """
        Forward pass through the autoencoder with Koopman prediction.

        Parameters:
            x: torch.Tensor
                Input tensor of shape (batch_size, input_sequence_length, variables, height, width).
            seq_length: int
                Sequence length for predictions (applicable for all samples in the batch).

        Returns:
            tuple: (reconstructed input, predictions, latent predictions, latent differences)
        """
        # Encode the input
        z = self.encode(x)  # Shape: (batch_size, latent_dim, spatial_dim, spatial_dim)
        z_pred = z
        z_preds = [z]

        # Roll out predictions for the given sequence length
        for _ in range(seq_length):
            z_pred = z_pred + self.koopman_operator(z_pred)
            z_pred = z_pred.view(z_pred.size(0), self.koopman_operator.latent_dim, 
                                 self.koopman_operator.spatial_dim, self.koopman_operator.spatial_dim)
            z_preds.append(z_pred)

        # Decode predictions
        x_recon = self.decode(z_preds[0])  # Reconstruction from initial z
        x_preds = torch.stack([self.decode(z_step) for z_step in z_preds[1:]], dim=1)

        # Compute latent prediction differences
        latent_pred_differences = torch.stack([
            z_preds[t + 1] - (z_preds[t] + self.koopman_operator(z_preds[t].view(z_preds[t].size(0), -1)).view_as(z_preds[t]))
            for t in range(seq_length)
        ], dim=1)

        return x_recon, x_preds, z_preds, latent_pred_differences
    