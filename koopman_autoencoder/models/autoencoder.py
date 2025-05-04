import torch
from torch import nn
from cnn import ConvEncoder, ConvDecoder


class KoopmanOperator(nn.Module):
    def __init__(self, latent_dim):
        """
        Koopman operator for linear dynamics in latent space.

        Parameters:
            latent_dim: int
                Dimensionality of the latent space.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.koopman_operator = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, z):
        """
        Apply Koopman operator to predict the next state.

        Parameters:
            z: torch.Tensor
                Latent representation of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor:
                Residual change in latent space.
        """
        return self.koopman_operator(z)


class KoopmanAutoencoder(nn.Module):
    def __init__(self, input_channels=6, height=64, width=64, latent_dim=32, 
                 hidden_dims=[64, 128, 64], block_size=2, kernel_size=3, **conv_kwargs):
        """
        Koopman Autoencoder for learning dynamical systems in latent space.

        Parameters:
            input_channels: int
                Number of input channels in the data.
            height: int
                Height of the input data.
            width: int
                Width of the input data.
            latent_dim: int
                Dimensionality of the latent space.
            hidden_dims: list of int
                List of hidden dimensions for encoder/decoder layers.
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
            C=2*input_channels, H=height, W=width, latent_dim=latent_dim, 
            hiddens=hidden_dims, block_size=block_size, kernel_size=kernel_size, 
            **conv_kwargs
        )

        # Initialize Decoder
        self.decoder = ConvDecoder(
            C=input_channels, H=height, W=width, latent_dim=latent_dim, 
            hiddens=hidden_dims, block_size=block_size, kernel_size=kernel_size, 
            **conv_kwargs
        )

        # Initialize Koopman Operator
        self.koopman_operator = KoopmanOperator(latent_dim)

    def encode(self, x):
        """
        Encode the input data into the latent space.

        Parameters:
            x: torch.Tensor
                Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Latent representation.
        """

        # Collapse the sequence and batch dimensions
        batch_size, sequence_length, channels, height, width = x.shape
        x = x.view(batch_size, sequence_length * channels, height, width)  # Merge batch and sequence dims
        z = self.encoder(x)  # Pass through encoder
        return z.view(batch_size, *z.shape[1:])  # Restore sequence dim

    def decode(self, z):
        """
        Decode the latent representation back to the input space.

        Parameters:
            z: torch.Tensor
                Latent representation of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed input.
        """
        return self.decoder(z)

    def predict_latent(self, z):
        """
        Predict the next state in latent space.

        Parameters:
            z: torch.Tensor
                Current latent state.

        Returns:
            torch.Tensor: Predicted next latent state.
        """
        return z + self.koopman_operator(z)

    def forward(self, x, seq_length=1):
        """
        Forward pass through the autoencoder with Koopman prediction.

        Parameters:
            x: torch.Tensor
                Input tensor of shape (batch_size, channels, height, width).
            seq_length: int
                Sequence length for predictions.

        Returns:
            tuple: (reconstructed input, predictions, latent predictions, latent differences)
        """
        # Encode the input
        z = self.encode(x)
        z_pred = z
        z_preds = [z]

        # Roll out predictions for the given sequence length
        for _ in range(seq_length):
            z_pred = self.predict_latent(z_pred)
            z_preds.append(z_pred)

        # Decode predictions
        x_recon = self.decode(z_preds[0])  # Reconstruction from initial z
        x_preds = torch.stack([self.decode(z_step) for z_step in z_preds[1:]], dim=1)

        # Compute latent prediction differences
        latent_pred_differences = torch.stack([
            z_preds[t + 1] - self.predict_latent(z_preds[t])
            for t in range(seq_length)
        ], dim=1)

        return x_recon, x_preds, z_preds, latent_pred_differences
    