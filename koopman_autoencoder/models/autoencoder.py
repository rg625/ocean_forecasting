import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KoopmanAutoencoder(nn.Module):
    def __init__(self, input_channels=2, latent_dim=32, hidden_dims=[64, 128, 64], spatial_dim=8):
        super(KoopmanAutoencoder, self).__init__()
        self.spatial_dim = spatial_dim

        # Encoder
        encoder_layers = []
        in_channels = input_channels

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, padding_mode='circular'),
                nn.ReLU(),
                nn.BatchNorm2d(hidden_dim),
                nn.MaxPool2d(2)
            ])
            in_channels = hidden_dim

        encoder_layers.append(nn.Conv2d(in_channels, latent_dim, kernel_size=3, padding=1, padding_mode='circular'))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        hidden_dims.reverse()
        in_channels = latent_dim

        for hidden_dim in hidden_dims:
            decoder_layers.extend([
                nn.ConvTranspose2d(in_channels, hidden_dim, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(hidden_dim),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode='circular')
            ])
            in_channels = hidden_dim

        decoder_layers.append(nn.Conv2d(in_channels, input_channels, kernel_size=3, padding=1, padding_mode='circular'))
        self.decoder = nn.Sequential(*decoder_layers)

        # Koopman operator (residual prediction version)
        self.koopman_operator = nn.Linear(latent_dim * spatial_dim * spatial_dim, latent_dim * spatial_dim * spatial_dim, bias=False)

    def encode(self, x):
        batch_size = x.size(0)
        encoded = self.encoder(x)
        return encoded.view(batch_size, -1)

    def decode(self, z):
        batch_size = z.size(0)
        z = z.view(batch_size, -1, self.spatial_dim, self.spatial_dim)
        return self.decoder(z)

    def forward(self, x, rollout_steps=1):
        z = self.encode(x)
        z_pred = z
        z_preds = [z]

        for _ in range(rollout_steps):
            z_pred = z_pred + self.koopman_operator(z_pred)  # residual prediction
            z_preds.append(z_pred)

        x_recon = self.decode(z_preds[0])  # reconstruction from initial z
        x_preds = [self.decode(z_step) for z_step in z_preds[1:]]

        # Compute latent losses directly here
        latent_pred_differences = [
            z_preds[t + 1] - (z_preds[t] + self.koopman_operator(z_preds[t]))
            for t in range(rollout_steps)
        ]

        return x_recon, x_preds, z_preds, latent_pred_differences

    def predict_next(self, x, steps=1):
        z = self.encode(x)
        preds = []

        for _ in range(steps):
            z = z + self.koopman_operator(z)
            preds.append(self.decode(z))

        return preds
