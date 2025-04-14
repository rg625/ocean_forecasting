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


class KoopmanLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, rollout_steps=1):
        super(KoopmanLoss, self).__init__()
        self.alpha = alpha  # weight for prediction loss
        self.beta = beta    # weight for latent loss
        self.rollout_steps = rollout_steps

    def forward(self, x_recon, x_preds, latent_pred_differences, x_true, x_future):
        recon_loss = F.mse_loss(x_recon, x_true)

        pred_loss = 0.0
        latent_loss = 0.0
        for t in range(self.rollout_steps):
            pred_loss += F.mse_loss(x_preds[t], x_future[:, t])
            latent_loss += F.mse_loss(latent_pred_differences[t], torch.zeros_like(latent_pred_differences[t]))

        pred_loss /= self.rollout_steps
        latent_loss /= self.rollout_steps

        total_loss = recon_loss + self.alpha * pred_loss + self.beta * latent_loss
        return total_loss, recon_loss, pred_loss, latent_loss

def train_koopman_autoencoder(model, train_loader, optimizer, criterion, device, num_epochs=100, rollout_steps=3):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x, x_future) in enumerate(train_loader):
            x = x.to(device)
            x_future = x_future.to(device)  # shape: [B, rollout_steps, C, H, W]

            optimizer.zero_grad()
            x_recon, x_preds, z_preds = model(x, rollout_steps=rollout_steps)
            loss, recon_loss, pred_loss, latent_loss = criterion(x_recon, x_preds, z_preds, x, x_future)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Recon: {recon_loss:.4f}, Pred: {pred_loss:.4f}, Latent: {latent_loss:.4f}')