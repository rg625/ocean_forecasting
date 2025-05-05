import torch
from torch import nn


class KoopmanLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        """
        Custom loss function for Koopman-based models.

        Args:
            alpha (float): Weight for the prediction loss term.
            beta (float): Weight for the latent loss term.
        """
        super(KoopmanLoss, self).__init__()
        self.alpha = alpha  # weight for prediction loss
        self.beta = beta  # weight for latent loss

    @staticmethod
    def recon_loss(x, y):
        """
        Explicit implementation of Mean Squared Error (MSE) loss for reconstruction.

        Args:
            x (torch.Tensor): Predicted tensor.
            y (torch.Tensor): Ground truth tensor.

        Returns:
            torch.Tensor: MSE loss.
        """
        assert x.shape == y.shape, "x_recon and x_true must have the same shape"
        diff = (x - y).flatten(start_dim=1)  # Flatten all non-batch dimensions
        return (torch.square(diff).sum(dim=-1)).mean()  # Mean over all batches

    @staticmethod
    def rollout_loss(x, y):
        """
        Explicit implementation of Mean Squared Error (MSE) loss for rollouts.

        Args:
            x (torch.Tensor): Predicted tensor of shape [batch, N, ...].
            y (torch.Tensor): Ground truth tensor of shape [batch, N, ...].

        Returns:
            torch.Tensor: MSE loss averaged over rollout steps.
        """
        assert x.shape == y.shape, "x_preds and x_future must have the same shape"
        diff = (x - y).flatten(
            start_dim=2
        )  # Flatten all non-batch and non-rollout dimensions
        per_step_loss = (torch.square(diff).sum(dim=-1)).mean(dim=0)  # Mean over batch
        return per_step_loss.mean()  # Average over rollout steps

    def forward(self, x_recon, x_preds, latent_pred_differences, x_true, x_future):
        """
        Compute the Koopman loss.

        Args:
            x_recon (torch.Tensor): Reconstructed input of shape [batch, channels, height, width].
            x_preds (torch.Tensor): Rollout predictions of shape [batch, N, channels, height, width].
            latent_pred_differences (torch.Tensor): Differences in latent space predictions, same shape as x_preds.
            x_true (torch.Tensor): Ground truth input of shape [batch, channels, height, width].
            x_future (torch.Tensor): Ground truth future states of shape [batch, N, channels, height, width].

        Returns:
            total_loss (torch.Tensor): Combined total loss.
            recon_loss (torch.Tensor): Reconstruction loss.
            pred_loss (torch.Tensor): Prediction loss.
            latent_loss (torch.Tensor): Latent consistency loss.
        """
        # Reconstruction loss
        recon_loss = self.recon_loss(x_recon, x_true)

        # Prediction loss (averaged over the rollout steps)
        pred_loss = self.rollout_loss(x_preds, x_future)

        # Latent consistency loss (averaged over the rollout steps)
        latent_loss = self.rollout_loss(
            latent_pred_differences, torch.zeros_like(latent_pred_differences)
        )

        # Combine the losses
        total_loss = recon_loss + self.alpha * pred_loss + self.beta * latent_loss

        return total_loss, recon_loss, pred_loss, latent_loss
