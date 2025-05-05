import torch
from torch import nn


class KoopmanLoss(nn.Module):
    """
    Custom loss function for Koopman-based models.

    Args:
        alpha (float): Weight for the prediction loss term.
        beta (float): Weight for the latent loss term.
    """

    def __init__(self, alpha=1.0, beta=0.1):
        super(KoopmanLoss, self).__init__()
        self.alpha = alpha  # weight for prediction loss
        self.beta = beta  # weight for latent loss

    @staticmethod
    def recon_loss(x, y):
        """
        Compute the reconstruction loss per variable.

        Args:
            x (TensorDict): Predicted TensorDict.
            y (TensorDict): Ground truth TensorDict.

        Returns:
            dict: Reconstruction loss per variable.
        """
        assert x.batch_size == y.batch_size, "Mismatch in batch size."
        losses = {}
        for key in x.keys():
            diff = (x[key] - y[key]).flatten(
                start_dim=1
            )  # Flatten all non-batch dimensions
            losses[key] = torch.square(diff).mean()  # Keep as tensor
        return losses

    @staticmethod
    def rollout_loss(x, y):
        """
        Compute the rollout loss per variable, ignoring the `seq_length` key.

        Args:
            x (TensorDict): Predicted TensorDict of shape [batch, N, ...].
            y (TensorDict): Ground truth TensorDict of shape [batch, N, ...].

        Returns:
            dict: Rollout loss per variable.
        """
        assert x.batch_size == y.batch_size, "Mismatch in batch size."
        x_filtered = x.exclude("seq_length")
        y_filtered = y.exclude("seq_length")
        losses = {}
        for key in x_filtered.keys():
            diff = (x_filtered[key] - y_filtered[key]).flatten(
                start_dim=2
            )  # Flatten non-batch and non-rollout dims
            per_step_loss = torch.square(diff).mean()  # Average over batches and steps
            losses[key] = per_step_loss  # Keep as tensor
        return losses

    def forward(self, x_recon, x_preds, latent_pred_differences, x_true, x_future):
        """
        Compute the Koopman loss.

        Args:
            x_recon (TensorDict): Reconstructed input of shape [batch, channels, height, width].
            x_preds (TensorDict): Rollout predictions of shape [batch, N, channels, height, width].
            latent_pred_differences (torch.Tensor): Differences in latent space predictions.
            x_true (TensorDict): Ground truth input TensorDict.
            x_future (TensorDict): Ground truth future states TensorDict.

        Returns:
            dict: Combined total loss, reconstruction loss (per variable), prediction loss (per variable), and latent loss.
        """
        # Reconstruction loss (per variable)
        recon_loss_per_variable = self.recon_loss(x_recon, x_true)
        # Prediction loss (per variable)
        pred_loss_per_variable = self.rollout_loss(x_preds, x_future)
        latent_loss = torch.mean(
            torch.square(latent_pred_differences.flatten(start_dim=2))
        )

        # Combine total losses (only total_loss will backpropagate)
        total_loss = (
            sum(recon_loss_per_variable.values())  # Tensors sum to a tensor
            + self.alpha * sum(pred_loss_per_variable.values())
            # + self.beta * latent_loss
        )

        return {
            "total_loss": total_loss,  # This will backpropagate
            "latent_loss": latent_loss.detach().item(),
            "recon_loss": {
                key: value.detach().item()
                for key, value in recon_loss_per_variable.items()
            },
            "pred_loss": {
                key: value.detach().item()
                for key, value in pred_loss_per_variable.items()
            },
        }
