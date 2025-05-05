import torch
from torch import nn
from tensordict import TensorDict


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
        Compute the reconstruction loss for each field in the TensorDict.

        Args:
            x (TensorDict): Predicted TensorDict.
            y (TensorDict): Ground truth TensorDict.

        Returns:
            TensorDict: Reconstruction loss for each field.
        """
        losses = {}
        for key in x.keys():
            assert x[key].shape == y[key].shape, f"Shape mismatch for key '{key}'"
            diff = (x[key] - y[key]).flatten(
                start_dim=1
            )  # Flatten all non-batch dimensions
            losses[key] = (torch.square(diff).sum(dim=-1)).mean()  # Mean over batches
        return TensorDict(losses, batch_size=[])

    @staticmethod
    def rollout_loss(x, y):
        """
        Compute the rollout loss for each field in the TensorDict.

        Args:
            x (TensorDict): Predicted TensorDict of shape [batch, N, ...].
            y (TensorDict): Ground truth TensorDict of shape [batch, N, ...].

        Returns:
            TensorDict: Rollout loss for each field.
        """
        losses = {}
        for key in x.keys():
            assert x[key].shape == y[key].shape, f"Shape mismatch for key '{key}'"
            diff = (x[key] - y[key]).flatten(
                start_dim=2
            )  # Flatten all non-batch and non-rollout dimensions
            per_step_loss = (
                torch.square(diff).sum(dim=-1).mean(dim=0)
            )  # Mean over batches
            losses[key] = per_step_loss.mean()  # Average over rollout steps
        return TensorDict(losses, batch_size=[])

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
            TensorDict: Combined total loss, reconstruction loss, prediction loss, and latent loss.
        """
        # Reconstruction loss
        recon_loss = self.recon_loss(x_recon, x_true)

        # Prediction loss (averaged over the rollout steps)
        pred_loss = self.rollout_loss(x_preds, x_future)

        # Latent consistency loss (averaged over the rollout steps)
        latent_loss_value = torch.mean(
            torch.square(latent_pred_differences.flatten(start_dim=2))
        )
        latent_loss = TensorDict({"latent": latent_loss_value}, batch_size=[])

        # Combine the losses
        total_loss_value = (
            sum(recon_loss.values())
            + self.alpha * sum(pred_loss.values())
            + self.beta * latent_loss["latent"]
        )
        total_loss = TensorDict({"total": total_loss_value}, batch_size=[])

        # Return all losses as TensorDicts
        return TensorDict(
            {
                "total_loss": total_loss,
                "recon_loss": recon_loss,
                "pred_loss": pred_loss,
                "latent_loss": latent_loss,
            },
            batch_size=[],
        )
