import torch
from torch import nn, Tensor, device
from tensordict import TensorDict
from einops import reduce, rearrange
from torchvision.transforms import GaussianBlur


class KoopmanLoss(nn.Module):
    """
    Custom loss function for Koopman-based models.

    Args:
        alpha (float): Weight for the prediction loss term.
        beta (float): Weight for the latent loss term.
    """

    def __init__(self, alpha=1.0, beta=0.1, weighting_type="cosine", sigma_blur=None):
        super(KoopmanLoss, self).__init__()
        self.alpha = alpha  # weight for prediction loss
        self.beta = beta  # weight for latent loss
        self.weighting_type = weighting_type  # weighting schedule for rollout loss

        if sigma_blur is not None:
            # Use a fixed kernel size; ensure it fits sigma
            self.gaussian_blur = GaussianBlur(kernel_size=5, sigma=sigma_blur)
        else:
            self.gaussian_blur = None

    def blur(self, tensor: Tensor):
        if self.gaussian_blur is None:
            return tensor

        if tensor.ndim == 5:  # [B, T, C, H, W]
            B, T, C, H, W = tensor.shape
            tensor = rearrange(tensor, "b t c h w -> (b t) c h w")
            tensor = self.gaussian_blur(tensor)
            return rearrange(tensor, "(b t) c h w -> b t c h w", b=B, t=T)

        elif tensor.ndim == 4:  # [B, C, H, W]
            return self.gaussian_blur(tensor)

        else:
            return tensor  # Do nothing for other shapes

    def recon_loss(self, x: TensorDict, y: TensorDict):
        """
        Compute the reconstruction loss per variable.

        Args:
            x (TensorDict): Predicted TensorDict.
            y (TensorDict): Ground truth TensorDict.

        Returns:
            dict: Reconstruction loss per variable.
        """
        assert x.batch_size == y.batch_size, "Mismatch in batch size."
        loss_dict = {}
        for key in x.keys():
            diff = x[key] - self.blur(y[key])
            loss_per_sample = reduce(diff**2, "b ... -> b", "mean")
            loss_dict[key] = reduce(loss_per_sample, "b ->", "mean")
        return loss_dict

    def rollout_loss(self, x: TensorDict, y: TensorDict):
        """
        Compute the rollout loss per variable, weighted over sequence steps.

        Args:
            x (TensorDict): Predicted TensorDict of shape [batch, N, ...].
            y (TensorDict): Ground truth TensorDict of same shape, with "seq_length" key.

        Returns:
            dict: Rollout loss per variable.
        """
        assert x.batch_size == y.batch_size, "Mismatch in batch size."

        # Get sequence length and device
        seq_len = y["seq_length"][0]

        # Compute step weights
        weights = self.weighting(
            timesteps=seq_len.item(), device=seq_len.device
        )  # shape [N]

        x_filtered = x.exclude("seq_length")
        y_filtered = y.exclude("seq_length")
        losses = {}

        for key in x_filtered.keys():
            diff = x_filtered[key] - self.blur(y_filtered[key])  # [B, N, ...]
            squared = diff**2
            per_step_loss = reduce(squared, "b n ... -> b n", "mean")  # [B, N]
            weighted_loss = per_step_loss * weights[None, :]  # [B, N]
            loss = reduce(weighted_loss, "b n ->", "mean")  # scalar
            losses[key] = loss

        return losses

    def weighting(self, timesteps: int, device: device):
        """
        Compute weights for rollout loss per step.

        Args:
            timesteps (int): Rollout sequence length.
            device (torch.device): Device to place the tensor on.
        Returns:
            Tensor: Weight vector of shape [timesteps].
        """
        idx = torch.arange(end=timesteps, device=device)

        if self.weighting_type == "cosine":
            # Prevent division by zero when timesteps is 1
            if timesteps > 1:
                weights = 0.5 * (1 + torch.cos(torch.pi * idx / (timesteps - 1)))
            else:
                weights = torch.tensor([1.0], device=device)

            weights = weights / torch.max(
                weights.sum(), torch.tensor(1e-8, device=device)
            )  # Avoid NaN by clamping the sum

        elif self.weighting_type == "uniform":
            weights = torch.ones(timesteps, device=device) / timesteps
        else:
            raise ValueError(f"Unknown weighting type: {self.weighting_type}")

        return weights

    @staticmethod
    def kl(mu: Tensor, var: Tensor):
        return -0.5 * reduce(1.0 + torch.log(var) - mu**2 - var, "b t -> t", "mean")

    def kl_divergence(self, latent_pred: Tensor):
        """
        Compute KL divergence between a Gaussian N(μ, σ²) (estimated from latent_diff)
        and a standard Gaussian N(0, 1).

        Args:
            latent_pred (Tensor): Tensor of shape [batch, time, latent_dim].

        Returns:
            Tensor: Mean KL divergence over the batch.
        """
        mu = reduce(latent_pred, "b t d -> b t", "mean")
        var = reduce((latent_pred - mu[:, :, None]) ** 2, "b t d -> b t", "mean") + 1e-6

        kl_loss = self.kl(mu=mu, var=var)
        return reduce(kl_loss, "b ->", "mean")

    @staticmethod
    def re(input: Tensor, predicted: Tensor):
        diff = (input - predicted).squeeze(-1)  # now shape: [b, t]
        return reduce(diff**2, "b t -> b", "mean")

    def re_loss(self, input: Tensor, predicted: Tensor):
        re_loss = self.re(input=input, predicted=predicted)
        return reduce(re_loss, "b ->", "mean")

    def forward(self, x_recon, x_preds, latent_pred, x_true, x_future, reynolds):
        """
        Compute the Koopman loss.

        Args:
            x_recon (TensorDict): Reconstructed input of shape [batch, channels, height, width].
            x_preds (TensorDict): Rollout predictions of shape [batch, N, channels, height, width].
            latent_pred (torch.Tensor): Latent space predictions.
            x_true (TensorDict): Ground truth input TensorDict.
            x_future (TensorDict): Ground truth future states TensorDict.
            reynolds (torch.Tensor): Reynold's Number estimates.

        Returns:
            dict: Combined total loss, reconstruction loss (per variable), prediction loss (per variable), and latent loss.
        """
        # Reconstruction loss (per variable)
        recon_loss_per_variable = self.recon_loss(x_recon, x_true)
        # Prediction loss (per variable)
        pred_loss_per_variable = self.rollout_loss(x_preds, x_future)
        # KL loss for latent space
        latent_loss = self.kl_divergence(latent_pred)
        # Reynolds Number estimation
        re_loss = (
            self.re_loss(reynolds, x_future["Re"]) if reynolds is not None else None
        )

        # Combine total losses (only total_loss will backpropagate)
        total_loss = (
            sum(recon_loss_per_variable.values())  # Tensors sum to a tensor
            + self.alpha * sum(pred_loss_per_variable.values())
            + self.beta * latent_loss
        )

        return {
            "total_loss": total_loss,  # This will backpropagate
            "latent_loss": latent_loss.detach().item(),
            "re_loss": re_loss,
            "recon_loss": {
                key: value.detach().item()
                for key, value in recon_loss_per_variable.items()
            },
            "pred_loss": {
                key: value.detach().item()
                for key, value in pred_loss_per_variable.items()
            },
        }
