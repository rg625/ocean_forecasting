import torch
from torch import nn
import torch.nn.functional as F

class KoopmanLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, rollout_steps=1):
        super(KoopmanLoss, self).__init__()
        self.alpha = alpha  # weight for prediction loss
        self.beta = beta    # weight for latent loss

    def forward(self, x_recon, x_preds, latent_pred_differences, x_true, x_future):
        recon_loss = F.mse_loss(x_recon, x_true)
        pred_loss = F.mse_loss(x_preds, x_future)
        latent_loss  = F.mse_loss(latent_pred_differences, torch.zeros_like(latent_pred_differences))

        pred_loss /= x_preds.size(1)
        latent_loss /= x_preds.size(1)

        total_loss = recon_loss + self.alpha * pred_loss + self.beta * latent_loss
        return total_loss, recon_loss, pred_loss, latent_loss
    