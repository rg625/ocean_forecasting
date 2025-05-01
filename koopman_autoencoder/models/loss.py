import torch
from torch import nn
import torch.nn.functional as F

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
            pred_loss += F.mse_loss(x_preds[:, t], x_future[:, t])
            latent_loss += F.mse_loss(latent_pred_differences[t], torch.zeros_like(latent_pred_differences[t]))

        pred_loss /= self.rollout_steps
        latent_loss /= self.rollout_steps

        total_loss = recon_loss + self.alpha * pred_loss + self.beta * latent_loss
        return total_loss, recon_loss, pred_loss, latent_loss
    