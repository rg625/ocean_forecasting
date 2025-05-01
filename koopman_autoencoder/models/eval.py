import torch

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_pred_loss = 0
    total_latent_loss = 0

    with torch.no_grad():
        for input, target in dataloader:
            input, target = input.to(device), target.to(device)
            x_recon, x_preds, z_preds, latent_pred_differences = model(input, seq_length=target.size(1))
            loss, recon_loss, pred_loss, latent_loss = criterion(x_recon, x_preds, latent_pred_differences, input[:, -1], target)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_pred_loss += pred_loss.item()
            total_latent_loss += latent_loss.item()

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'pred_loss': total_pred_loss / n_batches,
        'latent_loss': total_latent_loss / n_batches
    }
