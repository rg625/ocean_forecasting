import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
from koopman_autoencoder import KoopmanAutoencoder, KoopmanLoss
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

class QGDataset(Dataset):
    def __init__(self, data_path, sequence_length=2, variables=None):
        self.data = xr.open_dataset(data_path)
        self.sequence_length = sequence_length

        if variables is None:
            variables = list(self.data.data_vars.keys())
        self.variables = variables

        self.stacked_data = torch.stack([
            torch.FloatTensor(self.data[var].values)
            for var in variables
        ], dim=1)

        self.means = self.stacked_data.mean(dim=(0, 2, 3))
        self.stds = self.stacked_data.std(dim=(0, 2, 3))
        self.stacked_data = (self.stacked_data - self.means[None, :, None, None]) / self.stds[None, :, None, None]

    def __len__(self):
        return len(self.stacked_data) - self.sequence_length + 1

    def __getitem__(self, idx):
        data_seq = self.stacked_data[idx:idx + self.sequence_length]
        return data_seq[0], data_seq[1:]

    def denormalize(self, x):
        device = x.device  # Get the device of the input tensor
        means = self.means.to(device)  # Move means to the same device as x
        stds = self.stds.to(device)    # Move stds to the same device as x
        return x * stds[None, :, None, None] + means[None, :, None, None]

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_pred_loss = 0
    total_latent_loss = 0

    with torch.no_grad():
        for x, x_next_seq in dataloader:
            x, x_next_seq = x.to(device), x_next_seq.to(device)
            x_recon, x_preds, z_preds, latent_pred_differences = model(x, rollout_steps=x_next_seq.size(1))
            loss, recon_loss, pred_loss, latent_loss = criterion(x_recon, x_preds, latent_pred_differences, x, x_next_seq)
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

def visualize_results(model, dataset, num_samples=4, device='cpu', output_dir=None):
    model.eval()
    with torch.no_grad():
        x, x_next_seq = next(iter(DataLoader(dataset, batch_size=num_samples, shuffle=True)))
        x, x_next_seq = x.to(device), x_next_seq.to(device)

        # Updated unpacking to handle additional return value
        x_recon, x_preds, _, _ = model(x, rollout_steps=x_next_seq.size(1))

        x = dataset.denormalize(x)
        x_next_seq = dataset.denormalize(x_next_seq)
        x_recon = dataset.denormalize(x_recon)
        x_preds = [dataset.denormalize(pred) for pred in x_preds]

        # Only visualize reconstruction and first prediction
        x_pred = x_preds[0]

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

        # Reconstruction plot
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
        for i in range(num_samples):
            for j, (title, data) in enumerate([
                ('Input', x[i, 0]),
                ('Reconstructed', x_recon[i, 0])
            ]):
                axes[i, j].imshow(data.cpu().numpy(), cmap='RdBu_r')
                axes[i, j].set_title(title)
                axes[i, j].axis('off')
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / 'reconstruction_comparison.png')
        plt.close()

        # Prediction plot
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
        for i in range(num_samples):
            for j, (title, data) in enumerate([
                ('True Next', x_next_seq[i, 0, 0]),
                ('Predicted Next', x_pred[i, 0])
            ]):
                axes[i, j].imshow(data.cpu().numpy(), cmap='RdBu_r')
                axes[i, j].set_title(title)
                axes[i, j].axis('off')
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / 'prediction_comparison.png')
        plt.close()

        # Error maps
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
        for i in range(num_samples):
            recon_error = (x_recon[i, 0] - x[i, 0]).cpu().numpy()
            pred_error = (x_pred[i, 0] - x_next_seq[i, 0, 0]).cpu().numpy()

            axes[i, 0].imshow(recon_error, cmap='RdBu_r')
            axes[i, 0].set_title('Reconstruction Error')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(pred_error, cmap='RdBu_r')
            axes[i, 1].set_title('Prediction Error')
            axes[i, 1].axis('off')
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / 'error_maps.png')
        plt.close()

def train_with_validation(model, train_loader, val_loader, optimizer, criterion, 
                         device, num_epochs=100, patience=100, output_dir=None):
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_recon_loss': [], 
               'val_recon_loss': [], 'train_pred_loss': [], 'val_pred_loss': []}

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    progress_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    for epoch in progress_bar:
        model.train()
        total_loss = 0
        for x, x_next_seq in train_loader:
            x, x_next_seq = x.to(device), x_next_seq.to(device)
            optimizer.zero_grad()
            x_recon, x_preds, z_preds, latent_pred_differences = model(x, rollout_steps=x_next_seq.size(1))
            loss, recon_loss, pred_loss, latent_loss = criterion(x_recon, x_preds, latent_pred_differences, x, x_next_seq)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_metrics = evaluate_model(model, train_loader, criterion, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_recon_loss'].append(train_metrics['recon_loss'])
        history['val_recon_loss'].append(val_metrics['recon_loss'])
        history['train_pred_loss'].append(train_metrics['pred_loss'])
        history['val_pred_loss'].append(val_metrics['pred_loss'])

        # Update the progress bar with the current metrics
        progress_bar.set_postfix({
            'Train Loss': f"{train_metrics['loss']:.4f}",
            'Val Loss': f"{val_metrics['loss']:.4f}",
            'Recon (Train)': f"{train_metrics['recon_loss']:.4f}",
            'Pred (Train)': f"{train_metrics['pred_loss']:.4f}",
            'Recon (Val)': f"{val_metrics['recon_loss']:.4f}",
            'Pred (Val)': f"{val_metrics['pred_loss']:.4f}"
        })

        # Early stopping mechanism
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            if output_dir:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'history': history
                }, output_dir / 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                progress_bar.write(f'Early stopping triggered after {epoch+1} epochs')
                break

    if output_dir:
        plt.figure(figsize=(12, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.savefig(output_dir / 'training_history.png')
        plt.close()

        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f)

    return history

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path('model_outputs') / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path('data/data/two_layer')
    train_dataset = QGDataset(data_dir / 'qg_train_data.nc', sequence_length=2)
    val_dataset = QGDataset(data_dir / 'qg_val_data.nc', sequence_length=2)
    test_dataset = QGDataset(data_dir / 'qg_test_data.nc', sequence_length=2)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = KoopmanAutoencoder(
        input_channels=len(train_dataset.variables),
        latent_dim=32,
        hidden_dims=[64, 128, 64]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = KoopmanLoss(alpha=1.0)

    history = train_with_validation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=100,
        patience=100,
        output_dir=output_dir
    )

    test_metrics = evaluate_model(model, test_loader, criterion, device)
    print("\nTest Set Results:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Reconstruction Loss: {test_metrics['recon_loss']:.4f}")
    print(f"Prediction Loss: {test_metrics['pred_loss']:.4f}")

    visualize_results(model, test_dataset, num_samples=4, device=device, 
                     output_dir=output_dir / 'visualizations')

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'means': train_dataset.means,
        'stds': train_dataset.stds,
        'variables': train_dataset.variables,
        'test_metrics': test_metrics,
        'history': history
    }, output_dir / 'final_model.pth')

if __name__ == '__main__':
    main()