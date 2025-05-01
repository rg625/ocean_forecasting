import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from autoencoder import KoopmanAutoencoder
from loss import KoopmanLoss
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from datetime import datetime
from tqdm import tqdm
import wandb  # Import Weights & Biases

from dataloder import QGDataset, create_dataloaders
from eval import evaluate_model
from visualization import visualize_results


def load_config(config_path):
    """
    Load the YAML configuration file for the model and training process.

    Parameters:
        config_path: str
            Path to the YAML configuration file.

    Returns:
        dict: Loaded configuration.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
        for input, target in train_loader:
            input, target = input.to(device), target.to(device)
            seq_length = target.size(1)
            optimizer.zero_grad()
            x_recon, x_preds, z_preds, latent_pred_differences = model(input, seq_length=seq_length)
            loss, recon_loss, pred_loss, latent_loss = criterion(x_recon, x_preds, latent_pred_differences, input[:, -1], target)
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

        # Log metrics to W&B
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'train_recon_loss': train_metrics['recon_loss'],
            'val_recon_loss': val_metrics['recon_loss'],
            'train_pred_loss': train_metrics['pred_loss'],
            'val_pred_loss': val_metrics['pred_loss']
        })

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

        with open(output_dir / 'training_history.yaml', 'w') as f:
            yaml.dump(history, f)

    return history


def main(config_path):
    config = load_config(config_path)

    # Initialize W&B
    wandb.init(project="koopman-autoencoder", config=config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(config['output_dir']) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(config['data']['data_dir'])
    train_dataset = QGDataset(data_dir / config['data']['train_file'], 
                             max_sequence_length=config['data']['max_sequence_length'])
    val_dataset = QGDataset(data_dir / config['data']['val_file'], 
                           max_sequence_length=config['data']['max_sequence_length'])
    test_dataset = QGDataset(data_dir / config['data']['test_file'], 
                            max_sequence_length=config['data']['max_sequence_length'])
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        config=config,
        )
    
    model = KoopmanAutoencoder(
        input_channels=config['model']['input_channels'],
        height=config['model']['height'],
        width=config['model']['width'],
        latent_dim=config['model']['latent_dim'],
        hidden_dims=config['model']['hidden_dims'],
        **config['model']['conv_kwargs']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = KoopmanLoss(alpha=config['loss']['alpha'])

    # Log model architecture to W&B
    wandb.watch(model, log="all")

    history = train_with_validation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=config['training']['num_epochs'],
        patience=config['training']['patience'],
        output_dir=output_dir
    )

    test_metrics = evaluate_model(model, test_loader, criterion, device)
    print("\nTest Set Results:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Reconstruction Loss: {test_metrics['recon_loss']:.4f}")
    print(f"Prediction Loss: {test_metrics['pred_loss']:.4f}")

    visualize_results(model, test_dataset, num_samples=4, device=device, 
                     output_dir=output_dir / 'visualizations')

    # Save final model and log it to W&B
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'means': train_dataset.mins,
        'stds': train_dataset.maxs,
        'variables': train_dataset.variables,
        'test_metrics': test_metrics,
        'history': history
    }, output_dir / 'final_model.pth')
    wandb.save(str(output_dir / 'final_model.pth'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train Koopman Autoencoder with configuration file")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    main(args.config)
    