import argparse
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from omegaconf import OmegaConf

from models.cnn import TransformerConfig
from models.autoencoder import KoopmanAutoencoder
from models.loss import KoopmanLoss
from models.lr_schedule import CosineWarmup
from models.dataloader import create_ddp_dataloaders
from models.trainer import Trainer
from models.metrics import Metric
from models.utils import (
    load_checkpoint,
    get_dataset_class_and_kwargs,
    load_datasets,
)


def main_worker(rank, world_size, cfg):
    # Setup DDP
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Initialize W&B only on rank 0
    if rank == 0:
        import os
        if "WANDB_API_KEY" not in os.environ:
            wandb.login(key="your_api_key_here")  # or raise an error explicitly
        wandb.init(
            project="koopman-autoencoder",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Prepare output directory only on rank 0
    output_dir = Path(cfg.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets and dataloaders
    dataset_class, dataset_kwargs = get_dataset_class_and_kwargs(cfg)
    train_dataset, val_dataset, test_dataset = load_datasets(
        cfg, dataset_class, dataset_kwargs
    )

    train_loader, val_loader, test_loader = create_ddp_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        config=cfg,
        rank=rank,
        world_size=world_size,
    )

    # Initialize model
    model = KoopmanAutoencoder(
        input_frames=cfg.data.input_sequence_length,
        input_channels=cfg.model.input_channels,
        height=cfg.model.height,
        width=cfg.model.width,
        latent_dim=cfg.model.latent_dim,
        hidden_dims=cfg.model.hidden_dims,
        use_checkpoint=cfg.training.use_checkpoint,
        transformer_config=TransformerConfig(**cfg.model.transformer),
        predict_re=cfg.model.predict_re,
        **cfg.model.conv_kwargs,
    ).to(device)

    model = DDP(model, device_ids=[rank])

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr_scheduler.lr)
    criterion = KoopmanLoss(
        alpha=cfg.loss.alpha,
        beta=cfg.loss.beta,
        re_weight=cfg.loss.re_weight,
        weighting_type=cfg.loss.weighting_type,
        sigma_blur=cfg.loss.sigma_blur,
    )

    eval_metrics = Metric(mode=cfg.metric.type, variable_mode=cfg.metric.variable_mode)

    lr_scheduler = CosineWarmup(
        optimizer=optimizer,
        warmup=cfg.lr_scheduler.warmup,
        decay=cfg.lr_scheduler.decay,
        final_lr=cfg.lr_scheduler.final_lr,
    )

    if rank == 0:
        wandb.watch(model.module, log="all")

    start_epoch = 0
    history = {}

    if cfg.ckpt is not None:
        if rank == 0:
            print(f"Loading from checkpoint: {cfg.ckpt}")
        model, optimizer, history, start_epoch = load_checkpoint(
            checkpoint_path=cfg.ckpt, model=model, optimizer=optimizer
        )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        eval_metrics=eval_metrics,
        lr_scheduler=lr_scheduler,
        device=device,
        num_epochs=cfg.training.num_epochs,
        patience=cfg.training.patience,
        output_dir=output_dir if rank == 0 else None,
        start_epoch=start_epoch,
        log_epoch=cfg.log_epoch,
    )

    history = trainer.run_training_loop()

    if rank == 0:
        test_metrics = trainer.evaluate(test_loader, mode="test")
        print("\nTest Set Results:")
        print(f"Loss: {test_metrics['total_loss']:.4f}")
        print(f"Reconstruction Loss: {test_metrics['recon_loss']:.4f}")
        print(f"Prediction Loss: {test_metrics['pred_loss']:.4f}")

        torch.save(
            {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "means": train_dataset.mins,
                "stds": train_dataset.maxs,
                "variables": train_dataset.variables,
                "test_metrics": test_metrics,
                "history": history,
            },
            output_dir / "final_model.pth",
        )
        wandb.save(str(output_dir / "final_model.pth"))

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Koopman Autoencoder with DDP")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    # Load YAML config using OmegaConf
    cfg = OmegaConf.load(args.config)

    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, cfg), nprocs=world_size, join=True)
