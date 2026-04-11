"""
Main Training Script
Entry point for training the multimodal deepfake detector
"""

import argparse
import torch
import torch.nn as nn
import logging
from pathlib import Path

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataloader import create_dataloaders
from src.models.multimodal_detector import MultimodalDeepfakeDetector
from src.training.losses import get_loss_function
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Multimodal Deepfake Detector"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--train-manifest",
        type=str,
        default=None,
        help="Path to training manifest (overrides config)"
    )

    parser.add_argument(
        "--val-manifest",
        type=str,
        default=None,
        help="Path to validation manifest (overrides config)"
    )

    parser.add_argument(
        "--test-manifest",
        type=str,
        default=None,
        help="Path to test manifest (overrides config)"
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Dataset root directory (overrides config)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cuda/cpu, overrides auto-detection)"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of data loading workers (overrides config)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Override config with command line arguments
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
        config["evaluation"]["batch_size"] = args.batch_size

    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers

    # Set random seed
    set_seed(config["data"]["seed"])

    # Set device
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")

    # Create data paths
    train_manifest = args.train_manifest or "data/manifests/train.csv"
    val_manifest = args.val_manifest or "data/manifests/val.csv"
    test_manifest = args.test_manifest or "data/manifests/test.csv"
    dataset_root = args.dataset_root or config["data"]["dataset_path"]

    # Create dataloaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        test_manifest=test_manifest,
        dataset_root=dataset_root,
        config=config,
        num_workers=config["data"]["num_workers"]
    )

    # Create model
    logger.info("Creating model...")
    model = MultimodalDeepfakeDetector(config)

    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        logger.info(f"Resuming from epoch {start_epoch}")

    # Create loss function
    criterion = get_loss_function(config)

    # Create optimizer
    optimizer_config = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"],
        betas=optimizer_config.get("betas", [0.9, 0.999]),
        eps=optimizer_config.get("eps", 1e-8)
    )

    # Load optimizer state if resuming
    if args.checkpoint is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Create scheduler
    scheduler_config = config["training"]["scheduler"]
    if scheduler_config["name"] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config["mode"],
            factor=scheduler_config["factor"],
            patience=scheduler_config["patience"],
            min_lr=scheduler_config["min_lr"],
            verbose=scheduler_config.get("verbose", True)
        )
    elif scheduler_config["name"] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["num_epochs"],
            eta_min=scheduler_config.get("min_lr", 1e-6)
        )
    else:
        scheduler = None

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )

    # Start training
    logger.info("="*60)
    logger.info("Starting training")
    logger.info("="*60)

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info("="*60)
    logger.info("Training completed!")
    logger.info("="*60)

    # Print best checkpoint location
    checkpoint_dir = Path(config["training"]["checkpoint"]["save_dir"])
    best_checkpoint = checkpoint_dir / "best.pth"

    if best_checkpoint.exists():
        logger.info(f"\nBest model saved at: {best_checkpoint}")
        logger.info(f"Use this checkpoint for evaluation and inference")


if __name__ == "__main__":
    main()
