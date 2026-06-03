"""
Main Trainer Class
Handles training loop, validation, logging, and callbacks
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Dict, Optional

from .metrics import MetricsCalculator, AverageMeter
from .callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback
from ..utils.monitors import DataLoadingMonitor, TrainingProgressTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer for multimodal deepfake detection
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Dict,
        device: str = "cuda"
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: LR scheduler (optional)
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Training settings
        self.num_epochs = config["training"]["num_epochs"]
        self.gradient_clip = config["training"]["gradient_clip"]
        self.mixed_precision = config["training"]["mixed_precision"]
        self.log_interval = config["training"]["logging"]["log_every_n_steps"]

        # Initialize GradScaler for mixed precision
        # Use device-specific scaler based on hardware
        if self.mixed_precision:
            if device == "cuda":
                self.scaler = GradScaler("cuda")
            elif device == "cpu":
                self.scaler = GradScaler("cpu")
            else:  # mps or other devices
                self.scaler = None  # MPS doesn't support GradScaler yet
                self.mixed_precision = False  # Disable for MPS
                logger.warning(f"Mixed precision not supported on {device}, disabling")
        else:
            self.scaler = None

        # Initialize metrics calculator
        class_names = config.get("classes", [f"Class_{i}" for i in range(4)])
        self.metrics_calculator = MetricsCalculator(
            num_classes=config["model"]["classifier"]["num_classes"],
            class_names=class_names
        )

        # Initialize callbacks
        self._init_callbacks()

        # Initialize monitors
        self.data_monitor = DataLoadingMonitor(log_dir="logs/data_loading")
        self.progress_tracker = TrainingProgressTracker(log_dir="logs/training")

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0

        logger.info(f"Trainer initialized:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Mixed precision: {self.mixed_precision}")
        logger.info(f"  Total epochs: {self.num_epochs}")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")

    def _init_callbacks(self):
        """Initialize training callbacks"""
        # Early stopping
        if self.config["training"]["early_stopping"]["enabled"]:
            self.early_stopping = EarlyStopping(
                patience=self.config["training"]["early_stopping"]["patience"],
                min_delta=self.config["training"]["early_stopping"]["min_delta"],
                mode=self.config["training"]["early_stopping"]["mode"],
                verbose=True
            )
        else:
            self.early_stopping = None

        # Model checkpoint
        self.checkpoint = ModelCheckpoint(
            save_dir=self.config["training"]["checkpoint"]["save_dir"],
            metric_name=self.config["training"]["checkpoint"]["metric"],
            mode=self.config["training"]["checkpoint"]["mode"],
            save_best=self.config["training"]["checkpoint"]["save_best"],
            save_last=self.config["training"]["checkpoint"]["save_last"],
            verbose=True
        )

        # LR scheduler callback
        if self.scheduler is not None:
            self.lr_scheduler_callback = LRSchedulerCallback(
                scheduler=self.scheduler,
                metric_name=self.config["training"]["checkpoint"]["metric"],
                verbose=True
            )
        else:
            self.lr_scheduler_callback = None

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.metrics_calculator.reset()

        # Start monitoring
        self.data_monitor.start_epoch(self.current_epoch)

        loss_meter = AverageMeter("train_loss")
        acc_meter = AverageMeter("train_acc")

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}/{self.num_epochs}")

        for batch_idx, (videos, audios, labels, _) in enumerate(pbar):
            # Move to device
            videos = videos.to(self.device)
            audios = audios.to(self.device)
            labels = labels.to(self.device)

            # Forward pass with mixed precision
            if self.mixed_precision and self.scaler is not None:
                with autocast(device_type=self.device):
                    logits, _ = self.model(videos, audios)
                    loss = self.criterion(logits, labels)
            else:
                logits, _ = self.model(videos, audios)
                loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()

            if self.mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                self.optimizer.step()

            # Update metrics
            batch_acc = (torch.argmax(logits, dim=-1) == labels).float().mean().item()
            loss_meter.update(loss.item(), videos.size(0))
            acc_meter.update(batch_acc, videos.size(0))
            self.metrics_calculator.update(logits, labels)

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "acc": f"{acc_meter.avg:.4f}"
            })

            self.global_step += 1

        # Compute epoch metrics
        metrics = self.metrics_calculator.compute()
        metrics["train_loss"] = loss_meter.avg
        metrics["train_accuracy"] = acc_meter.avg

        # End monitoring and get report
        data_report = self.data_monitor.end_epoch(total_batches=len(self.train_loader))

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.metrics_calculator.reset()

        loss_meter = AverageMeter("val_loss")

        pbar = tqdm(self.val_loader, desc="Validation")

        for videos, audios, labels, _ in pbar:
            # Move to device
            videos = videos.to(self.device)
            audios = audios.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            if self.mixed_precision:
                with autocast():
                    logits, _ = self.model(videos, audios)
                    loss = self.criterion(logits, labels)
            else:
                logits, _ = self.model(videos, audios)
                loss = self.criterion(logits, labels)

            # Update metrics
            loss_meter.update(loss.item(), videos.size(0))
            self.metrics_calculator.update(logits, labels)

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        # Compute metrics
        metrics = self.metrics_calculator.compute()
        metrics["val_loss"] = loss_meter.avg

        # Rename metrics for clarity
        val_metrics = {}
        for key, value in metrics.items():
            if key != "confusion_matrix":
                val_metrics[f"val_{key}"] = value if not key.startswith("val_") else value
            else:
                val_metrics[key] = value

        return val_metrics

    def train(self):
        """
        Main training loop
        """
        logger.info("Starting training...")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch + 1

            # Train epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}

            # Log metrics with progress tracker
            self._log_metrics(all_metrics)
            self.progress_tracker.log_epoch(self.current_epoch, all_metrics)

            # Get monitored metric for callbacks
            monitored_metric = all_metrics.get(
                self.config["training"]["checkpoint"]["metric"],
                all_metrics.get("val_auc_roc", 0.0)
            )

            # Save checkpoint
            self.checkpoint(
                self.model,
                self.optimizer,
                self.current_epoch,
                monitored_metric,
                all_metrics
            )

            # Step LR scheduler
            if self.lr_scheduler_callback is not None:
                self.lr_scheduler_callback.step(all_metrics)

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(monitored_metric):
                    logger.info(f"Early stopping at epoch {self.current_epoch}")
                    break

        logger.info("Training completed!")

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics"""
        logger.info(f"\nEpoch {self.current_epoch}/{self.num_epochs}:")

        # Log main metrics
        for key in ["train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_auc_roc", "val_f1_macro"]:
            if key in metrics:
                logger.info(f"  {key}: {metrics[key]:.4f}")


def main():
    """Test trainer with dummy data"""
    import yaml
    from torch.utils.data import TensorDataset

    print("Testing Trainer...")

    # Load config
    config_path = "../../configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create dummy model
    from ..models.multimodal_detector import MultimodalDeepfakeDetector

    model = MultimodalDeepfakeDetector(config)

    # Create dummy data
    train_videos = torch.randn(100, 16, 3, 224, 224)
    train_audios = torch.randn(100, 1, 128, 94)
    train_labels = torch.randint(0, 4, (100,))

    val_videos = torch.randn(20, 16, 3, 224, 224)
    val_audios = torch.randn(20, 1, 128, 94)
    val_labels = torch.randint(0, 4, (20,))

    train_dataset = TensorDataset(train_videos, train_audios, train_labels)
    val_dataset = TensorDataset(val_videos, val_audios, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Create trainer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Modify config for testing
    config["training"]["num_epochs"] = 2
    config["training"]["early_stopping"]["enabled"] = False

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        config=config,
        device="cpu"
    )

    # Train for 1 epoch
    trainer.train()

    print("\n✓ Trainer test passed!")


if __name__ == "__main__":
    main()
