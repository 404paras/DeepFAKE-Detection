"""
Training Callbacks
Early stopping, learning rate scheduling, model checkpointing
"""

import torch
import os
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "max":
            self.is_better = lambda score, best: score > best + min_delta
        else:
            self.is_better = lambda score, best: score < best - min_delta

        logger.info(f"EarlyStopping initialized: patience={patience}, mode={mode}")

    def __call__(self, score: float) -> bool:
        """
        Args:
            score: Current validation metric value

        Returns:
            Whether to stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.is_better(score, self.best_score):
            # Improvement
            if self.verbose:
                logger.info(
                    f"Validation metric improved: {self.best_score:.4f} → {score:.4f}"
                )
            self.best_score = score
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"No improvement for {self.counter}/{self.patience} epochs "
                    f"(best: {self.best_score:.4f}, current: {score:.4f})"
                )

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"Early stopping triggered!")
                return True

        return False

    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class ModelCheckpoint:
    """
    Save model checkpoints based on validation metric
    """

    def __init__(
        self,
        save_dir: str,
        metric_name: str = "val_auc_roc",
        mode: str = "max",
        save_best: bool = True,
        save_last: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            metric_name: Metric name to monitor
            mode: 'max' or 'min'
            save_best: Whether to save best model
            save_last: Whether to save last model
            verbose: Whether to print messages
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.metric_name = metric_name
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.verbose = verbose

        self.best_score = None

        if mode == "max":
            self.is_better = lambda score, best: score > best
        else:
            self.is_better = lambda score, best: score < best

        logger.info(f"ModelCheckpoint initialized: save_dir={save_dir}, metric={metric_name}")

    def __call__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        score: float,
        metrics: dict
    ):
        """
        Save checkpoint if metric improved

        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            score: Current validation metric value
            metrics: Dictionary of all metrics
        """
        # Save last checkpoint
        if self.save_last:
            self._save_checkpoint(
                model, optimizer, epoch, score, metrics, "last.pth"
            )

        # Save best checkpoint
        if self.save_best:
            if self.best_score is None or self.is_better(score, self.best_score):
                if self.verbose:
                    if self.best_score is not None:
                        logger.info(
                            f"Saving best model: {self.metric_name} improved "
                            f"{self.best_score:.4f} → {score:.4f}"
                        )
                    else:
                        logger.info(f"Saving initial best model: {self.metric_name}={score:.4f}")

                self.best_score = score
                self._save_checkpoint(
                    model, optimizer, epoch, score, metrics, "best.pth"
                )

    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        score: float,
        metrics: dict,
        filename: str
    ):
        """Save checkpoint to file"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            f"{self.metric_name}": score,
            "metrics": metrics
        }

        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)

        if self.verbose:
            logger.info(f"Checkpoint saved: {save_path}")


class LRSchedulerCallback:
    """
    Wrapper for learning rate schedulers
    """

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        metric_name: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Args:
            scheduler: PyTorch LR scheduler
            metric_name: Metric to monitor (for ReduceLROnPlateau)
            verbose: Whether to print messages
        """
        self.scheduler = scheduler
        self.metric_name = metric_name
        self.verbose = verbose

        self.is_reduce_on_plateau = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        )

    def step(self, metrics: Optional[dict] = None):
        """
        Step the scheduler

        Args:
            metrics: Dictionary of metrics (required for ReduceLROnPlateau)
        """
        if self.is_reduce_on_plateau:
            if metrics is None or self.metric_name not in metrics:
                raise ValueError(
                    f"Metric '{self.metric_name}' required for ReduceLROnPlateau"
                )
            metric_value = metrics[self.metric_name]
            self.scheduler.step(metric_value)
        else:
            self.scheduler.step()

        # Log current LR
        if self.verbose:
            current_lr = self.scheduler.optimizer.param_groups[0]["lr"]
            logger.info(f"Learning rate: {current_lr:.2e}")

    def get_last_lr(self):
        """Get current learning rate"""
        return self.scheduler.get_last_lr()


def main():
    """Test callbacks"""
    print("Testing callbacks...")

    # Test EarlyStopping
    print("\n1. Testing EarlyStopping...")
    early_stopping = EarlyStopping(patience=3, mode="max", verbose=True)

    scores = [0.80, 0.85, 0.87, 0.86, 0.85, 0.84, 0.83]
    for epoch, score in enumerate(scores):
        print(f"Epoch {epoch}: score={score:.2f}")
        if early_stopping(score):
            print(f"Early stopping at epoch {epoch}!")
            break

    # Test ModelCheckpoint
    print("\n2. Testing ModelCheckpoint...")
    checkpoint = ModelCheckpoint(
        save_dir="test_checkpoints",
        metric_name="val_acc",
        mode="max",
        verbose=True
    )

    # Create dummy model
    model = torch.nn.Linear(10, 4)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch, score in enumerate([0.80, 0.85, 0.82, 0.90]):
        metrics = {"val_acc": score, "val_loss": 1.0 - score}
        print(f"Epoch {epoch}: val_acc={score:.2f}")
        checkpoint(model, optimizer, epoch, score, metrics)

    print("\n✓ Callbacks test passed!")

    # Cleanup
    import shutil
    if os.path.exists("test_checkpoints"):
        shutil.rmtree("test_checkpoints")


if __name__ == "__main__":
    main()
