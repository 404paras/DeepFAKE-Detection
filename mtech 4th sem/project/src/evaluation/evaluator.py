"""
Model Evaluator
Evaluate trained model on test set
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path
import logging

from ..training.metrics import MetricsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate model on test set"""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: str = "cuda",
        class_names: list = None
    ):
        """
        Args:
            model: Trained model
            test_loader: Test data loader
            criterion: Loss function
            device: Device to use
            class_names: List of class names
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device

        self.class_names = class_names or ["Real-Real", "Real-Fake", "Fake-Real", "Fake-Fake"]

        self.metrics_calculator = MetricsCalculator(
            num_classes=len(self.class_names),
            class_names=self.class_names
        )

    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Evaluate model on test set

        Returns:
            Dictionary of metrics
        """
        logger.info("Starting evaluation...")

        self.model.eval()
        self.metrics_calculator.reset()

        total_loss = 0.0
        num_samples = 0

        pbar = tqdm(self.test_loader, desc="Evaluating")

        for videos, audios, labels, _ in pbar:
            # Move to device
            videos = videos.to(self.device)
            audios = audios.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits, _ = self.model(videos, audios)
            loss = self.criterion(logits, labels)

            # Update metrics
            total_loss += loss.item() * videos.size(0)
            num_samples += videos.size(0)
            self.metrics_calculator.update(logits, labels)

        # Compute metrics
        metrics = self.metrics_calculator.compute()
        metrics["test_loss"] = total_loss / num_samples

        # Log results
        logger.info("\nTest Results:")
        logger.info(f"  Loss: {metrics['test_loss']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"  F1 (Macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
        logger.info(f"  Recall (Macro): {metrics['recall_macro']:.4f}")

        logger.info("\nPer-class metrics:")
        for i, class_name in enumerate(self.class_names):
            logger.info(f"  {class_name}:")
            logger.info(f"    Precision: {metrics[f'precision_{class_name}']:.4f}")
            logger.info(f"    Recall: {metrics[f'recall_{class_name}']:.4f}")
            logger.info(f"    F1: {metrics[f'f1_{class_name}']:.4f}")

        return metrics

    def save_results(self, metrics: dict, save_path: str):
        """
        Save evaluation results

        Args:
            metrics: Metrics dictionary
            save_path: Path to save results
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert confusion matrix to list for JSON serialization
        results = {}
        for key, value in metrics.items():
            if key == "confusion_matrix":
                results[key] = value.tolist()
            else:
                results[key] = float(value)

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {save_path}")
