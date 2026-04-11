"""
Training Metrics
Compute accuracy, AUC-ROC, F1, precision, recall for multiclass classification
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate classification metrics"""

    def __init__(self, num_classes: int = 4, class_names: Optional[list] = None):
        """
        Args:
            num_classes: Number of classes
            class_names: List of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        # Storage for batch-wise predictions
        self.reset()

    def reset(self):
        """Reset all stored predictions"""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Update with batch predictions

        Args:
            logits: Model logits of shape (B, num_classes)
            labels: Ground truth labels of shape (B,)
        """
        # Convert to numpy
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        self.all_probs.append(probs)
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics

        Returns:
            Dictionary of metrics
        """
        # Concatenate all batches
        preds = np.concatenate(self.all_preds)
        labels = np.concatenate(self.all_labels)
        probs = np.concatenate(self.all_probs)

        metrics = {}

        # Accuracy
        metrics["accuracy"] = accuracy_score(labels, preds)

        # AUC-ROC (one-vs-rest for multiclass)
        try:
            metrics["auc_roc"] = roc_auc_score(
                labels,
                probs,
                multi_class="ovr",
                average="macro"
            )
        except Exception as e:
            logger.warning(f"Failed to compute AUC-ROC: {e}")
            metrics["auc_roc"] = 0.0

        # F1 Score
        metrics["f1_macro"] = f1_score(labels, preds, average="macro", zero_division=0)
        metrics["f1_weighted"] = f1_score(labels, preds, average="weighted", zero_division=0)

        # Precision
        metrics["precision_macro"] = precision_score(labels, preds, average="macro", zero_division=0)
        metrics["precision_weighted"] = precision_score(labels, preds, average="weighted", zero_division=0)

        # Recall
        metrics["recall_macro"] = recall_score(labels, preds, average="macro", zero_division=0)
        metrics["recall_weighted"] = recall_score(labels, preds, average="weighted", zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

        for i in range(self.num_classes):
            metrics[f"precision_{self.class_names[i]}"] = precision_per_class[i]
            metrics[f"recall_{self.class_names[i]}"] = recall_per_class[i]
            metrics[f"f1_{self.class_names[i]}"] = f1_per_class[i]

        # Confusion matrix
        cm = confusion_matrix(labels, preds, labels=list(range(self.num_classes)))
        metrics["confusion_matrix"] = cm

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        preds = np.concatenate(self.all_preds)
        labels = np.concatenate(self.all_labels)
        return confusion_matrix(labels, preds, labels=list(range(self.num_classes)))


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy for a batch

    Args:
        logits: Model logits of shape (B, num_classes)
        labels: Ground truth labels of shape (B,)

    Returns:
        Accuracy value
    """
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0


def main():
    """Test metrics calculator"""
    print("Testing MetricsCalculator...")

    # Create calculator
    class_names = ["Real-Real", "Real-Fake", "Fake-Real", "Fake-Fake"]
    calculator = MetricsCalculator(num_classes=4, class_names=class_names)

    # Simulate some predictions
    num_batches = 10
    batch_size = 32

    for _ in range(num_batches):
        # Random logits and labels
        logits = torch.randn(batch_size, 4)
        labels = torch.randint(0, 4, (batch_size,))

        calculator.update(logits, labels)

    # Compute metrics
    metrics = calculator.compute()

    print("\nComputed metrics:")
    for key, value in metrics.items():
        if key != "confusion_matrix":
            print(f"  {key}: {value:.4f}")

    print(f"\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\n✓ MetricsCalculator test passed!")


if __name__ == "__main__":
    main()
