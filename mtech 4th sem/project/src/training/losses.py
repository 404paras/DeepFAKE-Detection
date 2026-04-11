"""
Loss Functions
Implements CrossEntropyLoss and optional FocalLoss for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        """
        Args:
            alpha: Weighting factor for each class (tensor of shape (num_classes,))
            gamma: Focusing parameter (gamma >= 0)
            reduction: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        logger.info(f"FocalLoss initialized: gamma={gamma}, reduction={reduction}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model outputs (B, num_classes)
            targets: Ground truth labels (B,)

        Returns:
            Loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Compute pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)

        # Compute focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Compute focal loss
        loss = focal_term * ce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing
    """

    def __init__(self, smoothing: float = 0.1):
        """
        Args:
            smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

        logger.info(f"LabelSmoothingCrossEntropy: smoothing={smoothing}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model outputs (B, num_classes)
            targets: Ground truth labels (B,)

        Returns:
            Loss value
        """
        log_probs = F.log_softmax(logits, dim=-1)
        num_classes = logits.size(-1)

        # One-hot encode targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        # Compute loss
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        return loss.mean()


def get_loss_function(config: dict) -> nn.Module:
    """
    Get loss function from config

    Args:
        config: Configuration dictionary

    Returns:
        Loss function module
    """
    loss_name = config["training"]["loss"]["name"]
    label_smoothing = config["training"]["loss"].get("label_smoothing", 0.0)

    if loss_name == "CrossEntropyLoss":
        if label_smoothing > 0:
            logger.info(f"Using CrossEntropyLoss with label_smoothing={label_smoothing}")
            return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            logger.info("Using standard CrossEntropyLoss")
            return nn.CrossEntropyLoss()

    elif loss_name == "FocalLoss":
        gamma = config["training"]["loss"].get("focal_gamma", 2.0)
        alpha = config["training"]["loss"].get("focal_alpha", None)

        if alpha is not None:
            alpha = torch.tensor(alpha)

        logger.info(f"Using FocalLoss with gamma={gamma}")
        return FocalLoss(alpha=alpha, gamma=gamma)

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def main():
    """Test loss functions"""
    print("Testing loss functions...")

    batch_size = 16
    num_classes = 4

    # Random logits and labels
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    print(f"\nLogits shape: {logits.shape}")
    print(f"Targets shape: {targets.shape}")

    # Test CrossEntropyLoss
    ce_loss = nn.CrossEntropyLoss()
    loss_ce = ce_loss(logits, targets)
    print(f"\nCrossEntropyLoss: {loss_ce.item():.4f}")

    # Test FocalLoss
    focal_loss = FocalLoss(gamma=2.0)
    loss_focal = focal_loss(logits, targets)
    print(f"FocalLoss (gamma=2.0): {loss_focal.item():.4f}")

    # Test LabelSmoothingCrossEntropy
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss_ls = ls_loss(logits, targets)
    print(f"LabelSmoothingCE (smoothing=0.1): {loss_ls.item():.4f}")

    # Test with class imbalance
    alpha = torch.tensor([0.5, 1.0, 1.5, 2.0])
    focal_loss_weighted = FocalLoss(alpha=alpha, gamma=2.0)
    loss_focal_weighted = focal_loss_weighted(logits, targets)
    print(f"FocalLoss (weighted): {loss_focal_weighted.item():.4f}")

    print("\n✓ Loss functions test passed!")


if __name__ == "__main__":
    main()
