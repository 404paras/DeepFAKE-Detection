"""
Video Encoder: ResNet50 + Bi-GRU
Extracts spatial-temporal features from video frames
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoEncoder(nn.Module):
    """
    Video encoder as specified in paper:
    - ResNet50 backbone (pretrained on ImageNet)
    - Freeze first 3 layers
    - 2-layer Bidirectional GRU for temporal modeling
    - Output: 2048-D feature vector
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_layers: list = [1, 2, 3],
        bigru_hidden: int = 512,
        bigru_layers: int = 2,
        bigru_dropout: float = 0.0,
        output_dim: int = 2048
    ):
        """
        Args:
            backbone: Backbone architecture (resnet50)
            pretrained: Whether to use ImageNet pretrained weights
            freeze_layers: List of layer indices to freeze [1, 2, 3]
            bigru_hidden: Hidden dimension of Bi-GRU
            bigru_layers: Number of Bi-GRU layers
            bigru_dropout: Dropout for Bi-GRU
            output_dim: Output feature dimension
        """
        super(VideoEncoder, self).__init__()

        self.output_dim = output_dim

        # Load ResNet50 backbone
        if backbone == "resnet50":
            if pretrained:
                self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                logger.info("Loaded pretrained ResNet50 (ImageNet)")
            else:
                self.resnet = models.resnet50(weights=None)
                logger.info("Initialized ResNet50 from scratch")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze specified layers
        self._freeze_layers(freeze_layers)

        # Remove final FC layer (we want features before classification)
        self.resnet.fc = nn.Identity()
        self.feature_dim = 2048  # ResNet50 feature dimension

        # Bidirectional GRU for temporal modeling
        self.bigru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=bigru_hidden,
            num_layers=bigru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=bigru_dropout if bigru_layers > 1 else 0.0
        )

        # Project Bi-GRU output back to output_dim
        bigru_output_dim = bigru_hidden * 2  # Bidirectional
        self.projection = nn.Linear(bigru_output_dim, output_dim)

        logger.info(f"VideoEncoder initialized:")
        logger.info(f"  Backbone: {backbone}")
        logger.info(f"  Frozen layers: {freeze_layers}")
        logger.info(f"  Bi-GRU: {bigru_layers} layers, hidden={bigru_hidden}")
        logger.info(f"  Output dim: {output_dim}")

    def _freeze_layers(self, freeze_layers: list):
        """
        Freeze specified ResNet layers

        Args:
            freeze_layers: List of layer indices [1, 2, 3, 4]
        """
        # Map layer indices to ResNet modules
        layer_map = {
            1: self.resnet.layer1,
            2: self.resnet.layer2,
            3: self.resnet.layer3,
            4: self.resnet.layer4
        }

        # Also freeze initial layers (conv1, bn1, relu, maxpool)
        for param in self.resnet.conv1.parameters():
            param.requires_grad = False
        for param in self.resnet.bn1.parameters():
            param.requires_grad = False

        # Freeze specified layers
        for layer_idx in freeze_layers:
            if layer_idx in layer_map:
                for param in layer_map[layer_idx].parameters():
                    param.requires_grad = False
                logger.info(f"  Frozen layer{layer_idx}")

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            frames: Tensor of shape (B, T, C, H, W)
                    B = batch size
                    T = number of frames (16)
                    C = channels (3)
                    H, W = height, width (224, 224)

        Returns:
            Feature tensor of shape (B, output_dim)
        """
        B, T, C, H, W = frames.shape

        # Reshape to process all frames through ResNet
        frames = frames.contiguous().reshape(B * T, C, H, W)  # (B*T, 3, 224, 224)

        # Extract per-frame features using ResNet
        with torch.set_grad_enabled(self.training):
            features = self.resnet(frames)  # (B*T, 2048)

        # Reshape back to sequence
        features = features.contiguous().reshape(B, T, self.feature_dim)  # (B, T, 2048)

        # Temporal modeling with Bi-GRU
        gru_out, _ = self.bigru(features)  # (B, T, bigru_hidden*2)

        # Take the output at the last timestep
        final_features = gru_out[:, -1, :]  # (B, bigru_hidden*2)

        # Project to output dimension
        output = self.projection(final_features)  # (B, output_dim)

        return output

    def get_trainable_parameters(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def main():
    """Test video encoder"""
    print("Testing VideoEncoder...")

    # Create model
    model = VideoEncoder(
        backbone="resnet50",
        pretrained=True,
        freeze_layers=[1, 2, 3],
        bigru_hidden=512,
        bigru_layers=2,
        output_dim=2048
    )

    print(f"\nModel architecture:")
    print(model)

    print(f"\nTrainable parameters: {model.get_trainable_parameters():,}")

    # Test forward pass
    batch_size = 4
    num_frames = 16
    dummy_input = torch.randn(batch_size, num_frames, 3, 224, 224)

    print(f"\nInput shape: {dummy_input.shape}")

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, 2048)")

    assert output.shape == (batch_size, 2048), "Output shape mismatch!"

    print("\n✓ VideoEncoder test passed!")


if __name__ == "__main__":
    main()
