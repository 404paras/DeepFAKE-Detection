"""
Complete Multimodal Deepfake Detector
Combines Video Encoder, Audio Encoder, Cross-Modal Attention, and Classifier
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

from .video_encoder import VideoEncoder
from .audio_encoder import AudioEncoder
from .cross_modal_attention import CrossModalAttention

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalDeepfakeDetector(nn.Module):
    """
    Complete multimodal deepfake detection model as specified in paper:
    - Video Stream: ResNet50 + Bi-GRU → 2048-D
    - Audio Stream: CNN + Bi-GRU → 512-D
    - Cross-Modal Fusion: 8-head attention → 2560-D
    - Classifier: FFN [2560→1536→768→256→4]
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary containing model hyperparameters
        """
        super(MultimodalDeepfakeDetector, self).__init__()

        # Video encoder
        self.video_encoder = VideoEncoder(
            backbone=config["model"]["video_encoder"]["backbone"],
            pretrained=config["model"]["video_encoder"]["pretrained"],
            freeze_layers=config["model"]["video_encoder"]["freeze_layers"],
            bigru_hidden=config["model"]["video_encoder"]["bigru_hidden"],
            bigru_layers=config["model"]["video_encoder"]["bigru_layers"],
            output_dim=config["model"]["video_encoder"]["output_dim"]
        )

        # Audio encoder
        self.audio_encoder = AudioEncoder(
            conv_channels=config["model"]["audio_encoder"]["conv_channels"],
            kernel_size=config["model"]["audio_encoder"]["kernel_size"],
            padding=config["model"]["audio_encoder"]["padding"],
            pool_size=config["model"]["audio_encoder"]["pool_size"],
            bigru_hidden=config["model"]["audio_encoder"]["bigru_hidden"],
            bigru_layers=config["model"]["audio_encoder"]["bigru_layers"],
            output_dim=config["model"]["audio_encoder"]["output_dim"]
        )

        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(
            video_dim=config["model"]["video_encoder"]["output_dim"],
            audio_dim=config["model"]["audio_encoder"]["output_dim"],
            num_heads=config["model"]["cross_modal_attention"]["num_heads"],
            dropout=config["model"]["cross_modal_attention"]["dropout"]
        )

        # Classification head
        embed_dim = config["model"]["cross_modal_attention"]["embed_dim"]
        hidden_dims = config["model"]["classifier"]["hidden_dims"]
        dropout = config["model"]["classifier"]["dropout"]
        num_classes = config["model"]["classifier"]["num_classes"]

        # Build classifier: FFN with dropout
        classifier_layers = []

        input_dim = embed_dim
        for hidden_dim in hidden_dims:
            classifier_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim

        # Final output layer
        classifier_layers.append(nn.Linear(input_dim, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

        # Store config
        self.config = config
        self.num_classes = num_classes

        # Log model info
        logger.info(f"MultimodalDeepfakeDetector initialized:")
        logger.info(f"  Total parameters: {self.get_total_parameters():,}")
        logger.info(f"  Trainable parameters: {self.get_trainable_parameters():,}")

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            video: Video frames of shape (B, T, C, H, W)
                   Expected: (B, 16, 3, 224, 224)
            audio: Mel-spectrograms of shape (B, 1, n_mels, time_steps)
                   Expected: (B, 1, 128, 94)
            return_attention: Whether to return attention weights

        Returns:
            logits: Class logits of shape (B, num_classes)
            attention_weights: Attention weights (optional)
        """
        # Extract video features
        video_features = self.video_encoder(video)  # (B, 2048)

        # Extract audio features
        audio_features = self.audio_encoder(audio)  # (B, 512)

        # Cross-modal fusion
        if return_attention:
            fused_features, attention_weights = self.cross_modal_attention(
                video_features,
                audio_features,
                return_attention_weights=True
            )
        else:
            fused_features = self.cross_modal_attention(
                video_features,
                audio_features,
                return_attention_weights=False
            )
            attention_weights = None

        # Classification
        logits = self.classifier(fused_features)  # (B, num_classes)

        return logits, attention_weights

    def get_total_parameters(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self):
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb


def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file"""
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Test complete model"""
    print("Testing MultimodalDeepfakeDetector...")

    # Load config
    config_path = "../../configs/config.yaml"
    config = load_config_from_yaml(config_path)

    # Create model
    model = MultimodalDeepfakeDetector(config)

    print(f"\nModel summary:")
    print(f"  Total parameters: {model.get_total_parameters():,}")
    print(f"  Trainable parameters: {model.get_trainable_parameters():,}")
    print(f"  Model size: {model.get_model_size_mb():.2f} MB")

    # Test forward pass
    batch_size = 4
    video = torch.randn(batch_size, 16, 3, 224, 224)
    audio = torch.randn(batch_size, 1, 128, 94)

    print(f"\nInput shapes:")
    print(f"  Video: {video.shape}")
    print(f"  Audio: {audio.shape}")

    model.eval()
    with torch.no_grad():
        # Without attention
        logits, _ = model(video, audio, return_attention=False)
        print(f"\nOutput shape: {logits.shape}")
        print(f"Expected: ({batch_size}, 4)")

        assert logits.shape == (batch_size, 4), "Output shape mismatch!"

        # With attention
        logits, attn = model(video, audio, return_attention=True)
        print(f"\nWith attention:")
        print(f"  Logits: {logits.shape}")
        print(f"  Attention: {attn.shape if attn is not None else None}")

        # Test predictions
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        print(f"\nSample predictions:")
        print(f"  Probabilities:\n{probs}")
        print(f"  Predicted classes: {preds}")

    print("\n✓ MultimodalDeepfakeDetector test passed!")


if __name__ == "__main__":
    main()
