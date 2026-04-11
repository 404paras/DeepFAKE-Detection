"""
Audio Encoder: CNN + Bi-GRU
Extracts spectro-temporal features from Mel-spectrograms
"""

import torch
import torch.nn as nn
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioEncoder(nn.Module):
    """
    Audio encoder as specified in paper:
    - Custom 4-layer CNN (64→128→256→512 channels)
    - 2-layer Bidirectional GRU for temporal modeling
    - Output: 512-D feature vector
    """

    def __init__(
        self,
        conv_channels: List[int] = [64, 128, 256, 512],
        kernel_size: int = 3,
        padding: int = 1,
        pool_size: int = 2,
        bigru_hidden: int = 256,
        bigru_layers: int = 2,
        bigru_dropout: float = 0.0,
        output_dim: int = 512
    ):
        """
        Args:
            conv_channels: List of channel dimensions for CNN blocks
            kernel_size: Convolution kernel size
            padding: Padding for convolutions
            pool_size: Max pooling size
            bigru_hidden: Hidden dimension of Bi-GRU
            bigru_layers: Number of Bi-GRU layers
            bigru_dropout: Dropout for Bi-GRU
            output_dim: Output feature dimension
        """
        super(AudioEncoder, self).__init__()

        self.output_dim = output_dim

        # Build CNN blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 1  # Mel-spectrogram has 1 channel

        for out_channels in conv_channels:
            block = self._make_conv_block(
                in_channels,
                out_channels,
                kernel_size,
                padding,
                pool_size
            )
            self.conv_blocks.append(block)
            in_channels = out_channels

        # Calculate CNN output dimensions
        # Input: (B, 1, 128, 94)
        # After 4x max pooling (2x2): (B, 512, 8, 5)
        self.cnn_output_height = 128 // (pool_size ** len(conv_channels))  # 128/16 = 8
        self.cnn_output_width = 94 // (pool_size ** len(conv_channels))  # 94/16 = 5
        self.cnn_output_channels = conv_channels[-1]

        # Calculate input size for GRU
        # We treat time dimension as sequence, so flatten spatial dims
        self.gru_input_size = self.cnn_output_channels * self.cnn_output_height

        # Bidirectional GRU for temporal modeling
        self.bigru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=bigru_hidden,
            num_layers=bigru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=bigru_dropout if bigru_layers > 1 else 0.0
        )

        # Project Bi-GRU output to output_dim
        bigru_output_dim = bigru_hidden * 2  # Bidirectional
        self.projection = nn.Linear(bigru_output_dim, output_dim)

        logger.info(f"AudioEncoder initialized:")
        logger.info(f"  CNN channels: {conv_channels}")
        logger.info(f"  CNN output shape: ({self.cnn_output_channels}, {self.cnn_output_height}, {self.cnn_output_width})")
        logger.info(f"  GRU input size: {self.gru_input_size}")
        logger.info(f"  Bi-GRU: {bigru_layers} layers, hidden={bigru_hidden}")
        logger.info(f"  Output dim: {output_dim}")

    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        pool_size: int
    ) -> nn.Module:
        """
        Create a CNN block: Conv2D → ReLU → MaxPool2D

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            padding: Padding
            pool_size: Pooling size

        Returns:
            Sequential CNN block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size)
        )

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            mel_spec: Tensor of shape (B, 1, n_mels, time_steps)
                      Expected: (B, 1, 128, 94)

        Returns:
            Feature tensor of shape (B, output_dim)
        """
        B = mel_spec.shape[0]

        # Pass through CNN blocks
        x = mel_spec
        for block in self.conv_blocks:
            x = block(x)
        # x shape: (B, 512, 8, 5)

        # Reshape for GRU: treat time dimension as sequence
        # (B, C, H, W) -> (B, W, C*H)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.reshape(B, W, C * H)  # (B, 5, 512*8=4096)

        # Temporal modeling with Bi-GRU
        gru_out, _ = self.bigru(x)  # (B, W, bigru_hidden*2)

        # Take output at last timestep
        final_features = gru_out[:, -1, :]  # (B, bigru_hidden*2)

        # Project to output dimension
        output = self.projection(final_features)  # (B, output_dim)

        return output

    def get_trainable_parameters(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def main():
    """Test audio encoder"""
    print("Testing AudioEncoder...")

    # Create model
    model = AudioEncoder(
        conv_channels=[64, 128, 256, 512],
        kernel_size=3,
        padding=1,
        pool_size=2,
        bigru_hidden=256,
        bigru_layers=2,
        output_dim=512
    )

    print(f"\nModel architecture:")
    print(model)

    print(f"\nTrainable parameters: {model.get_trainable_parameters():,}")

    # Test forward pass
    batch_size = 4
    n_mels = 128
    time_steps = 94

    dummy_input = torch.randn(batch_size, 1, n_mels, time_steps)

    print(f"\nInput shape: {dummy_input.shape}")

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, 512)")

    assert output.shape == (batch_size, 512), "Output shape mismatch!"

    print("\n✓ AudioEncoder test passed!")


if __name__ == "__main__":
    main()
