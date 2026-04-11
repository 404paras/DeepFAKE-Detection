"""
Cross-Modal Attention Mechanism
Fuses video and audio features with multi-head attention
"""

import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention fusion as specified in paper:
    - 8-head Multi-Head Attention
    - Fuses concatenated video (2048-D) and audio (512-D) features
    - Detects phoneme-viseme misalignments
    """

    def __init__(
        self,
        video_dim: int = 2048,
        audio_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            video_dim: Video feature dimension
            audio_dim: Audio feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(CrossModalAttention, self).__init__()

        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.embed_dim = video_dim + audio_dim  # 2560
        self.num_heads = num_heads

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        logger.info(f"CrossModalAttention initialized:")
        logger.info(f"  Video dim: {video_dim}")
        logger.info(f"  Audio dim: {audio_dim}")
        logger.info(f"  Embed dim: {self.embed_dim}")
        logger.info(f"  Num heads: {num_heads}")

    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
        return_attention_weights: bool = False
    ):
        """
        Forward pass

        Args:
            video_features: Video features of shape (B, video_dim)
            audio_features: Audio features of shape (B, audio_dim)
            return_attention_weights: Whether to return attention weights

        Returns:
            Fused features of shape (B, embed_dim)
            Attention weights (optional) of shape (B, num_heads, 1, 1)
        """
        B = video_features.shape[0]

        # Concatenate modalities
        fused = torch.cat([video_features, audio_features], dim=-1)  # (B, 2560)

        # Add sequence dimension for attention
        fused = fused.unsqueeze(1)  # (B, 1, 2560)

        # Self-attention for cross-modal reasoning
        # Query, Key, Value all use the same fused features
        attn_out, attn_weights = self.multihead_attn(
            query=fused,
            key=fused,
            value=fused,
            need_weights=return_attention_weights
        )
        # attn_out: (B, 1, 2560)
        # attn_weights: (B, num_heads, 1, 1) if return_attention_weights

        # Remove sequence dimension
        attn_out = attn_out.squeeze(1)  # (B, 2560)

        # Residual connection + layer norm
        output = self.layer_norm(fused.squeeze(1) + attn_out)

        if return_attention_weights:
            return output, attn_weights
        else:
            return output

    def get_trainable_parameters(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def main():
    """Test cross-modal attention"""
    print("Testing CrossModalAttention...")

    # Create model
    model = CrossModalAttention(
        video_dim=2048,
        audio_dim=512,
        num_heads=8,
        dropout=0.1
    )

    print(f"\nModel architecture:")
    print(model)

    print(f"\nTrainable parameters: {model.get_trainable_parameters():,}")

    # Test forward pass
    batch_size = 4
    video_features = torch.randn(batch_size, 2048)
    audio_features = torch.randn(batch_size, 512)

    print(f"\nInput shapes:")
    print(f"  Video: {video_features.shape}")
    print(f"  Audio: {audio_features.shape}")

    model.eval()
    with torch.no_grad():
        # Without attention weights
        output = model(video_features, audio_features, return_attention_weights=False)
        print(f"\nOutput shape: {output.shape}")
        print(f"Expected: ({batch_size}, 2560)")

        assert output.shape == (batch_size, 2560), "Output shape mismatch!"

        # With attention weights
        output, attn_weights = model(video_features, audio_features, return_attention_weights=True)
        print(f"\nWith attention weights:")
        print(f"  Output: {output.shape}")
        print(f"  Attention: {attn_weights.shape}")

    print("\n✓ CrossModalAttention test passed!")


if __name__ == "__main__":
    main()
