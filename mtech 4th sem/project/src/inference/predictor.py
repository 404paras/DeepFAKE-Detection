"""
Deepfake Predictor
Single video inference
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict
import logging

from ..data.preprocessing import VideoPreprocessor, AudioPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepfakePredictor:
    """
    Predictor for single video inference
    """

    def __init__(
        self,
        model: torch.nn.Module,
        video_preprocessor: VideoPreprocessor,
        audio_preprocessor: AudioPreprocessor,
        class_names: list = None,
        device: str = "cuda"
    ):
        """
        Args:
            model: Trained model
            video_preprocessor: Video preprocessing instance
            audio_preprocessor: Audio preprocessing instance
            class_names: List of class names
            device: Device to use
        """
        self.model = model.to(device)
        self.model.eval()
        self.video_preprocessor = video_preprocessor
        self.audio_preprocessor = audio_preprocessor
        self.device = device

        self.class_names = class_names or [
            "Real-Real",
            "Real-Fake",
            "Fake-Real",
            "Fake-Fake"
        ]

    @torch.no_grad()
    def predict(self, video_path: str, return_attention: bool = False) -> Dict:
        """
        Predict on a single video

        Args:
            video_path: Path to video file
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Processing: {video_path}")

        # Preprocess video
        video_tensor = self.video_preprocessor.preprocess(video_path)
        video_tensor = video_tensor.unsqueeze(0).to(self.device)  # Add batch dim

        # Preprocess audio
        audio_tensor = self.audio_preprocessor.preprocess(video_path)
        audio_tensor = audio_tensor.unsqueeze(0).to(self.device)  # Add batch dim

        # Inference
        logits, attention = self.model(video_tensor, audio_tensor, return_attention=return_attention)

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)[0]  # Remove batch dim
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

        # Build results
        results = {
            "video_path": str(video_path),
            "predicted_class": self.class_names[pred_idx],
            "predicted_index": pred_idx,
            "confidence": confidence,
            "probabilities": {
                class_name: prob.item()
                for class_name, prob in zip(self.class_names, probs)
            }
        }

        if return_attention and attention is not None:
            results["attention_weights"] = attention.cpu().numpy()

        return results

    def predict_batch(self, video_paths: list) -> list:
        """
        Predict on multiple videos

        Args:
            video_paths: List of video file paths

        Returns:
            List of prediction dictionaries
        """
        results = []
        for video_path in video_paths:
            try:
                result = self.predict(video_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                results.append({
                    "video_path": str(video_path),
                    "error": str(e)
                })

        return results


def load_model_from_checkpoint(checkpoint_path: str, config: dict, device: str = "cuda"):
    """
    Load model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Device to load model on

    Returns:
        Loaded model
    """
    from ..models.multimodal_detector import MultimodalDeepfakeDetector

    model = MultimodalDeepfakeDetector(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info(f"Model loaded from: {checkpoint_path}")
    if "epoch" in checkpoint:
        logger.info(f"  Epoch: {checkpoint['epoch']}")
    if "val_auc_roc" in checkpoint:
        logger.info(f"  Val AUC-ROC: {checkpoint['val_auc_roc']:.4f}")

    return model
