"""
Demo Script
CLI for single video inference
"""

import argparse
import json
import logging

from src.utils.config import load_config
from src.data.preprocessing import VideoPreprocessor, AudioPreprocessor
from src.inference.predictor import DeepfakePredictor, load_model_from_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deepfake Detection Demo")

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pth",
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save prediction results (JSON)"
    )

    parser.add_argument(
        "--show-attention",
        action="store_true",
        help="Return attention weights"
    )

    return parser.parse_args()


def main():
    """Main demo function"""
    args = parse_args()

    # Load config
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Load model
    logger.info(f"Loading model from: {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, config, args.device)

    # Create preprocessors
    video_prep = VideoPreprocessor(
        num_frames=config["preprocessing"]["video"]["num_frames"],
        img_size=config["preprocessing"]["video"]["img_size"],
        normalize=config["preprocessing"]["video"]["normalize"],
        mean=config["preprocessing"]["video"].get("mean"),
        std=config["preprocessing"]["video"].get("std")
    )

    audio_prep = AudioPreprocessor(
        sample_rate=config["preprocessing"]["audio"]["sample_rate"],
        duration=config["preprocessing"]["audio"]["duration"],
        n_fft=config["preprocessing"]["audio"]["n_fft"],
        hop_length=config["preprocessing"]["audio"]["hop_length"],
        n_mels=config["preprocessing"]["audio"]["n_mels"],
        f_min=config["preprocessing"]["audio"].get("f_min", 0.0),
        f_max=config["preprocessing"]["audio"].get("f_max", 8000.0)
    )

    # Create predictor
    class_names = config.get("classes", ["Real-Real", "Real-Fake", "Fake-Real", "Fake-Fake"])
    predictor = DeepfakePredictor(
        model=model,
        video_preprocessor=video_prep,
        audio_preprocessor=audio_prep,
        class_names=class_names,
        device=args.device
    )

    # Predict
    logger.info("="*60)
    logger.info(f"Analyzing video: {args.video}")
    logger.info("="*60)

    results = predictor.predict(args.video, return_attention=args.show_attention)

    # Print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\nVideo: {results['video_path']}")
    print(f"\nPredicted Class: {results['predicted_class']}")
    print(f"Confidence: {results['confidence']:.2%}")
    print(f"\nClass Probabilities:")
    for class_name, prob in results['probabilities'].items():
        print(f"  {class_name:15s}: {prob:.2%}")

    # Save results if output path provided
    if args.output:
        # Remove attention weights for JSON serialization
        output_results = {k: v for k, v in results.items() if k != "attention_weights"}

        with open(args.output, "w") as f:
            json.dump(output_results, f, indent=2)

        logger.info(f"\nResults saved to: {args.output}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
