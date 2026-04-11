"""
Evaluation Script
Evaluate trained model on test set
"""

import argparse
import torch
import logging
from pathlib import Path

from src.utils.config import load_config
from src.data.dataloader import create_dataloaders
from src.models.multimodal_detector import MultimodalDeepfakeDetector
from src.training.losses import get_loss_function
from src.evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate Multimodal Deepfake Detector")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--test-manifest",
        type=str,
        default="data/manifests/test.csv",
        help="Path to test manifest"
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Dataset root directory"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_results.json",
        help="Path to save evaluation results"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )

    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()

    # Load config
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Set device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading test data...")
    dataset_root = args.dataset_root or config["data"]["dataset_path"]

    # We only need test loader for evaluation
    _, _, test_loader = create_dataloaders(
        train_manifest="data/manifests/train.csv",  # Won't be used
        val_manifest="data/manifests/val.csv",  # Won't be used
        test_manifest=args.test_manifest,
        dataset_root=dataset_root,
        config=config,
        num_workers=config["data"]["num_workers"]
    )

    # Load model
    logger.info(f"Loading model from: {args.checkpoint}")
    model = MultimodalDeepfakeDetector(config)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info(f"Model loaded successfully")
    if "epoch" in checkpoint:
        logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")

    # Create loss function
    criterion = get_loss_function(config)

    # Create evaluator
    class_names = config.get("classes", ["Real-Real", "Real-Fake", "Fake-Real", "Fake-Fake"])
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names
    )

    # Evaluate
    logger.info("="*60)
    logger.info("Starting Evaluation")
    logger.info("="*60)

    metrics = evaluator.evaluate()

    # Save results
    evaluator.save_results(metrics, args.output)

    logger.info("="*60)
    logger.info("Evaluation Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
