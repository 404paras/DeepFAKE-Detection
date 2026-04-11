"""
Dataset Analyzer for Multimodal Deepfake Detection
Analyzes extracted dataset structure and creates train/val/test splits
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Analyze dataset structure and create data splits"""

    def __init__(self, dataset_root: str, output_dir: str = "data/manifests"):
        """
        Args:
            dataset_root: Root directory of extracted dataset
            output_dir: Directory to save manifest CSV files
        """
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.dataset_root.exists():
            raise ValueError(f"Dataset root not found: {self.dataset_root}")

    def scan_dataset(self) -> List[Dict]:
        """
        Scan dataset directory and collect all video files with metadata

        Returns:
            List of dictionaries containing video info
        """
        logger.info(f"Scanning dataset at: {self.dataset_root}")

        samples = []

        # Find all video files
        video_files = list(self.dataset_root.rglob("*.mp4"))
        logger.info(f"Found {len(video_files)} video files")

        for video_path in tqdm(video_files, desc="Analyzing videos"):
            # Get corresponding JSON metadata file
            json_path = video_path.with_suffix(".json")

            sample_info = {
                "video_path": str(video_path.relative_to(self.dataset_root)),
                "video_name": video_path.stem,
                "video_id": video_path.parent.name,
                "has_metadata": json_path.exists(),
            }

            # Load metadata if available
            if json_path.exists():
                try:
                    with open(json_path, "r") as f:
                        metadata = json.load(f)

                    # Extract relevant fields
                    # Note: Actual fields depend on dataset format
                    # Common fields: label, manipulation_type, video_real, audio_real
                    sample_info.update(self._parse_metadata(metadata))

                except Exception as e:
                    logger.warning(f"Failed to load metadata for {video_path.name}: {e}")
                    sample_info["metadata_error"] = str(e)

            samples.append(sample_info)

        logger.info(f"Collected {len(samples)} samples")
        return samples

    def _parse_metadata(self, metadata: Dict) -> Dict:
        """
        Parse metadata JSON and extract relevant fields

        Args:
            metadata: Dictionary from JSON file

        Returns:
            Dictionary with parsed fields
        """
        parsed = {}

        # Common fields in deepfake datasets
        # Adjust based on actual JSON structure
        field_mapping = {
            "label": "label",
            "manipulation": "manipulation_type",
            "video_real": "video_real",
            "audio_real": "audio_real",
            "subject_id": "subject_id",
            "gender": "gender",
            "race": "race",
            "duration": "duration",
        }

        for json_key, our_key in field_mapping.items():
            if json_key in metadata:
                parsed[our_key] = metadata[json_key]

        # Infer 4-class label from video_real and audio_real flags
        if "video_real" in parsed and "audio_real" in parsed:
            video_real = parsed["video_real"]
            audio_real = parsed["audio_real"]

            if video_real and audio_real:
                parsed["class_label"] = 0  # Real-Real
            elif video_real and not audio_real:
                parsed["class_label"] = 1  # Real-Fake
            elif not video_real and audio_real:
                parsed["class_label"] = 2  # Fake-Real
            else:
                parsed["class_label"] = 3  # Fake-Fake

        return parsed

    def create_splits(
        self,
        samples: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify_by: str = "class_label",
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits

        Args:
            samples: List of sample dictionaries
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            stratify_by: Column to stratify split (ensure balanced classes)
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        df = pd.DataFrame(samples)
        logger.info(f"Total samples: {len(df)}")

        # Check if stratify column exists
        if stratify_by not in df.columns:
            logger.warning(f"Stratify column '{stratify_by}' not found. Using random split.")
            stratify_col = None
        else:
            stratify_col = df[stratify_by]
            logger.info(f"Class distribution:\n{stratify_col.value_counts()}")

        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            stratify=stratify_col,
            random_state=random_state
        )

        # Second split: val vs test
        if stratify_col is not None:
            temp_stratify = temp_df[stratify_by]
        else:
            temp_stratify = None

        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_stratify,
            random_state=random_state
        )

        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Val samples: {len(val_df)}")
        logger.info(f"Test samples: {len(test_df)}")

        return train_df, val_df, test_df

    def save_manifests(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):
        """Save train/val/test manifests as CSV files"""
        train_path = self.output_dir / "train.csv"
        val_path = self.output_dir / "val.csv"
        test_path = self.output_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Saved train manifest: {train_path}")
        logger.info(f"Saved val manifest: {val_path}")
        logger.info(f"Saved test manifest: {test_path}")

    def get_dataset_stats(self, df: pd.DataFrame) -> Dict:
        """Get statistics about dataset"""
        stats = {
            "total_samples": len(df),
            "has_metadata": df["has_metadata"].sum() if "has_metadata" in df else 0,
        }

        if "class_label" in df.columns:
            stats["class_distribution"] = df["class_label"].value_counts().to_dict()

        if "manipulation_type" in df.columns:
            stats["manipulation_types"] = df["manipulation_type"].value_counts().to_dict()

        if "duration" in df.columns:
            stats["avg_duration"] = df["duration"].mean()

        return stats


def main():
    """Main analysis script"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze dataset and create splits")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="../dataset/extracted/train",
        help="Root directory of extracted dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/manifests",
        help="Output directory for manifest CSV files"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = DatasetAnalyzer(args.dataset_root, args.output_dir)

    # Scan dataset
    samples = analyzer.scan_dataset()

    if not samples:
        logger.error("No samples found!")
        return 1

    # Create splits
    train_df, val_df, test_df = analyzer.create_splits(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )

    # Save manifests
    analyzer.save_manifests(train_df, val_df, test_df)

    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)

    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{split_name} Set:")
        stats = analyzer.get_dataset_stats(split_df)
        for key, value in stats.items():
            print(f"  {key}: {value}")

    print("\n✓ Analysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
