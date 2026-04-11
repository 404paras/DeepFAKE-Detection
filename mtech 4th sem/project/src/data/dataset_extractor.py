"""
Dataset Extractor for Multimodal Deepfake Detection
Extracts split zip archives and organizes dataset structure
"""

import os
import subprocess
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetExtractor:
    """Extract split zip archives using 7z"""

    def __init__(self, dataset_dir: str, output_dir: str = None):
        """
        Args:
            dataset_dir: Directory containing split zip files
            output_dir: Directory to extract files (default: dataset_dir/extracted)
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir) if output_dir else self.dataset_dir / "extracted"

        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.dataset_dir}")

    def find_split_archives(self):
        """Find all split zip archive sets in dataset directory"""
        archives = {}

        for file in self.dataset_dir.glob("*.zip.*"):
            # Extract base name (e.g., train.zip from train.zip.001)
            base_name = file.stem  # train.zip
            if base_name not in archives:
                archives[base_name] = []
            archives[base_name].append(file)

        # Sort parts numerically
        for base_name in archives:
            archives[base_name] = sorted(archives[base_name])

        return archives

    def extract_archive(self, archive_name: str, parts: list):
        """
        Extract a split zip archive using 7z

        Args:
            archive_name: Base name of archive (e.g., 'train.zip')
            parts: List of split part files
        """
        logger.info(f"Extracting {archive_name} ({len(parts)} parts)...")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use 7z to extract (automatically handles split archives)
        first_part = parts[0]

        try:
            # Extract using 7z
            cmd = [
                "7z",
                "x",  # Extract with full paths
                str(first_part),
                f"-o{self.output_dir}",
                "-y"  # Yes to all prompts
            ]

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            logger.info(f"Successfully extracted {archive_name}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract {archive_name}")
            logger.error(f"Error: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("7z command not found. Please install p7zip:")
            logger.error("  macOS: brew install p7zip")
            logger.error("  Linux: sudo apt-get install p7zip-full")
            return False

    def extract_all(self):
        """Extract all split archives found in dataset directory"""
        archives = self.find_split_archives()

        if not archives:
            logger.warning("No split archives found")
            return False

        logger.info(f"Found {len(archives)} archive(s) to extract")

        success_count = 0
        for archive_name, parts in archives.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Archive: {archive_name}")
            logger.info(f"Parts: {len(parts)}")
            logger.info(f"Total size: {sum(p.stat().st_size for p in parts) / 1e9:.2f} GB")

            if self.extract_archive(archive_name, parts):
                success_count += 1

        logger.info(f"\n{'='*60}")
        logger.info(f"Extraction complete: {success_count}/{len(archives)} successful")
        logger.info(f"Output directory: {self.output_dir}")

        return success_count == len(archives)

    def get_extraction_stats(self):
        """Get statistics about extracted dataset"""
        if not self.output_dir.exists():
            return None

        stats = {
            "total_files": 0,
            "total_size_gb": 0,
            "video_files": 0,
            "json_files": 0,
            "directories": 0
        }

        for root, dirs, files in os.walk(self.output_dir):
            stats["directories"] += len(dirs)
            for file in files:
                file_path = Path(root) / file
                stats["total_files"] += 1
                stats["total_size_gb"] += file_path.stat().st_size / 1e9

                if file.endswith(".mp4"):
                    stats["video_files"] += 1
                elif file.endswith(".json"):
                    stats["json_files"] += 1

        return stats


def main():
    """Main extraction script"""
    import argparse

    parser = argparse.ArgumentParser(description="Extract split zip archives")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="../dataset",
        help="Directory containing split zip files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: dataset_dir/extracted)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show extraction statistics only"
    )

    args = parser.parse_args()

    extractor = DatasetExtractor(args.dataset_dir, args.output_dir)

    if args.stats:
        stats = extractor.get_extraction_stats()
        if stats:
            print("\nDataset Statistics:")
            print(f"  Total files: {stats['total_files']:,}")
            print(f"  Video files: {stats['video_files']:,}")
            print(f"  JSON files: {stats['json_files']:,}")
            print(f"  Directories: {stats['directories']:,}")
            print(f"  Total size: {stats['total_size_gb']:.2f} GB")
        else:
            print("Dataset not extracted yet")
    else:
        success = extractor.extract_all()

        if success:
            print("\n✓ Extraction successful!")
            stats = extractor.get_extraction_stats()
            if stats:
                print(f"\nExtracted {stats['video_files']:,} video files")
        else:
            print("\n✗ Extraction failed")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
