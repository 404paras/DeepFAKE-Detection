#!/usr/bin/env python3
"""
Dataset Validation and Cleaning Script
Validates all videos in manifests and creates cleaned versions
"""

import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import sys

def validate_single_video(video_path: Path) -> tuple[bool, str]:
    """
    Validate a single video file

    Returns:
        (is_valid, error_message)
    """
    if not video_path.exists():
        return False, "File not found"

    if video_path.stat().st_size == 0:
        return False, "Empty file"

    try:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return False, "Cannot open with OpenCV"

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count == 0:
            return False, "Zero frames"

        # Try reading first frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return False, "Cannot read frames"

        return True, "OK"

    except Exception as e:
        return False, f"Exception: {str(e)}"

def validate_manifest(manifest_path: Path, dataset_root: Path, split_name: str):
    """
    Validate all videos in a manifest

    Returns:
        cleaned_manifest, validation_stats
    """
    print(f"\n{'='*60}")
    print(f"Validating {split_name} split: {manifest_path}")
    print(f"{'='*60}")

    # Load manifest
    manifest = pd.read_csv(manifest_path)
    print(f"Total samples in manifest: {len(manifest)}")

    # Validation results
    valid_rows = []
    invalid_videos = []
    error_stats = {}

    # Validate each video
    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc=f"Validating {split_name}"):
        video_path = dataset_root / row['video_path']

        is_valid, error_msg = validate_single_video(video_path)

        if is_valid:
            valid_rows.append(row)
        else:
            invalid_videos.append({
                'idx': idx,
                'video_path': str(row['video_path']),
                'video_name': row.get('video_name', ''),
                'error': error_msg
            })

            # Track error types
            error_stats[error_msg] = error_stats.get(error_msg, 0) + 1

    # Create cleaned manifest
    cleaned_manifest = pd.DataFrame(valid_rows)

    # Statistics
    valid_count = len(valid_rows)
    invalid_count = len(invalid_videos)
    total_count = len(manifest)

    stats = {
        'split': split_name,
        'total': total_count,
        'valid': valid_count,
        'invalid': invalid_count,
        'valid_percentage': (valid_count / total_count * 100) if total_count > 0 else 0,
        'error_types': error_stats,
        'invalid_videos': invalid_videos
    }

    # Print summary
    print(f"\n{split_name.upper()} VALIDATION RESULTS:")
    print(f"  Total:   {total_count}")
    print(f"  Valid:   {valid_count} ({stats['valid_percentage']:.1f}%)")
    print(f"  Invalid: {invalid_count} ({invalid_count/total_count*100:.1f}%)")

    if error_stats:
        print(f"\n  Error breakdown:")
        for error, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {error}: {count}")

    return cleaned_manifest, stats

def main():
    parser = argparse.ArgumentParser(description="Validate and clean dataset manifests")
    parser.add_argument(
        "--manifests-dir",
        type=str,
        default="data/manifests",
        help="Directory containing manifest CSV files"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="dataset/extracted/train",
        help="Root directory of dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/manifests_cleaned",
        help="Directory to save cleaned manifests"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to validate"
    )

    args = parser.parse_args()

    # Setup paths
    manifests_dir = Path(args.manifests_dir)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*60}")
    print(f"DATASET VALIDATION AND CLEANING")
    print(f"{'='*60}")
    print(f"Manifests directory: {manifests_dir}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output directory: {output_dir}")
    print(f"Splits: {', '.join(args.splits)}")

    # Validate each split
    all_stats = {}

    for split in args.splits:
        manifest_path = manifests_dir / f"{split}.csv"

        if not manifest_path.exists():
            print(f"\n⚠️  Warning: Manifest not found: {manifest_path}")
            continue

        # Validate
        cleaned_manifest, stats = validate_manifest(manifest_path, dataset_root, split)
        all_stats[split] = stats

        # Save cleaned manifest
        output_path = output_dir / f"{split}.csv"
        cleaned_manifest.to_csv(output_path, index=False)
        print(f"✓ Saved cleaned manifest: {output_path}")

        # Save error log
        if stats['invalid_videos']:
            error_log_path = output_dir / f"{split}_errors.json"
            with open(error_log_path, 'w') as f:
                json.dump(stats['invalid_videos'], f, indent=2)
            print(f"✓ Saved error log: {error_log_path}")

    # Generate overall report
    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*60}")

    total_all = sum(s['total'] for s in all_stats.values())
    valid_all = sum(s['valid'] for s in all_stats.values())
    invalid_all = sum(s['invalid'] for s in all_stats.values())

    print(f"Total samples: {total_all}")
    print(f"Valid samples: {valid_all} ({valid_all/total_all*100:.1f}%)")
    print(f"Invalid samples: {invalid_all} ({invalid_all/total_all*100:.1f}%)")

    print(f"\nPer-split breakdown:")
    for split, stats in all_stats.items():
        print(f"  {split:6s}: {stats['valid']:5d}/{stats['total']:5d} valid ({stats['valid_percentage']:5.1f}%)")

    # Save full report
    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset_root': str(dataset_root),
        'summary': {
            'total': total_all,
            'valid': valid_all,
            'invalid': invalid_all,
            'valid_percentage': (valid_all / total_all * 100) if total_all > 0 else 0
        },
        'splits': all_stats
    }

    report_path = output_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Full report saved: {report_path}")
    print(f"\n{'='*60}")
    print(f"NEXT STEPS:")
    print(f"{'='*60}")
    print(f"1. Review validation report: {report_path}")
    print(f"2. Update config.yaml to use cleaned manifests:")
    print(f"   data:")
    print(f"     manifests_path: \"{output_dir.name}\"")
    print(f"3. Resume training with cleaned dataset")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
