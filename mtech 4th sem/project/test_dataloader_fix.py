#!/usr/bin/env python3
"""Quick test to verify dataloader works with fixed paths"""

import sys
sys.path.insert(0, 'src')

import yaml
from pathlib import Path

# Load config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("=" * 60)
print("Testing Dataloader with Fixed Paths")
print("=" * 60)

# Check paths
dataset_path = Path(config['data']['dataset_path'])
manifests_path = Path(config['data']['manifests_path'])

print(f"\nDataset path: {dataset_path}")
print(f"Dataset exists: {dataset_path.exists()}")

print(f"\nManifests path: {manifests_path}")
print(f"Manifests exists: {manifests_path.exists()}")

# Check manifest
train_manifest = manifests_path / "train.csv"
print(f"\nTrain manifest: {train_manifest}")
print(f"Train manifest exists: {train_manifest.exists()}")

if train_manifest.exists():
    import pandas as pd
    df = pd.read_csv(train_manifest)
    print(f"Train samples: {len(df)}")
    
    # Check first video
    first_video = dataset_path / df.iloc[0]['video_path']
    print(f"\nFirst video: {first_video}")
    print(f"First video exists: {first_video.exists()}")
    
    if first_video.exists():
        import cv2
        cap = cv2.VideoCapture(str(first_video))
        can_open = cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if can_open else 0
        cap.release()
        
        print(f"Can open with OpenCV: {can_open}")
        print(f"Frame count: {frame_count}")
        
        if can_open and frame_count > 0:
            print("\n✅ SUCCESS! Dataloader paths are correctly configured")
        else:
            print("\n❌ ERROR: Video exists but cannot be read")
    else:
        print("\n❌ ERROR: Video file not found")
else:
    print("\n❌ ERROR: Manifest not found")

print("=" * 60)
