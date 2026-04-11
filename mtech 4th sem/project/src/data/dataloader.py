"""
PyTorch Dataset and DataLoader for Multimodal Deepfake Detection
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

from .preprocessing import VideoPreprocessor, AudioPreprocessor
from .augmentations import VideoAugmentation, AudioAugmentation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for multimodal deepfake detection

    Loads video and audio, applies preprocessing and augmentation
    """

    def __init__(
        self,
        manifest_path: str,
        dataset_root: str,
        video_preprocessor: VideoPreprocessor,
        audio_preprocessor: AudioPreprocessor,
        video_augmentation: Optional[VideoAugmentation] = None,
        audio_augmentation: Optional[AudioAugmentation] = None,
        is_training: bool = True
    ):
        """
        Args:
            manifest_path: Path to CSV manifest file
            dataset_root: Root directory of dataset
            video_preprocessor: Video preprocessing instance
            audio_preprocessor: Audio preprocessing instance
            video_augmentation: Video augmentation (optional, for training)
            audio_augmentation: Audio augmentation (optional, for training)
            is_training: Whether this is training mode
        """
        self.dataset_root = Path(dataset_root)
        self.video_preprocessor = video_preprocessor
        self.audio_preprocessor = audio_preprocessor
        self.video_augmentation = video_augmentation if is_training else None
        self.audio_augmentation = audio_augmentation if is_training else None
        self.is_training = is_training

        # Load manifest
        self.manifest = pd.read_csv(manifest_path)
        logger.info(f"Loaded manifest with {len(self.manifest)} samples")

        # Check for required columns
        if "video_path" not in self.manifest.columns:
            raise ValueError("Manifest must contain 'video_path' column")

        if "class_label" not in self.manifest.columns:
            logger.warning("No 'class_label' column found. Using dummy labels.")
            self.manifest["class_label"] = 0

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, Dict]:
        """
        Get a single sample

        Returns:
            Tuple of (video_tensor, audio_tensor, label, metadata)
        """
        # Get sample info
        sample = self.manifest.iloc[idx]
        video_path = self.dataset_root / sample["video_path"]
        label = int(sample["class_label"])

        try:
            # Preprocess video
            video_tensor = self.video_preprocessor.preprocess(str(video_path))

            # Preprocess audio (from same video file)
            audio_tensor = self.audio_preprocessor.preprocess(str(video_path))

            # Apply augmentations if training
            if self.is_training:
                if self.video_augmentation is not None:
                    # Convert to numpy for augmentation
                    video_np = video_tensor.permute(0, 2, 3, 1).numpy()  # (T, C, H, W) -> (T, H, W, C)
                    video_np = (video_np * 255).astype('uint8')  # Denormalize for albumentations

                    # Augment
                    video_np = self.video_augmentation(video_np)

                    # Convert back to tensor
                    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0

                if self.audio_augmentation is not None:
                    # Apply spec augment to mel-spectrogram
                    mel_np = audio_tensor.squeeze(0).numpy()  # (1, H, W) -> (H, W)
                    _, aug_mel = self.audio_augmentation(audio=None, mel_spec=mel_np)
                    if aug_mel is not None:
                        audio_tensor = torch.from_numpy(aug_mel).unsqueeze(0).float()

            # Metadata
            metadata = {
                "video_path": str(video_path),
                "video_name": sample.get("video_name", ""),
                "idx": idx
            }

            return video_tensor, audio_tensor, label, metadata

        except Exception as e:
            logger.error(f"Error loading sample {idx} ({video_path}): {e}")
            # Return dummy data on error
            video_tensor = torch.zeros(16, 3, 224, 224)
            audio_tensor = torch.zeros(1, 128, 94)
            metadata = {"video_path": str(video_path), "error": str(e), "idx": idx}
            return video_tensor, audio_tensor, label, metadata


def collate_fn(batch):
    """
    Custom collate function to handle variable-size data

    Args:
        batch: List of tuples (video, audio, label, metadata)

    Returns:
        Batched tensors
    """
    videos, audios, labels, metadatas = zip(*batch)

    # Stack tensors
    videos = torch.stack(videos)
    audios = torch.stack(audios)
    labels = torch.tensor(labels, dtype=torch.long)

    return videos, audios, labels, metadatas


def create_dataloaders(
    train_manifest: str,
    val_manifest: str,
    test_manifest: str,
    dataset_root: str,
    config: Dict,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders

    Args:
        train_manifest: Path to training manifest
        val_manifest: Path to validation manifest
        test_manifest: Path to test manifest
        dataset_root: Root directory of dataset
        config: Configuration dictionary
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Initialize preprocessors
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

    # Initialize augmentations for training
    video_aug = None
    audio_aug = None

    if config["preprocessing"]["video"]["augmentation"]["enabled"]:
        video_aug = VideoAugmentation(
            img_size=config["preprocessing"]["video"]["img_size"],
            horizontal_flip_p=config["preprocessing"]["video"]["augmentation"]["horizontal_flip"],
            rotation_limit=config["preprocessing"]["video"]["augmentation"]["rotation"],
            apply_color_jitter=config["preprocessing"]["video"]["augmentation"]["color_jitter"],
            apply_blur=config["preprocessing"]["video"]["augmentation"]["gaussian_blur"],
            apply_erasing=config["preprocessing"]["video"]["augmentation"]["random_erasing"]
        )

    if config["preprocessing"]["audio"]["augmentation"]["enabled"]:
        audio_aug = AudioAugmentation(
            sample_rate=config["preprocessing"]["audio"]["sample_rate"],
            time_stretch=True,
            pitch_shift=True,
            add_noise=True,
            spec_augment=config["preprocessing"]["audio"]["augmentation"]["spec_augment"]
        )

    # Create datasets
    train_dataset = DeepfakeDataset(
        manifest_path=train_manifest,
        dataset_root=dataset_root,
        video_preprocessor=video_prep,
        audio_preprocessor=audio_prep,
        video_augmentation=video_aug,
        audio_augmentation=audio_aug,
        is_training=True
    )

    val_dataset = DeepfakeDataset(
        manifest_path=val_manifest,
        dataset_root=dataset_root,
        video_preprocessor=video_prep,
        audio_preprocessor=audio_prep,
        video_augmentation=None,
        audio_augmentation=None,
        is_training=False
    )

    test_dataset = DeepfakeDataset(
        manifest_path=test_manifest,
        dataset_root=dataset_root,
        video_preprocessor=video_prep,
        audio_preprocessor=audio_prep,
        video_augmentation=None,
        audio_augmentation=None,
        is_training=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    logger.info(f"Created dataloaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def main():
    """Test dataloader"""
    import yaml

    # Load config
    config_path = "../../configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_manifest="../data/manifests/train.csv",
        val_manifest="../data/manifests/val.csv",
        test_manifest="../data/manifests/test.csv",
        dataset_root="../../dataset/extracted/train",
        config=config,
        num_workers=2
    )

    # Test loading a batch
    print("\nLoading a batch...")
    for videos, audios, labels, metadatas in train_loader:
        print(f"Video batch shape: {videos.shape}")
        print(f"Audio batch shape: {audios.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")
        print(f"Metadata: {metadatas[0]}")
        break

    print("\n✓ Dataloader test passed!")


if __name__ == "__main__":
    main()
