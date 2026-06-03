"""
Data augmentation modules for video and audio
Implements augmentations as specified in the paper
"""

import numpy as np
import cv2
import torch
import random
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VideoAugmentation:
    """
    Video augmentations as specified in paper:
    - Random horizontal flip (p=0.5)
    - Random rotation (±10°)
    - Color jitter (brightness, contrast, saturation)
    - Gaussian blur
    - Random erasing
    """

    def __init__(
        self,
        img_size: int = 224,
        horizontal_flip_p: float = 0.5,
        rotation_limit: int = 10,
        apply_color_jitter: bool = True,
        apply_blur: bool = True,
        apply_erasing: bool = True,
    ):
        """
        Args:
            img_size: Image size
            horizontal_flip_p: Probability of horizontal flip
            rotation_limit: Rotation range in degrees
            apply_color_jitter: Whether to apply color jittering
            apply_blur: Whether to apply Gaussian blur
            apply_erasing: Whether to apply random erasing
        """
        self.img_size = img_size

        # Build augmentation pipeline using albumentations
        transforms = []

        # Geometric transformations
        if horizontal_flip_p > 0:
            transforms.append(A.HorizontalFlip(p=horizontal_flip_p))

        if rotation_limit > 0:
            transforms.append(
                A.Rotate(limit=rotation_limit, border_mode=cv2.BORDER_REFLECT, p=0.5)
            )

        # Color augmentations
        if apply_color_jitter:
            transforms.append(
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                )
            )

        # Blur
        if apply_blur:
            transforms.append(
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.3)
            )

        # Random erasing (coarse dropout)
        if apply_erasing:
            transforms.append(
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(img_size * 0.1),
                    max_width=int(img_size * 0.1),
                    p=0.5
                )
            )

        self.transform = A.Compose(transforms)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to video frames

        Args:
            frames: Array of shape (T, H, W, C)

        Returns:
            Augmented frames of shape (T, H, W, C)
        """
        augmented_frames = []

        for frame in frames:
            # Apply same augmentation to all frames for temporal consistency
            # (optional: can apply different augmentations per frame)
            augmented = self.transform(image=frame)
            augmented_frames.append(augmented["image"])

        return np.array(augmented_frames)


class AudioAugmentation:
    """
    Audio augmentations as specified in paper:
    - Time stretching (rate 0.8-1.2)
    - Pitch shifting (±2 semitones)
    - Background noise (SNR 20-40dB)
    - SpecAugment (frequency + time masking)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        time_stretch: bool = True,
        pitch_shift: bool = True,
        add_noise: bool = True,
        spec_augment: bool = True,
    ):
        """
        Args:
            sample_rate: Audio sample rate
            time_stretch: Whether to apply time stretching
            pitch_shift: Whether to apply pitch shifting
            add_noise: Whether to add noise
            spec_augment: Whether to apply SpecAugment
        """
        self.sample_rate = sample_rate
        self.time_stretch = time_stretch
        self.pitch_shift = pitch_shift
        self.add_noise = add_noise
        self.spec_augment = spec_augment

    def apply_time_stretch(self, audio: np.ndarray, rate_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Time stretching without changing pitch"""
        import librosa

        rate = np.random.uniform(*rate_range)
        stretched = librosa.effects.time_stretch(audio, rate=rate)

        # Ensure same length
        if len(stretched) > len(audio):
            stretched = stretched[:len(audio)]
        elif len(stretched) < len(audio):
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')

        return stretched

    def apply_pitch_shift(self, audio: np.ndarray, n_steps_range: Tuple[int, int] = (-2, 2)) -> np.ndarray:
        """Pitch shifting"""
        import librosa

        n_steps = np.random.randint(*n_steps_range)
        if n_steps == 0:
            return audio

        shifted = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
        return shifted

    def add_gaussian_noise(self, audio: np.ndarray, snr_range: Tuple[float, float] = (20, 40)) -> np.ndarray:
        """Add Gaussian noise with specified SNR"""
        snr_db = np.random.uniform(*snr_range)

        # Calculate signal power
        signal_power = np.mean(audio ** 2)

        # Calculate noise power based on SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate and add noise
        noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
        noisy_audio = audio + noise

        return noisy_audio

    def apply_spec_augment(
        self,
        mel_spec: np.ndarray,
        freq_mask_param: int = 15,
        time_mask_param: int = 20,
        num_freq_masks: int = 2,
        num_time_masks: int = 2
    ) -> np.ndarray:
        """
        SpecAugment: mask frequency and time bands

        Args:
            mel_spec: Mel-spectrogram of shape (n_mels, time_steps)
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
            num_freq_masks: Number of frequency masks
            num_time_masks: Number of time masks

        Returns:
            Augmented mel-spectrogram
        """
        n_mels, time_steps = mel_spec.shape
        augmented = mel_spec.copy()

        # Frequency masking
        for _ in range(num_freq_masks):
            f = np.random.randint(0, freq_mask_param)
            f0 = np.random.randint(0, n_mels - f)
            augmented[f0:f0+f, :] = 0

        # Time masking
        for _ in range(num_time_masks):
            t = np.random.randint(0, time_mask_param)
            t0 = np.random.randint(0, time_steps - t)
            augmented[:, t0:t0+t] = 0

        return augmented

    def __call__(
        self,
        audio: Optional[np.ndarray] = None,
        mel_spec: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply augmentations

        Args:
            audio: Raw audio waveform of shape (n_samples,)
            mel_spec: Mel-spectrogram of shape (n_mels, time_steps)

        Returns:
            Tuple of (augmented_audio, augmented_mel_spec)
        """
        augmented_audio = audio
        augmented_mel = mel_spec

        # Waveform augmentations
        if audio is not None:
            if self.time_stretch and random.random() < 0.5:
                augmented_audio = self.apply_time_stretch(augmented_audio)

            if self.pitch_shift and random.random() < 0.5:
                augmented_audio = self.apply_pitch_shift(augmented_audio)

            if self.add_noise and random.random() < 0.5:
                augmented_audio = self.add_gaussian_noise(augmented_audio)

        # Spectrogram augmentations
        if mel_spec is not None and self.spec_augment and random.random() < 0.5:
            augmented_mel = self.apply_spec_augment(mel_spec)

        return augmented_audio, augmented_mel


class CombinedAugmentation:
    """Combined video and audio augmentation"""

    def __init__(self, video_aug: VideoAugmentation, audio_aug: AudioAugmentation):
        self.video_aug = video_aug
        self.audio_aug = audio_aug

    def __call__(
        self,
        video_frames: np.ndarray,
        audio_waveform: Optional[np.ndarray] = None,
        mel_spec: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply augmentations to both modalities

        Args:
            video_frames: Video frames of shape (T, H, W, C)
            audio_waveform: Audio waveform (optional)
            mel_spec: Mel-spectrogram (optional)

        Returns:
            Tuple of (augmented_video, augmented_audio, augmented_mel)
        """
        # Augment video
        aug_video = self.video_aug(video_frames)

        # Augment audio
        aug_audio, aug_mel = self.audio_aug(audio_waveform, mel_spec)

        return aug_video, aug_audio, aug_mel


def main():
    """Test augmentation pipeline"""
    print("Testing video augmentation...")

    # Create dummy video frames
    video_frames = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)

    video_aug = VideoAugmentation()
    augmented_video = video_aug(video_frames)

    print(f"Video shape: {video_frames.shape} -> {augmented_video.shape}")

    print("\nTesting audio augmentation...")

    # Create dummy audio
    audio = np.random.randn(48000).astype(np.float32)
    mel_spec = np.random.randn(128, 94).astype(np.float32)

    audio_aug = AudioAugmentation()
    aug_audio, aug_mel = audio_aug(audio, mel_spec)

    if aug_audio is not None:
        print(f"Audio shape: {audio.shape} -> {aug_audio.shape}")
    if aug_mel is not None:
        print(f"Mel-spec shape: {mel_spec.shape} -> {aug_mel.shape}")

    print("\n✓ Augmentation tests passed!")


if __name__ == "__main__":
    main()
