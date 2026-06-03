"""
Preprocessing pipeline for video and audio data
Based on paper specifications
"""

import cv2
import numpy as np
import librosa
import soundfile as sf
import torch
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoPreprocessor:
    """
    Video preprocessing as specified in paper:
    - Extract 16 frames uniformly
    - Resize to 224x224
    - Normalize to [0, 1]
    """

    def __init__(
        self,
        num_frames: int = 16,
        img_size: int = 224,
        normalize: bool = True,
        mean: Optional[Tuple[float, float, float]] = (0.485, 0.456, 0.406),
        std: Optional[Tuple[float, float, float]] = (0.229, 0.224, 0.225)
    ):
        """
        Args:
            num_frames: Number of frames to extract
            img_size: Target image size (square)
            normalize: Whether to normalize with ImageNet stats
            mean: Mean for normalization
            std: Std for normalization
        """
        self.num_frames = num_frames
        self.img_size = img_size
        self.normalize = normalize
        self.mean = np.array(mean).reshape(1, 1, 3) if mean else None
        self.std = np.array(std).reshape(1, 1, 3) if std else None

    def extract_frames(self, video_path: str) -> np.ndarray:
        """
        Extract frames uniformly from video

        Args:
            video_path: Path to video file

        Returns:
            Array of shape (num_frames, height, width, 3)
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        if total_frames < self.num_frames:
            logger.warning(
                f"Video has {total_frames} frames, less than requested {self.num_frames}"
            )

        # Calculate frame indices to extract (uniform sampling)
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Repeat frames if video too short
            indices = np.array(
                list(range(total_frames)) * (self.num_frames // total_frames + 1)
            )[:self.num_frames]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                logger.warning(f"Failed to read frame {idx} from {video_path}")
                # Use last valid frame as fallback
                if frames:
                    frames.append(frames[-1])

        cap.release()

        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")

        return np.array(frames)

    def resize_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Resize frames to target size

        Args:
            frames: Array of shape (num_frames, H, W, 3)

        Returns:
            Resized frames of shape (num_frames, img_size, img_size, 3)
        """
        resized = []
        for frame in frames:
            resized_frame = cv2.resize(
                frame,
                (self.img_size, self.img_size),
                interpolation=cv2.INTER_LINEAR
            )
            resized.append(resized_frame)

        return np.array(resized)

    def normalize_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Normalize frames to [0, 1] and apply ImageNet normalization

        Args:
            frames: Array of shape (num_frames, H, W, 3) with values in [0, 255]

        Returns:
            Normalized frames
        """
        # Convert to float and scale to [0, 1]
        frames = frames.astype(np.float32) / 255.0

        # Apply ImageNet normalization if specified
        if self.normalize and self.mean is not None and self.std is not None:
            frames = (frames - self.mean) / self.std

        return frames

    def preprocess(self, video_path: str) -> torch.Tensor:
        """
        Full preprocessing pipeline

        Args:
            video_path: Path to video file

        Returns:
            Tensor of shape (num_frames, 3, img_size, img_size)
        """
        # Extract frames
        frames = self.extract_frames(video_path)

        # Resize
        frames = self.resize_frames(frames)

        # Normalize
        frames = self.normalize_frames(frames)

        # Convert to tensor and transpose to (T, C, H, W)
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)

        return frames_tensor


class AudioPreprocessor:
    """
    Audio preprocessing as specified in paper:
    - Resample to 16kHz
    - Extract 3-second clips (48,000 samples)
    - Compute Mel-spectrogram (128×94)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 3.0,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float = 8000.0
    ):
        """
        Args:
            sample_rate: Target sample rate in Hz
            duration: Duration of audio clip in seconds
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of Mel bands
            f_min: Minimum frequency
            f_max: Maximum frequency
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and resample audio

        Args:
            audio_path: Path to audio file (or video file to extract audio from)

        Returns:
            Audio waveform of shape (n_samples,)
        """
        try:
            # Load audio (suppress PySoundFile warnings)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            # Pad or trim to fixed length
            if len(audio) < self.n_samples:
                # Pad with zeros
                audio = np.pad(audio, (0, self.n_samples - len(audio)), mode='constant')
            else:
                # Trim to duration (take center portion)
                start = (len(audio) - self.n_samples) // 2
                audio = audio[start:start + self.n_samples]

            return audio

        except Exception as e:
            logger.error(f"Failed to load audio from {audio_path}: {e}")
            raise

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Mel-spectrogram

        Args:
            audio: Audio waveform of shape (n_samples,)

        Returns:
            Mel-spectrogram of shape (n_mels, time_steps)
        """
        # Compute Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max
        )

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1] range
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

        return mel_spec_norm

    def preprocess(self, audio_path: str) -> torch.Tensor:
        """
        Full preprocessing pipeline

        Args:
            audio_path: Path to audio file or video file

        Returns:
            Tensor of shape (1, n_mels, time_steps) - expected (1, 128, 94)
        """
        # Load audio
        audio = self.load_audio(audio_path)

        # Compute Mel-spectrogram
        mel_spec = self.compute_mel_spectrogram(audio)

        # Convert to tensor and add channel dimension
        mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).float()

        return mel_tensor


def preprocess_sample(
    video_path: str,
    video_preprocessor: VideoPreprocessor,
    audio_preprocessor: AudioPreprocessor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess both video and audio from a single file

    Args:
        video_path: Path to video file (contains both video and audio)
        video_preprocessor: Video preprocessor instance
        audio_preprocessor: Audio preprocessor instance

    Returns:
        Tuple of (video_tensor, audio_tensor)
    """
    # Process video
    video_tensor = video_preprocessor.preprocess(video_path)

    # Process audio (from same video file)
    audio_tensor = audio_preprocessor.preprocess(video_path)

    return video_tensor, audio_tensor


def main():
    """Test preprocessing pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Test preprocessing pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to test video")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")

    args = parser.parse_args()

    # Initialize preprocessors
    video_prep = VideoPreprocessor()
    audio_prep = AudioPreprocessor()

    # Preprocess
    print(f"Processing: {args.video}")
    video_tensor, audio_tensor = preprocess_sample(
        args.video,
        video_prep,
        audio_prep
    )

    print(f"\nVideo tensor shape: {video_tensor.shape}")
    print(f"Expected: (16, 3, 224, 224)")
    print(f"Audio tensor shape: {audio_tensor.shape}")
    print(f"Expected: (1, 128, ~94)")

    # Save sample outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    torch.save(video_tensor, output_dir / "sample_video.pt")
    torch.save(audio_tensor, output_dir / "sample_audio.pt")

    print(f"\n✓ Preprocessing successful!")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
