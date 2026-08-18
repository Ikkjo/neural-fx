from pathlib import Path
import warnings
from typing import Callable

from torch import Tensor
from torch.utils.data import Dataset
import torch
import torchaudio

from ..preprocessing.latency import LatencyCalibrator, LatencyCalibration


class AudioDataset(Dataset):
    """Dataset for input/target audio file pairs with optional random segment sampling."""

    def __init__(
        self,
        input_path: str | Path,
        target_path: str | Path,
        segment_length: int = 8192,
        sample_rate: int = 48000,
        normalize: bool = True,
        random_segments: bool = False,
        transform: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None,
        latency_calibration: LatencyCalibration | None = None,
    ):
        """
        Args:
            input_path: Path to input audio file (e.g., dry guitar signal).
            target_path: Path to target audio file (e.g., processed amp signal).
            segment_length: Number of samples per segment.
            sample_rate: Target sample rate (audio will be resampled if needed).
            normalize: Whether to normalize audio to [-1, 1] range.
            random_segments: If True, randomly sample segments instead of sequential.
            transform: Optional transform to apply to (input, target) pairs.
            latency_calibration: Optional latency calibration to apply.
        """
        super().__init__()
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.random_segments = random_segments
        self.transform = transform
        self.latency_calibration = latency_calibration

        input_path = Path(input_path)
        target_path = Path(target_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Target file not found: {target_path}")

        self.input_audio = self._load_audio(input_path)
        self.target_audio = self._load_audio(target_path)

        if self.normalize:
            pair_max = torch.maximum(
                self.input_audio.abs().max(), self.target_audio.abs().max()
            )
            if pair_max > 0:
                self.input_audio = self.input_audio / pair_max
                self.target_audio = self.target_audio / pair_max

        # Apply latency compensation if provided
        if latency_calibration is not None:
            calibrator = LatencyCalibrator()
            self.input_audio, self.target_audio = calibrator.compensate(
                self.input_audio, self.target_audio, latency_calibration
            )
            if latency_calibration.delay_samples != 0:
                warnings.warn(
                    f"Applied latency compensation of {latency_calibration.delay_samples} samples "
                    f"(correlation score: {latency_calibration.correlation_score:.4f})"
                )

        if self.input_audio.shape[-1] != self.target_audio.shape[-1]:
            min_len = min(self.input_audio.shape[-1], self.target_audio.shape[-1])
            warnings.warn(
                f"Input and target audio lengths differ "
                f"({self.input_audio.shape[-1]} vs {self.target_audio.shape[-1]}). "
                f"Truncating to {min_len} samples."
            )
            self.input_audio = self.input_audio[..., :min_len]
            self.target_audio = self.target_audio[..., :min_len]

        self.num_segments = self.input_audio.shape[-1] // segment_length
        if self.num_segments == 0:
            warnings.warn(
                f"Audio length ({self.input_audio.shape[-1]}) is shorter than "
                f"segment_length ({segment_length}). Creating empty dataset."
            )
        self.total_length = self.num_segments * segment_length

    def _load_audio(self, path: Path) -> Tensor:
        audio, sr = torchaudio.load(str(path))

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        return audio

    def __len__(self) -> int:
        return self.num_segments

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if self.random_segments:
            # Random segment sampling
            max_start = self.input_audio.shape[-1] - self.segment_length
            start = torch.randint(0, max_start + 1, (1,)).item()
        else:
            # Sequential segment sampling
            start = idx * self.segment_length

        end = start + self.segment_length

        x = self.input_audio[..., start:end]
        y = self.target_audio[..., start:end]

        x = x.squeeze(0)
        y = y.squeeze(0)

        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y

    @classmethod
    def train_val_split(
        cls,
        input_path: str | Path,
        target_path: str | Path,
        segment_length: int = 8192,
        sample_rate: int = 48000,
        normalize: bool = True,
        random_segments: bool = False,
        transform: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None,
        val_ratio: float = 0.1,
    ) -> tuple["AudioDataset", "AudioDataset"]:
        """
        Create train/validation split by splitting the audio temporally.

        Args:
            input_path: Path to input audio file.
            target_path: Path to target audio file.
            segment_length: Number of samples per segment.
            sample_rate: Target sample rate.
            normalize: Whether to normalize audio.
            random_segments: Use random segment sampling.
            transform: Optional transform to apply.
            val_ratio: Fraction of audio to use for validation (from the end).

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        input_path = Path(input_path)
        target_path = Path(target_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Target file not found: {target_path}")

        # Load full audio
        input_audio, sr = torchaudio.load(str(input_path))
        target_audio, _ = torchaudio.load(str(target_path))

        if sr != sample_rate:
            input_audio = torchaudio.functional.resample(input_audio, sr, sample_rate)
            target_audio = torchaudio.functional.resample(target_audio, sr, sample_rate)

        if input_audio.shape[0] > 1:
            input_audio = input_audio.mean(dim=0, keepdim=True)
        if target_audio.shape[0] > 1:
            target_audio = target_audio.mean(dim=0, keepdim=True)

        # Handle length mismatch
        min_len = min(input_audio.shape[-1], target_audio.shape[-1])
        input_audio = input_audio[..., :min_len]
        target_audio = target_audio[..., :min_len]

        if normalize:
            combined_max = max(input_audio.abs().max(), target_audio.abs().max())
            if combined_max > 0:
                input_audio = input_audio / combined_max
                target_audio = target_audio / combined_max

        # Calculate split point
        num_segments = min_len // segment_length
        val_segments = int(num_segments * val_ratio)
        train_segments = num_segments - val_segments

        train_length = train_segments * segment_length
        val_start = train_length

        # Split audio
        train_input = input_audio[..., :train_length]
        train_target = target_audio[..., :train_length]
        val_input = input_audio[
            ..., val_start : val_start + val_segments * segment_length
        ]
        val_target = target_audio[
            ..., val_start : val_start + val_segments * segment_length
        ]

        # Create datasets
        train_dataset = cls.__new__(cls)
        train_dataset.segment_length = segment_length
        train_dataset.sample_rate = sample_rate
        train_dataset.normalize = normalize
        train_dataset.random_segments = random_segments
        train_dataset.transform = transform
        # TODO: Could preserve parent's latency_calibration instead of always setting None,
        # or accept latency_calibration as an explicit parameter to split() method
        train_dataset.latency_calibration = None
        train_dataset.input_audio = train_input
        train_dataset.target_audio = train_target
        train_dataset.num_segments = train_segments
        train_dataset.total_length = train_length

        val_dataset = cls.__new__(cls)
        val_dataset.segment_length = segment_length
        val_dataset.sample_rate = sample_rate
        val_dataset.normalize = normalize
        val_dataset.random_segments = False  # Always sequential for validation
        val_dataset.transform = None  # No augmentation for validation
        # TODO: Could preserve parent's latency_calibration instead of always setting None
        val_dataset.latency_calibration = None
        val_dataset.input_audio = val_input
        val_dataset.target_audio = val_target
        val_dataset.num_segments = val_segments
        val_dataset.total_length = val_segments * segment_length

        return train_dataset, val_dataset
