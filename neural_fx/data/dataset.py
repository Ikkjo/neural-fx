import warnings
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..preprocessing.latency import LatencyCalibration
from .audio import AudioPair, load_audio_pair


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
        input_context: int = 0,
        include_partial_segment: bool = False,
        *,
        _audio_pair: AudioPair | None = None,
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
            input_context: Preceding input samples included without matching targets.
            include_partial_segment: Include a shorter final sequential segment.
        """
        super().__init__()
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.random_segments = random_segments
        self.transform = transform
        self.latency_calibration = latency_calibration
        self.input_context = input_context
        self.include_partial_segment = include_partial_segment

        audio_pair = _audio_pair or load_audio_pair(
            input_path,
            target_path,
            sample_rate=sample_rate,
            normalize=normalize,
            latency_calibration=latency_calibration,
        )
        self.input_audio = audio_pair.input_audio
        self.target_audio = audio_pair.target_audio

        available = max(0, self.input_audio.shape[-1] - input_context)
        self.num_segments = available // segment_length
        if include_partial_segment and available % segment_length:
            self.num_segments += 1
        if self.num_segments == 0:
            warnings.warn(
                f"Audio length ({self.input_audio.shape[-1]}) is shorter than "
                f"segment_length ({segment_length}). Creating empty dataset."
            )
        self.total_length = min(available, self.num_segments * segment_length)

    def __len__(self) -> int:
        return self.num_segments

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if self.random_segments:
            # Random segment sampling
            max_start = self.input_audio.shape[-1] - self.segment_length
            target_start = torch.randint(
                self.input_context, max_start + 1, (1,)
            ).item()
        else:
            # Sequential segment sampling
            target_start = self.input_context + idx * self.segment_length

        end = target_start + self.segment_length

        x = self.input_audio[..., target_start - self.input_context : end]
        y = self.target_audio[..., target_start:end]

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
        audio_pair = load_audio_pair(
            input_path,
            target_path,
            sample_rate=sample_rate,
            normalize=normalize,
        )

        # Calculate split point
        num_segments = audio_pair.num_samples // segment_length
        val_segments = int(num_segments * val_ratio)
        train_segments = num_segments - val_segments

        train_length = train_segments * segment_length
        val_start = train_length
        val_end = val_start + val_segments * segment_length

        train_pair = AudioPair(
            audio_pair.input_audio[..., :train_length],
            audio_pair.target_audio[..., :train_length],
            sample_rate,
        )
        val_pair = AudioPair(
            audio_pair.input_audio[..., val_start:val_end],
            audio_pair.target_audio[..., val_start:val_end],
            sample_rate,
        )

        train_dataset = cls(
            input_path,
            target_path,
            segment_length=segment_length,
            sample_rate=sample_rate,
            normalize=normalize,
            random_segments=random_segments,
            transform=transform,
            _audio_pair=train_pair,
        )
        val_dataset = cls(
            input_path,
            target_path,
            segment_length=segment_length,
            sample_rate=sample_rate,
            normalize=normalize,
            random_segments=False,
            transform=None,
            _audio_pair=val_pair,
        )

        return train_dataset, val_dataset
