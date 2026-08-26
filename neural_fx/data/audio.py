import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from torch import Tensor

from ..preprocessing.latency import LatencyCalibration, LatencyCalibrator


@dataclass(frozen=True)
class AudioPair:
    input_audio: Tensor
    target_audio: Tensor
    sample_rate: int

    @property
    def num_samples(self) -> int:
        return self.input_audio.shape[-1]


def _load_mono(path: Path, sample_rate: int) -> Tensor:
    audio, source_rate = torchaudio.load(path)
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio


def load_audio_pair(
    input_path: str | Path,
    target_path: str | Path,
    *,
    sample_rate: int = 48_000,
    normalize: bool = True,
    latency_calibration: LatencyCalibration | None = None,
    strict: bool = False,
) -> AudioPair:
    """Load an audio pair, optionally preserving both native tensors unchanged."""
    input_path = Path(input_path)
    target_path = Path(target_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    if strict:
        input_audio, input_rate = torchaudio.load(input_path)
        target_audio, target_rate = torchaudio.load(target_path)
        if input_rate != sample_rate or target_rate != sample_rate:
            raise ValueError(
                f"Expected {sample_rate} Hz; input={input_rate}, target={target_rate}"
            )
        return AudioPair(input_audio, target_audio, sample_rate)

    input_audio = _load_mono(input_path, sample_rate)
    target_audio = _load_mono(target_path, sample_rate)

    if normalize:
        pair_max = torch.maximum(input_audio.abs().max(), target_audio.abs().max())
        if pair_max > 0:
            input_audio = input_audio / pair_max
            target_audio = target_audio / pair_max

    if latency_calibration is not None:
        input_audio, target_audio = LatencyCalibrator().compensate(
            input_audio,
            target_audio,
            latency_calibration,
        )
        if latency_calibration.delay_samples != 0:
            warnings.warn(
                f"Applied latency compensation of {latency_calibration.delay_samples} "
                f"samples (correlation score: {latency_calibration.correlation_score:.4f})"
            )

    if input_audio.shape[-1] != target_audio.shape[-1]:
        input_length = input_audio.shape[-1]
        target_length = target_audio.shape[-1]
        aligned_length = min(input_length, target_length)
        warnings.warn(
            f"Input and target audio lengths differ ({input_length} vs {target_length}). "
            f"Truncating to {aligned_length} samples."
        )
        input_audio = input_audio[..., :aligned_length]
        target_audio = target_audio[..., :aligned_length]

    return AudioPair(input_audio, target_audio, sample_rate)
