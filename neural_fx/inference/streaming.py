import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torchaudio
from torch import Tensor

from ..losses.audio_losses import ESR, MSE

if TYPE_CHECKING:
    from ..models.base import BaseNeuralFXModel


def _resolve_model_sample_rate(
    model: "BaseNeuralFXModel", sample_rate: int | None
) -> int:
    """Resolve a model-aware sample rate without allowing conflicting overrides."""
    model_sample_rate = int(model.sample_rate)
    if sample_rate is not None and int(sample_rate) != model_sample_rate:
        warnings.warn(
            f"Requested sample rate {sample_rate} does not match the model sample "
            f"rate {model_sample_rate}; using the model sample rate.",
            UserWarning,
            stacklevel=2,
        )
    return model_sample_rate


def load_audio(path: str | Path, sample_rate: int = 48000) -> Tensor:
    """Load and preprocess audio file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = torchaudio.load(str(path))

    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    return audio


def save_audio(audio: Tensor, path: str | Path, sample_rate: int = 48000) -> None:
    """Save audio tensor to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Handle different input shapes
    if audio.ndim == 3:
        # [batch, channels, time] -> [channels, time]
        audio = audio.squeeze(0)
    if audio.ndim == 1:
        # [time] -> [channels, time]
        audio = audio.unsqueeze(0)

    torchaudio.save(str(path), audio, sample_rate)


def process_audio(
    model: "BaseNeuralFXModel",
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int | None = None,
    chunk_size: int = 8192,
    conditioning: Tensor | None = None,
) -> Tensor:
    """Process audio file through model."""
    sample_rate = _resolve_model_sample_rate(model, sample_rate)
    model.eval()
    model.reset_state()

    audio = load_audio(input_path, sample_rate)

    # Process in chunks to manage memory
    total_length = audio.shape[-1]
    output_chunks = []

    with torch.no_grad():
        for start in range(0, total_length, chunk_size):
            end = min(start + chunk_size, total_length)
            chunk = audio[..., start:end]

            if chunk.shape[-1] == 0:
                break

            # Add batch dimension if needed: [C, T] -> [B, C, T]
            if chunk.ndim == 2:
                chunk = chunk.unsqueeze(0)

            chunk_conditioning = conditioning
            if conditioning is not None and conditioning.ndim == 3:
                chunk_conditioning = conditioning[..., start:end]
            if chunk_conditioning is None:
                out_chunk = model(chunk)
            else:
                out_chunk = model(chunk, conditioning=chunk_conditioning)
            output_chunks.append(out_chunk)

    output = torch.cat(output_chunks, dim=-1)
    save_audio(output, output_path, sample_rate)

    return output


def evaluate_model(
    model: "BaseNeuralFXModel",
    input_path: str | Path,
    target_path: str | Path,
    sample_rate: int | None = None,
    burn_in: int = 0,
    conditioning: Tensor | None = None,
) -> dict[str, float]:
    """Evaluate model against target audio."""
    sample_rate = _resolve_model_sample_rate(model, sample_rate)
    model.eval()
    model.reset_state()

    input_audio = load_audio(input_path, sample_rate)
    target_audio = load_audio(target_path, sample_rate)

    min_length = min(input_audio.shape[-1], target_audio.shape[-1])
    input_audio = input_audio[..., :min_length]
    target_audio = target_audio[..., :min_length]

    # Add batch dimension: [C, T] -> [B, C, T]
    if input_audio.ndim == 2:
        input_audio = input_audio.unsqueeze(0)
    if target_audio.ndim == 2:
        target_audio = target_audio.unsqueeze(0)

    with torch.no_grad():
        if conditioning is None:
            pred_audio = model(input_audio)
        else:
            pred_audio = model(input_audio, conditioning=conditioning)

    if burn_in > 0:
        pred_audio = pred_audio[..., burn_in:]
        target_audio = target_audio[..., burn_in:]

    mse = MSE(pred_audio, target_audio).item()
    esr = ESR(pred_audio, target_audio).item()

    return {
        "mse": mse,
        "esr": esr,
    }


class StreamingProcessor:
    """Real-time streaming processor for model inference."""

    def __init__(self, model: "BaseNeuralFXModel", sample_rate: int | None = None):
        self.model = model
        self.sample_rate = _resolve_model_sample_rate(model, sample_rate)
        self.model.eval()
        self.model.reset_state()

    def process_block(
        self, block: Tensor, conditioning: Tensor | None = None
    ) -> Tensor:
        """Process a block of samples."""
        with torch.no_grad():
            if conditioning is None:
                return self.model(block)
            return self.model(block, conditioning=conditioning)

    def process_sample(
        self, sample: float, conditioning: Tensor | float | None = None
    ) -> float:
        """Process a single sample (for real-time use)."""
        device = next(self.model.parameters()).device
        x = torch.tensor([[sample]], device=device)
        if isinstance(conditioning, float):
            conditioning = torch.tensor([[conditioning]], device=device)
        elif conditioning is not None:
            conditioning = conditioning.to(device)
            if conditioning.ndim == 1:
                conditioning = conditioning.unsqueeze(0)
        if conditioning is None:
            output = self.model.process_sample(x)
        else:
            output = self.model.process_sample(x, conditioning=conditioning)
        return float(output.item())

    def reset(self) -> None:
        """Reset model state."""
        self.model.reset_state()
