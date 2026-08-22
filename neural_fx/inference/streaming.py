import warnings
from dataclasses import dataclass
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


@dataclass(frozen=True)
class InferenceResult:
    output: Tensor
    chunk_size: int
    chunks: int


class InferenceSession:
    """Persistent model stream whose state changes only on process or reset."""

    def __init__(self, model: "BaseNeuralFXModel", sample_rate: int | None = None):
        self.model = model
        self.sample_rate = _resolve_model_sample_rate(model, sample_rate)
        self.model.eval()
        self.model.reset_state()

    def process_block(
        self,
        block: Tensor,
        conditioning: Tensor | None = None,
    ) -> Tensor:
        with torch.inference_mode():
            if conditioning is None:
                return self.model(block)
            return self.model(block, conditioning=conditioning)

    def process_sample(
        self,
        sample: float,
        conditioning: Tensor | float | None = None,
    ) -> float:
        parameter = next(self.model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        x = torch.tensor([[sample]], device=device)
        if isinstance(conditioning, float):
            conditioning = torch.tensor([[conditioning]], device=device)
        elif conditioning is not None:
            conditioning = conditioning.to(device)
            if conditioning.ndim == 1:
                conditioning = conditioning.unsqueeze(0)
        with torch.inference_mode():
            if conditioning is None:
                output = self.model.process_sample(x)
            else:
                output = self.model.process_sample(x, conditioning=conditioning)
        return float(output.item())

    def reset(self) -> None:
        self.model.reset_state()


def run_inference(
    model: "BaseNeuralFXModel",
    audio: Tensor,
    *,
    chunk_size: int = 8192,
    conditioning: Tensor | None = None,
) -> InferenceResult:
    """Run one finite buffer with a single reset and state carried across chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if audio.shape[-1] == 0:
        raise ValueError("inference audio must contain at least one sample")
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)

    session = InferenceSession(model)
    outputs = []
    for start in range(0, audio.shape[-1], chunk_size):
        end = min(start + chunk_size, audio.shape[-1])
        chunk_conditioning = conditioning
        if conditioning is not None and conditioning.ndim == 3:
            chunk_conditioning = conditioning[..., start:end]
        outputs.append(
            session.process_block(
                audio[..., start:end],
                conditioning=chunk_conditioning,
            )
        )
    return InferenceResult(
        output=torch.cat(outputs, dim=-1),
        chunk_size=chunk_size,
        chunks=len(outputs),
    )


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
    audio = load_audio(input_path, sample_rate)
    output = run_inference(
        model,
        audio,
        chunk_size=chunk_size,
        conditioning=conditioning,
    ).output
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
    input_audio = load_audio(input_path, sample_rate)
    target_audio = load_audio(target_path, sample_rate)

    min_length = min(input_audio.shape[-1], target_audio.shape[-1])
    input_audio = input_audio[..., :min_length]
    target_audio = target_audio[..., :min_length]

    if target_audio.ndim == 2:
        target_audio = target_audio.unsqueeze(0)
    pred_audio = run_inference(
        model,
        input_audio,
        chunk_size=min_length,
        conditioning=conditioning,
    ).output

    if burn_in > 0:
        pred_audio = pred_audio[..., burn_in:]
        target_audio = target_audio[..., burn_in:]

    mse = MSE(pred_audio, target_audio).item()
    esr = ESR(pred_audio, target_audio).item()

    return {
        "mse": mse,
        "esr": esr,
    }


class StreamingProcessor(InferenceSession):
    """Compatibility name for the persistent inference session."""
