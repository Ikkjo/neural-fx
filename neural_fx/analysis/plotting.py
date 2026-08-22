"""Compatibility report generation for the legacy analysis command."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..config import NeuralFXConfig
from ..data.dataset import AudioDataset
from ..inference import run_inference
from ..losses.audio_losses import ESR, MSE
from ..models.base import BaseNeuralFXModel


def _esr_comment(esr: float) -> str:
    if esr < 0.001:
        return "Excellent! (ESR < 0.001)"
    if esr < 0.01:
        return "Great! (ESR < 0.01)"
    if esr < 0.05:
        return "Good (ESR < 0.05)"
    if esr < 0.1:
        return "Not bad (ESR < 0.1)"
    if esr < 0.3:
        return "Acceptable (ESR < 0.3)"
    return "Poor (ESR >= 0.3) - consider more training"


def _save_waveform_plot(
    input_audio: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    esr: float,
    path: Path,
) -> None:
    input_values = input_audio.numpy()
    target_values = target.numpy()
    prediction_values = prediction.numpy()
    figure, axes = plt.subplots(3, 1, figsize=(14, 8))
    axes[0].plot(input_values, linewidth=0.5, alpha=0.8)
    axes[0].set(title="Input Signal", ylabel="Amplitude")
    axes[1].plot(target_values, label="Target", linewidth=0.8, alpha=0.9)
    axes[1].plot(prediction_values, label="Prediction", linewidth=0.8, alpha=0.9)
    axes[1].set(title="Prediction vs Target", ylabel="Amplitude")
    axes[1].legend()
    axes[2].plot(target_values - prediction_values, linewidth=0.5, color="red")
    axes[2].set(
        title=f"Error (Target - Prediction) | ESR: {esr:.4f}",
        xlabel="Sample",
        ylabel="Amplitude",
    )
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _save_spectrogram_plot(
    target: torch.Tensor,
    prediction: torch.Tensor,
    sample_rate: int,
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].specgram(
        target.numpy(),
        Fs=sample_rate,
        NFFT=2048,
        noverlap=1024,
        cmap="viridis",
    )
    axes[0].set(title="Target Spectrogram", ylabel="Frequency (Hz)")
    axes[1].specgram(
        prediction.numpy(),
        Fs=sample_rate,
        NFFT=2048,
        noverlap=1024,
        cmap="viridis",
    )
    axes[1].set(
        title="Prediction Spectrogram",
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def generate_analysis_report(
    model: BaseNeuralFXModel,
    dataset: AudioDataset,
    output_dir: str | Path,
    num_samples: int = 48_000,
    config: NeuralFXConfig | None = None,
) -> dict:
    """Write the legacy plots and return their stable report payload."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_audio, target = dataset[0]
    num_samples = min(num_samples, input_audio.shape[-1])
    input_audio = input_audio[:num_samples]
    target = target[:num_samples]

    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    prediction = (
        run_inference(
            model,
            input_audio.unsqueeze(0).unsqueeze(0).to(device),
            chunk_size=num_samples,
        )
        .output.squeeze()
        .cpu()
    )
    esr = ESR(prediction[None, None], target[None, None]).item()
    mse = MSE(prediction[None, None], target[None, None]).item()
    prediction_std = prediction.std().item()
    target_std = target.std().item()
    correlation = (
        0.0
        if prediction_std == 0 or target_std == 0
        else float(np.corrcoef(prediction.numpy(), target.numpy())[0, 1])
    )

    waveform_path = output_dir / "waveform_comparison.png"
    spectrogram_path = output_dir / "spectrograms.png"
    _save_waveform_plot(input_audio, target, prediction, esr, waveform_path)
    sample_rate = config.sample_rate if config is not None else model.sample_rate
    _save_spectrogram_plot(target, prediction, sample_rate, spectrogram_path)
    return {
        "esr": esr,
        "esr_comment": _esr_comment(esr),
        "mse": mse,
        "correlation": correlation,
        "num_params": sum(parameter.numel() for parameter in model.parameters()),
        "plots": {
            "waveform": str(waveform_path),
            "spectrograms": str(spectrogram_path),
        },
    }


def write_html_report(report: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    css_class = (
        "good" if report["esr"] < 0.01 else "warning" if report["esr"] < 0.1 else "bad"
    )
    output_path.write_text(
        f"""<!DOCTYPE html>
<html>
<head>
  <title>Neural Audio Effects - Model Analysis Report</title>
  <style>
    body {{ font-family: sans-serif; max-width: 1200px; margin: auto; }}
    .good {{ color: #17823b; }} .warning {{ color: #a35f00; }} .bad {{ color: #b42318; }}
    img {{ max-width: 100%; }}
  </style>
</head>
<body>
  <h1>Model Analysis Report</h1>
  <p class="{css_class}">ESR: {report["esr"]:.6f} — {report["esr_comment"]}</p>
  <p>MSE: {report["mse"]:.6f}</p>
  <p>Correlation: {report["correlation"]:.4f}</p>
  <p>Model parameters: {report["num_params"]:,}</p>
  <h2>Waveform comparison</h2><img src="waveform_comparison.png">
  <h2>Spectrograms</h2><img src="spectrograms.png">
</body>
</html>
"""
    )


class TrainingAnalyzer:
    """Compatibility adapter for callers of the former analyzer class."""

    def __init__(
        self,
        model: BaseNeuralFXModel,
        config: NeuralFXConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config

    def generate_report(
        self,
        dataset: AudioDataset,
        output_dir: str | Path,
        num_samples: int = 48_000,
    ) -> dict:
        return generate_analysis_report(
            self.model,
            dataset,
            output_dir,
            num_samples,
            self.config,
        )
