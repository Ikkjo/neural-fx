"""Plotting utilities for analyzing neural audio effects models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from ..losses.audio_losses import ESR


class TrainingAnalyzer:
    """Analyzer for neural audio effects training results."""

    def __init__(self, model, config=None):
        """Initialize analyzer.

        Args:
            model: Trained model to analyze.
            config: Optional training configuration.
        """
        self.model = model
        self.config = config
        self.model.eval()

    def plot_comparison(
        self,
        dataset,
        num_samples: int = 48000,
        save_path: str | Path | None = None,
        start_idx: int = 0,
    ) -> plt.Figure:
        """Plot waveform comparison between prediction and target.

        Args:
            dataset: Dataset to get samples from.
            num_samples: Number of samples to plot.
            save_path: Optional path to save the figure.
            start_idx: Starting index in the dataset.

        Returns:
            Matplotlib figure.
        """
        # Get a segment from the dataset
        x, y_true = dataset[start_idx]

        # Ensure we have enough samples
        if x.shape[-1] < num_samples:
            num_samples = x.shape[-1]

        x = x[:num_samples]
        y_true = y_true[:num_samples]

        # Run model inference
        with torch.no_grad():
            x_batch = x.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
            self.model.reset_state()
            y_pred = self.model(x_batch).squeeze().cpu()

        # Convert to numpy
        x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
        y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else np.array(y_true)
        y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else np.array(y_pred)

        # Calculate error
        error = y_true_np - y_pred_np

        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 8))

        # Plot input
        axes[0].plot(x_np, linewidth=0.5, alpha=0.8)
        axes[0].set_title("Input Signal")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)

        # Plot prediction vs target
        axes[1].plot(y_true_np, label="Target", linewidth=0.8, alpha=0.9)
        axes[1].plot(y_pred_np, label="Prediction", linewidth=0.8, alpha=0.9)
        axes[1].set_title("Prediction vs Target")
        axes[1].set_ylabel("Amplitude")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot error
        axes[2].plot(error, linewidth=0.5, color="red", alpha=0.8)
        axes[2].set_title(f"Error (Target - Prediction) | ESR: {self.calculate_esr(y_pred, y_true):.4f}")
        axes[2].set_ylabel("Amplitude")
        axes[2].set_xlabel("Sample")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_spectrograms(
        self,
        pred: Tensor,
        target: Tensor,
        save_path: str | Path | None = None,
        sample_rate: int = 48000,
    ) -> plt.Figure:
        """Plot spectrograms of prediction and target.

        Args:
            pred: Predicted audio tensor.
            target: Target audio tensor.
            save_path: Optional path to save the figure.
            sample_rate: Sample rate for frequency axis.

        Returns:
            Matplotlib figure.
        """
        # Convert to numpy
        pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else np.array(pred)
        target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else np.array(target)

        # Ensure 1D arrays
        pred_np = pred_np.squeeze()
        target_np = target_np.squeeze()

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Target spectrogram
        axes[0].specgram(target_np, Fs=sample_rate, NFFT=2048, noverlap=1024, cmap="viridis")
        axes[0].set_title("Target Spectrogram")
        axes[0].set_ylabel("Frequency (Hz)")

        # Prediction spectrogram
        axes[1].specgram(pred_np, Fs=sample_rate, NFFT=2048, noverlap=1024, cmap="viridis")
        axes[1].set_title("Prediction Spectrogram")
        axes[1].set_ylabel("Frequency (Hz)")
        axes[1].set_xlabel("Time (s)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_training_curves(
        self,
        trainer,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot training curves from trainer logs.

        Args:
            trainer: Lightning trainer with logged metrics.
            save_path: Optional path to save the figure.

        Returns:
            Matplotlib figure.
        """
        # Get metrics from trainer
        metrics = trainer.callback_metrics

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot train loss if available
        if hasattr(trainer, "logger") and trainer.logger is not None:
            # Try to get from logger
            try:
                log_data = trainer.logger.experiment
                if hasattr(log_data, "data"):
                    data = log_data.data

                    if "train_loss" in data:
                        axes[0].plot(data["train_loss"], label="Train Loss")
                    if "val_loss" in data:
                        axes[0].plot(data["val_loss"], label="Val Loss")
            except Exception:
                pass

        # Fallback: use current metrics
        if "train_loss" in metrics:
            axes[0].set_title(f"Train Loss: {metrics['train_loss'].item():.4f}")
        if "val_loss" in metrics:
            axes[0].set_title(f"Val Loss: {metrics['val_loss'].item():.4f}")

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot learning rate if available
        if "lr" in metrics:
            axes[1].set_title(f"Learning Rate: {metrics['lr'].item():.6f}")

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Learning Rate")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def calculate_esr(self, pred: Tensor, target: Tensor) -> float:
        """Calculate ESR between prediction and target.

        Args:
            pred: Predicted audio tensor.
            target: Target audio tensor.

        Returns:
            ESR value.
        """
        # Ensure tensors are on same device and have batch dimension
        if pred.ndim == 1:
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.ndim == 2:
            pred = pred.unsqueeze(0)

        if target.ndim == 1:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.ndim == 2:
            target = target.unsqueeze(0)

        with torch.no_grad():
            esr = ESR(pred, target).item()

        return esr

    def get_esr_comment(self, esr: float) -> str:
        """Get a human-readable comment about ESR value.

        Args:
            esr: ESR value.

        Returns:
            Comment string.
        """
        if esr < 0.001:
            return "Excellent! (ESR < 0.001)"
        elif esr < 0.01:
            return "Great! (ESR < 0.01)"
        elif esr < 0.05:
            return "Good (ESR < 0.05)"
        elif esr < 0.1:
            return "Not bad (ESR < 0.1)"
        elif esr < 0.3:
            return "Acceptable (ESR < 0.3)"
        else:
            return "Poor (ESR >= 0.3) - consider more training"

    def generate_report(
        self,
        dataset,
        output_dir: str | Path,
        num_samples: int = 48000,
    ) -> dict:
        """Generate a complete analysis report.

        Args:
            dataset: Dataset to analyze.
            output_dir: Directory to save outputs.
            num_samples: Number of samples to analyze.

        Returns:
            Dictionary with report data.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get a sample
        x, y_true = dataset[0]
        if x.shape[-1] < num_samples:
            num_samples = x.shape[-1]

        x = x[:num_samples]
        y_true = y_true[:num_samples]

        # Run inference
        with torch.no_grad():
            x_batch = x.unsqueeze(0).unsqueeze(0)
            self.model.reset_state()
            y_pred = self.model(x_batch).squeeze().cpu()

        # Calculate metrics
        esr = self.calculate_esr(y_pred, y_true)
        mse = torch.mean((y_pred - y_true) ** 2).item()

        # Correlation coefficient
        pred_np = y_pred.numpy()
        true_np = y_true.numpy()
        correlation = np.corrcoef(pred_np, true_np)[0, 1]

        # Generate plots
        self.plot_comparison(dataset, num_samples, output_dir / "waveform_comparison.png")
        self.plot_spectrograms(y_pred, y_true, output_dir / "spectrograms.png")

        report = {
            "esr": esr,
            "esr_comment": self.get_esr_comment(esr),
            "mse": mse,
            "correlation": correlation,
            "num_params": sum(p.numel() for p in self.model.parameters()),
            "plots": {
                "waveform": str(output_dir / "waveform_comparison.png"),
                "spectrograms": str(output_dir / "spectrograms.png"),
            },
        }

        return report


def create_analyzer(model, config=None) -> TrainingAnalyzer:
    """Factory function to create a TrainingAnalyzer."""
    return TrainingAnalyzer(model, config)
