"""Custom Lightning callbacks for neural audio effects training."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

from ..config import NeuralFXConfig


class NeuralFXCheckpoint(ModelCheckpoint):
    """Enhanced checkpoint callback with metadata saving.

    Saves best checkpoints by validation loss, always saves last checkpoint,
    and writes metadata as separate .meta.json files.
    """

    def __init__(
        self,
        config: NeuralFXConfig,
        input_file: str | None = None,
        target_file: str | None = None,
        latency_calibration: dict | None = None,
        monitor: str = "val_loss",
        save_top_k: int = 3,
        save_last: bool = True,
        mode: str = "min",
        **kwargs,
    ):
        """Initialize NeuralFXCheckpoint.

        Args:
            config: The training configuration.
            input_file: Path to input audio file used for training.
            target_file: Path to target audio file used for training.
            latency_calibration: Latency calibration result dictionary.
            monitor: Metric to monitor for best checkpoints.
            save_top_k: Number of best checkpoints to keep.
            save_last: Whether to always save the last checkpoint.
            mode: "min" or "max" for the monitored metric.
            **kwargs: Additional arguments passed to ModelCheckpoint.
        """
        self._config = config
        self._input_file = input_file
        self._target_file = target_file
        self._latency_calibration = latency_calibration or {}
        self._best_val_loss = float("inf")

        super().__init__(
            monitor=monitor,
            save_top_k=save_top_k,
            save_last=save_last,
            mode=mode,
            **kwargs,
        )

    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        """Save checkpoint and metadata.

        Args:
            trainer: The Lightning trainer.
            filepath: Path to save the checkpoint.
        """
        # Save the checkpoint via parent
        super()._save_checkpoint(trainer, filepath)

        # Save metadata as .meta.json
        self._save_metadata(trainer, filepath)

    def _save_metadata(self, trainer: L.Trainer, ckpt_path: str | Path) -> None:
        """Save metadata alongside the checkpoint.

        Args:
            trainer: The Lightning trainer.
            ckpt_path: Path to the checkpoint file.
        """
        ckpt_path = Path(ckpt_path)
        meta_path = ckpt_path.with_suffix(".meta.json")

        # Get current metrics
        current_epoch = trainer.current_epoch
        global_step = trainer.global_step

        # Get validation loss if available
        val_loss: float | None = None
        if "val_loss" in trainer.callback_metrics:
            val_loss = trainer.callback_metrics["val_loss"].item()
            self._best_val_loss = min(self._best_val_loss, val_loss)

        # Get train loss
        train_loss: float | None = None
        if "train_loss" in trainer.callback_metrics:
            train_loss = trainer.callback_metrics["train_loss"].item()

        # Get hardware info
        hardware_info = self._get_hardware_info()

        # Get git commit
        git_commit = self._get_git_commit()

        # Build metadata
        metadata: dict[str, Any] = {
            "checkpoint_file": ckpt_path.name,
            "git_commit": git_commit,
            "config": self._config_to_dict(self._config),
            "training_info": {
                "epochs_trained": current_epoch + 1,
                "global_step": global_step,
                "best_val_loss": self._best_val_loss
                if self._best_val_loss != float("inf")
                else None,
                "current_val_loss": val_loss,
                "current_train_loss": train_loss,
                "hardware": hardware_info,
            },
            "data_info": {
                "input_file": self._input_file,
                "target_file": self._target_file,
                "latency_calibration": self._latency_calibration,
            },
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        # Write metadata file
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _get_hardware_info(self) -> dict:
        """Get hardware information."""
        info = {
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda

        return info

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.SubprocessError,
        ):
            pass
        return None

    def _config_to_dict(self, config: NeuralFXConfig) -> dict:
        """Convert config dataclass to dictionary."""
        import dataclasses

        def convert(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        return convert(config)


class ValidationEarlyStopping(EarlyStopping):
    """Early stopping based on validation metrics.

    Stops training when validation metric stops improving.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = "min",
        strict: bool = True,
        **kwargs,
    ):
        """Initialize ValidationEarlyStopping.

        Args:
            monitor: Metric to monitor for early stopping.
            min_delta: Minimum change to qualify as an improvement.
            patience: Number of epochs with no improvement after which training stops.
            mode: "min" or "max" for the monitored metric.
            strict: Whether to crash if the metric is not found.
            **kwargs: Additional arguments passed to EarlyStopping.
        """
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            mode=mode,
            strict=strict,
            **kwargs,
        )


class ESRThresholdStopping(Callback):
    """Stop training when ESR reaches a target threshold.

    This is useful for quickly stopping training when model quality is good enough.
    """

    def __init__(self, threshold: float = 0.01, monitor: str = "val_esr"):
        """Initialize ESR threshold stopping.

        Args:
            threshold: ESR threshold to stop training at.
            monitor: Metric to monitor (default: val_esr).
        """
        super().__init__()
        self.threshold = threshold
        self.monitor = monitor

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Check if ESR threshold is reached."""
        if self.monitor not in trainer.callback_metrics:
            return

        current_esr = trainer.callback_metrics[self.monitor].item()

        if current_esr <= self.threshold:
            trainer.should_stop = True
            trainer.callback_metrics["stopped_by_esr_threshold"] = torch.tensor(1.0)
