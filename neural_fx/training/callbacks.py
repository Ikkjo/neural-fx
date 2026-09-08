"""Custom Lightning callbacks for neural audio effects training."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

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

    def _save_last_if_needed(self, trainer: L.Trainer) -> None:
        """Keep ``last.ckpt`` resumable even after top-k stops improving."""
        if self.save_last and self._last_global_step_saved != trainer.global_step:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if (
            not self._should_skip_saving_checkpoint(trainer)
            and self._should_save_on_train_epoch_end(trainer)
        ):
            self._save_last_if_needed(trainer)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_validation_end(trainer, pl_module)
        if (
            not self._should_skip_saving_checkpoint(trainer)
            and not self._should_save_on_train_epoch_end(trainer)
        ):
            self._save_last_if_needed(trainer)

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
        min_delta_mode: str = "absolute",
        patience: int = 10,
        mode: str = "min",
        strict: bool = True,
        **kwargs,
    ):
        """Initialize ValidationEarlyStopping.

        Args:
            monitor: Metric to monitor for early stopping.
            min_delta: Minimum change to qualify as an improvement.
            min_delta_mode: Interpret min_delta as an absolute value or a
                fraction of the best score.
            patience: Number of epochs with no improvement after which training stops.
            mode: "min" or "max" for the monitored metric.
            strict: Whether to crash if the metric is not found.
            **kwargs: Additional arguments passed to EarlyStopping.
        """
        if min_delta_mode not in {"absolute", "relative"}:
            raise ValueError("min_delta_mode must be 'absolute' or 'relative'")
        if min_delta < 0:
            raise ValueError("min_delta cannot be negative")
        self.min_delta_mode = min_delta_mode
        self.relative_min_delta = min_delta if min_delta_mode == "relative" else 0.0
        super().__init__(
            monitor=monitor,
            min_delta=0.0 if min_delta_mode == "relative" else min_delta,
            patience=patience,
            mode=mode,
            strict=strict,
            **kwargs,
        )

    def _evaluate_stopping_criteria(
        self, current: torch.Tensor
    ) -> tuple[bool, str | None]:
        """Apply a scale-independent threshold when relative mode is enabled."""
        if self.min_delta_mode == "relative" and torch.isfinite(self.best_score):
            absolute_delta = abs(float(self.best_score)) * self.relative_min_delta
            self.min_delta = absolute_delta if self.mode == "max" else -absolute_delta
        return super()._evaluate_stopping_criteria(current)
