from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch import Tensor, nn


class UnsupportedExportError(NotImplementedError):
    """Raised when a model cannot represent an export format faithfully."""


class BaseNeuralFXModel(nn.Module, ABC):
    """Base class for neural audio effects models."""

    model_type: str = "base"

    def __init__(self, sample_rate: int = 48_000):
        super().__init__()
        self._sample_rate = sample_rate

    # Training
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Batch forward pass for training."""
        ...

    @abstractmethod
    def reset_state(self) -> None:
        """Reset internal state to initial values."""
        ...

    def detach_state(self) -> None:
        """Detach state from computation graph for TBPTT."""
        # Override in stateful models

    # Validation
    @abstractmethod
    def process_sample(self, x: Tensor) -> Tensor:
        """Single-sample forward for validating exported models."""
        ...

    # Properties
    @property
    @abstractmethod
    def receptive_field(self) -> int | float:
        """Number of past samples the model depends on."""
        ...

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def model_size_bytes(self) -> int:
        """Return the storage used by parameters and persistent buffers."""
        tensors = list(self.parameters()) + list(self.buffers())
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @property
    def supported_export_formats(self) -> tuple[str, ...]:
        """Return export formats that this model implements faithfully."""
        return ("onnx", "torchscript", "rtneural")

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseNeuralFXModel":
        """Create model from configuration dictionary."""
        ...

    # Export
    @abstractmethod
    def export_onnx(self, path: Path, opset_version: int = 17) -> None:
        """Export model to ONNX format."""
        ...

    @abstractmethod
    def export_torchscript(self, path: Path) -> None:
        """Export model to TorchScript format."""
        ...

    @abstractmethod
    def export_rtneural(self, path: Path) -> None:
        """Export model to RTNeural JSON format."""
        ...
