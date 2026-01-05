from abc import ABC, abstractmethod, property, classmethod
from typing import Optional, Union, Tuple, List 
import torch
import torch.nn as nn
from torch import Tensor

StateType = Optional[Union[Tensor, Tuple[Tensor, ...], List[Tensor]]]

class BaseNeuralFXModel(nn.Module, ABC):
    model_type: str = "base"
    supports_variable_length: bool = True

    def __init__(self, sample_rate: int = 48_000):
        super().__init__()
        self.sample_rate = sample_rate
    
    # Training
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
    
    @abstractmethod
    def reset_state(self) -> None: 
        """Reset the model state to initial values"""
        ...

    def detach_state(self) -> None: 
        """Detach the model state from the computation graph"""
        ...

    # Validation 
    @abstractmethod 
    def process_sample(self, x: Tensor) -> Tensor:
        """Single-sample forward pass for real-time use"""
        ...

    # Properties
    @property
    @abstractmethod
    def receptive_field(self) -> int:
        """The number of timesteps the model can see at once"""
        ...

    @property
    def sample_rate(self) -> int:
        """Sample rate of the model"""
        return self._sample_rate

    @property
    def num_params(self) -> int:
        """Number of parameters in the model"""
        ... 

    @property
    def has_state(self) -> bool:
        return True # Override to False if the model does not support state

    # Factory
    @classmethod
    @abstractmethod
    def from_config(self, config: dict) -> "BaseNeuralFXModel":
        """Create a model from a config dictionary"""
        ...

    # Export
    @abstractmethod
    def export_onnx(self, filename: str, opset_version: Optional[int] = None) -> None:
        """Export the model to ONNX format"""
        ...

    @abstractmethod
    def export_torchscript(self, filename: str) -> None:
        """Export the model to TorchScript format"""
        ...

    @abstractmethod
    def export_rtneural(self, path: Path) -> None:
        """Export the model to RTNeural JSON format"""
        ...