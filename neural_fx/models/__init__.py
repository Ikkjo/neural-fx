from ..config import ModelConfig
from .base import BaseNeuralFXModel, UnsupportedExportError
from .recurrent import NeuralfxGRU, NeuralfxLSTM, RecurrentNeuralFXModel
from .ssm import DiagonalStateSpace, S4DBlock, S4DModel
from .wavenet import CausalConv1d, DilatedResidualBlock, WaveNetModel

__all__ = [
    "BaseNeuralFXModel",
    "CausalConv1d",
    "DiagonalStateSpace",
    "DilatedResidualBlock",
    "NeuralfxGRU",
    "NeuralfxLSTM",
    "RecurrentNeuralFXModel",
    "S4DBlock",
    "S4DModel",
    "UnsupportedExportError",
    "WaveNetModel",
    "create_model_from_config",
]


_MODEL_TYPES: dict[str, type[BaseNeuralFXModel]] = {
    "lstm": NeuralfxLSTM,
    "gru": NeuralfxGRU,
    "wavenet": WaveNetModel,
    "s4": S4DModel,
}


def create_model_from_config(config: ModelConfig) -> BaseNeuralFXModel:
    """Create the configured built-in model type."""
    model_type = config.type.lower()

    if model_type not in _MODEL_TYPES:
        if model_type == "mamba":
            raise NotImplementedError(
                "Mamba is not part of the portable core backend; use the 's4' "
                "model type until an optional accelerated backend is implemented."
            )
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available types: {list(_MODEL_TYPES)}"
        )
    return _MODEL_TYPES[model_type].from_config(config)
