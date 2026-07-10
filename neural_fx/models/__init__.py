from .base import BaseNeuralFXModel
from .recurrent import NeuralfxLSTM, NeuralfxGRU, RecurrentNeuralFXModel

__all__ = [
    "BaseNeuralFXModel",
    "NeuralfxLSTM",
    "NeuralfxGRU",
    "RecurrentNeuralFXModel",
    "MODEL_REGISTRY",
    "create_model_from_config",
    "register_model",
    "get_available_models",
]


# Model registry mapping model type names to their classes
MODEL_REGISTRY: dict[str, type[BaseNeuralFXModel]] = {
    "lstm": NeuralfxLSTM,
    "gru": NeuralfxGRU,
    # Placeholder entries for models that may be implemented later
    # "wavenet": WaveNetModel,
    # "mamba": MambaModel,
    # "s4": S4Model,
}


def register_model(name: str, model_class: type[BaseNeuralFXModel]) -> None:
    """
    Register a new model class in the registry.

    Args:
        name: Model type name (used in config files).
        model_class: The model class to register.
    """
    MODEL_REGISTRY[name] = model_class


def create_model_from_config(config) -> BaseNeuralFXModel:
    """
    Create a model instance from a ModelConfig.

    Args:
        config: A ModelConfig instance containing the model type and parameters.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If the model type is not registered.
        NotImplementedError: If the model type is in registry but not implemented.
    """
    model_type = config.type.lower()

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available types: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_type]

    # Check if it's a placeholder (not actually implemented)
    # Models that are registered but not implemented should raise NotImplementedError
    # in their from_config method or be None
    if model_class is None:
        raise NotImplementedError(
            f"Model type '{model_type}' is registered but not yet implemented."
        )

    return model_class.from_config(config)


def get_available_models() -> list[str]:
    """Get list of available model types."""
    return list(MODEL_REGISTRY.keys())
