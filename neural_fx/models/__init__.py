from .base import BaseNeuralFXModel
from .errors import UnsupportedExportFormatError
from .mamba import MambaModel
from .recurrent import NeuralfxLSTM, NeuralfxGRU, RecurrentNeuralFXModel
from .ssm import S4Model
from .wavenet import WaveNetModel

__all__ = [
    "BaseNeuralFXModel",
    "NeuralfxLSTM",
    "NeuralfxGRU",
    "RecurrentNeuralFXModel",
    "WaveNetModel",
    "S4Model",
    "MambaModel",
    "MODEL_REGISTRY",
    "MODEL_EXPORT_FORMATS",
    "create_model_from_config",
    "register_model",
    "get_available_models",
    "get_supported_export_formats",
    "validate_export_formats",
]


# Model registry mapping model type names to their classes
MODEL_REGISTRY: dict[str, type[BaseNeuralFXModel]] = {
    "lstm": NeuralfxLSTM,
    "gru": NeuralfxGRU,
    "wavenet": WaveNetModel,
    "s4": S4Model,
    "mamba": MambaModel,
}

MODEL_EXPORT_FORMATS: dict[str, frozenset[str]] = {
    "lstm": frozenset({"onnx", "torchscript", "rtneural"}),
    "gru": frozenset({"onnx", "torchscript", "rtneural"}),
    "wavenet": frozenset({"onnx", "torchscript"}),
    "s4": frozenset({"onnx", "torchscript"}),
    "mamba": frozenset(),
}


def register_model(name: str, model_class: type[BaseNeuralFXModel]) -> None:
    """
    Register a new model class in the registry.

    Args:
        name: Model type name (used in config files).
        model_class: The model class to register.
    """
    normalized = name.lower()
    MODEL_REGISTRY[normalized] = model_class
    MODEL_EXPORT_FORMATS.setdefault(normalized, frozenset())


def get_supported_export_formats(model_type: str) -> frozenset[str]:
    """Return the export formats supported by a registered model type."""
    normalized = model_type.lower()
    if normalized not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            f"Available types: {list(MODEL_REGISTRY)}"
        )
    return MODEL_EXPORT_FORMATS[normalized]


def validate_export_formats(model_type: str, formats: list[str]) -> None:
    """Validate requested formats before constructing an optional model."""
    known_formats = {"onnx", "torchscript", "rtneural"}
    supported = get_supported_export_formats(model_type)
    for export_format in formats:
        if export_format not in known_formats:
            raise ValueError(f"Unknown export format: {export_format!r}")
        if export_format not in supported:
            raise UnsupportedExportFormatError(model_type.lower(), export_format)


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
