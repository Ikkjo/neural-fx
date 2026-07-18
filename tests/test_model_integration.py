"""Cross-architecture registry and export-capability integration tests."""

import pytest
import torch

from neural_fx.config import LSTMParams, ModelConfig, SSMParams, WaveNetParams
from neural_fx.models import (
    MODEL_REGISTRY,
    MambaModel,
    NeuralfxGRU,
    NeuralfxLSTM,
    S4Model,
    WaveNetModel,
    create_model_from_config,
    get_supported_export_formats,
    validate_export_formats,
)
from neural_fx.models.errors import UnsupportedExportFormatError


def test_all_architectures_are_registered() -> None:
    assert MODEL_REGISTRY == {
        "lstm": NeuralfxLSTM,
        "gru": NeuralfxGRU,
        "wavenet": WaveNetModel,
        "s4": S4Model,
        "mamba": MambaModel,
    }


@pytest.mark.parametrize(
    ("model_type", "params", "expected_class"),
    [
        ("lstm", LSTMParams(hidden_size=4), NeuralfxLSTM),
        ("gru", LSTMParams(hidden_size=4), NeuralfxGRU),
        ("wavenet", WaveNetParams(layers=2), WaveNetModel),
        ("s4", SSMParams(d_model=4, d_state=4, num_layers=1), S4Model),
    ],
)
def test_factory_constructs_portable_architectures(
    model_type, params, expected_class
) -> None:
    config = ModelConfig(type=model_type, params=params)

    assert isinstance(create_model_from_config(config), expected_class)


@pytest.mark.parametrize(
    ("model_type", "params"),
    [
        ("lstm", LSTMParams(hidden_size=4, num_layers=1)),
        ("gru", LSTMParams(hidden_size=4, num_layers=1)),
        (
            "wavenet",
            WaveNetParams(
                layers=2,
                stacks=1,
                dilation_channels=4,
                residual_channels=4,
                skip_channels=8,
            ),
        ),
        ("s4", SSMParams(d_model=4, d_state=4, num_layers=1)),
    ],
)
def test_portable_architecture_training_step_and_checkpoint_round_trip(
    model_type, params, tmp_path
) -> None:
    config = ModelConfig(type=model_type, params=params)
    model = create_model_from_config(config).train()
    inputs = torch.randn(2, 1, 16, requires_grad=True)

    output = model(inputs, reset_state=True)
    output.square().mean().backward()

    assert output.shape == inputs.shape
    assert inputs.grad is not None

    checkpoint = tmp_path / f"{model_type}.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = create_model_from_config(config)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    assert all(
        torch.equal(restored.state_dict()[name], value)
        for name, value in model.state_dict().items()
    )


def test_export_capabilities_are_explicit() -> None:
    assert get_supported_export_formats("lstm") == {
        "onnx",
        "torchscript",
        "rtneural",
    }
    assert get_supported_export_formats("wavenet") == {"onnx", "torchscript"}
    assert get_supported_export_formats("s4") == {"onnx", "torchscript"}
    assert get_supported_export_formats("mamba") == set()


def test_export_validation_runs_without_constructing_optional_model() -> None:
    with pytest.raises(UnsupportedExportFormatError, match="mamba"):
        validate_export_formats("mamba", ["onnx"])

    with pytest.raises(UnsupportedExportFormatError, match="rtneural"):
        validate_export_formats("wavenet", ["rtneural"])

    with pytest.raises(ValueError, match="Unknown export format"):
        validate_export_formats("lstm", ["pickle"])
