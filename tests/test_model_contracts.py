import torch

from neural_fx.config import LSTMParams, ModelConfig
from neural_fx.models import create_model_from_config


def test_model_reports_size_and_export_contract() -> None:
    model = create_model_from_config(
        ModelConfig(type="lstm", params=LSTMParams(hidden_size=4, num_layers=1))
    )

    parameter_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in model.parameters()
    )
    buffer_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in model.buffers()
    )

    assert model.model_size_bytes == parameter_bytes + buffer_bytes
    assert model.num_params == sum(p.numel() for p in model.parameters())
    assert model.supported_export_formats == ("onnx", "torchscript", "rtneural")
    assert model(torch.zeros(1, 1, 8)).shape == (1, 1, 8)
