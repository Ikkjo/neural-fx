from pathlib import Path

import pytest
import torch

from neural_fx.config import ModelConfig, WaveNetParams, load_config
from neural_fx.models import WaveNetModel, create_model_from_config


def _model() -> WaveNetModel:
    return WaveNetModel(
        ModelConfig(
            type="wavenet",
            params=WaveNetParams(
                layers=3,
                stacks=2,
                kernel_size=2,
                dilation_channels=4,
                residual_channels=4,
                skip_channels=8,
            ),
        )
    ).eval()


def test_wavenet_shape_receptive_field_and_registry() -> None:
    model = _model()

    assert model(torch.randn(2, 1, 64), reset_state=True).shape == (2, 1, 64)
    assert model.receptive_field == 15
    assert isinstance(create_model_from_config(model.config), WaveNetModel)


def test_wavenet_does_not_read_future_samples() -> None:
    model = _model()
    x = torch.randn(1, 1, 48)
    changed = x.clone()
    changed[..., 24:] += 10

    with torch.no_grad():
        original = model(x, reset_state=True)
        modified = model(changed, reset_state=True)

    torch.testing.assert_close(original[..., :24], modified[..., :24])


def test_chunked_and_sample_inference_match_full_sequence() -> None:
    model = _model()
    x = torch.randn(1, 1, 48)

    with torch.no_grad():
        full = model(x, reset_state=True)
        model.reset_state()
        chunked = torch.cat([model(x[..., :17]), model(x[..., 17:])], dim=-1)
        model.reset_state()
        sampled = torch.stack(
            [model.process_sample(x[..., i]).squeeze() for i in range(x.shape[-1])]
        ).reshape(1, 1, -1)

    torch.testing.assert_close(chunked, full, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(sampled, full, atol=1e-6, rtol=1e-5)


def test_wavenet_config_and_torchscript_export(tmp_path: Path) -> None:
    config = load_config("configs/models/wavenet/wavenet_nano.yaml")
    model = create_model_from_config(config.model).eval()
    path = tmp_path / "wavenet.pt"

    model.export_torchscript(path)
    exported = torch.jit.load(str(path))
    x = torch.randn(1, 1, 64)

    with torch.no_grad():
        expected = model._forward_sequence(x)
        actual = exported(x)

    assert config.model.params.layers == 6
    assert model.supported_export_formats == ("onnx", "torchscript")
    torch.testing.assert_close(actual, expected)


def test_wavenet_onnx_export_matches_pytorch(tmp_path: Path) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    model = _model()
    path = tmp_path / "wavenet.onnx"
    model.export_onnx(path)
    x = torch.randn(1, 1, 40)

    session = onnxruntime.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    actual = torch.from_numpy(session.run(None, {"input": x.numpy()})[0])

    with torch.no_grad():
        expected = model._forward_sequence(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
