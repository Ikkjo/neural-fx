"""Tests for portable S4D sequence and recurrent inference."""

from pathlib import Path

import pytest
import torch

from neural_fx.config import ModelConfig, SSMParams, load_config
from neural_fx.models.errors import UnsupportedExportFormatError
from neural_fx.models.ssm import S4DLayer, S4Model


def make_model(
    *, d_model: int = 4, d_state: int = 8, num_layers: int = 2
) -> S4Model:
    config = ModelConfig(
        type="s4",
        params=SSMParams(
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            dropout=0.0,
        ),
        input_size=1,
        output_size=1,
        sample_rate=44_100,
    )
    return S4Model(config)


def test_layer_sequence_and_recurrent_paths_match() -> None:
    torch.manual_seed(7)
    layer = S4DLayer(d_model=3, d_state=6).eval()
    input = torch.randn(2, 3, 48)

    sequence = layer(input, use_fft=True)
    layer.reset_state()
    recurrent = torch.stack(
        [layer.step(input[..., index]) for index in range(input.shape[-1])],
        dim=-1,
    )

    torch.testing.assert_close(sequence, recurrent, rtol=2e-4, atol=2e-5)


def test_layer_fft_and_direct_convolution_match() -> None:
    layer = S4DLayer(d_model=2, d_state=4).eval()
    input = torch.randn(1, 2, 31)

    fft_output = layer(input, use_fft=True, update_state=False)
    direct_output = layer(input, use_fft=False, update_state=False)

    torch.testing.assert_close(fft_output, direct_output, rtol=1e-4, atol=1e-5)


def test_complex_and_real_kernel_formulations_match() -> None:
    layer = S4DLayer(d_model=3, d_state=7)

    torch.testing.assert_close(
        layer._kernel(29), layer._real_kernel(29), rtol=2e-4, atol=2e-5
    )


def test_model_shape_gradients_and_finiteness() -> None:
    model = make_model()
    input = torch.randn(2, 1, 64, requires_grad=True)

    output = model(input)
    output.square().mean().backward()

    assert output.shape == input.shape
    assert torch.isfinite(output).all()
    assert input.grad is not None
    assert torch.isfinite(input.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_model_is_strictly_causal() -> None:
    torch.manual_seed(11)
    model = make_model().eval()
    original = torch.randn(1, 1, 80)
    changed = original.clone()
    changed[..., 40:] = torch.randn_like(changed[..., 40:])

    first = model(original, reset_state=True)
    second = model(changed, reset_state=True)

    torch.testing.assert_close(first[..., :40], second[..., :40])


def test_long_sequence_remains_finite() -> None:
    model = make_model(d_model=2, d_state=16, num_layers=1).eval()
    output = model(torch.randn(1, 1, 4096), reset_state=True)

    assert torch.isfinite(output).all()


def test_full_chunk_and_sample_inference_match() -> None:
    torch.manual_seed(19)
    model = make_model().eval()
    input = torch.randn(1, 1, 96)

    full = model(input, reset_state=True)

    model.reset_state()
    chunked = torch.cat(
        [model(input[..., :37]), model(input[..., 37:71]), model(input[..., 71:])],
        dim=-1,
    )

    model.reset_state()
    sampled = torch.stack(
        [model.process_sample(input[0, :, index]) for index in range(input.shape[-1])],
        dim=-1,
    ).unsqueeze(0)

    torch.testing.assert_close(full, chunked, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(full, sampled, rtol=3e-4, atol=3e-5)


def test_state_reset_detach_and_batch_validation() -> None:
    model = make_model()
    model(torch.randn(2, 1, 16))

    assert all(block.s4.state is not None for block in model.blocks)
    model.detach_state()
    assert all(not block.s4.state.requires_grad for block in model.blocks)

    with pytest.raises(ValueError, match="batch size"):
        model(torch.randn(1, 1, 8))

    model.reset_state()
    assert all(block.s4.state is None for block in model.blocks)


def test_factory_and_model_contract() -> None:
    model = S4Model.from_config(
        {
            "type": "s4",
            "sample_rate": 44_100,
            "params": {"d_model": 8, "d_state": 16, "num_layers": 2},
        }
    )

    assert model.model_type == "s4"
    assert model.sample_rate == 44_100
    assert model.receptive_field == -1
    assert model.has_state

    with pytest.raises(ValueError, match="Expected model type"):
        S4Model.from_config({"type": "mamba"})


@pytest.mark.parametrize(
    ("variant", "expected_count"),
    [
        ("nano", 3_065),
        ("small", 18_017),
        ("medium", 60_497),
        ("large", 126_529),
        ("xl", 207_457),
    ],
)
def test_size_configs_have_locked_parameter_counts(variant, expected_count) -> None:
    path = Path("configs/models/s4") / f"s4_{variant}.yaml"
    config = load_config(path)
    model = S4Model(config.model)

    assert model.num_params == expected_count


def test_torchscript_export_round_trip(tmp_path) -> None:
    model = make_model(d_model=2, d_state=4, num_layers=1).eval()
    path = tmp_path / "nested" / "model.pt"
    model.export_torchscript(path)

    exported = torch.jit.load(str(path))
    input = torch.randn(1, 1, 512)
    expected = model._forward_export(input)
    torch.testing.assert_close(exported(input), expected)


def test_onnx_export(tmp_path) -> None:
    pytest.importorskip("onnx")
    model = make_model(d_model=2, d_state=4, num_layers=1).eval()
    path = tmp_path / "model.onnx"

    model.export_onnx(path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_rtneural_export_is_explicitly_unsupported(tmp_path) -> None:
    model = make_model()
    with pytest.raises(UnsupportedExportFormatError, match="rtneural"):
        model.export_rtneural(tmp_path / "model.json")
