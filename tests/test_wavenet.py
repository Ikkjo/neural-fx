"""Direct tests for the causal WaveNet implementation."""

from pathlib import Path

import pytest
import torch

from neural_fx.config import ModelConfig, WaveNetParams, load_config
from neural_fx.models.errors import UnsupportedExportFormatError
from neural_fx.models.sizing import MODEL_SIZE_TARGETS
from neural_fx.models.wavenet import (
    CausalConv1d,
    DilatedResidualBlock,
    WaveNetModel,
)


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        type="wavenet",
        input_size=1,
        output_size=1,
        sample_rate=44100,
        params=WaveNetParams(
            layers=3,
            stacks=2,
            kernel_size=3,
            dilation_channels=4,
            residual_channels=4,
            skip_channels=8,
        ),
    )


def test_causal_convolution_preserves_shape_and_has_no_future_leakage():
    torch.manual_seed(0)
    convolution = CausalConv1d(2, 3, kernel_size=3, dilation=2)
    first = torch.randn(1, 2, 16)
    second = first.clone()
    second[..., 9:] = torch.randn_like(second[..., 9:])

    first_output = convolution(first)
    second_output = convolution(second)

    assert first_output.shape == (1, 3, 16)
    assert torch.allclose(first_output[..., :9], second_output[..., :9])


def test_causal_convolution_respects_dilation():
    convolution = CausalConv1d(1, 1, kernel_size=3, dilation=2, bias=False)
    convolution.weight.data.fill_(1.0)
    impulse = torch.zeros(1, 1, 7)
    impulse[..., 0] = 1.0

    output = convolution(impulse).flatten()

    assert torch.equal(output.nonzero().flatten(), torch.tensor([0, 2, 4]))


def test_cached_causal_convolution_matches_vectorized_path():
    torch.manual_seed(1)
    convolution = CausalConv1d(2, 3, kernel_size=3, dilation=4)
    samples = torch.randn(2, 2, 25)

    expected = convolution(samples)
    convolution.reset_state()
    actual = torch.stack(
        [convolution.process_sample(samples[..., index]) for index in range(25)],
        dim=-1,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_residual_block_preserves_residual_shape_and_projects_skip():
    block = DilatedResidualBlock(
        residual_channels=4,
        dilation_channels=6,
        skip_channels=8,
        kernel_size=3,
        dilation=2,
    )

    residual, skip = block(torch.randn(2, 4, 32))

    assert residual.shape == (2, 4, 32)
    assert skip.shape == (2, 8, 32)


def test_dilation_cycle_and_receptive_field_are_documented_contract(model_config):
    model = WaveNetModel(model_config)

    assert [block.dilation for block in model.blocks] == [1, 2, 4, 1, 2, 4]
    assert model.receptive_field == 1 + 2 * (3 - 1) * (2**3 - 1)
    assert model.sample_rate == 44100


def test_forward_is_strictly_causal(model_config):
    torch.manual_seed(2)
    model = WaveNetModel(model_config).eval()
    first = torch.randn(1, 1, 64)
    second = first.clone()
    second[..., 40:] = torch.randn_like(second[..., 40:])

    first_output = model(first, reset_state=True)
    second_output = model(second, reset_state=True)

    assert torch.allclose(first_output[..., :40], second_output[..., :40])


def test_full_chunked_and_sample_outputs_match(model_config):
    torch.manual_seed(3)
    model = WaveNetModel(model_config).eval()
    inputs = torch.randn(2, 1, 73)

    expected = model(inputs, reset_state=True)

    model.reset_state()
    chunked = torch.cat(
        [model(inputs[..., start : start + 11]) for start in range(0, 73, 11)],
        dim=-1,
    )

    model.reset_state()
    sample_wise = torch.stack(
        [model.process_sample(inputs[..., index]) for index in range(73)], dim=-1
    )

    assert torch.allclose(chunked, expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(sample_wise, expected, atol=1e-6, rtol=1e-5)


def test_reset_and_detach_manage_both_streaming_paths(model_config):
    model = WaveNetModel(model_config)
    inputs = torch.randn(1, 1, 12, requires_grad=True)

    expected = model(inputs, reset_state=True)
    assert model._input_history.grad_fn is not None
    model.detach_state()
    assert model._input_history.grad_fn is None

    model.reset_state()
    actual = model(inputs)
    assert torch.allclose(actual, expected)

    model.reset_state()
    model.process_sample(inputs.detach()[..., 0])
    assert all(block.filter_conv._stream_buffer.numel() for block in model.blocks)
    model.reset_state()
    assert all(
        block.filter_conv._stream_buffer.numel() == 0 for block in model.blocks
    )


def test_from_config_accepts_dict_and_rejects_other_model_types():
    model = WaveNetModel.from_config(
        {
            "type": "wavenet",
            "input_size": 2,
            "output_size": 1,
            "sample_rate": 32000,
            "params": {
                "layers": 2,
                "stacks": 1,
                "kernel_size": 3,
                "dilation_channels": 4,
                "residual_channels": 4,
                "skip_channels": 8,
            },
        }
    )

    assert model.config.input_size == 2
    assert model.sample_rate == 32000
    with pytest.raises(ValueError, match="Expected model type"):
        WaveNetModel(
            ModelConfig(type="lstm", params=WaveNetParams(layers=2))
        )


def _candidate_parameter_count(layers: int, stacks: int, channels: int) -> int:
    blocks = layers * stacks
    return blocks * (9 * channels**2 + 5 * channels) + 4 * channels**2 + 6 * channels + 1


def test_size_configs_are_deterministically_nearest_shared_targets():
    expected_counts = {
        "nano": 2_769,
        "small": 17_809,
        "medium": 60_545,
        "large": 129_745,
        "xl": 219_265,
    }
    channel_grid = [8, 12, 16, 24, 32, 48, 64]
    config_dir = Path("configs/models/wavenet")

    for size, target in MODEL_SIZE_TARGETS.items():
        config = load_config(config_dir / f"wavenet_{size}.yaml")
        model = WaveNetModel(config.model)
        params = config.model.params
        candidates = [
            (
                abs(_candidate_parameter_count(layers, stacks, channels) - target),
                layers,
                channels,
                stacks,
                _candidate_parameter_count(layers, stacks, channels),
            )
            for layers in range(4, 13)
            for stacks in range(1, 4)
            for channels in channel_grid
        ]
        nearest = min(candidates)

        assert (
            params.layers,
            params.residual_channels,
            params.stacks,
            model.num_params,
        ) == (nearest[1], nearest[2], nearest[3], nearest[4])
        assert model.num_params == expected_counts[size]
        assert params.dilation_channels == params.residual_channels
        assert params.skip_channels == 2 * params.residual_channels


def test_torchscript_export_matches_pytorch(model_config, tmp_path):
    model = WaveNetModel(model_config).eval()
    inputs = torch.randn(1, 1, 64)
    path = tmp_path / "wavenet.pt"

    model.export_torchscript(path)
    exported = torch.jit.load(str(path))

    assert torch.allclose(exported(inputs), model._forward_stateless(inputs))


def test_onnx_export_writes_model(model_config, tmp_path):
    pytest.importorskip("onnx")
    model = WaveNetModel(model_config).eval()
    path = tmp_path / "wavenet.onnx"

    model.export_onnx(path)

    assert path.is_file()
    assert path.stat().st_size > 0


def test_rtneural_export_is_explicitly_unsupported(model_config, tmp_path):
    model = WaveNetModel(model_config)

    with pytest.raises(UnsupportedExportFormatError, match="rtneural"):
        model.export_rtneural(tmp_path / "wavenet.json")
