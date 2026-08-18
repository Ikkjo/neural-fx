import math
from pathlib import Path

import pytest
import torch

from neural_fx.config import ModelConfig, SSMParams, load_config
from neural_fx.models import S4DModel, create_model_from_config


def _model() -> S4DModel:
    return S4DModel(
        ModelConfig(
            type="s4",
            params=SSMParams(d_model=4, d_state=4, num_layers=2),
        )
    ).eval()


def test_s4d_shape_registry_and_long_memory() -> None:
    model = _model()

    assert model(torch.randn(2, 1, 32), reset_state=True).shape == (2, 1, 32)
    assert math.isinf(model.receptive_field)
    assert isinstance(create_model_from_config(model.config), S4DModel)


def test_mamba_deferral_has_actionable_error() -> None:
    config = ModelConfig(type="mamba", params=SSMParams())

    with pytest.raises(NotImplementedError, match="portable core backend"):
        create_model_from_config(config)


def test_s4d_is_causal() -> None:
    model = _model()
    x = torch.randn(1, 1, 32)
    changed = x.clone()
    changed[..., 16:] += 10

    with torch.no_grad():
        original = model(x, reset_state=True)
        modified = model(changed, reset_state=True)

    torch.testing.assert_close(original[..., :16], modified[..., :16])


def test_s4d_chunks_and_recurrent_samples_match_full_sequence() -> None:
    model = _model()
    x = torch.randn(1, 1, 32)

    with torch.no_grad():
        full = model(x, reset_state=True)
        model.reset_state()
        chunked = torch.cat([model(x[..., :11]), model(x[..., 11:])], dim=-1)
        model.reset_state()
        sampled = torch.stack(
            [model.process_sample(x[..., i]).squeeze() for i in range(x.shape[-1])]
        ).reshape(1, 1, -1)

    torch.testing.assert_close(chunked, full, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(sampled, full, atol=2e-5, rtol=1e-4)


def test_s4d_config_and_torchscript_export(tmp_path: Path) -> None:
    config = load_config("configs/models/s4/s4_nano.yaml")
    model = create_model_from_config(config.model).eval()
    path = tmp_path / "s4.pt"

    model.export_torchscript(path)
    exported = torch.jit.load(str(path))
    x = torch.randn(1, 1, 32)
    state = torch.zeros(
        model.params.num_layers,
        1,
        model.params.d_model,
        model.params.d_state,
    )

    with torch.no_grad():
        expected = model(x, reset_state=True)
        samples = []
        for index in range(x.shape[-1]):
            output, state = exported(x[..., index : index + 1], state)
            samples.append(output)
        actual = torch.cat(samples, dim=-1)

    assert config.model.params.d_model == 8
    assert model.supported_export_formats == ("torchscript",)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
