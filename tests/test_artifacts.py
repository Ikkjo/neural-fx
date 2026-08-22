import dataclasses
import json
from pathlib import Path

import pytest
import torch

from neural_fx.artifacts import load_model
from neural_fx.config import load_config
from neural_fx.models import create_model_from_config

CONFIG_PATH = Path("configs/models/lstm/lstm_nano.yaml")


def _trained_model():
    config = load_config(CONFIG_PATH)
    model = create_model_from_config(config.model)
    torch.manual_seed(7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter))
    return model, config


def _assert_same_state(actual, expected) -> None:
    for name, expected_value in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[name], expected_value)


def test_loads_raw_state_dictionary_with_explicit_config(tmp_path: Path) -> None:
    model, config = _trained_model()
    checkpoint_path = tmp_path / "raw.pt"
    torch.save(model.state_dict(), checkpoint_path)

    loaded = load_model(
        checkpoint_path=checkpoint_path,
        config_path=CONFIG_PATH,
    )

    assert loaded.config == config
    assert loaded.provenance.config_origin == "explicit_file"
    assert loaded.provenance.weights_origin == "checkpoint"
    _assert_same_state(loaded.model, model)


def test_loads_lightning_state_and_embedded_config(tmp_path: Path) -> None:
    model, config = _trained_model()
    checkpoint_path = tmp_path / "lightning.ckpt"
    torch.save(
        {
            "state_dict": {
                **{f"model.{name}": value for name, value in model.state_dict().items()},
                "loss_fn.unrelated": torch.tensor(1.0),
            },
            "neural_fx_config": dataclasses.asdict(config),
        },
        checkpoint_path,
    )

    loaded = load_model(checkpoint_path=checkpoint_path)

    assert loaded.config == config
    assert loaded.provenance.config_origin == "checkpoint"
    _assert_same_state(loaded.model, model)


def test_loads_config_from_checkpoint_sidecar(tmp_path: Path) -> None:
    model, config = _trained_model()
    checkpoint_path = tmp_path / "legacy.ckpt"
    torch.save(model.state_dict(), checkpoint_path)
    checkpoint_path.with_suffix(".meta.json").write_text(
        json.dumps({"config": dataclasses.asdict(config)})
    )

    loaded = load_model(checkpoint_path=checkpoint_path)

    assert loaded.config == config
    assert loaded.provenance.config_origin == "sidecar"
    _assert_same_state(loaded.model, model)


def test_config_only_source_creates_an_initialized_model() -> None:
    loaded = load_model(config_path=CONFIG_PATH)

    assert loaded.provenance.weights_origin == "initialized"
    assert loaded.model.training is False


def test_rejects_checkpoint_without_model_state(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "invalid.ckpt"
    torch.save({"metadata": "not model weights"}, checkpoint_path)

    with pytest.raises(ValueError, match="model state dictionary"):
        load_model(checkpoint_path=checkpoint_path, config_path=CONFIG_PATH)


def test_requires_a_checkpoint_or_config() -> None:
    with pytest.raises(ValueError, match="checkpoint or config"):
        load_model()
