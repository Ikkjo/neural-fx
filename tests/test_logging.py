"""Tests for training logger configuration and metric naming."""

from pathlib import Path

import pytest
import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from neural_fx.config import (
    DataConfig,
    DataPaths,
    LoggingConfig,
    LossConfig,
    LRSchedulerConfig,
    LSTMParams,
    ModelConfig,
    NeuralFXConfig,
    OptimizerConfig,
    TrainingConfig,
    load_config,
)
from neural_fx.models.recurrent import NeuralfxLSTM
from neural_fx.training.lightning_module import NeuralFXModule
from scripts.train import build_loggers, resolve_logger_version


def _write_config(tmp_path: Path, logging_yaml: str = "") -> Path:
    path = tmp_path / "logging.yaml"
    path.write_text(
        "\n".join(
            [
                'version: "1.0"',
                'name: "logging_test"',
                "model:",
                '  type: "lstm"',
                "  params:",
                "    hidden_size: 8",
                "training: {}",
                "loss:",
                '  type: "mse"',
                "data:",
                "  train:",
                '    input: "input.wav"',
                '    target: "target.wav"',
                logging_yaml,
            ]
        )
        + "\n"
    )
    return path


def _config() -> NeuralFXConfig:
    return NeuralFXConfig(
        version="1.0",
        name="logging_test",
        model=ModelConfig(type="lstm", params=LSTMParams(hidden_size=8)),
        training=TrainingConfig(batch_size=2, epochs=3, segment_length=32),
        optimizer=OptimizerConfig(type="adam", lr=0.001),
        lr_scheduler=LRSchedulerConfig(type="exponential", gamma=0.99),
        loss=LossConfig(type="mse"),
        data=DataConfig(train=DataPaths(input="input.wav", target="target.wav")),
    )


def test_legacy_config_gets_both_logging_backends(tmp_path):
    config = load_config(_write_config(tmp_path))

    assert config.logging == LoggingConfig()


def test_logging_config_overrides_are_loaded(tmp_path):
    config = load_config(
        _write_config(
            tmp_path,
            "\n".join(
                [
                    "logging:",
                    '  backends: ["csv"]',
                    '  save_dir: "custom_logs"',
                    "  log_every_n_steps: 7",
                ]
            ),
        )
    )

    assert config.logging.backends == ["csv"]
    assert config.logging.save_dir == "custom_logs"
    assert config.logging.log_every_n_steps == 7


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backends": []}, "At least one"),
        ({"backends": ["csv", "csv"]}, "duplicates"),
        ({"backends": ["unknown"]}, "Unknown logging backend"),
        ({"log_every_n_steps": 0}, "greater than zero"),
    ],
)
def test_invalid_logging_config_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LoggingConfig(**kwargs)


def test_loggers_share_explicit_version(tmp_path):
    config = LoggingConfig(save_dir=str(tmp_path))

    loggers = build_loggers(config, "experiment", version=3)

    assert [type(logger) for logger in loggers] == [CSVLogger, TensorBoardLogger]
    assert all(logger.version == 3 for logger in loggers)


def test_new_and_resumed_runs_resolve_logger_versions(tmp_path):
    experiment_dir = tmp_path / "experiment"
    (experiment_dir / "version_0").mkdir(parents=True)
    (experiment_dir / "version_2").mkdir()

    assert resolve_logger_version(tmp_path, "experiment", None) == 3
    assert (
        resolve_logger_version(
            tmp_path,
            "experiment",
            str(experiment_dir / "version_2" / "checkpoints" / "last.ckpt"),
        )
        == 2
    )
    assert (
        resolve_logger_version(
            tmp_path,
            "experiment",
            str(experiment_dir / "legacy-last.ckpt"),
        )
        == 2
    )


def test_hyperparameters_are_tensorboard_safe_scalars():
    config = _config()
    model = NeuralfxLSTM(config.model)

    module = NeuralFXModule(model, config)

    assert module.hparams["parameter_count"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert module.hparams["sample_rate"] == 48000
    assert all(
        isinstance(value, (bool, float, int, str))
        for value in module.hparams.values()
    )
    assert -1 not in module.hparams.values()


def test_training_loss_has_distinct_step_and_epoch_series(monkeypatch):
    config = _config()
    module = NeuralFXModule(NeuralfxLSTM(config.model), config)
    calls = []
    monkeypatch.setattr(
        module,
        "log",
        lambda name, value, **kwargs: calls.append((name, value, kwargs)),
    )

    module._log_training_loss(torch.tensor(1.25), batch_size=2)

    assert [name for name, _, _ in calls] == ["train_loss_step", "train_loss"]
    assert calls[0][2]["on_step"] is True
    assert calls[0][2]["on_epoch"] is False
    assert calls[1][2]["on_step"] is False
    assert calls[1][2]["on_epoch"] is True
