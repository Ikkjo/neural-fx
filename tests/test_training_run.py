from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from neural_fx.config import config_from_dict
from neural_fx.models import create_model_from_config
from neural_fx.training.lightning_module import NeuralFXModule
from neural_fx.training.run import (
    TrainingRun,
    _create_training_model,
    _generate_plots,
    _training_callbacks,
    _validate_compile_request,
)
from scripts.train import _parse_args


def _config(
    *,
    normalize: bool = True,
    compile: bool = False,
    tbptt: bool = False,
):
    return config_from_dict(
        {
            "version": "1.0",
            "name": "training-run",
            "model": {"type": "lstm", "params": {"hidden_size": 4}},
            "training": {
                "epochs": 12,
                "compile": compile,
                "tbptt": {"enabled": tbptt},
                "early_stopping_patience": 14,
                "early_stopping_min_delta": 0.01,
                "early_stopping_min_delta_mode": "relative",
            },
            "loss": {"type": "mse"},
            "data": {
                "normalize": normalize,
                "train": {"input": "input.wav", "target": "target.wav"},
            },
            "latency": {
                "method": "xcorr",
                "calibration_duration_seconds": 0,
            },
        }
    )


def test_training_run_resolves_overrides_without_mutating_source_config() -> None:
    config = _config(compile=True)

    run = TrainingRun.resolve(
        config,
        checkpoint_dir="runs",
        max_epochs=3,
        latency_manual=27,
        patience=0,
        min_delta=0.02,
        min_delta_mode="absolute",
        resume_path="resume.ckpt",
        compile=False,
    )

    assert config.training.epochs == 12
    assert config.training.compile is True
    assert config.latency.method == "xcorr"
    assert config.latency.manual_delay is None
    assert run.config is not config
    assert run.config.training.epochs == 3
    assert run.config.training.compile is False
    assert run.config.latency.method == "manual"
    assert run.config.latency.manual_delay == 27
    assert run.checkpoint_dir == Path("runs")
    assert run.resume_path == Path("resume.ckpt")
    assert run.patience == 0
    assert run.min_delta == 0.02
    assert run.min_delta_mode == "absolute"


def test_training_run_inherits_compile_setting_without_override() -> None:
    run = TrainingRun.resolve(_config(compile=True))

    assert run.config.training.compile is True


def test_training_run_can_override_data_loader_workers() -> None:
    config = _config()

    run = TrainingRun.resolve(config, num_workers=0)

    assert config.training.num_workers == 4
    assert run.config.training.num_workers == 0
    assert run.patience == 14
    assert run.min_delta == 0.01
    assert run.min_delta_mode == "relative"


def test_relative_early_stopping_config_reaches_callback(tmp_path: Path) -> None:
    run = TrainingRun.resolve(_config(), checkpoint_dir=tmp_path)

    callbacks, _ = _training_callbacks(run, latency_calibration=None)
    early_stopping = callbacks[1]

    assert early_stopping.monitor == "train_loss"
    assert early_stopping.patience == 14
    assert early_stopping.relative_min_delta == pytest.approx(0.01)
    assert early_stopping.min_delta_mode == "relative"


@pytest.mark.parametrize(
    ("flag", "expected"),
    [(None, None), ("--compile", True), ("--no-compile", False)],
)
def test_compile_cli_override(flag: str | None, expected: bool | None) -> None:
    argv = ["--config", "config.yaml"]
    if flag is not None:
        argv.append(flag)

    assert _parse_args(argv).compile is expected


def test_num_workers_cli_override() -> None:
    assert _parse_args(["--config", "example.yaml", "--num-workers", "0"]).num_workers == 0


def test_early_stopping_cli_overrides_are_parsed() -> None:
    args = _parse_args(
        [
            "--config",
            "config.yaml",
            "--patience",
            "20",
            "--min_delta",
            "0.005",
            "--min_delta_mode",
            "relative",
        ]
    )

    assert args.patience == 20
    assert args.min_delta == 0.005
    assert args.min_delta_mode == "relative"


def test_compiled_training_rejects_multiple_gpus() -> None:
    run = TrainingRun.resolve(_config(compile=True), gpus=2)

    with pytest.raises(ValueError, match="at most one GPU"):
        _validate_compile_request(run)


def test_compiled_training_rejects_tbptt() -> None:
    run = TrainingRun.resolve(_config(compile=True, tbptt=True))

    with pytest.raises(ValueError, match="does not support TBPTT"):
        _validate_compile_request(run)


def test_compile_seam_preserves_model_and_optimizer_ownership() -> None:
    config = _config(compile=True)
    model = create_model_from_config(config.model)
    state_keys = tuple(model.state_dict())
    parameter_ids = {id(parameter) for parameter in model.parameters()}

    with (
        patch(
            "neural_fx.training.run.create_model_from_config",
            return_value=model,
        ),
        patch.object(model, "compile") as compile_model,
    ):
        compiled_model = _create_training_model(config)

    module = NeuralFXModule(compiled_model, config)
    optimizer = module.configure_optimizers()[0][0]

    compile_model.assert_called_once_with()
    assert compiled_model is model
    assert tuple(model.state_dict()) == state_keys
    assert {id(parameter) for parameter in model.parameters()} == parameter_ids
    assert tuple(module.state_dict()) == tuple(f"model.{key}" for key in state_keys)
    assert {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    } == parameter_ids


def test_compile_failure_propagates_without_eager_retry() -> None:
    config = _config(compile=True)
    model = MagicMock()
    model.compile.side_effect = RuntimeError("compiler failed")

    with patch(
        "neural_fx.training.run.create_model_from_config",
        return_value=model,
    ) as create_model:
        with pytest.raises(RuntimeError, match="compiler failed"):
            _create_training_model(config)

    create_model.assert_called_once_with(config.model)
    model.compile.assert_called_once_with()


def test_post_training_report_uses_resolved_normalization(tmp_path: Path) -> None:
    run = TrainingRun.resolve(
        _config(normalize=False),
        checkpoint_dir=tmp_path,
    )

    with (
        patch("neural_fx.training.run.AudioDataset") as dataset,
        patch("neural_fx.analysis.plotting.generate_analysis_report") as report,
    ):
        report.return_value = {
            "esr": 0.1,
            "esr_comment": "test",
        }
        _generate_plots(run, torch.nn.Identity(), "", None)

    assert dataset.call_args.kwargs["normalize"] is False
