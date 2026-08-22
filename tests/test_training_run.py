from pathlib import Path
from unittest.mock import patch

import torch

from neural_fx.config import config_from_dict
from neural_fx.training.run import TrainingRun, _generate_plots


def _config(*, normalize: bool = True):
    return config_from_dict(
        {
            "version": "1.0",
            "name": "training-run",
            "model": {"type": "lstm", "params": {"hidden_size": 4}},
            "training": {"epochs": 12},
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
    config = _config()

    run = TrainingRun.resolve(
        config,
        checkpoint_dir="runs",
        max_epochs=3,
        latency_manual=27,
        patience=0,
        resume_path="resume.ckpt",
    )

    assert config.training.epochs == 12
    assert config.latency.method == "xcorr"
    assert config.latency.manual_delay is None
    assert run.config is not config
    assert run.config.training.epochs == 3
    assert run.config.latency.method == "manual"
    assert run.config.latency.manual_delay == 27
    assert run.checkpoint_dir == Path("runs")
    assert run.resume_path == Path("resume.ckpt")
    assert run.patience == 0


def test_post_training_report_uses_resolved_normalization(tmp_path: Path) -> None:
    run = TrainingRun.resolve(
        _config(normalize=False),
        checkpoint_dir=tmp_path,
    )

    with (
        patch("neural_fx.training.run.AudioDataset") as dataset,
        patch("neural_fx.analysis.plotting.TrainingAnalyzer") as analyzer,
    ):
        analyzer.return_value.generate_report.return_value = {
            "esr": 0.1,
            "esr_comment": "test",
        }
        _generate_plots(run, torch.nn.Identity(), "", None)

    assert dataset.call_args.kwargs["normalize"] is False
