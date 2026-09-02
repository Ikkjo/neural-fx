"""Tests for controlled model evaluation."""

import json
from pathlib import Path

import pytest
import torch
import yaml
from scipy.io import wavfile

from neural_fx.analysis.evaluation import (
    build_comparison_report,
    evaluate_experiment,
    load_experiment_manifest,
    run_chunked_inference,
    write_evaluation_result,
)
from neural_fx.config import LSTMParams, ModelConfig, SSMParams, WaveNetParams
from neural_fx.models import create_model_from_config
from neural_fx.models.recurrent import NeuralfxGRU


def _write_test_config(path: Path) -> None:
    config = {
        "version": "1.0",
        "name": "test_gru",
        "model": {
            "type": "gru",
            "input_size": 1,
            "output_size": 1,
            "sample_rate": 48_000,
            "params": {"hidden_size": 4, "num_layers": 1},
        },
        "training": {"batch_size": 1, "epochs": 1, "segment_length": 4096},
        "optimizer": {"type": "adam", "lr": 0.001},
        "lr_scheduler": {"type": "exponential", "gamma": 0.99},
        "loss": {"type": "mse", "mask_first": 64},
        "data": {"train": {"input": "unused.wav", "target": "unused.wav"}},
    }
    path.write_text(yaml.safe_dump(config))


def test_manifest_evaluation_writes_metrics_and_listening_samples(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "model.pt"
    input_path = tmp_path / "input.wav"
    target_path = tmp_path / "target.wav"
    manifest_path = tmp_path / "manifest.yaml"
    _write_test_config(config_path)
    model = NeuralfxGRU(
        ModelConfig(type="gru", params=LSTMParams(hidden_size=4, num_layers=1))
    )
    torch.save(
        {
            "state_dict": {
                f"model.{key}": value for key, value in model.state_dict().items()
            },
            "epoch": 3,
            "global_step": 8,
            "callbacks": {
                "NeuralFXCheckpoint": {
                    "monitor": "val_loss",
                    "current_score": torch.tensor(0.25),
                }
            },
        },
        checkpoint_path,
    )
    time = torch.arange(4096) / 48_000
    input_audio = torch.sin(2 * torch.pi * 220 * time).numpy().astype("float32")
    wavfile.write(input_path, 48_000, input_audio)
    wavfile.write(target_path, 48_000, (input_audio * 0.5).astype("float32"))
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "experiment_id": "test-gru",
                "run_kind": "smoke",
                "esr_mode": "nam",
                "esr_pre_emphasis": None,
                "model": {
                    "config": "config.yaml",
                    "checkpoint": "model.pt",
                    "checkpoint_policy": "last",
                },
                "dataset": {
                    "input_audio": "input.wav",
                    "target_audio": "target.wav",
                    "split": "test",
                    "num_samples": 4096,
                    "metric_mask_first": 128,
                },
                "training": {"seed": 7, "epochs": 1},
            }
        )
    )

    manifest = load_experiment_manifest(manifest_path)
    result = evaluate_experiment(
        manifest,
        tmp_path / "result",
        inference_chunk_size=1024,
    )
    write_evaluation_result(result, tmp_path / "result" / "evaluation.json")

    assert set(result) == {
        "schema_version",
        "created_at",
        "experiment_id",
        "run_kind",
        "sources",
        "model",
        "checkpoint",
        "training",
        "inference",
        "dataset",
        "metrics",
        "performance",
        "artifacts",
        "notes",
    }
    assert result["run_kind"] == "smoke"
    assert set(result["metrics"]) == {
        "esr",
        "mse",
        "correlation",
        "multi_resolution_stft_distance",
    }
    assert result["dataset"]["evaluated_samples"] == 4096
    assert result["dataset"]["mask_first"] == 128
    assert result["dataset"]["configured_loss_mask_first"] == 64
    assert result["dataset"]["esr_mode"] == "nam"
    assert result["model"]["checkpoint_policy"] == "last"
    assert result["checkpoint"] == {
        "epoch": 3,
        "global_step": 8,
        "monitor": "val_loss",
        "monitor_value": 0.25,
    }
    assert result["dataset"]["metric_samples"] == 3968
    assert result["dataset"]["stft_window_starts"] == [0]
    assert result["inference"] == {
        "chunk_size": 1024,
        "chunks": 4,
        "state_reset_count": 1,
    }
    assert all(Path(path).exists() for path in result["artifacts"].values())
    assert {
        key: Path(path).name for key, path in result["artifacts"].items()
    } == {
        "input_audio": "input.wav",
        "target_audio": "target.wav",
        "prediction_audio": "prediction.wav",
    }
    assert (
        json.loads((tmp_path / "result" / "evaluation.json").read_text())[
            "experiment_id"
        ]
        == "test-gru"
    )


def test_comparison_report_groups_measured_sizes_and_marks_smoke_results() -> None:
    def result(experiment: str, parameters: int, model_type: str) -> dict:
        return {
            "experiment_id": experiment,
            "run_kind": "smoke",
            "sources": {
                "manifest": "/manifest.yaml",
                "config": "/config.yaml",
                "checkpoint": "/model.ckpt",
            },
            "model": {
                "type": model_type,
                "name": experiment,
                "trainable_parameters": parameters,
            },
            "metrics": {
                "esr": 1.0,
                "mse": 0.5,
                "correlation": 0.0,
                "multi_resolution_stft_distance": 2.0,
            },
            "dataset": {
                "input_audio": "/input.wav",
                "target_audio": "/target.wav",
                "split": "test",
                "start_sample": 0,
                "evaluated_samples": 4096,
                "sample_rate": 48_000,
                "latency_samples": 0,
                "preparation_delay_samples": 0,
                "normalization": "paired_peak",
                "mask_first": 0,
                "metric_samples": 4096,
                "esr_mode": "nam",
                "esr_pre_emphasis": None,
            },
            "performance": None,
        }

    report, markdown = build_comparison_report(
        [result("lstm", 1000, "lstm"), result("gru", 1200, "gru")],
        size_tolerance=1.25,
    )

    assert report["interpretation"] == "workflow_validation_only"
    assert report["primary_metric"] == {
        "name": "esr",
        "direction": "lower_is_better",
        "secondary_metrics": [
            "mse",
            "correlation",
            "multi_resolution_stft_distance",
        ],
    }
    assert [row["esr_rank"] for row in report["results"]] == [1, 2]
    assert "ESR is the primary ranking metric" in markdown
    assert report["size_groups"][0]["experiments"] == ["lstm", "gru"]
    assert "must not be used as a final quality ranking" in markdown
    assert "[checkpoint](/model.ckpt)" in markdown


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("esr_mode", "legacy"),
        ("esr_pre_emphasis", 0.85),
        ("preparation_delay_samples", -41),
    ],
)
def test_comparison_report_rejects_mismatched_esr_recipe(key: str, value: object) -> None:
    def result(experiment: str) -> dict:
        return {
            "experiment_id": experiment,
            "run_kind": "smoke",
            "sources": {},
            "model": {"trainable_parameters": 1},
            "metrics": {"esr": 1.0},
            "dataset": {
                "input_audio": "/input.wav",
                "target_audio": "/target.wav",
                "split": "test",
                "start_sample": 0,
                "evaluated_samples": 4096,
                "sample_rate": 48_000,
                "latency_samples": 0,
                "preparation_delay_samples": 0,
                "normalization": "paired_peak",
                "mask_first": 0,
                "metric_samples": 4096,
                "esr_mode": "nam",
                "esr_pre_emphasis": None,
            },
        }

    first, second = result("first"), result("second")
    second["dataset"][key] = value
    with pytest.raises(ValueError, match="same aligned dataset segment"):
        build_comparison_report([first, second])


def test_comparison_report_requires_esr_recipe() -> None:
    result = {
        "experiment_id": "incomplete",
        "run_kind": "smoke",
        "sources": {},
        "model": {"trainable_parameters": 1},
        "metrics": {"esr": 1.0},
        "dataset": {},
    }
    with pytest.raises(ValueError, match="must record esr_mode and esr_pre_emphasis"):
        build_comparison_report([result])


@pytest.mark.parametrize(
    "config",
    [
        ModelConfig(type="lstm", params=LSTMParams(hidden_size=4, num_layers=1)),
        ModelConfig(type="gru", params=LSTMParams(hidden_size=4, num_layers=1)),
        ModelConfig(
            type="wavenet",
            params=WaveNetParams(
                layers=2,
                stacks=1,
                kernel_size=2,
                dilation_channels=4,
                residual_channels=4,
                skip_channels=8,
            ),
        ),
        ModelConfig(
            type="s4",
            params=SSMParams(d_model=4, d_state=4, num_layers=1),
        ),
    ],
    ids=("lstm", "gru", "wavenet", "s4"),
)
def test_chunked_inference_matches_whole_inference(config: ModelConfig) -> None:
    torch.manual_seed(9)
    model = create_model_from_config(config).eval()
    audio = torch.randn(1, 1, 129)

    model.reset_state()
    with torch.inference_mode():
        whole = model(audio)
    chunked = run_chunked_inference(model, audio, chunk_size=31)

    torch.testing.assert_close(chunked, whole, atol=2e-5, rtol=1e-4)


def test_chunked_inference_does_not_build_an_autograd_graph() -> None:
    model = create_model_from_config(
        ModelConfig(type="lstm", params=LSTMParams(hidden_size=4, num_layers=1))
    ).eval()

    output = run_chunked_inference(
        model,
        torch.randn(1, 1, 129, requires_grad=True),
        chunk_size=31,
    )

    assert output.requires_grad is False
    assert output.grad_fn is None
