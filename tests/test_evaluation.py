"""Tests for controlled model evaluation."""

import json
from pathlib import Path

import pytest
import torch
import yaml
from scipy.io import wavfile

from neural_fx.analysis.evaluation import (
    build_architecture_report,
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
    torch.save(model.state_dict(), checkpoint_path)
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
                "model": {"config": "config.yaml", "checkpoint": "model.pt"},
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
    assert result["dataset"]["metric_samples"] == 3968
    assert result["dataset"]["stft_window_starts"] == [0]
    assert result["inference"] == {
        "chunk_size": 1024,
        "chunks": 4,
        "state_reset_count": 1,
    }
    assert all(Path(path).exists() for path in result["artifacts"].values())
    assert json.loads((tmp_path / "result" / "evaluation.json").read_text())[
        "experiment_id"
    ] == "test-gru"


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
                "normalization": "paired_peak",
            },
            "performance": None,
        }

    report, markdown = build_comparison_report(
        [result("lstm", 1000, "lstm"), result("gru", 1200, "gru")],
        size_tolerance=1.25,
    )

    assert report["interpretation"] == "workflow_validation_only"
    assert report["size_groups"][0]["experiments"] == ["lstm", "gru"]
    assert "must not be used as a final quality ranking" in markdown
    assert "[checkpoint](/model.ckpt)" in markdown


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


def _evaluation_result(
    architecture: str,
    seed: int,
    esr: float,
    *,
    parameters: int,
) -> dict:
    checkpoint = f"/checkpoints/{architecture}-{seed}.ckpt"
    return {
        "schema_version": "1.0",
        "experiment_id": f"{architecture}-{seed}",
        "run_kind": "final",
        "sources": {
            "manifest": f"/manifests/{architecture}-{seed}.yaml",
            "config": f"/configs/{architecture}.yaml",
            "checkpoint": checkpoint,
        },
        "model": {
            "type": architecture,
            "name": architecture,
            "trainable_parameters": parameters,
        },
        "training": {"seed": seed},
        "metrics": {
            "esr": esr,
            "mse": esr * 0.5,
            "correlation": 1.0 - esr * 0.1,
            "multi_resolution_stft_distance": esr * 2.0,
        },
        "artifacts": {"prediction_audio": f"/{architecture}-{seed}.wav"},
    }


def _benchmark_result(
    architecture: str,
    checkpoint: str,
    run: int,
    *,
    p95_ms: float,
    real_time_factor: float,
    memory_bytes: int,
) -> dict:
    return {
        "schema_version": "1.0",
        "model": {
            "type": architecture,
            "checkpoint_path": checkpoint,
        },
        "runtime": {"device": "cpu"},
        "offline": {"real_time_factor": real_time_factor},
        "blocks": [
            {
                "block_size": 128,
                "p95_ms": p95_ms,
                "deadline_ms": 128 / 48_000 * 1000,
                "deadline_misses": 0,
            }
        ],
        "memory": {"process_peak_rss_bytes": memory_bytes},
        "_result_path": f"/{architecture}-benchmark-{run}.json",
    }


def test_architecture_report_aggregates_seeds_benchmarks_and_conclusions() -> None:
    results = [
        *[
            _evaluation_result("lstm", seed, esr, parameters=1000)
            for seed, esr in zip((17, 42, 137), (0.08, 0.09, 0.10))
        ],
        *[
            _evaluation_result("gru", seed, esr, parameters=1005)
            for seed, esr in zip((17, 42, 137), (0.12, 0.13, 0.14))
        ],
    ]
    benchmarks = [
        *[
            _benchmark_result(
                "lstm",
                "/checkpoints/lstm-42.ckpt",
                run,
                p95_ms=1.0 + run * 0.02,
                real_time_factor=0.10 + run * 0.002,
                memory_bytes=1000 + run,
            )
            for run in range(3)
        ],
        *[
            _benchmark_result(
                "gru",
                "/checkpoints/gru-42.ckpt",
                run,
                p95_ms=1.4 + run * 0.02,
                real_time_factor=0.14 + run * 0.002,
                memory_bytes=1200 + run,
            )
            for run in range(3)
        ],
    ]

    report, markdown = build_architecture_report(
        results,
        benchmarks,
        required_architectures=("lstm", "gru"),
    )

    lstm = report["architectures"][0]
    assert report["all_architectures_size_matched"] is True
    assert lstm["metrics"]["esr"]["median"] == pytest.approx(0.09)
    assert lstm["metrics"]["esr"]["raw"] == {
        "17": 0.08,
        "42": 0.09,
        "137": 0.10,
    }
    assert lstm["representative"]["seed"] == 42
    assert lstm["representative"]["listening_samples"]["prediction_audio"] == (
        "/lstm-42.wav"
    )
    assert lstm["benchmarks"]["cpu"]["status"] == "complete"
    assert lstm["benchmarks"]["cpu"]["blocks"]["128"]["real_time_capable"]
    assert report["conclusion"]["quality"]["winner"] == "lstm"
    assert report["conclusion"]["performance"]["pareto_winner"] == "lstm"
    assert report["conclusion"]["performance"]["material_differences"][
        "cpu_128_sample_p95"
    ]["winner"] == "lstm"
    assert "lstm is the clear quality winner" in markdown


def test_architecture_report_refuses_conclusion_for_missing_seed() -> None:
    results = [
        _evaluation_result("lstm", seed, esr, parameters=1000)
        for seed, esr in zip((17, 42), (0.08, 0.09))
    ]
    results.extend(
        _evaluation_result("gru", seed, esr, parameters=1000)
        for seed, esr in zip((17, 42, 137), (0.10, 0.11, 0.12))
    )

    report, _ = build_architecture_report(
        results,
        required_architectures=("lstm", "gru"),
    )

    assert report["architectures"][0]["missing_seeds"] == [137]
    assert report["conclusion"]["quality"] == {
        "status": "incomplete",
        "winner": None,
        "statement": (
            "architecture comparison is incomplete; no quality conclusion is allowed"
        ),
        "conditions": {},
    }


def test_architecture_report_emits_no_clear_winner_when_rule_fails() -> None:
    results = [
        *[
            _evaluation_result("lstm", seed, esr, parameters=1000)
            for seed, esr in zip((17, 42, 137), (0.100, 0.101, 0.102))
        ],
        *[
            _evaluation_result("gru", seed, esr, parameters=1000)
            for seed, esr in zip((17, 42, 137), (0.102, 0.103, 0.104))
        ],
    ]

    report, markdown = build_architecture_report(
        results,
        required_architectures=("lstm", "gru"),
    )

    quality = report["conclusion"]["quality"]
    assert quality["winner"] is None
    assert quality["statement"] == "no clear quality winner under this budget"
    assert "no clear quality winner under this budget" in markdown
