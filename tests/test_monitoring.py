"""Contract tests for fixed-suite offline monitoring."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torchaudio
import yaml

from neural_fx.config import LSTMParams, ModelConfig
from neural_fx.models.recurrent import NeuralfxGRU
from neural_fx.monitoring import (
    MonitoringError,
    load_monitoring_report,
    monitor_artifact,
)
from neural_fx.monitoring.execution import latency_summary
from neural_fx.monitoring.schema import (
    fingerprint_monitoring_suite,
    load_monitoring_manifest,
)


def _manifest_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": "1.0",
        "suite_id": "fixed-suite",
        "sample_rate": 48_000,
        "channels": 1,
        "segment_length": 4096,
        "burn_in_samples": 64,
        "inference_chunk_size": 1024,
        "latency_block_sizes": [128, 512],
        "warmup_runs": 1,
        "measurement_runs": 3,
        "quality_metrics": [
            "esr",
            "mse",
            "multi_resolution_stft_distance",
        ],
        "esr_pre_emphasis": None,
        "clipping_threshold": 0.999,
        "max_abs": 1.0,
        "cases": [
            {
                "id": "case-a",
                "input": "audio/input.wav",
                "target": "audio/target.wav",
            }
        ],
    }
    data.update(overrides)
    return data


def _write_suite(root: Path, **overrides: object) -> Path:
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "input.wav").write_bytes(b"input-audio")
    (audio_dir / "target.wav").write_bytes(b"target-audio")
    manifest_path = root / "suite.yaml"
    manifest_path.write_text(yaml.safe_dump(_manifest_data(**overrides)))
    return manifest_path


def _write_audio(root: Path, sample_rate: int = 48_000) -> tuple[Path, Path]:
    time_axis = torch.arange(4096) / sample_rate
    input_audio = (0.4 * torch.sin(2 * torch.pi * 220 * time_axis)).unsqueeze(0)
    target_audio = torch.tanh(input_audio * 1.5)
    input_path = root / "audio" / "input.wav"
    target_path = root / "audio" / "target.wav"
    torchaudio.save(input_path, input_audio, sample_rate)
    torchaudio.save(target_path, target_audio, sample_rate)
    return input_path, target_path


def _write_model_artifacts(
    root: Path, *, include_torchscript: bool = False
) -> tuple[Path, Path, Path]:
    config_path = root / "config.yaml"
    checkpoint_path = root / "model.ckpt"
    torchscript_path = root / "model.pt"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "1.0",
                "name": "monitor-gru",
                "model": {
                    "type": "gru",
                    "input_size": 1,
                    "output_size": 1,
                    "sample_rate": 48_000,
                    "params": {"hidden_size": 4, "num_layers": 1},
                },
                "training": {
                    "batch_size": 1,
                    "epochs": 1,
                    "segment_length": 4096,
                },
                "optimizer": {"type": "adam", "lr": 0.001},
                "lr_scheduler": {"type": "exponential", "gamma": 0.99},
                "loss": {"type": "mse", "mask_first": 64},
                "data": {
                    "train": {"input": "unused.wav", "target": "unused.wav"}
                },
                "latency": {"calibration_duration_seconds": 0},
            }
        )
    )
    model = NeuralfxGRU(
        ModelConfig(
            type="gru",
            params=LSTMParams(hidden_size=4, num_layers=1),
        )
    ).eval()
    torch.save(model.state_dict(), checkpoint_path)
    if include_torchscript:
        model.export_torchscript(torchscript_path)
    return config_path, checkpoint_path, torchscript_path


def test_manifest_resolves_paths_and_applies_compatibility_defaults(
    tmp_path: Path,
) -> None:
    manifest_path = _write_suite(tmp_path)

    manifest = load_monitoring_manifest(manifest_path)

    case = manifest.cases[0]
    assert case.input_path == (tmp_path / "audio" / "input.wav").resolve()
    assert case.target_path == (tmp_path / "audio" / "target.wav").resolve()
    assert case.start_sample == 0
    assert case.num_samples == 4096
    assert manifest.allow_target_full_scale is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_rate", True, "sample_rate must be an integer"),
        ("unexpected", "value", "unknown fields: unexpected"),
    ],
)
def test_manifest_rejects_invalid_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    manifest_path = _write_suite(tmp_path, **{field: value})

    with pytest.raises(MonitoringError, match=message) as error:
        load_monitoring_manifest(manifest_path)

    assert error.value.category == "manifest"


def test_suite_fingerprint_is_path_independent_and_content_sensitive(
    tmp_path: Path,
) -> None:
    first_path = _write_suite(tmp_path / "first")
    second_path = _write_suite(tmp_path / "second")
    first = load_monitoring_manifest(first_path)
    second = load_monitoring_manifest(second_path)

    assert fingerprint_monitoring_suite(first) == fingerprint_monitoring_suite(second)

    second.cases[0].target_path.write_bytes(b"changed-target")

    assert fingerprint_monitoring_suite(first) != fingerprint_monitoring_suite(second)


def test_suite_fingerprint_covers_workload_settings(tmp_path: Path) -> None:
    first = load_monitoring_manifest(_write_suite(tmp_path / "first"))
    second = load_monitoring_manifest(
        _write_suite(tmp_path / "second", allow_target_full_scale=True)
    )

    assert fingerprint_monitoring_suite(first) != fingerprint_monitoring_suite(second)


def test_latency_summary_uses_linear_p95() -> None:
    summary = latency_summary([1.0, 2.0, 4.0], samples=48_000, sample_rate=48_000)

    assert summary["p50_latency_ms"] == 2.0
    assert summary["p95_latency_ms"] == pytest.approx(3.8)
    assert summary["real_time_factor"] == pytest.approx(0.002)


def test_report_loader_preserves_the_established_version_one_shape(
    tmp_path: Path,
) -> None:
    data = {
        "schema_version": "1.0",
        "created_at": "2026-08-26T10:00:00+00:00",
        "suite": {"id": "fixed-suite", "fingerprint": "suite-hash"},
        "artifact": {"type": "checkpoint", "sha256": "artifact-hash"},
        "runtime": {"device": "cpu"},
        "workload": {"sample_rate": 48_000},
        "validation": [
            {
                "case_id": "case-a",
                "name": "files_exist",
                "passed": True,
                "severity": "error",
                "message": "Input and target files exist",
                "value": None,
            }
        ],
        "cases": [
            {
                "case_id": "case-a",
                "input_sha256": "input-hash",
                "target_sha256": "target-hash",
                "evaluated_samples": 4096,
                "metric_samples": 4032,
                "metrics": {"mse": 0.1},
                "latency": {"full": {"p50_latency_ms": 1.0}},
            }
        ],
        "aggregate": {"metrics": {"mse": 0.1}},
        "ignored_additive_field": "old loader compatibility",
    }
    report_path = tmp_path / "monitoring.json"
    report_path.write_text(json.dumps(data))

    report = load_monitoring_report(report_path)

    assert report.schema_version == "1.0"
    assert report.cases[0].metrics == {"mse": 0.1}
    assert "ignored_additive_field" not in report.to_dict()


def test_monitoring_error_carries_category_and_diagnostics() -> None:
    error = MonitoringError("Artifact is invalid", category="artifact")

    assert str(error) == "Artifact is invalid"
    assert error.category == "artifact"
    assert error.diagnostics == ()


def test_checkpoint_monitoring_produces_required_measurements(tmp_path: Path) -> None:
    manifest_path = _write_suite(tmp_path)
    _write_audio(tmp_path)
    config_path, checkpoint_path, _ = _write_model_artifacts(tmp_path)

    report = monitor_artifact(
        manifest_path,
        checkpoint_path,
        config_path=config_path,
    )

    assert report.artifact["type"] == "checkpoint"
    assert report.artifact["trainable_parameters"] > 0
    assert set(report.cases[0].metrics) == {
        "esr",
        "mse",
        "multi_resolution_stft_distance",
    }
    assert report.cases[0].latency["full"]["runs"] == 3
    assert report.aggregate["metrics"]["artifact_size_bytes"] > 0
    assert report.suite["validation_passed"] is True


def test_monitoring_marks_process_memory_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _write_suite(
        tmp_path,
        warmup_runs=0,
        measurement_runs=1,
        quality_metrics=["mse"],
    )
    _write_audio(tmp_path)
    config_path, checkpoint_path, _ = _write_model_artifacts(tmp_path)
    monkeypatch.setattr("neural_fx.monitoring.pipeline.peak_rss_bytes", lambda: None)

    report = monitor_artifact(
        manifest_path,
        checkpoint_path,
        config_path=config_path,
    )

    assert report.aggregate["metrics"]["peak_memory_bytes"] is None
    assert report.aggregate["memory"]["kind"] == "unavailable"


def test_torchscript_uses_the_same_monitoring_interface(tmp_path: Path) -> None:
    manifest_path = _write_suite(tmp_path, quality_metrics=["mse"])
    _write_audio(tmp_path)
    config_path, _, torchscript_path = _write_model_artifacts(
        tmp_path, include_torchscript=True
    )

    report = monitor_artifact(
        manifest_path,
        torchscript_path,
        artifact_type="torchscript",
        config_path=config_path,
    )

    assert report.artifact["type"] == "torchscript"
    assert report.artifact["inference_category"] == "stateless_sequence"
    assert report.cases[0].metrics["mse"] >= 0


def test_monitoring_rejects_native_sample_rate_mismatch(tmp_path: Path) -> None:
    manifest_path = _write_suite(tmp_path)
    _write_audio(tmp_path, sample_rate=44_100)
    config_path, checkpoint_path, _ = _write_model_artifacts(tmp_path)

    with pytest.raises(MonitoringError, match="Expected 48000 Hz") as error:
        monitor_artifact(
            manifest_path,
            checkpoint_path,
            config_path=config_path,
        )

    assert error.value.category == "validation"


def test_target_full_scale_override_is_recorded(tmp_path: Path) -> None:
    manifest_path = _write_suite(tmp_path, allow_target_full_scale=True)
    _, target_path = _write_audio(tmp_path)
    target, sample_rate = torchaudio.load(target_path)
    target[0, 0] = -1.0
    torchaudio.save(target_path, target, sample_rate)
    config_path, checkpoint_path, _ = _write_model_artifacts(tmp_path)

    report = monitor_artifact(
        manifest_path,
        checkpoint_path,
        config_path=config_path,
    )

    check = next(item for item in report.validation if item.name == "target_full_scale")
    assert check.passed
    assert check.value == 1
    assert report.workload["allow_target_full_scale"] is True


def test_target_full_scale_fails_without_override(tmp_path: Path) -> None:
    manifest_path = _write_suite(tmp_path)
    _, target_path = _write_audio(tmp_path)
    target, sample_rate = torchaudio.load(target_path)
    target[0, 0] = -1.0
    torchaudio.save(target_path, target, sample_rate)
    config_path, checkpoint_path, _ = _write_model_artifacts(tmp_path)

    with pytest.raises(MonitoringError, match="target_full_scale") as error:
        monitor_artifact(
            manifest_path,
            checkpoint_path,
            config_path=config_path,
        )

    assert error.value.category == "validation"


def _run_monitor_command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "monitor.py"), *arguments],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )


def test_monitor_command_writes_required_outputs(tmp_path: Path) -> None:
    manifest_path = _write_suite(
        tmp_path,
        warmup_runs=0,
        measurement_runs=1,
        quality_metrics=["mse"],
    )
    _write_audio(tmp_path)
    config_path, checkpoint_path, _ = _write_model_artifacts(tmp_path)
    output_dir = tmp_path / "result"

    result = _run_monitor_command(
        [
            "--manifest",
            str(manifest_path),
            "--artifact",
            str(checkpoint_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--html",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert json.loads((output_dir / "monitoring.json").read_text())[
        "schema_version"
    ] == "1.0"
    assert (output_dir / "monitoring.csv").is_file()
    assert (output_dir / "monitoring.html").is_file()


def test_monitor_command_returns_two_for_expected_failure(tmp_path: Path) -> None:
    manifest_path = _write_suite(tmp_path)
    _write_audio(tmp_path, sample_rate=44_100)
    config_path, checkpoint_path, _ = _write_model_artifacts(tmp_path)
    output_dir = tmp_path / "result"

    result = _run_monitor_command(
        [
            "--manifest",
            str(manifest_path),
            "--artifact",
            str(checkpoint_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result.returncode == 2
    assert "Expected 48000 Hz" in result.stderr
    assert not output_dir.exists()
