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
from neural_fx.metrics import silence_policy_metadata
from neural_fx.models.recurrent import NeuralfxGRU
from neural_fx.monitoring import (
    MonitoringError,
    load_comparison_policy,
    load_monitoring_report,
    monitor_artifact,
    write_monitoring_outputs,
)
from neural_fx.monitoring.comparison import build_monitoring_comparison
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
    root: Path,
    *,
    include_torchscript: bool = False,
    output_bias: float | None = None,
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
                "data": {"train": {"input": "unused.wav", "target": "unused.wav"}},
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
    if output_bias is not None:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.fc_out.bias.fill_(output_bias)
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
    assert manifest.esr_mode == "legacy"


def test_monitoring_records_and_applies_the_selected_esr_mode(tmp_path: Path) -> None:
    legacy_path = _write_suite(
        tmp_path / "legacy",
        quality_metrics=["esr"],
        esr_pre_emphasis=0.85,
        warmup_runs=0,
        measurement_runs=1,
    )
    nam_path = _write_suite(
        tmp_path / "nam",
        quality_metrics=["esr"],
        esr_pre_emphasis=0.85,
        esr_mode="nam",
        warmup_runs=0,
        measurement_runs=1,
    )
    _write_audio(tmp_path / "legacy")
    _write_audio(tmp_path / "nam")
    config_path, checkpoint_path, _ = _write_model_artifacts(tmp_path / "legacy")

    legacy = monitor_artifact(legacy_path, checkpoint_path, config_path=config_path)
    nam = monitor_artifact(nam_path, checkpoint_path, config_path=config_path)

    assert legacy.workload["esr_mode"] == "legacy"
    assert nam.workload["esr_mode"] == "nam"
    assert legacy.suite["fingerprint"] != nam.suite["fingerprint"]
    assert legacy.cases[0].metrics["esr"] != nam.cases[0].metrics["esr"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_rate", True, "sample_rate must be an integer"),
        ("esr_mode", "other", "esr_mode must be 'legacy' or 'nam'"),
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
        "schema_version": "1.1",
        "created_at": "2026-08-26T10:00:00+00:00",
        "suite": {"id": "fixed-suite", "fingerprint": "suite-hash"},
        "artifact": {"type": "checkpoint", "sha256": "artifact-hash"},
        "runtime": {"device": "cpu"},
        "workload": {
            "sample_rate": 48_000,
            "silence_policy": silence_policy_metadata(),
        },
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
                "diagnostics": {
                    "digital_silence": False,
                    "relative_score_status": "eligible",
                    "mse": 0.1,
                    "prediction_rms": 0.1,
                    "prediction_abs_peak": 0.2,
                },
                "latency": {"full": {"p50_latency_ms": 1.0}},
            }
        ],
        "aggregate": {
            "metrics": {"mse": 0.1},
            "quality": {"mse": 0.1},
        },
        "ignored_additive_field": "old loader compatibility",
    }
    report_path = tmp_path / "monitoring.json"
    report_path.write_text(json.dumps(data))

    report = load_monitoring_report(report_path)

    assert report.schema_version == "1.1"
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
    assert report.artifact["effective_inference_chunk_size"] == 1024
    assert report.artifact["trainable_parameters"] > 0
    assert set(report.cases[0].metrics) == {
        "esr",
        "mse",
        "multi_resolution_stft_distance",
    }
    assert report.cases[0].latency["full"]["runs"] == 3
    assert report.aggregate["metrics"]["artifact_size_bytes"] > 0
    assert report.suite["validation_passed"] is True


def test_silent_monitoring_keeps_absolute_diagnostics_and_nulls_relative_scores(
    tmp_path: Path,
) -> None:
    manifest_path = _write_suite(
        tmp_path,
        warmup_runs=0,
        measurement_runs=1,
    )
    _write_audio(tmp_path)
    _, target_path = _write_audio(tmp_path)
    torchaudio.save(target_path, torch.zeros(1, 4096), 48_000)
    config_path, checkpoint_path, _ = _write_model_artifacts(
        tmp_path, output_bias=0.001
    )

    report = monitor_artifact(
        manifest_path,
        checkpoint_path,
        config_path=config_path,
    )
    case = report.cases[0]

    assert case.metrics["esr"] is None
    assert case.metrics["multi_resolution_stft_distance"] is None
    assert case.metrics["mse"] == pytest.approx(1e-6)
    assert case.diagnostics == {
        "digital_silence": True,
        "relative_score_status": "excluded",
        "mse": pytest.approx(1e-6),
        "prediction_rms": pytest.approx(0.001),
        "prediction_abs_peak": pytest.approx(0.001),
    }
    assert report.aggregate["metrics"]["esr"] is None
    assert report.aggregate["metrics"]["multi_resolution_stft_distance"] is None
    assert report.aggregate["relative_score_counts"] == {
        "esr": {"eligible": 0, "excluded": 1},
        "multi_resolution_stft_distance": {"eligible": 0, "excluded": 1},
    }
    assert report.aggregate["silent_case_count"] == 1

    paths = write_monitoring_outputs(report, tmp_path / "report", include_html=True)
    rows = (paths["csv"]).read_text().splitlines()
    assert any(",excluded,True," in row for row in rows)
    assert "N/A" in paths["html"].read_text()
    assert load_monitoring_report(paths["json"]).cases[0].metrics["esr"] is None


def test_quiet_nonzero_target_remains_eligible(tmp_path: Path) -> None:
    manifest_path = _write_suite(
        tmp_path,
        warmup_runs=0,
        measurement_runs=1,
    )
    _write_audio(tmp_path)
    _, target_path = _write_audio(tmp_path)
    target = torch.zeros(1, 4096)
    target[0, 2048] = 0.001
    torchaudio.save(target_path, target, 48_000)
    config_path, checkpoint_path, _ = _write_model_artifacts(tmp_path, output_bias=0.0)

    report = monitor_artifact(
        manifest_path,
        checkpoint_path,
        config_path=config_path,
    )

    assert report.cases[0].diagnostics["digital_silence"] is False
    assert report.cases[0].metrics["esr"] is not None
    assert report.cases[0].metrics["multi_resolution_stft_distance"] is not None
    assert report.aggregate["relative_score_counts"]["esr"] == {
        "eligible": 1,
        "excluded": 0,
    }


def test_mixed_monitoring_cases_average_only_eligible_relative_scores(
    tmp_path: Path,
) -> None:
    manifest_path = _write_suite(
        tmp_path,
        warmup_runs=0,
        measurement_runs=1,
        cases=[
            {
                "id": "nonzero",
                "input": "audio/input.wav",
                "target": "audio/target.wav",
                "start_sample": 0,
                "num_samples": 4096,
            },
            {
                "id": "silent",
                "input": "audio/input.wav",
                "target": "audio/target.wav",
                "start_sample": 4096,
                "num_samples": 4096,
            },
        ],
    )
    _write_audio(tmp_path)
    input_audio, _ = torchaudio.load(tmp_path / "audio" / "input.wav")
    target_audio, _ = torchaudio.load(tmp_path / "audio" / "target.wav")
    torchaudio.save(
        tmp_path / "audio" / "input.wav",
        torch.cat((input_audio, input_audio), dim=-1),
        48_000,
    )
    torchaudio.save(
        tmp_path / "audio" / "target.wav",
        torch.cat((target_audio, torch.zeros_like(target_audio)), dim=-1),
        48_000,
    )
    config_path, checkpoint_path, _ = _write_model_artifacts(tmp_path, output_bias=0.0)

    report = monitor_artifact(
        manifest_path,
        checkpoint_path,
        config_path=config_path,
    )

    nonzero, silent = report.cases
    assert nonzero.diagnostics["digital_silence"] is False
    assert silent.diagnostics["digital_silence"] is True
    assert report.aggregate["relative_score_counts"]["esr"] == {
        "eligible": 1,
        "excluded": 1,
    }
    assert report.aggregate["quality"]["esr"] == pytest.approx(nonzero.metrics["esr"])
    assert report.aggregate["quality"][
        "multi_resolution_stft_distance"
    ] == pytest.approx(nonzero.metrics["multi_resolution_stft_distance"])


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

    paths = write_monitoring_outputs(report, tmp_path / "result", include_html=True)
    assert "N/A" in paths["html"].read_text()
    assert (
        load_monitoring_report(paths["json"]).aggregate["metrics"]["peak_memory_bytes"]
        is None
    )


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
    assert report.artifact["effective_inference_chunk_size"] is None
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


def test_monitoring_warns_about_out_of_range_predictions(tmp_path: Path) -> None:
    manifest_path = _write_suite(
        tmp_path,
        warmup_runs=0,
        measurement_runs=1,
        quality_metrics=["mse"],
    )
    _write_audio(tmp_path)
    config_path, checkpoint_path, _ = _write_model_artifacts(tmp_path, output_bias=2.0)

    report = monitor_artifact(
        manifest_path,
        checkpoint_path,
        config_path=config_path,
    )

    failed_warnings = {
        check.name
        for check in report.validation
        if not check.passed and check.severity == "warning"
    }
    assert failed_warnings == {"prediction_amplitude", "prediction_clipping"}
    assert report.suite["validation_warnings"] == 2


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


def _run_compare_monitor_command(
    arguments: list[str],
) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    return subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "compare_monitoring.py"),
            *arguments,
        ],
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

    arguments = [
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
    result = _run_monitor_command(arguments)

    assert result.returncode == 0, result.stderr
    assert (
        json.loads((output_dir / "monitoring.json").read_text())["schema_version"]
        == "1.1"
    )
    assert (output_dir / "monitoring.csv").is_file()
    csv_header = (output_dir / "monitoring.csv").read_text().splitlines()[0]
    assert "inference_category" in csv_header
    assert "effective_inference_chunk_size" in csv_header
    assert (output_dir / "monitoring.html").is_file()

    blocked = _run_monitor_command(arguments)
    assert blocked.returncode == 2
    assert "already exists" in blocked.stderr

    overwritten = _run_monitor_command([*arguments, "--overwrite"])
    assert overwritten.returncode == 0, overwritten.stderr


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


def _comparison_report(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": "1.1",
        "created_at": "2026-09-06T00:00:00+00:00",
        "suite": {
            "id": "suite",
            "fingerprint": "fingerprint",
            "validation_passed": True,
        },
        "artifact": {
            "path": "artifact.ckpt",
            "sha256": "artifact-hash",
            "config_path": "config.yaml",
            "config_sha256": "config-hash",
            "model_name": "model",
            "model_type": "lstm",
            "type": "checkpoint",
            "inference_category": "native_stateful",
            "effective_inference_chunk_size": 8192,
        },
        "runtime": {"device_class": "cpu", "device_name": "CPU", "dtype": "float32"},
        "workload": {
            "silence_policy": silence_policy_metadata(),
            "esr_mode": "nam",
            "esr_pre_emphasis": None,
            "sample_rate": 44100,
            "burn_in_samples": 4096,
            "segment_length": 44100,
            "quality_metrics": ["esr", "mse", "multi_resolution_stft_distance"],
            "case_ids": ["case"],
            "latency_block_sizes": [64, 128, 256, 512],
            "warmup_runs": 3,
            "measurement_runs": 20,
        },
        "validation": [],
        "cases": [
            {
                "case_id": "case",
                "input_sha256": "input",
                "target_sha256": "target",
                "evaluated_samples": 44100,
                "metric_samples": 40004,
                "metrics": {"mse": 1.0},
                "diagnostics": {
                    "digital_silence": False,
                    "relative_score_status": "eligible",
                    "mse": 1.0,
                    "prediction_rms": 0.1,
                    "prediction_abs_peak": 0.2,
                },
                "latency": {"full": {}},
            }
        ],
        "aggregate": {
            "metrics": {
                "esr": 1.0,
                "mse": 1.0,
                "multi_resolution_stft_distance": 1.0,
                "p95_latency_ms": 1.0,
                "real_time_factor": 1.0,
                "peak_memory_bytes": 100.0,
                "artifact_size_bytes": 100.0,
            },
            "quality": {"esr": 1.0, "mse": 1.0, "multi_resolution_stft_distance": 1.0},
        },
    }
    data.update(overrides)
    return data


def _comparison_policy_path(root: Path, **overrides: object) -> Path:
    data: dict[str, object] = {
        "schema_version": "1.0",
        "policy_id": "policy",
        "reject_thresholds": {"esr": 0.1, "multi_resolution_stft_distance": 0.1},
        "investigate_thresholds": {
            "mse": 0.1,
            "p95_latency_ms": 0.2,
            "real_time_factor": 0.2,
            "peak_memory_bytes": 0.2,
            "artifact_size_bytes": 0.1,
        },
        "response": "Keep the baseline.",
    }
    data.update(overrides)
    path = root / "policy.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"unexpected": True}, "unknown fields: unexpected"),
        (
            {"reject_thresholds": {"esr": -1, "multi_resolution_stft_distance": 0.1}},
            "finite and non-negative",
        ),
        (
            {
                "investigate_thresholds": {
                    "mse": float("inf"),
                    "p95_latency_ms": 0.2,
                    "real_time_factor": 0.2,
                    "peak_memory_bytes": 0.2,
                    "artifact_size_bytes": 0.1,
                }
            },
            "finite and non-negative",
        ),
    ],
)
def test_comparison_policy_is_strict(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    with pytest.raises(MonitoringError, match=message):
        load_comparison_policy(_comparison_policy_path(tmp_path, **override))


def test_comparison_thresholds_zero_and_precedence(tmp_path: Path) -> None:
    policy = load_comparison_policy(_comparison_policy_path(tmp_path))
    baseline = load_monitoring_report(
        _write_comparison_report(tmp_path / "baseline", _comparison_report())
    )
    candidate_data = _comparison_report()
    candidate_data["aggregate"]["metrics"].update(
        {"esr": 1.1, "mse": 1.2, "artifact_size_bytes": 0.0}
    )
    candidate = load_monitoring_report(
        _write_comparison_report(tmp_path / "candidate", candidate_data)
    )

    report, _ = build_monitoring_comparison(
        baseline, candidate, policy, scenario="controlled_rollback_failure"
    )

    assert report["decision"] == "investigate"  # Exactly 10% ESR does not cross.
    assert report["metrics"]["artifact_size_bytes"]["relative_percent"] == -100.0
    candidate_data["aggregate"]["metrics"]["esr"] = 1.101
    candidate = load_monitoring_report(
        _write_comparison_report(tmp_path / "candidate-reject", candidate_data)
    )
    report, _ = build_monitoring_comparison(
        baseline, candidate, policy, scenario="controlled_rollback_failure"
    )
    assert report["decision"] == "reject"  # Reject overrides the MSE investigate check.


def test_comparison_nulls_zero_or_unavailable_baseline_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    policy = load_comparison_policy(_comparison_policy_path(tmp_path))
    baseline_data = _comparison_report()
    baseline_data["aggregate"]["metrics"]["esr"] = None
    baseline_data["aggregate"]["metrics"]["artifact_size_bytes"] = 0.0
    baseline_data["aggregate"]["quality"]["esr"] = None
    baseline = load_monitoring_report(
        _write_comparison_report(tmp_path / "baseline", baseline_data)
    )
    candidate = load_monitoring_report(
        _write_comparison_report(tmp_path / "candidate", _comparison_report())
    )
    report, _ = build_monitoring_comparison(
        baseline, candidate, policy, scenario="controlled_rollback_failure"
    )
    assert report["metrics"]["esr"]["relative_percent"] is None
    assert report["metrics"]["artifact_size_bytes"]["relative_percent"] is None

    mismatched_data = _comparison_report()
    mismatched_data["suite"]["fingerprint"] = "other"
    mismatched = load_monitoring_report(
        _write_comparison_report(tmp_path / "mismatch", mismatched_data)
    )
    with pytest.raises(MonitoringError, match="suite.fingerprint"):
        build_monitoring_comparison(
            baseline, mismatched, policy, scenario="controlled_rollback_failure"
        )


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("workload", "esr_mode", "legacy", "workload.esr_mode"),
        ("runtime", "device_name", "other CPU", "runtime.device_name"),
        ("artifact", "model_type", "gru", "artifact.model_type"),
    ],
)
def test_comparison_rejects_incomparable_runtime_and_workload(
    tmp_path: Path, section: str, field: str, value: object, message: str
) -> None:
    policy = load_comparison_policy(_comparison_policy_path(tmp_path))
    baseline = load_monitoring_report(
        _write_comparison_report(tmp_path / "baseline", _comparison_report())
    )
    candidate_data = _comparison_report()
    candidate_data[section][field] = value
    candidate = load_monitoring_report(
        _write_comparison_report(tmp_path / "candidate", candidate_data)
    )
    with pytest.raises(MonitoringError, match=message):
        build_monitoring_comparison(
            baseline, candidate, policy, scenario="controlled_rollback_failure"
        )


def test_compare_monitor_command_refuses_overwrite(tmp_path: Path) -> None:
    policy = _comparison_policy_path(tmp_path)
    baseline = _write_comparison_report(tmp_path / "baseline", _comparison_report())
    candidate = _write_comparison_report(tmp_path / "candidate", _comparison_report())
    output_dir = tmp_path / "output"
    arguments = [
        "--baseline-report",
        str(baseline),
        "--candidate-report",
        str(candidate),
        "--policy",
        str(policy),
        "--scenario",
        "controlled_rollback_failure",
        "--output-dir",
        str(output_dir),
    ]
    first = _run_compare_monitor_command(arguments)
    assert first.returncode == 0, first.stderr
    assert (
        json.loads((output_dir / "comparison.json").read_text())["decision"] == "accept"
    )
    blocked = _run_compare_monitor_command(arguments)
    assert blocked.returncode == 2
    assert "already exists" in blocked.stderr


def _write_comparison_report(root: Path, data: dict[str, object]) -> Path:
    root.mkdir()
    path = root / "monitoring.json"
    path.write_text(json.dumps(data))
    return path
