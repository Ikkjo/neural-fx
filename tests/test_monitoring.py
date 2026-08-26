"""Contract tests for fixed-suite offline monitoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from neural_fx.monitoring import MonitoringError, load_monitoring_report
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
