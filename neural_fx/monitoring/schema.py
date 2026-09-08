"""Manifest, fingerprint, error, and report contracts for offline monitoring."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

MONITORING_MANIFEST_SCHEMA_VERSION = "1.0"
MONITORING_REPORT_SCHEMA_VERSION = "1.0"
SUPPORTED_QUALITY_METRICS = (
    "esr",
    "mse",
    "multi_resolution_stft_distance",
)

FailureCategory = Literal[
    "manifest",
    "validation",
    "artifact",
    "execution",
    "output",
]


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a mapping")
    return value


def _check_fields(
    data: dict[str, Any],
    *,
    name: str,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = required - set(data)
    if missing:
        raise ValueError(f"{name} is missing: {', '.join(sorted(missing))}")
    unknown = set(data) - required - optional
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")


def _strict_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _strict_float(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be true or false")
    return value


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _resolve_path(value: Any, manifest_path: Path, name: str) -> Path:
    path = Path(_non_empty_string(value, name)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


@dataclass(frozen=True)
class ValidationCheck:
    """One preflight validation result."""

    case_id: str
    name: str
    passed: bool
    severity: str
    message: str
    value: float | int | str | None = None


class MonitoringError(RuntimeError):
    """An expected monitoring failure with a stable machine-readable category."""

    def __init__(
        self,
        message: str,
        *,
        category: FailureCategory,
        diagnostics: tuple[ValidationCheck, ...] = (),
    ) -> None:
        super().__init__(message)
        self.category = category
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class MonitoringCase:
    """One fixed input and target slice in a monitoring suite."""

    case_id: str
    input_path: Path
    target_path: Path
    start_sample: int
    num_samples: int

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        manifest_path: Path,
        segment_length: int,
    ) -> MonitoringCase:
        _check_fields(
            data,
            name="Monitoring case",
            required={"id", "input", "target"},
            optional={"start_sample", "num_samples"},
        )
        case_id = _non_empty_string(data["id"], "Monitoring case id")
        start_sample = _strict_int(
            data.get("start_sample", 0), f"Case '{case_id}' start_sample"
        )
        num_samples = _strict_int(
            data.get("num_samples", segment_length),
            f"Case '{case_id}' num_samples",
        )
        if start_sample < 0:
            raise ValueError(f"Case '{case_id}' start_sample must be non-negative")
        if num_samples != segment_length:
            raise ValueError(
                f"Case '{case_id}' num_samples must match segment_length"
            )
        return cls(
            case_id=case_id,
            input_path=_resolve_path(
                data["input"], manifest_path, f"Case '{case_id}' input"
            ),
            target_path=_resolve_path(
                data["target"], manifest_path, f"Case '{case_id}' target"
            ),
            start_sample=start_sample,
            num_samples=num_samples,
        )


@dataclass(frozen=True)
class MonitoringManifest:
    """A parsed version 1 fixed-suite monitoring manifest."""

    suite_id: str
    sample_rate: int
    channels: int
    segment_length: int
    burn_in_samples: int
    inference_chunk_size: int
    latency_block_sizes: tuple[int, ...]
    warmup_runs: int
    measurement_runs: int
    quality_metrics: tuple[str, ...]
    esr_pre_emphasis: float | None
    clipping_threshold: float
    max_abs: float
    allow_target_full_scale: bool
    cases: tuple[MonitoringCase, ...]
    manifest_path: Path
    schema_version: str = MONITORING_MANIFEST_SCHEMA_VERSION

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], manifest_path: str | Path
    ) -> MonitoringManifest:
        _check_fields(
            data,
            name="Monitoring manifest",
            required={
                "schema_version",
                "suite_id",
                "sample_rate",
                "channels",
                "segment_length",
                "burn_in_samples",
                "inference_chunk_size",
                "latency_block_sizes",
                "warmup_runs",
                "measurement_runs",
                "quality_metrics",
                "esr_pre_emphasis",
                "clipping_threshold",
                "max_abs",
                "cases",
            },
            optional={"allow_target_full_scale"},
        )
        manifest_path = Path(manifest_path).expanduser().resolve()
        schema_version = data["schema_version"]
        if schema_version != MONITORING_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported monitoring manifest schema: {schema_version}"
            )

        suite_id = _non_empty_string(data["suite_id"], "suite_id")
        sample_rate = _strict_int(data["sample_rate"], "sample_rate")
        channels = _strict_int(data["channels"], "channels")
        segment_length = _strict_int(data["segment_length"], "segment_length")
        burn_in_samples = _strict_int(data["burn_in_samples"], "burn_in_samples")
        inference_chunk_size = _strict_int(
            data["inference_chunk_size"], "inference_chunk_size"
        )
        warmup_runs = _strict_int(data["warmup_runs"], "warmup_runs")
        measurement_runs = _strict_int(
            data["measurement_runs"], "measurement_runs"
        )
        clipping_threshold = _strict_float(
            data["clipping_threshold"], "clipping_threshold"
        )
        max_abs = _strict_float(data["max_abs"], "max_abs")
        allow_target_full_scale = _strict_bool(
            data.get("allow_target_full_scale", False),
            "allow_target_full_scale",
        )

        if sample_rate <= 0 or channels <= 0:
            raise ValueError("sample_rate and channels must be positive")
        if segment_length < 2048:
            raise ValueError("segment_length must be at least 2048 samples")
        if burn_in_samples < 0 or burn_in_samples > segment_length - 2048:
            raise ValueError("burn_in_samples must leave at least 2048 metric samples")
        if inference_chunk_size <= 0:
            raise ValueError("inference_chunk_size must be positive")
        if warmup_runs < 0 or measurement_runs <= 0:
            raise ValueError(
                "warmup_runs must be non-negative and measurement_runs must be positive"
            )
        if not 0.0 < clipping_threshold <= max_abs or max_abs <= 0.0:
            raise ValueError("Require 0 < clipping_threshold <= max_abs")

        latency_data = data["latency_block_sizes"]
        if not isinstance(latency_data, list):
            raise TypeError("latency_block_sizes must contain a list")
        latency_block_sizes = tuple(
            _strict_int(value, "latency block size") for value in latency_data
        )
        if not latency_block_sizes or any(value <= 0 for value in latency_block_sizes):
            raise ValueError("latency_block_sizes must contain positive values")
        if any(value > segment_length for value in latency_block_sizes):
            raise ValueError("latency block sizes cannot exceed segment_length")

        metrics_data = data["quality_metrics"]
        if not isinstance(metrics_data, list):
            raise TypeError("quality_metrics must contain a list")
        quality_metrics = tuple(
            _non_empty_string(value, "quality metric") for value in metrics_data
        )
        unknown_metrics = sorted(set(quality_metrics) - set(SUPPORTED_QUALITY_METRICS))
        if unknown_metrics:
            raise ValueError(
                f"Unsupported quality metrics: {', '.join(unknown_metrics)}"
            )
        if not quality_metrics or len(set(quality_metrics)) != len(quality_metrics):
            raise ValueError("quality_metrics must be non-empty and unique")

        esr_value = data["esr_pre_emphasis"]
        esr_pre_emphasis = (
            None
            if esr_value is None
            else _strict_float(esr_value, "esr_pre_emphasis")
        )
        if esr_pre_emphasis is not None and not 0.0 <= esr_pre_emphasis < 1.0:
            raise ValueError("esr_pre_emphasis must be null or in [0, 1)")

        cases_data = data["cases"]
        if not isinstance(cases_data, list) or not cases_data:
            raise ValueError("Monitoring manifest requires at least one case")
        cases = tuple(
            MonitoringCase.from_dict(
                _require_mapping(case, "Monitoring case"),
                manifest_path,
                segment_length,
            )
            for case in cases_data
        )
        case_ids = [case.case_id for case in cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Monitoring case ids must be unique")

        return cls(
            suite_id=suite_id,
            sample_rate=sample_rate,
            channels=channels,
            segment_length=segment_length,
            burn_in_samples=burn_in_samples,
            inference_chunk_size=inference_chunk_size,
            latency_block_sizes=latency_block_sizes,
            warmup_runs=warmup_runs,
            measurement_runs=measurement_runs,
            quality_metrics=quality_metrics,
            esr_pre_emphasis=esr_pre_emphasis,
            clipping_threshold=clipping_threshold,
            max_abs=max_abs,
            allow_target_full_scale=allow_target_full_scale,
            cases=cases,
            manifest_path=manifest_path,
            schema_version=schema_version,
        )

    def settings_dict(self) -> dict[str, Any]:
        """Return normalized path-independent settings for suite identity."""
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "segment_length": self.segment_length,
            "burn_in_samples": self.burn_in_samples,
            "inference_chunk_size": self.inference_chunk_size,
            "latency_block_sizes": list(self.latency_block_sizes),
            "warmup_runs": self.warmup_runs,
            "measurement_runs": self.measurement_runs,
            "quality_metrics": list(self.quality_metrics),
            "esr_pre_emphasis": self.esr_pre_emphasis,
            "clipping_threshold": self.clipping_threshold,
            "max_abs": self.max_abs,
            "allow_target_full_scale": self.allow_target_full_scale,
        }


def load_monitoring_manifest(path: str | Path) -> MonitoringManifest:
    """Load a strict YAML manifest and resolve its paths."""
    resolved_path = Path(path).expanduser().resolve()
    try:
        data = yaml.safe_load(resolved_path.read_text())
        return MonitoringManifest.from_dict(
            _require_mapping(data, "Monitoring manifest"), resolved_path
        )
    except MonitoringError:
        raise
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise MonitoringError(str(exc), category="manifest") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MonitoringError(
            f"Cannot fingerprint file '{path}': {exc}", category="validation"
        ) from exc
    return digest.hexdigest()


def monitoring_case_hashes(manifest: MonitoringManifest) -> list[dict[str, Any]]:
    """Hash ordered case files and return their path-independent identities."""
    return [
        {
            "id": case.case_id,
            "input_sha256": sha256_file(case.input_path),
            "target_sha256": sha256_file(case.target_path),
            "start_sample": case.start_sample,
            "num_samples": case.num_samples,
        }
        for case in manifest.cases
    ]


def fingerprint_monitoring_suite(
    manifest: MonitoringManifest,
    case_hashes: list[dict[str, Any]] | None = None,
) -> str:
    """Return the SHA-256 identity of one complete ordered monitoring workload."""
    payload = {
        "settings": manifest.settings_dict(),
        "cases": case_hashes if case_hashes is not None else monitoring_case_hashes(manifest),
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode()).hexdigest()


@dataclass(frozen=True)
class MonitoringCaseResult:
    """Quality and operational measurements for one suite case."""

    case_id: str
    input_sha256: str
    target_sha256: str
    evaluated_samples: int
    metric_samples: int
    metrics: dict[str, float]
    latency: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class MonitoringReport:
    """Version 1 result for one artifact and fixed suite."""

    created_at: str
    suite: dict[str, Any]
    artifact: dict[str, Any]
    runtime: dict[str, Any]
    workload: dict[str, Any]
    validation: tuple[ValidationCheck, ...]
    cases: tuple[MonitoringCaseResult, ...]
    aggregate: dict[str, Any]
    schema_version: str = MONITORING_REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to JSON-compatible data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MonitoringReport:
        """Load the established permissive version 1 report shape."""
        if data.get("schema_version") != MONITORING_REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported monitoring report schema: {data.get('schema_version')}"
            )
        for key in (
            "created_at",
            "suite",
            "artifact",
            "runtime",
            "workload",
            "validation",
            "cases",
            "aggregate",
        ):
            if key not in data:
                raise ValueError(f"Monitoring report is missing '{key}'")
        validations = tuple(ValidationCheck(**item) for item in data["validation"])
        cases = tuple(MonitoringCaseResult(**item) for item in data["cases"])
        if not cases:
            raise ValueError("Monitoring report must contain at least one case")
        metric_values = list(data["aggregate"].get("metrics", {}).values())
        metric_values.extend(value for case in cases for value in case.metrics.values())
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in metric_values
        ):
            raise ValueError("Monitoring report metrics must be finite numbers")
        return cls(
            schema_version=data["schema_version"],
            created_at=str(data["created_at"]),
            suite=dict(data["suite"]),
            artifact=dict(data["artifact"]),
            runtime=dict(data["runtime"]),
            workload=dict(data["workload"]),
            validation=validations,
            cases=cases,
            aggregate=dict(data["aggregate"]),
        )


def load_monitoring_report(path: str | Path) -> MonitoringReport:
    """Load one established version 1 monitoring JSON report."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise TypeError("Monitoring report must contain a JSON object")
    return MonitoringReport.from_dict(data)
