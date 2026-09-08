"""Compare two compatible offline monitoring reports against a fixed policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import MonitoringError, MonitoringReport, load_monitoring_report

POLICY_SCHEMA_VERSION = "1.0"
REJECT_METRICS = {"esr", "multi_resolution_stft_distance"}
INVESTIGATE_METRICS = {
    "mse",
    "p95_latency_ms",
    "real_time_factor",
    "peak_memory_bytes",
    "artifact_size_bytes",
}
METRICS = (
    "esr",
    "mse",
    "multi_resolution_stft_distance",
    "p95_latency_ms",
    "real_time_factor",
    "peak_memory_bytes",
    "artifact_size_bytes",
)


@dataclass(frozen=True)
class ComparisonPolicy:
    policy_id: str
    reject_thresholds: dict[str, float]
    investigate_thresholds: dict[str, float]
    response: str
    schema_version: str = POLICY_SCHEMA_VERSION


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a mapping")
    return value


def _fields(data: dict[str, Any], name: str, expected: set[str]) -> None:
    missing = expected - set(data)
    unknown = set(data) - expected
    if missing:
        raise ValueError(f"{name} is missing: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")


def _thresholds(value: Any, name: str, metrics: set[str]) -> dict[str, float]:
    data = _mapping(value, name)
    _fields(data, name, metrics)
    result: dict[str, float] = {}
    for metric, threshold in data.items():
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"{name}.{metric} must be a number")
        threshold = float(threshold)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError(f"{name}.{metric} must be finite and non-negative")
        result[metric] = threshold
    return result


def load_comparison_policy(path: str | Path) -> ComparisonPolicy:
    """Load one strict version-1 artifact comparison policy."""
    try:
        data = _mapping(yaml.safe_load(Path(path).read_text()), "Comparison policy")
        _fields(
            data,
            "Comparison policy",
            {
                "schema_version",
                "policy_id",
                "reject_thresholds",
                "investigate_thresholds",
                "response",
            },
        )
        if data["schema_version"] != POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported comparison policy schema: {data['schema_version']}"
            )
        if not isinstance(data["policy_id"], str) or not data["policy_id"].strip():
            raise ValueError("policy_id must be a non-empty string")
        if not isinstance(data["response"], str) or not data["response"].strip():
            raise ValueError("response must be a non-empty string")
        return ComparisonPolicy(
            policy_id=data["policy_id"],
            reject_thresholds=_thresholds(
                data["reject_thresholds"], "reject_thresholds", REJECT_METRICS
            ),
            investigate_thresholds=_thresholds(
                data["investigate_thresholds"],
                "investigate_thresholds",
                INVESTIGATE_METRICS,
            ),
            response=data["response"],
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise MonitoringError(str(exc), category="manifest") from exc


def _same(left: Any, right: Any, name: str) -> None:
    if left != right:
        raise MonitoringError(
            f"Monitoring reports differ in {name}", category="validation"
        )


def _validate_report_pair(
    baseline: MonitoringReport, candidate: MonitoringReport
) -> None:
    _same(baseline.schema_version, candidate.schema_version, "schema_version")
    for key in ("id", "fingerprint"):
        _same(baseline.suite.get(key), candidate.suite.get(key), f"suite.{key}")
    if not baseline.suite.get("validation_passed") or not candidate.suite.get(
        "validation_passed"
    ):
        raise MonitoringError(
            "Monitoring report validation did not pass", category="validation"
        )
    for key in (
        "silence_policy",
        "esr_mode",
        "esr_pre_emphasis",
        "sample_rate",
        "burn_in_samples",
        "segment_length",
        "quality_metrics",
        "case_ids",
        "latency_block_sizes",
        "warmup_runs",
        "measurement_runs",
    ):
        _same(
            baseline.workload.get(key), candidate.workload.get(key), f"workload.{key}"
        )
    for key in ("device_class", "device_name", "dtype"):
        _same(baseline.runtime.get(key), candidate.runtime.get(key), f"runtime.{key}")
    for key in (
        "type",
        "inference_category",
        "effective_inference_chunk_size",
        "model_type",
    ):
        _same(
            baseline.artifact.get(key), candidate.artifact.get(key), f"artifact.{key}"
        )


def _value(report: MonitoringReport, metric: str) -> float | None:
    value = report.aggregate["metrics"].get(metric)
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise MonitoringError(
            f"Monitoring report has invalid {metric}", category="validation"
        )
    return float(value)


def _contract_failed(report: MonitoringReport) -> bool:
    return any(
        not check.passed
        and check.name in {"prediction_amplitude", "prediction_clipping"}
        for check in report.validation
    )


def build_monitoring_comparison(
    baseline: MonitoringReport,
    candidate: MonitoringReport,
    policy: ComparisonPolicy,
    *,
    scenario: str,
) -> tuple[dict[str, Any], str]:
    """Return a policy decision and concise Markdown for a compatible pair."""
    if not scenario.strip():
        raise ValueError("scenario must be a non-empty string")
    _validate_report_pair(baseline, candidate)
    metrics: dict[str, dict[str, float | None]] = {}
    for metric in METRICS:
        baseline_value = _value(baseline, metric)
        candidate_value = _value(candidate, metric)
        relative = (
            None
            if baseline_value is None or candidate_value is None or baseline_value == 0
            else (candidate_value / baseline_value - 1.0) * 100.0
        )
        metrics[metric] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "absolute_delta": (
                None
                if baseline_value is None or candidate_value is None
                else candidate_value - baseline_value
            ),
            "relative_percent": relative,
        }
    checks: list[dict[str, Any]] = []
    for severity, thresholds in (
        ("reject", policy.reject_thresholds),
        ("investigate", policy.investigate_thresholds),
    ):
        for metric, threshold in thresholds.items():
            relative = metrics[metric]["relative_percent"]
            baseline_value = metrics[metric]["baseline"]
            candidate_value = metrics[metric]["candidate"]
            crossed = (
                relative is not None
                and candidate_value is not None
                and baseline_value is not None
                and candidate_value > baseline_value * (1.0 + threshold)
            )
            checks.append(
                {
                    "metric": metric,
                    "severity": severity,
                    "threshold_percent": threshold * 100.0,
                    "observed_relative_percent": relative,
                    "crossed": crossed,
                    "reason": (
                        "unavailable relative delta"
                        if relative is None
                        else f"{relative:.3f}% {'exceeds' if crossed else 'does not exceed'} {threshold * 100.0:.3f}%"
                    ),
                }
            )
    if _contract_failed(baseline) or _contract_failed(candidate):
        checks.append(
            {
                "metric": "output_contract",
                "severity": "reject",
                "threshold_percent": None,
                "observed_relative_percent": None,
                "crossed": True,
                "reason": "a report recorded an output amplitude or clipping contract failure",
            }
        )
    decision = (
        "reject"
        if any(check["crossed"] and check["severity"] == "reject" for check in checks)
        else ("investigate" if any(check["crossed"] for check in checks) else "accept")
    )
    report = {
        "schema_version": "1.0",
        "scenario": scenario,
        "policy": {"id": policy.policy_id, "schema_version": policy.schema_version},
        "suite": {
            "id": baseline.suite["id"],
            "fingerprint": baseline.suite["fingerprint"],
        },
        "baseline": {
            key: baseline.artifact.get(key)
            for key in (
                "path",
                "sha256",
                "config_path",
                "config_sha256",
                "model_name",
                "model_type",
            )
        },
        "candidate": {
            key: candidate.artifact.get(key)
            for key in (
                "path",
                "sha256",
                "config_path",
                "config_sha256",
                "model_name",
                "model_type",
            )
        },
        "comparability": {
            "passed": True,
            "device_name": baseline.runtime.get("device_name"),
            "dtype": baseline.runtime.get("dtype"),
            "inference_category": baseline.artifact.get("inference_category"),
            "effective_inference_chunk_size": baseline.artifact.get(
                "effective_inference_chunk_size"
            ),
        },
        "metrics": metrics,
        "checks": checks,
        "decision": decision,
        "response": policy.response
        if decision == "reject"
        else "No promotion action is required."
        if decision == "accept"
        else "Inspect the flagged version difference before promotion.",
    }
    return report, _markdown(report)


def _display(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6g}"


def _markdown(report: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {name} | {_display(values['baseline'])} | {_display(values['candidate'])} | {_display(values['absolute_delta'])} | {_display(values['relative_percent'])}% |"
        for name, values in report["metrics"].items()
    )
    checks = "\n".join(
        f"| {check['severity']} | {check['metric']} | {_display(check['observed_relative_percent'])}% | {_display(check['threshold_percent'])}% | {'yes' if check['crossed'] else 'no'} |"
        for check in report["checks"]
    )
    return (
        f"# TAMU artifact comparison\n\n"
        f"Scenario: `{report['scenario']}`. This is a controlled rollback failure, not a claim that the historical artifact was deployed after the baseline.\n\n"
        f"Suite `{report['suite']['id']}` (`{report['suite']['fingerprint']}`). Both reports used `{report['comparability']['inference_category']}` inference, chunk `{report['comparability']['effective_inference_chunk_size']}`, dtype `{report['comparability']['dtype']}`, on `{report['comparability']['device_name']}`.\n\n"
        f"Baseline `{report['baseline']['sha256']}`: `{report['baseline']['path']}`. Candidate `{report['candidate']['sha256']}`: `{report['candidate']['path']}`.\n\n"
        "| Metric | Baseline | Candidate | Delta | Change |\n| --- | ---: | ---: | ---: | ---: |\n"
        f"{rows}\n\n| Severity | Metric | Observed | Threshold | Crossed |\n| --- | --- | ---: | ---: | --- |\n{checks}\n\n"
        f"Decision: **{report['decision']}**. {report['response']}\n"
    )


def compare_monitoring_reports(
    baseline_path: str | Path,
    candidate_path: str | Path,
    policy_path: str | Path,
    *,
    scenario: str,
) -> tuple[dict[str, Any], str]:
    """Load and compare two report files using one policy manifest."""
    return build_monitoring_comparison(
        load_monitoring_report(baseline_path),
        load_monitoring_report(candidate_path),
        load_comparison_policy(policy_path),
        scenario=scenario,
    )
