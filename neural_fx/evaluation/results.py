"""Seed-level comparison results and deterministic aggregation."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import (
    SCHEMA_VERSION,
    ExperimentManifest,
    ManifestValidationError,
    _require_mapping,
    _require_sequence,
    _required_text,
)


@dataclass(frozen=True)
class RunResult:
    """Numeric quality/performance metrics for one declared model run."""

    run_id: str
    metrics: Mapping[str, float]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], index: int) -> RunResult:
        prefix = f"records[{index}]"
        raw_metrics = _require_mapping(data.get("metrics"), f"{prefix}.metrics")
        if not raw_metrics:
            raise ManifestValidationError(f"{prefix}.metrics must not be empty")
        metrics: dict[str, float] = {}
        for name, value in raw_metrics.items():
            metric_name = _required_text(name, f"{prefix}.metrics key")
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ManifestValidationError(
                    f"{prefix}.metrics.{metric_name} must be a finite number"
                )
            metrics[metric_name] = float(value)
        return cls(
            run_id=_required_text(data.get("run_id"), f"{prefix}.run_id"),
            metrics=metrics,
        )


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}


@dataclass(frozen=True)
class AggregateResult:
    """Metrics aggregated across seeds for one architecture and size."""

    architecture: str
    size_label: str
    parameter_count_mean: float
    seeds: tuple[int, ...]
    metrics: Mapping[str, MetricSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "size_label": self.size_label,
            "parameter_count_mean": self.parameter_count_mean,
            "seeds": list(self.seeds),
            "metrics": {
                name: summary.to_dict()
                for name, summary in sorted(self.metrics.items())
            },
        }


def load_run_results(path: str | Path) -> tuple[RunResult, ...]:
    """Load the versioned JSON interchange produced by future evaluators."""
    with Path(path).open(encoding="utf-8") as handle:
        data = _require_mapping(json.load(handle), "results")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ManifestValidationError(
            f"Unsupported results schema_version {data.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    raw_records = _require_sequence(data.get("records"), "records")
    if not raw_records:
        raise ManifestValidationError("records must not be empty")
    return tuple(
        RunResult.from_dict(_require_mapping(item, f"records[{index}]"), index)
        for index, item in enumerate(raw_records)
    )


def aggregate_results(
    manifest: ExperimentManifest, records: Sequence[RunResult]
) -> tuple[AggregateResult, ...]:
    """Validate run results and compute sample mean/std across declared seeds.

    Standard deviation is the sample standard deviation (``n - 1``). A
    single-seed experiment reports a standard deviation of zero.
    """
    runs_by_id = {run.id: run for run in manifest.models}
    records_by_id: dict[str, RunResult] = {}
    for record in records:
        if record.run_id not in runs_by_id:
            raise ManifestValidationError(
                f"Result references unknown run id: {record.run_id}"
            )
        if record.run_id in records_by_id:
            raise ManifestValidationError(
                f"Duplicate result for run id: {record.run_id}"
            )
        records_by_id[record.run_id] = record

    missing = sorted(set(runs_by_id) - set(records_by_id))
    if missing:
        raise ManifestValidationError(f"Missing results for run ids: {missing}")

    groups: dict[tuple[str, str], list[tuple[Any, RunResult]]] = {}
    for run in manifest.models:
        groups.setdefault((run.architecture, run.size_label), []).append(
            (run, records_by_id[run.id])
        )

    output: list[AggregateResult] = []
    for (architecture, size_label), run_records in sorted(groups.items()):
        metric_names = set(run_records[0][1].metrics)
        for run, record in run_records[1:]:
            if set(record.metrics) != metric_names:
                raise ManifestValidationError(
                    f"Run {run.id!r} has inconsistent metric names; expected "
                    f"{sorted(metric_names)}, got {sorted(record.metrics)}"
                )
        summaries: dict[str, MetricSummary] = {}
        for metric_name in sorted(metric_names):
            values = [record.metrics[metric_name] for _, record in run_records]
            summaries[metric_name] = MetricSummary(
                mean=statistics.fmean(values),
                std=statistics.stdev(values) if len(values) > 1 else 0.0,
            )
        output.append(
            AggregateResult(
                architecture=architecture,
                size_label=size_label,
                parameter_count_mean=statistics.fmean(
                    run.parameter_count for run, _ in run_records
                ),
                seeds=tuple(sorted(run.seed for run, _ in run_records)),
                metrics=summaries,
            )
        )
    return tuple(output)
