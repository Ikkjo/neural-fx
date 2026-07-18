"""Reusable experiment comparison schemas, aggregation, and reports."""

from .manifest import (
    SCHEMA_VERSION,
    AudioPair,
    DatasetSplits,
    ExperimentManifest,
    ManifestValidationError,
    ModelRun,
    ParameterBudget,
    load_manifest,
)
from .reporting import write_reports
from .results import (
    AggregateResult,
    MetricSummary,
    RunResult,
    aggregate_results,
    load_run_results,
)

__all__ = [
    "SCHEMA_VERSION",
    "AggregateResult",
    "AudioPair",
    "DatasetSplits",
    "ExperimentManifest",
    "ManifestValidationError",
    "MetricSummary",
    "ModelRun",
    "ParameterBudget",
    "RunResult",
    "aggregate_results",
    "load_manifest",
    "load_run_results",
    "write_reports",
]
