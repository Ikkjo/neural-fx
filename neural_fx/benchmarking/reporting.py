"""Writers for machine-readable and human-readable benchmark artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .schema import SCHEMA_VERSION, BenchmarkResult


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(results: list[BenchmarkResult], path: Path | str) -> Path:
    """Write benchmark results as a versioned JSON document."""
    destination = Path(path)
    _ensure_parent(destination)
    schema_versions = {result.schema_version for result in results}
    if len(schema_versions) > 1:
        raise ValueError("All results in one artifact must use the same schema version")
    payload = {
        "schema_version": next(iter(schema_versions), SCHEMA_VERSION),
        "results": [result.to_dict() for result in results],
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def write_csv(results: list[BenchmarkResult], path: Path | str) -> Path:
    """Write one flat CSV row per model/block-size measurement."""
    destination = Path(path)
    _ensure_parent(destination)
    fieldnames = [
        "schema_version",
        "timestamp_utc",
        "git_sha",
        "device",
        "device_name",
        "torch_num_threads",
        "seed",
        "model_name",
        "model_type",
        "sample_rate",
        "parameter_count",
        "parameter_bytes",
        "config_path",
        "checkpoint_path",
        "mode",
        "block_size",
        "warmup_iterations",
        "iterations",
        "latency_median_ms",
        "latency_p95_ms",
        "real_time_factor",
        "throughput_samples_per_second",
        "state_bytes",
        "peak_cpu_rss_bytes",
        "peak_cuda_allocated_bytes",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for measurement in result.measurements:
                writer.writerow(
                    {
                        "schema_version": result.schema_version,
                        "timestamp_utc": result.environment.timestamp_utc,
                        "git_sha": result.environment.git_sha,
                        "device": result.environment.device,
                        "device_name": result.environment.device_name,
                        "torch_num_threads": result.environment.torch_num_threads,
                        "seed": result.environment.seed,
                        "model_name": result.model.name,
                        "model_type": result.model.type,
                        "sample_rate": result.model.sample_rate,
                        "parameter_count": result.model.parameter_count,
                        "parameter_bytes": result.model.parameter_bytes,
                        "config_path": result.model.config_path,
                        "checkpoint_path": result.model.checkpoint_path,
                        **measurement.__dict__,
                    }
                )
    return destination


def write_markdown(results: list[BenchmarkResult], path: Path | str) -> Path:
    """Write a compact latency comparison table."""
    destination = Path(path)
    _ensure_parent(destination)
    lines = [
        "# Neural-FX benchmark results",
        "",
        "Lower latency and real-time factor (RTF) are better. RTF below 1.0 "
        "indicates faster-than-real-time processing.",
        "",
        "| Model | Device | Block | Median (ms) | p95 (ms) | RTF | Params |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for measurement in result.measurements:
            lines.append(
                f"| {result.model.name} | {result.environment.device_name} "
                f"| {measurement.block_size} "
                f"| {measurement.latency_median_ms:.4f} "
                f"| {measurement.latency_p95_ms:.4f} "
                f"| {measurement.real_time_factor:.4f} "
                f"| {result.model.parameter_count:,} |"
            )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
