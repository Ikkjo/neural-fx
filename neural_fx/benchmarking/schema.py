"""Versioned data structures for benchmark results."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class BenchmarkEnvironment:
    """Execution environment needed to interpret or reproduce a benchmark."""

    timestamp_utc: str
    git_sha: str | None
    os: str
    machine: str
    processor: str
    python_version: str
    torch_version: str
    device: str
    device_name: str
    torch_num_threads: int
    seed: int


@dataclass(frozen=True)
class BenchmarkModel:
    """Identity and static size information for one model."""

    name: str
    type: str
    sample_rate: int
    input_size: int
    output_size: int
    parameter_count: int
    parameter_bytes: int
    config_path: str | None = None
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class BenchmarkMeasurement:
    """Timing statistics for one inference mode and block size.

    Memory fields are optional in schema 1.0. Task 5B can populate them without
    changing existing consumers of benchmark artifacts.
    """

    mode: str
    block_size: int
    warmup_iterations: int
    iterations: int
    latency_median_ms: float
    latency_p95_ms: float
    real_time_factor: float
    throughput_samples_per_second: float
    state_bytes: int | None = None
    peak_cpu_rss_bytes: int | None = None
    peak_cuda_allocated_bytes: int | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    """Complete, serializable benchmark result for one model."""

    environment: BenchmarkEnvironment
    model: BenchmarkModel
    measurements: list[BenchmarkMeasurement] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def _git_sha(repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return platform.processor() or platform.machine() or "unknown"


def collect_environment(
    *, device: torch.device, seed: int, repo_root: Path | None = None
) -> BenchmarkEnvironment:
    """Collect stable environment metadata for a benchmark run."""
    return BenchmarkEnvironment(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_sha=_git_sha(repo_root),
        os=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        processor=platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        device=str(device),
        device_name=_device_name(device),
        torch_num_threads=torch.get_num_threads(),
        seed=seed,
    )
