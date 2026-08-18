"""Reproducible inference benchmarks for neural audio models."""

from __future__ import annotations

import json
import math
import platform
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..config import load_config
from ..models import BaseNeuralFXModel, create_model_from_config

BENCHMARK_SCHEMA_VERSION = "1.0"


def load_model_for_evaluation(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> tuple[BaseNeuralFXModel, Any]:
    """Create a model from a config and optionally load checkpoint weights."""
    config_path = Path(config_path)
    config = load_config(config_path)
    model = create_model_from_config(config.model)

    if checkpoint_path is not None:
        checkpoint = torch.load(
            Path(checkpoint_path), map_location="cpu", weights_only=False
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint does not contain a state dictionary")
        model_state = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if not key.startswith("model.") or key.removeprefix("model.")
        }
        model.load_state_dict(model_state, strict=True)

    model.to(device)
    model.eval()
    return model, config


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_rss_bytes() -> int:
    """Return the process high-water RSS in bytes on Linux and macOS."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak)
    return int(peak * 1024)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_summary(
    latencies_ms: list[float], samples_per_call: int, sample_rate: int
) -> dict[str, float | int]:
    median_ms = statistics.median(latencies_ms)
    audio_duration_ms = samples_per_call / sample_rate * 1000.0
    return {
        "runs": len(latencies_ms),
        "median_ms": median_ms,
        "p95_ms": _percentile(latencies_ms, 0.95),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "throughput_samples_per_second": samples_per_call / (median_ms / 1000.0),
        "real_time_factor": median_ms / audio_duration_ms,
    }


def _time_forward(model: BaseNeuralFXModel, audio: Tensor) -> float:
    device = audio.device
    _synchronize(device)
    start = time.perf_counter_ns()
    model(audio)
    _synchronize(device)
    return (time.perf_counter_ns() - start) / 1_000_000.0


def benchmark_model(
    model: BaseNeuralFXModel,
    *,
    model_name: str,
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    block_sizes: list[int] | None = None,
    num_samples: int = 48_000,
    warmup_runs: int = 3,
    measurement_runs: int = 20,
    batch_size: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Benchmark offline and stateful block inference for one model."""
    if num_samples <= 0 or warmup_runs < 0 or measurement_runs <= 0:
        raise ValueError("num_samples and measurement_runs must be positive")
    block_sizes = block_sizes or [64, 128, 256, 512]
    if any(size <= 0 for size in block_sizes):
        raise ValueError("block sizes must be positive")

    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    input_size = int(model.config.input_size)
    sample_rate = int(model.sample_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    offline_audio = torch.randn(
        batch_size, input_size, num_samples, generator=generator
    ).to(device)

    model.reset_state()
    with torch.inference_mode():
        for _ in range(warmup_runs):
            model.reset_state()
            model(offline_audio)
        _synchronize(device)

        offline_latencies = []
        for _ in range(measurement_runs):
            model.reset_state()
            offline_latencies.append(_time_forward(model, offline_audio))

        block_results = []
        for block_size in block_sizes:
            block = offline_audio[..., :block_size]
            model.reset_state()
            for _ in range(warmup_runs):
                model(block)
            _synchronize(device)

            block_latencies = [
                _time_forward(model, block) for _ in range(measurement_runs)
            ]
            summary = _latency_summary(block_latencies, block_size, sample_rate)
            deadline_ms = block_size / sample_rate * 1000.0
            summary.update(
                {
                    "block_size": block_size,
                    "deadline_ms": deadline_ms,
                    "deadline_misses": sum(
                        latency > deadline_ms for latency in block_latencies
                    ),
                }
            )
            block_results.append(summary)

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    state_bytes = sum(
        value.numel() * value.element_size() for value in model.state_dict().values()
    )
    checkpoint_size = (
        Path(checkpoint_path).stat().st_size if checkpoint_path is not None else None
    )
    memory: dict[str, Any] = {
        "model_state_bytes": state_bytes,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "process_peak_rss_note": (
            "Process high-water RSS; run one model per CLI process for comparable CPU results."
        ),
    }
    if device.type == "cuda":
        memory["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        memory["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)

    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or platform.machine()
    )
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": model_name,
            "type": model.config.type,
            "config_path": str(Path(config_path).resolve()),
            "checkpoint_path": (
                str(Path(checkpoint_path).resolve())
                if checkpoint_path is not None
                else None
            ),
            "checkpoint_size_bytes": checkpoint_size,
            "trainable_parameters": trainable_parameters,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "device_name": device_name,
            "dtype": str(offline_audio.dtype).removeprefix("torch."),
            "torch_num_threads": torch.get_num_threads(),
        },
        "workload": {
            "sample_rate": sample_rate,
            "batch_size": batch_size,
            "input_channels": input_size,
            "offline_samples": num_samples,
            "block_sizes": block_sizes,
            "warmup_runs": warmup_runs,
            "measurement_runs": measurement_runs,
            "seed": seed,
        },
        "offline": _latency_summary(
            offline_latencies, num_samples, sample_rate
        ),
        "blocks": block_results,
        "memory": memory,
    }


def write_benchmark_result(result: dict[str, Any], output_path: str | Path) -> None:
    """Write a benchmark result as stable, readable JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")


def load_benchmark_result(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a benchmark result."""
    result = json.loads(Path(path).read_text())
    if result.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported benchmark schema: {result.get('schema_version')}"
        )
    for key in ("model", "runtime", "workload", "offline", "blocks", "memory"):
        if key not in result:
            raise ValueError(f"Benchmark result is missing '{key}'")
    return result


def format_benchmark_table(results: list[dict[str, Any]]) -> str:
    """Render benchmark JSON results as a Markdown comparison table."""
    if not results:
        raise ValueError("At least one benchmark result is required")
    block_sizes = sorted(
        {block["block_size"] for result in results for block in result["blocks"]}
    )
    headers = [
        "Model",
        "Type",
        "Parameters",
        "State (MiB)",
        "Peak RSS (MiB)",
        "Offline RTF",
    ] + [f"{size}-sample p95 (ms)" for size in block_sizes]
    rows = [headers, ["---"] * len(headers)]
    for result in results:
        by_size = {block["block_size"]: block for block in result["blocks"]}
        row = [
            str(result["model"]["name"]),
            str(result["model"]["type"]),
            f"{result['model']['trainable_parameters']:,}",
            f"{result['memory']['model_state_bytes'] / 2**20:.3f}",
            f"{result['memory']['process_peak_rss_bytes'] / 2**20:.1f}",
            f"{result['offline']['real_time_factor']:.4f}",
        ]
        row.extend(
            f"{by_size[size]['p95_ms']:.3f}" if size in by_size else "—"
            for size in block_sizes
        )
        rows.append(row)
    return "\n".join("| " + " | ".join(row) + " |" for row in rows) + "\n"
