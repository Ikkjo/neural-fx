"""Model timing and statistics for neural audio inference."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable

import torch

from neural_fx.models.base import BaseNeuralFXModel

from .schema import BenchmarkMeasurement


def percentile(values: list[float], quantile: float) -> float:
    """Calculate a linearly interpolated percentile without NumPy."""
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_forward(
    model: BaseNeuralFXModel, input_block: torch.Tensor, device: torch.device
) -> float:
    _synchronize(device)
    started_ns = time.perf_counter_ns()
    model(input_block)
    _synchronize(device)
    return (time.perf_counter_ns() - started_ns) / 1_000_000.0


def benchmark_model(
    model: BaseNeuralFXModel,
    *,
    block_sizes: Iterable[int] = (1, 64, 128, 256, 512, 1024),
    warmup_iterations: int = 20,
    iterations: int = 100,
    device: str | torch.device = "cpu",
) -> list[BenchmarkMeasurement]:
    """Benchmark stateful streaming inference for each requested block size.

    Model state is preserved between blocks to represent a real audio stream.
    It is reset before warm-up and again before measured iterations so warm-up
    samples do not affect the measured stream.
    """
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    normalized_sizes = list(block_sizes)
    if not normalized_sizes or any(size <= 0 for size in normalized_sizes):
        raise ValueError("block_sizes must contain only positive integers")

    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
    measurements: list[BenchmarkMeasurement] = []

    with torch.inference_mode():
        for block_size in normalized_sizes:
            input_block = torch.randn(
                1,
                model.config.input_size,
                block_size,
                device=target_device,
            )

            model.reset_state()
            for _ in range(warmup_iterations):
                model(input_block)
            _synchronize(target_device)

            model.reset_state()
            latency_ms = [
                _time_forward(model, input_block, target_device)
                for _ in range(iterations)
            ]

            median_ms = percentile(latency_ms, 0.5)
            p95_ms = percentile(latency_ms, 0.95)
            audio_duration_ms = block_size / model.sample_rate * 1000.0
            measurements.append(
                BenchmarkMeasurement(
                    mode="streaming_block",
                    block_size=block_size,
                    warmup_iterations=warmup_iterations,
                    iterations=iterations,
                    latency_median_ms=median_ms,
                    latency_p95_ms=p95_ms,
                    real_time_factor=median_ms / audio_duration_ms,
                    throughput_samples_per_second=block_size / (median_ms / 1000.0),
                )
            )

    model.reset_state()
    return measurements
