"""Tests for benchmark schema, timing, and report writers."""

from __future__ import annotations

import csv
import json
from dataclasses import replace

import pytest

from neural_fx.benchmarking import (
    SCHEMA_VERSION,
    BenchmarkEnvironment,
    BenchmarkMeasurement,
    BenchmarkModel,
    BenchmarkResult,
    benchmark_model,
    percentile,
    write_csv,
    write_json,
    write_markdown,
)
from neural_fx.config import LSTMParams, ModelConfig
from neural_fx.models.recurrent import NeuralfxGRU


@pytest.fixture
def result() -> BenchmarkResult:
    return BenchmarkResult(
        environment=BenchmarkEnvironment(
            timestamp_utc="2026-01-01T00:00:00+00:00",
            git_sha="abc123",
            os="Test OS",
            machine="x86_64",
            processor="Test CPU",
            python_version="3.12.0",
            torch_version="2.7.0",
            device="cpu",
            device_name="Test CPU",
            torch_num_threads=1,
            seed=42,
        ),
        model=BenchmarkModel(
            name="gru_test",
            type="gru",
            sample_rate=48_000,
            input_size=1,
            output_size=1,
            parameter_count=123,
            parameter_bytes=492,
            config_path="gru.yaml",
        ),
        measurements=[
            BenchmarkMeasurement(
                mode="streaming_block",
                block_size=64,
                warmup_iterations=2,
                iterations=5,
                latency_median_ms=0.2,
                latency_p95_ms=0.3,
                real_time_factor=0.15,
                throughput_samples_per_second=320_000.0,
            )
        ],
    )


def test_percentile_interpolates_and_validates() -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.5
    assert percentile([1.0, 2.0, 3.0], 0.95) == pytest.approx(2.9)
    with pytest.raises(ValueError, match="empty"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="between"):
        percentile([1.0], 1.1)


def test_benchmark_recurrent_model() -> None:
    config = ModelConfig(
        type="gru",
        params=LSTMParams(hidden_size=4, num_layers=1, conv1d=None),
        input_size=1,
        output_size=1,
        sample_rate=48_000,
    )
    model = NeuralfxGRU(config)

    measurements = benchmark_model(
        model, block_sizes=[1, 16], warmup_iterations=1, iterations=3
    )

    assert [item.block_size for item in measurements] == [1, 16]
    assert all(item.mode == "streaming_block" for item in measurements)
    assert all(item.latency_median_ms > 0 for item in measurements)
    assert all(item.latency_p95_ms >= item.latency_median_ms for item in measurements)
    assert all(item.real_time_factor > 0 for item in measurements)
    assert model.hidden_state is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"block_sizes": []}, "block_sizes"),
        ({"block_sizes": [0]}, "block_sizes"),
        ({"warmup_iterations": -1}, "warmup_iterations"),
        ({"iterations": 0}, "iterations"),
    ],
)
def test_benchmark_validates_arguments(kwargs, message) -> None:
    config = ModelConfig(type="gru", params=LSTMParams(hidden_size=2, conv1d=None))
    model = NeuralfxGRU(config)
    with pytest.raises(ValueError, match=message):
        benchmark_model(model, **kwargs)


def test_json_writer_uses_versioned_envelope(tmp_path, result) -> None:
    path = write_json([result], tmp_path / "nested" / "results.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["results"][0]["schema_version"] == SCHEMA_VERSION
    assert payload["results"][0]["model"]["name"] == "gru_test"
    assert payload["results"][0]["measurements"][0]["state_bytes"] is None


def test_json_writer_rejects_mixed_versions(tmp_path, result) -> None:
    incompatible = replace(result, schema_version="2.0")
    with pytest.raises(ValueError, match="same schema"):
        write_json([result, incompatible], tmp_path / "results.json")


def test_csv_writer_flattens_measurements(tmp_path, result) -> None:
    path = write_csv([result], tmp_path / "results.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["model_type"] == "gru"
    assert rows[0]["block_size"] == "64"
    assert rows[0]["peak_cpu_rss_bytes"] == ""


def test_markdown_writer_builds_comparison_table(tmp_path, result) -> None:
    path = write_markdown([result], tmp_path / "results.md")
    markdown = path.read_text(encoding="utf-8")

    assert "| Model | Device | Block |" in markdown
    assert "| gru_test | Test CPU | 64 | 0.2000 | 0.3000 | 0.1500 | 123 |" in markdown
