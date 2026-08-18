"""Tests for inference benchmarking and result formatting."""

import json

from neural_fx.analysis.benchmarking import (
    benchmark_model,
    format_benchmark_table,
    load_benchmark_result,
    write_benchmark_result,
)
from neural_fx.config import LSTMParams, ModelConfig
from neural_fx.models.recurrent import NeuralfxGRU


def test_benchmark_model_records_latency_memory_and_deadlines(tmp_path) -> None:
    model = NeuralfxGRU(
        ModelConfig(
            type="gru",
            params=LSTMParams(hidden_size=4, num_layers=1),
            sample_rate=48_000,
        )
    )

    result = benchmark_model(
        model,
        model_name="test_gru",
        config_path=tmp_path / "config.yaml",
        block_sizes=[32],
        num_samples=128,
        warmup_runs=1,
        measurement_runs=2,
    )

    assert result["model"]["type"] == "gru"
    assert result["model"]["trainable_parameters"] == model.num_params
    assert result["offline"]["runs"] == 2
    assert result["offline"]["real_time_factor"] > 0
    assert result["blocks"][0]["block_size"] == 32
    assert 0 <= result["blocks"][0]["deadline_misses"] <= 2
    assert result["memory"]["model_state_bytes"] > 0


def test_benchmark_result_round_trip_and_markdown_table(tmp_path) -> None:
    result = {
        "schema_version": "1.0",
        "model": {
            "name": "nano",
            "type": "lstm",
            "trainable_parameters": 123,
        },
        "runtime": {},
        "workload": {},
        "offline": {"real_time_factor": 0.25},
        "blocks": [{"block_size": 64, "p95_ms": 0.5}],
        "memory": {
            "model_state_bytes": 1024,
            "process_peak_rss_bytes": 2**20,
        },
    }
    path = tmp_path / "result.json"
    write_benchmark_result(result, path)

    loaded = load_benchmark_result(path)
    table = format_benchmark_table([loaded])

    assert json.loads(path.read_text())["schema_version"] == "1.0"
    assert "| nano | lstm | 123 |" in table
    assert "64-sample p95 (ms)" in table
