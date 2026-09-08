"""Tests for inference benchmarking and result formatting."""

import json

import pytest
import torch

from neural_fx.analysis.benchmarking import (
    benchmark_model,
    format_benchmark_table,
    load_benchmark_result,
    load_model_for_evaluation,
    write_benchmark_result,
)
from neural_fx.config import (
    DataConfig,
    DataPaths,
    LossConfig,
    LRSchedulerConfig,
    LSTMParams,
    ModelConfig,
    NeuralFXConfig,
    OptimizerConfig,
    TrainingConfig,
)
from neural_fx.models import create_model_from_config
from neural_fx.models.recurrent import NeuralfxGRU
from neural_fx.training.lightning_module import NeuralFXModule


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


def test_benchmark_model_rejects_blocks_larger_than_workload() -> None:
    model = NeuralfxGRU(
        ModelConfig(
            type="gru",
            params=LSTMParams(hidden_size=4, num_layers=1),
            sample_rate=48_000,
        )
    )

    with pytest.raises(ValueError, match="cannot exceed num_samples"):
        benchmark_model(
            model,
            model_name="test_gru",
            block_sizes=[129],
            num_samples=128,
            warmup_runs=0,
            measurement_runs=1,
        )


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


def test_model_loader_uses_checkpoint_embedded_config(tmp_path) -> None:
    model_config = ModelConfig(
        type="gru",
        params=LSTMParams(hidden_size=5, num_layers=1),
        sample_rate=44_100,
    )
    config = NeuralFXConfig(
        version="1.0",
        name="embedded-gru",
        model=model_config,
        training=TrainingConfig(segment_length=4096),
        optimizer=OptimizerConfig(),
        lr_scheduler=LRSchedulerConfig(),
        loss=LossConfig(mask_first=32),
        data=DataConfig(train=DataPaths(input="input.wav", target="target.wav")),
    )
    module = NeuralFXModule(create_model_from_config(model_config), config)
    checkpoint = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(checkpoint)
    checkpoint_path = tmp_path / "embedded.ckpt"
    torch.save(checkpoint, checkpoint_path)

    model, loaded_config = load_model_for_evaluation(
        checkpoint_path=checkpoint_path
    )

    assert loaded_config == config
    assert model.config.type == "gru"
    assert model.sample_rate == 44_100
