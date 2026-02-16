"""Tests for training callbacks module."""

import sys
import os
import tempfile
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest
from unittest.mock import Mock, MagicMock, patch

from neural_fx.training.callbacks import (
    NeuralFXCheckpoint,
    ValidationEarlyStopping,
)
from neural_fx.config import (
    NeuralFXConfig,
    ModelConfig,
    TrainingConfig,
    OptimizerConfig,
    LRSchedulerConfig,
    LossConfig,
    DataConfig,
    DataPaths,
    LSTMParams,
)


class TestNeuralFXCheckpoint:
    """Test suite for NeuralFXCheckpoint callback."""

    @pytest.fixture
    def base_config(self):
        """Create base configuration for testing."""
        return NeuralFXConfig(
            version="1.0",
            name="test_model",
            model=ModelConfig(
                type="lstm",
                params=LSTMParams(hidden_size=16, num_layers=1),
                input_size=1,
                output_size=1,
            ),
            training=TrainingConfig(
                batch_size=2,
                epochs=2,
                segment_length=2048,
            ),
            optimizer=OptimizerConfig(type="adam", lr=0.001),
            lr_scheduler=LRSchedulerConfig(type="exponential", gamma=0.99),
            loss=LossConfig(type="mse"),
            data=DataConfig(
                train=DataPaths(input="data/DI.wav", target="data/effect.wav"),
            ),
        )

    def test_checkpoint_creation(self, base_config):
        """Test NeuralFXCheckpoint can be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = NeuralFXCheckpoint(
                config=base_config,
                dirpath=tmpdir,
                monitor="val_loss",
            )
            assert callback is not None
            assert callback._config == base_config

    def test_checkpoint_with_latency_calibration(self, base_config):
        """Test checkpoint with latency calibration data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            latency_cal = {
                "delay_samples": 100,
                "method": "xcorr",
                "correlation_score": 0.95,
            }

            callback = NeuralFXCheckpoint(
                config=base_config,
                input_file="input.wav",
                target_file="target.wav",
                latency_calibration=latency_cal,
                dirpath=tmpdir,
                monitor="val_loss",
            )

            assert callback._input_file == "input.wav"
            assert callback._target_file == "target.wav"
            assert callback._latency_calibration == latency_cal

    def test_get_hardware_info(self, base_config):
        """Test hardware info collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = NeuralFXCheckpoint(
                config=base_config,
                dirpath=tmpdir,
                monitor="val_loss",
            )

            info = callback._get_hardware_info()
            assert "cuda_available" in info
            assert isinstance(info["cuda_available"], bool)

    def test_config_to_dict(self, base_config):
        """Test config conversion to dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = NeuralFXCheckpoint(
                config=base_config,
                dirpath=tmpdir,
                monitor="val_loss",
            )

            config_dict = callback._config_to_dict(base_config)
            assert isinstance(config_dict, dict)
            assert config_dict["version"] == "1.0"
            assert config_dict["name"] == "test_model"


class TestValidationEarlyStopping:
    """Test suite for ValidationEarlyStopping callback."""

    def test_early_stopping_creation(self):
        """Test ValidationEarlyStopping can be created."""
        callback = ValidationEarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=10,
            mode="min",
        )
        assert callback is not None
        assert callback.monitor == "val_loss"
        assert callback.patience == 10

    def test_early_stopping_with_esr_monitor(self):
        """Test early stopping with ESR monitor."""
        callback = ValidationEarlyStopping(
            monitor="val_esr",
            min_delta=0.001,
            patience=5,
            mode="min",
        )
        assert callback.monitor == "val_esr"

    def test_early_stopping_mode_max(self):
        """Test early stopping with max mode."""
        callback = ValidationEarlyStopping(
            monitor="accuracy",
            mode="max",
        )
        assert callback.mode == "max"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
