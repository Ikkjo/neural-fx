"""Tests for training callbacks module."""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

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
from neural_fx.training.callbacks import (
    NeuralFXCheckpoint,
    ValidationEarlyStopping,
)


class _CheckpointModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        loss = (self.weight * batch[0]).square().mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = (self.weight * batch[0]).square().mean()
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=2)

    def val_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=2)


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

    def test_native_checkpointing_saves_best_last_and_resumes(
        self, base_config, tmp_path
    ):
        callback = NeuralFXCheckpoint(
            base_config,
            dirpath=tmp_path,
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=3,
            save_last=True,
        )
        trainer = L.Trainer(
            max_epochs=2,
            accelerator="cpu",
            callbacks=[callback],
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(_CheckpointModule())

        best = Path(callback.best_model_path)
        last = Path(callback.last_model_path)
        assert best.is_file()
        assert last.is_file()
        assert best.with_suffix(".meta.json").is_file()
        assert (
            json.loads(last.with_suffix(".meta.json").read_text())["checkpoint_file"]
            == "last.ckpt"
        )
        assert torch.load(last, weights_only=False)["epoch"] == 1

        resumed = L.Trainer(
            max_epochs=3,
            accelerator="cpu",
            callbacks=[NeuralFXCheckpoint(base_config, dirpath=tmp_path)],
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        resumed.fit(_CheckpointModule(), ckpt_path=last)
        assert resumed.global_step == 6


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

    def test_relative_min_delta_scales_with_best_score(self):
        callback = ValidationEarlyStopping(
            monitor="val_loss",
            min_delta=0.005,
            min_delta_mode="relative",
            patience=2,
            mode="min",
        )

        should_stop, _ = callback._evaluate_stopping_criteria(torch.tensor(1.0))
        assert should_stop is False
        assert callback.wait_count == 0

        should_stop, _ = callback._evaluate_stopping_criteria(torch.tensor(0.996))
        assert should_stop is False
        assert callback.wait_count == 1
        assert callback.best_score.item() == pytest.approx(1.0)

        should_stop, _ = callback._evaluate_stopping_criteria(torch.tensor(0.994))
        assert should_stop is False
        assert callback.wait_count == 0
        assert callback.best_score.item() == pytest.approx(0.994)

    def test_relative_min_delta_stops_after_patience(self):
        callback = ValidationEarlyStopping(
            monitor="val_loss",
            min_delta=0.005,
            min_delta_mode="relative",
            patience=2,
            mode="min",
        )

        callback._evaluate_stopping_criteria(torch.tensor(1.0))
        callback._evaluate_stopping_criteria(torch.tensor(0.999))
        should_stop, _ = callback._evaluate_stopping_criteria(torch.tensor(0.998))

        assert should_stop is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
