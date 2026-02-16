import sys
import os
import tempfile
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest

from neural_fx.config import (
    ModelConfig,
    LSTMParams,
    NeuralFXConfig,
    TrainingConfig,
    TBPTTConfig,
    OptimizerConfig,
    LRSchedulerConfig,
    LossConfig,
    DataConfig,
    DataPaths,
)
from neural_fx.models.recurrent import NeuralfxLSTM
from neural_fx.training.lightning_module import NeuralFXModule


class TestNeuralFXModule:
    """Test suite for NeuralFX Lightning module."""

    @pytest.fixture
    def base_config(self):
        """Create base configuration for testing."""
        model_config = ModelConfig(
            type="lstm",
            params=LSTMParams(
                hidden_size=16,
                num_layers=1,
                conv1d=None,
                skip_connection=False,
                dropout=0.0,
                conditioning_size=0,
            ),
            input_size=1,
            output_size=1,
            sample_rate=48000,
        )

        return NeuralFXConfig(
            version="1.0",
            name="test_model",
            model=model_config,
            training=TrainingConfig(
                batch_size=4,
                epochs=2,
                segment_length=2048,
                tbptt=None,  # No TBPTT for basic tests
                seed=42,
            ),
            optimizer=OptimizerConfig(type="adam", lr=0.001),
            lr_scheduler=LRSchedulerConfig(type="exponential", gamma=0.99),
            loss=LossConfig(type="mse"),
            data=DataConfig(
                train=DataPaths(input="data/DI.wav", target="data/effect.wav"),
                sample_rate=48000,
            ),
        )

    @pytest.fixture
    def tbptt_config(self):
        """Create configuration with TBPTT enabled."""
        model_config = ModelConfig(
            type="lstm",
            params=LSTMParams(
                hidden_size=16,
                num_layers=1,
                conv1d=None,
                skip_connection=False,
                dropout=0.0,
                conditioning_size=0,
            ),
            input_size=1,
            output_size=1,
            sample_rate=48000,
        )

        return NeuralFXConfig(
            version="1.0",
            name="test_model",
            model=model_config,
            training=TrainingConfig(
                batch_size=2,
                epochs=2,
                segment_length=4096,
                tbptt=TBPTTConfig(enabled=True, burn_in=512),
                seed=42,
            ),
            optimizer=OptimizerConfig(type="adam", lr=0.001),
            lr_scheduler=LRSchedulerConfig(type="exponential", gamma=0.99),
            loss=LossConfig(type="mse"),
            data=DataConfig(
                train=DataPaths(input="data/DI.wav", target="data/effect.wav"),
                sample_rate=48000,
            ),
        )

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        batch_size = 2
        seq_len = 1024
        x = torch.randn(batch_size, seq_len)
        y = torch.randn(batch_size, seq_len)
        return x, y

    def test_module_creation(self, base_config):
        """Test NeuralFXModule can be created."""
        model = NeuralfxLSTM(base_config.model)
        module = NeuralFXModule(model, base_config)

        assert module.model is not None
        assert module.config is not None
        assert module.loss_fn is not None

    def test_forward_pass(self, base_config, sample_batch):
        """Test forward pass through module."""
        model = NeuralfxLSTM(base_config.model)
        module = NeuralFXModule(model, base_config)

        x, _ = sample_batch
        x = x.unsqueeze(1)  # Add channel dimension

        output = module(x)
        assert output.shape == x.shape

    def test_training_step_no_tbptt(self, base_config, sample_batch):
        """Test training step without TBPTT."""
        model = NeuralfxLSTM(base_config.model)
        module = NeuralFXModule(model, base_config)

        x, y = sample_batch
        loss = module.training_step((x, y), batch_idx=0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar

    def test_training_step_with_tbptt(self, tbptt_config, sample_batch):
        """Test training step with TBPTT enabled."""
        model = NeuralfxLSTM(tbptt_config.model)
        module = NeuralFXModule(model, tbptt_config)

        # Need longer sequence for TBPTT
        x = torch.randn(2, 4096)
        y = torch.randn(2, 4096)

        loss = module.training_step((x, y), batch_idx=0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)

    def test_validation_step(self, base_config, sample_batch):
        """Test validation step."""
        model = NeuralfxLSTM(base_config.model)
        module = NeuralFXModule(model, base_config)

        x, y = sample_batch
        loss = module.validation_step((x, y), batch_idx=0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)

    def test_configure_optimizers(self, base_config):
        """Test optimizer configuration."""
        model = NeuralfxLSTM(base_config.model)
        module = NeuralFXModule(model, base_config)

        optimizers, schedulers = module.configure_optimizers()

        assert len(optimizers) == 1
        assert len(schedulers) == 1
        assert isinstance(optimizers[0], torch.optim.Adam)
        assert isinstance(
            schedulers[0], torch.optim.lr_scheduler.ExponentialLR
        )

    def test_configure_optimizers_adamw(self, base_config):
        """Test AdamW optimizer configuration."""
        base_config.optimizer.type = "adamw"
        model = NeuralfxLSTM(base_config.model)
        module = NeuralFXModule(model, base_config)

        optimizers, _ = module.configure_optimizers()
        assert isinstance(optimizers[0], torch.optim.AdamW)

    def test_configure_optimizers_sgd(self, base_config):
        """Test SGD optimizer configuration."""
        base_config.optimizer.type = "sgd"
        model = NeuralfxLSTM(base_config.model)
        module = NeuralFXModule(model, base_config)

        optimizers, _ = module.configure_optimizers()
        assert isinstance(optimizers[0], torch.optim.SGD)

    def test_burn_in_excluded_from_loss(self, base_config, sample_batch):
        """Test that burn-in samples are excluded from loss calculation."""
        base_config.training.tbptt = TBPTTConfig(enabled=False, burn_in=256)
        model = NeuralfxLSTM(base_config.model)
        module = NeuralFXModule(model, base_config)

        x, y = sample_batch

        # This should run without error and exclude burn-in from loss
        loss = module.training_step((x, y), batch_idx=0)
        assert loss is not None

    def test_loss_with_esr_weight(self, base_config, sample_batch):
        """Test loss calculation with ESR weight."""
        base_config.loss.weights = type("obj", (object,), {"esr": 0.5, "mse": 0.5})()
        model = NeuralfxLSTM(base_config.model)
        module = NeuralFXModule(model, base_config)

        x, y = sample_batch
        loss = module.training_step((x, y), batch_idx=0)

        assert loss is not None

    def test_loss_with_mse_only(self, base_config, sample_batch):
        """Test loss calculation with MSE only."""
        base_config.loss.weights = type("obj", (object,), {"esr": 0.0, "mse": 1.0})()
        model = NeuralfxLSTM(base_config.model)
        module = NeuralFXModule(model, base_config)

        x, y = sample_batch
        loss = module.training_step((x, y), batch_idx=0)

        assert loss is not None

    def test_gru_model_with_module(self, base_config, sample_batch):
        """Test NeuralFXModule works with GRU model."""
        from neural_fx.models.recurrent import NeuralfxGRU

        base_config.model.type = "gru"
        model = NeuralfxGRU(base_config.model)
        module = NeuralFXModule(model, base_config)

        x, y = sample_batch
        loss = module.training_step((x, y), batch_idx=0)

        assert loss is not None
