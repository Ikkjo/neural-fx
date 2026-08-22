"""Tests for analysis/plotting module."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torchaudio

from neural_fx.analysis.plotting import TrainingAnalyzer
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
    load_config,
)
from neural_fx.data.dataset import AudioDataset
from neural_fx.models import create_model_from_config
from neural_fx.models.recurrent import NeuralfxLSTM
from neural_fx.training.lightning_module import NeuralFXModule
from scripts.analyze import load_checkpoint


class TestTrainingAnalyzer:
    """Test suite for TrainingAnalyzer."""

    @pytest.fixture
    def temp_model(self):
        """Create a simple model for testing."""
        config = ModelConfig(
            type="lstm",
            params=LSTMParams(hidden_size=16, num_layers=1),
            input_size=1,
            output_size=1,
        )
        model = NeuralfxLSTM(config)
        model.eval()
        return model

    @pytest.fixture
    def temp_dataset(self):
        """Create a temporary dataset for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            sample_rate = 48000
            duration = 2.0
            num_samples = int(sample_rate * duration)

            # Create input and target audio
            t = torch.linspace(0, duration, num_samples)
            input_audio = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
            target_audio = input_audio * 0.5

            input_path = tmpdir / "input.wav"
            target_path = tmpdir / "target.wav"

            torchaudio.save(str(input_path), input_audio, sample_rate)
            torchaudio.save(str(target_path), target_audio, sample_rate)

            dataset = AudioDataset(
                input_path=input_path,
                target_path=target_path,
                segment_length=8192,
                sample_rate=sample_rate,
            )

            yield dataset


    def test_generate_report(self, temp_model, temp_dataset):
        """Test report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            analyzer = TrainingAnalyzer(temp_model)
            report = analyzer.generate_report(
                temp_dataset, output_dir, num_samples=4800
            )

            assert "esr" in report
            assert "esr_comment" in report
            assert "mse" in report
            assert "correlation" in report
            assert "num_params" in report
            assert "plots" in report

            # Check plots were created
            assert (output_dir / "waveform_comparison.png").exists()
            assert (output_dir / "spectrograms.png").exists()

    def test_report_esr_value(self, temp_model, temp_dataset):
        """Test that ESR value is reasonable in report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = TrainingAnalyzer(temp_model)
            report = analyzer.generate_report(
                temp_dataset, Path(tmpdir), num_samples=4800
            )

            # ESR should be non-negative
            assert report["esr"] >= 0
            # Should have comment
            assert isinstance(report["esr_comment"], str)

    def test_report_correlation(self, temp_model, temp_dataset):
        """Test correlation coefficient in report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = TrainingAnalyzer(temp_model)
            report = analyzer.generate_report(temp_dataset, Path(tmpdir))

            # Correlation should be between -1 and 1
            assert -1 <= report["correlation"] <= 1

    def test_checkpoint_reconstructs_complete_model_config(self):
        """Analysis can rebuild a model without locating its original YAML."""
        model_config = ModelConfig(
            type="gru",
            params=LSTMParams(hidden_size=7, num_layers=1),
            sample_rate=44100,
        )
        config = NeuralFXConfig(
            version="1.0",
            name="roundtrip",
            model=model_config,
            training=TrainingConfig(batch_size=2, segment_length=1024),
            optimizer=OptimizerConfig(lr=0.002),
            lr_scheduler=LRSchedulerConfig(),
            loss=LossConfig(mask_first=64),
            data=DataConfig(
                train=DataPaths(input="input.wav", target="target.wav")
            ),
        )
        module = NeuralFXModule(create_model_from_config(model_config), config)
        checkpoint = {"state_dict": module.state_dict()}
        module.on_save_checkpoint(checkpoint)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.ckpt"
            torch.save(checkpoint, checkpoint_path)
            loaded_model, loaded_config = load_checkpoint(str(checkpoint_path))

        assert loaded_config == config
        assert loaded_model.config.type == "gru"
        assert loaded_model.sample_rate == 44100

    def test_checkpoint_loads_raw_state_with_explicit_config(self, tmp_path):
        config_path = Path("configs/models/lstm/lstm_nano.yaml")
        config = load_config(config_path)
        model = create_model_from_config(config.model)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.25)
        checkpoint_path = tmp_path / "raw-state.pt"
        torch.save(model.state_dict(), checkpoint_path)

        loaded_model, loaded_config = load_checkpoint(
            str(checkpoint_path), str(config_path)
        )

        assert loaded_config == config
        for name, value in model.state_dict().items():
            assert torch.equal(loaded_model.state_dict()[name], value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
