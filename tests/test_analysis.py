"""Tests for analysis/plotting module."""

import sys
import os
import tempfile
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchaudio
import pytest

from neural_fx.analysis.plotting import TrainingAnalyzer, create_analyzer
from neural_fx.config import ModelConfig, LSTMParams
from neural_fx.models.recurrent import NeuralfxLSTM
from neural_fx.data.dataset import AudioDataset


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

    def test_analyzer_creation(self, temp_model):
        """Test TrainingAnalyzer can be created."""
        analyzer = TrainingAnalyzer(temp_model)
        assert analyzer is not None
        assert analyzer.model == temp_model

    def test_create_analyzer_factory(self, temp_model):
        """Test factory function creates analyzer."""
        analyzer = create_analyzer(temp_model)
        assert isinstance(analyzer, TrainingAnalyzer)

    def test_analyzer_uses_model_sample_rate(self):
        """Plotting inherits the authoritative model sample rate."""
        config = ModelConfig(
            type="lstm",
            params=LSTMParams(hidden_size=4, num_layers=1),
            sample_rate=44100,
        )
        analyzer = TrainingAnalyzer(NeuralfxLSTM(config))

        assert analyzer._resolve_sample_rate() == 44100
        with pytest.warns(UserWarning, match="using the configured sample rate"):
            assert analyzer._resolve_sample_rate(48000) == 44100

    def test_calculate_esr(self, temp_model):
        """Test ESR calculation."""
        analyzer = TrainingAnalyzer(temp_model)

        pred = torch.randn(1, 1, 1000)
        target = torch.randn(1, 1, 1000)

        esr = analyzer.calculate_esr(pred, target)
        assert isinstance(esr, float)
        assert esr >= 0

    def test_calculate_esr_1d_input(self, temp_model):
        """Test ESR calculation with 1D input."""
        analyzer = TrainingAnalyzer(temp_model)

        pred = torch.randn(1000)
        target = torch.randn(1000)

        esr = analyzer.calculate_esr(pred, target)
        assert isinstance(esr, float)
        assert esr >= 0

    def test_calculate_esr_identical_signals(self, temp_model):
        """Test ESR is low for identical signals."""
        analyzer = TrainingAnalyzer(temp_model)

        signal = torch.randn(1, 1, 1000)

        esr = analyzer.calculate_esr(signal, signal)
        assert esr < 0.001  # Should be very small

    def test_get_esr_comment(self, temp_model):
        """Test ESR comment generation."""
        analyzer = TrainingAnalyzer(temp_model)

        assert "Excellent" in analyzer.get_esr_comment(0.0005)
        assert "Great" in analyzer.get_esr_comment(0.005)
        assert "Good" in analyzer.get_esr_comment(0.03)
        assert "Not bad" in analyzer.get_esr_comment(0.07)
        assert "Acceptable" in analyzer.get_esr_comment(0.2)
        assert "Poor" in analyzer.get_esr_comment(0.5)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
