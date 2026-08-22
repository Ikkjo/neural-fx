"""Tests for latency calibration module."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torchaudio

from neural_fx.preprocessing.latency import (
    LatencyCalibration,
    LatencyCalibrator,
)


class TestLatencyCalibration:
    """Test suite for latency calibration."""

    @pytest.fixture
    def temp_audio_files(self):
        """Create temporary audio files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            sample_rate = 48000
            duration = 2.0
            num_samples = int(sample_rate * duration)

            # Create input audio (sine wave)
            t = torch.linspace(0, duration, num_samples)
            input_audio = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)

            # Create target audio with known delay
            delay_samples = 500
            target_audio = torch.zeros_like(input_audio)
            target_audio[:, delay_samples:] = input_audio[:, :-delay_samples] * 0.5

            input_path = tmpdir / "input.wav"
            target_path = tmpdir / "target.wav"

            torchaudio.save(str(input_path), input_audio, sample_rate)
            torchaudio.save(str(target_path), target_audio, sample_rate)

            yield input_path, target_path, sample_rate, delay_samples

    @pytest.fixture
    def temp_tensors(self):
        """Create temporary tensors for testing."""
        sample_rate = 48000
        duration = 1.0
        num_samples = int(sample_rate * duration)

        t = torch.linspace(0, duration, num_samples)
        input_audio = torch.sin(2 * 3.14159 * 440 * t)

        # Create target with known delay
        delay_samples = 100
        target_audio = torch.zeros_like(input_audio)
        target_audio[delay_samples:] = input_audio[:-delay_samples]

        return input_audio, target_audio, delay_samples

    def test_latency_calibration_dataclass(self):
        """Test LatencyCalibration dataclass."""
        cal = LatencyCalibration(
            delay_samples=100,
            method="xcorr",
            correlation_score=0.95,
        )

        assert cal.delay_samples == 100
        assert cal.method == "xcorr"
        assert cal.correlation_score == 0.95

    def test_calibrator_creation(self):
        """Test LatencyCalibrator can be created."""
        calibrator = LatencyCalibrator()
        assert calibrator is not None

    def test_xcorr_calibration(self, temp_tensors):
        """Test xcorr method detects delay."""
        input_audio, target_audio, true_delay = temp_tensors

        calibrator = LatencyCalibrator()
        calibration = calibrator.calibrate(
            input_audio,
            target_audio,
            method="xcorr",
            max_delay=500,
        )

        # Should detect approximately the correct delay
        # Allow for some tolerance due to correlation resolution
        assert abs(calibration.delay_samples - true_delay) < 5
        assert calibration.method == "xcorr"
        assert 0 <= calibration.correlation_score <= 1

    def test_manual_calibration(self):
        """Test manual calibration method."""
        input_audio = torch.randn(1000)
        target_audio = torch.randn(1000)

        calibrator = LatencyCalibrator()
        calibration = calibrator.calibrate(
            input_audio,
            target_audio,
            method="manual",
            manual_delay=250,
        )

        assert calibration.delay_samples == 250
        assert calibration.method == "manual"
        assert calibration.correlation_score == 1.0

    def test_manual_without_delay_raises(self):
        """Test that manual method without delay raises error."""
        input_audio = torch.randn(1000)
        target_audio = torch.randn(1000)

        calibrator = LatencyCalibrator()

        with pytest.raises(ValueError, match="manual_delay must be provided"):
            calibrator.calibrate(
                input_audio,
                target_audio,
                method="manual",
            )

    def test_unknown_method_raises(self):
        """Test that unknown method raises error."""
        input_audio = torch.randn(1000)
        target_audio = torch.randn(1000)

        calibrator = LatencyCalibrator()

        with pytest.raises(ValueError, match="Unknown calibration method"):
            calibrator.calibrate(
                input_audio,
                target_audio,
                method="unknown",
            )

    def test_apply_delay_positive(self):
        """Test apply_delay with positive delay."""
        x = torch.randn(1, 1000)
        y = torch.randn(1, 1000)
        delay = 100

        calibrator = LatencyCalibrator()
        x_aligned, y_aligned = calibrator.apply_delay(x, y, delay)

        # With positive delay, y lags x, so y should be trimmed
        # Expected: y_aligned = y[..., delay:], x_aligned = x[..., :len(y_aligned)]
        assert y_aligned.shape[-1] == 1000 - delay
        assert x_aligned.shape[-1] == y_aligned.shape[-1]

    def test_apply_delay_negative(self):
        """Test apply_delay with negative delay."""
        x = torch.randn(1, 1000)
        y = torch.randn(1, 1000)
        delay = -100

        calibrator = LatencyCalibrator()
        x_aligned, y_aligned = calibrator.apply_delay(x, y, delay)

        # With negative delay, x lags y, so x should be trimmed
        assert x_aligned.shape[-1] == 1000 - abs(delay)
        assert x_aligned.shape[-1] == y_aligned.shape[-1]

    def test_apply_delay_zero(self):
        """Test apply_delay with zero delay."""
        x = torch.randn(1, 1000)
        y = torch.randn(1, 950)

        calibrator = LatencyCalibrator()
        x_aligned, y_aligned = calibrator.apply_delay(x, y, 0)

        # Should match to min length
        assert x_aligned.shape[-1] == 950
        assert y_aligned.shape[-1] == 950

    def test_apply_delay_1d_tensors(self):
        """Test apply_delay with 1D tensors."""
        x = torch.randn(1000)
        y = torch.randn(1000)
        delay = 50

        calibrator = LatencyCalibrator()
        x_aligned, y_aligned = calibrator.apply_delay(x, y, delay)

        assert x_aligned.ndim == 1
        assert y_aligned.ndim == 1
        assert x_aligned.shape[0] == 950
        assert y_aligned.shape[0] == 950

    def test_compensate_method(self, temp_tensors):
        """Test compensate method with calibration result."""
        input_audio, target_audio, true_delay = temp_tensors

        calibrator = LatencyCalibrator()
        calibration = calibrator.calibrate(
            input_audio,
            target_audio,
            method="xcorr",
            max_delay=500,
        )

        x_aligned, y_aligned = calibrator.compensate(
            input_audio, target_audio, calibration
        )

        assert x_aligned.shape == y_aligned.shape

    def test_calibration_on_audio_files(self, temp_audio_files):
        """Test calibration on actual audio files."""
        input_path, target_path, sample_rate, true_delay = temp_audio_files

        # Load audio
        input_audio, sr = torchaudio.load(str(input_path))
        target_audio, _ = torchaudio.load(str(target_path))

        calibrator = LatencyCalibrator()
        calibration = calibrator.calibrate(
            input_audio,
            target_audio,
            method="xcorr",
            max_delay=1000,
        )

        # Should detect approximately correct delay
        assert abs(calibration.delay_samples - true_delay) < 20
        assert calibration.correlation_score > 0.5

    def test_multi_channel_input(self):
        """Test calibration with multi-channel audio."""
        # Create stereo audio
        input_audio = torch.randn(2, 10000)
        target_audio = torch.randn(2, 10000)

        # Add delay to second channel
        delay = 100
        target_audio[:, delay:] = input_audio[:, :-delay] * 0.5

        calibrator = LatencyCalibrator()
        calibration = calibrator.calibrate(
            input_audio,
            target_audio,
            method="xcorr",
            max_delay=500,
        )

        # Should detect delay
        assert abs(calibration.delay_samples - delay) < 10

    def test_no_delay_case(self):
        """Test calibration when there's no delay."""
        input_audio = torch.randn(10000)
        target_audio = input_audio * 0.8  # Same signal, different amplitude

        calibrator = LatencyCalibrator()
        calibration = calibrator.calibrate(
            input_audio,
            target_audio,
            method="xcorr",
            max_delay=500,
        )

        # Should find approximately zero delay
        assert abs(calibration.delay_samples) < 10
        assert calibration.correlation_score > 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
