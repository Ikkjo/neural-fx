"""Tests for data validation module."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torchaudio

from neural_fx.preprocessing.validation import (
    CheckResult,
    DataValidator,
    ValidationReport,
)


class TestCheckResult:
    """Test suite for CheckResult dataclass."""

    def test_check_result_creation(self):
        """Test CheckResult can be created."""
        result = CheckResult(
            passed=True,
            message="Test passed",
            value=0.5,
        )
        assert result.passed is True
        assert result.message == "Test passed"
        assert result.value == 0.5

    def test_check_result_without_value(self):
        """Test CheckResult without optional value."""
        result = CheckResult(
            passed=False,
            message="Test failed",
        )
        assert result.passed is False
        assert result.value is None


class TestValidationReport:
    """Test suite for ValidationReport dataclass."""

    def test_report_creation(self):
        """Test ValidationReport can be created."""
        checks = {
            "test_check": CheckResult(passed=True, message="OK"),
        }
        report = ValidationReport(
            passed=True,
            checks=checks,
            warnings=["Warning 1"],
        )
        assert report.passed is True
        assert "test_check" in report.checks
        assert len(report.warnings) == 1

    def test_report_summary(self):
        """Test get_summary method."""
        checks = {
            "check1": CheckResult(passed=True, message="OK", value=0.5),
            "check2": CheckResult(passed=False, message="Failed", value=0.9),
        }
        report = ValidationReport(
            passed=False,
            checks=checks,
            warnings=["Test warning"],
        )
        summary = report.get_summary()

        assert "Validation Report" in summary
        assert "PASSED" not in summary  # Overall is failed
        assert "FAILED" in summary
        assert "check1" in summary
        assert "check2" in summary
        assert "Warning 1" not in summary  # Warning content not index


class TestDataValidator:
    """Test suite for DataValidator."""

    @pytest.fixture
    def temp_audio_files(self):
        """Create temporary valid audio files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            sample_rate = 48000
            duration = 2.0
            num_samples = int(sample_rate * duration)

            # Create clean input audio
            t = torch.linspace(0, duration, num_samples)
            input_audio = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)

            # Create target audio (attenuated input)
            target_audio = input_audio * 0.5

            input_path = tmpdir / "input.wav"
            target_path = tmpdir / "target.wav"

            torchaudio.save(str(input_path), input_audio, sample_rate)
            torchaudio.save(str(target_path), target_audio, sample_rate)

            yield input_path, target_path

    def test_validator_creation(self):
        """Test DataValidator can be created."""
        validator = DataValidator()
        assert validator is not None
        assert validator.check_clipping is True
        assert validator.check_dc_offset is True

    def test_files_exist_check(self, temp_audio_files):
        """Test files exist check."""
        input_path, target_path = temp_audio_files

        validator = DataValidator()
        report = validator.validate(input_path, target_path)

        assert "files_exist" in report.checks
        assert report.checks["files_exist"].passed is True

    def test_missing_file_check(self):
        """Test validation fails for missing files."""
        validator = DataValidator()
        report = validator.validate(
            "/nonexistent/input.wav",
            "/nonexistent/target.wav",
        )

        assert report.passed is False
        assert "files_exist" in report.checks
        assert report.checks["files_exist"].passed is False

    def test_sample_rate_mismatch(self):
        """Test sample rate mismatch detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create audio at different sample rates
            input_audio = torch.randn(1, 48000)
            target_audio = torch.randn(1, 44100)

            input_path = tmpdir / "input.wav"
            target_path = tmpdir / "target.wav"

            torchaudio.save(str(input_path), input_audio, 48000)
            torchaudio.save(str(target_path), target_audio, 44100)

            validator = DataValidator()
            report = validator.validate(input_path, target_path)

            assert "sample_rate_match" in report.checks
            assert report.checks["sample_rate_match"].passed is False
            assert len(report.warnings) > 0

    def test_clipping_detection(self):
        """Test clipping detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create audio with clipping
            input_audio = torch.ones(1, 10000) * 1.5  # Exceeds 1.0
            target_audio = torch.randn(1, 10000) * 0.5

            input_path = tmpdir / "input.wav"
            target_path = tmpdir / "target.wav"

            torchaudio.save(str(input_path), input_audio, 48000)
            torchaudio.save(str(target_path), target_audio, 48000)

            validator = DataValidator(clipping_threshold=0.99)
            report = validator.validate(input_path, target_path)

            assert "input_clipping" in report.checks
            assert report.checks["input_clipping"].passed is False
            assert len(report.warnings) > 0

    def test_dc_offset_detection(self):
        """Test DC offset detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create audio with DC offset
            input_audio = torch.randn(1, 10000) * 0.1 + 0.5  # DC offset of 0.5
            target_audio = torch.randn(1, 10000) * 0.1

            input_path = tmpdir / "input.wav"
            target_path = tmpdir / "target.wav"

            torchaudio.save(str(input_path), input_audio, 48000)
            torchaudio.save(str(target_path), target_audio, 48000)

            validator = DataValidator(dc_offset_threshold=0.01)
            report = validator.validate(input_path, target_path)

            assert "input_dc_offset" in report.checks
            assert report.checks["input_dc_offset"].passed is False

    def test_length_compatibility(self):
        """Test length compatibility check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create audio with very different lengths
            input_audio = torch.randn(1, 100000)
            target_audio = torch.randn(1, 10000)

            input_path = tmpdir / "input.wav"
            target_path = tmpdir / "target.wav"

            torchaudio.save(str(input_path), input_audio, 48000)
            torchaudio.save(str(target_path), target_audio, 48000)

            validator = DataValidator()
            report = validator.validate(input_path, target_path)

            assert "length_compatible" in report.checks
            assert report.checks["length_compatible"].passed is False
            assert report.passed is False  # Critical check failure

    def test_signal_level_check(self):
        """Test signal level check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create very low signal audio
            input_audio = torch.randn(1, 10000) * 1e-8
            target_audio = torch.randn(1, 10000) * 1e-8

            input_path = tmpdir / "input.wav"
            target_path = tmpdir / "target.wav"

            torchaudio.save(str(input_path), input_audio, 48000)
            torchaudio.save(str(target_path), target_audio, 48000)

            validator = DataValidator()
            report = validator.validate(input_path, target_path)

            assert "signal_level" in report.checks
            assert report.checks["signal_level"].passed is False

    def test_validation_does_not_compare_dry_and_processed_signals(
        self, temp_audio_files
    ):
        input_path, target_path = temp_audio_files

        report = DataValidator().validate(input_path, target_path)

        assert "replicability" not in report.checks

    def test_valid_audio_passes(self, temp_audio_files):
        """Test that valid audio passes all checks."""
        input_path, target_path = temp_audio_files

        validator = DataValidator()
        report = validator.validate(input_path, target_path)

        # Should pass critical checks
        assert report.passed is True
        assert report.checks["files_exist"].passed is True
        assert report.checks["can_load"].passed is True
        assert report.checks["length_compatible"].passed is True
        assert report.checks["signal_level"].passed is True

    def test_clipping_check_disabled(self):
        """Test that clipping check can be disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create clipped audio
            input_audio = torch.ones(1, 10000) * 2.0
            target_audio = torch.randn(1, 10000)

            input_path = tmpdir / "input.wav"
            target_path = tmpdir / "target.wav"

            torchaudio.save(str(input_path), input_audio, 48000)
            torchaudio.save(str(target_path), target_audio, 48000)

            validator = DataValidator(check_clipping=False)
            report = validator.validate(input_path, target_path)

            # Clipping check should not be in report
            assert "input_clipping" not in report.checks
            assert "output_clipping" not in report.checks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
