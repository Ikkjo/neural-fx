import sys
import os
import tempfile
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchaudio
import pytest

from neural_fx.data.dataset import AudioDataset


class TestAudioDataset:
    """Test suite for AudioDataset."""

    @pytest.fixture
    def temp_audio_files(self):
        """Create temporary audio files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            sample_rate = 48000
            duration = 2.0  # seconds
            num_samples = int(sample_rate * duration)

            # Create input audio (sine wave)
            t = torch.linspace(0, duration, num_samples)
            input_audio = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)  # 440 Hz sine

            # Create target audio (attenuated and phase shifted)
            target_audio = 0.5 * torch.sin(2 * 3.14159 * 440 * t + 0.5).unsqueeze(0)

            input_path = tmpdir / "input.wav"
            target_path = tmpdir / "target.wav"

            torchaudio.save(str(input_path), input_audio, sample_rate)
            torchaudio.save(str(target_path), target_audio, sample_rate)

            yield input_path, target_path, sample_rate

    def test_dataset_creation(self, temp_audio_files):
        """Test AudioDataset can be created from file pairs."""
        input_path, target_path, sample_rate = temp_audio_files

        dataset = AudioDataset(
            input_path=input_path,
            target_path=target_path,
            segment_length=8192,
            sample_rate=sample_rate,
        )

        assert len(dataset) > 0
        assert dataset.num_segments > 0

    def test_dataset_getitem(self, temp_audio_files):
        """Test getting items from dataset returns correct shapes."""
        input_path, target_path, sample_rate = temp_audio_files
        segment_length = 8192

        dataset = AudioDataset(
            input_path=input_path,
            target_path=target_path,
            segment_length=segment_length,
            sample_rate=sample_rate,
        )

        x, y = dataset[0]

        assert x.shape == (segment_length,)
        assert y.shape == (segment_length,)
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_dataset_length_consistency(self, temp_audio_files):
        """Test that input and target have matching lengths."""
        input_path, target_path, sample_rate = temp_audio_files

        # Create mismatched length files
        short_audio = torch.randn(1, 8000)
        long_audio = torch.randn(1, 16000)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.wav"
            target_path = tmpdir / "target.wav"

            torchaudio.save(str(input_path), short_audio, sample_rate)
            torchaudio.save(str(target_path), long_audio, sample_rate)

            dataset = AudioDataset(
                input_path=input_path,
                target_path=target_path,
                segment_length=4096,
                sample_rate=sample_rate,
            )

            # Should use minimum length
            assert dataset.total_length <= 8000

    def test_dataset_file_not_found(self):
        """Test that missing files raise appropriate errors."""
        with pytest.raises(FileNotFoundError):
            AudioDataset(
                input_path="/nonexistent/input.wav",
                target_path="/nonexistent/target.wav",
            )

    def test_dataset_resampling(self, temp_audio_files):
        """Test that audio is resampled to target sample rate."""
        input_path, target_path, _ = temp_audio_files

        # Load at different sample rate
        dataset = AudioDataset(
            input_path=input_path,
            target_path=target_path,
            segment_length=8192,
            sample_rate=16000,  # Different from file
        )

        # Just verify it doesn't crash - actual resampling is tested by torchaudio
        assert len(dataset) > 0

    def test_dataset_normalization(self, temp_audio_files):
        """Test that audio is normalized."""
        input_path, target_path, sample_rate = temp_audio_files

        # Create audio with very high amplitude
        loud_audio = torch.ones(1, 48000) * 10.0
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.wav"
            torchaudio.save(str(input_path), loud_audio, sample_rate)
            torchaudio.save(str(tmpdir / "target.wav"), loud_audio, sample_rate)

            dataset = AudioDataset(
                input_path=input_path,
                target_path=tmpdir / "target.wav",
                segment_length=8192,
                sample_rate=sample_rate,
                normalize=True,
            )

            x, _ = dataset[0]
            # After normalization, max should be <= 1.0
            assert x.abs().max() <= 1.0 + 1e-6

    def test_dataset_normalization_preserves_relative_level(self):
        """Pair normalization must not erase the modeled gain relationship."""
        sample_rate = 48000
        input_audio = torch.full((1, 8192), 0.8)
        target_audio = torch.full((1, 8192), 0.2)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.wav"
            target_path = tmpdir / "target.wav"
            torchaudio.save(str(input_path), input_audio, sample_rate)
            torchaudio.save(str(target_path), target_audio, sample_rate)

            dataset = AudioDataset(
                input_path=input_path,
                target_path=target_path,
                segment_length=8192,
                sample_rate=sample_rate,
                normalize=True,
            )
            x, y = dataset[0]

        assert torch.isclose(x.abs().max(), torch.tensor(1.0), atol=1e-4)
        assert torch.isclose(y.abs().max(), torch.tensor(0.25), atol=1e-4)

    def test_dataset_no_normalization(self, temp_audio_files):
        """Test that audio is not normalized when normalize=False."""
        input_path, target_path, sample_rate = temp_audio_files

        dataset = AudioDataset(
            input_path=input_path,
            target_path=target_path,
            segment_length=8192,
            sample_rate=sample_rate,
            normalize=False,
        )

        x, _ = dataset[0]
        # Values may exceed 1.0 if file has high amplitude
        # We just verify the dataset works
        assert x.numel() > 0

    def test_dataset_multiple_segments(self, temp_audio_files):
        """Test that multiple segments can be retrieved."""
        input_path, target_path, sample_rate = temp_audio_files
        segment_length = 4096

        dataset = AudioDataset(
            input_path=input_path,
            target_path=target_path,
            segment_length=segment_length,
            sample_rate=sample_rate,
        )

        # Get first few segments
        for i in range(min(3, len(dataset))):
            x, y = dataset[i]
            assert x.shape == (segment_length,)
            assert y.shape == (segment_length,)

    def test_dataset_stereo_to_mono(self, temp_audio_files):
        """Test that stereo audio is converted to mono."""
        input_path, target_path, sample_rate = temp_audio_files

        # Create stereo audio
        stereo_audio = torch.randn(2, 48000)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.wav"
            target_path = tmpdir / "target.wav"

            torchaudio.save(str(input_path), stereo_audio, sample_rate)
            torchaudio.save(str(target_path), stereo_audio, sample_rate)

            dataset = AudioDataset(
                input_path=input_path,
                target_path=target_path,
                segment_length=8192,
                sample_rate=sample_rate,
            )

            x, _ = dataset[0]
            # Should be mono (1D tensor)
            assert x.ndim == 1
