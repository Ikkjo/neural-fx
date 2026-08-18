import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torchaudio

from neural_fx.config import LSTMParams, ModelConfig
from neural_fx.inference.streaming import (
    StreamingProcessor,
    evaluate_model,
    load_audio,
    process_audio,
    save_audio,
)
from neural_fx.models.recurrent import NeuralfxLSTM


class TestStreamingInference:
    """Test suite for streaming inference functionality."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple LSTM model for testing."""
        config = ModelConfig(
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
        model = NeuralfxLSTM(config)
        model.eval()
        return model

    @pytest.fixture
    def temp_audio_file(self):
        """Create a temporary audio file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_rate = 48000
            duration = 1.0  # 1 second
            audio = torch.randn(1, int(sample_rate * duration))

            audio_path = Path(tmpdir) / "test_audio.wav"
            torchaudio.save(str(audio_path), audio, sample_rate)

            yield audio_path, sample_rate

    def test_streaming_processor_creation(self, simple_model):
        """Test StreamingProcessor can be created."""
        processor = StreamingProcessor(simple_model)
        assert processor.model is not None
        assert processor.sample_rate == 48000

    def test_streaming_processor_warns_for_conflicting_rate(self, simple_model):
        """Warn and retain the model rate when an inference override conflicts."""
        with pytest.warns(UserWarning, match="using the model sample rate"):
            processor = StreamingProcessor(simple_model, sample_rate=44100)

        assert processor.sample_rate == 48000

    def test_streaming_processor_process_sample(self, simple_model):
        """Test processing single samples."""
        processor = StreamingProcessor(simple_model, sample_rate=48000)

        # Process a few samples
        for i in range(10):
            input_sample = 0.5 * torch.sin(torch.tensor(0.1 * i))
            output_sample = processor.process_sample(float(input_sample))
            assert isinstance(output_sample, float)

    def test_streaming_processor_process_block(self, simple_model):
        """Test processing blocks of samples."""
        processor = StreamingProcessor(simple_model, sample_rate=48000)

        block = torch.randn(1, 1, 512)
        output = processor.process_block(block)

        assert output.shape == block.shape

    def test_streaming_processor_reset(self, simple_model):
        """Test that reset clears model state."""
        processor = StreamingProcessor(simple_model, sample_rate=48000)

        # Process some samples
        for _ in range(10):
            processor.process_sample(0.5)

        # Reset
        processor.reset()

        # State should be cleared
        assert simple_model.hidden_state is None

    def test_streaming_processor_stateful(self, simple_model):
        """Test that processor maintains state between calls."""
        processor = StreamingProcessor(simple_model, sample_rate=48000)

        # First call
        block1 = torch.randn(1, 1, 256)
        out1 = processor.process_block(block1)

        # State should be populated
        assert simple_model.hidden_state is not None

        # Second call - should use previous state
        block2 = torch.randn(1, 1, 256)
        out2 = processor.process_block(block2)

        # Outputs should be different tensors
        assert out1 is not out2

    def test_streaming_processor_accepts_conditioning(self):
        """Block and sample streaming expose model conditioning controls."""
        config = ModelConfig(
            type="lstm",
            params=LSTMParams(hidden_size=4, conditioning_size=1),
            input_size=1,
            output_size=1,
        )
        processor = StreamingProcessor(NeuralfxLSTM(config).eval())

        block = processor.process_block(
            torch.randn(1, 1, 8), conditioning=torch.tensor([[0.5]])
        )
        processor.reset()
        sample = processor.process_sample(0.1, conditioning=0.5)

        assert block.shape == (1, 1, 8)
        assert isinstance(sample, float)

    def test_load_audio(self, temp_audio_file):
        """Test loading audio from file."""
        audio_path, sample_rate = temp_audio_file

        audio = load_audio(audio_path, sample_rate=sample_rate)

        assert audio.ndim == 2  # [channels, samples]
        assert audio.shape[0] == 1  # Mono
        assert audio.shape[1] > 0

    def test_load_audio_resampling(self, temp_audio_file):
        """Test that audio is resampled to target rate."""
        audio_path, _ = temp_audio_file

        # Load at different sample rate
        target_sr = 16000
        audio = load_audio(audio_path, sample_rate=target_sr)

        # Should not crash, audio should be resampled
        assert audio.ndim == 2

    def test_save_audio(self):
        """Test saving audio to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = torch.randn(1, 48000)
            output_path = Path(tmpdir) / "output.wav"

            save_audio(audio, output_path, sample_rate=48000)

            assert output_path.exists()

            # Verify it can be loaded back
            loaded = load_audio(output_path, sample_rate=48000)
            assert loaded.shape == audio.shape

    def test_process_audio(self, simple_model, temp_audio_file):
        """Test processing audio file through model."""
        input_path, sample_rate = temp_audio_file

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "processed.wav"

            output = process_audio(
                model=simple_model,
                input_path=input_path,
                output_path=output_path,
                sample_rate=sample_rate,
                chunk_size=4096,
            )

            assert output_path.exists()
            assert output.ndim == 3  # [batch, channels, time]

    def test_process_audio_uses_model_sample_rate(self):
        """Processing resamples and saves using a non-default model rate."""
        config = ModelConfig(
            type="lstm",
            params=LSTMParams(hidden_size=4, num_layers=1),
            input_size=1,
            output_size=1,
            sample_rate=44100,
        )
        model = NeuralfxLSTM(config).eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.wav"
            output_path = tmpdir / "output.wav"
            torchaudio.save(str(input_path), torch.randn(1, 4800), 48000)

            output = process_audio(model, input_path, output_path, chunk_size=1024)
            output_info = torchaudio.info(str(output_path))

            assert output_info.sample_rate == 44100
            assert output.shape[-1] == 4410

    def test_evaluate_model(self, simple_model, temp_audio_file):
        """Test model evaluation against target."""
        input_path, _ = temp_audio_file

        # Use same file as target (not realistic but good for testing)
        metrics = evaluate_model(
            model=simple_model,
            input_path=input_path,
            target_path=input_path,
            burn_in=0,
        )

        assert "mse" in metrics
        assert "esr" in metrics
        assert metrics["mse"] >= 0
        assert metrics["esr"] >= 0

    def test_evaluate_model_with_burn_in(self, simple_model, temp_audio_file):
        """Test model evaluation with burn-in period."""
        input_path, sample_rate = temp_audio_file

        metrics = evaluate_model(
            model=simple_model,
            input_path=input_path,
            target_path=input_path,
            sample_rate=sample_rate,
            burn_in=4096,
        )

        assert "mse" in metrics
        assert "esr" in metrics

    def test_load_audio_file_not_found(self):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/audio.wav", sample_rate=48000)

    def test_process_audio_chunked(self, simple_model):
        """Test that audio is processed in chunks correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create longer audio file
            sample_rate = 48000
            duration = 3.0  # 3 seconds
            audio = torch.randn(1, int(sample_rate * duration))

            input_path = tmpdir / "input.wav"
            output_path = tmpdir / "output.wav"
            torchaudio.save(str(input_path), audio, sample_rate)

            # Process with small chunks
            output = process_audio(
                model=simple_model,
                input_path=input_path,
                output_path=output_path,
                sample_rate=sample_rate,
                chunk_size=2048,  # Small chunks
            )

            # Output should have same length as input
            assert output.shape[-1] == audio.shape[-1]
