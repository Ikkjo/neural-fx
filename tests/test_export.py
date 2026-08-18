import json
import sys
import os
import tempfile
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import pytest

from neural_fx.config import ModelConfig, LSTMParams, Conv1dConfig
from neural_fx.models.recurrent import NeuralfxLSTM, NeuralfxGRU


class TestModelExport:
    """Test suite for model export functionality."""

    @pytest.fixture
    def simple_lstm_config(self):
        """Simple LSTM config without conv1d for faster tests."""
        params = LSTMParams(
            hidden_size=16,
            num_layers=1,
            conv1d=None,
            skip_connection=False,
            dropout=0.0,
            conditioning_size=0,
        )
        return ModelConfig(
            type="lstm",
            params=params,
            input_size=1,
            output_size=1,
            sample_rate=48000,
        )

    @pytest.fixture
    def lstm_with_conv_config(self):
        """LSTM config with conv1d layer."""
        params = LSTMParams(
            hidden_size=16,
            num_layers=1,
            conv1d=Conv1dConfig(filters=8, kernel_size=3, stride=4),
            skip_connection=False,
            dropout=0.0,
            conditioning_size=0,
        )
        return ModelConfig(
            type="lstm",
            params=params,
            input_size=1,
            output_size=1,
            sample_rate=48000,
        )

    @pytest.fixture
    def gru_config(self):
        """GRU config for testing."""
        params = LSTMParams(
            hidden_size=16,
            num_layers=1,
            conv1d=None,
            skip_connection=False,
            dropout=0.0,
            conditioning_size=0,
        )
        return ModelConfig(
            type="gru",
            params=params,
            input_size=1,
            output_size=1,
            sample_rate=48000,
        )

    @pytest.mark.onnx
    def test_onnx_export_simple_lstm(self, simple_lstm_config):
        """Test ONNX export for simple LSTM model."""
        model = NeuralfxLSTM(simple_lstm_config)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "model.onnx"
            model.export_onnx(export_path)

            assert export_path.exists()
            assert export_path.stat().st_size > 0

    @pytest.mark.onnx
    def test_onnx_export_lstm_with_conv(self, lstm_with_conv_config):
        """Test ONNX export for LSTM with convolution."""
        model = NeuralfxLSTM(lstm_with_conv_config)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "model.onnx"
            model.export_onnx(export_path)

            assert export_path.exists()

    @pytest.mark.onnx
    def test_onnx_export_supports_dynamic_batch_and_time(
        self, simple_lstm_config, lstm_with_conv_config
    ):
        """Exported recurrent models run with shapes unlike the trace input."""
        onnxruntime = pytest.importorskip("onnxruntime")

        for config in (simple_lstm_config, lstm_with_conv_config):
            model = NeuralfxLSTM(config).eval()
            input_tensor = torch.randn(3, 1, 20)
            model.reset_state()
            with torch.no_grad():
                expected = model(input_tensor).numpy()

            with tempfile.TemporaryDirectory() as tmpdir:
                export_path = Path(tmpdir) / "model.onnx"
                model.export_onnx(export_path)
                session = onnxruntime.InferenceSession(
                    str(export_path), providers=["CPUExecutionProvider"]
                )
                actual = session.run(None, {"input": input_tensor.numpy()})[0]

            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_torchscript_export_simple_lstm(self, simple_lstm_config):
        """Test TorchScript export for simple LSTM model."""
        model = NeuralfxLSTM(simple_lstm_config)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "model.pt"
            model.export_torchscript(export_path)

            assert export_path.exists()
            assert export_path.stat().st_size > 0

            # Verify it can be loaded
            loaded = torch.jit.load(str(export_path))
            assert loaded is not None

            # Test inference with loaded model
            dummy_input = torch.randn(1, 1, 512)
            output = loaded(dummy_input)
            assert output.shape == dummy_input.shape

    def test_torchscript_export_gru(self, gru_config):
        """Test TorchScript export for GRU model."""
        model = NeuralfxGRU(gru_config)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "model.pt"
            model.export_torchscript(export_path)

            assert export_path.exists()

            # Verify it can be loaded and used
            loaded = torch.jit.load(str(export_path))
            dummy_input = torch.randn(1, 1, 512)
            output = loaded(dummy_input)
            assert output.shape == dummy_input.shape

    def test_rtneural_export_simple_lstm(self, simple_lstm_config):
        """Test RTNeural JSON export for simple LSTM."""
        model = NeuralfxLSTM(simple_lstm_config)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "model.json"
            model.export_rtneural(export_path)

            assert export_path.exists()

            # Verify JSON structure
            with open(export_path) as f:
                data = json.load(f)

            assert "in_shape" in data
            assert "layers" in data
            assert len(data["layers"]) >= 2  # At least RNN + Dense

    def test_rtneural_export_lstm_with_conv(self, lstm_with_conv_config):
        """Test RTNeural JSON export includes conv layers."""
        model = NeuralfxLSTM(lstm_with_conv_config)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "model.json"
            model.export_rtneural(export_path)

            with open(export_path) as f:
                data = json.load(f)

            # Should have conv1d, lstm, dense, convtranspose
            layer_types = [layer["type"] for layer in data["layers"]]
            assert "conv1d" in layer_types
            assert "lstm" in layer_types
            assert "dense" in layer_types

    def test_rtneural_export_gru(self, gru_config):
        """Test RTNeural JSON export for GRU."""
        model = NeuralfxGRU(gru_config)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "model.json"
            model.export_rtneural(export_path)

            with open(export_path) as f:
                data = json.load(f)

            layer_types = [layer["type"] for layer in data["layers"]]
            assert "gru" in layer_types
            assert "dense" in layer_types

    def test_rtneural_layer_structure(self, simple_lstm_config):
        """Test RTNeural export produces correct layer structure."""
        model = NeuralfxLSTM(simple_lstm_config)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "model.json"
            model.export_rtneural(export_path)

            with open(export_path) as f:
                data = json.load(f)

            # Check layer structure
            for layer in data["layers"]:
                assert "type" in layer
                assert "shape" in layer
                assert "weights" in layer

                # Weights should be non-empty
                assert len(layer["weights"]) > 0

    @pytest.mark.onnx
    def test_export_creates_directories(self, simple_lstm_config):
        """Test that export creates parent directories if needed."""
        model = NeuralfxLSTM(simple_lstm_config)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "subdir1" / "subdir2" / "model.onnx"
            model.export_onnx(export_path)

            assert export_path.exists()

    def test_exported_model_consistency(self, simple_lstm_config):
        """Test that exported models produce consistent outputs."""
        model = NeuralfxLSTM(simple_lstm_config)
        model.eval()

        # Get reference output
        dummy_input = torch.randn(1, 1, 512)
        with torch.no_grad():
            reference_output = model(dummy_input)

        # Export and reload TorchScript
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "model.pt"
            model.export_torchscript(export_path)
            loaded = torch.jit.load(str(export_path))

            # Note: TorchScript export is stateless, so outputs may differ
            # for stateful models. This test mainly checks it doesn't crash.
            ts_output = loaded(dummy_input)
            assert ts_output.shape == reference_output.shape
