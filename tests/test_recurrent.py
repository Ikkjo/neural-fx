import copy
import os
import sys

import pytest
import torch

from neural_fx.config import Conv1dConfig, LSTMParams, ModelConfig
from neural_fx.models.recurrent import NeuralfxGRU, NeuralfxLSTM

# Ensure the package is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestRecurrentModels:
    @pytest.fixture
    def lstm_config(self):
        params = LSTMParams(
            hidden_size=20,
            num_layers=1,
            conv1d=Conv1dConfig(filters=16, kernel_size=3, stride=4),
            conditioning_size=2,
        )
        return ModelConfig(
            type="lstm", params=params, input_size=1, output_size=1, sample_rate=48000
        )

    @pytest.fixture
    def gru_config(self):
        params = LSTMParams(
            hidden_size=20,
            num_layers=1,
            conv1d=None,  # No conv
            conditioning_size=0,
        )
        return ModelConfig(type="gru", params=params, input_size=1, output_size=1)

    def test_lstm_forward_shape(self, lstm_config):
        """Test LSTM forward pass output shape with convolution and conditioning."""
        model = NeuralfxLSTM(lstm_config)

        # Input: [Batch, Channels, Time]
        # Length 1024 is divisible by stride 4
        x = torch.randn(2, 1, 1024)
        cond = torch.randn(2, 2)  # Conditioning [Batch, C_cond]

        y = model(x, conditioning=cond)

        assert y.shape == x.shape, f"Output shape mismatch! {y.shape} != {x.shape}"

    def test_gru_forward_shape(self, gru_config):
        """Test GRU forward pass output shape without convolution."""
        model = NeuralfxGRU(gru_config)

        x = torch.randn(2, 1, 100)
        y = model(x)

        assert y.shape == x.shape

    def test_state_management(self, lstm_config):
        """Test resetting and detaching state."""
        model = NeuralfxLSTM(lstm_config)
        x = torch.randn(1, 1, 100)

        # Run once to populate state
        model(x)
        assert model.hidden_state is not None

        # Detach
        model.detach_state()
        if isinstance(model.hidden_state, tuple):
            assert not model.hidden_state[0].requires_grad
        else:
            assert not model.hidden_state.requires_grad

        # Reset
        model.reset_state()
        assert model.hidden_state is None

    def test_process_sample(self, lstm_config):
        """Test single sample processing."""
        model = NeuralfxLSTM(lstm_config)
        model.eval()

        # Single sample input [Channels]
        x = torch.randn(1)
        y = model.process_sample(x)

        assert y.ndim == 0 or y.ndim == 1
        assert model.hidden_state is not None

    def test_causal_conv_matches_block_and_sample_streaming(self):
        """Causal strided convolution is independent of future block layout."""
        config = ModelConfig(
            type="lstm",
            params=LSTMParams(
                hidden_size=6,
                num_layers=1,
                conv1d=Conv1dConfig(filters=4, kernel_size=3, stride=4),
                conditioning_size=1,
            ),
            input_size=1,
            output_size=1,
        )
        whole_model = NeuralfxLSTM(config).eval()
        block_model = copy.deepcopy(whole_model)
        sample_model = copy.deepcopy(whole_model)
        future_model = copy.deepcopy(whole_model)
        x = torch.randn(1, 1, 23)
        changed_future = x.clone()
        changed_future[..., 12:] += 10
        conditioning = torch.tensor([[0.25]])

        with torch.no_grad():
            whole = whole_model(x, conditioning=conditioning)
            blocks = torch.cat(
                [
                    block_model(x[..., :5], conditioning=conditioning),
                    block_model(x[..., 5:12], conditioning=conditioning),
                    block_model(x[..., 12:], conditioning=conditioning),
                ],
                dim=-1,
            )
            samples = torch.cat(
                [
                    sample_model.process_sample(
                        x[..., index], conditioning=conditioning
                    ).reshape(1, 1, 1)
                    for index in range(x.shape[-1])
                ],
                dim=-1,
            )
            future = future_model(changed_future, conditioning=conditioning)

        torch.testing.assert_close(blocks, whole)
        torch.testing.assert_close(samples, whole)
        torch.testing.assert_close(future[..., :12], whole[..., :12])

    def test_skip_projection_and_recurrent_bias_initialization(self):
        """Construction initializes skip projection and recurrent gate biases."""
        skip_config = ModelConfig(
            type="lstm",
            params=LSTMParams(hidden_size=4, skip_connection=True),
            input_size=2,
            output_size=1,
        )
        lstm = NeuralfxLSTM(skip_config)
        gru = NeuralfxGRU(
            ModelConfig(
                type="gru",
                params=LSTMParams(hidden_size=4),
                input_size=1,
                output_size=1,
            )
        )

        assert lstm.skip_projection is not None
        assert lstm(torch.randn(1, 2, 8), reset_state=True).shape == (1, 1, 8)
        for name, bias in lstm.rnn.named_parameters():
            if "bias" in name:
                torch.testing.assert_close(bias[4:8], torch.ones(4))
        for name, bias in gru.rnn.named_parameters():
            if "bias" in name:
                torch.testing.assert_close(bias[4:8], torch.ones(4))
