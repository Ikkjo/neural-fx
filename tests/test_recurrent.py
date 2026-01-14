import torch
import pytest
import sys
import os

# Ensure the package is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neural_fx.config import ModelConfig, LSTMParams, Conv1dConfig
from neural_fx.models.recurrent import NeuralfxLSTM, NeuralfxGRU

class TestRecurrentModels:
    @pytest.fixture
    def lstm_config(self):
        params = LSTMParams(
            hidden_size=20,
            num_layers=1,
            conv1d=Conv1dConfig(filters=16, kernel_size=3, stride=4),
            conditioning_size=2
        )
        return ModelConfig(
            type="lstm",
            params=params,
            input_size=1,
            output_size=1,
            sample_rate=48000
        )

    @pytest.fixture
    def gru_config(self):
        params = LSTMParams(
            hidden_size=20,
            num_layers=1,
            conv1d=None, # No conv
            conditioning_size=0
        )
        return ModelConfig(
            type="gru",
            params=params,
            input_size=1,
            output_size=1
        )

    def test_lstm_forward_shape(self, lstm_config):
        """Test LSTM forward pass output shape with convolution and conditioning."""
        model = NeuralfxLSTM(lstm_config)
        
        # Input: [Batch, Channels, Time]
        # Length 1024 is divisible by stride 4
        x = torch.randn(2, 1, 1024) 
        cond = torch.randn(2, 2) # Conditioning [Batch, C_cond]
        
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
