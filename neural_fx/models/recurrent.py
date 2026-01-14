from typing import Dict, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import ModelConfig, LSTMParams, _load_model_params
from .base import BaseNeuralFXModel

class RecurrentNeuralFXModel(BaseNeuralFXModel):
    """Base class for recurrent neural audio effects (LSTM/GRU)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config.sample_rate)
        self.config = config
        self.params = config.params
        
        # Verify params type
        if not isinstance(self.params, LSTMParams):
             raise ValueError(f"Expected LSTMParams, got {type(self.params)}")

        # Optional Input Convolution (Feature Extraction / Downsampling)
        self.conv1d = None
        self.conv_transpose = None
        rnn_input_size = config.input_size
        
        if self.params.conv1d:
            self.conv1d = nn.Sequential(
                nn.Conv1d(
                    in_channels=config.input_size,
                    out_channels=self.params.conv1d.filters,
                    kernel_size=self.params.conv1d.kernel_size,
                    stride=self.params.conv1d.stride,
                    padding=(self.params.conv1d.kernel_size - 1) // 2 
                ),
                nn.ELU()
            )
            rnn_input_size = self.params.conv1d.filters
            
            # Upsampling if stride > 1
            if self.params.conv1d.stride > 1:
                self.conv_transpose = nn.ConvTranspose1d(
                    in_channels=config.output_size, # Assuming we project to output_size first
                    out_channels=config.output_size,
                    kernel_size=self.params.conv1d.kernel_size,
                    stride=self.params.conv1d.stride,
                    padding=(self.params.conv1d.kernel_size - 1) // 2,
                    output_padding=self.params.conv1d.stride - 1 # Ensure length match
                )

        # Add conditioning channels to RNN input
        rnn_input_size += self.params.conditioning_size

        # Abstract RNN Core
        self.rnn = self._build_rnn(rnn_input_size)
        
        # Output Projection
        # Project hidden_size -> output_size (channels)
        self.fc_out = nn.Linear(self.params.hidden_size, config.output_size)
        
        # Hidden State
        self.hidden_state = None

    def _build_rnn(self, input_size: int) -> nn.Module:
        raise NotImplementedError

    def forward(self, x: Tensor, conditioning: Optional[Tensor] = None) -> Tensor:
        # x: [Batch, Channels, Time]
        identity = x
        
        if self.conv1d:
            x = self.conv1d(x)
        
        # Handle Conditioning
        if self.params.conditioning_size > 0:
            if conditioning is None:
                # Default to zeros if not provided
                conditioning = torch.zeros(x.shape[0], self.params.conditioning_size, device=x.device)
            
            # conditioning: [Batch, C_cond] or [Batch, C_cond, Time]
            if conditioning.ndim == 2:
                 # Expand to time: [Batch, C_cond, 1] -> [Batch, C_cond, Time]
                 cond = conditioning.unsqueeze(2).expand(-1, -1, x.shape[2])
            elif conditioning.ndim == 3:
                 cond = conditioning
                 if cond.shape[2] != x.shape[2]:
                     cond = F.interpolate(cond, size=x.shape[2])
            else:
                raise ValueError(f"Invalid conditioning shape: {conditioning.shape}")
            
            if cond.shape[1] != self.params.conditioning_size:
                 raise ValueError(f"Expected {self.params.conditioning_size} conditioning channels, got {cond.shape[1]}")
                 
            x = torch.cat([x, cond], dim=1)
        
        # Transpose for RNN: [Batch, Time, Channels]
        x = x.transpose(1, 2)
        
        # Run RNN
        x, new_state = self.rnn(x, self.hidden_state)
        self.hidden_state = new_state
        
        # FC
        x = self.fc_out(x)
        
        # Back to [Batch, Channels, Time]
        x = x.transpose(1, 2)
        
        # Upsample if needed
        if self.conv_transpose:
            x = self.conv_transpose(x)
            
        # Skip Connection
        if self.params.skip_connection:
             if x.shape == identity.shape:
                 x = x + identity
        
        return x

    def reset_state(self) -> None:
        self.hidden_state = None

    def detach_state(self) -> None:
        if self.hidden_state is None:
            return
        if isinstance(self.hidden_state, tuple):
            self.hidden_state = tuple(h.detach() for h in self.hidden_state)
        else:
            self.hidden_state = self.hidden_state.detach()

    def process_sample(self, x: Tensor) -> Tensor:
        # x: [Channels] or [Channels, 1]
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(2) # [1, C, 1]
        elif x.ndim == 2:
            x = x.unsqueeze(2) # [B, C, 1]
            
        with torch.no_grad():
            out = self.forward(x)
            
        return out.squeeze()

    @property
    def receptive_field(self) -> int:
        return 1 # Placeholder

    # Export stubs
    def export_onnx(self, path, opset_version=17): pass
    def export_torchscript(self, path): pass
    def export_rtneural(self, path): pass

    @classmethod
    def from_config(cls, config: Union[Dict[str, Any], ModelConfig]) -> "RecurrentNeuralFXModel":
        if isinstance(config, dict):
            # Parse dict to ModelConfig
            model_type = config.get("type", "lstm")
            params_dict = config.get("params", {})
            params = _load_model_params(model_type, params_dict)
            
            config_obj = ModelConfig(
                type=model_type,
                params=params,
                input_size=config.get("input_size", 1),
                output_size=config.get("output_size", 1),
                sample_rate=config.get("sample_rate", 48000)
            )
            config = config_obj
            
        if config.type == "lstm":
            return NeuralfxLSTM(config)
        elif config.type == "gru":
            return NeuralfxGRU(config)
        else:
            raise ValueError(f"Unknown recurrent model type: {config.type}")


class NeuralfxLSTM(RecurrentNeuralFXModel):
    def _build_rnn(self, input_size: int) -> nn.Module:
        return nn.LSTM(
            input_size=input_size,
            hidden_size=self.params.hidden_size,
            num_layers=self.params.num_layers,
            batch_first=True,
            dropout=self.params.dropout
        )
        
    def reset_parameters(self):
        # Forget gate bias = 1
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)

class NeuralfxGRU(RecurrentNeuralFXModel):
    def _build_rnn(self, input_size: int) -> nn.Module:
        return nn.GRU(
            input_size=input_size,
            hidden_size=self.params.hidden_size,
            num_layers=self.params.num_layers,
            batch_first=True,
            dropout=self.params.dropout
        )