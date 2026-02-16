import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import LSTMParams, ModelConfig, _load_model_params
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
                    padding=(self.params.conv1d.kernel_size - 1) // 2,
                ),
                nn.ELU(),
            )
            rnn_input_size = self.params.conv1d.filters

            # projection layer to ensure shape matches if needed
            self.skip_projection = None
            if self.params.skip_connection:
                needs_projection = (
                    self.params.conv1d.stride > 1
                    or config.input_size != config.output_size
                )
                if needs_projection:
                    self.skip_projection = nn.Conv1d(
                        config.input_size, config.output_size, 1
                    )

            # Upsampling if stride > 1
            if self.params.conv1d.stride > 1:
                self.conv_transpose = nn.ConvTranspose1d(
                    in_channels=config.output_size,  # After fc_out projection
                    out_channels=config.output_size,
                    kernel_size=self.params.conv1d.kernel_size,
                    stride=self.params.conv1d.stride,
                    padding=(self.params.conv1d.kernel_size - 1) // 2,
                    output_padding=self.params.conv1d.stride - 1,  # Ensure length match
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

    def forward(
        self,
        x: Tensor,
        conditioning: Optional[Tensor] = None,
        reset_state=False,
        detach_state=False,
    ) -> Tensor:
        # x: [Batch, Channels, Time]
        identity = x

        if reset_state:
            self.reset_state()
        if detach_state and self.hidden_state is not None:
            self.detach_state()

        if self.conv1d:
            x = self.conv1d(x)

        # Handle Conditioning
        if self.params.conditioning_size > 0:
            if conditioning is None:
                # Default to zeros if not provided
                conditioning = torch.zeros(
                    x.shape[0], self.params.conditioning_size, device=x.device
                )

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
                raise ValueError(
                    f"Expected {self.params.conditioning_size} conditioning channels, got {cond.shape[1]}"
                )

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
            skip = self.skip_projection(identity) if self.skip_projection else identity
            if x.shape == skip.shape:
                x = x + skip

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

    def process_sample(self, x: Tensor, reset=False) -> Tensor:
        # x: [Channels] or [Channels, 1]
        if reset:
            self.reset_state()

        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(2)  # [1, C, 1]
        elif x.ndim == 2:
            x = x.unsqueeze(2)  # [B, C, 1]

        with torch.no_grad():
            out = self.forward(x)

        return out.squeeze()

    @property
    def receptive_field(self) -> int:
        rf = 1
        if self.params.conv1d:
            rf = self.params.conv1d.kernel_size
            if self.params.conv1d.stride > 1:
                rf = rf * self.params.conv1d.stride  # Approximate
        return rf

    def export_onnx(self, path: str | Path, opset_version: int = 17) -> None:
        """Export model to ONNX format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.eval()
        self.reset_state()

        dummy_input = torch.randn(1, self.config.input_size, 512)
        if self.params.conditioning_size > 0:
            dummy_cond = torch.randn(1, self.params.conditioning_size)
        else:
            dummy_cond = None

        input_names = ["input"]
        output_names = ["output"]
        dynamic_axes = {
            "input": {0: "batch_size", 2: "time"},
            "output": {0: "batch_size", 2: "time"},
        }

        args = (dummy_input, dummy_cond) if dummy_cond is not None else (dummy_input,)

        torch.onnx.export(
            self,
            args,
            path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

    def export_torchscript(self, path: str | Path) -> None:
        """Export model to TorchScript format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.eval()

        # Create a stateless wrapper for export
        class TorchScriptWrapper(torch.nn.Module):
            def __init__(self, model: "RecurrentNeuralFXModel"):
                super().__init__()
                self.conv1d = model.conv1d
                self.conv_transpose = model.conv_transpose
                self.rnn = model.rnn
                self.fc_out = model.fc_out
                self.skip_projection = getattr(model, "skip_projection", None)
                self.params = model.params
                self.config = model.config

            def forward(self, x: Tensor) -> Tensor:
                identity = x

                if self.conv1d is not None:
                    x = self.conv1d(x)

                x = x.transpose(1, 2)
                x, _ = self.rnn(x, None)  # stateless - always pass None
                x = self.fc_out(x)
                x = x.transpose(1, 2)

                if self.conv_transpose is not None:
                    x = self.conv_transpose(x)

                if self.params.skip_connection:
                    skip = (
                        self.skip_projection(identity)
                        if self.skip_projection
                        else identity
                    )
                    if x.shape == skip.shape:
                        x = x + skip

                return x

        wrapper = TorchScriptWrapper(self)

        # Use tracing for stateless export
        dummy_input = torch.randn(1, self.config.input_size, 512)
        scripted = torch.jit.trace(wrapper, dummy_input)
        scripted.save(str(path))

    def export_rtneural(self, path: str | Path) -> None:
        """Export model to RTNeural JSON format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.eval()

        model_dict: dict[str, Any] = {
            "in_shape": [None, None, self.config.input_size],
            "layers": [],
        }

        # Add Conv1D layer if present
        if self.conv1d is not None:
            conv_layer = self._export_rtneural_conv1d()
            model_dict["layers"].append(conv_layer)

        # Add RNN layers
        rnn_type = "lstm" if self.config.type == "lstm" else "gru"
        for layer_idx in range(self.params.num_layers):
            rnn_layer = self._export_rtneural_rnn(layer_idx, rnn_type)
            model_dict["layers"].append(rnn_layer)

        # Add Dense output layer
        dense_layer = self._export_rtneural_dense()
        model_dict["layers"].append(dense_layer)

        # Add ConvTranspose if present
        if self.conv_transpose is not None:
            convt_layer = self._export_rtneural_convtranspose()
            model_dict["layers"].append(convt_layer)

        with open(path, "w") as f:
            json.dump(model_dict, f, indent=2)

    def _export_rtneural_conv1d(self) -> dict[str, Any]:
        conv = self.conv1d[0]  # Get Conv1d from Sequential
        activation = self.conv1d[1]  # Get activation

        layer_dict: dict[str, Any] = {
            "type": "conv1d",
            "activation": "elu" if isinstance(activation, nn.ELU) else "tanh",
            "shape": [None, None, conv.out_channels],
            "kernel_size": conv.kernel_size[0],
            "dilation": conv.dilation[0],
            "groups": conv.groups,
            "weights": [],
        }

        # Conv1d weights: [out_channels, in_channels, kernel_size]
        # Transpose to [kernel_size, in_channels, out_channels] for RTNeural
        weight = conv.weight.data
        weight_rtneural = weight.permute(2, 1, 0).cpu().numpy().tolist()
        bias = (
            conv.bias.data.cpu().numpy().tolist()
            if conv.bias is not None
            else [0.0] * conv.out_channels
        )

        layer_dict["weights"] = [weight_rtneural, bias]
        return layer_dict

    def _export_rtneural_rnn(self, layer_idx: int, rnn_type: str) -> dict[str, Any]:
        layer_dict: dict[str, Any] = {
            "type": rnn_type,
            "activation": "tanh",
            "shape": [None, None, self.params.hidden_size],
            "weights": [],
        }

        # Get weights for this layer
        weight_ih = getattr(self.rnn, f"weight_ih_l{layer_idx}").data
        weight_hh = getattr(self.rnn, f"weight_hh_l{layer_idx}").data
        bias_ih = getattr(self.rnn, f"bias_ih_l{layer_idx}").data
        bias_hh = getattr(self.rnn, f"bias_hh_l{layer_idx}").data

        # For LSTM: PyTorch uses [4 * hidden_size, input_size] for weight_ih
        # RTNeural expects split weights for each gate
        hidden_size = self.params.hidden_size

        if rnn_type == "lstm":
            # LSTM gates: input, forget, cell, output
            # PyTorch order: i, f, g, o
            # RTNeural expects: W_ih, W_hh, b_ih, b_hh for each gate
            weights = []
            for gate_idx, gate_name in enumerate(["i", "f", "g", "o"]):
                start = gate_idx * hidden_size
                end = (gate_idx + 1) * hidden_size

                W_ih = weight_ih[start:end, :].cpu().numpy().tolist()
                W_hh = weight_hh[start:end, :].cpu().numpy().tolist()
                b_ih = bias_ih[start:end].cpu().numpy().tolist()
                b_hh = bias_hh[start:end].cpu().numpy().tolist()

                weights.extend([W_ih, W_hh, b_ih, b_hh])

            layer_dict["weights"] = weights
        else:
            # GRU gates: reset, update, new
            # PyTorch order: r, z, n
            weights = []
            for gate_idx in range(3):
                start = gate_idx * hidden_size
                end = (gate_idx + 1) * hidden_size

                W_ih = weight_ih[start:end, :].cpu().numpy().tolist()
                W_hh = weight_hh[start:end, :].cpu().numpy().tolist()
                b_ih = bias_ih[start:end].cpu().numpy().tolist()
                b_hh = bias_hh[start:end].cpu().numpy().tolist()

                weights.extend([W_ih, W_hh, b_ih, b_hh])

            layer_dict["weights"] = weights

        return layer_dict

    def _export_rtneural_dense(self) -> dict[str, Any]:
        layer_dict: dict[str, Any] = {
            "type": "dense",
            "activation": "",
            "shape": [None, None, self.config.output_size],
            "weights": [],
        }

        # Linear layer weights: [out_features, in_features]
        # RTNeural expects: W^T, bias
        weight = self.fc_out.weight.data
        bias = (
            self.fc_out.bias.data
            if self.fc_out.bias is not None
            else torch.zeros(self.config.output_size)
        )

        weight_rtneural = weight.cpu().numpy().tolist()
        bias_rtneural = bias.cpu().numpy().tolist()

        layer_dict["weights"] = [weight_rtneural, bias_rtneural]
        return layer_dict

    def _export_rtneural_convtranspose(self) -> dict[str, Any]:
        """Export ConvTranspose1d as Conv1d for RTNeural compatibility.

        RTNeural doesn't have a native ConvTranspose1d layer, so we export
        it as a Conv1d layer with appropriate weight permutation.
        """
        layer_dict: dict[str, Any] = {
            "type": "conv1d",
            "activation": "",
            "shape": [None, None, self.conv_transpose.out_channels],
            "kernel_size": self.conv_transpose.kernel_size[0],
            "dilation": 1,
            "groups": 1,
            "weights": [],
        }

        weight = self.conv_transpose.weight.data
        weight_rtneural = weight.permute(2, 1, 0).cpu().numpy().tolist()
        bias = (
            self.conv_transpose.bias.data.cpu().numpy().tolist()
            if self.conv_transpose.bias is not None
            else [0.0] * self.conv_transpose.out_channels
        )

        layer_dict["weights"] = [weight_rtneural, bias]
        return layer_dict

    @classmethod
    def from_config(
        cls, config: Union[Dict[str, Any], ModelConfig]
    ) -> "RecurrentNeuralFXModel":
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
                sample_rate=config.get("sample_rate", 48000),
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
            dropout=self.params.dropout,
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
            dropout=self.params.dropout,
        )

    def reset_parameters(self):
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                n = param.size(0)
                start, end = n // 3, 2 * n // 3
                param.data[start:end].fill_(1.0)
