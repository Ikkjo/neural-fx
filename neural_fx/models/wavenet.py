"""Causal WaveNet model with vectorized and cached streaming inference paths."""

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig, WaveNetParams, _load_model_params
from .base import BaseNeuralFXModel
from .errors import UnsupportedExportFormatError


class CausalConv1d(nn.Conv1d):
    """A stride-one convolution padded only on the left.

    The regular ``forward`` path is vectorized. ``process_sample`` uses a
    fixed-size circular cache, avoiding receptive-field recomputation during
    real-time inference.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=bias,
        )
        self.left_padding = dilation * (kernel_size - 1)
        self.register_buffer("_stream_buffer", torch.empty(0), persistent=False)
        self._stream_index = 0

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))

    def _ensure_stream_buffer(self, x: Tensor) -> None:
        buffer_length = self.left_padding + 1
        expected_shape = (x.shape[0], self.in_channels, buffer_length)
        if (
            tuple(self._stream_buffer.shape) != expected_shape
            or self._stream_buffer.device != x.device
            or self._stream_buffer.dtype != x.dtype
        ):
            self._stream_buffer = x.new_zeros(expected_shape)
            self._stream_index = 0

    @torch.no_grad()
    def process_sample(self, x: Tensor) -> Tensor:
        """Process ``[batch, channels]`` using a circular history cache."""
        if x.ndim != 2 or x.shape[1] != self.in_channels:
            raise ValueError(
                "Expected sample shape [batch, "
                f"{self.in_channels}], got {tuple(x.shape)}"
            )

        self._ensure_stream_buffer(x)
        self._stream_buffer[:, :, self._stream_index] = x

        dilation = self.dilation[0]
        kernel_size = self.kernel_size[0]
        buffer_length = self._stream_buffer.shape[-1]
        indices = [
            (self._stream_index - dilation * (kernel_size - 1 - offset))
            % buffer_length
            for offset in range(kernel_size)
        ]
        window = self._stream_buffer[:, :, indices]
        output = torch.einsum("ock,bck->bo", self.weight, window)
        if self.bias is not None:
            output = output + self.bias

        self._stream_index = (self._stream_index + 1) % buffer_length
        return output

    def reset_state(self) -> None:
        self._stream_buffer = self.weight.new_empty(0)
        self._stream_index = 0

    def detach_state(self) -> None:
        self._stream_buffer = self._stream_buffer.detach()


class DilatedResidualBlock(nn.Module):
    """WaveNet gated activation with residual and skip projections."""

    def __init__(
        self,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.dilation = dilation
        self.filter_conv = CausalConv1d(
            residual_channels,
            dilation_channels,
            kernel_size,
            dilation=dilation,
        )
        self.gate_conv = CausalConv1d(
            residual_channels,
            dilation_channels,
            kernel_size,
            dilation=dilation,
        )
        self.residual_projection = nn.Conv1d(
            dilation_channels, residual_channels, kernel_size=1
        )
        self.skip_projection = nn.Conv1d(
            dilation_channels, skip_channels, kernel_size=1
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        gated = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        return x + self.residual_projection(gated), self.skip_projection(gated)

    @staticmethod
    def _pointwise_sample(layer: nn.Conv1d, x: Tensor) -> Tensor:
        return F.linear(x, layer.weight[:, :, 0], layer.bias)

    def process_sample(self, x: Tensor) -> tuple[Tensor, Tensor]:
        filtered = self.filter_conv.process_sample(x)
        gated = self.gate_conv.process_sample(x)
        activation = torch.tanh(filtered) * torch.sigmoid(gated)
        residual = x + self._pointwise_sample(self.residual_projection, activation)
        skip = self._pointwise_sample(self.skip_projection, activation)
        return residual, skip

    def reset_state(self) -> None:
        self.filter_conv.reset_state()
        self.gate_conv.reset_state()

    def detach_state(self) -> None:
        self.filter_conv.detach_state()
        self.gate_conv.detach_state()


class WaveNetModel(BaseNeuralFXModel):
    """Causal WaveNet for sample-aligned neural audio effects.

    Each stack repeats the dilation cycle ``1, 2, 4, ...,
    2**(layers - 1)``. The receptive field is therefore
    ``1 + stacks * (kernel_size - 1) * (2**layers - 1)`` samples.
    """

    model_type = "wavenet"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config.sample_rate)
        if config.type != "wavenet":
            raise ValueError(f"Expected model type 'wavenet', got {config.type!r}")
        if not isinstance(config.params, WaveNetParams):
            raise ValueError(f"Expected WaveNetParams, got {type(config.params)}")

        self.config = config
        self.params = config.params
        self.input_projection = nn.Conv1d(
            config.input_size, self.params.residual_channels, kernel_size=1
        )

        dilations = [
            2**layer
            for _stack in range(self.params.stacks)
            for layer in range(self.params.layers)
        ]
        self.blocks = nn.ModuleList(
            [
                DilatedResidualBlock(
                    residual_channels=self.params.residual_channels,
                    dilation_channels=self.params.dilation_channels,
                    skip_channels=self.params.skip_channels,
                    kernel_size=self.params.kernel_size,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )
        self.output_projection = nn.Conv1d(
            self.params.skip_channels, self.params.skip_channels, kernel_size=1
        )
        self.output = nn.Conv1d(
            self.params.skip_channels, config.output_size, kernel_size=1
        )
        self.register_buffer("_input_history", torch.empty(0), persistent=False)

    @property
    def receptive_field(self) -> int:
        return 1 + self.params.stacks * (self.params.kernel_size - 1) * (
            2**self.params.layers - 1
        )

    @staticmethod
    def _pointwise_sample(layer: nn.Conv1d, x: Tensor) -> Tensor:
        return F.linear(x, layer.weight[:, :, 0], layer.bias)

    def _forward_stateless(self, x: Tensor) -> Tensor:
        residual = self.input_projection(x)
        skip_total: Tensor | None = None
        for block in self.blocks:
            residual, skip = block(residual)
            skip_total = skip if skip_total is None else skip_total + skip

        if skip_total is None:
            raise RuntimeError("WaveNet requires at least one residual block")
        output = F.relu(skip_total)
        output = F.relu(self.output_projection(output))
        return self.output(output)

    def forward(
        self, x: Tensor, reset_state: bool = False, detach_state: bool = False
    ) -> Tensor:
        """Process a block while preserving enough input for the next block."""
        if x.ndim != 3 or x.shape[1] != self.config.input_size:
            raise ValueError(
                "Expected input shape [batch, "
                f"{self.config.input_size}, time], got {tuple(x.shape)}"
            )
        if x.shape[-1] == 0:
            raise ValueError("WaveNet cannot process an empty time dimension")
        if reset_state:
            self.reset_state()
        if detach_state:
            self.detach_state()

        history_matches = (
            self._input_history.ndim == 3
            and self._input_history.shape[0] == x.shape[0]
            and self._input_history.device == x.device
            and self._input_history.dtype == x.dtype
        )
        if not history_matches:
            self._input_history = x.new_empty(x.shape[0], x.shape[1], 0)

        context = torch.cat((self._input_history, x), dim=-1)
        output = self._forward_stateless(context)[..., -x.shape[-1] :]
        history_length = self.receptive_field - 1
        self._input_history = (
            context[..., -history_length:]
            if history_length > 0
            else context[..., :0]
        )
        return output

    @torch.no_grad()
    def process_sample(self, x: Tensor) -> Tensor:
        """Process one sample per batch with cached dilated-convolution state."""
        original_ndim = x.ndim
        if original_ndim == 1:
            x = x.unsqueeze(0)
        elif original_ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]
        if x.ndim != 2 or x.shape[1] != self.config.input_size:
            raise ValueError(
                "Expected sample shape [channels], [batch, channels], or "
                f"[batch, channels, 1], got {tuple(x.shape)}"
            )

        residual = self._pointwise_sample(self.input_projection, x)
        skip_total: Tensor | None = None
        for block in self.blocks:
            residual, skip = block.process_sample(residual)
            skip_total = skip if skip_total is None else skip_total + skip

        if skip_total is None:
            raise RuntimeError("WaveNet requires at least one residual block")
        output = F.relu(skip_total)
        output = F.relu(self._pointwise_sample(self.output_projection, output))
        output = self._pointwise_sample(self.output, output)
        return output.squeeze(0) if original_ndim == 1 else output

    def reset_state(self) -> None:
        self._input_history = self.input_projection.weight.new_empty(0)
        for block in self.blocks:
            block.reset_state()

    def detach_state(self) -> None:
        self._input_history = self._input_history.detach()
        for block in self.blocks:
            block.detach_state()

    @classmethod
    def from_config(cls, config: dict[str, Any] | ModelConfig) -> "WaveNetModel":
        if isinstance(config, dict):
            model_type = config.get("type", "wavenet")
            params = _load_model_params(model_type, config.get("params", {}))
            config = ModelConfig(
                type=model_type,
                params=params,
                input_size=config.get("input_size", 1),
                output_size=config.get("output_size", 1),
                sample_rate=config.get("sample_rate", 48000),
            )
        return cls(config)

    def export_onnx(self, path: str | Path, opset_version: int = 17) -> None:
        """Export the vectorized stateless path to ONNX."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.eval()

        class StatelessWrapper(nn.Module):
            def __init__(self, model: "WaveNetModel") -> None:
                super().__init__()
                self.model = model

            def forward(self, x: Tensor) -> Tensor:
                return self.model._forward_stateless(x)

        wrapper = StatelessWrapper(self)
        dummy_input = torch.randn(1, self.config.input_size, 512)
        torch.onnx.export(
            wrapper,
            dummy_input,
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "time"},
                "output": {0: "batch_size", 2: "time"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    def export_torchscript(self, path: str | Path) -> None:
        """Export the vectorized stateless path to TorchScript."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.eval()

        class StatelessWrapper(nn.Module):
            def __init__(self, model: "WaveNetModel") -> None:
                super().__init__()
                self.model = model

            def forward(self, x: Tensor) -> Tensor:
                return self.model._forward_stateless(x)

        wrapper = StatelessWrapper(self)
        dummy_input = torch.randn(1, self.config.input_size, 512)
        traced = torch.jit.trace(wrapper, dummy_input)
        traced.save(str(path))

    def export_rtneural(self, path: str | Path) -> None:
        """Report the deliberate lack of an RTNeural WaveNet mapping."""
        raise UnsupportedExportFormatError(self.model_type, "rtneural")
